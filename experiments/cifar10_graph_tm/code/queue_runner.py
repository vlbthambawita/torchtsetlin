"""Run the programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu.sh <K>`` (one per GPU / tmux session; the plan uses four V100s). Each runner
repeatedly takes the first job that is not finished, not claimed and not waiting, in this order
(plan §6):

  1. pricing   short timings of every cell (seconds per epoch, peak memory). A cell above the gate
               (1 500 s per epoch, or out of memory) is not run as planned: for the depth study
               its smaller options are priced next (inbox 1 024, then 40 000 clauses); a paper
               run above the gate is reported and not started (``CGT_PAPER_FORCE=1`` starts it)
  2. paper     the runs of the paper's setting (GraphTM and CoTM baseline): they need no pilot
  3. pilot     6 runs at depth 1, 20 000 clauses; when all are final, T / C and s are written to
               results/pilot/choice.json (rule: common.choose)
  4. depth     the depth study, longest first; every run waits for the pilot and for its pricing

With four runners, pricing takes a few minutes; then two take the paper runs and two the pilot.
``CGT_PHASES=pricing,pilot`` restricts a runner to some phases; ``CGT_TS=0.5,5`` fixes T / C and
s without a pilot; ``CGT_GATE=off`` runs every cell as planned, without pricing.

A job is claimed by atomically creating ``.cache/claims/<phase>/<tag>/`` (with owner.json). A job
whose record is final (done / oom) is skipped. A crashed job leaves ``<tag>.failed-gpu<K>`` so
this GPU does not retry it (another GPU may). Restarting a runner removes its own stale claims
(dead pid on this host), so interrupted runs resume from their checkpoints.

Stop gracefully: ``touch STOP`` in the programme folder (runners finish their current job and exit).

Usage: python code/queue_runner.py --gpu 0               # CUDA_VISIBLE_DEVICES is set by run_gpu.sh
       python code/queue_runner.py --gpu 0 --phases smoke
       python code/queue_runner.py --gpu 0 --dry-run     # show what this GPU would run next
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

import torch
from common import (
    CACHE,
    FINAL,
    LOGS,
    PHASES,
    PILOT_MAX_EPOCHS,
    PROG,
    RESULTS,
    choose,
    est_cost,
    jobs,
    read_json,
    status_of,
    ts_choice,
    write_json,
)

SMOKE_ARGS = ["--subset", "1000", "--min-epochs", "1", "--max-epochs", "2", "--viz-every", "1"]
SMOKE_WANDB = {"gtm_d2_c500_m64_grid_s0"}
CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("CGT_DEVICE", "cuda:0")      # CGT_DEVICE=cpu: test the harness without a GPU
INTERRUPTED = {143, -15, 130, -2, 137, -9}          # SIGTERM / SIGINT / SIGKILL: not a failure
HOST = socket.gethostname()
POLL_S = int(os.environ.get("CGT_POLL_S", 120))


def ordered_jobs(phases):
    """The job list in priority order (recomputed every round: the gate adds pricing jobs)."""
    out = []
    for ph in ("pricing", "paper", "pilot"):
        if ph in phases:
            out += jobs(ph)
    if "depth" in phases:
        out += sorted(jobs("depth"), key=est_cost, reverse=True)
    return out


def pilot_choice():
    """Once every pilot run is final, write T / C and s to results/pilot/choice.json."""
    f = RESULTS / "pilot" / "choice.json"
    if f.exists():
        return
    val = {}
    for j in jobs("pilot"):
        r = read_json(RESULTS / "pilot" / f"{j['tag']}.json")
        if r is None or r.get("status") not in FINAL:
            return
        if r.get("status") != "done":
            write_json({"error": f"pilot run {r['tag']} ended with status {r.get('status')}"},
                       RESULTS / "pilot" / "choice_failed.json")
            return
        val[(j["T"], j["s"])] = r["best_val_acc"]
    c = dict(choose(val), written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(c, f)
    print(f"== pilot choice: T = {c['T']} at {jobs('pilot')[0]['clauses']} clauses (T / C = {c['ratio']:g}), "
          f"s = {c['s']:g}", flush=True)


def stop_reason():
    if (PROG / "STOP").exists():
        return "STOP file present"
    return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_stale_claims():
    for d in CLAIMS.glob("*/*"):
        if not d.is_dir():
            continue
        o = read_json(d / "owner.json")
        if o and o.get("host") == HOST and not any(alive(int(o.get(k) or -1)) for k in ("pid", "child")):
            shutil.rmtree(d, ignore_errors=True)
            print(f"== released stale claim {d.parent.name}/{d.name} (pid {o.get('pid')} gone)", flush=True)


def claim(job, gpu):
    d = CLAIMS / job["phase"] / job["tag"]
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        return None
    write_json({"host": HOST, "pid": os.getpid(), "gpu": gpu, "started": time.strftime("%c")}, d / "owner.json")
    return d


def command(job):
    cmd = [sys.executable, "train.py", "--kind", job["kind"], "--clauses", str(job["clauses"]),
           "--T", str(job["T"]), "--s", f"{job['s']:g}", "--seed", str(job["seed"]), "--phase", job["phase"],
           "--tag", job["tag"], "--device", DEVICE]
    if job["kind"] == "grid":
        cmd += ["--depth", str(job["depth"]), "--edges", job["edges"]]
        if job["depth"] > 1:
            cmd += ["--inbox", str(job["inbox"])]
    if job.get("planned"):
        cmd += ["--planned", job["planned"], "--role", job["role"]]
    if job["phase"] == "pricing":
        cmd += ["--calibrate", "100", "--no-wandb"]
    if os.environ.get("CGT_TIME_CAP_H") and job["phase"] != "pricing":
        cmd += ["--time-cap-h", os.environ["CGT_TIME_CAP_H"]]
    if job["phase"] == "pilot":
        cmd += ["--max-epochs", str(PILOT_MAX_EPOCHS)]
    if job["phase"] == "smoke":
        cmd += SMOKE_ARGS
        if job["tag"] not in SMOKE_WANDB:
            cmd.append("--no-wandb")
    return cmd


def gpu_name(slot, cvd):
    """Name of the one visible GPU, or a diagnosis of why there is none."""
    try:
        return torch.cuda.get_device_name(0)
    except RuntimeError as e:
        try:
            smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e2:  # noqa: BLE001
            smi = f"nvidia-smi failed: {e2}"
        raise SystemExit(
            f"GPU slot {slot}: PyTorch sees no GPU with CUDA_VISIBLE_DEVICES={cvd} ({e}).\n"
            f"  outer CUDA_VISIBLE_DEVICES (before run_gpu.sh): {os.environ.get('CGT_OUTER_CVD') or 'unset'}\n"
            f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}\n"
            f"  nvidia-smi -L:\n    " + smi.replace("\n", "\n    ")) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="runner number (run_gpu.sh sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--phases", default=os.environ.get("CGT_PHASES") or ",".join(PHASES),
                    help=f"comma-separated: {', '.join(PHASES)} (default: all), or smoke")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    smoke = phases == ["smoke"]
    if not smoke and any(p not in PHASES for p in phases):
        raise SystemExit(f"unknown phase in {phases}; choose from {PHASES} or smoke")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if DEVICE != "cpu" and not args.dry_run and (not cvd or "," in cvd):
        raise SystemExit(f"start me through ./run_gpu.sh {args.gpu}: it gives this runner exactly one GPU "
                         f"(CUDA_VISIBLE_DEVICES is {cvd or 'unset'})")
    name = "dry run" if args.dry_run else gpu_name(args.gpu, cvd) if DEVICE != "cpu" else "cpu (harness test)"
    os.environ["CGT_GPU"] = str(args.gpu)
    me = f"gpu{args.gpu}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        clear_stale_claims()
        print(f"== {me}: {name} on {HOST}, phases {', '.join(phases)}", flush=True)

    idle_since, lonely = None, 0
    while True:
        if not smoke:
            if "pilot" in phases and not args.dry_run:
                pilot_choice()
            why = stop_reason()
            if why and not args.dry_run:
                print(f"== {me}: stopping ({why})", flush=True)
                return
        todo = jobs("smoke") if smoke else ordered_jobs(phases)
        pending, waiting, stuck = [], 0, []
        for j in todo:
            if status_of(j["phase"], j["tag"]) in FINAL:
                continue
            if (CLAIMS / j["phase"] / f"{j['tag']}.failed-{me}").exists():
                continue
            if j["wait"]:
                stuck.append(j)
            elif (CLAIMS / j["phase"] / j["tag"]).exists():
                waiting += 1
            else:
                pending.append(j)
        if args.dry_run:
            for j in pending[:30]:
                print(f"{j['phase']:8s} {j['tag']:44s} T {j['T']} s {j['s']:g}")
            for j in stuck:
                print(f"{j['phase']:8s} {j['tag']:44s} {j['wait']}")
            print(f"... {len(pending)} runnable now, {waiting} claimed elsewhere, {len(stuck)} waiting "
                  f"(T / C, s = {ts_choice()})")
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            # Waiting jobs are released by the pilot and by pricing. If nobody holds a claim, those
            # are not running anywhere (another runner may be between two jobs: look three times).
            lonely = lonely + 1 if waiting == 0 else 0
            if waiting == 0 and (not stuck or lonely >= 3):
                for why in sorted({j["wait"] for j in stuck}):
                    print(f"== {me}: {sum(j['wait'] == why for j in stuck)} job(s): {why}, but nothing that "
                          f"would release them is running (see ./status.sh; CGT_TS / CGT_GATE=off override)",
                          flush=True)
                print(f"== {me}: nothing left to do", flush=True)
                return
            if idle_since is None:
                idle_since = time.time()
                print(f"== {me}: {waiting} job(s) running elsewhere, {len(stuck)} waiting; polling", flush=True)
            time.sleep(POLL_S)
            continue
        lonely = 0
        idle_since = None
        log = LOGS / "runs" / f"{job['phase']}__{job['tag']}.log"
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['phase']} {job['tag']}", flush=True)
        with open(log, "a") as fh:
            p = subprocess.Popen(command(job), cwd=PROG / "code", stdout=fh, stderr=subprocess.STDOUT)
            owner = CLAIMS / job["phase"] / job["tag"] / "owner.json"
            write_json(dict(read_json(owner) or {}, child=p.pid), owner)
            try:
                rc = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                rc = p.wait()
        st = status_of(job["phase"], job["tag"])
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['tag']} exit {rc} status {st}", flush=True)
        if rc in INTERRUPTED:                   # stopped by the user: resumes on the next start
            shutil.rmtree(CLAIMS / job["phase"] / job["tag"], ignore_errors=True)
            print(f"== {me}: interrupted, exiting", flush=True)
            return
        if rc != 0 and st not in FINAL:
            (CLAIMS / job["phase"] / f"{job['tag']}.failed-{me}").touch()
        shutil.rmtree(CLAIMS / job["phase"] / job["tag"], ignore_errors=True)


if __name__ == "__main__":
    main()
