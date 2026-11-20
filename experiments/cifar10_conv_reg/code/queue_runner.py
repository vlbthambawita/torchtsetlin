"""Run the programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu.sh <slot>`` (one per GPU; the plan uses slots 0, 3 and 1). Each runner
repeatedly takes the first job that is not finished, not claimed and not blocked, in this order
(plan §8):

  1. pricing     8 short timings (4 sizes, with and without drop clause): the cost table
  2. factorial   ``base`` and ``aug`` at the largest size (64 000 clauses): they need no screen
  3. screen      8 runs at 8 000 clauses; when all are final, p* is written to
                 results/screen/choice.json (rule: common.choose)
  4. factorial   everything else, longest first; ``drop`` and ``aug_drop`` wait for p*

With three runners, two start the 64 000-clause runs while the third works through the screen.
``CCR_PHASES=pricing,screen`` restricts a runner to some phases; ``CCR_DROP_P=0.25`` fixes p*
without a screen.

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
    AUGS,
    CACHE,
    DROP_PS,
    FACTORIAL_CLAUSES,
    FINAL,
    LOGS,
    PHASES,
    PROG,
    RESULTS,
    choose,
    drop_choice,
    est_cost,
    jobs,
    read_json,
    status_of,
    tag_of,
    write_json,
)

SMOKE_ARGS = ["--subset", "2000", "--min-epochs", "1", "--max-epochs", "2", "--viz-every", "1"]
SMOKE_WANDB = {"conv_c1000_b32_geocol_dc0.25_s0"}
CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("CCR_DEVICE", "cuda:0")      # CCR_DEVICE=cpu: test the harness without a GPU
INTERRUPTED = {143, -15, 130, -2, 137, -9}          # SIGTERM / SIGINT / SIGKILL: not a failure
HOST = socket.gethostname()
POLL_S = int(os.environ.get("CCR_POLL_S", 120))


def ordered_jobs(phases):
    """The job list in priority order (static; ``blocked`` is checked separately)."""
    out = list(jobs("pricing")) if "pricing" in phases else []
    fact = jobs("factorial") if "factorial" in phases else []
    first = [j for j in fact if j["clauses"] == max(FACTORIAL_CLAUSES) and j["arm"] in ("base", "aug")]
    rest = sorted((j for j in fact if j not in first), key=est_cost, reverse=True)
    return out + first + (list(jobs("screen")) if "screen" in phases else []) + rest


def blocked(job):
    """Why a job cannot start yet (None = it can)."""
    return "waits for the screen's choice of p*" if job["p"] is None else None


def screen_choice():
    """Once every screen run is final, write p* (plan §5) to results/screen/choice.json."""
    f = RESULTS / "screen" / "choice.json"
    if f.exists():
        return
    val = {}
    for a in AUGS:
        for p in DROP_PS:
            r = read_json(RESULTS / "screen" / f"{tag_of(jobs('screen')[0]['clauses'], a, p)}.json")
            if r is None or r.get("status") not in FINAL:
                return
            if r.get("status") != "done":
                write_json({"error": f"screen run {r['tag']} ended with status {r.get('status')}"},
                           RESULTS / "screen" / "choice_failed.json")
                return
            val[(a, p)] = r["best_val_acc"]
    c = dict(choose(val), written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(c, f)
    print(f"== screen choice: p* = {c['p_star']:g} (drop clause {'helps' if c['helps'] else 'does not help'} "
          f"at the screen size)", flush=True)


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
    cmd = [sys.executable, "train.py", "--clauses", str(job["clauses"]), "--aug", job["aug"],
           "--drop-clause-p", f"{job['p']:g}", "--seed", str(job["seed"]), "--phase", job["phase"],
           "--device", DEVICE]
    if job.get("arm"):
        cmd += ["--arm", job["arm"]]
    if job["phase"] == "pricing":
        cmd += ["--calibrate", "200", "--no-wandb"]
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
            f"  outer CUDA_VISIBLE_DEVICES (before run_gpu.sh): {os.environ.get('CCR_OUTER_CVD') or 'unset'}\n"
            f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}\n"
            f"  nvidia-smi -L:\n    " + smi.replace("\n", "\n    ")) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="GPU slot (run_gpu.sh sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--phases", default=os.environ.get("CCR_PHASES") or ",".join(PHASES),
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
    os.environ["CCR_GPU"] = str(args.gpu)
    me = f"gpu{args.gpu}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        clear_stale_claims()
        print(f"== {me}: {name} on {HOST}, phases {', '.join(phases)}", flush=True)

    idle_since, lonely = None, 0
    while True:
        if not smoke:
            if "screen" in phases and not args.dry_run:
                screen_choice()
            why = stop_reason()
            if why and not args.dry_run:
                print(f"== {me}: stopping ({why})", flush=True)
                return
        todo = jobs("smoke") if smoke else ordered_jobs(phases)
        pending, waiting, stuck = [], 0, 0
        for j in todo:
            if j["p"] is not None and status_of(j["phase"], j["tag"]) in FINAL:
                continue
            if (CLAIMS / j["phase"] / f"{j['tag']}.failed-{me}").exists():
                continue
            if blocked(j):
                stuck += 1
            elif (CLAIMS / j["phase"] / j["tag"]).exists():
                waiting += 1
            else:
                pending.append(j)
        if args.dry_run:
            for j in pending[:30]:
                print(f"{j['phase']:9s} {j.get('arm') or '':8s} {j['tag']:40s} est {est_cost(j):6.0f} s/epoch")
            print(f"... {len(pending)} runnable now, {waiting} claimed elsewhere, {stuck} waiting for p* "
                  f"(p* = {drop_choice()})")
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            # Blocked jobs are released by the screen. If nobody holds a claim, the screen is not
            # running anywhere (another runner may be between two jobs: look three times).
            lonely = lonely + 1 if waiting == 0 else 0
            if waiting == 0 and (stuck == 0 or lonely >= 3):
                if stuck:
                    print(f"== {me}: {stuck} job(s) wait for p*, but no screen run is running "
                          f"(finish the screen, or set CCR_DROP_P)", flush=True)
                print(f"== {me}: nothing left to do", flush=True)
                return
            if idle_since is None:
                idle_since = time.time()
                print(f"== {me}: {waiting} job(s) running elsewhere, {stuck} waiting for p*; polling", flush=True)
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
