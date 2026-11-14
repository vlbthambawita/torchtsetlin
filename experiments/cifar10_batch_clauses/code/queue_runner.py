"""Run the whole programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu<K>.sh`` (one per V100). Each runner repeatedly takes the first job that is
not finished, not claimed and whose dependencies are met, in this order (plan §7):

  1. pricing      27 short timings (one per model x C x B corner), feeds the ordering below
  2. pilot        6 flat runs choosing (T_ref, s) on validation (D3); choice.json written automatically
  3. sanity       conv C=2k B=32 (D11); if below the gate, results/STOP.json halts every runner
  4. grid         all cells, longest first (keeps the 4 GPUs busy until the end); flat after the pilot
  5. control_seq, control_T

A job is claimed by atomically creating ``.cache/claims/<arm>/<tag>/`` (with owner.json). A job
whose record is final (done / oom) is skipped. A crashed job leaves ``<tag>.failed-gpu<K>`` so this
GPU does not retry it (another GPU may). Restarting a runner removes its own stale claims (dead
pid on this host), so interrupted runs resume from their checkpoints.

Stop gracefully: ``touch STOP`` in the programme folder (runners finish their current job and exit).

Usage: python code/queue_runner.py --gpu 0               # full programme on GPU 0 (CUDA_VISIBLE_DEVICES=0)
       python code/queue_runner.py --gpu 0 --arm smoke  # smoke test only (setup.sh)
       python code/queue_runner.py --gpu 0 --dry-run    # show what this GPU would run next
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
    FLAT_PILOT,
    LOGS,
    PROG,
    RESULTS,
    SANITY,
    est_cost,
    flat_choice,
    jobs,
    read_json,
    status_of,
    write_json,
)

SMOKE_ARGS = ["--subset", "2000", "--min-epochs", "1", "--max-epochs", "1"]
SMOKE_WANDB = {"conv_c1000_b32_Tscaled_s0", "conv3_c1000_b32_Tscaled_s0",
               "flat_c1000_b32_Tscaled_Tref50_spec10_s0"}
CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("CBC_DEVICE", "cuda:0")      # CBC_DEVICE=cpu: test the harness without a GPU
INTERRUPTED = {143, -15, 130, -2, 137, -9}          # SIGTERM / SIGINT / SIGKILL: not a failure
HOST = socket.gethostname()


def ordered_jobs():
    """The full job list in priority order (static; dependencies are checked separately)."""
    out = list(jobs("pricing")) + list(jobs("pilot"))
    grid = jobs("grid")
    sanity = [j for j in grid if j["tag"] == SANITY["tag"]]
    rest = sorted((j for j in grid if j["tag"] != SANITY["tag"]), key=est_cost, reverse=True)
    return out + sanity + rest + list(jobs("control_seq")) + list(jobs("control_T"))


def blocked(job):
    """Why a job cannot start yet (None = it can)."""
    if job["model"] == "flat" and job["arm"] in ("grid", "control_T", "control_seq") and flat_choice() is None:
        return "waits for the flat pilot"
    return None


def pilot_choice():
    """D3: once every pilot run is final, pick (T_ref, s) by best validation accuracy; ties go to
    the earlier entry of FLAT_PILOT (B4's MNIST values first)."""
    if flat_choice() is not None:
        return
    recs = [read_json(RESULTS / "pilot" / f"{j['tag']}.json") for j in jobs("pilot")]
    if any(r is None or r.get("status") not in FINAL for r in recs):
        return
    done = [(r["best_val_acc"], -i, p, r["tag"]) for i, (r, p) in enumerate(zip(recs, FLAT_PILOT))
            if r.get("status") == "done"]
    if not done:
        write_json({"error": "every flat pilot run failed"}, RESULTS / "pilot" / "choice_failed.json")
        return
    val, _, (t_ref, s), tag = max(done)
    write_json({"T_ref": t_ref, "s": s, "best_val_acc": val, "tag": tag,
                "rule": "max validation accuracy at C=2000, B=32, <=60 epochs; ties -> earlier FLAT_PILOT entry",
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, RESULTS / "pilot" / "choice.json")
    print(f"== pilot choice: flat T_ref={t_ref} s={s:g} (val {val:.2f})", flush=True)


def sanity_gate():
    """D11: halt everything if the reference cell is far below the published value."""
    r = read_json(RESULTS / "grid" / f"{SANITY['tag']}.json")
    if r and r.get("status") == "done" and r["test_acc_at_best"] < SANITY["min_test"]:
        if not (RESULTS / "STOP.json").exists():
            write_json({"reason": f"sanity gate: {SANITY['tag']} test {r['test_acc_at_best']:.2f} < "
                        f"{SANITY['min_test']} (Toolbox Table IV: 63.6 %)", "time": time.strftime("%c")},
                       RESULTS / "STOP.json")


def stop_reason():
    if (PROG / "STOP").exists():
        return "STOP file present"
    s = read_json(RESULTS / "STOP.json")
    return s["reason"] if s else None


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
    d = CLAIMS / job["arm"] / job["tag"]
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        return None
    write_json({"host": HOST, "pid": os.getpid(), "gpu": gpu, "started": time.strftime("%c")}, d / "owner.json")
    return d


def command(job, smoke=False):
    cmd = [sys.executable, "train.py", "--model", job["model"], "--clauses", str(job["clauses"]),
           "--batch-size", str(job["batch"]), "--t-mode", job["t_mode"], "--seed", str(job["seed"]),
           "--arm", job["arm"], "--device", DEVICE]
    if job["sequential"]:
        cmd.append("--sequential")
    if job["pilot"] is not None:
        cmd += ["--pilot-tref", str(job["pilot"][0]), "--pilot-s", f"{job['pilot'][1]:g}"]
    if job["arm"] == "pricing":
        cmd += ["--calibrate", "200", "--no-wandb"]
    if smoke:
        cmd += SMOKE_ARGS
        if job["tag"] not in SMOKE_WANDB:
            cmd.append("--no-wandb")
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="physical GPU index (CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--arm", default="all", choices=["all", "smoke"])
    ap.add_argument("--any-gpu", action="store_true", help="allow a non-V100 GPU (timings then differ)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise SystemExit(f"run me with CUDA_VISIBLE_DEVICES={args.gpu} (run_gpu{args.gpu}.sh does this)")
    name = torch.cuda.get_device_name(0) if DEVICE != "cpu" else "cpu (harness test)"
    if "V100" not in name and not args.any_gpu and DEVICE != "cpu":
        raise SystemExit(f"GPU {args.gpu} is a {name}; the plan's timings are V100 timings (--any-gpu to override)")
    os.environ["CBC_GPU"] = str(args.gpu)
    me = f"gpu{args.gpu}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    clear_stale_claims()
    smoke = args.arm == "smoke"
    print(f"== {me}: {name} on {HOST}, programme {'smoke' if smoke else 'all'}", flush=True)

    idle_since = None
    while True:
        if not smoke:
            pilot_choice()
            sanity_gate()
            why = stop_reason()
            if why:
                print(f"== {me}: stopping ({why})", flush=True)
                return
        todo = jobs("smoke") if smoke else ordered_jobs()
        pending, waiting = [], 0
        for j in todo:
            if status_of(j["arm"], j["tag"]) in FINAL:
                continue
            if (CLAIMS / j["arm"] / f"{j['tag']}.failed-{me}").exists():
                continue
            if (CLAIMS / j["arm"] / j["tag"]).exists() or blocked(j):
                waiting += 1
                continue
            pending.append(j)
        if args.dry_run:
            for j in pending[:20]:
                print(j["arm"], j["tag"], f"est {est_cost(j):.0f}")
            print(f"... {len(pending)} runnable now, {waiting} claimed elsewhere or blocked")
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            if waiting == 0:
                print(f"== {me}: nothing left to do", flush=True)
                return
            if idle_since is None:
                idle_since = time.time()
                print(f"== {me}: {waiting} job(s) running elsewhere or waiting for the pilot; polling", flush=True)
            time.sleep(120)
            continue
        idle_since = None
        log = LOGS / "runs" / f"{job['arm']}__{job['tag']}.log"
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['arm']} {job['tag']}", flush=True)
        with open(log, "a") as fh:
            p = subprocess.Popen(command(job, smoke), cwd=PROG / "code", stdout=fh, stderr=subprocess.STDOUT)
            owner = CLAIMS / job["arm"] / job["tag"] / "owner.json"
            write_json(dict(read_json(owner) or {}, child=p.pid), owner)
            try:
                rc = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                rc = p.wait()
        st = status_of(job["arm"], job["tag"])
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['tag']} exit {rc} status {st}", flush=True)
        if rc in INTERRUPTED:                   # stopped by the user: resumes on the next start
            shutil.rmtree(CLAIMS / job["arm"] / job["tag"], ignore_errors=True)
            print(f"== {me}: interrupted, exiting", flush=True)
            return
        if rc != 0 and st not in FINAL:
            (CLAIMS / job["arm"] / f"{job['tag']}.failed-{me}").touch()
        shutil.rmtree(CLAIMS / job["arm"] / job["tag"], ignore_errors=True)


if __name__ == "__main__":
    main()
