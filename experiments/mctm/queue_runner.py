"""Run a list of arms sequentially on one GPU, skipping those already in results/.

The shell wave runner is the documented entry point, but bash re-reads a script file as it
executes it, so editing the wave list while a wave is running corrupts the running shell.
This runner reads its job list once, at start.

Job list format: one ``name|args`` line per arm, ``#`` comments ignored, e.g.

  mctm-size-k6|--l1 greedy --calib 0.20 --max-size 6
  mctm-random-calib|--l1 random --calib 0.20

  python queue_runner.py --gpu 0 --jobs jobs.txt --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--jobs", required=True, help="one 'name|args' line per arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    with open(a.jobs) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # One queue per GPU. Two queues on a 10 GB card OOM each other instantly, and because a
    # crashed run writes no result file the queue then burns through its whole list in
    # seconds, leaving nothing behind but a pile of tracebacks.
    lock = os.path.join(HERE, "logs", f".gpu{a.gpu}.lock")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(lock) as f:
            owner = f.read().strip()
        if owner.isdigit() and os.path.exists(f"/proc/{owner}"):
            sys.exit(f"gpu {a.gpu} is already driven by queue pid {owner} ({lock})")
        os.unlink(lock)                      # stale lock from a killed runner
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(lambda: os.path.exists(lock) and os.unlink(lock))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1")
    for seed in a.seeds:
        for job in jobs:
            name, args = job.split("|", 1)
            out = os.path.join(HERE, "results", f"{name}_seed{seed}.json")
            if os.path.exists(out):
                print(f"skip {name} seed {seed}", flush=True)
                continue
            cmd = [sys.executable, os.path.join(HERE, "run_ideas.py"), "--arm", name,
                   "--seed", str(seed), "--epochs", str(a.epochs),
                   "--epochs-l1", str(a.epochs), *args.split(), "--out", out]
            print(f"=== {name} seed {seed} (gpu {a.gpu}) ===", flush=True)
            t0 = time.time()
            log = os.path.join(HERE, "logs", f"{name}_seed{seed}.log")
            with open(log, "w") as lf:
                p = subprocess.run(cmd, env=env, cwd=HERE, stdout=lf,
                                   stderr=subprocess.STDOUT)
            print(f"    rc={p.returncode}  {time.time()-t0:.0f}s  -> {log}", flush=True)
    print(f"=== queue {a.jobs} done on gpu {a.gpu} ===", flush=True)


if __name__ == "__main__":
    main()
