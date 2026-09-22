"""Run a list of arms sequentially on one GPU, skipping those already in results/.

Adapted from ``experiments/mctm/queue_runner.py``; its two hard-won behaviours are kept.

**The per-GPU lock.** Two queues on the 10 GB card OOM each other instantly, and because a
crashed run writes no result file, the queue then burns through its whole list in seconds and
leaves nothing behind but a pile of tracebacks. One queue per GPU, enforced by an exclusive
lock file carrying the owner pid (a stale lock from a killed runner is reclaimed).

**Skip-if-exists.** Jobs are idempotent, so a queue can be re-run after a partial crash.

Added here: **crash accounting** (PLAN Section 7.4.7). Every job's return code is logged, and
the queue prints a final table of failures with the tail of each failing log. A queue that
finishes suspiciously fast is a queue whose jobs all crashed; this makes that visible instead
of leaving it to be discovered three phases later.

The job list is read once, at start, so it can be edited while a queue runs.

Job list format: one ``arm|args`` line per job, or ``arm:label|args`` to name the result
explicitly. **Use the second form whenever two jobs in one file differ only in their
arguments** — a clause ladder, a budget sweep — because otherwise they share ``arm + tag`` as a
label and therefore share one output path: the first is written and the rest are silently
skipped as "exists", so the wave appears to have run. ``#`` comments ignored::

    ctm-vanilla|--patch 10 --n-clauses 8000
    ctm-small|--bool therm8

    python code/queue_runner.py --gpu 1 --jobs jobs/p0.txt --seeds 0 1 2 --epochs 30

A ``--tag`` suffixes the result name *and* the record's ``arm`` field, so a sweep of one arm
does not collide in ``results/`` or in ``results/preds/``. A tag beginning with ``-`` must be
written ``--tag=-chunk29``; ``--tag -chunk29`` is read by argparse as another option.
"""
from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")


def _acquire(gpu: str) -> str:
    os.makedirs(LOGS, exist_ok=True)
    lock = os.path.join(LOGS, f".gpu{gpu}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        owner = open(lock).read().strip()
        if owner.isdigit() and os.path.exists(f"/proc/{owner}"):
            sys.exit(f"gpu {gpu} is already driven by queue pid {owner} ({lock})")
        os.unlink(lock)                       # stale lock from a killed runner
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(lambda: os.path.exists(lock) and os.unlink(lock))
    return lock


def acquire_or_wait(gpu: str, wait_s: float = 0.0, poll_s: float = 60.0) -> str:
    """:func:`_acquire`, optionally polling for the card instead of exiting when it is held.

    ``_acquire`` is left exactly as it was, because ``code/cnn/queue_cnn.py`` imports it
    directly (DR-005 Decision 3: one lock, two drivers) and its ``sys.exit``-on-contention
    behaviour is what that module's own wait loop is built on.

    This is what makes a **detached** launch safe: a queue can be started while another wave
    still owns the GPU and will sit in the poll loop rather than dying or -- far worse --
    contending, which would corrupt both waves' throughput numbers. It never bypasses the
    lock.

    ``poll_s`` is a priority dial, and using it as one is deliberate rather than accidental:
    two waiters both poll, and at a lock release the one with the shorter period almost always
    wins. A queue that must go next says so with a short period; the default is long, so
    nothing acquires this behaviour by accident.
    """
    t0 = time.time()
    while True:
        try:
            return _acquire(gpu)
        except SystemExit as exc:
            if time.time() - t0 >= wait_s:
                raise
            print(f"[wait] {exc}  ({time.time()-t0:.0f}s of {wait_s:.0f}s elapsed); "
                  f"retry in {poll_s:.0f}s", flush=True)
            time.sleep(poll_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--jobs", required=True, help="one 'name|args' line per job")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None, help="override every job's epochs")
    # NOTE: a tag that starts with '-' must be passed as --tag=-foo, not --tag -foo:
    # argparse reads a leading-dash value as another option and errors out.
    ap.add_argument("--tag", default="",
                    help="suffix appended to the result name and to --label. A tag starting "
                         "with '-' must use the --tag=-foo form.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="poll the GPU lock for up to S seconds instead of exiting when the "
                         "card is busy. Required for a detached launch behind another wave.")
    ap.add_argument("--poll", type=float, default=60.0,
                    help="seconds between lock polls. Shorter = higher priority at a release "
                         "point, since the shortest-period waiter usually wins the lock.")
    a = ap.parse_args()

    with open(a.jobs) as f:
        jobs = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not a.dry_run:
        acquire_or_wait(a.gpu, a.wait, a.poll)
    else:
        _acquire(a.gpu)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1")
    os.makedirs(RESULTS, exist_ok=True)
    failures, done, skipped = [], 0, 0
    t_queue = time.time()

    for seed in a.seeds:
        for job in jobs:
            name, _, args = job.partition("|")
            name = name.strip()
            # A job may name its own label as `arm:label|args`. Without this, two jobs in one
            # file that differ only in their arguments -- a clause ladder, a budget sweep --
            # share `arm + tag` as a label and therefore share ONE output path: the first is
            # written, the rest are skipped as "exists", and the wave looks like it ran.
            if ":" in name:
                name, label = (p.strip() for p in name.split(":", 1))
                label = f"{label}{a.tag}"
            else:
                label = f"{name}{a.tag}"
            out = os.path.join(RESULTS, f"{label}_seed{seed}.json")
            if os.path.exists(out):
                print(f"skip {label} seed {seed} (exists)", flush=True)
                skipped += 1
                continue
            cmd = [sys.executable, os.path.join(HERE, "run_arm.py"), "--arm", name,
                   "--label", label, "--seed", str(seed), "--out", out]
            if a.epochs:
                cmd += ["--epochs", str(a.epochs)]
            cmd += args.split()
            if a.dry_run:
                print(" ".join(cmd), flush=True)
                continue
            print(f"=== {label} seed {seed} (gpu {a.gpu}) ===", flush=True)
            t0 = time.time()
            log = os.path.join(LOGS, f"{label}_seed{seed}.log")
            with open(log, "w") as lf:
                proc = subprocess.Popen(cmd, env=env, cwd=HERE, stdout=lf,
                                        stderr=subprocess.STDOUT)
                # Printed BEFORE the job runs. The per-job rc= line below only appears once a
                # job returns, so without this an in-progress wave and a wave whose jobs all
                # died look identical in the wave log -- which is the first log a supervisor
                # checks. This names the pid and the per-job log, i.e. the discriminators.
                print(f"    [start] pid={proc.pid} -> {log}", flush=True)
                proc.wait()
            p = proc
            dt = time.time() - t0
            ok = p.returncode == 0 and os.path.exists(out)
            print(f"    rc={p.returncode}  {dt:.0f}s  result={'yes' if os.path.exists(out) else 'NO'}"
                  f"  -> {log}", flush=True)
            if ok:
                done += 1
            else:
                failures.append((label, seed, p.returncode, log))

    print(f"\n=== queue {os.path.basename(a.jobs)} on gpu {a.gpu}: {done} ok, "
          f"{skipped} skipped, {len(failures)} FAILED, {time.time()-t_queue:.0f}s ===", flush=True)
    for label, seed, rc, log in failures:          # crash accounting, never silent
        print(f"  FAIL {label} seed {seed} rc={rc}", flush=True)
        try:
            tail = subprocess.check_output(["tail", "-n", "6", log]).decode()
            print("".join(f"      | {ln}\n" for ln in tail.strip().splitlines()), flush=True)
        except Exception:
            pass
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
