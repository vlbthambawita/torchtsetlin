"""Run a list of CNN arms sequentially on one GPU, under the programme's shared GPU lock.

Why this exists rather than `code/queue_runner.py`
--------------------------------------------------
`queue_runner.py` hardcodes `code/run_arm.py`, which is the **Tsetlin** driver: it calls
`data.load_boolean(cfg['booleanization'])` unconditionally, reads `arm.model._chunk_size(...)`,
and calls `arm.accuracy()` / `arm.final_diagnostics()`. A float CNN arm has no Booleanization,
no chunk size and none of those three methods, so it cannot go through that driver. The CNN
driver is `code/cnn/train.py` (it has been since P0; `screen/cnn-ctmshape-*.json` were written
by it).

Rather than edit a file the research-engineer owns and the TM queues depend on, this module
**imports `queue_runner`'s lock helper** so that the two queues still exclude each other on a
card, and drives `train.py` instead. One lock, two drivers.

Everything else is deliberately identical to `queue_runner.py`: `name|args` job lines, `#`
comments, skip-if-exists, per-job logs under `logs/`, crash accounting at the end.

    python code/cnn/queue_cnn.py --gpu 1 --jobs jobs/p2_cnn.txt --seeds 0 1 2
    python code/cnn/queue_cnn.py --gpu 1 --jobs jobs/p2_cnn.txt --seeds 0 --wait 86400

``--wait S`` polls the lock for up to ``S`` seconds instead of exiting when the card is busy.
That is what makes a detached launch safe: the queue can be started while another programme's
wave still owns the GPU, and it will sit in the poll loop rather than either dying or
contending. **Throughput numbers are only meaningful on an uncontended card**, which is the
whole reason the lock exists, so `--wait` never bypasses it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))       # code/cnn
CODE = os.path.dirname(HERE)                            # code
ROOT = os.path.dirname(CODE)                            # experiments/convtm
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

import queue_runner as QR  # noqa: E402  the shared lock, not the shared driver


def acquire(gpu: str, wait_s: float = 0.0, poll_s: float = 60.0) -> str:
    """`queue_runner._acquire`, optionally waiting for the card instead of exiting.

    `_acquire` calls `sys.exit()` when the lock is held, so the wait loop catches SystemExit.
    The lock file carries the owner pid and `_acquire` reclaims it when that pid is gone, so a
    queue that dies does not block this one forever.
    """
    t0 = time.time()
    while True:
        try:
            return QR._acquire(gpu)
        except SystemExit as exc:
            if time.time() - t0 >= wait_s:
                raise
            print(f"[wait] {exc}  ({time.time()-t0:.0f}s of {wait_s:.0f}s elapsed); "
                  f"retry in {poll_s:.0f}s", flush=True)
            time.sleep(poll_s)


def out_for(arm: str, seed: int, args: str) -> str:
    """The path `train.py` would choose, recomputed here so skip-if-exists works.

    Must stay in step with `code/cnn/train.py::out_path`: a sample-efficiency point carries an
    `-n<subset>` suffix so a curve point can never be confused with the full-data arm.
    """
    toks = args.split()
    subset = None
    for i, t in enumerate(toks):
        if t == "--subset" and i + 1 < len(toks):
            subset = int(toks[i + 1])
        elif t.startswith("--subset="):
            subset = int(t.split("=", 1)[1])
    tag = f"{arm}-n{subset}" if subset else arm
    return os.path.join(RESULTS, f"{tag}_seed{seed}.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--jobs", required=True, nargs="+",
                    help="one or more job files, run in the order given")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None, help="override every job's epochs")
    ap.add_argument("--wait", type=float, default=0.0,
                    help="seconds to wait for the GPU lock before giving up (0 = fail fast)")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    files = []
    for jf in a.jobs:
        path = jf if os.path.isabs(jf) else os.path.join(ROOT, jf)
        if not os.path.exists(path):
            path = jf if os.path.exists(jf) else path
        with open(path) as f:
            jobs = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        files.append((os.path.basename(path), jobs))

    if not a.dry_run:
        acquire(a.gpu, wait_s=a.wait, poll_s=a.poll)
        print(f"[lock] acquired gpu {a.gpu} (pid {os.getpid()})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu, PYTHONUNBUFFERED="1")
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    failures, done, skipped = [], 0, 0
    t_queue = time.time()

    for fname, jobs in files:
        for seed in a.seeds:
            for job in jobs:
                name, _, args = job.partition("|")
                name = name.strip()
                out = out_for(name, seed, args)
                label = os.path.basename(out)[: -len(f"_seed{seed}.json")]
                if os.path.exists(out):
                    print(f"skip {label} seed {seed} (exists)", flush=True)
                    skipped += 1
                    continue
                cmd = [sys.executable, os.path.join(HERE, "train.py"), "--arm", name,
                       "--seed", str(seed), "--out", out, "--device", "cuda"]
                if a.epochs:
                    cmd += ["--epochs", str(a.epochs)]
                cmd += args.split()
                if a.dry_run:
                    print(" ".join(cmd), flush=True)
                    continue
                print(f"=== [{fname}] {label} seed {seed} (gpu {a.gpu}) ===", flush=True)
                t0 = time.time()
                log = os.path.join(LOGS, f"{label}_seed{seed}.log")
                with open(log, "w") as lf:
                    proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=lf,
                                            stderr=subprocess.STDOUT)
                    # Printed BEFORE the job runs: the rc= line only appears once a job
                    # returns, so without this an in-progress wave and a wave whose jobs all
                    # died look identical in the wave log (CHARTER silent-failure mode 7).
                    print(f"    [start] pid={proc.pid} -> {log}", flush=True)
                    proc.wait()
                dt = time.time() - t0
                ok = proc.returncode == 0 and os.path.exists(out)
                print(f"    rc={proc.returncode}  {dt:.0f}s  "
                      f"result={'yes' if os.path.exists(out) else 'NO'}  -> {log}", flush=True)
                if ok:
                    done += 1
                else:
                    failures.append((label, seed, proc.returncode, log))

    print(f"\n=== p2 cnn queue on gpu {a.gpu}: {done} ok, {skipped} skipped, "
          f"{len(failures)} FAILED, {time.time()-t_queue:.0f}s ===", flush=True)
    for label, seed, rc, log in failures:
        print(f"  FAIL {label} seed {seed} rc={rc}", flush=True)
        try:
            tail = subprocess.check_output(["tail", "-n", "8", log]).decode()
            print("".join(f"      | {ln}\n" for ln in tail.strip().splitlines()), flush=True)
        except Exception:
            pass
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
