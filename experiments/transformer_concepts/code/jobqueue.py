"""Lock-file job queue: any number of workers share one JSON job list.

A job is a dict of ``runner.run`` keyword arguments (``stage``, ``key``, ``task``, ...). A worker
claims a job by creating ``results/<stage>/<key>.lock`` with O_EXCL; a job whose record exists is
skipped. A crashed job leaves its lock and an ``.err`` file and is not retried automatically.

Run one worker per process, with the GPU pinned outside:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.jobqueue jobs.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

from experiments.transformer_concepts.code.runner import RESULTS, run


def claim(stage: str, key: str) -> bool:
    d = RESULTS / stage
    d.mkdir(parents=True, exist_ok=True)
    if (d / f"{key}.json").exists():
        return False
    try:
        fd = os.open(d / f"{key}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{os.getpid()} {time.strftime('%H:%M:%S')}".encode())
    os.close(fd)
    return True


def work(paths) -> None:
    for path in paths:
        jobs = json.loads(open(path).read())
        for job in jobs:
            stage, key = job["stage"], job["key"]
            if not claim(stage, key):
                continue
            t0 = time.time()
            try:
                rec = run(**job)
                (RESULTS / stage / f"{key}.lock").unlink(missing_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] {stage}/{key} val={rec['best_val']:.4f} "
                      f"ep={rec['epochs_run']} {time.time() - t0:.0f}s", flush=True)
            except Exception:
                (RESULTS / stage / f"{key}.err").write_text(traceback.format_exc())
                print(f"FAILED {stage}/{key}", flush=True)


if __name__ == "__main__":
    work(sys.argv[1:])
