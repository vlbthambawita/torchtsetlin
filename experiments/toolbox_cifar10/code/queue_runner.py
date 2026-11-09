"""Run the 22 specialists on one GPU, sharing the queue with the other GPU.

Each GPU works through its own list (QUEUES), then takes unclaimed jobs from the other list. A job is
claimed by atomically creating ``.cache/claims/<tag>``. A job whose record says ``done`` is skipped.
A job that fails (e.g. out of memory on the 10 GB card) releases its claim and leaves a
``.cache/claims/<tag>.failed-<device>`` marker, so the other GPU can take it and this one skips it.
An interrupted job resumes from its checkpoint when the runner is restarted (remove its claim).

Order: the plain specialists first (the 11-specialist composite completes early), then the
augmented ones; the longest first within each half.

Usage: nohup python code/queue_runner.py --device cuda:0 > logs/queue_gpu0.log 2>&1 &
"""

import argparse
import json
import os
import subprocess
import sys

from common import CACHE, PROG, RESULTS
from specialists import BEST_CLAUSES

# HOG at 64 000 (27.8 GiB) and 32 000 clauses/class do not fit 24 GB; 16 000 is its largest size.
CLAUSES = {"hog": 16000, "hog_aug": 16000}
QUEUES = {
    # RTX 3090, 24 GB: the 4x4 / 5x5 thermometers (10-17 GiB at 64k) and HOG (21 GiB at 16k)
    "cuda:0": ["therm5", "therm4", "atherm5", "atherm4", "hog",
               "therm5_aug", "therm4_aug", "atherm5_aug", "atherm4_aug", "hog_aug"],
    # RTX 3080, 10 GB: the 3-plane specialists and the 3x3 thermometers (<= 7 GiB at 64k)
    "cuda:1": ["therm3", "atherm3", "canny10", "amean10", "otsu10", "agauss5",
               "therm3_aug", "atherm3_aug", "canny10_aug", "amean10_aug", "otsu10_aug", "agauss5_aug"],
}


def tag_of(spec):
    return f"{spec}_c{CLAUSES.get(spec, BEST_CLAUSES)}_seed0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, choices=sorted(QUEUES))
    args = ap.parse_args()
    other = [d for d in QUEUES if d != args.device][0]
    jobs = QUEUES[args.device] + QUEUES[other][::-1]
    claims = CACHE / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for spec in jobs:
        tag = tag_of(spec)
        rec = RESULTS / f"{tag}.json"
        if rec.exists() and json.loads(rec.read_text()).get("status") == "done":
            continue
        if (claims / f"{tag}.failed-{args.device}").exists():
            continue
        try:
            os.mkdir(claims / tag)
        except FileExistsError:
            continue
        print(f"== {args.device}: {tag}", flush=True)
        cmd = [sys.executable, "train.py", "--spec", spec, "--clauses", str(CLAUSES.get(spec, BEST_CLAUSES)),
               "--device", args.device]
        with open(PROG / "logs" / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, cwd=PROG / "code", stdout=log, stderr=subprocess.STDOUT)
        print(f"== {args.device}: {tag} exit {rc}", flush=True)
        if rc != 0:
            (claims / f"{tag}.failed-{args.device}").touch()
            os.rmdir(claims / tag)


if __name__ == "__main__":
    main()
