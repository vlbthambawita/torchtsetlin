"""Run the specialist jobs on one GPU, pulling from a queue shared with the other GPU.

A job is claimed by atomically creating ``.cache/claims/<spec>_seed<seed>``; a GPU skips jobs it
cannot hold (HoG peaks at 19.5 GiB). A job whose record says ``done`` is skipped, and an
interrupted one resumes from its checkpoint when the runner is started again (remove its claim).

Usage: python code/queue_runner.py --device cuda:0 --seeds 0 1 2
"""

import argparse
import json
import os
import subprocess
import sys

import torch
from common import CACHE, PROG, RESULTS

ORDER = ["HoG", "C4", "C3", "T10"]          # longest-first per seed, so the GPUs finish together
NEEDS_GIB = {"HoG": 20.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variant", default="", help="e.g. softbudget")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra train_specialist.py flags, e.g. --budget-mode soft")
    args = ap.parse_args()
    idx = torch.device(args.device).index or 0
    gib = torch.cuda.get_device_properties(idx).total_memory / 2**30
    claims = CACHE / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for spec in ORDER:
            tag = f"{spec}_seed{seed}" + (f"_{args.variant}" if args.variant else "")
            if NEEDS_GIB.get(spec, 0) > gib:
                continue
            rec = RESULTS / f"{tag}.json"
            if rec.exists() and json.loads(rec.read_text()).get("status") == "done":
                continue
            try:
                os.mkdir(claims / tag)
            except FileExistsError:
                continue
            print(f"== {args.device}: {tag}", flush=True)
            cmd = [sys.executable, "train_specialist.py", "--spec", spec, "--seed", str(seed),
                   "--device", args.device] + (["--variant", args.variant] if args.variant else []) + args.extra
            with open(PROG / "logs" / f"{tag}.log", "a") as log:
                rc = subprocess.call(cmd, cwd=PROG / "code", stdout=log, stderr=subprocess.STDOUT)
            print(f"== {args.device}: {tag} exit {rc}", flush=True)


if __name__ == "__main__":
    main()
