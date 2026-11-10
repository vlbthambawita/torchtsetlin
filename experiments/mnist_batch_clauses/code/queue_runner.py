"""Work through one arm of the plan on one GPU, sharing the queue with other runners.

A job is claimed by atomically creating ``.cache/claims/<arm>/<tag>``; a job whose record says
``done`` is skipped. A failed job (e.g. out of memory) releases its claim and leaves a
``<tag>.failed-<device>`` marker so another GPU can take it. An interrupted job resumes from its
checkpoint when the runner is restarted (remove its claim directory first).

D8: seed-0 grid cells give the paper's timings, so they only run on an RTX 3090 (checked by name),
one at a time. Order: smallest models first within a seed, so complete rows of the heatmap
arrive early; B = 1 last within each clause count (the slowest).

Usage: nohup python code/queue_runner.py --arm grid --seeds 0 --device cuda:0 > logs/q_grid_s0.log 2>&1 &
       python code/queue_runner.py --arm smoke --device cuda:1 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

import torch

from common import CACHE, LOGS, PROG, RESULTS, SEEDS, jobs, tag_of

SMOKE_ARGS = ["--subset", "2000", "--min-epochs", "1", "--max-epochs", "1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["grid", "control_T", "control_seq", "smoke"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--device", required=True)
    ap.add_argument("--models", nargs="+", default=["flat", "conv"])
    ap.add_argument("--wandb-only", nargs="*", default=None,
                    help="smoke: mirror only these tags to W&B (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(torch.device(args.device))
    todo = [j for j in jobs(args.arm, args.seeds) if j[0] in args.models]
    todo.sort(key=lambda j: (j[4], j[1], j[2] == 1, j[2], j[0]))
    claims = CACHE / "claims" / args.arm
    claims.mkdir(parents=True, exist_ok=True)
    (LOGS / args.arm).mkdir(parents=True, exist_ok=True)
    for model, c, b, t_mode, seed, seq in todo:
        tag = tag_of(model, c, b, t_mode, seed, seq)
        if args.arm == "grid" and seed == 0 and "3090" not in gpu:
            continue                                       # D8: timing cells on the 3090 only
        rec = RESULTS / args.arm / f"{tag}.json"
        if rec.exists() and json.loads(rec.read_text()).get("status") == "done":
            continue
        if (claims / f"{tag}.failed-{args.device}").exists():
            continue
        if args.dry_run:
            print(tag)
            continue
        try:
            os.mkdir(claims / tag)
        except FileExistsError:
            continue
        cmd = [sys.executable, "train.py", "--model", model, "--clauses", str(c), "--batch-size", str(b),
               "--t-mode", t_mode, "--seed", str(seed), "--arm", args.arm, "--device", args.device]
        if seq:
            cmd.append("--sequential")
        if args.arm == "smoke":
            cmd += SMOKE_ARGS
            if args.wandb_only is not None and tag not in args.wandb_only:
                cmd.append("--no-wandb")
        print(f"== {args.device}: {tag}", flush=True)
        with open(LOGS / args.arm / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, cwd=PROG / "code", stdout=log, stderr=subprocess.STDOUT)
        print(f"== {args.device}: {tag} exit {rc}", flush=True)
        if rc != 0:
            (claims / f"{tag}.failed-{args.device}").touch()
            os.rmdir(claims / tag)


if __name__ == "__main__":
    main()
