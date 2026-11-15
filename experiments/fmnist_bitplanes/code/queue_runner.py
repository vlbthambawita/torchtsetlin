"""Work through one arm of the plan on one GPU.

A job is claimed by atomically creating ``.cache/claims/<arm>/<tag>``; a job whose record says
``done`` is skipped. A failed job (e.g. out of memory) releases its claim and leaves a
``<tag>.failed-<device>`` marker. An interrupted job resumes from its checkpoint when the runner
is restarted (remove its claim directory first).

GPU (D11, user 2026-11-13): the RTX 3080 only; the RTX 3090 is refused unless ``--allow-3090``
(only when the user asks for it). Stage A (default) = the conv3 cells only; stage B = the
flat/conv2 comparison cells, run only after the user has checked stage A. Order: cheapest first (flat, conv2, conv3; small C first).

Usage: nohup python code/queue_runner.py --arm smoke --device cuda:1 > logs/q_smoke.log 2>&1 &
       python code/queue_runner.py --arm pilot --device cuda:1 --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

import torch

from common import ARCHS, AUG_POLICY, CACHE, LOGS, PROG, RESULTS, STAGES, jobs, tag_of

SMOKE_ARGS = ["--subset", "2000", "--min-epochs", "1", "--max-epochs", "1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["smoke", "pricing", "pilot", "main", "seeds", "aug", "replay"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--allow-3090", action="store_true", help="only when the user asked for the 3090")
    ap.add_argument("--wandb-only", nargs="*", default=None,
                    help="smoke: mirror only these tags to W&B (default: all)")
    ap.add_argument("--calibrate", type=int, default=300, help="pricing: updates to time")
    ap.add_argument("--stage", default="A", choices=sorted(STAGES),
                    help="A = 3D conv cells (now); B = the comparison cells, only after the user OKs it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpu = torch.cuda.get_device_name(torch.device(args.device))
    if "3090" in gpu and not args.allow_3090:
        raise SystemExit(f"{args.device} is the {gpu}: this programme runs on the 3080 only (D11)")
    todo = jobs(args.arm, args.seeds, None if args.arm in ("smoke", "aug", "replay") else STAGES[args.stage])
    augment = AUG_POLICY if args.arm == "aug" else "none"
    todo.sort(key=lambda j: (j[5], ARCHS.index(j[0]), j[2], j[4] or 0, j[3], j[1]))
    claims = CACHE / "claims" / args.arm
    claims.mkdir(parents=True, exist_ok=True)
    (LOGS / args.arm).mkdir(parents=True, exist_ok=True)
    for arch, enc, c, s, kd, seed in todo:
        tag = tag_of(arch, enc, c, s, kd, seed, augment)
        rec = RESULTS / args.arm / f"{tag}.json"
        if rec.exists() and (args.arm == "pricing" or json.loads(rec.read_text()).get("status") == "done"):
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
        cmd = [sys.executable, "train.py", "--arch", arch, "--enc", enc, "--clauses", str(c),
               "--s", f"{s:g}", "--seed", str(seed), "--arm", args.arm, "--device", args.device]
        if kd:
            cmd += ["--kd", str(kd)]
        cmd += ["--augment", augment]
        if args.arm == "smoke":
            cmd += SMOKE_ARGS
            if args.wandb_only is not None and tag not in args.wandb_only:
                cmd.append("--no-wandb")
        if args.arm == "pricing":
            cmd += ["--calibrate", str(args.calibrate), "--no-wandb"]
        print(f"== {args.device}: {tag}", flush=True)
        with open(LOGS / args.arm / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, cwd=PROG / "code", stdout=log, stderr=subprocess.STDOUT)
        print(f"== {args.device}: {tag} exit {rc}", flush=True)
        if rc != 0:
            (claims / f"{tag}.failed-{args.device}").touch()
        os.rmdir(claims / tag)


if __name__ == "__main__":
    main()
