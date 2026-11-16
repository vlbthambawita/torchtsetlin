"""Work through one phase of the plan on one GPU, cheapest jobs first.

A job whose record says ``done`` is skipped; a failed job leaves ``.cache/claims/<phase>/<arm>-<tag>.failed``
(delete it to retry); an interrupted job resumes from its checkpoint when the runner is restarted.

GPU: the RTX 3080 only; the RTX 3090 is refused unless ``--allow-3090`` (only when the user asks).
Phases beyond ``smoke`` and ``pricing`` need the user's go-ahead.

Usage: nohup python code/queue_runner.py --phase pilot --device cuda:1 > logs/q_pilot.log 2>&1 &
       python code/queue_runner.py --phase main --seeds 0 1 2 --dry-run
"""

import argparse
import os
import subprocess
import sys

import torch
from common import CACHE, LOGS, MEMBERS, PROG, RESULTS, RULES, TEAMS, pilot_choice, read_json
from train import spec_of

COST = {"A10": 1, "H3": 2, "H4": 2, "H5": 3, "H10": 4}          # order only


def jobs(phase, seeds, jown=False):
    """[(arm, member, team, rule, seed)]"""
    out = []
    if phase in ("smoke", "pricing", "pilot"):
        out += [("I", m, None, None, 0) for m in MEMBERS]
        out += [("J", None, t, r, 0) for t in TEAMS for r in RULES]
        if phase == "smoke":
            out += [("Jown", None, "S", "sum", 0), ("W", "H10", "S", "sum", 0), ("W", "H10", "S", "one", 0)]
        if phase == "pricing":                       # W: every member a pilot could pick
            out += [("W", m, t, "sum", 0) for t, ms in TEAMS.items() for m in ms]
        return out
    ch = pilot_choice()
    if ch is None:
        raise SystemExit("no pilot choice yet: python code/report.py --phase pilot --choose")
    for seed in seeds:
        if jown:
            out += [("Jown", None, "S", r, seed) for r in ch["S"]["rules"]]
            continue
        out += [("I", m, None, None, seed) for m in MEMBERS]
        for t in TEAMS:
            for r in ch[t]["rules"]:
                out += [("J", None, t, r, seed), ("W", ch[t]["w_member"], t, r, seed)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["smoke", "pricing", "pilot", "main"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--jown", action="store_true", help="main: the own-negative ablation on Team S")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--allow-3090", action="store_true", help="only when the user asked for the 3090")
    ap.add_argument("--calibrate", type=int, default=300, help="pricing: updates to time")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    gpu = torch.cuda.get_device_name(torch.device(args.device))
    if "3090" in gpu and not args.allow_3090:
        raise SystemExit(f"{args.device} is the {gpu}: this programme runs on the 3080 only")
    todo = jobs(args.phase, args.seeds, args.jown)
    todo.sort(key=lambda j: (j[4], {"I": 0, "W": 1, "J": 2, "Jown": 3}[j[0]],
                             COST[j[1]] if j[1] else sum(COST[m] for m in TEAMS[j[2]])))
    claims = CACHE / "claims" / args.phase
    claims.mkdir(parents=True, exist_ok=True)
    for arm, member, team, rule, seed in todo:
        tag = spec_of(arm, member, team, rule, seed)[0]
        rec = RESULTS / args.phase / arm / f"{tag}.json"
        if rec.exists() and (args.phase == "pricing" or read_json(rec).get("status") == "done"):
            continue
        failed = claims / f"{arm}-{tag}.failed"
        if failed.exists():
            continue
        if args.dry_run:
            print(arm, tag)
            continue
        cmd = [sys.executable, "train.py", "--phase", args.phase, "--arm", arm, "--seed", str(seed),
               "--device", args.device]
        for k, v in (("--member", member), ("--team", team), ("--rule", rule)):
            if v:
                cmd += [k, v]
        if args.phase == "smoke":
            cmd += ["--subset", "2000", "--no-wandb"]
        if args.phase == "pricing":
            cmd += ["--calibrate", str(args.calibrate), "--no-wandb"]
        if arm == "I" and (args.phase == "smoke" or (args.phase == "main" and seed == 0)):
            cmd.append("--keep-epochs")              # the composite-best model (activation maps: seed 0)
        (LOGS / args.phase).mkdir(parents=True, exist_ok=True)
        print(f"== {args.device}: {arm}/{tag}", flush=True)
        with open(LOGS / args.phase / f"{arm}-{tag}.log", "a") as log:
            rc = subprocess.call(cmd, cwd=PROG / "code", stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
        print(f"== {args.device}: {arm}/{tag} exit {rc}", flush=True)
        if rc != 0:
            failed.touch()


if __name__ == "__main__":
    main()
