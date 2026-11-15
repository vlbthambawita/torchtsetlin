"""Load runs trained before W&B tracking existed into W&B (code/tracking.py), once.

For every run record (results/*_seed0.json and results/diagnostics/therm5_c*_b*.json) whose W&B run
(id ``tb-<tag>``) does not exist yet, creates it and logs the recorded epochs. A finished run gets
its summary; an unfinished one is left resumable, so the training run continues it from its
checkpoint. Safe to re-run: existing W&B runs are skipped.

Usage: python code/wandb_backfill.py [--dry-run]
"""

import argparse
import glob

import wandb

from common import RESULTS, read_json
from tracking import ENTITY, PROJECT, Tracker, run_id


def exists(api, tag):
    try:
        api.run(f"{ENTITY}/{PROJECT}/{run_id(tag)}")
        return True
    except (wandb.errors.CommError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    api = wandb.Api()
    files = sorted(glob.glob(str(RESULTS / "*_seed0.json")) +
                   glob.glob(str(RESULTS / "diagnostics" / "therm5_c*_b*.json")))
    for f in files:
        r = read_json(f)
        if r.get("kind") != "specialist":
            continue
        tag = str(f).split("/results/")[1].removesuffix(".json")
        if exists(api, tag):
            print(f"{tag}: already in W&B")
            continue
        print(f"{tag}: {len(r['curve'])} epochs, {r['status']}")
        if args.dry_run:
            continue
        t = Tracker(tag, r["spec"], r["hp"], env=r["env"])
        for row in r["curve"]:
            t.epoch(row)
        if r["status"] == "done":
            t.end({k: r.get(k) for k in ("status", "test_acc_final", "last25_mean", "last25_variance",
                                         "train_s_per_epoch_mean", "test_s_per_epoch_mean")}
                  | {"epochs_done": r["curve"][-1]["epoch"]})
        else:                                       # left resumable: training continues this run
            t.run.summary.update({"status": "paused", "epochs_done": r["curve"][-1]["epoch"]})
            t.run.finish(exit_code=0)


if __name__ == "__main__":
    main()
