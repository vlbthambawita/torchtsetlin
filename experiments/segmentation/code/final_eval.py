"""Evaluate frozen E4 checkpoints ONCE on the sealed final split (PLAN section 8).

    python code/final_eval.py --dataset kvasir [--device cuda:1]

Refuses to run unless DECISIONS.md contains the freeze record for the dataset
(``D-011`` for Kvasir, ``D-012`` for ACDC) and never re-evaluates a run that already has a
FINAL record. Writes ``results/FINAL_<run_id>.json`` with native-grid metrics, per-image
sufficient statistics (for the paired bootstrap) and, for ACDC, per-volume 3D metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402
import record  # noqa: E402
import run as R  # noqa: E402

FREEZE = {"kvasir": "## D-011", "acdc": "## D-012"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    with open(os.path.join(record.HERE, "DECISIONS.md")) as f:
        if FREEZE[a.dataset] not in f.read():
            raise SystemExit("no freeze record in DECISIONS.md; refusing to unseal")
    recs = record.load_records(f"E4_{a.dataset}_F_")
    for rid, rec in sorted(recs.items()):
        out = os.path.join(record.RESULTS, f"FINAL_{rid}.json")
        if rec["status"] != "completed" or os.path.exists(out):
            continue
        t = time.time()
        cfg, d, scores = evalio.predict(rec, "final", a.device)
        nat = R.eval_native(cfg, scores, d, "final")
        p = os.path.join(record.ARTIFACTS, "votes", f"{rid}_final.pt")
        torch.save(scores, p)
        row = {"run_id": f"FINAL_{rid}", "status": "completed", "source_run": rid, "split": "final",
               "env": record.environment(a.device), "final_native": nat["summary"], "final_native_stats": nat["stats"],
               "final_votes": os.path.relpath(p, record.HERE), "eval_s": time.time() - t}
        if "volumes" in nat:
            row["final_volumes"] = nat["volumes"]
        record.write_record(row)
        print(rid, round(nat["summary"]["miou"], 4), flush=True)


if __name__ == "__main__":
    main()
