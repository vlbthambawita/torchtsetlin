"""Included literals per clause and clause-weight magnitudes from torchtsetlin checkpoints,
for comparison with tmu (code/tmu_clause_stats.py). Writes results/ttl_clausestats.json.

Usage: python code/clause_stats_torchtsetlin.py C3_seed0 C3_seed0_nocap C3_seed0_bs10 C3_seed0_seq
"""
import sys

import numpy as np
import torch
from common import CACHE, RESULTS, new_record, write_json


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return {"mean": float(a.mean()), "q0": q[0], "q5": q[1], "q25": q[2], "q50": q[3], "q75": q[4],
            "q95": q[5], "q100": q[6], "frac_gt_32": float((a > 32).mean()), "frac_eq_0": float((a == 0).mean())}


def main(tags):
    runs = {}
    for tag in tags:
        st = torch.load(CACHE / "ckpt" / f"{tag}.pt", map_location="cpu", weights_only=False)
        m, n_states = st["model"], int(st["rec"]["hp"]["n_states"])
        inc = (m["ta_state"] >= n_states).sum(dim=1).numpy()
        w = m["weights"].numpy() if "weights" in m else None
        runs[tag] = {"epoch": int(st["epoch"]), "test_acc": st["rec"]["curve"][-1]["test_acc"],
                     "hp": {k: st["rec"]["hp"].get(k) for k in ("batch_size", "feedback_mode", "max_included_literals")},
                     "included_literals": stats(inc),
                     "weight_magnitude": stats(w) if w is not None else None}
        r = runs[tag]
        print(f"{tag:20s} ep {r['epoch']:3d} acc {r['test_acc']:.2f} | size q50 {r['included_literals']['q50']:.0f} "
              f"mean {r['included_literals']['mean']:.1f} >32 {r['included_literals']['frac_gt_32']:.3f} "
              f"empty {r['included_literals']['frac_eq_0']:.3f} | weight q50 "
              f"{r['weight_magnitude']['q50']:.0f} mean {r['weight_magnitude']['mean']:.1f}")
    write_json(new_record("clausestats_torchtsetlin", runs=runs), RESULTS / "ttl_clausestats.json")


if __name__ == "__main__":
    main(sys.argv[1:])
