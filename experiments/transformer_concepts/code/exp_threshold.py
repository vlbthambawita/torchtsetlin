"""PROOF_PLAN Part II, D2: the specificity a pair clause needs grows with the number of pairs.

Prediction, written before the runs (THEORY C2 with the library default
``boost_true_positive=True``, i.e. p_mem = 1): an under-specified clause ``[q = M]`` on T4 matches
the L pairs (p, j) of the marker; the key literal ``[k0 = b]`` is true on the drawn pair with
probability rho = 1/L + (1 - 1/L) * pi, where pi is the rate of b among fillers (0 without
distractors, 1/(V-1) with). It drifts towards inclusion iff rho > 1/(s + 1), i.e.
s > s_pred = 1/rho - 1. So s* (smallest s that learns) should track s_pred: ~ L - 1 without
distractors, saturating near V - 1 with them. Control: the conv TM with kernel d + 1 has one
matching window per example, no dilution, so its s* should not grow with L.

    python -m experiments.transformer_concepts.code.exp_threshold jobs   # writes jobs_threshold.json
"""
from __future__ import annotations

import json
import sys

from experiments.transformer_concepts.code.runner import ROOT

LS = (8, 12, 16, 24, 32)
SS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SEEDS = (1, 2)


def jobs():
    out = []
    for seed in SEEDS:
        for L in LS:
            for task, arms in (("T4n", ("A2_pair_w2", "B3_conv_kd1")), ("T4", ("A2_pair_w2",))):
                for arm in arms:
                    for s in SS:
                        out.append(dict(stage="threshold", key=f"{task}_L{L}_{arm}_s{s}_seed{seed}",
                                        task=task, L=L, arm_name=arm, s=float(s), seed=seed,
                                        n_train=8000, n_valid=2000, epochs=15, patience=15, device="cuda"))
    return out


def s_pred(L: int, V: int = 16, distractors: bool = False) -> float:
    pi = 1.0 / (V - 1) if distractors else 0.0
    rho = 1.0 / L + (1.0 - 1.0 / L) * pi
    return 1.0 / rho - 1.0


if __name__ == "__main__":
    if sys.argv[1] == "jobs":
        j = jobs()
        (ROOT / "jobs_threshold.json").write_text(json.dumps(j))
        print(len(j), "jobs")
