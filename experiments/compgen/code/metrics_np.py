"""The benchmark's clause metrics in plain NumPy, with the reference script's formulas.

``clause_out``: (N, C) Boolean clause outputs in prediction mode; ``target``: (C,) the label each
clause votes for (reference: 1 for a non-negative clause weight, else 0; torchtsetlin with two
classes: the clause's class for a positive clause, the other class for a negative one).

Used by ``run_ref.py`` (the reference library's environment has no torch) and as the naive
reference of check M1. Percentiles are NumPy's default (linear interpolation), as in the script.
"""

import numpy as np

PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


def clause_metrics(clause_out, y, target):
    """Per clause: precision (0 if it never matches) and recall; per example: the number of
    matching clauses and the highest precision among them (0 if none matches)."""
    co = np.asarray(clause_out) > 0
    y = np.asarray(y).astype(np.int64)
    C = co.shape[1]
    precision, recall = np.zeros(C), np.zeros(C)
    for i in range(C):
        hit = co[:, i]
        agree = (y[hit] == target[i]).sum()
        precision[i] = agree / hit.sum() if hit.sum() > 0 else 0.0
        recall[i] = agree / max(1, (y == target[i]).sum())
    votes = co.sum(axis=1)
    max_precision = np.where(co, precision[None, :], 0.0).max(axis=1)
    return precision, recall, votes, max_precision


def summarise(precision, recall, votes, max_precision):
    """The scalars the reference script prints and writes (percentages)."""
    out = {"coverage": 100.0 * float((votes > 0).mean()), "votes_mean": float(votes.mean()),
           "precision_mean": 100.0 * float(precision.mean()), "recall_mean": 100.0 * float(recall.mean())}
    for name, v in (("precision", precision), ("recall", recall), ("max_precision", max_precision)):
        for p in PERCENTILES:
            out[f"{name}_p{p}"] = 100.0 * float(np.percentile(v, p))
    hist = np.bincount(votes.astype(np.int64))
    out["votes_hist"] = {int(k): float(c) / len(votes) for k, c in enumerate(hist) if c}
    return out
