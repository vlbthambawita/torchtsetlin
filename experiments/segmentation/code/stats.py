"""Paired hierarchical bootstrap for arbitrary linear contrasts of arms (PLAN section 6).

``contrast(arms, weights, K, metric)``: ``arms`` is a list of per-arm lists of per-seed evaluator
stats (same seed order across arms, same evaluation IDs); the statistic is
``sum_a w_a * metric(arm a)`` averaged over seeds. Resampling: seed indices with replacement,
then evaluation groups with replacement (shared across arms and seeds within a draw).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

import metrics as M


def _metric_fn(metric: str, K: int):
    def f(cms: np.ndarray, bcs, idx):
        sel = cms if idx is None else cms[idx]
        cm = sel.sum(0)
        if metric == "miou":
            return M._nanmean(M.iou_from_cm(cm))
        if metric == "fg_dice":
            return M._nanmean(M.dice_from_cm(cm)[1:])
        if metric == "fg_iou":
            return M._nanmean(M.iou_from_cm(cm)[1:])
        if metric.startswith("bf1_"):  # e.g. bf1_n_fg  -> tolerance key n, foreground mean
            tol = metric.split("_")[1]
            b = bcs[tol] if idx is None else bcs[tol][idx]
            bf = M.boundary_f1_from_counts(b.sum(0))
            return M._nanmean(bf[1:])
        raise ValueError(metric)
    return f


def contrast(arms: Sequence[List[Dict]], weights: Sequence[float], K: int, metric: str = "miou",
             n_boot: int = 2000, seed: int = 20261013) -> Dict:
    S = min(len(a) for a in arms)
    if S == 0:
        return {"n_seeds": 0}
    groups = arms[0][0].get("groups") or arms[0][0]["ids"]
    for a in arms:
        for s in a[:S]:
            assert (s.get("groups") or s["ids"]) == groups, "unpaired evaluation IDs"
    uniq, inv = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    CM = [[np.asarray(s["cm"]) for s in a[:S]] for a in arms]
    BC = [[{k: np.asarray(v) for k, v in s.get("bc", {}).items()} for s in a[:S]] for a in arms]
    f = _metric_fn(metric, K)

    def stat(seed_idx, idx):
        vals = []
        for s in seed_idx:
            vals.append(sum(w * f(CM[a][s], BC[a][s], idx) for a, w in enumerate(weights)))
        return float(np.mean(vals))

    point = stat(range(S), None)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        ss = rng.integers(0, S, S)
        gs = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in gs])
        draws.append(stat(ss, idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    per_seed = [stat([s], None) for s in range(S)]
    return {"metric": metric, "estimate": point, "ci95": [float(lo), float(hi)], "n_seeds": S,
            "n_groups": len(uniq), "per_seed": per_seed, "n_boot": n_boot, "boot_seed": seed}
