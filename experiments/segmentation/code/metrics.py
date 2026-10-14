"""Evaluation independent of the training loop (PLAN section 6).

All functions take integer label maps (numpy or torch) and an optional ignore value. The
dataset-level statistics are built from *sufficient statistics per image* (confusion matrix,
boundary hit counts) so the paired bootstrap can recompute any metric from resampled IDs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import ndimage

IGNORE = 255


def confusion(pred: np.ndarray, tgt: np.ndarray, K: int, ignore: Optional[int] = IGNORE) -> np.ndarray:
    p, t = np.asarray(pred).reshape(-1).astype(np.int64), np.asarray(tgt).reshape(-1).astype(np.int64)
    if ignore is not None:
        ok = t != ignore
        p, t = p[ok], t[ok]
    return np.bincount(t * K + p, minlength=K * K).reshape(K, K)


def iou_from_cm(cm: np.ndarray) -> np.ndarray:
    """Per-class IoU; classes with zero union are NaN (absent, excluded from the mean)."""
    tp = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, tp / union, np.nan)


def dice_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


# ---------------------------------------------------------------------------- boundaries
def _class_boundary(lbl: np.ndarray, k: int) -> np.ndarray:
    """Inner 4-neighbour boundary of the binary mask ``lbl == k`` (image border is not an edge)."""
    m = lbl == k
    er = ndimage.binary_erosion(m, structure=ndimage.generate_binary_structure(2, 1), border_value=1)
    return m & ~er


def boundary_counts(pred: np.ndarray, tgt: np.ndarray, K: int, tol: int, ignore: Optional[int] = IGNORE) -> np.ndarray:
    """Class-aware boundary sufficient statistics for one 2D image.

    Returns ``(K, 4)``: predicted boundary pixels, matched predicted, true boundary pixels,
    matched true. A predicted boundary of class ``k`` is matched only by a *class-k* true
    boundary within Chebyshev distance ``tol`` (and vice versa). Boundary pixels whose
    ``(2 tol + 1)`` neighbourhood touches void/padding are excluded from both sides.
    """
    pred, tgt = np.asarray(pred), np.asarray(tgt)
    valid = np.ones_like(tgt, bool) if ignore is None else tgt != ignore
    tg = np.where(valid, tgt, -1)
    pr = np.where(valid, pred, -1)
    sz = 2 * tol + 1
    near_void = ndimage.maximum_filter((~valid).astype(np.uint8), size=sz) > 0
    out = np.zeros((K, 4), np.int64)
    for k in range(K):
        bp = _class_boundary(pr, k) & ~near_void
        bt = _class_boundary(tg, k) & ~near_void
        dt = ndimage.maximum_filter(bt.astype(np.uint8), size=sz) > 0
        dp = ndimage.maximum_filter(bp.astype(np.uint8), size=sz) > 0
        out[k] = (bp.sum(), (bp & dt).sum(), bt.sum(), (bt & dp).sum())
    return out


def boundary_f1_from_counts(c: np.ndarray) -> np.ndarray:
    """Per-class BF1 from summed ``(K, 4)`` counts; NaN when neither side has a boundary."""
    out = np.full(c.shape[0], np.nan)
    for k in range(c.shape[0]):
        n_p, h_p, n_t, h_t = c[k]
        if n_p == 0 and n_t == 0:
            continue
        if n_p == 0 or n_t == 0:
            out[k] = 0.0
            continue
        P, R = h_p / n_p, h_t / n_t
        out[k] = 0.0 if P + R == 0 else 2 * P * R / (P + R)
    return out


# ------------------------------------------------------------------------------ records
class SegEvaluator:
    """Accumulates per-image sufficient statistics; ``summary()`` computes the reported metrics."""

    def __init__(self, K: int, tols: Sequence[int] = (1, 2), ignore: Optional[int] = IGNORE, boundaries: bool = True):
        self.K, self.tols, self.ignore, self.boundaries = K, tuple(tols), ignore, boundaries
        self.cms: List[np.ndarray] = []
        self.bcs: Dict[int, List[np.ndarray]] = {t: [] for t in self.tols}
        self.ids: List = []

    def add(self, pred, tgt, id_=None, norm_tol=None):
        """``norm_tol``: per-image pixel tolerance for the tolerance key ``"n"`` (the
        resolution-normalised tolerance: 2 px of the 128-px model grid, in native pixels)."""
        pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        tgt = tgt.cpu().numpy() if isinstance(tgt, torch.Tensor) else np.asarray(tgt)
        self.cms.append(confusion(pred, tgt, self.K, self.ignore))
        if self.boundaries:
            for t in self.tols:
                tol = int(norm_tol) if t == "n" else t
                self.bcs[t].append(boundary_counts(pred, tgt, self.K, tol, self.ignore))
        self.ids.append(id_ if id_ is not None else len(self.ids))

    def stats(self) -> Dict:
        return {"ids": list(self.ids), "cm": np.stack(self.cms).tolist(),
                "bc": {str(t): np.stack(v).tolist() for t, v in self.bcs.items()} if self.boundaries else {}}

    def summary(self) -> Dict:
        return summarize(np.stack(self.cms), {t: np.stack(v) for t, v in self.bcs.items()} if self.boundaries else {}, self.K)


def _nanmean(a) -> Optional[float]:
    a = np.asarray(a, np.float64)
    return None if np.all(np.isnan(a)) else float(np.nanmean(a))


def summarize(cms: np.ndarray, bcs: Dict[int, np.ndarray], K: int, idx: Optional[np.ndarray] = None) -> Dict:
    """Metrics from stacked per-image statistics, optionally restricted/resampled by ``idx``."""
    if idx is not None:
        cms = cms[idx]
        bcs = {t: v[idx] for t, v in bcs.items()}
    cm = cms.sum(0)
    iou = iou_from_cm(cm)
    dice = dice_from_cm(cm)
    out = {
        "miou": _nanmean(iou),
        "iou": [None if np.isnan(v) else float(v) for v in iou],
        "dice": [None if np.isnan(v) else float(v) for v in dice],
        "macro_dice": _nanmean(dice),
        "pixel_acc": float(np.trace(cm) / max(1, cm.sum())),
        "absent_classes": [int(k) for k in range(K) if np.isnan(iou[k])],
    }
    if K > 1:
        out["fg_miou"] = _nanmean(iou[1:])
        out["fg_dice"] = _nanmean(dice[1:])
    # mean per-image foreground dice / IoU (binary: class 1; multiclass: macro over fg present)
    per_img_d, per_img_i = [], []
    for c in cms:
        d, i = dice_from_cm(c)[1:], iou_from_cm(c)[1:]
        if not np.all(np.isnan(d)):
            per_img_d.append(np.nanmean(d))
            per_img_i.append(np.nanmean(i))
    out["mean_img_fg_dice"] = _nanmean(per_img_d) if per_img_d else None
    out["mean_img_fg_iou"] = _nanmean(per_img_i) if per_img_i else None
    for t, v in bcs.items():
        bf = boundary_f1_from_counts(v.sum(0))
        out[f"bf1_t{t}"] = [None if np.isnan(x) else float(x) for x in bf]
        out[f"bf1_t{t}_mean"] = _nanmean(bf)
        out[f"bf1_t{t}_fg_mean"] = _nanmean(bf[1:])
    return out


# ------------------------------------------------------------------------------ 3D ACDC
def dice_3d(pred: np.ndarray, gt: np.ndarray, k: int) -> Optional[float]:
    p, g = pred == k, gt == k
    if not p.any() and not g.any():
        return None  # both empty: undefined, counted
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))


def hd95_3d(pred: np.ndarray, gt: np.ndarray, k: int, spacing) -> Optional[float]:
    """Symmetric 95th-percentile Hausdorff distance (mm) between class-k surfaces.

    Convention: surface = mask minus its 6-connected erosion; distances from every surface
    voxel of each mask to the other surface (both directions pooled), 95th percentile.
    Undefined (None) if either mask is empty."""
    p, g = pred == k, gt == k
    if not p.any() or not g.any():
        return None
    st = ndimage.generate_binary_structure(3, 1)
    sp = p & ~ndimage.binary_erosion(p, st, border_value=0)
    sg = g & ~ndimage.binary_erosion(g, st, border_value=0)
    dg = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dp = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dg[sp], dp[sg]])
    return float(np.percentile(d, 95))


def acdc_volume_metrics(pred: np.ndarray, gt: np.ndarray, spacing) -> Dict:
    out = {}
    for k, name in ((1, "RV"), (2, "MYO"), (3, "LV")):
        out[f"dice_{name}"] = dice_3d(pred, gt, k)
        out[f"hd95_{name}"] = hd95_3d(pred, gt, k, spacing)
    return out


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap(
    stats_a: List[Dict], stats_b: List[Dict], K: int, metric: str = "miou", n_boot: int = 2000, seed: int = 20261013
) -> Dict:
    """Paired hierarchical bootstrap of ``metric(b) - metric(a)``.

    ``stats_a[s]`` / ``stats_b[s]`` are evaluator stats of the *same* training seed index ``s``
    on the *same* evaluation IDs (groups). Resample seed pairs, then group IDs with
    replacement, and recompute each metric from the selected confusion counts."""
    rng = np.random.default_rng(seed)
    S = len(stats_a)
    assert S == len(stats_b) and S > 0
    A = [np.asarray(s["cm"]) for s in stats_a]
    B = [np.asarray(s["cm"]) for s in stats_b]
    groups = stats_a[0].get("groups") or stats_a[0]["ids"]
    for s in stats_a + stats_b:
        assert (s.get("groups") or s["ids"]) == groups, "unpaired evaluation IDs"
    uniq, inv = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]

    def metric_of(cms, idx):
        cm = (cms if idx is None else cms[idx]).sum(0)
        if metric == "miou":
            return _nanmean(iou_from_cm(cm))
        if metric == "fg_dice":
            return _nanmean(dice_from_cm(cm)[1:])
        if metric == "fg_miou":
            return _nanmean(iou_from_cm(cm)[1:])
        return summarize(cms, {}, K, idx)[metric]

    point = float(np.mean([metric_of(B[s], None) - metric_of(A[s], None) for s in range(S)]))
    diffs = []
    for _ in range(n_boot):
        ss = rng.integers(0, S, S)
        gs = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in gs])
        diffs.append(np.mean([metric_of(B[s], idx) - metric_of(A[s], idx) for s in ss]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": metric, "diff": point, "ci95": [float(lo), float(hi)], "n_boot": n_boot, "seed": seed, "n_seeds": S}
