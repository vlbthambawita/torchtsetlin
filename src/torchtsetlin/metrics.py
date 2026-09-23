"""Lightweight, device-agnostic metrics for Tsetlin machine outputs.

Vote sums behave like logits, so ``torchmetrics`` / scikit-learn work too; these helpers
avoid extra dependencies and stay on the GPU.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision_recall_f1",
    "classification_report",
    "regression_metrics",
    "multilabel_metrics",
    "vote_margin",
    "expected_calibration_error",
    "trustworthiness_curve",
    # dense / segmentation
    "segmentation_confusion_matrix",
    "pixel_accuracy",
    "iou",
    "iou_from_confusion",
    "mean_iou",
    "dice",
    "dice_from_confusion",
    "segmentation_metrics",
    "segmentation_report",
    "boundary_f1",
]


def _pred(votes_or_pred: Tensor) -> Tensor:
    return votes_or_pred.argmax(dim=1) if votes_or_pred.dim() == 2 else votes_or_pred


def accuracy(votes_or_pred: Tensor, y: Tensor) -> float:
    """Top-1 accuracy from vote sums ``(B, K)`` or predictions ``(B,)``."""
    return float((_pred(votes_or_pred) == y).float().mean())


def confusion_matrix(
    votes_or_pred: Tensor,
    y: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """``(K, K)`` confusion matrix with rows = true class, columns = predicted class.

    ``ignore_index`` drops every entry whose *true* label equals it (the "void" class of
    segmentation datasets), so it is counted neither as a hit nor as a miss.
    """
    pred = _pred(votes_or_pred)
    y = y.long()
    if ignore_index is not None:
        keep = y != int(ignore_index)
        pred, y = pred[keep], y[keep]
    if n_classes is None:
        n_classes = int(max(pred.max(), y.max())) + 1 if y.numel() else 1
    idx = y * n_classes + pred.long()
    return torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes)


def precision_recall_f1(cm: Tensor) -> Dict[str, Tensor]:
    """Per-class precision / recall / F1 from a confusion matrix."""
    cm = cm.to(torch.float64)
    tp = cm.diag()
    precision = tp / cm.sum(dim=0).clamp_min(1)
    recall = tp / cm.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {"precision": precision, "recall": recall, "f1": f1, "support": cm.sum(dim=1)}


def classification_report(
    votes_or_pred: Tensor, y: Tensor, class_names: Optional[Sequence[str]] = None, digits: int = 3
) -> str:
    """Text report similar to scikit-learn's ``classification_report``."""
    cm = confusion_matrix(votes_or_pred, y)
    m = precision_recall_f1(cm)
    K = cm.shape[0]
    names = list(class_names) if class_names is not None else [str(k) for k in range(K)]
    w = max(len(n) for n in names + ["macro avg"])
    lines = [f"{'':{w}}  precision  recall  f1-score  support"]
    for k in range(K):
        lines.append(
            f"{names[k]:{w}}  {m['precision'][k]:.{digits}f}      {m['recall'][k]:.{digits}f}   "
            f"{m['f1'][k]:.{digits}f}     {int(m['support'][k])}"
        )
    acc = float(cm.diag().sum() / cm.sum().clamp_min(1))
    lines.append("")
    lines.append(f"{'accuracy':{w}}  {acc:.{digits}f}")
    lines.append(
        f"{'macro avg':{w}}  {m['precision'].mean():.{digits}f}      {m['recall'].mean():.{digits}f}   "
        f"{m['f1'].mean():.{digits}f}     {int(m['support'].sum())}"
    )
    return "\n".join(lines)


def regression_metrics(pred: Tensor, y: Tensor) -> Dict[str, float]:
    """MAE, RMSE and R^2."""
    pred = pred.to(torch.float64).view(-1)
    y = y.to(torch.float64).view(-1)
    err = pred - y
    ss_res = float((err**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "mae": float(err.abs().mean()),
        "rmse": float((err**2).mean().sqrt()),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def multilabel_metrics(pred: Tensor, y: Tensor) -> Dict[str, float]:
    """Subset accuracy, Hamming accuracy and micro-F1 for Boolean ``(B, K)`` predictions."""
    pred = pred.bool()
    y = y.bool()
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "subset_accuracy": float((pred == y).all(dim=1).float().mean()),
        "hamming_accuracy": float((pred == y).float().mean()),
        "micro_f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def vote_margin(votes: Tensor) -> Tensor:
    """Difference between the best and second best vote sum per example (``(B,)``)."""
    top2 = votes.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def expected_calibration_error(proba: Tensor, y: Tensor, n_bins: int = 10) -> float:
    """ECE of ``(B, K)`` probabilities."""
    conf, pred = proba.max(dim=1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1, device=proba.device)
    ece = torch.zeros((), device=proba.device, dtype=torch.float64)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += m.float().mean().double() * (correct[m].mean() - conf[m].mean()).abs().double()
    return float(ece)


def trustworthiness_curve(votes: Tensor, y: Tensor, n_levels: int = 10) -> Dict[str, Tensor]:
    """Accuracy of the predictions whose confidence is at least each level (Chapter 7).

    Confidence is the winning vote sum. Returns ``{"confidence": (L,), "accuracy": (L,),
    "coverage": (L,)}``; a *trustworthy* model has monotonically increasing accuracy.
    """
    conf = votes.max(dim=1).values
    correct = (votes.argmax(dim=1) == y).float()
    levels = torch.quantile(conf.float(), torch.linspace(0, 0.9, n_levels, device=votes.device))
    acc, cov = [], []
    for c in levels:
        m = conf >= c
        acc.append(correct[m].mean())
        cov.append(m.float().mean())
    return {"confidence": levels, "accuracy": torch.stack(acc), "coverage": torch.stack(cov)}


# --------------------------------------------------------------------------------------
# Dense prediction (semantic segmentation)
# --------------------------------------------------------------------------------------
def _flat_labels(votes_or_pred: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    """Flatten a dense prediction / target pair to matching 1-D label vectors.

    Accepts vote maps ``(B, K, H, W)``, label maps ``(B, H, W)`` / ``(H, W)``, folded vote
    sums ``(N, K)`` and plain label vectors ``(N,)``.
    """
    p = votes_or_pred
    t = target.long().reshape(-1)
    if p.dim() == 4:  # (B, K, H, W) vote map
        p = p.argmax(dim=1)
    elif p.dim() == 2 and p.shape[0] == t.shape[0] and p.shape[1] != 1:
        p = p.argmax(dim=1)  # (N, K) folded vote sums
    p = p.reshape(-1).long()
    if p.shape[0] != t.shape[0]:
        raise ValueError(f"prediction has {p.shape[0]} pixels but target has {t.shape[0]}")
    return p, t


def segmentation_confusion_matrix(
    votes_or_pred: Tensor,
    target: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """``(K, K)`` pixel confusion matrix for dense predictions (see :func:`confusion_matrix`)."""
    pred, y = _flat_labels(votes_or_pred, target)
    return confusion_matrix(pred, y, n_classes=n_classes, ignore_index=ignore_index)


def pixel_accuracy(
    votes_or_pred: Tensor, target: Tensor, ignore_index: Optional[int] = None
) -> float:
    """Fraction of pixels predicted correctly."""
    pred, y = _flat_labels(votes_or_pred, target)
    if ignore_index is not None:
        keep = y != int(ignore_index)
        pred, y = pred[keep], y[keep]
    if y.numel() == 0:
        return float("nan")
    return float((pred == y).float().mean())


def iou_from_confusion(cm: Tensor) -> Tensor:
    """Per-class intersection-over-union from a confusion matrix; ``NaN`` for absent classes.

    A class is *absent* when it appears neither in the ground truth nor in the prediction,
    in which case its IoU is undefined rather than 0 — averaging a 0 there would punish a
    model for a class the image does not contain.
    """
    cm = cm.to(torch.float64)
    inter = cm.diag()
    union = cm.sum(dim=0) + cm.sum(dim=1) - inter
    return torch.where(union > 0, inter / union, torch.full_like(union, float("nan")))


def iou(
    votes_or_pred: Tensor,
    target: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Per-class IoU ``(K,)``, ``NaN`` where the class is absent from both sides."""
    return iou_from_confusion(
        segmentation_confusion_matrix(votes_or_pred, target, n_classes, ignore_index)
    )


def mean_iou(
    votes_or_pred: Tensor,
    target: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> float:
    """Mean IoU over the classes that are present (the usual ``mIoU``)."""
    v = iou(votes_or_pred, target, n_classes, ignore_index)
    return float(torch.nanmean(v)) if v.numel() else float("nan")


def dice_from_confusion(cm: Tensor) -> Tensor:
    """Per-class Dice / F1 coefficient from a confusion matrix (``2|A∩B| / (|A|+|B|)``)."""
    cm = cm.to(torch.float64)
    inter = cm.diag()
    total = cm.sum(dim=0) + cm.sum(dim=1)
    return torch.where(total > 0, 2 * inter / total, torch.full_like(total, float("nan")))


def dice(
    votes_or_pred: Tensor,
    target: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Per-class Dice coefficient ``(K,)`` — the medical-imaging convention for overlap."""
    return dice_from_confusion(
        segmentation_confusion_matrix(votes_or_pred, target, n_classes, ignore_index)
    )


def segmentation_metrics(
    votes_or_pred: Tensor,
    target: Tensor,
    n_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    ignore_index: Optional[int] = None,
    per_class: bool = True,
) -> Dict[str, float]:
    """Pixel accuracy, mean IoU, mean Dice and (optionally) per-class IoU in one dict."""
    cm = segmentation_confusion_matrix(votes_or_pred, target, n_classes, ignore_index)
    ious = iou_from_confusion(cm)
    dices = dice_from_confusion(cm)
    total = float(cm.sum())
    out: Dict[str, float] = {
        "pixel_accuracy": float(cm.diag().sum() / total) if total else float("nan"),
        "mean_iou": float(torch.nanmean(ious)),
        "mean_dice": float(torch.nanmean(dices)),
    }
    if per_class:
        names = list(class_names) if class_names is not None else [str(k) for k in range(cm.shape[0])]
        out.update({f"iou_{n}": float(v) for n, v in zip(names, ious)})
    return out


def segmentation_report(
    votes_or_pred: Tensor,
    target: Tensor,
    class_names: Optional[Sequence[str]] = None,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    digits: int = 3,
) -> str:
    """Per-class IoU / Dice / precision / recall table, plus pixel accuracy and mIoU."""
    cm = segmentation_confusion_matrix(votes_or_pred, target, n_classes, ignore_index)
    prf = precision_recall_f1(cm)
    ious, dices = iou_from_confusion(cm), dice_from_confusion(cm)
    K = cm.shape[0]
    names = list(class_names) if class_names is not None else [str(k) for k in range(K)]
    w = max(len(n) for n in names + ["macro avg"])
    lines = [f"{'':{w}}      IoU    Dice  precision  recall   pixels"]
    for k in range(K):
        lines.append(
            f"{names[k]:{w}}  {float(ious[k]):{7}.{digits}f} {float(dices[k]):{7}.{digits}f}  "
            f"{float(prf['precision'][k]):{9}.{digits}f}  {float(prf['recall'][k]):{6}.{digits}f}  "
            f"{int(prf['support'][k]):>7}"
        )
    total = float(cm.sum())
    lines += [
        "",
        f"{'pixel accuracy':{w}}  {(float(cm.diag().sum() / total) if total else float('nan')):{7}.{digits}f}",
        f"{'mean IoU':{w}}  {float(torch.nanmean(ious)):{7}.{digits}f}",
        f"{'mean Dice':{w}}  {float(torch.nanmean(dices)):{7}.{digits}f}",
    ]
    return "\n".join(lines)


def _boundary_mask(labels: Tensor) -> Tensor:
    """``(B, 1, H, W)`` mask of pixels whose 4-neighbourhood contains another label."""
    x = labels.unsqueeze(1).float()
    pad = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    diff = (
        (pad[:, :, 1:-1, :-2] != x)
        | (pad[:, :, 1:-1, 2:] != x)
        | (pad[:, :, :-2, 1:-1] != x)
        | (pad[:, :, 2:, 1:-1] != x)
    )
    return diff


def boundary_f1(
    votes_or_pred: Tensor,
    target: Tensor,
    tolerance: int = 2,
    ignore_index: Optional[int] = None,
) -> float:
    """Boundary F1 (BF) score: how well the predicted class *outlines* match the true ones.

    A predicted boundary pixel counts as a hit when a true boundary pixel lies within
    ``tolerance`` pixels (Chebyshev distance), and vice versa; the score is the harmonic mean
    of the two rates. Unlike IoU this is sensitive to thin structures and to how crisp the
    edges are, which is what a per-pixel model with no smoothing tends to get wrong.

    Both arguments must be dense: vote maps ``(B, K, H, W)`` or label maps ``(B, H, W)``.
    """
    pred = votes_or_pred.argmax(dim=1) if votes_or_pred.dim() == 4 else votes_or_pred
    pred, tgt = pred.long(), target.long()
    if pred.dim() == 2:
        pred, tgt = pred.unsqueeze(0), tgt.unsqueeze(0)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction {tuple(pred.shape)} and target {tuple(tgt.shape)} differ")
    valid = None
    if ignore_index is not None:
        valid = (tgt != int(ignore_index)).unsqueeze(1)
        # Ignored pixels must not create spurious edges: give them the predicted label.
        tgt = torch.where(tgt == int(ignore_index), pred, tgt)
    bp, bt = _boundary_mask(pred), _boundary_mask(tgt)
    if valid is not None:
        bp, bt = bp & valid, bt & valid
    k = 2 * int(tolerance) + 1
    pool = torch.nn.functional.max_pool2d
    dt = pool(bt.float(), k, stride=1, padding=int(tolerance)) > 0
    dp = pool(bp.float(), k, stride=1, padding=int(tolerance)) > 0
    n_p, n_t = float(bp.sum()), float(bt.sum())
    if n_p == 0 and n_t == 0:
        return float("nan")  # neither side has an edge: the score is undefined, not perfect
    if n_p == 0 or n_t == 0:
        return 0.0  # one side found no edge at all — a total miss, not an undefined case
    precision = float((bp & dt).sum()) / n_p
    recall = float((bt & dp).sum()) / n_t
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
