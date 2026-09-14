"""Lightweight, device-agnostic metrics for Tsetlin machine outputs.

Vote sums behave like logits, so ``torchmetrics`` / scikit-learn work too; these helpers
avoid extra dependencies and stay on the GPU.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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
]


def _pred(votes_or_pred: Tensor) -> Tensor:
    return votes_or_pred.argmax(dim=1) if votes_or_pred.dim() == 2 else votes_or_pred


def accuracy(votes_or_pred: Tensor, y: Tensor) -> float:
    """Top-1 accuracy from vote sums ``(B, K)`` or predictions ``(B,)``."""
    return float((_pred(votes_or_pred) == y).float().mean())


def confusion_matrix(votes_or_pred: Tensor, y: Tensor, n_classes: Optional[int] = None) -> Tensor:
    """``(K, K)`` confusion matrix with rows = true class, columns = predicted class."""
    pred = _pred(votes_or_pred)
    if n_classes is None:
        n_classes = int(max(pred.max(), y.max())) + 1
    idx = y.long() * n_classes + pred.long()
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
