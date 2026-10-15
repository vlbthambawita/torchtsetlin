"""Evaluation metrics shared by every arm."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def to_proba(p: np.ndarray, n_classes: int) -> np.ndarray:
    """Accept ``(N,)`` P(class 1) for binary tasks or ``(N, C)``; return ``(N, C)``."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    if p.shape[1] != n_classes:
        raise ValueError(f"expected {n_classes} probability columns, got {p.shape[1]}")
    p = np.clip(p, 1e-7, 1.0)
    return p / p.sum(1, keepdims=True)


def classification_metrics(y: np.ndarray, p: np.ndarray, n_classes: int) -> Dict[str, float]:
    """AUROC (binary; macro one-vs-rest for multi-class, see DECISIONS.md D5), accuracy, log-loss."""
    P = to_proba(p, n_classes)
    labels = np.arange(n_classes)
    try:
        if n_classes == 2:
            auroc = roc_auc_score(y, P[:, 1])
        else:
            present = np.unique(y)
            if len(present) < n_classes:  # tiny splits can miss a class; score present ones
                auroc = roc_auc_score(y, P[:, present] / P[:, present].sum(1, keepdims=True),
                                      multi_class="ovr", average="macro", labels=present)
            else:
                auroc = roc_auc_score(y, P, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        auroc = float("nan")
    return {
        "auroc": float(auroc),
        "accuracy": float(accuracy_score(y, P.argmax(1))),
        "log_loss": float(log_loss(y, P, labels=labels)),
    }
