"""Stage B / baseline classifier loop with validation checkpointing (spec §8), and metrics.

Equivalent to ``tt.Trainer`` + ``EarlyStopping(patience=10, restore_best=True)``: strict ``>``
improvement, earliest best epoch kept, stop after ``patience`` epochs without a strictly higher
validation accuracy (D15: own loop so that evaluation time is not charged to training time).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from .common import sync
from .tm_adapter import TMAdapter


def metrics_from_votes(votes: Tensor, y: Tensor) -> Dict:
    pred = votes.argmax(1).cpu().numpy()
    yt = y.cpu().numpy()
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (yt, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"accuracy": float((pred == yt).mean()), "errors": int((pred != yt).sum()), "n": int(len(yt)),
            "macro_f1": float(f1.mean()), "per_class_accuracy": rec.tolist(), "confusion": cm.tolist()}


def fit_classifier(head: TMAdapter, Xtr: Tensor, ytr: Tensor, Xva: Tensor, yva: Tensor, *,
                   max_epochs: int = 50, patience: int = 10, batch_size: int = 32,
                   log: Callable = print, time_limit_s: Optional[float] = None) -> Dict:
    dev = head.device
    best_acc, best_epoch, stale = -1.0, -1, 0
    best_state = None
    hist = []
    train_time = 0.0
    stop = "max_epochs"
    t_start = time.time()
    for ep in range(max_epochs):
        sync(dev)
        t0 = time.time()
        head.fit_epoch(Xtr, ytr, shuffle=True, batch_size=batch_size)
        sync(dev)
        dt = time.time() - t0
        train_time += dt
        t1 = time.time()
        acc = float((head.votes(Xva).argmax(1) == yva.to(dev)).double().mean())
        et = time.time() - t1
        improved = acc > best_acc
        if improved:
            best_acc, best_epoch, stale = acc, ep, 0
            best_state = head.state_dict()
        else:
            stale += 1
        hist.append({"epoch": ep + 1, "val_accuracy": acc, "epoch_train_s": dt, "eval_s": et,
                     "cum_train_s": train_time, "improved": improved})
        log(f"  epoch {ep + 1:2d} val_acc={acc:.4f} best={best_acc:.4f}@{best_epoch + 1} "
            f"train={dt:.1f}s eval={et:.1f}s")
        if stale >= patience:
            stop = "patience"
            break
        if time_limit_s is not None and time.time() - t_start > time_limit_s:
            stop = "time_limit"
            break
    head.load_state(best_state)
    return {"best_val_accuracy": best_acc, "best_epoch": best_epoch + 1, "epochs_run": len(hist),
            "stop_reason": stop, "history": hist, "train_time_s": train_time}
