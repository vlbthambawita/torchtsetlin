"""Trainer for the neural arms, replicating the official IMN training loop.

From ``models/model.py`` of https://github.com/ArlindKadra/IMN: Adam(lr, weight_decay), linear
warm-up over 5 epochs, then ``CosineAnnealingWarmRestarts(T_0, T_mult=2)`` with
``T_0 = total_steps * (T_mult - 1) // (T_mult**restarts - 1)``, snapshots at steps
``T_0, 3 T_0, 7 T_0`` (counted from step 0, as in the official code), BCE with one logit for
binary tasks / CE otherwise, and an L1 penalty ``weight_norm * mean(|generated weights|)``
(bias row included, as in the official code). Augmentation is off (official default
``augmentation_probability=0``).

Deviation (DECISIONS.md D6): snapshots are ensembled by averaging *logits* (official: averaging
probabilities). Averaging logits of models that are linear in their generated parameters is the
same as averaging the generated parameters, so the ensemble is itself one local model per
example -- the model that is explained is exactly the model that predicts.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR, SequentialLR

NN_DEFAULTS: Dict = {
    "epochs": 500,
    "batch_size": 64,
    "lr": 0.01,
    "weight_decay": 0.01,
    "weight_norm": 0.1,      # L1 on generated weights (paper's lambda)
    "l1_mode": "all",        # "all" (official) | "active": only |w * feature| terms |
                             # "prior": |w - w_prior| (clause weights shrink toward the TM's votes)
    "dropout": 0.25,
    "width": 128,
    "n_blocks": 2,
    "restarts": 3,
    "t_mult": 2,
    "warmup_epochs": 5,
    "weight_norm_linear": None,  # separate L1 weight for the linear block (None = weight_norm)
    "max_steps": None,       # optional cap on optimizer steps (DECISIONS.md D7)
    "tm_init_scale": 1.0,    # clause_init = tm_weight * tm_init_scale / T (D9)
    "tm_init": True,
}


def nn_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = dict(NN_DEFAULTS)
    cfg.update(overrides or {})
    return cfg


class Batches:
    """GPU-resident tensors with optional clause matrix, sliced into mini-batches."""

    def __init__(self, X: Tensor, y: Tensor, C: Optional[Tensor]):
        self.X, self.y, self.C = X, y, C

    def __len__(self) -> int:
        return self.X.shape[0]

    def get(self, idx):
        c = None if self.C is None else self.C[idx].float()
        return self.X[idx], self.y[idx], c


def _loss_fn(n_out: int):
    if n_out == 1:
        bce = nn.BCEWithLogitsLoss()
        return lambda z, y: bce(z.squeeze(1), y.float())
    ce = nn.CrossEntropyLoss()
    return lambda z, y: ce(z, y)


def _l1(model, parts: Dict[str, Tensor], x: Tensor, c: Optional[Tensor], mode: str,
        lam: float, lam_lin: Optional[float] = None) -> Tensor:
    """``lam * mean(|.|)``; with ``lam_lin`` the linear block is weighted by ``lam_lin`` instead
    (same normalisation, so ``lam_lin == lam`` reproduces the single-weight penalty)."""
    w = parts["w"]
    if mode == "all":
        a = w.abs()
    elif mode == "prior":
        a = (w - model.w_prior).abs()
    else:
        feats = model.local_features(x, c if model.K else None)  # (B, P)
        a = (w * feats.unsqueeze(2)).abs()
    if lam_lin is None or not getattr(model, "use_linear", False):
        return lam * a.mean()
    d = model.d_in
    return (lam_lin * a[:, :d].sum() + lam * a[:, d:].sum()) / a.numel()


def fit(model: nn.Module, train: Batches, cfg: Dict, log=print) -> Dict:
    """Train in place; returns ``{"snapshots": [state_dict...], "history": [...], ...}``."""
    dev = train.X.device
    n = len(train)
    bs = int(cfg["batch_size"])
    steps_per_epoch = max(1, (n + bs - 1) // bs)
    epochs = int(cfg["epochs"])
    if cfg.get("max_steps"):
        epochs = max(1, min(epochs, int(cfg["max_steps"]) // steps_per_epoch))
    total = epochs * steps_per_epoch
    t_mult, restarts = int(cfg["t_mult"]), int(cfg["restarts"])
    T0 = max((total * (t_mult - 1)) // (t_mult ** restarts - 1), 1)
    warm = int(cfg["warmup_epochs"]) * steps_per_epoch
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    s1 = LambdaLR(opt, lr_lambda=lambda step: float(step / max(1, warm)))
    s2 = CosineAnnealingWarmRestarts(opt, T0, t_mult)
    sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[warm])
    snap_at = {T0, (t_mult + 1) * T0, (t_mult ** 2 + t_mult + 1) * T0}
    loss_fn = _loss_fn(model.n_out if hasattr(model, "n_out") else model.head.out_features)
    lam = float(cfg["weight_norm"]) if getattr(model, "interpretable", False) else 0.0
    snapshots: List[Dict] = []
    hist = []
    step = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() == 1:  # BatchNorm cannot train on a single example
                continue
            x, y, c = train.get(idx)
            z, parts = model(x, c)
            loss = loss_fn(z, y)
            if lam > 0:
                loss = loss + _l1(model, parts, x, c, cfg["l1_mode"], lam, cfg.get("weight_norm_linear"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += float(loss.detach()) * idx.numel()
            cnt += idx.numel()
            if step in snap_at:
                snapshots.append(copy.deepcopy(model.state_dict()))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            hist.append({"epoch": epoch, "train_loss": tot / max(cnt, 1), "time": time.time() - t0})
        if epoch % 100 == 0:
            log(f"    epoch {epoch}/{epochs} loss {tot / max(cnt, 1):.4f} ({time.time() - t0:.0f}s)")
    if not snapshots:  # very short runs: fall back to the final model
        snapshots.append(copy.deepcopy(model.state_dict()))
    return {"snapshots": snapshots, "history": hist, "train_time": time.time() - t0,
            "epochs": epochs, "steps": step, "T0": T0}


@torch.no_grad()
def ensemble_forward(model: nn.Module, snapshots: List[Dict], X: Tensor, C: Optional[Tensor],
                     chunk: int = 4096, return_w: bool = False):
    """Snapshot-averaged logits ``(N, n_out)`` (and averaged generated weights ``(N, P, n_out)``)."""
    z_acc, w_acc = None, None
    with torch.no_grad():  # bound the generated (rows, P, n_out) tensor to ~64M elements per chunk
        model.eval()
        per_row = model(X[:1], None if C is None else C[:1].float())[1].get("w", torch.empty(0)).numel()
    chunk = max(1, min(chunk, 64_000_000 // max(per_row, 1)))
    for sd in snapshots:
        model.load_state_dict(sd)
        model.eval()
        zs, ws = [], []
        for i in range(0, X.shape[0], chunk):
            c = None if C is None else C[i:i + chunk].float()
            z, parts = model(X[i:i + chunk], c)
            zs.append(z)
            if return_w:
                ws.append(parts["w"])
        z = torch.cat(zs, 0)
        z_acc = z if z_acc is None else z_acc + z
        if return_w:
            w = torch.cat(ws, 0)
            w_acc = w if w_acc is None else w_acc + w
    k = float(len(snapshots))
    if return_w:
        return z_acc / k, w_acc / k
    return z_acc / k


def logits_to_proba(z: Tensor) -> np.ndarray:
    if z.shape[1] == 1:
        return torch.sigmoid(z[:, 0]).cpu().numpy()
    return torch.softmax(z, 1).cpu().numpy()
