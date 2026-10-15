"""Explanations and interpretability metrics, uniform across arms.

Feature attribution (paper Eq. 2): the impact of feature ``m`` on example ``x`` is the change
of the explained output when ``x_m`` is set to 0 (= the training mean after standardization)
*with the local model held fixed*:

    phi_m(x) = z(x; w(x)) - z(x with x_m = 0; w(x))

For IMN's linear local model this is exactly ``w_m(x) x_m`` (tested). For a clause local model
the clauses are re-evaluated on the modified input (a thermometer bit of column m can flip,
switching clauses on or off) while the generated clause weights stay those of ``x``. Arms
without a local model (TM, TabResNet, GBDT, ...) use the same removal on the full model
(occlusion), so every arm is scored with the same protocol.

The explained output is the logit for binary tasks and the logit of the predicted class for
multi-class tasks (fixed per example from the unmodified prediction).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch
from torch import Tensor


class Explainable:
    """What :mod:`explain` needs from an arm.

    ``full(X)`` -> ``(N, n_out)`` logits of the whole model (``n_out = 1`` for binary tasks:
    the log-odds of class 1). Arms with a per-example local model also set
    ``local(Xmod, rows)`` -> logits of the *frozen* local models of examples ``rows``
    evaluated at ``Xmod`` (one modified input per row), plus ``weights(rows)`` returning the
    parts of those local models for size/fidelity metrics.
    """

    def __init__(self, full: Callable[[Tensor], Tensor],
                 local: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 parts: Optional[Callable[[Tensor], Dict[str, Tensor]]] = None,
                 clause_fn: Optional[Callable[[Tensor], Tensor]] = None):
        self.full, self.local, self.parts, self.clause_fn = full, local, parts, clause_fn


def _target(z: Tensor) -> Tensor:
    """Index of the explained output per example."""
    if z.shape[1] == 1:
        return torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
    return z.argmax(1)


@torch.no_grad()
def removal_attributions(ex: Explainable, X: Tensor, baseline: float = 0.0,
                         use_local: bool = True, chunk_rows: int = 65536) -> np.ndarray:
    """``(N, d)`` attributions ``phi_m(x)`` (see module docstring)."""
    N, d = X.shape
    z0 = ex.full(X)
    tgt = _target(z0)
    rows_all = torch.arange(N, device=X.device)
    local = use_local and ex.local is not None
    if local:
        base = ex.local(X, rows_all).gather(1, tgt[:, None])[:, 0]
    else:
        base = z0.gather(1, tgt[:, None])[:, 0]
    out = torch.empty(N, d, device=X.device)
    per = max(1, min(chunk_rows // d, 20_000_000 // (d * d)))  # masked copies hold per*d*d floats
    eye = torch.eye(d, device=X.device, dtype=torch.bool)
    for i in range(0, N, per):
        xs = X[i:i + per]
        b = xs.shape[0]
        Xm = xs.unsqueeze(1).expand(b, d, d).clone()
        Xm[:, eye] = baseline
        Xm = Xm.reshape(b * d, d)
        rows = rows_all[i:i + b].repeat_interleave(d)
        z = ex.local(Xm, rows) if local else ex.full(Xm)
        zt = z.gather(1, tgt[rows][:, None])[:, 0].view(b, d)
        out[i:i + b] = base[i:i + b, None] - zt
    return out.cpu().numpy()


def explanation_size(contrib: np.ndarray, mass: float = 0.9) -> np.ndarray:
    """Smallest number of terms whose |contribution| covers ``mass`` of the total, per row."""
    a = np.sort(np.abs(contrib), axis=1)[:, ::-1]
    tot = a.sum(1, keepdims=True)
    cum = np.cumsum(a, 1) / np.maximum(tot, 1e-12)
    return (cum < mass).sum(1) + 1


@torch.no_grad()
def local_model_summary(ex: Explainable, X: Tensor, lengths: Optional[Tensor] = None) -> Dict[str, float]:
    """Sizes of the per-example local models (what a user has to read)."""
    if ex.parts is None:
        return {}
    z = ex.full(X)
    tgt = _target(z)
    rows = torch.arange(X.shape[0], device=X.device)
    parts = ex.parts(rows)
    res: Dict[str, float] = {}
    if "linear" in parts:
        w = parts["linear"].gather(2, tgt[:, None, None].expand(-1, parts["linear"].shape[1], 1))[..., 0]
        contrib = (w * X).cpu().numpy()
        res["linear_terms_90"] = float(explanation_size(contrib).mean())
    if "clauses" in parts and ex.clause_fn is not None:
        c = ex.clause_fn(X)
        W = parts["clauses"].gather(2, tgt[:, None, None].expand(-1, parts["clauses"].shape[1], 1))[..., 0]
        contrib = (W * c).cpu().numpy()
        active = c.sum(1).cpu().numpy()
        res["active_clauses_mean"] = float(active.mean())
        sizes = explanation_size(contrib)
        sizes = np.minimum(sizes, np.maximum(active, 1))
        res["clause_terms_90"] = float(sizes.mean())
        if lengths is not None:
            L = lengths.float()
            res["active_clause_len_mean"] = float(((c * L).sum(1) / c.sum(1).clamp_min(1)).mean())
    return res


@torch.no_grad()
def local_fidelity(ex: Explainable, X: Tensor, y: np.ndarray, k_per_class=(10, 25, 50, 100, 200),
                   n_anchor: int = 500, seed: int = 0) -> Dict[str, float]:
    """Paper Table 1: accuracy of the local model of ``x_n`` on its ``k`` nearest neighbours
    from every class (Euclidean, preprocessed space, drawn from the same split)."""
    if ex.local is None:
        return {}
    g = np.random.default_rng(seed)
    N = X.shape[0]
    anchors = g.choice(N, size=min(n_anchor, N), replace=False)
    yt = torch.as_tensor(y, device=X.device)
    classes = torch.unique(yt)
    res = {}
    for k in k_per_class:
        accs = []
        for a in anchors:
            nb = []
            dist = ((X - X[a]) ** 2).sum(1)
            dist[a] = float("inf")
            for cl in classes:
                idx = (yt == cl).nonzero().flatten()
                kk = min(k, idx.numel() - int(yt[a] == cl))
                if kk <= 0:
                    continue
                nb.append(idx[dist[idx].topk(kk, largest=False).indices])
            nb = torch.cat(nb)
            z = ex.local(X[nb], torch.full((nb.numel(),), int(a), device=X.device))
            pred = (z[:, 0] > 0).long() if z.shape[1] == 1 else z.argmax(1)
            accs.append(float((pred == yt[nb]).float().mean()))
        res[f"local_acc_k{k}"] = float(np.mean(accs))
    return res
