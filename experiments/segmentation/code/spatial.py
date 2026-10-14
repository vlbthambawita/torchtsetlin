"""Structured inference on frozen unaries (PLAN H4, H5) and the cheap controls.

All functions take score maps ``scores (B, K, H, W)`` (TM vote sums or CNN logits), an image
``img (B, C, H, W)`` in [0, 1] and a ``valid (B, H, W)`` mask; padding/void pixels have no
edges. Four-neighbour grid, each undirected edge counted once.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F


def edge_weights(img: torch.Tensor, sigma: Optional[float], valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Horizontal ``(B, H, W-1)`` and vertical ``(B, H-1, W)`` weights
    ``exp(-||I_i - I_j||^2 / (2 sigma^2))``; ``sigma=None`` -> unweighted (w = 1)."""
    dh = ((img[..., :, 1:] - img[..., :, :-1]) ** 2).sum(1)
    dv = ((img[..., 1:, :] - img[..., :-1, :]) ** 2).sum(1)
    if sigma is None:
        wh, wv = torch.ones_like(dh), torch.ones_like(dv)
    else:
        wh, wv = torch.exp(-dh / (2 * sigma**2)), torch.exp(-dv / (2 * sigma**2))
    vf = valid.float()
    wh = wh * vf[..., :, 1:] * vf[..., :, :-1]
    wv = wv * vf[..., 1:, :] * vf[..., :-1, :]
    return wh, wv


def _neighbor_cost(lbl: torch.Tensor, wh: torch.Tensor, wv: torch.Tensor, K: int) -> torch.Tensor:
    """``(B, K, H, W)``: sum_j w_ij [k != y_j] for every pixel and candidate label k."""
    oh = F.one_hot(lbl, K).permute(0, 3, 1, 2).float()  # (B, K, H, W)
    agree = torch.zeros_like(oh)
    agree[..., :, 1:] += wh.unsqueeze(1) * oh[..., :, :-1]  # left neighbour
    agree[..., :, :-1] += wh.unsqueeze(1) * oh[..., :, 1:]  # right
    agree[..., 1:, :] += wv.unsqueeze(1) * oh[..., :-1, :]  # up
    agree[..., :-1, :] += wv.unsqueeze(1) * oh[..., 1:, :]  # down
    deg = torch.zeros_like(lbl, dtype=torch.float32)
    deg[..., :, 1:] += wh
    deg[..., :, :-1] += wh
    deg[..., 1:, :] += wv
    deg[..., :-1, :] += wv
    return deg.unsqueeze(1) - agree


def potts_energy(U: torch.Tensor, lbl: torch.Tensor, wh, wv, lam: float) -> torch.Tensor:
    """Per-image energy ``sum_i U_i(y_i) + lam sum_edges w_ij [y_i != y_j]`` (valid pixels only
    contribute edges; unary of every pixel included — void unaries are zeroed by the caller)."""
    un = U.gather(1, lbl.unsqueeze(1)).squeeze(1).sum((1, 2))
    ph = (wh * (lbl[..., :, 1:] != lbl[..., :, :-1]).float()).sum((1, 2))
    pv = (wv * (lbl[..., 1:, :] != lbl[..., :-1, :]).float()).sum((1, 2))
    return un + lam * (ph + pv)


def potts_icm(
    scores: torch.Tensor, img: torch.Tensor, valid: torch.Tensor, T: float, lam: float, sigma: Optional[float],
    max_sweeps: int = 10, log: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """Red/black coordinate descent on the H4 Potts energy, started at the unary argmin.

    Each colour class has no adjacent vertices, so the exact conditional minimisation of all
    its pixels at once cannot increase the energy. The current label is kept on ties."""
    K = scores.shape[1]
    U = -scores.float() / T
    U = U * valid.unsqueeze(1).float()
    lbl = U.argmin(1)  # == scores.argmax(1) with the same first-index tie rule
    wh, wv = edge_weights(img.float(), sigma, valid)
    H, W = lbl.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=lbl.device), torch.arange(W, device=lbl.device), indexing="ij")
    colours = [((yy + xx) % 2 == c) for c in (0, 1)]
    hist = {"energy": [float(potts_energy(U, lbl, wh, wv, lam).sum())], "changed": []}
    init = lbl.clone()
    for _ in range(max_sweeps):
        n_changed = 0
        for col in colours:
            cost = U + lam * _neighbor_cost(lbl, wh, wv, K)
            cur = cost.gather(1, lbl.unsqueeze(1)).squeeze(1)
            best_cost, best = cost.min(1)
            upd = col & valid & (best_cost < cur - 1e-9)
            n_changed += int(upd.sum())
            lbl = torch.where(upd, best, lbl)
            if log:
                hist["energy"].append(float(potts_energy(U, lbl, wh, wv, lam).sum()))
            hist["changed"].append(int(upd.sum()))
        if n_changed == 0:
            break
    hist["changed_pixel_rate"] = float(((lbl != init) & valid).float().sum() / valid.float().sum().clamp_min(1))
    return lbl, hist


def laplacian_apply(Q: torch.Tensor, wh: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
    """``L Q`` for ``L = D - W`` on the grid; ``Q (B, K, H, W)``."""
    out = torch.zeros_like(Q)
    whk, wvk = wh.unsqueeze(1), wv.unsqueeze(1)
    dh = Q[..., :, 1:] - Q[..., :, :-1]
    dv = Q[..., 1:, :] - Q[..., :-1, :]
    out[..., :, :-1] -= whk * dh
    out[..., :, 1:] += whk * dh
    out[..., :-1, :] -= wvk * dv
    out[..., 1:, :] += wvk * dv
    return out


def degree(wh, wv):
    d = torch.zeros(wh.shape[0], wv.shape[1] + 1, wh.shape[2] + 1, device=wh.device)
    d[..., :, 1:] += wh
    d[..., :, :-1] += wh
    d[..., 1:, :] += wv
    d[..., :-1, :] += wv
    return d


def diffuse(
    scores: torch.Tensor, img: torch.Tensor, valid: torch.Tensor, T: float, lam: float, sigma: Optional[float],
    temperature: float = 1.0, method: str = "explicit", max_iter: int = 100, tol: float = 1e-4,
) -> Tuple[torch.Tensor, Dict]:
    """Anchored diffusion (H5): solve ``(I + lam L) Q = P`` with ``P = softmax(v / T / temp)``.

    ``method="explicit"``: ``Q <- Q - eta[(Q - P) + lam L Q]`` from ``Q = P`` with
    ``eta = 1 / (1 + 2 lam d_max)`` (keeps Q a non-negative mixture); ``"cg"``: conjugate
    gradients on the same SPD system. Stops at relative residual < ``tol`` or ``max_iter``."""
    P = torch.softmax(scores.float() / T / temperature, dim=1)
    wh, wv = edge_weights(img.float(), sigma, valid)
    if lam == 0:
        return P, {"iters": 0, "converged": True, "rel_residual": 0.0}

    def A(Q):
        return Q + lam * laplacian_apply(Q, wh, wv)

    pn = P.norm()
    info = {"converged": False}
    if method == "explicit":
        dmax = float(degree(wh, wv).max())
        eta = 1.0 if dmax == 0 else 1.0 / (1.0 + 2 * lam * dmax)
        Q = P.clone()
        for it in range(1, max_iter + 1):
            Q = Q - eta * ((Q - P) + lam * laplacian_apply(Q, wh, wv))
            r = float((A(Q) - P).norm() / pn)
            if r < tol:
                info["converged"] = True
                break
        info.update(iters=it, rel_residual=r, eta=eta)
        return Q, info
    # conjugate gradients
    Q = P.clone()
    R = P - A(Q)
    D = R.clone()
    rs = (R * R).sum()
    for it in range(1, max_iter + 1):
        AD = A(D)
        alpha = rs / (D * AD).sum().clamp_min(1e-30)
        Q = Q + alpha * D
        R = R - alpha * AD
        rs_new = (R * R).sum()
        r = float(rs_new.sqrt() / pn)
        if r < tol:
            info["converged"] = True
            break
        D = R + (rs_new / rs) * D
        rs = rs_new
    info.update(iters=it, rel_residual=r)
    return Q, info


def majority_filter(scores: torch.Tensor, valid: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Control: ``k x k`` mode filter of the raw argmax (ties -> lowest class id among maxima,
    keeping the centre label when it is among them)."""
    K = scores.shape[1]
    lbl = scores.argmax(1)
    oh = F.one_hot(lbl, K).permute(0, 3, 1, 2).float() * valid.unsqueeze(1).float()
    cnt = F.avg_pool2d(oh, k, stride=1, padding=k // 2) * k * k
    best = cnt.max(1, keepdim=True).values
    centre_is_max = cnt.gather(1, lbl.unsqueeze(1)) >= best - 1e-6
    return torch.where(centre_is_max.squeeze(1), lbl, cnt.argmax(1))
