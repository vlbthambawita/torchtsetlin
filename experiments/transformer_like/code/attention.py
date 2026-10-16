"""Score normalisation, masked scaled-dot-product attention, value mixing and quantisation
(spec §7, §10). float32 main path; the float64 loop reference is ``attention_ref.py``.

``alpha[b, i, j]`` = weight of **sender j** in **receiver i**; rows sum to 1 over allowed senders.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

CONTEXT_THRESHOLDS = (0.25, 0.50, 0.75)
DIRECT_THRESHOLDS = (-0.5, 0.0, 0.5)
MODES = ("qk", "uniform", "self_only", "shuffled_keys")
EPS = 1e-6


def normalize_votes(u: Tensor, eps: float = EPS) -> Tensor:
    """Per-token centring and scaling over the 10 class coordinates (spec §7.1)."""
    centered = u - u.mean(dim=-1, keepdim=True)
    # a constant vector maps to exactly zero (spec §7.1); without this guard the float32
    # rounding residual of the mean (~1e-8) is amplified by 1/sqrt(eps) (D16)
    const = (u.amax(dim=-1, keepdim=True) == u.amin(dim=-1, keepdim=True))
    centered = torch.where(const, torch.zeros_like(centered), centered)
    var = (centered * centered).mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps)


def compatibility(q: Tensor, k: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return q @ k.transpose(-1, -2) / (math.sqrt(q.shape[-1]) * temperature)


def masked_softmax(S: Tensor, allowed: Tensor) -> Tensor:
    """Softmax over the sender axis restricted to ``allowed``; all-masked rows are zero."""
    row_any = allowed.any(dim=-1, keepdim=True)
    Sm = S.masked_fill(~allowed, float("-inf"))
    m = Sm.amax(dim=-1, keepdim=True)
    m = torch.where(row_any, m, torch.zeros_like(m))  # empty rows handled before exp
    e = torch.exp(Sm - m) * allowed
    den = e.sum(dim=-1, keepdim=True)
    return torch.where(row_any, e / den.clamp_min(torch.finfo(S.dtype).tiny), torch.zeros_like(e))


def shuffle_keys(k: Tensor, active: Tensor) -> Tensor:
    """A3: cyclically shift K over the active slots (row-major) by one: the key at the r-th
    active slot becomes the key of the ((r+1) mod m)-th active slot. Identity if m < 2."""
    B, n, _ = k.shape
    ar = torch.arange(n, device=k.device).expand(B, n)
    # active slots first, in row-major order
    order = torch.sort(torch.where(active, ar, ar + n), dim=1).indices
    m = active.sum(dim=1, keepdim=True)
    rank = torch.cumsum(active.long(), dim=1) - 1
    nxt = torch.where(m > 0, (rank + 1) % m.clamp_min(1), rank.clamp_min(0))
    src = torch.gather(order, 1, nxt.clamp_min(0))
    src = torch.where(active, src, ar)
    return torch.gather(k, 1, src.unsqueeze(-1).expand_as(k))


def attention_weights(active: Tensor, mode: str, q: Optional[Tensor] = None, k: Optional[Tensor] = None,
                      temperature: float = 1.0) -> Tensor:
    allowed = active.unsqueeze(-1) & active.unsqueeze(-2)
    if mode == "uniform":
        a = allowed.float()
        return a / a.sum(-1, keepdim=True).clamp_min(1.0)
    if mode == "self_only":
        eye = torch.eye(active.shape[1], dtype=torch.bool, device=active.device)
        return (allowed & eye).float()
    if mode == "shuffled_keys":
        k = shuffle_keys(k, active)
    elif mode != "qk":
        raise ValueError(mode)
    return masked_softmax(compatibility(q, k, temperature), allowed)


def aggregate(alpha: Tensor, values: Tensor) -> Tensor:
    return alpha @ values


def quantize(x: Tensor, thresholds=CONTEXT_THRESHOLDS) -> Tensor:
    """``z[..., t] = 1[x >= beta_t]`` (``>=`` exactly, no epsilon)."""
    th = torch.tensor(thresholds, dtype=x.dtype, device=x.device)
    return x.unsqueeze(-1) >= th


def context_features(image_bits: Tensor, context_bits: Tensor) -> Tensor:
    """``[vec(x); vec(z)]`` in token, feature, threshold order -> ``(B, 784 + 16*55*3)``."""
    B = image_bits.shape[0]
    return torch.cat([image_bits.reshape(B, -1), context_bits.reshape(B, -1)], dim=1)


def direct_features(image_bits: Tensor, uq: Tensor, uk: Tensor, active: Tensor) -> Tensor:
    """A2: clipped unnormalised Q/K votes thresholded at [-0.5, 0, 0.5], zeroed for inactive
    tokens; ``[image; Q bits; K bits]`` -> ``(B, 784 + 16*2*10*3)``."""
    B = image_bits.shape[0]
    m = active.unsqueeze(-1).unsqueeze(-1)
    qb = quantize(uq, DIRECT_THRESHOLDS) & m
    kb = quantize(uk, DIRECT_THRESHOLDS) & m
    return torch.cat([image_bits.reshape(B, -1), qb.reshape(B, -1), kb.reshape(B, -1)], dim=1)
