"""Pair tokens: attention as a clause over (receiver, sender) pairs (THEORY.md section B).

``pair_tokens`` turns a token sequence ``(N, L)`` into ``(N, F_pair, P)`` Boolean columns, one
per ordered pair ``(i, j)``, so that ``Conv1dTsetlinMachine(kernel_size=1,
position_encoding=False)`` evaluates every clause on every pair and ORs over pairs. The
library is used unchanged (rule C1). Column ``(i, j)`` holds, in order:

- ``onehot(x_i)``                              receiver ("query" literals), V bits
- ``onehot(x_{j+k})`` for ``k = 0..window-1``  sender ("key/value" literals), V+1 bits each
  (the extra bit marks "past the end")
- the position block, by ``rel``:
    ``"therm"``  thermometer of ``j - i``: 2L-2 bits ``[j - i > t]`` (default)
    ``"bucket"`` thermometer of a monotone log-bucket of ``j - i`` (THEORY A.4)
    ``"abs"``    thermometer of ``i`` then of ``j``: 2(L-1) bits (absolute codes, Theorem A7)
    ``"none"``   no position block
- ``therm(i)`` if ``abs_receiver``             receiver absolute position, L-1 bits
- ``[i == L-1]`` if ``last_bit``               receiver is the last token (query slot)

``rel_perm`` (``(N, P)`` long) permutes the position block across pairs per example (the A2-shuf
ablation). Pair columns are cheap to gather, so build them per mini-batch (DECISIONS D8).
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

REL_MODES = ("therm", "bucket", "abs", "none")


def thermometer(v: Tensor, lo: int, hi: int) -> Tensor:
    """``[v > t]`` for ``t = lo .. hi-1``: ``hi - lo`` bits encoding ``v`` in ``[lo, hi]``."""
    t = torch.arange(lo, hi, device=v.device)
    return v.unsqueeze(-1) > t


def offset_bucket(delta: Tensor, exact: int = 4) -> Tensor:
    """Monotone signed log-bucket of an offset: exact for ``|d| <= exact``, then one bucket per
    doubling (``5..8``, ``9..16``, ... for ``exact = 4``)."""
    a = delta.abs()
    big = exact + torch.ceil(torch.log2(a.clamp(min=1).float() / exact)).long()
    b = torch.where(a <= exact, a, big)
    return torch.sign(delta) * b


def n_buckets(L: int, exact: int = 4) -> int:
    """Largest bucket index for offsets up to ``L - 1``."""
    return int(offset_bucket(torch.tensor([L - 1]), exact).item())


def pair_feature_names(L: int, V: int, window: int = 1, abs_receiver: bool = False,
                       last_bit: bool = False, rel: str = "therm") -> List[str]:
    names = [f"q={a}" for a in range(V)]
    for k in range(window):
        names += [f"k{k}={a}" for a in range(V)] + [f"k{k}=END"]
    if rel == "therm":
        names += [f"rel>{t}" for t in range(-(L - 1), L - 1)]
    elif rel == "bucket":
        bm = n_buckets(L)
        names += [f"relb>{t}" for t in range(-bm, bm)]
    elif rel == "abs":
        names += [f"i>{t}" for t in range(L - 1)] + [f"j>{t}" for t in range(L - 1)]
    elif rel != "none":
        raise ValueError(rel)
    if abs_receiver:
        names += [f"i>{t}" for t in range(L - 1)]
    if last_bit:
        names += ["i=last"]
    return names


def pair_index(L: int, self_only: bool = False, device=None):
    """Receiver and sender index of every pair column, ``(P,)`` each (row-major over ``i``)."""
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    if self_only:
        keep = ii == jj
        ii, jj = ii[keep], jj[keep]
    return ii, jj


def pair_tokens(x: Tensor, V: int, window: int = 1, abs_receiver: bool = False,
                last_bit: bool = False, self_only: bool = False, rel: str = "therm",
                rel_perm: Optional[Tensor] = None) -> Tensor:
    """``(N, L)`` int tokens -> ``(N, F_pair, P)`` bool; ``P = L*L`` (``L`` with ``self_only``)."""
    N, L = x.shape
    dev = x.device
    ii, jj = pair_index(L, self_only, dev)
    P = ii.numel()
    eye = torch.eye(V, dtype=torch.bool, device=dev)
    parts = [eye[x[:, ii]]]  # (N, P, V)
    eye_end = torch.eye(V + 1, dtype=torch.bool, device=dev)
    for k in range(window):
        jk = jj + k
        tok = torch.where(jk < L, x[:, jk.clamp(max=L - 1)], torch.full_like(x[:, :1], V))
        parts.append(eye_end[tok])  # (N, P, V+1)
    if rel == "therm":
        pos = thermometer(jj - ii, -(L - 1), L - 1)  # (P, 2L-2)
    elif rel == "bucket":
        bm = n_buckets(L)
        pos = thermometer(offset_bucket(jj - ii), -bm, bm)
    elif rel == "abs":
        pos = torch.cat([thermometer(ii, 0, L - 1), thermometer(jj, 0, L - 1)], dim=1)
    elif rel == "none":
        pos = None
    else:
        raise ValueError(rel)
    if pos is not None:
        pos = pos.unsqueeze(0).expand(N, P, -1)
        if rel_perm is not None:
            pos = torch.gather(pos, 1, rel_perm.to(dev).long().unsqueeze(-1).expand(N, P, pos.shape[-1]))
        parts.append(pos)
    if abs_receiver:
        parts.append(thermometer(ii, 0, L - 1).unsqueeze(0).expand(N, P, -1))
    if last_bit:
        parts.append((ii == L - 1).view(1, P, 1).expand(N, P, 1))
    return torch.cat(parts, dim=2).transpose(1, 2).contiguous()  # (N, F, P)


def token_columns(x: Tensor, V: int, abs_pos: str = "none") -> Tensor:
    """Per-token columns for the conv baselines: ``(N, V + pos_bits, L)`` bool.

    ``abs_pos``: ``none``, ``onehot``, ``thermometer``, ``binary`` or ``gray``.
    """
    N, L = x.shape
    dev = x.device
    eye = torch.eye(V, dtype=torch.bool, device=dev)
    parts = [eye[x]]
    p = torch.arange(L, device=dev)
    if abs_pos == "onehot":
        parts.append(torch.eye(L, dtype=torch.bool, device=dev).unsqueeze(0).expand(N, L, L))
    elif abs_pos == "thermometer":
        parts.append(thermometer(p, 0, L - 1).unsqueeze(0).expand(N, L, -1))
    elif abs_pos in ("binary", "gray"):
        m = max(1, (L - 1).bit_length())
        code = p ^ (p >> 1) if abs_pos == "gray" else p
        bits = ((code.unsqueeze(1) >> torch.arange(m, device=dev)) & 1).bool()
        parts.append(bits.unsqueeze(0).expand(N, L, m))
    elif abs_pos != "none":
        raise ValueError(abs_pos)
    return torch.cat(parts, dim=2).transpose(1, 2).contiguous()
