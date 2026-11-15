"""Synthetic sequence tasks with controllable length (torchtsetlin_plans/experiments/transformer_concepts/PLAN.md section 2).

Every generator returns ``(tokens, labels)``: ``tokens`` a ``(N, L)`` int64 tensor over a
vocabulary of ``V`` symbols, ``labels`` a ``(N,)`` int64 tensor. All randomness comes from the
``torch.Generator`` passed in.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _fillers(N: int, L: int, lo: int, V: int, g: torch.Generator) -> Tensor:
    return torch.randint(lo, V, (N, L), generator=g)


def ordered_pair(N: int, L: int, V: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T1: exactly one A (=0) and one B (=1) at distinct random positions; label [pos A < pos B]."""
    x = _fillers(N, L, 2, V, g)
    perm = torch.rand(N, L, generator=g).argsort(dim=1)
    pa, pb = perm[:, 0], perm[:, 1]
    rows = torch.arange(N)
    x[rows, pa] = 0
    x[rows, pb] = 1
    return x, (pa < pb).long()


def associative_recall(N: int, L: int, n_keys: int, n_values: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T2: ``k1 v1 k2 v2 ... kn vn q`` with distinct keys, ``q`` one of them; label = its value.

    Symbols: keys ``0..n_keys-1``, values ``n_keys..n_keys+n_values-1``. ``L`` must be odd.
    """
    if L % 2 == 0 or (L - 1) // 2 > n_keys:
        raise ValueError("L must be odd and (L-1)/2 <= n_keys")
    n = (L - 1) // 2
    keys = torch.rand(N, n_keys, generator=g).argsort(dim=1)[:, :n]
    vals = torch.randint(0, n_values, (N, n), generator=g)
    which = torch.randint(0, n, (N,), generator=g)
    x = torch.empty(N, L, dtype=torch.long)
    x[:, 0:2 * n:2] = keys
    x[:, 1:2 * n:2] = vals + n_keys
    rows = torch.arange(N)
    x[:, -1] = keys[rows, which]
    return x, vals[rows, which]


def needle(N: int, L: int, V: int, d: int, n_classes: int, g: torch.Generator,
           distractors: bool = True) -> Tuple[Tensor, Tensor]:
    """T4: one marker M (=0) at a random position p <= L-1-d; label = symbol at p+d.

    Label symbols are ``1..n_classes``. With ``distractors`` every non-marker position is drawn
    from ``1..V-1``, so label symbols also appear elsewhere; without, fillers are drawn from
    ``n_classes+1..V-1`` and a label symbol occurs only at ``p+d``.
    """
    lo = 1 if distractors else n_classes + 1  # without distractors label symbols occur only at p+d
    if lo >= V:
        raise ValueError("V too small for fillers")
    x = torch.randint(lo, V, (N, L), generator=g)
    p = torch.randint(0, L - d, (N,), generator=g)
    y = torch.randint(0, n_classes, (N,), generator=g)
    rows = torch.arange(N)
    x[rows, p] = 0
    x[rows, p + d] = y + 1
    return x, y


def repeat_detect(N: int, L: int, V: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T3: label 1 iff some symbol occurs twice. Half the examples have exactly one repeat."""
    if V < L:
        raise ValueError("need V >= L for repeat-free sequences")
    x = torch.rand(N, V, generator=g).argsort(dim=1)[:, :L].clone()
    y = (torch.rand(N, generator=g) < 0.5).long()
    perm = torch.rand(N, L, generator=g).argsort(dim=1)
    rows = torch.arange(N)
    src, dst = perm[:, 0], perm[:, 1]
    rep = y.bool()
    x[rows[rep], dst[rep]] = x[rows[rep], src[rep]]
    return x, y


def bag_control(N: int, L: int, V: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T5: label 1 iff both symbols 0 and 1 occur (order irrelevant); ~balanced by construction."""
    x = _fillers(N, L, 2, V, g)
    y = (torch.rand(N, generator=g) < 0.5).long()
    perm = torch.rand(N, L, generator=g).argsort(dim=1)
    rows = torch.arange(N)
    x[rows, perm[:, 0]] = 0
    # negatives: one of the two symbols is missing; positives: both present
    second = torch.where(y.bool(), torch.ones_like(y), torch.zeros_like(y))
    x[rows, perm[:, 1]] = second
    flip = (~y.bool()) & (torch.rand(N, generator=g) < 0.5)  # some negatives contain only 1
    x[rows[flip], perm[flip, 0]] = 1
    x[rows[flip], perm[flip, 1]] = 1
    return x, y


def absolute_position(N: int, L: int, V: int, pos: int, n_classes: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T6: label = symbol at a fixed position ``pos`` (absolute position needed, relative not)."""
    x = torch.randint(0, V, (N, L), generator=g)
    y = torch.randint(0, n_classes, (N,), generator=g)
    x[:, pos] = y
    return x, y


def presence(N: int, L: int, V: int, g: torch.Generator) -> Tuple[Tensor, Tensor]:
    """T7 (no-routing control): label 1 iff symbol 0 occurs (exactly once in positives)."""
    x = _fillers(N, L, 1, V, g)
    y = (torch.rand(N, generator=g) < 0.5).long()
    pos = torch.randint(0, L, (N,), generator=g)
    rows = torch.arange(N)[y.bool()]
    x[rows, pos[y.bool()]] = 0
    return x, y
