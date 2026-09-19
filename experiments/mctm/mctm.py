"""Multilayer Convolutional Tsetlin Machine (MCTM).

A CTM layer normally OR-pools each clause over all patch positions and votes. Here the
*spatial* clause-activation map is kept instead and used as the Boolean input channels of
the next CTM layer, following the greedy layer-wise scheme of
``ideas/multilayer_convolutional_tsetlin_machine_pseudocode.md``.

The library computes the full ``(B, P, C)`` patch-match tensor inside
``_ConvMixin._evaluate`` and discards the layout; :func:`conv_clause_maps` reshapes it to
``(B, C, Py, Px)``. It uses the model's private encode/evaluate hooks on purpose -- keeping
the prototype out of the public API surface.

Design decisions that deviate from the pseudocode, and why, are documented per function.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as TF

from torchtsetlin.utils import chunk_indices


# --------------------------------------------------------------------------- feature maps
@torch.no_grad()
def conv_clause_maps(model, x, empty_value: bool = False) -> Tensor:
    """Spatial clause activations ``(B, C, Py, Px)`` (bool) for a convolutional model.

    ``empty_value=False`` is forced by default: a clause with no included literals must not
    emit an all-ones channel into the next layer (every clause is empty at init, and a
    constant-1 channel is exactly what a downstream conjunction latches onto).
    """
    xb = model._prepare(x)
    Py, Px = model._grid(xb.shape[2], xb.shape[3])
    outs: List[Tensor] = []
    for sl in chunk_indices(xb.shape[0], model._chunk_size(xb)):
        lits = model._encode(xb[sl])
        _, matches = model._evaluate(lits, empty_value=empty_value)  # (b, P, C)
        b = matches.shape[0]
        outs.append(matches.to(torch.bool).permute(0, 2, 1).reshape(b, -1, Py, Px))
    return torch.cat(outs, dim=0)


def or_pool(maps: Tensor, k: int) -> Tensor:
    """``k x k`` stride-``k`` OR pooling of a Boolean map stack -- max-pool on 0/1 bits.

    Its role here is *density control*, not just downsampling: clause maps are sparse, and a
    conjunction over sparse bits is satisfied with probability ~p^j. A 2x2 OR lifts a 2%
    firing rate to ~8%, which is what keeps layer-2 clauses from degenerating into
    conjunctions of negated (i.e. near-constant) literals.
    """
    if k <= 1:
        return maps
    return TF.max_pool2d(maps.to(torch.float32), kernel_size=k, stride=k) > 0.5


# --------------------------------------------------------------------------- random clauses
@torch.no_grad()
def randomize_clauses(model, n_include: int = 3, seed: Optional[int] = None) -> None:
    """Replace the learned state with random clauses: ``n_include`` literals per clause,
    each a random feature with a random sign (never both ``x`` and ``NOT x``, which would
    make the clause unsatisfiable).

    This is the control that answers "does the supervised local objective buy anything?" --
    if a random layer 1 matches a trained layer 1, the greedy objective is contributing
    nothing beyond a random Boolean feature basis.
    """
    if seed is not None:
        torch.manual_seed(seed)
    C, L = model.ta_state.shape
    Fn = L // 2
    dev = model.ta_state.device
    model.ta_state.fill_(model.n_states - 1)  # all excluded, one step below the boundary
    feats = torch.stack([torch.randperm(Fn, device=dev)[:n_include] for _ in range(C)])
    sign = torch.randint(0, 2, (C, n_include), device=dev)
    cols = feats + sign * Fn
    rows = torch.arange(C, device=dev).unsqueeze(1).expand_as(cols)
    model.ta_state[rows, cols] = model.n_states  # included
    model._refresh_include()


# --------------------------------------------------------------------------- the stack
class MCTM:
    """A frozen stack of CTM feature layers followed by a CTM classifier.

    ``layers`` are trained greedily: each is fit on the representation produced by the ones
    below it, then frozen. Because a frozen layer's output is constant across epochs the
    driver materialises it once (50k CIFAR images at 128x14x14 bool is 1.25 GB, which fits
    on the GPU); :meth:`transform` is the streaming path used at evaluation time.
    """

    def __init__(self, layers: Sequence, pools: Sequence[int], head) -> None:
        self.layers = list(layers)
        self.pools = list(pools)
        self.head = head

    def transform(self, x: Tensor, upto: Optional[int] = None, batch_size: int = 256) -> Tensor:
        """Push ``x`` through the first ``upto`` frozen feature layers."""
        upto = len(self.layers) if upto is None else upto
        for li in range(upto):
            model, pool = self.layers[li], self.pools[li]
            was_training = model.training
            model.eval()
            outs = []
            for sl in chunk_indices(x.shape[0], batch_size):
                outs.append(or_pool(conv_clause_maps(model, x[sl]), pool))
            x = torch.cat(outs, dim=0)
            model.train(was_training)
        return x

    def forward(self, x: Tensor, batch_size: int = 256) -> Tensor:
        return self.head(self.transform(x, batch_size=batch_size))

    def n_clauses_total(self) -> int:
        return sum(int(m.n_clauses_total) for m in self.layers) + int(self.head.n_clauses_total)

    def n_automata(self) -> int:
        return sum(int(m.ta_state.numel()) for m in [*self.layers, self.head])


# --------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def firing_stats(maps: Tensor) -> dict:
    """Per-channel firing rate of a Boolean map stack ``(N, C, H, W)``."""
    rate = maps.float().mean(dim=(0, 2, 3))  # (C,)
    q = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], device=rate.device)
    quant = torch.quantile(rate, q).tolist()
    return {
        "mean": float(rate.mean()),
        "median": float(rate.median()),
        "quantiles": {f"q{int(v*100)}": s for v, s in zip(q.tolist(), quant)},
        "frac_dead": float((rate < 1e-4).float().mean()),
        "frac_saturated": float((rate > 0.99).float().mean()),
        "per_channel": rate.tolist(),
    }


def effective_receptive_field(patches: Sequence[int], strides: Sequence[int],
                              pools: Sequence[int]) -> Tuple[int, int]:
    """Receptive field (in input pixels) and jump of a patch/pool stack."""
    rf, jump = 1, 1
    for p, s, k in zip(patches, strides, pools):
        rf += (p - 1) * jump
        jump *= s
        if k > 1:
            rf += (k - 1) * jump
            jump *= k
    return rf, jump


@torch.no_grad()
def clause_composition(model) -> dict:
    """How a layer's clauses are built: size, and how much of that size is *negations*.

    This is the direct test of the density-collapse failure mode. Over a sparse input a
    positive literal ("channel k fires here") is rarely satisfied, while its negation
    ("channel k does not fire here") is almost always satisfied, and Type I feedback rewards
    including literals that are 1 in the drawn patch. A layer that has degenerated reports a
    negation fraction near 1.0: its clauses assert only absences and carry little
    information.
    """
    inc = model.included_mask()                      # (C, 2F) bool
    Fn = int(model.n_features)
    pos = inc[:, :Fn].sum(dim=1).float()
    neg = inc[:, Fn:].sum(dim=1).float()
    total = pos + neg
    nonempty = total > 0
    frac_neg = torch.where(nonempty, neg / total.clamp(min=1), torch.zeros_like(total))
    return {
        "size_median": float(total.median()),
        "size_mean": float(total.mean()),
        "size_max": float(total.max()),
        "frac_empty": float((~nonempty).float().mean()),
        "pos_median": float(pos.median()),
        "neg_median": float(neg.median()),
        "neg_fraction_mean": float(frac_neg[nonempty].mean()) if bool(nonempty.any()) else 0.0,
        "neg_fraction_median": float(frac_neg[nonempty].median()) if bool(nonempty.any()) else 0.0,
    }
