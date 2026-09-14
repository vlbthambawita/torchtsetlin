"""Regression Tsetlin machine."""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..utils import as_float_tensor
from .base import FeedbackAccumulator, TsetlinMachineBase

__all__ = ["RegressionTsetlinMachine"]


class RegressionTsetlinMachine(TsetlinMachineBase):
    """Regression Tsetlin machine (Abeyrathna et al.).

    All clauses have positive polarity; the vote sum ``v = sum_j w_j c_j`` is clamped to
    ``[0, T]`` and mapped linearly onto the target range:
    ``y_hat = y_min + (v / T) * (y_max - y_min)``.

    Learning: with the target expressed in vote units ``v* = T (y - y_min)/(y_max - y_min)``,
    every clause receives Type I feedback with probability ``(v* - v)/T`` when the prediction
    is too low, and Type II feedback with probability ``(v - v*)/T`` when it is too high.

    Args:
        n_features: Boolean feature count (``None`` = lazy).
        n_clauses: number of clauses.
        T: vote margin — also the output resolution (``T`` distinct predictions).
        s: specificity.
        y_range: ``(y_min, y_max)`` of the targets. If ``None`` it is inferred from the first
            training batch (a warning is issued); prefer to pass it explicitly.
        weighted: learn integer clause weights (Type Ia ``+1``, Type II ``-1``, min 0).
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachineBase`.
    """

    weights: Tensor

    def __init__(
        self,
        n_features: Optional[int],
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        y_range: Optional[Tuple[float, float]] = None,
        weighted: bool = False,
        **kwargs,
    ) -> None:
        if T <= 0:
            raise ValueError("T must be positive")
        self.n_clauses = int(n_clauses)
        self.T = float(T)
        self.weighted = bool(weighted)
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None
        if y_range is not None:
            self.set_range(*y_range)
        super().__init__(n_features, self.n_clauses, s=s, **kwargs)
        self.register_buffer("weights", torch.ones(self.n_clauses, dtype=torch.int32))

    n_outputs = 1

    def set_range(self, y_min: float, y_max: float) -> None:
        if y_max <= y_min:
            raise ValueError("y_max must be larger than y_min")
        self.y_min, self.y_max = float(y_min), float(y_max)

    def _check_range(self) -> None:
        if self.y_min is None or self.y_max is None:
            raise RuntimeError("Target range unknown: pass y_range or call update() first.")

    # ------------------------------------------------------------------ output layer
    def _votes(self, clause_out: Tensor, clamp: bool = True) -> Tensor:
        w = self.weights.to(torch.float32) if self.weighted else None
        c = clause_out.to(torch.float32)
        v = (c * w).sum(dim=1, keepdim=True) if w is not None else c.sum(dim=1, keepdim=True)
        if clamp:
            v = v.clamp_(0.0, self.T)
        return v  # (B, 1)

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:
        y = as_float_tensor(y, device=device).view(-1)
        if y.shape[0] != n:
            raise ValueError(f"x has {n} examples but y has {y.shape[0]} targets")
        if self.y_min is None or self.y_max is None:
            lo, hi = float(y.min()), float(y.max())
            if hi <= lo:
                hi = lo + 1.0
            warnings.warn(
                f"Inferring target range ({lo}, {hi}) from the first batch; pass y_range= for "
                "a stable scale.",
                stacklevel=3,
            )
            self.set_range(lo, hi)
        return y

    def votes_to_targets(self, votes: Tensor) -> Tensor:
        self._check_range()
        return self.y_min + (votes.squeeze(-1) / self.T) * (self.y_max - self.y_min)  # type: ignore[operator]

    def targets_to_votes(self, y: Tensor) -> Tensor:
        self._check_range()
        return ((y - self.y_min) / (self.y_max - self.y_min) * self.T).clamp(0.0, self.T)  # type: ignore[operator]

    def predict(self, x) -> Tensor:
        """Continuous predictions ``(B,)`` in ``[y_min, y_max]``."""
        return self.votes_to_targets(self.forward(x))

    # ------------------------------------------------------------------ feedback policy
    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:
        B = votes.shape[0]
        v = votes.squeeze(1)
        v_star = self.targets_to_votes(y)
        err = (v_star - v) / self.T  # >0: too low -> Type I ; <0: too high -> Type II
        p = err.abs().clamp(max=1.0)
        sel = torch.rand(B, self.n_clauses, device=votes.device) < p.unsqueeze(1)
        low = (err > 0).unsqueeze(1)
        return sel & low, sel & ~low, None

    def _accumulate_weights(
        self, acc: FeedbackAccumulator, aux: Any, clause_out: Tensor, sel_fire_i: Tensor, sel_fire_ii: Tensor
    ) -> None:
        if not self.weighted:
            return
        dw = sel_fire_i.sum(dim=0) - sel_fire_ii.sum(dim=0)
        acc.extra["dw"] = acc.extra["dw"] + dw if "dw" in acc.extra else dw

    def _commit_weights(self, acc: FeedbackAccumulator) -> None:
        if self.weighted and "dw" in acc.extra:
            self.weights.add_(acc.extra["dw"].to(self.weights.dtype)).clamp_(min=0)

    def rules(self, feature_names: Optional[Sequence[str]] = None, skip_empty: bool = True) -> List[str]:
        """``IF ... THEN +w`` rules: each matching clause adds ``w * (y_max - y_min) / T``."""
        self._check_range()
        step = (self.y_max - self.y_min) / self.T  # type: ignore[operator]
        out = []
        for j in range(self.n_clauses_total):
            if skip_empty and int(self.include_count[j]) == 0:
                continue
            w = int(self.weights[j]) if self.weighted else 1
            out.append(f"IF {self.clause_expression(j, feature_names)} THEN +{w * step:.4g}")
        return out

    def extra_repr(self) -> str:
        rng = f"({self.y_min}, {self.y_max})" if self.y_min is not None else "lazy"
        return f"n_clauses={self.n_clauses}, T={self.T}, y_range={rng}, " + super().extra_repr()
