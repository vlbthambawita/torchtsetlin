"""Coalesced Tsetlin machine: one shared clause pool with signed weights per output."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .. import functional as F
from ..utils import as_bool_tensor, as_long_tensor
from .base import FeedbackAccumulator, TsetlinMachineBase

__all__ = ["CoalescedTsetlinMachine"]


class CoalescedTsetlinMachine(TsetlinMachineBase):
    """Coalesced (multi-output) Tsetlin machine with clause sharing (Glimsdal & Granmo, 2021).

    All ``n_clauses`` clauses are shared by the ``n_outputs`` outputs. Clause ``j`` votes for
    output ``i`` with an integer weight ``W[j, i]``: positive weights vote *for*, negative
    weights *against*. Vote sums are ``v = c @ W`` clamped to ``[-T, T]``.

    Learning per output ``i`` with binary target ``y_i``: with probability
    ``d_i = |q_i - v_i| / (2T)`` (``q_i = T`` if ``y_i = 1`` else ``-T``) a clause receives
    Type I feedback when its weight sign agrees with ``y_i`` and Type II feedback otherwise.
    Weights of matching, selected clauses move by ``+1`` for ``y_i = 1`` and ``-1`` for
    ``y_i = 0``, so a clause can change sides over time.

    In multi-class mode (``multi_label=False``) the target class and one random negative
    class are updated per example; in multi-label mode every output is updated with its own
    probability (negative outputs scaled by ``negative_scale``).

    Args:
        n_features: Boolean feature count (``None`` = lazy).
        n_outputs: number of classes (multi-class) or labels (multi-label).
        n_clauses: total number of shared clauses.
        T: vote margin.
        s: specificity.
        multi_label: interpret ``y`` as a ``(B, n_outputs)`` binary matrix and predict
            ``v_i > 0`` per output.
        negative_scale: multiplier of the Type II probability for negative outputs in
            multi-label mode (the ``q`` hyper-parameter of the multi-task CTM literature).
        focused_negative_sampling: multi-class only; bias the negative class towards strongly
            voting classes.
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachineBase`.
    """

    weights: Tensor

    def __init__(
        self,
        n_features: Optional[int],
        n_outputs: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        multi_label: bool = False,
        negative_scale: float = 1.0,
        focused_negative_sampling: bool = False,
        **kwargs,
    ) -> None:
        if n_outputs < 1 or (not multi_label and n_outputs < 2):
            raise ValueError("n_outputs must be >= 2 for multi-class, >= 1 for multi-label")
        if T <= 0:
            raise ValueError("T must be positive")
        self.n_outputs = int(n_outputs)
        self.n_clauses = int(n_clauses)
        self.T = float(T)
        self.multi_label = bool(multi_label)
        self.negative_scale = float(negative_scale)
        self.focused_negative_sampling = bool(focused_negative_sampling)
        super().__init__(n_features, self.n_clauses, s=s, **kwargs)
        w = torch.randint(0, 2, (self.n_clauses, self.n_outputs), dtype=torch.int32) * 2 - 1
        self.register_buffer("weights", w)
        self.class_names: Optional[List[str]] = None

    @property
    def n_classes(self) -> int:
        return self.n_outputs

    # ------------------------------------------------------------------ output layer
    def _votes(self, clause_out: Tensor, clamp: bool = True) -> Tensor:
        return F.vote_sums(clause_out, self.weights, self.T if clamp else None)

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:
        if self.multi_label:
            yb = as_bool_tensor(y, device=device)
            if yb.dim() == 1:
                yb = yb.view(n, -1)
            if yb.shape != (n, self.n_outputs):
                raise ValueError(f"multi-label targets must have shape ({n}, {self.n_outputs})")
            return yb
        y = as_long_tensor(y, device=device).view(-1)
        if y.shape[0] != n:
            raise ValueError(f"x has {n} examples but y has {y.shape[0]} labels")
        if bool((y < 0).any()) or bool((y >= self.n_outputs).any()):
            raise ValueError(f"labels must lie in [0, {self.n_outputs - 1}]")
        return y

    def predict(self, x) -> Tensor:
        """Class indices ``(B,)`` (multi-class) or a Boolean ``(B, n_outputs)`` matrix."""
        votes = self.forward(x)
        if self.multi_label:
            return votes > 0
        return votes.argmax(dim=1)

    def predict_proba(self, x, method: str = "linear", temperature: float = 1.0) -> Tensor:
        """Probabilities: per-output ``(1 + v/T)/2`` (multi-label) or normalised over classes."""
        votes = self.forward(x)
        p = 0.5 * (1.0 + votes / self.T)
        if self.multi_label:
            return p
        if method == "softmax":
            return F.predict_proba_from_votes(votes, self.T, temperature)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def confidence(self, x) -> Tensor:
        return F.confidence_from_votes(self.forward(x), self.T)

    # ------------------------------------------------------------------ feedback policy
    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:
        B = votes.shape[0]
        dev = votes.device
        if self.multi_label:
            # d_i for every output; negative outputs get scaled Type II probability.
            yb = y.to(torch.bool)
            q = torch.where(yb, self.T, -self.T)
            d = (q - votes).abs() / (2.0 * self.T)  # (B, K)
            d = torch.where(yb, d, (d * self.negative_scale).clamp(max=1.0))
            r = torch.rand(B, self.n_outputs, self.n_clauses, device=dev)
            sel = r < d.unsqueeze(2)  # (B, K, C)
            wpos = (self.weights.transpose(0, 1) >= 0).unsqueeze(0)  # (1, K, C)
            agree = yb.unsqueeze(2) == wpos
            type_i = (sel & agree).any(dim=1)
            type_ii = (sel & ~agree).any(dim=1)
            # weight deltas: +1 where y_i=1, -1 where y_i=0 for selected & matching pairs
            sign = torch.where(yb, 1.0, -1.0).unsqueeze(2)  # (B, K, 1)
            aux = (sel.to(self.compute_dtype) * sign)  # (B, K, C)
            return type_i, type_ii, aux

        y_neg = F.sample_negative_classes(
            y, self.n_outputs, votes=votes, focused=self.focused_negative_sampling
        )
        ar = torch.arange(B, device=dev)
        p_tgt = F.feedback_probabilities(votes[ar, y], self.T, target=True)
        p_neg = F.feedback_probabilities(votes[ar, y_neg], self.T, target=False)
        sel_tgt = torch.rand(B, self.n_clauses, device=dev) < p_tgt.unsqueeze(1)
        sel_neg = torch.rand(B, self.n_clauses, device=dev) < p_neg.unsqueeze(1)
        w_t = self.weights.transpose(0, 1)  # (K, C)
        pos_tgt = w_t[y] >= 0  # (B, C)
        pos_neg = w_t[y_neg] >= 0
        type_i = (sel_tgt & pos_tgt) | (sel_neg & ~pos_neg)
        type_ii = (sel_tgt & ~pos_tgt) | (sel_neg & pos_neg)
        aux = (sel_tgt, sel_neg, y, y_neg)
        return type_i, type_ii, aux

    def _accumulate_weights(
        self, acc: FeedbackAccumulator, aux: Any, clause_out: Tensor, sel_fire_i: Tensor, sel_fire_ii: Tensor
    ) -> None:
        cd = self.compute_dtype
        fire = clause_out.to(cd)
        if self.multi_label:
            # aux: (B, K, C) signed selection -> sum over batch of sel * fire -> (C, K)
            dw = torch.einsum("bkc,bc->ck", aux, fire)
        else:
            sel_tgt, sel_neg, y, y_neg = aux
            oh_t = torch.nn.functional.one_hot(y, self.n_outputs).to(cd)
            oh_n = torch.nn.functional.one_hot(y_neg, self.n_outputs).to(cd)
            dw = (sel_tgt.to(cd) * fire).transpose(0, 1) @ oh_t - (
                sel_neg.to(cd) * fire
            ).transpose(0, 1) @ oh_n
        if "dw" in acc.extra:
            acc.extra["dw"] += dw
        else:
            acc.extra["dw"] = dw

    def _commit_weights(self, acc: FeedbackAccumulator) -> None:
        if "dw" in acc.extra:
            self.weights.add_(acc.extra["dw"].to(self.weights.dtype))

    # ------------------------------------------------------------------ interpretation
    def rules(
        self,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        skip_empty: bool = True,
        min_abs_weight: int = 1,
    ) -> List[str]:
        """``IF ... THEN`` rules listing every output the clause votes for/against with its weight."""
        names = list(class_names or self.class_names or [str(k) for k in range(self.n_outputs)])
        out = []
        W = self.weights.tolist()
        for j in range(self.n_clauses_total):
            if skip_empty and int(self.include_count[j]) == 0:
                continue
            votes = []
            for i, w in enumerate(W[j]):
                if abs(w) >= min_abs_weight:
                    votes.append(f"{'+' if w >= 0 else '-'}{abs(w)} {names[i]}")
            if not votes:
                continue
            out.append(f"IF {self.clause_expression(j, feature_names)} THEN {', '.join(votes)}")
        return out

    def extra_repr(self) -> str:
        return (
            f"n_outputs={self.n_outputs}, n_clauses={self.n_clauses}, T={self.T}, "
            f"multi_label={self.multi_label}, " + super().extra_repr()
        )
