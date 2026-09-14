"""Multi-class Tsetlin machine classifier (vanilla and weighted)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .. import functional as F
from ..utils import as_long_tensor
from .base import FeedbackAccumulator, TsetlinMachineBase

__all__ = ["TsetlinMachine"]


class TsetlinMachine(TsetlinMachineBase):
    """Multi-class Tsetlin machine with class-owned clauses of positive/negative polarity.

    Each of the ``n_classes`` classes owns ``n_clauses`` clauses. The first half of them
    vote *for* the class (positive polarity), the second half vote *against* it (negative
    polarity). The class sum is ``v_k = sum(w_j c_j) - sum(w_j c_j)`` over the two halves,
    clamped to ``[-T, T]``, and the prediction is ``argmax_k v_k``.

    Learning follows the standard algorithm (Granmo, 2018): for an example of class ``y``
    the clauses of ``y`` receive feedback with probability ``(T - v_y)/(2T)`` (Type I to
    positive clauses, Type II to negative ones) and the clauses of one randomly chosen
    other class ``q`` receive feedback with probability ``(T + v_q)/(2T)`` (Type II to
    positive, Type I to negative clauses).

    Args:
        n_features: number of Boolean features (``None`` = infer from the first batch).
        n_classes: number of classes (``>= 2``). Labels are integers ``0..n_classes-1``.
        n_clauses: clauses **per class**.
        T: vote margin (threshold). Larger ``T`` makes more clauses cooperate per class.
        s: specificity.
        weighted: learn an integer weight per clause (Integer-Weighted TM). Weights start
            at 1, grow with Type Ia feedback and shrink with Type II feedback (min 0).
        focused_negative_sampling: sample the negative class proportionally to how strongly
            it currently votes, instead of uniformly.
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachineBase`
            (``n_states``, ``boost_true_positive``, ``max_included_literals``,
            ``drop_clause_p``, ``drop_literal_p``, ``init``, ``state_dtype``, ...).

    Shape:
        - input ``x``: ``(B, F)`` Boolean (extra trailing dimensions are flattened).
        - output of :meth:`forward`: ``(B, n_classes)`` float vote sums in ``[-T, T]``.

    Example:
        >>> model = TsetlinMachine(n_features=12, n_classes=2, n_clauses=20, T=10, s=3.9)
        >>> votes = model.update(x, y)      # learn from a batch, returns pre-update votes
        >>> pred = model(x).argmax(dim=1)   # predict
    """

    weights: Tensor
    clause_class: Tensor
    clause_polarity: Tensor

    def __init__(
        self,
        n_features: Optional[int],
        n_classes: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        weighted: bool = False,
        focused_negative_sampling: bool = False,
        **kwargs,
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2 (use two classes for binary problems)")
        if n_clauses < 2:
            raise ValueError("n_clauses (per class) must be >= 2")
        if T <= 0:
            raise ValueError("T must be positive")
        self.n_classes = int(n_classes)
        self.n_clauses = int(n_clauses)
        self.T = float(T)
        self.weighted = bool(weighted)
        self.focused_negative_sampling = bool(focused_negative_sampling)
        super().__init__(n_features, self.n_classes * self.n_clauses, s=s, **kwargs)

        C = self.n_clauses_total
        clause_class = torch.arange(C) // self.n_clauses
        within = torch.arange(C) % self.n_clauses
        n_pos = (self.n_clauses + 1) // 2
        polarity = torch.where(within < n_pos, 1.0, -1.0)
        self.register_buffer("clause_class", clause_class, persistent=False)
        self.register_buffer("clause_polarity", polarity, persistent=False)
        self.register_buffer("weights", torch.ones(C, dtype=torch.int32))
        self.class_names: Optional[List[str]] = None

    # ------------------------------------------------------------------ output layer
    @property
    def n_outputs(self) -> int:
        return self.n_classes

    def _signed_weights(self) -> Tensor:
        """Per-clause signed vote weight ``polarity * weight`` -> ``(C,)`` float."""
        w = self.clause_polarity
        if self.weighted:
            w = w * self.weights.to(w.dtype)
        return w

    def _votes(self, clause_out: Tensor, clamp: bool = True) -> Tensor:
        B = clause_out.shape[0]
        w = self._signed_weights().view(1, self.n_classes, self.n_clauses)
        v = (clause_out.to(w.dtype).view(B, self.n_classes, self.n_clauses) * w).sum(dim=2)
        if clamp:
            v = v.clamp_(-self.T, self.T)
        return v

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:
        y = as_long_tensor(y, device=device).view(-1)
        if y.shape[0] != n:
            raise ValueError(f"x has {n} examples but y has {y.shape[0]} labels")
        if bool((y < 0).any()) or bool((y >= self.n_classes).any()):
            raise ValueError(f"labels must lie in [0, {self.n_classes - 1}]")
        return y

    # ------------------------------------------------------------------ inference API
    def predict(self, x) -> Tensor:
        """Predicted class index per example ``(B,)``."""
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x, method: str = "linear", temperature: float = 1.0) -> Tensor:
        """Class probabilities ``(B, n_classes)``.

        ``method="linear"`` uses the calibrated score of Abeyrathna et al.,
        ``p_k ∝ (1 + v_k/T)/2``; ``method="softmax"`` applies a softmax to ``v/T``.
        """
        votes = self.forward(x)
        if method == "softmax":
            return F.predict_proba_from_votes(votes, self.T, temperature)
        p = 0.5 * (1.0 + votes / self.T)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def confidence(self, x) -> Tensor:
        """Confidence of the winning class in ``[0, 1]``: ``(v_max + T) / (2T)``."""
        return F.confidence_from_votes(self.forward(x), self.T)

    # ------------------------------------------------------------------ feedback policy
    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:
        B = votes.shape[0]
        dev = votes.device
        y_neg = F.sample_negative_classes(
            y, self.n_classes, votes=votes, focused=self.focused_negative_sampling
        )
        ar = torch.arange(B, device=dev)
        p_tgt = F.feedback_probabilities(votes[ar, y], self.T, target=True)
        p_neg = F.feedback_probabilities(votes[ar, y_neg], self.T, target=False)

        # Per-class feedback probability (0 for classes that are neither the target nor the
        # sampled negative class), broadcast over the clauses of each class.
        p_class = torch.zeros(B, self.n_classes, device=dev)
        p_class[ar, y] = p_tgt
        p_class[ar, y_neg] = p_neg
        r = torch.rand(B, self.n_classes, self.n_clauses, device=dev)
        selected = r < p_class.unsqueeze(2)  # (B, K, n)

        is_tgt = torch.zeros(B, self.n_classes, dtype=torch.bool, device=dev)
        is_tgt[ar, y] = True
        is_neg = torch.zeros_like(is_tgt)
        is_neg[ar, y_neg] = True
        positive = (self.clause_polarity.view(self.n_classes, self.n_clauses) > 0).unsqueeze(0)
        type_i = selected & ((is_tgt.unsqueeze(2) & positive) | (is_neg.unsqueeze(2) & ~positive))
        type_ii = selected & ((is_tgt.unsqueeze(2) & ~positive) | (is_neg.unsqueeze(2) & positive))
        return type_i.view(B, -1), type_ii.view(B, -1), None

    def _accumulate_weights(
        self, acc: FeedbackAccumulator, aux: Any, clause_out: Tensor, sel_fire_i: Tensor, sel_fire_ii: Tensor
    ) -> None:
        if not self.weighted:
            return
        # Type Ia strengthens, Type II weakens the vote of a matching clause.
        dw = sel_fire_i.sum(dim=0) - sel_fire_ii.sum(dim=0)
        if "dw" in acc.extra:
            acc.extra["dw"] += dw
        else:
            acc.extra["dw"] = dw

    def _commit_weights(self, acc: FeedbackAccumulator) -> None:
        if self.weighted and "dw" in acc.extra:
            self.weights.add_(acc.extra["dw"].to(self.weights.dtype)).clamp_(min=0)

    # ------------------------------------------------------------------ interpretation
    def clause_info(self, clause: int, feature_names: Optional[Sequence[str]] = None) -> Dict:
        """Metadata for one clause: class, polarity, weight and the literal expression."""
        cls = int(self.clause_class[clause])
        pol = int(self.clause_polarity[clause])
        name = self.class_names[cls] if self.class_names else str(cls)
        return {
            "clause": int(clause),
            "class": cls,
            "class_name": name,
            "polarity": pol,
            "weight": int(self.weights[clause]) if self.weighted else 1,
            "n_literals": int(self.include_count[clause]),
            "expression": self.clause_expression(clause, feature_names),
        }

    def rules(
        self,
        feature_names: Optional[Sequence[str]] = None,
        class_names: Optional[Sequence[str]] = None,
        skip_empty: bool = True,
        polarity: Optional[int] = None,
        class_index: Optional[int] = None,
    ) -> List[str]:
        """Human readable ``IF ... THEN`` rules for all (non-empty) clauses."""
        names = list(class_names or self.class_names or [str(k) for k in range(self.n_classes)])
        out = []
        for j in range(self.n_clauses_total):
            if skip_empty and int(self.include_count[j]) == 0:
                continue
            pol = int(self.clause_polarity[j])
            if polarity is not None and pol != polarity:
                continue
            k = int(self.clause_class[j])
            if class_index is not None and k != class_index:
                continue
            expr = self.clause_expression(j, feature_names)
            w = f" (weight {int(self.weights[j])})" if self.weighted else ""
            verdict = f"THEN {names[k]}" if pol > 0 else f"THEN NOT {names[k]}"
            out.append(f"IF {expr} {verdict}{w}")
        return out

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"n_classes={self.n_classes}, n_clauses_per_class={self.n_clauses}, T={self.T}, "
            f"weighted={self.weighted}, " + base
        )
