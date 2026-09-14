"""Interpretability tools: rules, clause activity, explanations and closed-form feature
importance (Blakely & Granmo, 2020)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor

from .models import (
    CoalescedTsetlinMachine,
    RegressionTsetlinMachine,
    TsetlinMachine,
    TsetlinMachineBase,
)
from .utils import as_bool_tensor

__all__ = [
    "rules",
    "clause_activity",
    "clause_precision",
    "global_feature_importance",
    "local_feature_importance",
    "aggregate_literal_importance",
    "explain",
    "Explanation",
    "MatchedClause",
]


def rules(model: TsetlinMachineBase, **kwargs) -> List[str]:
    """Human readable rules of any model (delegates to ``model.rules``)."""
    return model.rules(**kwargs)  # type: ignore[attr-defined]


def _clause_weight_matrix(model: TsetlinMachineBase) -> Tensor:
    """Signed clause weights per output ``(C, K)`` as float."""
    if isinstance(model, TsetlinMachine):
        w = model._signed_weights()  # (C,)
        W = torch.zeros(model.n_clauses_total, model.n_classes, device=w.device)
        W[torch.arange(model.n_clauses_total, device=w.device), model.clause_class] = w
        return W
    if isinstance(model, CoalescedTsetlinMachine):
        return model.weights.to(torch.float32)
    if isinstance(model, RegressionTsetlinMachine):
        w = model.weights.to(torch.float32) if model.weighted else torch.ones(model.n_clauses_total, device=model.device)
        return w.unsqueeze(1)
    raise TypeError("Unsupported model type")


@torch.no_grad()
def clause_activity(model: TsetlinMachineBase, x, y: Optional[Tensor] = None, n_classes: Optional[int] = None) -> Tensor:
    """How often each clause matches.

    Returns ``(C,)`` firing frequencies, or ``(K, C)`` per-class frequencies when labels
    ``y`` are given (row ``k`` = fraction of class-``k`` examples matched by each clause).
    """
    out = model.evaluate_clauses(x).to(torch.float32)  # (B, C)
    if y is None:
        return out.mean(dim=0)
    y = torch.as_tensor(y, device=out.device).long().view(-1)
    K = n_classes or int(y.max()) + 1
    oh = torch.nn.functional.one_hot(y, K).to(out.dtype)  # (B, K)
    counts = oh.sum(dim=0).clamp_min(1).unsqueeze(1)
    return (oh.transpose(0, 1) @ out) / counts


@torch.no_grad()
def clause_precision(model: TsetlinMachine, x, y) -> Dict[str, Tensor]:
    """For each clause of a multi-class model: how often it fires (``support``) and how often
    the example belongs to the clause's own class when it fires (``precision``). Negative
    polarity clauses are scored against *not* their class."""
    out = model.evaluate_clauses(x)  # (B, C)
    y = torch.as_tensor(y, device=out.device).long().view(-1)
    own = (y.unsqueeze(1) == model.clause_class.unsqueeze(0))  # (B, C)
    target = torch.where(model.clause_polarity.unsqueeze(0) > 0, own, ~own)
    fires = out.to(torch.float32)
    support = fires.sum(dim=0)
    precision = (fires * target.to(torch.float32)).sum(dim=0) / support.clamp_min(1)
    return {"support": support / out.shape[0], "precision": precision}


def global_feature_importance(model: TsetlinMachineBase, class_index: Optional[int] = None, normalize: bool = True) -> Tensor:
    """Closed-form *global* literal importance.

    For output ``k``, the strength of literal ``l`` is the (weighted) fraction of positive
    clauses voting for ``k`` that include ``l``:
    ``g_k[l] = (1/m) * sum_{j: W[j,k] > 0} W[j,k] * include[j, l]``.

    Returns ``(2F,)`` for one class or ``(K, 2F)`` for all classes. The first ``F`` entries
    correspond to the positive literals ``x_l`` and the last ``F`` to ``NOT x_l``.
    """
    inc = model.included_mask().to(torch.float32)  # (C, 2F)
    W = _clause_weight_matrix(model).clamp(min=0)  # positive votes only
    m = (W > 0).sum(dim=0).clamp_min(1).to(torch.float32)  # positive clauses per output
    g = (W.transpose(0, 1) @ inc) / m.unsqueeze(1)  # (K, 2F)
    if normalize:
        g = g / g.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return g if class_index is None else g[class_index]


@torch.no_grad()
def local_feature_importance(model: TsetlinMachineBase, x, class_index: Optional[Tensor] = None, normalize: bool = True) -> Tensor:
    """Closed-form *local* literal importance for each example.

    ``l_k[l, X] = sum_{j: W[j,k] > 0, C_j(X) = 1} W[j,k] * include[j, l] * literal_l(X)`` —
    the weight of the matching positive clauses of class ``k`` that rely on literal ``l``.
    By default ``k`` is the predicted class. Returns ``(B, 2F)``.
    """
    xb = model._prepare(x)
    votes = model(xb)
    if class_index is None:
        k = votes.argmax(dim=1)
    else:
        k = torch.as_tensor(class_index, device=votes.device).long().view(-1).expand(votes.shape[0]) if torch.as_tensor(class_index).numel() == 1 else torch.as_tensor(class_index, device=votes.device).long()
    fires = model.evaluate_clauses(xb).to(torch.float32)  # (B, C)
    W = _clause_weight_matrix(model).clamp(min=0)  # (C, K)
    w_sel = W[:, k].transpose(0, 1)  # (B, C)
    inc = model.included_mask().to(torch.float32)  # (C, 2F)
    lit = torch.cat([xb.reshape(xb.shape[0], -1), ~xb.reshape(xb.shape[0], -1)], dim=1).to(torch.float32) if not hasattr(model, "patch_size") else None
    contrib = (fires * w_sel) @ inc  # (B, 2F)
    if lit is not None:
        contrib = contrib * lit
    if normalize:
        contrib = contrib / contrib.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return contrib


def aggregate_literal_importance(importance: Tensor, groups: Sequence[int], n_groups: Optional[int] = None, combine_negated: bool = True) -> Tensor:
    """Sum literal importances over groups of Boolean features (e.g. the thermometer bits of
    one original feature). ``groups[i]`` is the group id of Boolean feature ``i``.

    ``importance`` has ``2F`` entries (or ``(..., 2F)``); with ``combine_negated`` the
    ``NOT`` literals are added to the same group, otherwise ``2 * n_groups`` values are
    returned (positive groups first).
    """
    g = torch.as_tensor(list(groups), device=importance.device).long()
    Fn = g.numel()
    G = int(n_groups or int(g.max()) + 1)
    pos = importance[..., :Fn]
    neg = importance[..., Fn:]
    out_pos = torch.zeros(*importance.shape[:-1], G, device=importance.device, dtype=importance.dtype)
    out_neg = torch.zeros_like(out_pos)
    out_pos.index_add_(-1, g, pos)
    out_neg.index_add_(-1, g, neg)
    return out_pos + out_neg if combine_negated else torch.cat([out_pos, out_neg], dim=-1)


@dataclass
class MatchedClause:
    clause: int
    output: int
    weight: float
    expression: str


@dataclass
class Explanation:
    """Why the model predicted what it predicted for one example."""

    prediction: int
    votes: List[float]
    matched: List[MatchedClause] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"prediction: {self.prediction}  votes: {[round(v, 2) for v in self.votes]}"]
        for m in self.matched:
            sign = "+" if m.weight >= 0 else "-"
            lines.append(f"  [{sign}{abs(m.weight):g} -> output {m.output}] clause {m.clause}: IF {m.expression}")
        return "\n".join(lines)


@torch.no_grad()
def explain(model: TsetlinMachineBase, x, feature_names: Optional[Sequence[str]] = None, output: Optional[int] = None, max_clauses: int = 20) -> Explanation:
    """List the clauses that matched a single example and how they voted.

    Args:
        model: a classifier (vanilla or coalesced).
        x (Tensor | ndarray): one example (``(F,)`` or ``(1, F)``; one image for convolutional
            models).
        feature_names: names used to render clause expressions.
        output: restrict to clauses voting on this output (default: the predicted class).
        max_clauses: keep the ``max_clauses`` strongest matches.
    """
    xb = as_bool_tensor(x)
    if xb.dim() == 1 or (hasattr(model, "input_shape") and model.input_shape is not None and tuple(xb.shape) == tuple(model.input_shape)):  # type: ignore[attr-defined]
        xb = xb.unsqueeze(0)
    votes = model(xb)[0]
    pred = int(votes.argmax())
    k = pred if output is None else int(output)
    fires = model.evaluate_clauses(xb)[0]  # (C,)
    W = _clause_weight_matrix(model)[:, k]
    idx = torch.nonzero(fires & (W != 0), as_tuple=False).flatten().tolist()
    idx.sort(key=lambda j: -abs(float(W[j])))
    matched = [MatchedClause(j, k, float(W[j]), model.clause_expression(j, feature_names)) for j in idx[:max_clauses]]
    return Explanation(pred, votes.tolist(), matched)
