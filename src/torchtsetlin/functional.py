"""Stateless building blocks of Tsetlin machine learning, expressed as tensor operations.

Everything here works on any device (CPU / CUDA / MPS) and never touches autograd.
The model classes in :mod:`torchtsetlin.models` are thin, stateful wrappers around
these functions, so advanced users can compose their own Tsetlin machine variants.

Conventions
-----------
* A **literal vector** is ``[x, ~x]`` (length ``2F`` for ``F`` Boolean features).
* Each clause owns one **Tsetlin automaton (TA)** per literal. Its state is an integer
  in ``[0, 2N-1]``; states ``>= N`` mean the literal is *included* (memorized) in the
  clause, states ``< N`` mean *excluded* (forgotten). ``N`` is ``n_states``.
* Clause output is the AND of its included literals. An empty clause (no included
  literal) evaluates to ``1`` during learning and ``0`` during prediction.
* Feedback probabilities follow the *specificity* ``s``: memorize probability
  ``(s-1)/s``, forget probability ``1/s`` (Type Ia/Ib), Type II is deterministic.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "to_literals",
    "clause_violations",
    "clause_outputs",
    "vote_sums",
    "feedback_probabilities",
    "sample_negative_classes",
    "type_i_counts",
    "type_ii_counts",
    "apply_feedback",
    "predict_proba_from_votes",
    "confidence_from_votes",
    "boolean_pool",
    "thermometer_cast",
]


# --------------------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------------------
def to_literals(x: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Turn Boolean features ``x`` of shape ``(..., F)`` into literals ``[x, ~x]`` of
    shape ``(..., 2F)`` with the requested floating dtype (for matmul-based evaluation)."""
    if x.dtype != torch.bool:
        x = x != 0
    return torch.cat([x, ~x], dim=-1).to(dtype)


def clause_violations(literals: Tensor, include: Tensor) -> Tensor:
    """Number of *included but False* literals per clause.

    Args:
        literals: ``(B, 2F)`` floating tensor with 0/1 values.
        include: ``(C, 2F)`` floating tensor with 0/1 values (``1`` = literal included).

    Returns:
        ``(B, C)`` floating tensor; a clause matches an example iff its entry is ``0``.
    """
    # (1 - literals) marks False literals; a False literal that is included violates the AND.
    return torch.matmul(1.0 - literals, include.transpose(0, 1))


def clause_outputs(
    literals: Tensor,
    include: Tensor,
    include_count: Optional[Tensor] = None,
    empty_value: bool = True,
) -> Tensor:
    """Evaluate all clauses on a batch of literal vectors.

    Args:
        literals: ``(B, 2F)`` 0/1 floating tensor.
        include: ``(C, 2F)`` 0/1 floating tensor of included literals.
        include_count: optional ``(C,)`` number of included literals per clause; needed
            when ``empty_value`` is ``False``.
        empty_value: value of a clause without included literals (``True`` while
            learning, ``False`` when predicting).

    Returns:
        ``(B, C)`` boolean tensor of clause outputs.
    """
    out = clause_violations(literals, include) == 0
    if not empty_value:
        if include_count is None:
            include_count = include.sum(dim=1)
        out = out & (include_count > 0).unsqueeze(0)
    return out


def vote_sums(clause_out: Tensor, weights: Tensor, T: Optional[float] = None) -> Tensor:
    """Weighted vote (class) sums ``clause_out @ weights`` optionally clamped to ``[-T, T]``.

    Args:
        clause_out: ``(B, C)`` boolean / 0-1 tensor.
        weights: ``(C, K)`` integer or floating weights (``K`` outputs).
        T: vote margin; if given the sums are clamped to ``[-T, T]``.
    """
    votes = torch.matmul(clause_out.to(torch.float32), weights.to(torch.float32))
    if T is not None:
        votes = votes.clamp_(-float(T), float(T))
    return votes


# --------------------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------------------
def feedback_probabilities(votes: Tensor, T: float, target: bool) -> Tensor:
    """Probability of giving feedback to a clause given the (clamped) vote sum.

    * For the **target** class the probability is ``(T - v) / (2T)``: far below the margin
      -> update aggressively, at the margin -> stop.
    * For a **negative** class it is ``(T + v) / (2T)``.
    """
    v = votes.clamp(-float(T), float(T))
    if target:
        return (float(T) - v) / (2.0 * float(T))
    return (float(T) + v) / (2.0 * float(T))


def sample_negative_classes(
    y: Tensor,
    n_classes: int,
    votes: Optional[Tensor] = None,
    focused: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample one negative (non-target) class per example.

    With ``focused=True`` ("focused negative sampling") classes are sampled with
    probability proportional to how strongly they currently vote, i.e. the classes the
    machine confuses with the target are corrected first. Otherwise sampling is uniform.
    """
    if n_classes < 2:
        raise ValueError("Need at least two classes to sample a negative class.")
    if focused and votes is not None:
        w = votes.detach().to(torch.float32)
        w = w - w.min(dim=1, keepdim=True).values + 1.0  # strictly positive
        w = w.scatter(1, y.view(-1, 1), 0.0)  # never the target itself
        return torch.multinomial(w, 1, generator=generator).squeeze(1)
    offset = torch.randint(1, n_classes, (y.shape[0],), device=y.device, generator=generator)
    return (y + offset) % n_classes


def type_i_counts(
    literals: Tensor, sel_fire: Tensor, sel_nofire: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Aggregate Type I feedback events over a batch.

    Args:
        literals: ``(B, 2F)`` 0/1 floating tensor.
        sel_fire: ``(B, C)`` 0/1 floating tensor; example ``b`` gives Type I feedback to a
            clause ``j`` **that matched** it (Type Ia).
        sel_nofire: ``(B, C)`` 0/1 floating tensor; Type I feedback to a clause that did
            **not** match (Type Ib).

    Returns:
        ``(n_true, n_false, n_ib)`` where ``n_true[j,k]`` counts Type Ia events with literal
        ``k`` True (candidates for memorization), ``n_false[j,k]`` counts Type Ia events with
        literal ``k`` False (candidates for forgetting) and ``n_ib[j]`` counts Type Ib events
        (all literals are candidates for forgetting).
    """
    sel_fire_t = sel_fire.transpose(0, 1)  # (C, B)
    n_true = torch.matmul(sel_fire_t, literals)  # (C, 2F)
    n_false = torch.matmul(sel_fire_t, 1.0 - literals)
    n_ib = sel_nofire.sum(dim=0)  # (C,)
    return n_true, n_false, n_ib


def type_ii_counts(literals: Tensor, sel_fire: Tensor) -> Tensor:
    """Aggregate Type II feedback events: ``n2[j,k]`` counts (example, clause) pairs where the
    clause matched the example, received Type II feedback and literal ``k`` was False."""
    return torch.matmul(sel_fire.transpose(0, 1), 1.0 - literals)


def _binomial(count: Tensor, p: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """Binomial samples with a scalar probability (``count`` is a floating tensor)."""
    if p >= 1.0:
        return count
    if p <= 0.0:
        return torch.zeros_like(count)
    prob = torch.full((1,) * count.dim(), float(p), dtype=count.dtype, device=count.device)
    return torch.binomial(count, prob.expand_as(count), generator=generator)


def apply_feedback(
    state: Tensor,
    n_true: Tensor,
    n_false: Tensor,
    n_ib: Tensor,
    n2: Tensor,
    *,
    n_states: int,
    s: float,
    boost_true_positive: bool = True,
    max_included_literals: Optional[int] = None,
    include_count: Optional[Tensor] = None,
    literal_active: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Turn aggregated feedback counts into Tsetlin automata transitions (in place).

    Args:
        state: ``(C, 2F)`` integer TA states, modified in place.
        n_true: ``(C, 2F)`` Type Ia events with the literal True (from :func:`type_i_counts`).
        n_false: ``(C, 2F)`` Type Ia events with the literal False.
        n_ib: ``(C,)`` Type Ib events per clause.
        n2: ``(C, 2F)`` Type II events with the literal False (from :func:`type_ii_counts`).
            All four count tensors **are used as scratch space** (modified in place) — do not
            reuse them afterwards.
        n_states: ``N`` states per action (include iff ``state >= N``).
        s: specificity; memorize probability ``(s-1)/s`` and forget probability ``1/s``.
        boost_true_positive: if ``True`` the memorization of True literals in matching
            clauses is deterministic (probability 1) instead of ``(s-1)/s``.
        max_included_literals: clause size budget ``b``. A matching clause that includes more
            than ``b`` literals gets no Type Ia feedback; all its Type I events act as Type Ib
            (forgetting) until it shrinks back (Abeyrathna et al., 2023).
        include_count: ``(C,)`` included literals per clause (needed for the constraint).
        literal_active: optional ``(2F,)`` 0/1 mask; inactive literals receive no feedback
            (literal dropout).
        generator: optional RNG.

    Returns:
        The updated ``state`` tensor (same object).
    """
    N = int(n_states)
    p_forget = 1.0 / float(s)
    p_mem = 1.0 if boost_true_positive else (float(s) - 1.0) / float(s)

    if max_included_literals is not None:
        if include_count is None:
            include_count = (state >= N).sum(dim=1)
        oversize = (include_count > int(max_included_literals)).unsqueeze(1)
        if bool(oversize.any()):
            # Every Type Ia event of an oversized clause becomes a Type Ib event.
            n_ia_events = n_true[:, :1] + n_false[:, :1]  # identical for every literal
            n_ib = n_ib + (n_ia_events * oversize).squeeze(1)
            keep = (~oversize).to(n_true.dtype)
            n_true.mul_(keep)
            n_false.mul_(keep)

    # Type Ia – memorize True literals of matching clauses.
    inc = _binomial(n_true, p_mem, generator)

    # Type Ia (False literals of matching clauses) + Type Ib (all literals of non-matching
    # clauses) – forget with probability 1/s. Both are Bernoulli(1/s) events, so the sum of
    # counts is again binomial.
    n_false.add_(n_ib.unsqueeze(1))
    dec = _binomial(n_false, p_forget, generator)

    # Type II – deterministically push excluded False literals towards inclusion. Each event
    # moves the automaton one step, but never beyond the include boundary (as in the
    # sequential algorithm, where the clause stops matching once the literal is included).
    room = torch.sub(N, state).clamp_(min=0)
    inc2 = torch.minimum(n2, room.to(n2.dtype), out=n2)

    delta = inc.sub_(dec).add_(inc2)
    if literal_active is not None:
        delta.mul_(literal_active.to(delta.dtype).unsqueeze(0))
    state.add_(delta.to(state.dtype)).clamp_(0, 2 * N - 1)
    return state


# --------------------------------------------------------------------------------------
# Probabilities & confidence
# --------------------------------------------------------------------------------------
def predict_proba_from_votes(votes: Tensor, T: float, temperature: float = 1.0) -> Tensor:
    """Softmax over vote sums normalised by ``T`` – a convenient (heuristic) probability.

    ``votes / T`` lies in ``[-1, 1]``; dividing by ``temperature`` sharpens (<1) or flattens
    (>1) the distribution.
    """
    z = votes.to(torch.float32) / (float(T) * float(temperature))
    return torch.softmax(z, dim=-1)


def confidence_from_votes(votes: Tensor, T: float) -> Tensor:
    """Confidence in the predicted class, as in Chapter 7 of *An Introduction to Tsetlin
    Machines*: the winning vote sum normalised to ``[0, 1]`` by the vote margin,
    ``(v_max + T) / (2T)``. Returns shape ``(B,)``."""
    v = votes.to(torch.float32).clamp(-float(T), float(T))
    return (v.max(dim=-1).values + float(T)) / (2.0 * float(T))


# --------------------------------------------------------------------------------------
# TM-native context primitives (dense / stacked models)
# --------------------------------------------------------------------------------------
def boolean_pool(x: Tensor, factor: int = 2, mode: str = "or") -> Tensor:
    """Pool Boolean planes ``(B, Z, H, W)`` by an integer ``factor``.

    ``mode="or"`` keeps a bit that is set anywhere in the window (the Boolean analogue of
    max-pooling, and what *CTM-UNet* uses between blocks); ``mode="and"`` keeps only bits set
    everywhere in it. Both stay Boolean, so the result can be fed straight back into another
    Tsetlin machine — no float parameter and no gradient anywhere, unlike a strided
    convolution.

    Args:
        x: ``(B, Z, H, W)`` Boolean-like tensor.
        factor: pooling window and stride.
        mode: ``"or"`` or ``"and"``.

    Returns:
        ``(B, Z, H // factor, W // factor)`` bool.
    """
    if x.dim() != 4:
        raise ValueError("boolean_pool expects (B, Z, H, W)")
    k = int(factor)
    if k < 1:
        raise ValueError("factor must be >= 1")
    if k == 1:
        return x.to(torch.bool)
    f = x.to(torch.float32)
    if mode == "or":
        return torch.nn.functional.max_pool2d(f, k) > 0.5
    if mode == "and":
        return torch.nn.functional.max_pool2d(-f, k).neg() > 0.5
    raise ValueError("mode must be 'or' or 'and'")


def thermometer_cast(votes: Tensor, levels: Sequence[float]) -> Tensor:
    """Turn vote sums into Boolean planes without losing their ordering.

    A Tsetlin machine can only read Boolean features, so passing one machine's votes to
    another needs a Booleanization that preserves *how strong* the vote was. A thermometer
    does: the planes of ``v`` are ``[v >= l]`` for each level ``l``, so a larger vote sets a
    superset of the bits a smaller one does, and a clause can refer to "at least this
    confident" with one literal.

    Args:
        votes: ``(B, K, H, W)`` vote map (or ``(B, K)`` vote sums).
        levels: ascending thresholds, e.g. ``(-40, -20, -8, 0, 8, 20, 40)``.

    Returns:
        ``(B, K * len(levels), H, W)`` bool for a vote map, ``(B, K * len(levels))`` for
        vote sums. Channel order is class-major: all levels of class 0, then class 1, ...
    """
    lv = torch.as_tensor(list(levels), dtype=votes.dtype, device=votes.device)
    if votes.dim() == 4:
        planes = votes.unsqueeze(2) >= lv.view(1, 1, -1, 1, 1)
        return planes.reshape(votes.shape[0], -1, votes.shape[2], votes.shape[3])
    if votes.dim() == 2:
        return (votes.unsqueeze(2) >= lv.view(1, 1, -1)).reshape(votes.shape[0], -1)
    raise ValueError("thermometer_cast expects (B, K, H, W) or (B, K)")
