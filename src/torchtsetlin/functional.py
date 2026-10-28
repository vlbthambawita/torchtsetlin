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

import math
import os
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

__all__ = [
    "to_literals",
    "clause_violations",
    "clause_outputs",
    "vote_sums",
    "feedback_probabilities",
    "sample_negative_classes",
    "sample_forget_events",
    "apply_feedback_rows",
    "commit_single_example",
    "commit_dense",
    "set_compile",
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
    # The literals follow the dtype of ``include`` (the models keep it in float16 on CUDA).
    if literals.dtype != include.dtype:
        literals = literals.to(include.dtype)
    if literals.dim() == 2 and literals.device.type == "cpu" and literals.shape[0] <= 256:
        # clause-major (C, 2F) @ (2F, B) is up to ~2x faster than (B, 2F) @ (2F, C) on the CPU
        # for a few rows (flat mini-batches), but ~3x slower for the thousands of patch rows of
        # a convolutional batch (B=16 MNIST images: 582 vs 187 ms on 16 threads)
        return torch.matmul(include, (1.0 - literals).transpose(0, 1)).transpose(0, 1).contiguous()
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
    T: Optional[float] = None,
) -> Tensor:
    """Sample one negative (non-target) class per example.

    With ``focused=True`` ("focused negative sampling") a non-target class ``k`` is sampled
    with probability proportional to its negative-feedback probability
    ``(T + clip(v_k, -T, T)) / 2T``, as in tmu and green-tsetlin, so the classes the machine
    confuses with the target are corrected first. This needs the vote margin ``T``.
    Otherwise sampling is uniform.
    """
    if n_classes < 2:
        raise ValueError("Need at least two classes to sample a negative class.")
    if focused and votes is not None:
        if T is None:
            raise ValueError("focused negative sampling needs the vote margin T.")
        w = feedback_probabilities(votes.detach().to(torch.float32), T, target=False)
        w = w + 1e-16  # as green-tsetlin: a class at -T keeps a tiny chance
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


# --------------------------------------------------------------------------------------
# Compiled kernels (torch.compile)
# --------------------------------------------------------------------------------------
# The update samples millions of small binomials per batch. Eager CPU RNG is single-threaded
# and `torch.binomial` is slow, so on the CPU (and for single examples on CUDA) the hot loops
# are written as elementwise inverse-CDF kernels that torch.compile fuses, drawing the
# uniforms inside the kernel. Every compiled path has an eager twin that is used when the work
# is small (compiling costs seconds), when an explicit generator is passed (compiled RNG is
# seeded from the default generator only), when disabled, or when compilation fails.
_COMPILE: Dict[str, Any] = {"enabled": os.environ.get("TORCHTSETLIN_COMPILE", "1") != "0"}
_KERNELS: Dict[Callable, Callable] = {}
_FAILED: set = set()
#: Tensor elements below which the eager code paths are used.
COMPILE_MIN_ELEMENTS = 1 << 18
# The inverse-CDF kernels read one fixed-shape table (every new shape would recompile them):
# counts up to _TABLE_N - 1 events, the first _TABLE_K CDF columns. A draw beyond the table
# is resampled exactly from the tail; if that would be common, the eager sampler is used.
_TABLE_N = 64
_TABLE_K = 32
_TABLE_TAIL = 1e-3
_CDF_CACHE: Dict[float, Tensor] = {}


def set_compile(enabled: bool) -> None:
    """Enable or disable the ``torch.compile`` kernels of the update (default: enabled, or
    the environment variable ``TORCHTSETLIN_COMPILE=0`` to disable). The compiled and eager
    paths draw from the same distributions but consume random numbers differently, so a seeded
    run reproduces only under the same setting."""
    _COMPILE["enabled"] = bool(enabled)


def _compile_wanted(numel: int, generator: Optional[torch.Generator] = None) -> bool:
    return bool(_COMPILE["enabled"]) and generator is None and numel >= COMPILE_MIN_ELEMENTS


def _run_compiled(fn: Callable, *args):
    """``torch.compile(fn)(*args)``, or ``None`` if compilation is unavailable (no C++
    compiler or Triton, unsupported Python, code dynamo cannot trace, ...). A failure warns
    once and disables ``fn`` for the rest of the process; the caller then takes its eager
    path."""
    if not _COMPILE["enabled"] or fn in _FAILED:
        return None
    try:
        k = _KERNELS.get(fn)
        if k is None:
            # Kernels whose sizes change every update are compiled shape-generic up front;
            # the default would recompile for each new size before going dynamic.
            k = _KERNELS[fn] = torch.compile(fn, dynamic=True if fn in _DYNAMIC else None)
        return k(*args)
    except Exception as err:  # pragma: no cover - depends on the toolchain
        _FAILED.add(fn)
        warnings.warn(
            f"torch.compile failed for {fn.__qualname__} ({type(err).__name__}: {err}); "
            "torchtsetlin uses the eager code path. Set TORCHTSETLIN_COMPILE=0 to skip the attempt.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _binomial_cdf(p: float) -> Tensor:
    """``(_TABLE_N, _TABLE_N)`` float64 table ``P(Binomial(n, p) <= k)``, exactly ``2.0``
    where ``k >= n`` (so that a uniform ``u < 1`` never passes it)."""
    tab = _CDF_CACHE.get(float(p))
    if tab is None:
        n = torch.arange(_TABLE_N, dtype=torch.float64).unsqueeze(1)
        k = torch.arange(_TABLE_N, dtype=torch.float64)
        log_pmf = (torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k).clamp(min=0) + 1)
                   + k * math.log(p) + (n - k) * math.log1p(-p))
        cdf = torch.where(k <= n, log_pmf.exp(), torch.zeros(())).cumsum(1)
        tab = _CDF_CACHE[float(p)] = cdf.masked_fill_(k >= n, 2.0)
    return tab


def _table_fits(p: float, n_max: int) -> bool:
    """Whether counts up to ``n_max`` can use the kernel table: within its rows, and draws
    beyond its columns (resampled from the exact tail) are rare."""
    if n_max >= _TABLE_N:
        return False
    return n_max < _TABLE_K or 1.0 - float(_binomial_cdf(p)[n_max, _TABLE_K - 1]) < _TABLE_TAIL


def _thresholds(p: float, device: torch.device) -> Tensor:
    """The kernel table: float32 ``(_TABLE_N, _TABLE_K)``."""
    return _binomial_cdf(p)[:, :_TABLE_K].to(dtype=torch.float32, device=device)


def _icdf_count(u: Tensor, thr: Tensor) -> Tensor:
    """``#{k : u >= thr[..., k]}``, unrolled over the (small, fixed) table width."""
    out = (u >= thr[..., 0]).to(u.dtype)
    for k in range(1, thr.shape[-1]):
        out = out + (u >= thr[..., k]).to(u.dtype)
    return out


# A batched update aggregates up to one Type I event per example into an automaton, so the
# dense GPU commit (and the CPU kernels, for large batches or small s) needs counts beyond the
# narrow table. It searches one fixed square table
# (counts up to _WIDE_TABLE_N - 1, every CDF column) by bisection: log2(_WIDE_TABLE_N) gathers per
# automaton instead of an unrolled compare per column, and no tail to resample.
_WIDE_TABLE_N = 256
_WIDE_CDF_CACHE: Dict[float, Tensor] = {}
_WIDE_ON_DEVICE: Dict[Tuple[float, torch.device], Tensor] = {}


def _wide_thresholds(p: float, device: torch.device) -> Tensor:
    """Flattened float32 ``P(Binomial(n, p) <= k)`` for ``n, k < _WIDE_TABLE_N`` (``2.0`` where
    ``k >= n``)."""
    on_dev = _WIDE_ON_DEVICE.get((float(p), torch.device(device)))
    if on_dev is not None:  # cached per device: no host-to-device copy (and sync) per commit
        return on_dev
    tab = _WIDE_CDF_CACHE.get(float(p))
    if tab is None:
        n = torch.arange(_WIDE_TABLE_N, dtype=torch.float64).unsqueeze(1)
        k = torch.arange(_WIDE_TABLE_N, dtype=torch.float64)
        log_pmf = (torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma((n - k).clamp(min=0) + 1)
                   + k * math.log(p) + (n - k) * math.log1p(-p))
        cdf = torch.where(k <= n, log_pmf.exp(), torch.zeros(())).cumsum(1)
        tab = _WIDE_CDF_CACHE[float(p)] = cdf.masked_fill_(k >= n, 2.0).reshape(-1)
    out = _WIDE_ON_DEVICE[(float(p), torch.device(device))] = tab.to(dtype=torch.float32, device=device)
    return out


def _bisect_binomial(n: Tensor, thr: Tensor, flip: bool, width: int = _WIDE_TABLE_N) -> Tensor:
    """``Binomial(n, q)`` from the flattened :func:`_wide_thresholds` table of ``q``:
    ``#{k : u >= cdf[n, k]}`` by bisection over the non-decreasing row, searching the first
    ``width`` (a power of two ``> n``) columns. ``flip`` returns ``n - X`` instead, i.e.
    ``Binomial(n, 1 - q)``."""
    u = torch.rand(n.shape, dtype=thr.dtype, device=n.device)
    base = n.long() * _WIDE_TABLE_N
    lo = torch.zeros(n.shape, dtype=torch.long, device=n.device)
    step = width // 2
    while step:
        cand = lo + step
        lo = torch.where(u >= thr[base + cand - 1], cand, lo)
        step //= 2
    out = lo.to(n.dtype)
    return n - out if flip else out


def _binomial_kernel(count: Tensor, thr: Tensor) -> Tensor:
    u = torch.rand(count.shape, dtype=thr.dtype, device=count.device)
    return _icdf_count(u, thr[count.long()]).to(count.dtype)


def _forget_rows_kernel(sub: Tensor, n: Tensor, thr: Tensor, active: Optional[Tensor], N: int):
    u = torch.rand(sub.shape, dtype=thr.dtype, device=sub.device)
    dec = _icdf_count(u, thr[n.long()])  # (R, L) from per-row counts (R, 1)
    if active is not None:
        dec = dec * active
    old = sub.to(torch.int32)
    new = (old - dec.to(torch.int32)).clamp(min=0)
    return new.to(sub.dtype), (old >= N) & (new < N)


def _bisect_kernel(count: Tensor, thr: Tensor, width: int) -> Tensor:
    return _bisect_binomial(count, thr, False, width)


def _forget_rows_bisect_kernel(sub: Tensor, n: Tensor, thr: Tensor, active: Optional[Tensor], N: int,
                               width: int):
    dec = _bisect_binomial(n.expand(sub.shape), thr, False, width)  # (R, L) from (R, 1)
    if active is not None:
        dec = dec * active
    old = sub.to(torch.int32)
    new = (old - dec.to(torch.int32)).clamp(min=0)
    return new.to(sub.dtype), (old >= N) & (new < N)


def _bisect_width(n_max: int) -> int:
    """Bisection steps for counts up to ``n_max`` (a few distinct widths, each compiled once)."""
    return max(16, 1 << int(n_max).bit_length())


_DYNAMIC = {_binomial_kernel, _forget_rows_kernel, _bisect_kernel, _forget_rows_bisect_kernel}


def _tail_extra(n: Tensor, p: float) -> Tensor:
    """Exact ``X - K`` for ``X ~ Binomial(n, p)`` conditioned on ``X >= K = _TABLE_K``."""
    K = _TABLE_K
    cdf = _binomial_cdf(p).to(n.device)[n.long()]  # (M, _TABLE_N)
    lo = cdf[:, K - 1]
    v = lo + torch.rand(n.shape, dtype=torch.float64, device=n.device) * (1.0 - lo)
    return (v.unsqueeze(1) >= cdf).sum(1).to(n.dtype) - K


def _binomial_compiled(count: Tensor, p: float) -> Optional[Tensor]:
    """``Binomial(count, p)`` by a compiled inverse-CDF kernel (``0 < p <= 0.5``), or
    ``None`` when the counts are too large for the table (the caller then samples eagerly)."""
    n_max = int(count.max()) if count.numel() else 0
    if n_max == 0:
        return torch.zeros_like(count)
    if not _table_fits(p, n_max):
        # counts beyond the narrow table (large batches, small s): bisect the wide one
        if n_max >= _WIDE_TABLE_N:
            return None
        return _run_compiled(_bisect_kernel, count, _wide_thresholds(p, count.device),
                             _bisect_width(n_max))
    out = _run_compiled(_binomial_kernel, count, _thresholds(p, count.device))
    if out is None:
        return None
    K = _TABLE_K
    if n_max > K:  # rare draws beyond the table: resample them from the exact tail
        hit = ((out == K) & (count > K)).nonzero(as_tuple=True)
        if hit[0].numel():
            out[hit] += _tail_extra(count[hit], p)
    return out


def _binomial(count: Tensor, p: float, generator: Optional[torch.Generator] = None) -> Tensor:
    """Binomial samples with a scalar probability (``count`` is a floating tensor)."""
    if p >= 1.0:
        return count
    if p <= 0.0:
        return torch.zeros_like(count)
    if count.device.type == "cpu" and _compile_wanted(count.numel(), generator):
        out = _binomial_compiled(count, p) if p <= 0.5 else None
        if p > 0.5:
            rest = _binomial_compiled(count, 1.0 - p)
            out = None if rest is None else count - rest
        if out is not None:
            return out
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
    budget_mode: str = "soft",
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
            (forgetting) until it shrinks back (Abeyrathna et al., 2023). Under
            ``budget_mode="hard"`` the budget is also a hard limit on growth.
        include_count: ``(C,)`` included literals per clause (needed for the constraint).
        literal_active: optional ``(2F,)`` 0/1 mask; inactive literals receive no feedback
            (literal dropout).
        generator: optional RNG.
        budget_mode: ``"soft"`` (default) is the reference implementation's rule (tmu): the
            budget only gates Type Ia, so a clause may overshoot ``b`` and is then pruned by
            Type Ib until it shrinks back. ``"hard"`` also reverts new inclusions beyond the
            budget, so a clause never ends a call above ``b``, however many events were
            aggregated into it.

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
    room = torch.sub(N, state.to(torch.int32)).clamp_(min=0)
    inc2 = torch.minimum(n2, room.to(n2.dtype), out=n2)

    delta = inc.sub_(dec).add_(inc2)
    if literal_active is not None:
        delta.mul_(literal_active.to(delta.dtype).unsqueeze(0))
    # Add in int32 and clamp before casting back, so that narrow state dtypes (uint8, int16)
    # never see a negative or overflowing intermediate.
    # The new state is formed in the (floating) delta buffer and clamped before it is written
    # back, so narrow state dtypes (uint8, int16) never see a negative or overflowing value.
    # All values are small integers, exactly representable in the compute dtype.
    hard = max_included_literals is not None and budget_mode != "soft"
    was_included = state >= N if hard else None
    new = delta.add_(state).clamp_(0, 2 * N - 1)
    if hard:
        _cap_new_inclusions(new, was_included, N, int(max_included_literals), generator)
    state.copy_(new)
    return state


def sample_forget_events(
    counts: Tensor, n_literals: int, p: float, generator: Optional[torch.Generator] = None
) -> Tuple[Tensor, Tensor]:
    """Sparse exact draw of per-literal forgetting for row-broadcast event counts.

    Row ``c`` holds ``counts[c]`` Bernoulli(``p``) trials for each of its ``n_literals``
    literals (Type Ib feedback applies the same count to every literal of a clause). Every
    literal therefore needs a ``Binomial(counts[c], p)`` decrement. Instead of one draw per
    literal, the successes of each row's ``counts[c] * n_literals`` trials are generated by
    geometric gaps, so the work scales with the number of successes. Trial ``t`` of row ``c``
    belongs to literal ``t // counts[c]``.

    Args:
        counts: ``(C,)`` non-negative integer-valued counts (any numeric dtype).
        n_literals: literals per row ``L``.
        p: success probability, ``0 < p <= 0.5`` (the gaps are long only for small ``p``).
        generator: optional RNG.

    Returns:
        ``(pos, cnt)``: sorted unique flat positions ``c * L + l`` with at least one success
        and the number of successes at each (``int64``).
    """
    dev = counts.device
    L = int(n_literals)
    rows = (counts > 0).nonzero().squeeze(1)
    empty = torch.zeros(0, dtype=torch.long, device=dev)
    if rows.numel() == 0 or p <= 0.0:
        return empty, empty
    n = counts[rows].round().long()
    trials = n * L
    log_q = math.log1p(-float(p))
    mu = trials.double() * float(p)
    # Enough gaps per row that running out is rare; rows that do run out continue below.
    k = (mu + 6.0 * mu.sqrt() + 8.0).ceil().long()
    starts = torch.zeros_like(trials)  # first unexamined trial of each row
    pos_parts, todo = [], torch.arange(rows.numel(), device=dev)
    while todo.numel():
        kk = k[todo]
        row_of = torch.repeat_interleave(torch.arange(todo.numel(), device=dev), kk)
        u = torch.rand(row_of.numel(), device=dev, generator=generator, dtype=torch.float64)
        step = (torch.log(u) / log_q).floor_().long().add_(1)  # trials up to the next success
        csum = step.cumsum(0)
        seg_end = kk.cumsum(0) - 1
        seg_base = csum[seg_end] - kk.new_zeros(kk.shape).index_add_(0, row_of, step)
        t = csum - seg_base[row_of] - 1 + starts[todo][row_of]
        lim = trials[todo][row_of]
        ok = t < lim
        pos_parts.append(rows[todo][row_of][ok] * L + t[ok] // n[todo][row_of][ok])
        last = t[seg_end]
        more = last < trials[todo] - 1
        starts[todo] = last + 1
        todo = todo[more]
        if todo.numel():
            k[todo] = k[todo] * 2
    pos = pos_parts[0] if len(pos_parts) == 1 else torch.sort(torch.cat(pos_parts)).values
    if pos.numel() == 0:
        return empty, empty
    return torch.unique_consecutive(pos, return_counts=True)


def apply_feedback_rows(
    state: Tensor,
    n_true: Tensor,
    n_false: Tensor,
    n_ib: Tensor,
    n2: Tensor,
    rows: Tensor,
    *,
    n_states: int,
    s: float,
    boost_true_positive: bool = True,
    max_included_literals: Optional[int] = None,
    include_count: Optional[Tensor] = None,
    literal_active: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    budget_mode: str = "soft",
) -> Optional[Tuple[Tensor, Tensor]]:
    """:func:`apply_feedback` for count tensors that are zero outside ``rows``.

    Type Ia and Type II counts only exist for clauses that matched and were selected, a small
    fraction of all clauses. Those rows are updated as a dense slab; the Type Ib forgetting of
    every other clause is drawn sparsely with :func:`sample_forget_events`. The transitions
    follow the same distribution as :func:`apply_feedback`. The count tensors are read, not
    modified.

    Args:
        rows: ``(C,)`` bool mask of the clauses with Type Ia or Type II counts. The other
            arguments are as in :func:`apply_feedback`.

    Returns:
        ``(slab_rows, other_pos)``: the clause indices updated as a dense slab and the flat
        positions changed elsewhere, so that cached include masks can be refreshed
        incrementally. ``None`` when the dense path was used (refresh everything).
    """
    N = int(n_states)
    p_forget = 1.0 / float(s)
    if p_forget > 0.5:  # forgetting is not sparse; the dense update is as fast
        apply_feedback(state, n_true.clone(), n_false.clone(), n_ib.clone(), n2.clone(),
                       n_states=N, s=s, boost_true_positive=boost_true_positive,
                       max_included_literals=max_included_literals, include_count=include_count,
                       literal_active=literal_active, generator=generator,
                       budget_mode=budget_mode)
        return None
    p_mem = 1.0 if boost_true_positive else (float(s) - 1.0) / float(s)
    C, L = state.shape
    idx = rows.nonzero().squeeze(1)
    nt, nf, n2s = n_true[idx], n_false[idx], n2[idx]
    nib = n_ib.clone()
    if max_included_literals is not None and idx.numel():
        if include_count is None:
            include_count = (state >= N).sum(dim=1)
        over = include_count[idx] > int(max_included_literals)
        if bool(over.any()):
            # Every Type Ia event of an oversized clause becomes a Type Ib event.
            nib.index_add_(0, idx[over], (nt[over, 0] + nf[over, 0]).to(nib.dtype))
            keep = (~over).to(nt.dtype).unsqueeze(1)
            nt.mul_(keep)
            nf.mul_(keep)

    # Type Ib forgetting of the clauses outside the slab: one compiled dense pass over their
    # rows on the CPU, else an exact sparse draw of the forget events.
    other = None
    if state.device.type == "cpu" and _COMPILE["enabled"] and generator is None:
        ib_rows = ((nib > 0) & ~rows).nonzero().squeeze(1)
        if _compile_wanted(ib_rows.numel() * L):
            other = _forget_rows(state, ib_rows, nib[ib_rows], p_forget, literal_active, N)
    if other is None:
        pos, cnt = sample_forget_events(nib, L, p_forget, generator)
        if literal_active is not None and pos.numel():
            alive = literal_active.reshape(-1)[pos % L] > 0
            pos, cnt = pos[alive], cnt[alive]
        slab_of = torch.full((C,), -1, dtype=torch.long, device=state.device)
        slab_of[idx] = torch.arange(idx.numel(), device=state.device)
        s_i = slab_of[pos // L]
        in_slab = s_i >= 0

    if idx.numel():
        inc = _binomial(nt, p_mem, generator)
        if other is not None:  # the slab's Type Ib events are part of its dense draw
            dec = _binomial(nf + nib[idx].unsqueeze(1), p_forget, generator)
        else:
            dec = _binomial(nf, p_forget, generator)
            if bool(in_slab.any()):
                dec.view(-1).index_add_(0, s_i[in_slab] * L + pos[in_slab] % L,
                                        cnt[in_slab].to(dec.dtype))
        sub = state[idx].to(torch.int32)
        room = torch.sub(N, sub).clamp_(min=0)
        inc2 = torch.minimum(n2s, room.to(n2s.dtype))
        delta = inc.sub_(dec).add_(inc2)
        if literal_active is not None:
            delta.mul_(literal_active.to(delta.dtype).unsqueeze(0))
        was_included = sub >= N
        sub.add_(delta.to(torch.int32)).clamp_(0, 2 * N - 1)
        if max_included_literals is not None and budget_mode != "soft":
            _cap_new_inclusions(sub, was_included, N, int(max_included_literals), generator)
        state[idx] = sub.to(state.dtype)

    if other is not None:
        return idx, other
    other = pos[~in_slab]
    if other.numel():
        flat = state.view(-1)
        vals = flat[other].to(torch.int32).sub_(cnt[~in_slab].to(torch.int32)).clamp_(min=0)
        flat[other] = vals.to(state.dtype)
    return idx, other


def _forget_rows(
    state: Tensor, rows: Tensor, n: Tensor, p: float, literal_active: Optional[Tensor], N: int
) -> Optional[Tensor]:
    """Type Ib forgetting of whole clauses ``rows`` (``n[i]`` events for every literal of
    row ``i``) by the compiled kernel, in place. Returns the flat positions whose include bit
    changed, or ``None`` (nothing modified) if the compiled kernel cannot be used."""
    L = state.shape[1]
    empty = torch.zeros(0, dtype=torch.long, device=state.device)
    n_max = int(n.max()) if n.numel() else 0
    if n_max == 0:
        return empty
    active = None if literal_active is None else literal_active.reshape(1, -1).to(torch.float32)
    sub = state[rows]
    K = _TABLE_K
    if not _table_fits(p, n_max):
        # counts beyond the narrow table (large batches, small s): bisect the wide one, which
        # covers every count it holds, so there is no tail to resample
        if n_max >= _WIDE_TABLE_N:
            return None
        out = _run_compiled(_forget_rows_bisect_kernel, sub, n.unsqueeze(1),
                            _wide_thresholds(p, state.device), active, N, _bisect_width(n_max))
        K = _WIDE_TABLE_N
    else:
        out = _run_compiled(_forget_rows_kernel, sub, n.unsqueeze(1), _thresholds(p, state.device),
                            active, N)
    if out is None:
        return None
    new, crossed = out
    if n_max > K:  # rare draws beyond the table: resample them from the exact tail
        tail = (n > K).nonzero().squeeze(1)
        old_t, new_t = sub[tail].to(torch.int32), new[tail].to(torch.int32)
        hit = ((old_t - new_t) == K) & (new_t > 0)  # exactly K decrements, not clamped
        r, c = hit.nonzero(as_tuple=True)
        if r.numel():
            extra = _tail_extra(n[tail][r], p).to(torch.int32)
            new_t[r, c] = (new_t[r, c] - extra).clamp_(min=0)
            new[tail] = new_t.to(new.dtype)
            crossed[tail] = (old_t >= N) & (new_t < N)
    state[rows] = new
    r, c = crossed.nonzero(as_tuple=True)
    return rows[r] * L + c


def _single_example_kernel(
    state: Tensor, include: Tensor, lit: Tensor, lit_ii: Tensor, fire_i: Tensor,
    nofire_i: Tensor, fire_ii: Tensor, include_count: Tensor, active: Optional[Tensor], N: int,
    p_mem: float, p_forget: float, budget: int,
) -> Tensor:
    # lit / lit_ii: (1, 2F) for a flat model, (C, 2F) when every clause is updated from its
    # own literals (the random matching patch of a convolutional model)
    neg = 1.0 - lit
    fi = fire_i.unsqueeze(1)
    nib = nofire_i.unsqueeze(1)
    if budget >= 0:  # an oversized matching clause gets Type Ib instead of Type Ia
        over = (include_count > budget).unsqueeze(1).to(fi.dtype)
        nib = nib + fi * over
        fi = fi * (1.0 - over)
    old = state.to(torch.float32)
    inc = fi * lit
    if p_mem < 1.0:
        inc = inc * (torch.rand(state.shape, device=state.device) < p_mem).to(inc.dtype)
    # fire and no-fire are exclusive, so every count is 0 or 1 and the binomials are Bernoulli
    dec = (fi * neg + nib) * (torch.rand(state.shape, device=state.device) < p_forget).to(inc.dtype)
    inc2 = torch.minimum(fire_ii.unsqueeze(1) * (1.0 - lit_ii), (N - old).clamp(min=0))
    delta = inc - dec + inc2
    if active is not None:
        delta = delta * active
    new = (old + delta).clamp(0, 2 * N - 1)
    included = new >= N
    state.copy_(new.to(state.dtype))
    include.copy_(included.to(include.dtype))
    count = included.sum(dim=1).to(include_count.dtype)
    include_count.copy_(count)  # in place: CUDA graphs keep pointing at the same buffers
    return count


def commit_single_example(
    state: Tensor,
    include: Tensor,
    include_count: Tensor,
    literals: Tensor,
    fire_i: Tensor,
    nofire_i: Tensor,
    fire_ii: Tensor,
    *,
    n_states: int,
    s: float,
    boost_true_positive: bool = True,
    max_included_literals: Optional[int] = None,
    literal_active: Optional[Tensor] = None,
    literals_ii: Optional[Tensor] = None,
) -> Optional[Tensor]:
    """The feedback of **one** example, applied by a single fused ``torch.compile`` kernel.

    Equivalent in distribution to accumulating the example's counts (the outer products of
    ``fire_i`` / ``nofire_i`` / ``fire_ii`` with the literals) and calling
    :func:`apply_feedback` with the soft budget, but without materialising any ``(C, 2F)``
    count tensor and without host synchronisation. Used by sequential mode on CUDA.

    Args:
        state: ``(C, 2F)`` TA states, updated in place.
        include: ``(C, 2F)`` cached include mask, updated in place.
        include_count: ``(C,)`` included literals per clause *before* the update; updated in
            place.
        literals: ``(2F,)`` 0/1 literals of the example, or ``(C, 2F)`` literals per clause
            (a convolutional model updates each clause from its own random matching patch).
        fire_i: ``(C,)`` 0/1 mask of the clauses that get Type Ia feedback.
        nofire_i: ``(C,)`` 0/1 mask of the clauses that get Type Ib feedback.
        fire_ii: ``(C,)`` 0/1 mask of the clauses that get Type II feedback.
        n_states: as in :func:`apply_feedback`.
        s: as in :func:`apply_feedback`.
        boost_true_positive: as in :func:`apply_feedback`.
        max_included_literals: as in :func:`apply_feedback` (soft budget).
        literal_active: as in :func:`apply_feedback`.
        literals_ii: literals of the Type II events, shaped like ``literals``; defaults to
            ``literals`` (a convolutional model draws the patch of each kind independently).

    Returns:
        The new ``(C,)`` include counts, or ``None`` when the compiled kernel is unavailable
        (nothing has been modified; the caller falls back to :func:`apply_feedback_rows`).
    """
    lit = literals if literals.dim() == 2 else literals.unsqueeze(0)
    lit_ii = lit if literals_ii is None else (
        literals_ii if literals_ii.dim() == 2 else literals_ii.unsqueeze(0))
    N = int(n_states)
    p_mem = 1.0 if boost_true_positive else (float(s) - 1.0) / float(s)
    budget = -1 if max_included_literals is None else int(max_included_literals)
    active = None if literal_active is None else literal_active.reshape(1, -1).to(torch.float32)
    return _run_compiled(_single_example_kernel, state, include, lit, lit_ii, fire_i, nofire_i,
                         fire_ii, include_count, active, N, p_mem, 1.0 / float(s), budget)


def _dense_commit_kernel(
    state: Tensor, include: Tensor, include_count: Tensor, n_true: Tensor, n_false: Tensor,
    n_ib: Tensor, n2: Tensor, thr_mem: Optional[Tensor], thr_forget: Optional[Tensor],
    active: Optional[Tensor], N: int, mem_flip: bool, forget_flip: bool, budget: int, width: int,
) -> Tensor:
    nt, nf, nib = n_true, n_false, n_ib.unsqueeze(1)
    if budget >= 0:  # an oversized matching clause gets Type Ib instead of Type Ia
        over = (include_count > budget).unsqueeze(1).to(nt.dtype)
        nib = nib + (nt[:, :1] + nf[:, :1]) * over
        nt = nt * (1.0 - over)
        nf = nf * (1.0 - over)
    inc = nt if thr_mem is None else _bisect_binomial(nt, thr_mem, mem_flip, width)
    nf = nf + nib
    dec = nf if thr_forget is None else _bisect_binomial(nf, thr_forget, forget_flip, width)
    old = state.to(torch.float32)
    inc2 = torch.minimum(n2, (N - old).clamp(min=0))
    delta = inc - dec + inc2
    if active is not None:
        delta = delta * active
    new = (old + delta).clamp(0, 2 * N - 1)
    included = new >= N
    state.copy_(new.to(state.dtype))
    include.copy_(included.to(include.dtype))
    n_true.zero_()
    n_false.zero_()
    n2.zero_()
    # include_count is not summed here: fused in, inductor schedules the whole kernel as a row
    # reduction at half the speed (G16); the caller sums the new state separately


def commit_dense(
    state: Tensor,
    include: Tensor,
    include_count: Tensor,
    n_true: Tensor,
    n_false: Tensor,
    n_ib: Tensor,
    n2: Tensor,
    *,
    n_states: int,
    s: float,
    boost_true_positive: bool = True,
    max_included_literals: Optional[int] = None,
    literal_active: Optional[Tensor] = None,
    max_events: Optional[int] = None,
) -> Optional[Tensor]:
    """:func:`apply_feedback` (soft budget) as one fused ``torch.compile`` kernel that also
    refreshes the include cache.

    The binomials are drawn inside the kernel by inverse-CDF lookup, so the update reads the
    counts and the state once and writes the state, the include mask and zeroed counts once,
    instead of the ~20 full passes of the eager update. Used by batched updates on CUDA.

    Args:
        state: ``(C, 2F)`` TA states, updated in place.
        include: ``(C, 2F)`` cached include mask, updated in place.
        include_count: ``(C,)`` included literals per clause *before* the update; updated in
            place.
        n_true: as in :func:`apply_feedback`; zeroed on return (as are ``n_false`` and
            ``n2``; ``n_ib`` is left as it was).
        n_false: as in :func:`apply_feedback`.
        n_ib: as in :func:`apply_feedback`.
        n2: as in :func:`apply_feedback`.
        n_states: as in :func:`apply_feedback`.
        s: as in :func:`apply_feedback`.
        boost_true_positive: as in :func:`apply_feedback`.
        max_included_literals: as in :func:`apply_feedback` (soft budget).
        literal_active: as in :func:`apply_feedback`.
        max_events: an upper bound on any count (e.g. the examples in the update). With it the
            table width is chosen without reading the counts back to the host.

    Returns:
        The new ``(C,)`` include counts (``include_count`` itself), or ``None`` (nothing
        modified) when the kernel cannot be used: compilation disabled or failed, or more than
        ``_WIDE_TABLE_N - 1`` events aggregated into one automaton. The caller then takes the
        eager path.
    """
    p_mem = 1.0 if boost_true_positive else (float(s) - 1.0) / float(s)
    if (not _compile_wanted(state.numel()) or _dense_commit_kernel in _FAILED
            or p_mem <= 0.0):
        return None
    # Type Ia events per clause are n_true + n_false of any one literal; with Type Ib they
    # bound every count that is sampled.
    if max_events is not None and int(max_events) < _WIDE_TABLE_N:
        most = int(max_events)
    else:  # no bound given (or a large one): read the actual largest count back
        events = n_true[:, 0] + n_false[:, 0] + n_ib
        most = int(events.max())
    if most >= _WIDE_TABLE_N:
        return None
    width = _bisect_width(most)
    N = int(n_states)
    dev = state.device

    def table(p: float) -> Tuple[Optional[Tensor], bool]:
        if p >= 1.0:
            return None, False
        q = min(p, 1.0 - p)
        return _wide_thresholds(q, dev), p > 0.5

    thr_mem, mem_flip = table(p_mem)
    thr_forget, forget_flip = table(1.0 / float(s))
    budget = -1 if max_included_literals is None else int(max_included_literals)
    active = None if literal_active is None else literal_active.reshape(1, -1).to(torch.float32)
    done = _run_compiled(_dense_commit_kernel, state, include, include_count, n_true, n_false,
                         n_ib, n2, thr_mem, thr_forget, active, N, mem_flip, forget_flip, budget,
                         width)
    if done is None and _dense_commit_kernel in _FAILED:
        return None
    torch.sum(state >= N, dim=1, out=include_count)
    return include_count

def _cap_new_inclusions(
    state: Tensor,
    was_included: Tensor,
    N: int,
    budget: int,
    generator: Optional[torch.Generator] = None,
) -> None:
    """Hard clause-size budget: undo newly crossed include boundaries beyond ``budget``.

    One commit aggregates many Type Ia / Type II events, so a clause that was under budget at
    the start of the commit can cross it many times over. Per clause, at most
    ``budget - (literals that were included and still are)`` of the newly included literals
    are kept (a uniformly random subset); the rest are put back just below the boundary.
    Forgetting is untouched, so a clause that is already oversized can only shrink.
    """
    included = state >= N
    newly = included & ~was_included
    allowed = (budget - (included & was_included).sum(dim=1)).clamp_(min=0)
    over = newly.sum(dim=1) > allowed
    if not bool(over.any()):
        return
    rows = over.nonzero().squeeze(1)
    cand = newly[rows]
    # Rank the candidates of each row in a random order; non-candidates sort last.
    keys = torch.rand(cand.shape, device=state.device, generator=generator)
    keys.masked_fill_(~cand, 2.0)
    rank = keys.argsort(dim=1).argsort(dim=1)
    revert = cand & (rank >= allowed[rows].unsqueeze(1))
    sub = state[rows]
    sub.masked_fill_(revert, N - 1)
    state[rows] = sub


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


_DYNAMIC.add(_dense_commit_kernel)  # model sizes differ between models; compile shape-generic
