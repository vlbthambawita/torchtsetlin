"""Base class shared by all Tsetlin machine modules."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .. import functional as F
from ..utils import as_bool_tensor, chunk_indices

__all__ = ["TsetlinMachineBase", "FeedbackAccumulator"]


class FeedbackAccumulator:
    """Running sums of feedback events for one :meth:`TsetlinMachineBase.update` call."""

    def __init__(self, n_clauses: int, n_literals: int, device: torch.device, dtype: torch.dtype):
        self.n_true = torch.zeros(n_clauses, n_literals, device=device, dtype=dtype)
        self.n_false = torch.zeros(n_clauses, n_literals, device=device, dtype=dtype)
        self.n_ib = torch.zeros(n_clauses, device=device, dtype=dtype)
        self.n2 = torch.zeros(n_clauses, n_literals, device=device, dtype=dtype)
        self.extra: Dict[str, Tensor] = {}


class TsetlinMachineBase(nn.Module):
    """Common machinery for Tsetlin machines: automata state, clause evaluation, dropout,
    and the batched learning loop.

    The learnable state lives in the integer buffer ``ta_state`` of shape
    ``(n_clauses_total, 2 * n_features)``. It moves with ``.to(device)``, is saved in
    ``state_dict()`` and can be inspected directly.

    Subclasses implement the *output layer* (how clause outputs become votes) and the
    *feedback policy* (which clauses receive Type I / Type II feedback for a labelled
    example) through :meth:`_votes`, :meth:`_select_feedback` and the weight hooks.
    Convolutional variants additionally override :meth:`_encode`, :meth:`_evaluate` and
    :meth:`_feedback_counts`.

    Args:
        n_features: number of Boolean input features ``F``. ``None`` defers allocation until
            the first batch is seen (lazy initialisation, like ``torch.nn.LazyLinear``).
        n_clauses_total: total number of clauses ``C`` (all classes / outputs together).
        n_states: automaton states per action ``N``. A literal is included when its state
            is ``>= N``. Larger values give a deeper memory (slower forgetting).
        s: specificity. Memorize probability ``(s-1)/s``, forget probability ``1/s``.
            Larger ``s`` -> more literals per clause (more specific patterns).
        boost_true_positive: memorize True literals of matching clauses with probability 1
            instead of ``(s-1)/s`` (the default in the reference implementations).
        max_included_literals: optional cap on the number of included literals per clause
            (clause size constraint). Oversized clauses only receive forgetting feedback
            until they shrink back.
        drop_clause_p: probability of dropping a clause during learning (a Tsetlin machine
            analogue of dropout). The mask is resampled on every :meth:`update` call, or once
            per epoch if ``drop_granularity="epoch"`` (call :meth:`resample_dropout`).
        drop_literal_p: probability of freezing a literal (no feedback) per resample.
        drop_granularity: ``"batch"`` or ``"epoch"``.
        init: ``"boundary"`` starts every automaton just below the include boundary
            (``N-1``); ``"random"`` starts randomly at ``N-1`` or ``N``.
        state_dtype: integer dtype of the automata states.
        compute_dtype: floating dtype used for matmul-based clause evaluation.
        max_chunk_elements: memory budget (in tensor elements) for the largest intermediate
            tensor of an update; batches are processed in chunks that respect it.
        feedback_mode: ``"batch"`` accumulates the feedback of all examples in a batch and
            applies it once (fast, mini-batch style). ``"sequential"`` applies the feedback
            example by example, re-evaluating the clauses in between — the exact classical
            algorithm, at the cost of one small step per example.
    """

    ta_state: Tensor
    include: Tensor
    include_count: Tensor

    def __init__(
        self,
        n_features: Optional[int],
        n_clauses_total: int,
        *,
        n_states: int = 128,
        s: float = 10.0,
        boost_true_positive: bool = True,
        max_included_literals: Optional[int] = None,
        drop_clause_p: float = 0.0,
        drop_literal_p: float = 0.0,
        drop_granularity: str = "batch",
        init: str = "boundary",
        state_dtype: torch.dtype = torch.int32,
        compute_dtype: torch.dtype = torch.float32,
        max_chunk_elements: int = 2**27,
        feedback_mode: str = "batch",
    ) -> None:
        super().__init__()
        if n_clauses_total <= 0:
            raise ValueError("n_clauses_total must be positive")
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if s < 1.0:
            raise ValueError("s (specificity) must be >= 1")
        if not (0.0 <= drop_clause_p < 1.0) or not (0.0 <= drop_literal_p < 1.0):
            raise ValueError("dropout probabilities must lie in [0, 1)")
        if init not in ("boundary", "random"):
            raise ValueError("init must be 'boundary' or 'random'")
        if drop_granularity not in ("batch", "epoch"):
            raise ValueError("drop_granularity must be 'batch' or 'epoch'")
        if feedback_mode not in ("batch", "sequential"):
            raise ValueError("feedback_mode must be 'batch' or 'sequential'")
        self.n_features: Optional[int] = None
        self.n_clauses_total = int(n_clauses_total)
        self.n_states = int(n_states)
        self.s = float(s)
        self.boost_true_positive = bool(boost_true_positive)
        self.max_included_literals = max_included_literals
        self.drop_clause_p = float(drop_clause_p)
        self.drop_literal_p = float(drop_literal_p)
        self.drop_granularity = drop_granularity
        self.init = init
        self.state_dtype = state_dtype
        self.compute_dtype = compute_dtype
        self.max_chunk_elements = int(max_chunk_elements)
        self.feedback_mode = feedback_mode
        self.feature_names: Optional[List[str]] = None
        self._clause_active: Optional[Tensor] = None
        self._literal_active: Optional[Tensor] = None

        # Buffers (empty until the feature count is known).
        self.register_buffer("ta_state", torch.empty(0, dtype=state_dtype))
        self.register_buffer("include", torch.empty(0, dtype=compute_dtype), persistent=False)
        self.register_buffer("include_count", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_load_state_dict_post_hook(_refresh_after_load)
        if n_features is not None:
            self._initialize_state(int(n_features))

    # ------------------------------------------------------------------ properties
    @property
    def device(self) -> torch.device:
        return self.ta_state.device

    @property
    def is_initialized(self) -> bool:
        return self.n_features is not None and self.ta_state.numel() > 0

    @property
    def n_literals(self) -> int:
        self._check_initialized()
        return 2 * int(self.n_features)  # type: ignore[arg-type]

    def _check_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "The Tsetlin machine has not seen any data yet (lazy initialisation). "
                "Pass n_features to the constructor or call update()/forward() first."
            )

    # ------------------------------------------------------------------ state init
    def _initialize_state(self, n_features: int, device: Optional[torch.device] = None) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        device = device if device is not None else self.ta_state.device
        self.n_features = int(n_features)
        shape = (self.n_clauses_total, 2 * self.n_features)
        N = self.n_states
        if self.init == "random":
            state = torch.randint(N - 1, N + 1, shape, device=device, dtype=self.state_dtype)
        else:
            state = torch.full(shape, N - 1, device=device, dtype=self.state_dtype)
        self.ta_state = state
        self._refresh_include()
        self._on_initialized()

    def _on_initialized(self) -> None:
        """Hook for subclasses that keep extra per-feature state."""

    def reset_parameters(self) -> None:
        """Re-initialise all automata (keeps the shape)."""
        if self.is_initialized:
            self._initialize_state(int(self.n_features), self.ta_state.device)  # type: ignore[arg-type]

    def _refresh_include(self) -> None:
        """Recompute the cached include mask after the automata changed."""
        inc = self.ta_state >= self.n_states
        self.include = inc.to(self.compute_dtype)
        self.include_count = inc.sum(dim=1)

    # ------------------------------------------------------------------ input handling
    def _coerce_input(self, x) -> Tensor:
        """Boolean tensor on the model device with shape ``(B, ...)``."""
        return as_bool_tensor(x, device=self.ta_state.device)

    def _infer_n_features(self, xb: Tensor) -> int:
        return int(xb[0].numel())

    def _encode(self, xb: Tensor) -> Tensor:
        """Boolean input batch ``(B, ...)`` -> literals ``(B, 2F)`` (flat models)."""
        xb = xb.reshape(xb.shape[0], -1)
        if xb.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} Boolean features per example, got {xb.shape[1]}."
            )
        return F.to_literals(xb, dtype=self.compute_dtype)

    def _prepare(self, x) -> Tensor:
        """Coerce ``x`` and lazily initialise the state. Returns the Boolean batch."""
        xb = self._coerce_input(x)
        if xb.dim() == 1:
            xb = xb.unsqueeze(0)
        if not self.is_initialized:
            self._initialize_state(self._infer_n_features(xb), xb.device)
        return xb

    def _patches_per_example(self, xb: Tensor) -> int:
        return 1

    def _chunk_elements_per_example(self, xb: Tensor) -> int:
        """Largest per-example intermediate of an update, in elements (flat models: the
        ``(B, C)`` clause matrix and the ``(B, 2F)`` literals)."""
        return 4 * (self.n_clauses_total + self.n_literals)

    def _chunk_size(self, xb: Tensor) -> int:
        """Examples per chunk so that the intermediates fit ``max_chunk_elements``."""
        per_example = max(1, self._chunk_elements_per_example(xb))
        return max(1, min(xb.shape[0], self.max_chunk_elements // per_example))

    # ------------------------------------------------------------------ clause evaluation
    def _evaluate(self, literals: Tensor, empty_value: bool) -> Tuple[Tensor, Any]:
        """Clause outputs ``(B, C)`` plus an opaque context for feedback (flat: ``None``)."""
        out = F.clause_outputs(literals, self.include, self.include_count, empty_value)
        return out, None

    def clause_outputs(self, literals: Tensor, empty_value: Optional[bool] = None) -> Tensor:
        """Evaluate every clause on encoded literals -> ``(B, C)`` bool.

        ``empty_value`` defaults to ``self.training`` (empty clauses are True while learning
        and False when predicting).
        """
        if empty_value is None:
            empty_value = self.training
        return self._evaluate(literals, empty_value)[0]

    def evaluate_clauses(self, x) -> Tensor:
        """Clause outputs for raw Boolean input ``x`` (prediction semantics) -> ``(B, C)``."""
        with torch.no_grad():
            xb = self._prepare(x)
            outs = []
            for sl in chunk_indices(xb.shape[0], self._chunk_size(xb)):
                outs.append(self._evaluate(self._encode(xb[sl]), empty_value=False)[0])
            return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------ output layer hooks
    def _votes(self, clause_out: Tensor, clamp: bool = True) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:  # pragma: no cover
        """Return ``(type_i, type_ii, aux)`` masks of shape ``(B, C)`` for this chunk."""
        raise NotImplementedError

    def _accumulate_weights(
        self, acc: FeedbackAccumulator, aux: Any, clause_out: Tensor, sel_fire_i: Tensor, sel_fire_ii: Tensor
    ) -> None:
        """Optional: accumulate clause weight deltas (default: no weights)."""

    def _commit_weights(self, acc: FeedbackAccumulator) -> None:
        """Optional: apply accumulated weight deltas."""

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x) -> Tensor:
        """Vote sums ``(B, n_outputs)``. Uses prediction semantics for empty clauses when the
        module is in ``eval()`` mode."""
        with torch.no_grad():
            xb = self._prepare(x)
            votes = []
            for sl in chunk_indices(xb.shape[0], self._chunk_size(xb)):
                out, _ = self._evaluate(self._encode(xb[sl]), empty_value=self.training)
                votes.append(self._votes(out))
            return torch.cat(votes, dim=0)

    # ------------------------------------------------------------------ learning loop
    def update(self, x, y, sequential: Optional[bool] = None) -> Tensor:
        """One learning step on a batch — the Tsetlin machine analogue of
        ``loss.backward(); optimizer.step()``.

        In ``"batch"`` mode every example of the batch is evaluated against the current
        automata; the resulting Type I / Type II feedback events are accumulated and applied
        once. With a batch size of 1 this is exactly the classical sequential algorithm;
        larger batches trade a little fidelity for throughput (like mini-batch SGD). In
        ``"sequential"`` mode (``feedback_mode`` or the ``sequential`` argument) the examples
        are processed one by one with the automata updated in between.

        Args:
            x (Tensor | ndarray | list): Boolean input batch (any Boolean-like dtype).
            y (Tensor | ndarray | list): targets (class indices, multi-label matrix or
                regression values, depending on the model).
            sequential: override ``self.feedback_mode`` for this call.

        Returns:
            The vote sums computed **before** the update (``(B, n_outputs)``) so that batch
            metrics can be logged without a second forward pass.
        """
        with torch.no_grad():
            xb = self._prepare(x)
            B = xb.shape[0]
            y_t = self._coerce_targets(y, B, xb.device)
            if self.drop_granularity == "batch":
                self.resample_dropout()
            seq = self.feedback_mode == "sequential" if sequential is None else bool(sequential)
            votes_all = []
            if seq:
                for i in range(B):
                    acc = self._new_accumulator()
                    votes_all.append(self._accumulate_chunk(xb[i : i + 1], y_t[i : i + 1], acc))
                    self._commit(acc)
            else:
                acc = self._new_accumulator()
                for sl in chunk_indices(B, self._chunk_size(xb)):
                    votes_all.append(self._accumulate_chunk(xb[sl], y_t[sl], acc))
                self._commit(acc)
            return torch.cat(votes_all, dim=0)

    def _new_accumulator(self) -> FeedbackAccumulator:
        return FeedbackAccumulator(
            self.n_clauses_total, self.n_literals, self.ta_state.device, self.compute_dtype
        )

    def _accumulate_chunk(self, xb: Tensor, y_t: Tensor, acc: FeedbackAccumulator) -> Tensor:
        """Evaluate a chunk, decide feedback per (example, clause) and accumulate counts.
        Returns the (pre-update) vote sums of the chunk."""
        return self._accumulate_literals(self._encode(xb), y_t, acc)

    def _accumulate_literals(
        self, literals: Tensor, y_t: Tensor, acc: FeedbackAccumulator
    ) -> Tensor:
        """Same as :meth:`_accumulate_chunk` for already-encoded literals.

        Split out so that models which encode once and then learn from the result in several
        commits (see :mod:`torchtsetlin.models.segmentation`) do not have to re-encode.
        """
        clause_active = self._clause_active
        clause_out, ctx = self._evaluate(literals, empty_value=True)
        if clause_active is not None:
            clause_out = clause_out & clause_active.unsqueeze(0)
        votes = self._votes(clause_out)
        type_i, type_ii, aux = self._select_feedback(votes, y_t, clause_out)
        if clause_active is not None:
            type_i = type_i & clause_active.unsqueeze(0)
            type_ii = type_ii & clause_active.unsqueeze(0)
        cd = self.compute_dtype
        sel_fire_i = (type_i & clause_out).to(cd)
        sel_nofire_i = (type_i & ~clause_out).to(cd)
        sel_fire_ii = (type_ii & clause_out).to(cd)
        self._feedback_counts(literals, ctx, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)
        self._accumulate_weights(acc, aux, clause_out, sel_fire_i, sel_fire_ii)
        return votes

    def _commit(self, acc: FeedbackAccumulator) -> None:
        """Apply accumulated feedback to the automata (and weights) and refresh caches."""
        F.apply_feedback(
            self.ta_state,
            acc.n_true,
            acc.n_false,
            acc.n_ib,
            acc.n2,
            n_states=self.n_states,
            s=self.s,
            boost_true_positive=self.boost_true_positive,
            max_included_literals=self.max_included_literals,
            include_count=self.include_count,
            literal_active=self._literal_active,
        )
        self._commit_weights(acc)
        self._refresh_include()

    def _feedback_counts(
        self,
        literals: Tensor,
        ctx: Any,
        sel_fire_i: Tensor,
        sel_nofire_i: Tensor,
        sel_fire_ii: Tensor,
        acc: FeedbackAccumulator,
    ) -> None:
        """Accumulate feedback event counts for a chunk (flat models: dense matmuls)."""
        sel_fire_t = sel_fire_i.transpose(0, 1)  # (C, B)
        acc.n_true.addmm_(sel_fire_t, literals)
        neg_lit = 1.0 - literals
        acc.n_false.addmm_(sel_fire_t, neg_lit)
        acc.n_ib += sel_nofire_i.sum(dim=0)
        acc.n2.addmm_(sel_fire_ii.transpose(0, 1), neg_lit)

    # ------------------------------------------------------------------ dropout masks
    def resample_dropout(self) -> None:
        """Draw new clause / literal dropout masks (called automatically per update when
        ``drop_granularity="batch"``; call it once per epoch otherwise)."""
        dev = self.ta_state.device
        if self.drop_clause_p > 0.0:
            self._clause_active = torch.rand(self.n_clauses_total, device=dev) >= self.drop_clause_p
        else:
            self._clause_active = None
        if self.drop_literal_p > 0.0 and self.is_initialized:
            keep = torch.rand(self.n_literals, device=dev) >= self.drop_literal_p
            self._literal_active = keep.to(self.compute_dtype)
        else:
            self._literal_active = None

    # ------------------------------------------------------------------ introspection
    def included_mask(self) -> Tensor:
        """Boolean ``(C, 2F)`` mask of included literals (first ``F`` columns = positive
        literals ``x_k``, last ``F`` columns = negated literals ``NOT x_k``)."""
        self._check_initialized()
        return self.ta_state >= self.n_states

    def clause_literals(self, clause: int) -> Dict[str, List[int]]:
        """Indices of the included literals of one clause: ``{"positive": [...], "negated": [...]}``."""
        mask = self.included_mask()[clause]
        Fn = int(self.n_features)  # type: ignore[arg-type]
        pos = torch.nonzero(mask[:Fn], as_tuple=False).flatten().tolist()
        neg = torch.nonzero(mask[Fn:], as_tuple=False).flatten().tolist()
        return {"positive": pos, "negated": neg}

    def default_feature_names(self) -> List[str]:
        Fn = int(self.n_features)  # type: ignore[arg-type]
        return [f"x{i}" for i in range(Fn)]

    def literal_names(self, feature_names: Optional[Sequence[str]] = None) -> List[str]:
        """Names of all ``2F`` literals, e.g. ``["x0", "x1", "NOT x0", "NOT x1"]``."""
        Fn = int(self.n_features)  # type: ignore[arg-type]
        names = list(feature_names or self.feature_names or self.default_feature_names())
        if len(names) != Fn:
            raise ValueError(f"Expected {Fn} feature names, got {len(names)}")
        return names + [f"NOT {n}" for n in names]

    def clause_expression(
        self, clause: int, feature_names: Optional[Sequence[str]] = None, joiner: str = " AND "
    ) -> str:
        """Human readable conjunction for one clause (``"x1 AND NOT x3"``; ``"TRUE"`` if empty)."""
        lits = self.literal_names(feature_names)
        mask = self.included_mask()[clause]
        idx = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not idx:
            return "TRUE"
        return joiner.join(lits[i] for i in idx)

    def literal_frequency(self) -> Tensor:
        """How many clauses include each literal -> ``(2F,)`` long tensor."""
        return self.included_mask().sum(dim=0)

    def state_summary(self) -> Dict[str, float]:
        """Quick statistics about the automata (useful for logging)."""
        self._check_initialized()
        counts = self.include_count.to(torch.float32)
        return {
            "included_literals_mean": float(counts.mean()),
            "included_literals_max": float(counts.max()),
            "empty_clauses": int((counts == 0).sum()),
            "n_clauses": int(self.n_clauses_total),
            "n_literals": int(self.n_literals),
        }

    def extra_repr(self) -> str:
        nf = self.n_features if self.n_features is not None else "lazy"
        parts = [
            f"n_features={nf}",
            f"n_clauses_total={self.n_clauses_total}",
            f"n_states={self.n_states}",
            f"s={self.s}",
        ]
        if self.max_included_literals is not None:
            parts.append(f"max_included_literals={self.max_included_literals}")
        if self.drop_clause_p:
            parts.append(f"drop_clause_p={self.drop_clause_p}")
        if self.drop_literal_p:
            parts.append(f"drop_literal_p={self.drop_literal_p}")
        if self.feedback_mode != "batch":
            parts.append(f"feedback_mode={self.feedback_mode!r}")
        return ", ".join(parts)

    # ------------------------------------------------------------------ (de)serialisation
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[override]
        key = prefix + "ta_state"
        if key in state_dict and not self.is_initialized:
            loaded = state_dict[key]
            if loaded.dim() == 2 and loaded.numel() > 0:
                self._initialize_state(loaded.shape[1] // 2, self.ta_state.device)
        for name in list(self._buffers.keys()):
            k = prefix + name
            buf = self._buffers[name]
            if k in state_dict and buf is not None and buf.shape != state_dict[k].shape:
                self._buffers[name] = torch.empty(
                    state_dict[k].shape, dtype=buf.dtype, device=buf.device
                )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        params = list(super().parameters(recurse))
        if not params:
            warnings.warn(
                "Tsetlin machines have no gradient-trained parameters; the learnable state "
                "is stored in buffers (ta_state, weights). Use model.update(x, y) to learn.",
                stacklevel=2,
            )
        return iter(params)


def _refresh_after_load(module: TsetlinMachineBase, incompatible_keys) -> None:
    if module.is_initialized:
        module._refresh_include()
