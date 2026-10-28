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
        #: clauses with Type Ia or Type II counts (all other rows of the counts are zero)
        self.rows = torch.zeros(n_clauses, dtype=torch.bool, device=device)
        #: commit with the row-sparse update instead of the dense one
        self.sparse = device.type == "cpu"
        #: record a single example as factors for :func:`F.commit_single_example` (see
        #: :meth:`TsetlinMachineBase._feedback_counts`) instead of writing the counts
        self.fuse = False
        #: ``(literals, literals_ii, fire_i, nofire_i, fire_ii)`` of that example
        self.single: Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]] = None
        self.extra: Dict[str, Tensor] = {}

    def matches(self, n_clauses: int, n_literals: int, device: torch.device, dtype: torch.dtype) -> bool:
        return (tuple(self.n_true.shape) == (n_clauses, n_literals) and self.n_true.device == device
                and self.n_true.dtype == dtype)

    def reset(self, counts_clear: bool = False) -> None:
        """Zero the counts for reuse; only the rows that can be non-zero are touched.
        ``counts_clear``: the commit already zeroed ``n_true``, ``n_false`` and ``n2``."""
        if self.single is not None:  # the counts were never written
            self.single = None
        elif counts_clear:
            pass
        elif not self.sparse:
            for t in (self.n_true, self.n_false, self.n2):
                t.zero_()
        else:
            idx = self.rows.nonzero().squeeze(1)
            if idx.numel():
                for t in (self.n_true, self.n_false, self.n2):
                    t.index_fill_(0, idx, 0.0)
        self.n_ib.zero_()
        self.rows.zero_()
        self.extra = {}


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
        max_included_literals: optional budget on the number of included literals per clause
            (clause size constraint). A matching clause above the budget receives only
            forgetting (Type Ib) feedback until it shrinks back (see ``budget_mode``).
            Clause weights are still updated for such a clause, as in tmu (pyTsetlinMachine
            skips the weight increment of an over-budget clause).
        drop_clause_p: probability of dropping a clause during learning (a Tsetlin machine
            analogue of dropout). The mask is resampled on every :meth:`update` call, or once
            per epoch if ``drop_granularity="epoch"`` (call :meth:`resample_dropout`).
        drop_literal_p: probability of freezing a literal (no feedback) per resample.
        drop_granularity: ``"batch"`` or ``"epoch"``.
        init: ``"boundary"`` starts every automaton just below the include boundary
            (``N-1``); ``"random"`` starts randomly at ``N-1`` or ``N``.
        clamp_prediction_votes: clamp vote sums to ``[-T, T]`` in ``forward()`` / ``predict()``
            (pyTsetlinMachine's rule, under which saturated classes tie and the lowest index
            wins). ``None`` keeps the model's default (unclamped for classifiers, as in tmu).
        state_dtype: integer dtype of the automata states. ``None`` (default) picks the
            smallest safe one: ``uint8`` when ``2 * n_states - 1 <= 255`` (8 bits per automaton,
            as in the reference implementations), else ``int16`` or ``int32``. Cast before
            doing arithmetic on ``ta_state`` yourself (``model.ta_state.long() - N``).
        compute_dtype: floating dtype of the literals and feedback counts. On CUDA, clauses
            are evaluated in ``float16`` regardless: the evaluation only tests whether a count
            of violated literals is zero, which half precision decides exactly (a sum of
            non-negative integers never rounds to zero).
        max_chunk_elements: memory budget (in tensor elements) for the largest intermediate
            tensor of an update; batches are processed in chunks that respect it.
        feedback_mode: ``"batch"`` accumulates the feedback of all examples in a batch and
            applies it once (fast, mini-batch style). ``"sequential"`` applies the feedback
            example by example, re-evaluating the clauses in between — the exact classical
            algorithm, at the cost of one small step per example.
        budget_mode: how ``max_included_literals`` is enforced. ``"soft"`` (default) is the
            reference implementation's rule (tmu): the budget only gates Type Ia, so a clause can
            overshoot it and is then pruned by Type Ib, which keeps clause contents changing.
            ``"hard"`` (the 0.3.x default) also reverts every inclusion beyond the budget. A
            clause then never exceeds it, so the Type Ib switch never fires and clauses freeze
            at exactly the budget, which cost 10-11 pp on CIFAR-10 colour-thermometer models.
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
        state_dtype: Optional[torch.dtype] = None,
        compute_dtype: torch.dtype = torch.float32,
        max_chunk_elements: int = 2**27,
        feedback_mode: str = "batch",
        budget_mode: str = "soft",
        clamp_prediction_votes: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if clamp_prediction_votes is not None:
            # instance override of the class default (pyTsetlinMachine / PyTsetlinMachineCUDA
            # take the argmax over votes clamped to [-T, T]; tmu and green-tsetlin do not)
            self.clamp_prediction_votes = bool(clamp_prediction_votes)
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
        if budget_mode not in ("hard", "soft"):
            raise ValueError("budget_mode must be 'hard' or 'soft'")
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
        if state_dtype is None:
            # 8 bits per automaton, as in the reference implementations, when 2N - 1 fits.
            top = 2 * int(n_states) - 1
            state_dtype = torch.uint8 if top <= 255 else torch.int16 if top <= 32767 else torch.int32
        self.state_dtype = state_dtype
        self.compute_dtype = compute_dtype
        self.max_chunk_elements = int(max_chunk_elements)
        self.feedback_mode = feedback_mode
        self.budget_mode = budget_mode
        self.feature_names: Optional[List[str]] = None
        self._clause_active: Optional[Tensor] = None
        self._literal_active: Optional[Tensor] = None

        # Buffers (empty until the feature count is known).
        self.register_buffer("ta_state", torch.empty(0, dtype=self.state_dtype))
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

    def _include_dtype(self) -> torch.dtype:
        """dtype of the cached include mask, the operand of clause evaluation: ``float16`` on
        CUDA (exact for the zero test, and the matmul runs on tensor cores), else
        ``compute_dtype``."""
        return torch.float16 if self.ta_state.device.type == "cuda" else self.compute_dtype

    def _refresh_include(self) -> None:
        """Recompute the cached include mask after the automata changed."""
        inc = self.ta_state >= self.n_states
        self.include = inc.to(self._include_dtype())
        self.include_count = inc.sum(dim=1)

    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        # .to(device) keeps buffer dtypes; re-derive the include cache for the new device.
        out = super()._apply(fn, recurse)
        if self.is_initialized and self.include.dtype != self._include_dtype():
            self._refresh_include()
        return out

    def _refresh_include_at(self, rows: Tensor, pos: Tensor) -> None:
        """Refresh the include caches where :func:`F.apply_feedback_rows` changed the state:
        whole clauses ``rows`` and the flat positions ``pos`` (unique) elsewhere."""
        N = self.n_states
        if rows.numel():
            inc = self.ta_state[rows] >= N
            self.include[rows] = inc.to(self.include.dtype)
            self.include_count[rows] = inc.sum(dim=1).to(self.include_count.dtype)
        if pos.numel():
            flat = self.include.view(-1)
            old = flat[pos]
            new = (self.ta_state.view(-1)[pos] >= N).to(flat.dtype)
            flat[pos] = new
            self.include_count.index_add_(0, pos // self.include.shape[1],
                                          (new - old).to(self.include_count.dtype))

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

    #: Whether ``forward()`` clamps vote sums to the vote margin. Learning always uses the
    #: clamped sum (the feedback probability is defined on it), but a classifier's prediction
    #: should not: every class saturating at ``+T`` ties for the argmax. Regression keeps the
    #: clamp because its output is the clamped sum rescaled to the target range.
    clamp_prediction_votes: bool = False

    def forward(self, x) -> Tensor:
        """Vote sums ``(B, n_outputs)``. Uses prediction semantics for empty clauses when the
        module is in ``eval()`` mode. The sums are *not* clamped to ``[-T, T]`` (unless the
        model sets ``clamp_prediction_votes``), as in the reference implementations."""
        with torch.no_grad():
            xb = self._prepare(x)
            votes = []
            for sl in chunk_indices(xb.shape[0], self._chunk_size(xb)):
                out, _ = self._evaluate(self._encode(xb[sl]), empty_value=self.training)
                votes.append(self._votes(out, clamp=self.clamp_prediction_votes))
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
                fuse = self._fuse_single_examples(B)
                for i in range(B):
                    acc = self._new_accumulator(sparse=True, fuse=fuse)
                    if fuse:
                        votes_all.append(self._accumulate_one(xb[i : i + 1], y_t[i : i + 1], acc))
                    else:
                        votes_all.append(self._accumulate_chunk(xb[i : i + 1], y_t[i : i + 1], acc))
                    self._commit(acc)
            else:
                acc = self._new_accumulator()
                for sl in chunk_indices(B, self._chunk_size(xb)):
                    votes_all.append(self._accumulate_chunk(xb[sl], y_t[sl], acc))
                self._commit(acc)
            return torch.cat(votes_all, dim=0)

    def train(self, mode: bool = True) -> TsetlinMachineBase:
        """As :meth:`torch.nn.Module.train`; switching to ``eval()`` also releases the count
        buffers that learning keeps between updates."""
        if not mode:
            self.__dict__.pop("_acc_cache", None)
        return super().train(mode)

    def _new_accumulator(self, sparse: Optional[bool] = None, fuse: bool = False) -> FeedbackAccumulator:
        """A zeroed accumulator. The dense count buffers are reused between updates (only
        the rows an update touched are cleared), so sequential mode does not allocate three
        ``(C, 2F)`` tensors per example. ``fuse`` lets a single example be committed by
        :func:`F.commit_single_example`."""
        dev, cd = self.ta_state.device, self.compute_dtype
        acc = self.__dict__.get("_acc_cache")
        if acc is None or not acc.matches(self.n_clauses_total, self.n_literals, dev, cd):
            acc = FeedbackAccumulator(self.n_clauses_total, self.n_literals, dev, cd)
            self.__dict__["_acc_cache"] = acc
        elif self.__dict__.get("_acc_dirty", False):
            # a previous update raised before its commit: clear everything
            for t in (acc.n_true, acc.n_false, acc.n2, acc.n_ib):
                t.zero_()
            acc.rows.zero_()
            acc.extra = {}
            acc.single = None
        self.__dict__["_acc_dirty"] = True
        # Row-sparse commit (F.apply_feedback_rows) on the CPU and for single examples; a
        # batched GPU update is faster dense (no host syncs for the sparse bookkeeping).
        acc.sparse = (dev.type == "cpu") if sparse is None else bool(sparse)
        acc.fuse = bool(fuse)
        return acc

    def _fuse_single_examples(self, n_examples: int = 1) -> bool:
        """Whether sequential mode commits each example with the fused compiled kernel: on
        CUDA, where the row-sparse commit is dominated by kernel launches and host syncs (on
        the CPU the row-sparse commit is cheaper than any dense pass), and not under the hard
        budget, which the kernel does not implement. The compile cost is paid once, so the
        threshold is on the work of the whole call (``n_examples`` steps), not of one step:
        a small model trained example by example is exactly where launches dominate."""
        return (self.ta_state.device.type == "cuda"
                and getattr(type(self)._feedback_counts, "_records_single", False)
                and (self.max_included_literals is None or self.budget_mode == "soft")
                and F._compile_wanted(self.ta_state.numel() * max(1, int(n_examples))))

    def _accumulate_one(self, xb: Tensor, y_t: Tensor, acc: FeedbackAccumulator) -> Tensor:
        """:meth:`_accumulate_chunk` for one example of fused sequential mode, through
        ``torch.compile``: it fuses the ~50 small kernels of clause evaluation and feedback
        selection, whose launch overhead otherwise dominates a single-example step."""
        literals = self._encode(xb)
        votes = F._run_compiled(type(self)._accumulate_literals, self, literals, y_t, acc)
        return self._accumulate_literals(literals, y_t, acc) if votes is None else votes

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
        acc.rows |= ((type_i | type_ii) & clause_out).any(dim=0)
        self._feedback_counts(literals, ctx, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)
        self._accumulate_weights(acc, aux, clause_out, sel_fire_i, sel_fire_ii)
        return votes

    def _commit(self, acc: FeedbackAccumulator) -> None:
        """Apply accumulated feedback to the automata (and weights) and refresh caches."""
        if acc.single is not None:
            lit, lit_ii, fire_i, nofire_i, fire_ii = acc.single
            count = F.commit_single_example(
                self.ta_state, self.include, self.include_count, lit, fire_i, nofire_i, fire_ii,
                n_states=self.n_states, s=self.s, boost_true_positive=self.boost_true_positive,
                max_included_literals=self.max_included_literals,
                literal_active=self._literal_active, literals_ii=lit_ii,
            )
            if count is not None:
                self.include_count = count
                self._commit_weights(acc)
                acc.reset()
                self.__dict__["_acc_dirty"] = False
                return
            self._write_single_counts(acc)  # no compiled kernel: commit the counts instead
        if (not acc.sparse and acc.single is None and self.ta_state.device.type == "cuda"
                and (self.max_included_literals is None or self.budget_mode == "soft")):
            # one fused pass: sample, update the state, refresh the include cache, zero counts
            count = F.commit_dense(
                self.ta_state, self.include, self.include_count, acc.n_true, acc.n_false,
                acc.n_ib, acc.n2, n_states=self.n_states, s=self.s,
                boost_true_positive=self.boost_true_positive,
                max_included_literals=self.max_included_literals,
                literal_active=self._literal_active,
            )
            if count is not None:
                self.include_count = count
                self._commit_weights(acc)
                acc.reset(counts_clear=True)
                self.__dict__["_acc_dirty"] = False
                return
        if acc.sparse:
            changed = F.apply_feedback_rows(
                self.ta_state, acc.n_true, acc.n_false, acc.n_ib, acc.n2, acc.rows,
                n_states=self.n_states, s=self.s, boost_true_positive=self.boost_true_positive,
                max_included_literals=self.max_included_literals,
                include_count=self.include_count, literal_active=self._literal_active,
                budget_mode=self.budget_mode,
            )
        else:
            changed = None
            F.apply_feedback(  # uses the counts as scratch; reset() zeroes them afterwards
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
                budget_mode=self.budget_mode,
            )
        self._commit_weights(acc)
        if changed is None:
            self._refresh_include()
        else:
            self._refresh_include_at(*changed)
        acc.reset()
        self.__dict__["_acc_dirty"] = False

    def _feedback_counts(
        self,
        literals: Tensor,
        ctx: Any,
        sel_fire_i: Tensor,
        sel_nofire_i: Tensor,
        sel_fire_ii: Tensor,
        acc: FeedbackAccumulator,
    ) -> None:
        """Accumulate feedback event counts for a chunk (flat models: matmuls over the
        clauses that matched and were selected; usually a small fraction of all clauses)."""
        if acc.fuse and literals.shape[0] == 1:
            acc.single = (literals[0], literals[0], sel_fire_i[0], sel_nofire_i[0], sel_fire_ii[0])
            return
        acc.n_ib += sel_nofire_i.sum(dim=0)
        neg_lit = 1.0 - literals
        idx = ((sel_fire_i + sel_fire_ii).sum(dim=0) > 0).nonzero().squeeze(1) if acc.sparse else None
        if idx is None or idx.numel() * 4 > sel_fire_i.shape[1]:  # dense is cheaper
            sel_fire_t = sel_fire_i.transpose(0, 1)  # (C, B)
            acc.n_true.addmm_(sel_fire_t, literals)
            acc.n_false.addmm_(sel_fire_t, neg_lit)
            acc.n2.addmm_(sel_fire_ii.transpose(0, 1), neg_lit)
            return
        if idx.numel() == 0:
            return
        sel_i = sel_fire_i[:, idx].transpose(0, 1)  # (R, B)
        acc.n_true.index_add_(0, idx, sel_i @ literals)
        acc.n_false.index_add_(0, idx, sel_i @ neg_lit)
        acc.n2.index_add_(0, idx, sel_fire_ii[:, idx].transpose(0, 1) @ neg_lit)

    _feedback_counts._records_single = True  # type: ignore[attr-defined]

    def _write_single_counts(self, acc: FeedbackAccumulator) -> None:
        """Turn the factors recorded for :func:`F.commit_single_example` into counts."""
        lit, lit_ii, fire_i, nofire_i, fire_ii = acc.single  # type: ignore[misc]
        acc.single = None
        acc.fuse = False
        lit = lit if lit.dim() == 2 else lit.unsqueeze(0)  # (1 or C, 2F)
        lit_ii = lit_ii if lit_ii.dim() == 2 else lit_ii.unsqueeze(0)
        fi = fire_i.unsqueeze(1)
        acc.n_true += fi * lit
        acc.n_false += fi * (1.0 - lit)
        acc.n2 += fire_ii.unsqueeze(1) * (1.0 - lit_ii)
        acc.n_ib += nofire_i

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
            if self.budget_mode != "soft":
                parts.append(f"budget_mode={self.budget_mode!r}")
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
