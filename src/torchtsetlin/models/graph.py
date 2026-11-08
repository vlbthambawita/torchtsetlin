"""Graph Tsetlin machine (Granmo et al., 2025, arXiv:2507.14874).

A *deep clause* ``C_j = C_j^0 AND C_j^1 AND ... AND C_j^{D-1}`` has one component per layer.
``C^0`` reads a node's own features. ``C^i`` reads the node's *inbox* of layer ``i``: the
messages its in-neighbours sent because clause ``k``'s components ``0 .. i-1`` all matched
there, each bound to the type of the edge it travelled along. The clause is True for a graph
iff the deep clause holds at some node (the OR over patches of the convolutional TM), and the
votes and feedback are those of the coalesced TM.

Implementation (``plans/graph-tm.md``): the per-layer inputs of a node are concatenated into
one literal row ``[H^0 | H^1 | ... | H^{D-1}]``, so a clause is a single row of automata and
the deep clause is that row's AND (Lemma 1). Feedback goes, for every layer at once, to the
row of one random node where the full clause matches, which is exactly the convolutional
feedback on the concatenated row (Lemma 2). Only evaluation is new: it runs layer by layer,
because ``H^i`` depends on the layer ``i-1`` matches of the neighbours.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .. import _triton
from .. import functional as F
from ..data.graphs import GraphBatch, SymbolTable, _prev_prime
from ..utils import chunk_indices
from .base import FeedbackAccumulator
from .coalesced import CoalescedTsetlinMachine

__all__ = ["GraphTsetlinMachine"]

_ENCODINGS = ("random", "double_hashing", "one_hot")


class _Encoded:
    """A coerced :class:`GraphBatch` flattened for evaluation: what ``_encode`` returns."""

    __slots__ = ("x0", "node_ok", "node_type", "src", "dst", "etype", "emask", "B", "P", "full")

    def __init__(self, x0, node_ok, node_type, src, dst, etype, B, P, emask=None, full=False):
        self.x0, self.node_ok, self.node_type = x0, node_ok, node_type
        self.src, self.dst, self.etype = src, dst, etype
        self.emask = emask  # (E,) real edges of a padded batch, None when all are real
        self.B, self.P = B, P
        self.full = full  # every graph has P nodes (GraphBatch.full_nodes)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.B, self.P, self.x0.shape[1])


class _GraphMixin:
    """Graph input, layered evaluation with message passing, and the node-level feedback."""

    depth: int
    message_size: int
    message_bits: int
    message_encoding: str
    n_edge_types: int
    n_node_types: int
    shared_node_draw: bool
    layer_s: Optional[Tuple[float, ...]]
    msg_bits: Tensor

    # ---- sizes ---------------------------------------------------------------------
    @property
    def n_node_features(self) -> int:
        """Features per node ``F`` (the model's ``n_features`` also counts the inboxes)."""
        self._check_initialized()  # type: ignore[attr-defined]
        return int(self.n_features) - (self.depth - 1) * self.message_size  # type: ignore[attr-defined]

    def _layer_bounds(self) -> List[int]:
        """Offsets of the layers in the positive half of a literal row (``depth + 1`` values)."""
        F0 = self.n_node_features
        return [0] + [F0 + i * self.message_size for i in range(self.depth)]

    # ---- input ---------------------------------------------------------------------
    def _coerce_input(self, x) -> GraphBatch:  # type: ignore[override]
        if not isinstance(x, GraphBatch):
            raise TypeError("graph Tsetlin machines take a GraphBatch (torchtsetlin.data.GraphBatch)")
        if x.n_edge_types > self.n_edge_types:
            raise ValueError(f"the batch has {x.n_edge_types} edge types, the model {self.n_edge_types}")
        if x.n_node_types > self.n_node_types:
            raise ValueError(f"the batch has {x.n_node_types} node types, the model {self.n_node_types}")
        if self.is_initialized and x.n_features != self.n_node_features:  # type: ignore[attr-defined]
            raise ValueError(f"expected {self.n_node_features} node features, got {x.n_features}")
        self.__dict__["_input_padded"] = x.is_padded  # read by _fuse_single_examples / _graphs_ok
        return x.to(self.ta_state.device)  # type: ignore[attr-defined]

    def _infer_n_features(self, xb: GraphBatch) -> int:  # type: ignore[override]
        return xb.n_features + (self.depth - 1) * self.message_size

    def _encode(self, xb: GraphBatch) -> _Encoded:  # type: ignore[override]
        B, P, F0 = xb.shape
        dt = self._include_dtype()  # type: ignore[attr-defined]
        src, dst, et = xb.flat_edges()
        nt = None if xb.node_type is None else xb.node_type.reshape(B * P)
        em = xb.flat_edge_mask()
        return _Encoded(xb.node_features.reshape(B * P, F0).to(dt), xb.node_mask.reshape(B * P),
                        nt, src, dst, et, B, P, None if em is None else em.unsqueeze(1).to(dt),
                        bool(xb.full_nodes))

    # ---- chunking and the fast paths we do not take --------------------------------
    def _patches_per_example(self, xb: GraphBatch) -> int:  # type: ignore[override]
        return max(1, int(xb.shape[1]))

    def _chunk_elements_per_example(self, xb: GraphBatch) -> int:  # type: ignore[override]
        P = self._patches_per_example(xb)
        C = self.n_clauses_total  # type: ignore[attr-defined]
        Ft = int(self.n_features)  # type: ignore[attr-defined]
        E = xb.edge_slots / max(1, len(xb))
        per_node = C * (3 + self.n_edge_types) + 3 * Ft + self.message_size
        return int(P * per_node + E * C)

    def _event_block(self, P: int) -> int:
        """Feedback events drawn together (as :meth:`_ConvMixin._event_block`)."""
        L = self.n_literals  # type: ignore[attr-defined]
        return max(1, self.max_chunk_elements // (2 * P + 2 * L))  # type: ignore[attr-defined]

    # ---- depth 1: the convolutional kernel -------------------------------------------
    def _bitpacked(self, full: bool) -> bool:
        """Whether clauses are evaluated by :func:`_triton.conv_eval` on bit-packed node
        literals: depth 1 (a graph clause is then a convolutional clause over the node
        features), every graph with all ``P`` nodes (the kernel has no node mask), one node
        type (no clause ownership), CUDA with Triton, outside ``torch.compile``."""
        return (bool(full) and self.depth == 1 and self.n_node_types == 1
                and _triton.available(self.ta_state.device)  # type: ignore[attr-defined]
                and not torch.compiler.is_compiling())

    def _node_literals(self, enc: _Encoded) -> Tensor:
        """``(B, P, 2F)`` literal rows of the nodes (depth 1: own features and negations)."""
        x = enc.x0
        return torch.cat([x, 1.0 - x], dim=1).view(enc.B, enc.P, -1)

    def _bitpacked_chunk_size(self, xb: GraphBatch) -> int:
        """Graphs per chunk on the kernel path: no ``(P, C)`` intermediate (as
        :meth:`_ConvMixin._fused_chunk_size`)."""
        P = max(1, int(xb.shape[1]))
        per = 4 * P * self.n_literals + 8 * self.n_clauses_total  # type: ignore[attr-defined]
        return max(1, min(len(xb), self.max_chunk_elements // per))  # type: ignore[attr-defined]

    def _chunk_size(self, xb: GraphBatch) -> int:  # type: ignore[override]
        if not self.training and self._bitpacked(xb.full_nodes):  # type: ignore[attr-defined]
            return self._bitpacked_chunk_size(xb)
        return super()._chunk_size(xb)  # type: ignore[misc]

    def _events_chunk_size(self, xb: GraphBatch) -> int:  # type: ignore[override]
        if self._bitpacked(xb.full_nodes):
            return self._bitpacked_chunk_size(xb)
        return super()._events_chunk_size(xb)  # type: ignore[misc]

    def _evaluate_learning(self, literals: _Encoded, acc: FeedbackAccumulator) -> Tuple[Tensor, Any]:  # type: ignore[override]
        if acc.record_events and self._bitpacked(literals.full):
            # one kernel: clause outputs plus, per (graph, clause), the random matching node of
            # Type Ia and an independent one of Type II
            lit = self._node_literals(literals)
            B, P, L = lit.shape
            res = _triton.conv_eval(_triton.pack_bits(lit.reshape(B * P, L) == 0),
                                    self.include_bits(), B, P, draw=True)  # type: ignore[attr-defined]
            if res is not None:
                p_i, p_ii = res[1], res[2]
                if self.shared_node_draw:
                    p_ii = p_i
                return res[0], ("nodes", lit, p_i, p_ii)
        return self._evaluate(literals, empty_value=True)

    def _events_eligible(self, n_examples: int) -> bool:  # type: ignore[override]
        """The base rule (CUDA with Triton, soft budget, counts within the kernel table), for
        one ``s`` (the Triton commit takes a scalar specificity)."""
        return self.layer_s is None and super()._events_eligible(n_examples)  # type: ignore[misc]

    def _fuse_single_examples(self, n_examples: int = 1) -> bool:  # type: ignore[override]
        """Fused sequential commits on CUDA for padded batches (the compiled step is traced per
        input shape, and flat batches change shape with every graph), one ``s`` and the soft
        budget. Unlike the base rule there is no minimum model size: an eager graph step is a
        few hundred kernel launches and host syncs (~5 ms on an RTX 3090) whatever the size,
        so fusing (and replaying it from CUDA graphs) pays off even for tiny models."""
        return (self.layer_s is None and self.__dict__.get("_input_padded", False)
                and self.ta_state.device.type == "cuda"  # type: ignore[attr-defined]
                and (self.max_included_literals is None or self.budget_mode == "soft")  # type: ignore[attr-defined]
                and bool(F._COMPILE["enabled"]))

    def _graphs_ok(self, seq: bool, xb, fuse: bool) -> bool:  # type: ignore[override]
        """CUDA graphs for padded batches only: their slices have static shapes and the update
        is free of host synchronisation (flat batches change shape with every graph)."""
        return bool(getattr(xb, "is_padded", False)) and super()._graphs_ok(seq, xb, fuse)  # type: ignore[misc]

    def _graph_key(self, seq: bool, xb: GraphBatch, y_t: Tensor) -> tuple:  # type: ignore[override]
        """The base key on the node features, plus the padded edge layout."""
        e, _, _ = xb._padded  # type: ignore[misc]
        return super()._graph_key(seq, xb.node_features, y_t) + (  # type: ignore[misc]
            tuple(e.shape), xb.node_type is not None, xb.n_edge_types)

    # ---- message passing -----------------------------------------------------------
    def _bind_matrix(self, dtype: torch.dtype) -> Tensor:
        """``(n_edge_types * C, M)``: row ``t * C + j`` holds clause ``j``'s message bits
        shifted by ``t`` (binding to the edge type). Cached; rebuilt when ``msg_bits`` or the
        device/dtype change."""
        key = (self.msg_bits.device, dtype, self.msg_bits.data_ptr(), self.msg_bits._version)
        cache = self.__dict__.get("_bind_cache")
        if cache is not None and cache[0] == key:
            return cache[1]
        C, M, nt = self.msg_bits.shape[0], self.message_size, self.n_edge_types
        bind = torch.zeros(nt, C, M, dtype=dtype, device=self.msg_bits.device)
        rows = torch.arange(C, device=self.msg_bits.device).unsqueeze(1).expand_as(self.msg_bits)
        for t in range(nt):
            bind[t, rows, (self.msg_bits + t) % M] = 1
        bind = bind.reshape(nt * C, M)
        self.__dict__["_bind_cache"] = (key, bind)
        return bind

    def _inbox(self, sent: Tensor, enc: _Encoded) -> Tensor:
        """``(N, M)`` 0/1 inboxes from the ``(N, C)`` Boolean matches of the previous layer."""
        N, C = sent.shape
        nt, M = self.n_edge_types, self.message_size
        dt = enc.x0.dtype
        if enc.src.numel() == 0:
            return torch.zeros(N, M, dtype=dt, device=sent.device)
        # S[n, t, j] = number of in-neighbours of n over type t at which clause j matched
        S = torch.zeros(N * nt, C, dtype=dt, device=sent.device)
        msg = sent[enc.src].to(dt)
        if enc.emask is not None:  # unused edge slots of a padded batch carry nothing
            msg = msg * enc.emask
        S.index_add_(0, enc.dst * nt + enc.etype, msg)
        S = S.view(N, nt, C) > 0
        if self.message_encoding == "one_hot":  # bit j * nt + t: no collisions, no matmul
            return S.transpose(1, 2).reshape(N, C * nt).to(dt)
        if self._dense_binding(N, sent.device):
            hits = torch.matmul(S.reshape(N, nt * C).to(dt), self._bind_matrix(dt))
            return (hits > 0).to(dt)
        # scatter every message bit into its (shifted) inbox column: N * nt * C * bits work
        # instead of the N * nt * C * M of the product, and no (nt * C, M) binding matrix
        hits = torch.zeros(N, M, dtype=torch.float32, device=sent.device)
        Sf = S.to(torch.float32)
        bits = self.msg_bits.to(sent.device)
        for t in range(nt):
            for k in range(bits.shape[1]):
                hits.index_add_(1, (bits[:, k] + t) % M, Sf[:, t])
        return (hits > 0).to(dt)

    def _dense_binding(self, N: int, device: torch.device) -> bool:
        """Hashed messages: bind them by a product with the cached binding matrix (CUDA:
        tensor cores make it as fast as or faster than a scatter, unless the matrix gets
        large) or by a column scatter (CPU, where it is ~15x faster for inboxes of 1024 bits
        and more; the product still wins for small inboxes)."""
        C, M, nt = self.msg_bits.shape[0], self.message_size, self.n_edge_types
        if device.type == "cuda":
            return nt * C * M <= 1 << 28
        return M < 512

    def _evaluate_layers(self, enc: _Encoded) -> Tuple[List[Tensor], List[Tensor]]:
        """Run the layers: ``(running matches per layer, inputs per layer)``, both lists of
        ``(N, ...)`` tensors. ``running[i]`` is ``M^i``: components ``0 .. i`` all match.

        An empty component matches (it constrains nothing), in learning and prediction alike:
        the messages are the next layer's *input*, and other clauses learn to rely on those of
        an empty clause ("has a left neighbour"), so they must not change at prediction time.
        Only the empty clause's own vote is dropped when predicting (:meth:`_evaluate`)."""
        inc = self.include  # type: ignore[attr-defined]
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        C = inc.shape[0]
        bounds = self._layer_bounds()
        dt = enc.x0.dtype
        if inc.dtype != dt:
            inc = inc.to(dt)
        running: List[Tensor] = []
        xs: List[Tensor] = []
        x = enc.x0
        for i in range(self.depth):
            if i:
                x = self._inbox(running[-1], enc)
            xs.append(x)
            a, b = bounds[i], bounds[i + 1]
            # included-but-False literals: positives on 0 inputs, negations on 1 inputs
            viol = torch.matmul(1.0 - x, inc[:, a:b].t()) + torch.matmul(x, inc[:, Fn + a:Fn + b].t())
            m = viol == 0
            if i == 0:
                m &= enc.node_ok.unsqueeze(1)
                if self.n_node_types > 1 and enc.node_type is not None:
                    owner = torch.arange(C, device=m.device) % self.n_node_types
                    m &= enc.node_type.unsqueeze(1) == owner.unsqueeze(0)
                running.append(m)
            else:
                running.append(running[-1] & m)
        return running, xs

    def _evaluate(self, literals: _Encoded, empty_value: bool) -> Tuple[Tensor, Any]:  # type: ignore[override]
        if not empty_value and self._bitpacked(literals.full):
            # prediction at depth 1: the convolutional kernel, no (B, P, C) tensor
            lit = self._node_literals(literals)
            B, P, L = lit.shape
            res = _triton.conv_eval(_triton.pack_bits(lit.reshape(B * P, L) == 0),
                                    self.include_bits(), B, P, draw=False)  # type: ignore[attr-defined]
            if res is not None:
                return res[0] & (self.include_count > 0).unsqueeze(0), None  # type: ignore[attr-defined]
        running, xs = self._evaluate_layers(literals)
        matches = running[-1].view(literals.B, literals.P, -1)  # (B, P, C)
        out = matches.any(dim=1)
        if not empty_value:  # an empty clause does not vote when predicting
            out = out & (self.include_count > 0).unsqueeze(0)  # type: ignore[attr-defined]
        return out, (matches, xs)

    # ---- feedback ------------------------------------------------------------------
    def _feedback_counts(  # type: ignore[override]
        self,
        literals: _Encoded,
        ctx: Any,
        sel_fire_i: Tensor,
        sel_nofire_i: Tensor,
        sel_fire_ii: Tensor,
        acc: FeedbackAccumulator,
    ) -> None:
        if isinstance(ctx[0], str):  # ("nodes", ...): events drawn by the kernel at depth 1
            _, lit, p_i, p_ii = ctx
            acc.n_ib += sel_nofire_i.sum(dim=0)
            acc.n_ia += sel_fire_i.sum(dim=0)
            acc.events = (lit, sel_fire_i, sel_fire_ii, p_i, p_ii)
            return
        matches, xs = ctx
        B, P = literals.B, literals.P
        N, C = B * P, matches.shape[2]
        if acc.record_events or (acc.fuse and B == 1):
            self._record_fast(xs, matches, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)
            return
        acc.n_ib += sel_nofire_i.sum(dim=0)  # Type Ib needs no node features
        if self.shared_node_draw:
            # the reference implementation: one random matching node per (graph, clause),
            # shared by every class and by Type I and Type II
            node = (torch.rand(matches.shape, device=matches.device) * matches).argmax(dim=1)  # (B, C)
        E = self._event_block(P)
        x = None
        for sel, kind in ((sel_fire_i, "i"), (sel_fire_ii, "ii")):
            nz = torch.nonzero(sel, as_tuple=True)
            if nz[0].numel() == 0:
                continue
            nodes = []
            for ev in chunk_indices(nz[0].numel(), E):
                b_idx, j_idx = nz[0][ev], nz[1][ev]
                if self.shared_node_draw:
                    p_idx = node[b_idx, j_idx]
                else:
                    # an independent random matching node per feedback event, drawn exactly
                    # as _ConvMixin._feedback_counts does (same blocks, same random stream)
                    cand = matches[b_idx, :, j_idx]  # (n_sel, P)
                    p_idx = (torch.rand(cand.shape, device=cand.device) * cand).argmax(dim=1)
                nodes.append(b_idx * P + p_idx)
            # The literal row of a node is [H | 1 - H], so the summed rows of the events
            # follow from W @ H, W[j, n] = events of clause j drawn at node n: one sparse
            # product over the positive half instead of gathering 2F-wide rows per event.
            if x is None:
                x = torch.cat(xs, dim=1).to(acc.dtype)  # (N, F)
            idx = torch.stack([nz[1], torch.cat(nodes)])
            W = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], dtype=acc.dtype, device=x.device),
                                        (C, N))
            pos = torch.sparse.mm(W, x)  # (C, F): events whose node has the feature
            cnt = torch.zeros(C, dtype=acc.dtype, device=x.device).index_add_(
                0, nz[1], torch.ones(nz[1].numel(), dtype=acc.dtype, device=x.device)).unsqueeze(1)
            neg = cnt - pos  # events whose node lacks it: the negated literal is True
            Fn = pos.shape[1]
            if kind == "i":
                acc.n_true[:, :Fn] += pos
                acc.n_true[:, Fn:] += neg
                acc.n_false[:, :Fn] += neg
                acc.n_false[:, Fn:] += pos
            else:
                acc.n2[:, :Fn] += neg
                acc.n2[:, Fn:] += pos

    _feedback_counts._records_single = True  # type: ignore[attr-defined]
    _feedback_counts._records_events = True  # type: ignore[attr-defined]

    def _draw_nodes(self, matches: Tensor) -> Tuple[Tensor, Tensor]:
        """``(B, C)`` int32 random matching node for Type Ia and an independent one for Type II
        (the same one under ``shared_node_draw``), drawn densely: no host synchronisation and
        no data-dependent shapes, as the CUDA fast paths need. Equal in distribution to the
        draw of the eager path (arbitrary where the clause matches nowhere, as there)."""
        m = matches.to(torch.float32)
        p_i = (torch.rand(m.shape, device=m.device) * m).argmax(dim=1).to(torch.int32)
        if self.shared_node_draw:
            return p_i, p_i
        p_ii = (torch.rand(m.shape, device=m.device) * m).argmax(dim=1).to(torch.int32)
        return p_i, p_ii

    def _record_fast(self, xs: List[Tensor], matches: Tensor, sel_fire_i: Tensor, sel_nofire_i: Tensor,
                     sel_fire_ii: Tensor, acc: FeedbackAccumulator) -> None:
        """Events mode (``acc.events`` for :func:`_triton.fused_commit`) and fused sequential
        mode (``acc.single`` for :func:`F.commit_single_example`): the convolutional records,
        with nodes for patches and the concatenated layer inputs as the literal row."""
        B, P, C = matches.shape
        full = torch.cat(xs + [1.0 - x for x in xs], dim=1).view(B, P, -1)  # (B, P, 2F)
        p_i, p_ii = self._draw_nodes(matches)
        if acc.record_events:
            acc.n_ib += sel_nofire_i.sum(dim=0)
            acc.n_ia += sel_fire_i.sum(dim=0)
            acc.events = (full, sel_fire_i, sel_fire_ii, p_i, p_ii)
            return
        acc.single = (full[0, p_i[0].long()], full[0, p_ii[0].long()], sel_fire_i[0], sel_nofire_i[0],
                      sel_fire_ii[0])

    def _layer_columns(self) -> List[Tensor]:
        """Literal columns of every layer (positive and negated), into the ``2F`` axis."""
        cached = self.__dict__.get("_layer_cols")
        dev = self.ta_state.device  # type: ignore[attr-defined]
        if (cached is not None and cached[0].device == dev
                and sum(c.numel() for c in cached) == self.ta_state.shape[1]):  # type: ignore[attr-defined]
            return cached
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        bounds = self._layer_bounds()
        cols = [torch.cat([torch.arange(a, b), torch.arange(Fn + a, Fn + b)]).to(dev)
                for a, b in zip(bounds[:-1], bounds[1:])]
        self.__dict__["_layer_cols"] = cols
        return cols

    def _commit(self, acc: FeedbackAccumulator) -> None:  # type: ignore[override]
        """The base commit, or with ``s`` per layer, one :func:`F.apply_feedback_blocks`."""
        if self.layer_s is None:
            super()._commit(acc)  # type: ignore[misc]
            return
        self.__dict__["_bits_fresh"] = False
        F.apply_feedback_blocks(
            self.ta_state, acc.n_true, acc.n_false, acc.n_ib, acc.n2,  # type: ignore[attr-defined]
            list(zip(self._layer_columns(), self.layer_s)),
            n_states=self.n_states,  # type: ignore[attr-defined]
            boost_true_positive=self.boost_true_positive,  # type: ignore[attr-defined]
            max_included_literals=self.max_included_literals,  # type: ignore[attr-defined]
            include_count=self.include_count,  # type: ignore[attr-defined]
            literal_active=self._literal_active,  # type: ignore[attr-defined]
        )
        self._commit_weights(acc)  # type: ignore[attr-defined]
        self._refresh_include()  # type: ignore[attr-defined]
        acc.reset()
        self.__dict__["_acc_dirty"] = False

    # ---- introspection -------------------------------------------------------------
    @torch.no_grad()
    def node_clause_outputs(self, x: GraphBatch, layer: Optional[int] = None,
                            empty_value: Optional[bool] = None) -> Tensor:
        """``(B, P, C)`` Boolean ``M^layer``: clause components ``0 .. layer`` all match at the
        node (default: the last layer, i.e. the full deep clause). Padding nodes are False.
        With prediction semantics (``empty_value=False``, the default in ``eval()`` mode) an
        entirely empty clause is False everywhere, as in :meth:`forward`."""
        layer = self.depth - 1 if layer is None else int(layer)
        if not 0 <= layer < self.depth:
            raise ValueError(f"layer must lie in [0, {self.depth})")
        ev = self.training if empty_value is None else bool(empty_value)  # type: ignore[attr-defined]
        xb = self._prepare(x)  # type: ignore[attr-defined]
        outs = []
        for sl in chunk_indices(len(xb), self._chunk_size(xb)):  # type: ignore[attr-defined]
            enc = self._encode(xb[sl])
            running, _ = self._evaluate_layers(enc)
            m = running[layer]
            if not ev:
                m = m & (self.include_count > 0).unsqueeze(0)  # type: ignore[attr-defined]
            outs.append(m.view(enc.B, enc.P, -1))
        return _pad_cat(outs)

    @torch.no_grad()
    def node_messages(self, x: GraphBatch, layer: int) -> Tensor:
        """``(B, P, M)`` Boolean inbox ``H^layer`` of every node (``1 <= layer < depth``). The
        inboxes are the same when learning and predicting."""
        if not 1 <= layer < self.depth:
            raise ValueError(f"layer must lie in [1, {self.depth})")
        xb = self._prepare(x)  # type: ignore[attr-defined]
        outs = []
        for sl in chunk_indices(len(xb), self._chunk_size(xb)):  # type: ignore[attr-defined]
            enc = self._encode(xb[sl])
            _, xs = self._evaluate_layers(enc)
            outs.append(xs[layer].view(enc.B, enc.P, -1) > 0.5)
        return _pad_cat(outs)

    def message_collision_rate(self) -> float:
        """Fraction of (clause, edge type) messages whose bits are all set by *other*
        messages too, so that a message literal cannot tell them apart (0 for one-hot)."""
        if self.message_encoding == "one_hot":
            return 0.0
        bind = self._bind_matrix(torch.float32) > 0  # (nt*C, M)
        load = bind.sum(dim=0)  # messages per bit
        shared = (load.unsqueeze(0) > 1) | ~bind  # bit used by another message as well
        return float(shared.all(dim=1).float().mean())

    # ---- names and decoded clauses -------------------------------------------------
    def _edge_name(self, t: int, edge_type_names: Optional[Sequence[str]]) -> str:
        names = edge_type_names or self.edge_type_names
        return str(names[t]) if names is not None else f"e{t}"

    def _message_names(self, edge_type_names: Optional[Sequence[str]] = None) -> List[str]:
        """Names of the ``(depth - 1) * M`` message features: ``"<edge><layer>:<clause>"`` for
        one-hot messages (the paper's ``r1:0``), ``"m<layer>[<bit>]"`` for hashed ones."""
        names: List[str] = []
        nt = self.n_edge_types
        for i in range(1, self.depth):
            if self.message_encoding == "one_hot":
                names += [f"{self._edge_name(b % nt, edge_type_names)}{i}:{b // nt}"
                          for b in range(self.message_size)]
            else:
                names += [f"m{i}[{b}]" for b in range(self.message_size)]
        return names

    def default_feature_names(self) -> List[str]:  # type: ignore[override]
        return [f"x{k}" for k in range(self.n_node_features)] + self._message_names()

    def literal_names(self, feature_names: Optional[Sequence[str]] = None) -> List[str]:  # type: ignore[override]
        """Names of all ``2F`` literals. ``feature_names`` may name only the node features;
        the message features are named automatically."""
        names = list(feature_names or self.feature_names or [f"x{k}" for k in range(self.n_node_features)])  # type: ignore[attr-defined]
        if len(names) == self.n_node_features:
            names = names + self._message_names()
        if len(names) != int(self.n_features):  # type: ignore[attr-defined]
            raise ValueError(f"Expected {self.n_node_features} node feature names, got {len(names)}")
        return names + [f"NOT {n}" for n in names]

    def clause_expression(  # type: ignore[override]
        self,
        clause: int,
        feature_names: Optional[Sequence[str]] = None,
        joiner: str = " AND ",
        symbols: Optional[SymbolTable] = None,
        edge_type_names: Optional[Sequence[str]] = None,
        layers: Optional[Sequence[int]] = None,
    ) -> str:
        """Readable deep clause, e.g. ``"A AND right1:0 AND NOT left1:3"``.

        ``right1:0`` is a message literal: clause 0's components ``0 .. 0`` matched at the node
        a *right*-going edge came from (the left neighbour in a sequence), received in layer 1.
        ``layers`` restricts the output to some components, e.g. ``layers=[0]`` for the part
        that a message about this clause vouches for at the sending node.

        With a :class:`~torchtsetlin.data.SymbolTable`, node literals are decoded to symbols:
        ``A`` when all of A's positions are included, ``NOT A`` when all of them are included
        negated. Hashed messages are decoded the same way to ``<edge><layer>:<clause>``. Bits
        that decode to nothing are printed raw (``hv[k]``, ``m1[b]``).
        """
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        pos, neg = mask[:Fn], mask[Fn:]
        bounds = self._layer_bounds()
        terms: List[str] = []
        F0 = self.n_node_features
        show = set(range(self.depth)) if layers is None else {int(i) for i in layers}
        # layer 0: node features
        if 0 not in show:
            pass
        elif symbols is not None:
            if symbols.dim != F0:
                raise ValueError(f"the symbol table has dim {symbols.dim}, the model {F0} node features")
            terms += _decode(pos[:F0], neg[:F0], symbols.positions.to(mask.device),
                             [str(s) for s in symbols.symbols], [f"hv[{k}]" for k in range(F0)])
        else:
            names = self.literal_names(feature_names)
            terms += [names[k] for k in torch.nonzero(pos[:F0]).flatten().tolist()]
            terms += [names[Fn + k] for k in torch.nonzero(neg[:F0]).flatten().tolist()]
        # message layers
        M = self.message_size
        if self.depth > 1:
            groups, labels = self._message_groups(mask.device)
            for i in range(1, self.depth):
                if i not in show:
                    continue
                a, b = bounds[i], bounds[i + 1]
                names = [f"{self._edge_name(t, edge_type_names)}{i}:{j}" for (t, j) in labels]
                terms += _decode(pos[a:b], neg[a:b], groups.to(mask.device), names,
                                 [f"m{i}[{k}]" for k in range(M)])
        return joiner.join(terms) if terms else "TRUE"

    def _message_groups(self, device: torch.device) -> Tuple[Tensor, List[Tuple[int, int]]]:
        """``(groups, labels)``: the inbox positions ``(nt*C, bits)`` of every message and its
        ``(edge type, clause)``, in the same order."""
        M, nt = self.message_size, self.n_edge_types
        C = self.msg_bits.shape[0]
        if self.message_encoding == "one_hot":
            groups = torch.arange(nt * C, device=device).view(C, nt).t().reshape(-1, 1)  # (t, j) -> bit j*nt+t
        else:
            groups = torch.cat([(self.msg_bits.to(device) + t) % M for t in range(nt)])  # (nt*C, bits)
        return groups, [(t, j) for t in range(nt) for j in range(C)]

    @torch.no_grad()
    def explain_node(
        self,
        x: GraphBatch,
        clause: int,
        graph: int = 0,
        node: Optional[int] = None,
        symbols: Optional[SymbolTable] = None,
        edge_type_names: Optional[Sequence[str]] = None,
        max_senders: int = 3,
    ) -> str:
        """Why clause ``clause`` holds (or not) at a node of one graph, traced back to the
        symbols of the nodes that sent the messages it relies on (paper Sec. 3.1, Table 1).

        Every component is printed with :meth:`clause_expression`; under each positive message
        literal ``<edge><i>:<k>`` the in-neighbours that sent it are listed (at most
        ``max_senders``), each with its own explanation of clause ``k`` up to layer ``i - 1``,
        recursively down to layer 0. ``node=None`` picks the first node where the whole clause
        holds. Uses the learning semantics (an empty component matches), which are also the
        ones the messages are computed with. Hashed message bits that decode to no complete
        message are not traced.
        """
        clause, graph = int(clause), int(graph)
        if not 0 <= clause < self.n_clauses:  # type: ignore[attr-defined]
            raise ValueError(f"clause must lie in [0, {self.n_clauses})")  # type: ignore[attr-defined]
        xb = self._prepare(x)  # type: ignore[attr-defined]
        if not 0 <= graph < len(xb):
            raise ValueError(f"graph must lie in [0, {len(xb)})")
        enc = self._encode(xb[graph:graph + 1])
        running, xs = self._evaluate_layers(enc)
        n = int(xb.n_nodes[graph])
        mask = self.included_mask()  # type: ignore[attr-defined]
        bounds = self._layer_bounds()
        dev = mask.device
        groups, labels = self._message_groups(dev) if self.depth > 1 else (None, [])
        real = slice(None) if enc.emask is None else enc.emask.squeeze(1) > 0
        src, dst, et = enc.src[real].tolist(), enc.dst[real].tolist(), enc.etype[real].tolist()

        def node_label(v: int) -> str:
            f = xs[0][v] > 0.5
            if symbols is not None:
                names = [str(s) for s in symbols.decode(f.cpu())]
            else:
                names = [f"x{k}" for k in torch.nonzero(f).flatten().tolist()]
            return "[" + " ".join(names) + "]"

        def expr(k: int, i: int) -> str:
            return self.clause_expression(k, symbols=symbols, edge_type_names=edge_type_names, layers=[i])

        lines: List[str] = []

        def explain(k: int, v: int, top: int, pad: str) -> None:
            for i in range(top + 1):
                ok = bool(running[i][v, k])  # components 0 .. i-1 held, or we returned
                mark = "" if ok else "   <- fails here"
                lines.append(f"{pad}layer {i}: {expr(k, i)}{mark}")
                if not ok:
                    return
                if i == 0 or groups is None:
                    continue
                a, b = bounds[i], bounds[i + 1]
                full = mask[k, a:b][groups].all(dim=1)  # positive messages this component needs
                for g in torch.nonzero(full).flatten().tolist():
                    t, kk = labels[g]
                    senders = [u for u, w, ty in zip(src, dst, et)
                               if w == v and ty == t and bool(running[i - 1][u, kk])]
                    name = f"{self._edge_name(t, edge_type_names)}{i}:{kk}"
                    if not senders:  # a hashed bit set by another message
                        lines.append(f"{pad}  {name}: no sender of this message (bits set by others)")
                        continue
                    for u in senders[:max_senders]:
                        lines.append(f"{pad}  {name} from node {u} {node_label(u)} over "
                                     f"{self._edge_name(t, edge_type_names)}: clause {kk} up to layer {i - 1}")
                        explain(kk, u, i - 1, pad + "    ")
                    if len(senders) > max_senders:
                        lines.append(f"{pad}  ... and {len(senders) - max_senders} more senders")

        last = self.depth - 1
        if node is None:
            hits = torch.nonzero(running[last][:n, clause]).flatten().tolist()
            if not hits:
                return f"clause {clause} holds at no node of graph {graph}"
            node = hits[0]
        node = int(node)
        if not 0 <= node < n:
            raise ValueError(f"node must lie in [0, {n})")
        holds = bool(running[last][node, clause])
        lines.append(f"clause {clause} {'holds' if holds else 'does not hold'} at node {node} "
                     f"{node_label(node)} of graph {graph}")
        explain(clause, node, last, "  ")
        return "\n".join(lines)


def _decode(pos: Tensor, neg: Tensor, groups: Tensor, names: List[str], raw: List[str]) -> List[str]:
    """Literals of one layer as named groups: a group (a symbol / message: its bit positions)
    whose bits are all included positively is ``name``, all included negated ``NOT name``;
    included bits that no complete group explains are printed raw."""
    out: List[str] = []
    used_p = torch.zeros_like(pos)
    used_n = torch.zeros_like(neg)
    full_p = pos[groups].all(dim=1)
    full_n = neg[groups].all(dim=1)
    for g in torch.nonzero(full_p).flatten().tolist():
        out.append(names[g])
        used_p[groups[g]] = True
    for g in torch.nonzero(full_n).flatten().tolist():
        out.append(f"NOT {names[g]}")
        used_n[groups[g]] = True
    out += [raw[k] for k in torch.nonzero(pos & ~used_p).flatten().tolist()]
    out += [f"NOT {raw[k]}" for k in torch.nonzero(neg & ~used_n).flatten().tolist()]
    return out


def _pad_cat(parts: List[Tensor]) -> Tensor:
    """Concatenate ``(b, P_i, ...)`` chunks, padding the node axis to the largest ``P_i``."""
    P = max(p.shape[1] for p in parts)
    out = []
    for p in parts:
        if p.shape[1] < P:
            pad = torch.zeros((p.shape[0], P - p.shape[1]) + tuple(p.shape[2:]), dtype=p.dtype, device=p.device)
            p = torch.cat([p, pad], dim=1)
        out.append(p)
    return torch.cat(out, dim=0)


class GraphTsetlinMachine(_GraphMixin, CoalescedTsetlinMachine):
    """Graph Tsetlin machine (Granmo et al., 2025): deep clauses over graph-structured input.

    Every clause has ``depth`` components. Component 0 matches a node's own Boolean features;
    component ``i >= 1`` matches the node's inbox of layer ``i``, which holds a message
    ``(clause k, edge type t)`` for every in-neighbour, reached over an edge of type ``t``, at
    which clause ``k``'s components ``0 .. i-1`` all match. A clause is True for a graph when
    its components ``0 .. depth-1`` all match at one node. Votes, clause sharing and the
    feedback probabilities are those of :class:`CoalescedTsetlinMachine`; each feedback event
    updates every layer of the clause from one random node where the whole clause matches.

    Input: a :class:`~torchtsetlin.data.GraphBatch`. With ``depth=1`` and the graphs of
    :meth:`GraphBatch.from_images`, this is exactly
    :class:`~torchtsetlin.models.ConvCoalescedTsetlinMachine`.

    Args:
        n_features: Boolean features per node ``F`` (``None`` = lazy, from the first batch).
        n_outputs: number of classes (multi-class) or labels (``multi_label=True``).
        n_clauses: number of shared clauses.
        T: vote margin.
        s: specificity, or one per layer ``(s0, s1, ..., s_{D-1})`` as the reference
            implementation allows (``s0`` for the node features, ``s_i`` for the inbox of
            layer ``i``). A tuple is kept in :attr:`layer_s`; the scalar attribute ``s`` is
            then ``s0``. Per-layer ``s`` needs the soft budget (``budget_mode="soft"``).
        depth: number of layers ``D`` (``1`` = no message passing; each extra layer reaches
            one hop further).
        message_size: inbox size ``M`` per layer for hashed messages. One-hot messages use
            ``M = n_clauses * n_edge_types`` and ignore it.
        message_bits: positions per clause message (hashed ``"random"`` encoding).
        message_encoding: ``"random"`` (default, as the reference implementation),
            ``"double_hashing"`` (2 positions: ``j mod M`` and ``p - (j mod p)``) or
            ``"one_hot"`` (one position per clause and edge type: exact, collision free, and
            readable as the paper's ``r1:0`` notation, but the inbox grows with the clause count).
        n_edge_types: edge types of the dataset (a message is bound to its edge type by a cyclic
            shift of its positions).
        n_node_types: node types; clause ``j`` only matches nodes of type
            ``j mod n_node_types``, as in the reference implementation.
        shared_node_draw: draw one random matching node per (graph, clause) and use it for all
            classes and for Type I and II, as the reference implementation does. The default
            draws a node per feedback event, as the convolutional models do.
        edge_type_names: names used when printing message literals (default ``e0, e1, ...``).
        **kwargs: forwarded to :class:`CoalescedTsetlinMachine` (``multi_label``,
            ``negative_scale``, ``max_included_literals``, ``feedback_mode``, ...).

    The reference implementation updates every class with probability scaled by
    ``q / (K - 1)`` for the negative ones; ``multi_label=True`` with one-hot targets and
    ``negative_scale = q / (K - 1)`` is that rule. It also learns one graph at a time:
    ``feedback_mode="sequential"``.
    """

    def __init__(
        self,
        n_features: Optional[int],
        n_outputs: int,
        n_clauses: int,
        T: float,
        s: Union[float, Sequence[float]] = 10.0,
        *,
        depth: int = 1,
        message_size: int = 256,
        message_bits: int = 2,
        message_encoding: str = "random",
        n_edge_types: int = 1,
        n_node_types: int = 1,
        shared_node_draw: bool = False,
        edge_type_names: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if message_encoding not in _ENCODINGS:
            raise ValueError(f"message_encoding must be one of {_ENCODINGS}")
        if n_edge_types < 1 or n_node_types < 1:
            raise ValueError("n_edge_types and n_node_types must be >= 1")
        C = int(n_clauses)
        if message_encoding == "one_hot":
            message_size, message_bits = C * int(n_edge_types), 1
        elif message_encoding == "double_hashing":
            message_bits = 2
        if message_encoding != "one_hot" and not 1 <= message_bits <= message_size:
            raise ValueError("message_bits must lie in [1, message_size]")
        self.depth = int(depth)
        self.message_size = int(message_size)
        self.message_bits = int(message_bits)
        self.message_encoding = message_encoding
        self.n_edge_types = int(n_edge_types)
        self.n_node_types = int(n_node_types)
        self.shared_node_draw = bool(shared_node_draw)
        self.edge_type_names = None if edge_type_names is None else [str(n) for n in edge_type_names]
        if self.edge_type_names is not None and len(self.edge_type_names) != self.n_edge_types:
            raise ValueError("edge_type_names needs one name per edge type")
        if isinstance(s, (int, float)):
            layer_s: Optional[Tuple[float, ...]] = None
        else:
            layer_s = tuple(float(v) for v in s)
            if len(layer_s) != self.depth:
                raise ValueError(f"s needs one value per layer ({self.depth}), got {len(layer_s)}")
            if kwargs.get("budget_mode", "soft") != "soft":
                raise ValueError("per-layer s needs budget_mode='soft'")
            if len(set(layer_s)) == 1:  # all equal: the scalar path, bit for bit
                s, layer_s = layer_s[0], None
            else:
                s = layer_s[0]
        if any(v < 1.0 for v in (layer_s or (float(s),))):
            raise ValueError("s must be >= 1")
        self.layer_s = layer_s
        total = None if n_features is None else int(n_features) + (self.depth - 1) * self.message_size
        super().__init__(total, n_outputs, n_clauses, T, float(s), **kwargs)
        self.register_buffer("msg_bits", self._make_message_bits())

    def _make_message_bits(self) -> Tensor:
        """``(C, message_bits)`` positions of every clause's message in the inbox."""
        C, M = self.n_clauses, self.message_size
        j = torch.arange(C)
        if self.message_encoding == "one_hot":
            return (j * self.n_edge_types).unsqueeze(1)
        if self.message_encoding == "double_hashing":
            p = _prev_prime(M)
            return torch.stack([j % M, p - (j % p)], dim=1)
        return torch.rand(C, M).argsort(dim=1)[:, : self.message_bits]

    def extra_repr(self) -> str:
        s = "" if self.layer_s is None else f"layer_s={self.layer_s}, "
        return (f"depth={self.depth}, {s}message_size={self.message_size}, "
                f"message_encoding={self.message_encoding!r}, n_edge_types={self.n_edge_types}, "
                + super().extra_repr())
