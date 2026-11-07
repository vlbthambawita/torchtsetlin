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

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..data.graphs import GraphBatch, SymbolTable, _prev_prime
from ..utils import chunk_indices
from .base import FeedbackAccumulator
from .coalesced import CoalescedTsetlinMachine
from .conv import _ConvMixin

__all__ = ["GraphTsetlinMachine"]

_ENCODINGS = ("random", "double_hashing", "one_hot")


class _Encoded:
    """A coerced :class:`GraphBatch` flattened for evaluation: what ``_encode`` returns."""

    __slots__ = ("x0", "node_ok", "node_type", "src", "dst", "etype", "B", "P")

    def __init__(self, x0, node_ok, node_type, src, dst, etype, B, P):
        self.x0, self.node_ok, self.node_type = x0, node_ok, node_type
        self.src, self.dst, self.etype = src, dst, etype
        self.B, self.P = B, P

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
        return x.to(self.ta_state.device)  # type: ignore[attr-defined]

    def _infer_n_features(self, xb: GraphBatch) -> int:  # type: ignore[override]
        return xb.n_features + (self.depth - 1) * self.message_size

    def _encode(self, xb: GraphBatch) -> _Encoded:  # type: ignore[override]
        B, P, F0 = xb.shape
        dt = self._include_dtype()  # type: ignore[attr-defined]
        src, dst, et = xb.flat_edges()
        nt = None if xb.node_type is None else xb.node_type.reshape(B * P)
        return _Encoded(xb.node_features.reshape(B * P, F0).to(dt), xb.node_mask.reshape(B * P),
                        nt, src, dst, et, B, P)

    # ---- chunking and the fast paths we do not take --------------------------------
    def _patches_per_example(self, xb: GraphBatch) -> int:  # type: ignore[override]
        return max(1, int(xb.shape[1]))

    def _chunk_elements_per_example(self, xb: GraphBatch) -> int:  # type: ignore[override]
        P = self._patches_per_example(xb)
        C = self.n_clauses_total  # type: ignore[attr-defined]
        Ft = int(self.n_features)  # type: ignore[attr-defined]
        E = int(xb.edges.shape[0]) / max(1, len(xb))
        per_node = C * (3 + self.n_edge_types) + 3 * Ft + self.message_size
        return int(P * per_node + E * C)

    def _event_block(self, P: int) -> int:
        """Feedback events drawn together (as :meth:`_ConvMixin._event_block`)."""
        L = self.n_literals  # type: ignore[attr-defined]
        return max(1, self.max_chunk_elements // (2 * P + 2 * L))  # type: ignore[attr-defined]

    def _events_eligible(self, n_examples: int) -> bool:  # type: ignore[override]
        return False  # no Triton events path for graphs (yet)

    def _fuse_single_examples(self, n_examples: int = 1) -> bool:  # type: ignore[override]
        return False

    def _graphs_ok(self, seq: bool, xb, fuse: bool) -> bool:  # type: ignore[override]
        return False  # graph shapes vary between batches: no CUDA-graph capture

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
        S.index_add_(0, enc.dst * nt + enc.etype, sent[enc.src].to(dt))
        S = S.view(N, nt, C) > 0
        if self.message_encoding == "one_hot":  # bit j * nt + t: no collisions, no matmul
            return S.transpose(1, 2).reshape(N, C * nt).to(dt)
        hits = torch.matmul(S.reshape(N, nt * C).to(dt), self._bind_matrix(dt))
        return (hits > 0).to(dt)

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
        matches, xs = ctx
        B, P = literals.B, literals.P
        # one literal row per node, laid out like ta_state: [H0 | H1 | ...] then the negations
        full = torch.cat(xs + [1.0 - x for x in xs], dim=1).view(B, P, -1)
        if not self.shared_node_draw:
            # an independent random matching node per feedback event (the conv feedback)
            _ConvMixin._feedback_counts(self, full, matches, sel_fire_i, sel_nofire_i,  # type: ignore[arg-type]
                                        sel_fire_ii, acc)
            return
        # the reference implementation: one random matching node per (graph, clause), shared
        # by every class and by Type I and Type II
        acc.n_ib += sel_nofire_i.sum(dim=0)
        node = (torch.rand(matches.shape, device=matches.device) * matches).argmax(dim=1)  # (B, C)
        flat = full.reshape(B * P, -1)
        for sel, kind in ((sel_fire_i, "i"), (sel_fire_ii, "ii")):
            nz = torch.nonzero(sel, as_tuple=True)
            for ev in chunk_indices(nz[0].numel(), self._event_block(P)):
                b_idx, j_idx = nz[0][ev], nz[1][ev]
                rows = flat[b_idx * P + node[b_idx, j_idx]].to(acc.dtype)
                if kind == "i":
                    acc.n_true.index_add_(0, j_idx, rows)
                    acc.n_false.index_add_(0, j_idx, 1.0 - rows)
                else:
                    acc.n2.index_add_(0, j_idx, 1.0 - rows)

    _feedback_counts._records_single = False  # type: ignore[attr-defined]
    _feedback_counts._records_events = False  # type: ignore[attr-defined]

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
        M, nt = self.message_size, self.n_edge_types
        if self.depth > 1:
            C = self.msg_bits.shape[0]
            if self.message_encoding == "one_hot":
                groups = torch.arange(nt * C, device=mask.device).view(C, nt).t().reshape(-1, 1)  # (t, j) -> bit j*nt+t
            else:
                groups = torch.cat([(self.msg_bits + t) % M for t in range(nt)])  # (nt*C, bits)
            labels = [(t, j) for t in range(nt) for j in range(C)]
            for i in range(1, self.depth):
                if i not in show:
                    continue
                a, b = bounds[i], bounds[i + 1]
                names = [f"{self._edge_name(t, edge_type_names)}{i}:{j}" for (t, j) in labels]
                terms += _decode(pos[a:b], neg[a:b], groups.to(mask.device), names,
                                 [f"m{i}[{k}]" for k in range(M)])
        return joiner.join(terms) if terms else "TRUE"


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
        s: specificity.
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
        s: float = 10.0,
        *,
        depth: int = 1,
        message_size: int = 256,
        message_bits: int = 2,
        message_encoding: str = "random",
        n_edge_types: int = 1,
        n_node_types: int = 1,
        shared_node_draw: bool = False,
        edge_type_names: Optional[Sequence[str]] = None,
        **kwargs,
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
        total = None if n_features is None else int(n_features) + (self.depth - 1) * self.message_size
        super().__init__(total, n_outputs, n_clauses, T, s, **kwargs)
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
        return (f"depth={self.depth}, message_size={self.message_size}, "
                f"message_encoding={self.message_encoding!r}, n_edge_types={self.n_edge_types}, "
                + super().extra_repr())
