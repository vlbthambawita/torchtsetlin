"""Graph-structured input for :class:`~torchtsetlin.models.GraphTsetlinMachine`.

A :class:`GraphBatch` holds a batch of directed multigraphs whose nodes carry Boolean feature
vectors and whose edges carry an integer *type*. Node features are usually sparse
hypervectors of symbols, built by a :class:`SymbolTable` (Halenka et al., 2024; Granmo et al.,
2025, arXiv:2507.14874), but any Boolean vector works: an image patch with its position bits
is a node, so a graph without edges is the input of a convolutional Tsetlin machine.

Storage is padded: features ``(B, P, F)`` with ``P`` the largest node count of the batch and a
``(B, P)`` node mask. Edges are kept per graph, sorted by graph, with *local* node indices,
so that slicing a batch (``batch[idx]``) is cheap and keeps every edge inside its graph.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..utils import _to_device, as_bool_tensor

__all__ = ["GraphBatch", "SymbolTable"]


def _prev_prime(n: int) -> int:
    """Largest prime strictly below ``n`` (``sympy.prevprime``), for double hashing."""
    for k in range(int(n) - 1, 1, -1):
        if all(k % d for d in range(2, int(k**0.5) + 1)):
            return k
    raise ValueError("double hashing needs a size of at least 3")


class SymbolTable:
    """Map symbol names to sparse hypervector positions (the node *properties* of a graph).

    Every symbol owns ``bits`` positions out of ``dim``. A node's feature vector is the OR
    (*bundle*) of its symbols' positions. The three encodings follow the reference
    GraphTsetlinMachine (``Graphs`` in ``graphs.py``):

    * ``"random"``: ``bits`` distinct random positions per symbol (seeded);
    * ``"double_hashing"``: positions ``i mod dim`` and ``p - (i mod p)`` with ``p`` the largest
      prime below ``dim`` (always 2 bits);
    * ``"one_hot"``: ``dim = len(symbols)`` and symbol ``i`` owns position ``i``. This is a plain
      Boolean feature per symbol, and it makes decoding exact.

    Args:
        symbols: the symbol names (any hashable values).
        dim: hypervector size (ignored for ``"one_hot"``).
        bits: positions per symbol (ignored for ``"double_hashing"`` and ``"one_hot"``).
        encoding: ``"random"``, ``"double_hashing"`` or ``"one_hot"``.
        seed: seed of the random positions.
    """

    def __init__(
        self,
        symbols: Iterable,
        dim: int = 128,
        bits: int = 2,
        encoding: str = "random",
        seed: int = 0,
    ) -> None:
        self.symbols = list(symbols)
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbol names must be unique")
        self.index = {s: i for i, s in enumerate(self.symbols)}
        n = len(self.symbols)
        if encoding == "one_hot":
            dim, bits = n, 1
            pos = torch.arange(n).unsqueeze(1)
        elif encoding == "double_hashing":
            bits = 2
            p = _prev_prime(dim)
            i = torch.arange(n)
            pos = torch.stack([i % dim, p - (i % p)], dim=1)
        elif encoding == "random":
            if not 1 <= bits <= dim:
                raise ValueError("bits must lie in [1, dim]")
            g = torch.Generator().manual_seed(int(seed))
            pos = torch.stack([torch.randperm(dim, generator=g)[:bits] for _ in range(n)]) if n else \
                torch.zeros(0, bits, dtype=torch.long)
        else:
            raise ValueError("encoding must be 'random', 'double_hashing' or 'one_hot'")
        self.dim = int(dim)
        self.bits = int(bits)
        self.encoding = encoding
        self.positions: Tensor = pos.long()  # (n_symbols, bits)

    def __len__(self) -> int:
        return len(self.symbols)

    def vector(self, symbols: Iterable) -> Tensor:
        """``(dim,)`` Boolean hypervector of a set of symbols (their bundle)."""
        out = torch.zeros(self.dim, dtype=torch.bool)
        for s in symbols:
            out[self.positions[self.index[s]]] = True
        return out

    def encode(self, nodes: Sequence[Iterable]) -> Tensor:
        """``(n, dim)`` Boolean feature matrix, one row per node's symbol set."""
        if not len(nodes):
            return torch.zeros(0, self.dim, dtype=torch.bool)
        return torch.stack([self.vector(s) for s in nodes])

    def decode(self, vector: Tensor) -> List:
        """Symbols whose positions are all set in ``vector`` (``(dim,)`` Boolean)."""
        v = as_bool_tensor(vector).flatten()
        hit = v[self.positions].all(dim=1)
        return [self.symbols[i] for i in torch.nonzero(hit).flatten().tolist()]

    def feature_names(self) -> List[str]:
        """Names of the ``dim`` positions: the symbol for one-hot tables, else ``hv[k]``."""
        if self.encoding == "one_hot":
            return [str(s) for s in self.symbols]
        return [f"hv[{k}]" for k in range(self.dim)]

    def __repr__(self) -> str:
        return f"SymbolTable({len(self)} symbols, dim={self.dim}, bits={self.bits}, encoding={self.encoding!r})"


IndexLike = Union[int, slice, Tensor, Sequence[int]]


class GraphBatch:
    """A padded batch of directed multigraphs with typed edges.

    Args:
        node_features: ``(B, P, F)`` Boolean node features; rows beyond a graph's node count
            are padding and are ignored.
        n_nodes: ``(B,)`` node count per graph.
        edges: ``(E, 2)`` long ``(source, destination)`` pairs of *local* node indices,
            grouped by graph in batch order.
        edge_type: ``(E,)`` integer edge types in ``[0, n_edge_types)``.
        edges_per_graph: ``(B,)`` number of edges of each graph (they sum to ``E``).
        node_type: optional ``(B, P)`` integer node types in ``[0, n_node_types)``.
        n_edge_types: number of edge types of the *dataset* (kept by slices, so a batch that
            happens to lack a type still declares it). Defaults to ``max(edge_type) + 1``.
        n_node_types: number of node types of the dataset. Defaults to ``max(node_type) + 1``.
        edge_type_names: optional names of the edge types, for printing clauses.

    Messages travel along an edge from its source to its destination, so a node learns about
    the nodes that have an edge *into* it. Undirected relations need an edge each way, usually
    with two types (as in the sequence demo: "left" and "right").

    Build batches with :meth:`from_graphs`, :meth:`from_sequences` or :meth:`from_images`;
    index them like tensors along the graph axis (``batch[3]``, ``batch[2:10]``,
    ``batch[torch.tensor([4, 1])]``) and concatenate them with :meth:`cat`.
    """

    def __init__(
        self,
        node_features: Tensor,
        n_nodes: Tensor,
        edges: Optional[Tensor] = None,
        edge_type: Optional[Tensor] = None,
        edges_per_graph: Optional[Tensor] = None,
        node_type: Optional[Tensor] = None,
        n_edge_types: Optional[int] = None,
        n_node_types: Optional[int] = None,
        edge_type_names: Optional[Sequence[str]] = None,
    ) -> None:
        x = as_bool_tensor(node_features)
        if x.dim() != 3:
            raise ValueError("node_features must have shape (B, P, F)")
        dev = x.device
        B, P, _ = x.shape
        n_nodes = torch.as_tensor(n_nodes, dtype=torch.long, device=dev).reshape(-1)
        if n_nodes.numel() != B:
            raise ValueError("n_nodes must hold one count per graph")
        if B and (int(n_nodes.min()) < 0 or int(n_nodes.max()) > P):
            raise ValueError("node counts must lie in [0, P]")
        if edges is None:
            edges = torch.zeros(0, 2, dtype=torch.long, device=dev)
        edges = torch.as_tensor(edges, dtype=torch.long, device=dev).reshape(-1, 2)
        E = edges.shape[0]
        edge_type = (torch.zeros(E, dtype=torch.long, device=dev) if edge_type is None
                     else torch.as_tensor(edge_type, dtype=torch.long, device=dev).reshape(-1))
        if edge_type.numel() != E:
            raise ValueError("edge_type must hold one type per edge")
        if edges_per_graph is None:
            if E and B != 1:
                raise ValueError("edges_per_graph is required for a batch of several graphs")
            edges_per_graph = torch.full((B,), E, dtype=torch.long, device=dev)
        edges_per_graph = torch.as_tensor(edges_per_graph, dtype=torch.long, device=dev).reshape(-1)
        if edges_per_graph.numel() != B or int(edges_per_graph.sum()) != E:
            raise ValueError("edges_per_graph must hold one count per graph and sum to E")
        if E:
            graph = torch.repeat_interleave(torch.arange(B, device=dev), edges_per_graph)
            if int(edges.min()) < 0 or bool((edges >= n_nodes[graph].unsqueeze(1)).any()):
                raise ValueError("an edge refers to a node outside its graph")
            if int(edge_type.min()) < 0:
                raise ValueError("edge types must be non-negative")
        if node_type is not None:
            node_type = torch.as_tensor(node_type, dtype=torch.long, device=dev)
            if node_type.shape != (B, P):
                raise ValueError("node_type must have shape (B, P)")
        self.node_features = x
        self.n_nodes = n_nodes
        self.edges = edges
        self.edge_type = edge_type
        self.edges_per_graph = edges_per_graph
        self.node_type = node_type
        inferred_et = int(edge_type.max()) + 1 if E else 1
        self.n_edge_types = int(n_edge_types) if n_edge_types is not None else inferred_et
        if E and inferred_et > self.n_edge_types:
            raise ValueError("an edge type is >= n_edge_types")
        inferred_nt = int(node_type.max()) + 1 if node_type is not None and node_type.numel() else 1
        self.n_node_types = int(n_node_types) if n_node_types is not None else inferred_nt
        if inferred_nt > self.n_node_types:
            raise ValueError("a node type is >= n_node_types")
        self.edge_type_names = None if edge_type_names is None else [str(n) for n in edge_type_names]
        if self.edge_type_names is not None and len(self.edge_type_names) != self.n_edge_types:
            raise ValueError("edge_type_names needs one name per edge type")

    # ------------------------------------------------------------------ tensor-like API
    @property
    def shape(self) -> torch.Size:
        """``(B, P, F)``, the shape of the padded node features."""
        return self.node_features.shape

    @property
    def device(self) -> torch.device:
        return self.node_features.device

    @property
    def n_features(self) -> int:
        return int(self.node_features.shape[2])

    def dim(self) -> int:
        return 3

    def __len__(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def node_mask(self) -> Tensor:
        """``(B, P)`` Boolean: which rows are real nodes."""
        P = self.node_features.shape[1]
        return torch.arange(P, device=self.device).unsqueeze(0) < self.n_nodes.unsqueeze(1)

    @property
    def edge_graph(self) -> Tensor:
        """``(E,)`` graph index of every edge."""
        return torch.repeat_interleave(torch.arange(len(self), device=self.device), self.edges_per_graph)

    def to(self, device) -> GraphBatch:
        device = torch.device(device)
        if device == self.device:
            return self
        mv = lambda t: None if t is None else _to_device(t, device)  # noqa: E731
        return GraphBatch(mv(self.node_features), mv(self.n_nodes), mv(self.edges), mv(self.edge_type),
                          mv(self.edges_per_graph), mv(self.node_type), self.n_edge_types, self.n_node_types,
                          self.edge_type_names)

    def __getitem__(self, index: IndexLike) -> GraphBatch:
        B = len(self)
        if isinstance(index, int):
            if not -B <= index < B:
                raise IndexError("graph index out of range")
            idx = torch.tensor([index % B], device=self.device)
        elif isinstance(index, slice):
            idx = torch.arange(B, device=self.device)[index]
        else:
            idx = torch.as_tensor(index, device=self.device)
            if idx.dtype == torch.bool:
                idx = torch.nonzero(idx).flatten()
            idx = idx.long().reshape(-1)
            idx = torch.where(idx < 0, idx + B, idx)
            if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= B):
                raise IndexError("graph index out of range")
        n_nodes = self.n_nodes[idx]
        P = int(n_nodes.max()) if idx.numel() else 0  # trim the padding to the subset
        feats = self.node_features[idx, :P]
        ntype = None if self.node_type is None else self.node_type[idx, :P]
        start = torch.cumsum(self.edges_per_graph, 0) - self.edges_per_graph
        counts = self.edges_per_graph[idx]
        total = int(counts.sum())
        if total:
            first = torch.repeat_interleave(start[idx], counts)
            offset = torch.arange(total, device=self.device) - torch.repeat_interleave(
                torch.cumsum(counts, 0) - counts, counts)
            sel = first + offset
            edges, etype = self.edges[sel], self.edge_type[sel]
        else:
            edges = self.edges[:0]
            etype = self.edge_type[:0]
        return GraphBatch(feats, n_nodes, edges, etype, counts, ntype, self.n_edge_types, self.n_node_types,
                          self.edge_type_names)

    @staticmethod
    def cat(batches: Sequence[GraphBatch]) -> GraphBatch:
        """Concatenate batches along the graph axis (padding to the largest node count)."""
        if not batches:
            raise ValueError("nothing to concatenate")
        P = max(b.shape[1] for b in batches)
        F_ = batches[0].n_features
        if any(b.n_features != F_ for b in batches):
            raise ValueError("all batches need the same feature count")
        dev = batches[0].device
        pad = lambda t, v: torch.nn.functional.pad(t, (0, 0, 0, P - t.shape[1]), value=v) if t.dim() == 3 \
            else torch.nn.functional.pad(t, (0, P - t.shape[1]), value=v)  # noqa: E731
        feats = torch.cat([pad(b.node_features.to(dev), False) for b in batches])
        has_type = any(b.node_type is not None for b in batches)
        ntype = None
        if has_type:
            ntype = torch.cat([pad(b.node_type.to(dev) if b.node_type is not None else
                                   torch.zeros(b.shape[:2], dtype=torch.long, device=dev), 0) for b in batches])
        return GraphBatch(
            feats, torch.cat([b.n_nodes.to(dev) for b in batches]),
            torch.cat([b.edges.to(dev) for b in batches]), torch.cat([b.edge_type.to(dev) for b in batches]),
            torch.cat([b.edges_per_graph.to(dev) for b in batches]), ntype,
            max(b.n_edge_types for b in batches), max(b.n_node_types for b in batches),
            batches[0].edge_type_names)

    def flat_edges(self) -> Tuple[Tensor, Tensor, Tensor]:
        """``(src, dst, type)`` with node indices into the flattened ``B * P`` node axis."""
        P = self.node_features.shape[1]
        base = self.edge_graph * P
        return base + self.edges[:, 0], base + self.edges[:, 1], self.edge_type

    def __repr__(self) -> str:
        return (f"GraphBatch({len(self)} graphs, max {self.shape[1]} nodes, {self.n_features} features, "
                f"{self.edges.shape[0]} edges, {self.n_edge_types} edge types)")

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_graphs(
        cls,
        node_features: Sequence,
        edges: Optional[Sequence] = None,
        edge_types: Optional[Sequence] = None,
        node_types: Optional[Sequence] = None,
        n_edge_types: Optional[int] = None,
        n_node_types: Optional[int] = None,
        edge_type_names: Optional[Sequence[str]] = None,
    ) -> GraphBatch:
        """Build a batch from per-graph lists.

        Args:
            node_features: per graph, an ``(n_i, F)`` Boolean array.
            edges: per graph, an ``(E_i, 2)`` array of ``(source, destination)`` local node
                indices (or ``None`` / empty for no edges).
            edge_types: per graph, an ``(E_i,)`` array of edge types (default: all ``0``).
            node_types: per graph, an ``(n_i,)`` array of node types (default: none).
        """
        feats = [as_bool_tensor(f) for f in node_features]
        if not feats:
            raise ValueError("no graphs")
        F_ = feats[0].shape[1] if feats[0].dim() == 2 else 0
        n = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        P = max(1, int(n.max()))
        X = torch.zeros(len(feats), P, F_, dtype=torch.bool)
        for i, f in enumerate(feats):
            if f.dim() != 2 or f.shape[1] != F_:
                raise ValueError("every graph needs (n_i, F) node features with the same F")
            X[i, : f.shape[0]] = f.cpu()
        E_list, T_list, per = [], [], []
        for i in range(len(feats)):
            e = edges[i] if edges is not None else None
            e = torch.zeros(0, 2, dtype=torch.long) if e is None else torch.as_tensor(e, dtype=torch.long).reshape(-1, 2)
            t = edge_types[i] if edge_types is not None else None
            t = torch.zeros(e.shape[0], dtype=torch.long) if t is None else torch.as_tensor(t, dtype=torch.long).reshape(-1)
            E_list.append(e)
            T_list.append(t)
            per.append(e.shape[0])
        nt = None
        if node_types is not None:
            nt = torch.zeros(len(feats), P, dtype=torch.long)
            for i, t in enumerate(node_types):
                t = torch.as_tensor(t, dtype=torch.long).reshape(-1)
                nt[i, : t.numel()] = t
        return cls(X, n, torch.cat(E_list), torch.cat(T_list), torch.tensor(per, dtype=torch.long), nt,
                   n_edge_types, n_node_types, edge_type_names)

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[Sequence],
        symbols: SymbolTable,
        edge_types: Tuple[str, str] = ("left", "right"),
        unknown: str = "ignore",
    ) -> GraphBatch:
        """One node per sequence element, chained in both directions.

        Node ``k`` sends to ``k - 1`` over edge type 0 (named ``edge_types[0]``) and to
        ``k + 1`` over type 1, as in the reference ``SequenceClassificationDemo``. An element
        may be a single symbol or a set of symbols. Symbols missing from the table are
        ignored (``unknown="ignore"``) or rejected (``"error"``).
        """
        if unknown not in ("ignore", "error"):
            raise ValueError("unknown must be 'ignore' or 'error'")
        feats, edges, types = [], [], []
        for seq in sequences:
            nodes = []
            for el in seq:
                group = el if isinstance(el, (set, frozenset, list, tuple)) else [el]
                kept = [s for s in group if s in symbols.index]
                if unknown == "error" and len(kept) != len(group):
                    raise KeyError(f"unknown symbol in {group!r}")
                nodes.append(kept)
            feats.append(symbols.encode(nodes) if nodes else torch.zeros(0, symbols.dim, dtype=torch.bool))
            L = len(nodes)
            k = torch.arange(L)
            left = torch.stack([k[1:], k[:-1]], dim=1)   # k -> k-1, type 0
            right = torch.stack([k[:-1], k[1:]], dim=1)  # k -> k+1, type 1
            edges.append(torch.cat([left, right]))
            types.append(torch.cat([torch.zeros(max(L - 1, 0), dtype=torch.long),
                                    torch.ones(max(L - 1, 0), dtype=torch.long)]))
        return cls.from_graphs(feats, edges, types, n_edge_types=2, edge_type_names=edge_types)

    @classmethod
    def from_images(
        cls,
        images: Any,
        patch_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        position_encoding: bool = True,
    ) -> GraphBatch:
        """One node per image patch and no edges (paper Sec. 3.2, "disconnected nodes").

        Node features are the ``Z * kh * kw`` patch pixels followed by thermometer-coded
        patch coordinates, exactly the patch features of
        :class:`~torchtsetlin.models.ConvTsetlinMachine` with the same arguments. A depth-1
        graph machine on these graphs is therefore the convolutional coalesced machine.

        Args:
            images: Boolean ``(B, Z, H, W)`` (or ``(B, H, W)``) images.
        """
        x = as_bool_tensor(images)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4:
            raise ValueError("images must have shape (B, Z, H, W) or (B, H, W)")
        kh, kw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        B, Z, H, W = x.shape
        Py, Px = (H - kh) // sh + 1, (W - kw) // sw + 1
        if Py < 1 or Px < 1:
            raise ValueError("image smaller than the patch")
        patches = torch.nn.functional.unfold(x.float(), kernel_size=(kh, kw), stride=(sh, sw))
        patches = patches.transpose(1, 2) > 0.5  # (B, P, Z*kh*kw)
        if position_encoding:
            ys = torch.arange(Py, device=x.device).repeat_interleave(Px)
            xs = torch.arange(Px, device=x.device).repeat(Py)
            pos = torch.cat([ys.unsqueeze(1) > torch.arange(Py - 1, device=x.device).unsqueeze(0),
                             xs.unsqueeze(1) > torch.arange(Px - 1, device=x.device).unsqueeze(0)], dim=1)
            patches = torch.cat([patches, pos.unsqueeze(0).expand(B, -1, -1)], dim=2)
        P = Py * Px
        n = torch.full((B,), P, dtype=torch.long, device=x.device)
        return cls(patches, n, None, None, torch.zeros(B, dtype=torch.long, device=x.device), None, 1, 1)
