"""Image -> model input, built per mini-batch on the GPU (plan §3, §4, §5).

A stored ``(N, P, F)`` node-feature array would be 4 GB for arm D and far more for arm P, so only
the Boolean image planes live on the GPU and the node features of a batch are unfolded from them.

* ``GridInput`` (arm D): planes = the 8-level colour thermometer (24 x 32 x 32). Nodes = the
  windows of ``GraphBatch.from_images(patch 4, stride 2)`` (15 x 15 = 225 nodes, 384 patch bits +
  28 thermometer position bits: exactly the patch features of the convolutional models). Every
  node sends to its right, left, lower and upper neighbour over four edge types (840 edges).
  ``edges="rewired"``: the same number of edges and types into every node, but each sender is
  replaced by a random other node (one fixed rewiring per run, the same for every image).
* ``PaperInput`` (arm P, GraphTM): planes = adaptive Gaussian threshold (3) + thermometer (24).
  Nodes = 8 x 8 windows at stride 1 (625), features = 192 + 1 536 window bits + one-hot row (25)
  and column (25) = 1 778 symbols, one feature each; no edges.
* ``ConvInput`` (arm P, CoTM baseline): the 3 threshold planes, as an image tensor.

Batches use the padded layout (fixed shapes), which is what the CUDA fast paths need. Only public
attributes of ``GraphBatch`` are used: a padded template per batch size, with the node features
replaced.
"""

import copy

import numpy as np
import torch
from common import EDGE_TYPES, GRID, PAPER, PATCH, STRIDE

from torchtsetlin.data import GraphBatch

_LEVELS = torch.tensor(np.arange(1, 9) * 255 / 9, dtype=torch.float64)


def thermometer(x, dev, chunk=2000):
    """uint8 (N, 32, 32, 3) array -> bool (N, 24, 32, 32) on ``dev``: the colour thermometer of
    cifar10_batch_clauses (``x >= k*255/9``, k = 1..8, plane = channel * 8 + level)."""
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    lv = _LEVELS.to(dev)
    out = []
    for i in range(0, len(xt), chunk):
        b = xt[i:i + chunk]
        bits = b.unsqueeze(-1).to(torch.float64) >= lv                       # (n, 32, 32, 3, 8)
        out.append(bits.reshape(len(b), 32, 32, 24).permute(0, 3, 1, 2).contiguous())
    return torch.cat(out)


def adaptive_threshold(x):
    """``cv2.adaptiveThreshold(channel, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2)`` per
    colour channel (the reference ``CIFAR10Demo.py``): uint8 (N, 32, 32, 3) -> bool (N, 3, 32, 32)."""
    import cv2
    out = np.empty((x.shape[0], 3, 32, 32), dtype=bool)
    for i in range(x.shape[0]):
        for j in range(3):
            out[i, j] = cv2.adaptiveThreshold(np.ascontiguousarray(x[i, :, :, j]), 1,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) > 0
    return out


def two_views(x, dev):
    """uint8 images -> bool (N, 27, 32, 32) on ``dev``: planes 0-2 threshold, 3-26 thermometer."""
    return torch.cat([torch.from_numpy(adaptive_threshold(x)).to(dev), thermometer(x, dev)], dim=1)


def grid_edges(edges="grid", seed=0):
    """(E, 2) (source, destination) node indices and (E,) types of one image graph.

    Type t is the direction the message travels (EDGE_TYPES): a node's ``right`` in-edge comes
    from its left neighbour. Rewired: every in-edge keeps its destination and type, its source
    becomes a random other node (distinct per destination)."""
    src, dst, typ = [], [], []
    for y in range(GRID):
        for x in range(GRID):
            n = y * GRID + x
            for t, (dy, dx) in enumerate(((0, 1), (0, -1), (1, 0), (-1, 0))):
                yy, xx = y + dy, x + dx
                if 0 <= yy < GRID and 0 <= xx < GRID:
                    src.append(n), dst.append(yy * GRID + xx), typ.append(t)
    src, dst, typ = np.array(src), np.array(dst), np.array(typ)
    if edges == "rewired":
        rng = np.random.default_rng([seed, 20261120])
        for d in range(GRID * GRID):
            k = np.flatnonzero(dst == d)
            others = np.delete(np.arange(GRID * GRID), d)
            src[k] = rng.choice(others, size=len(k), replace=False)
    elif edges != "grid":
        raise KeyError(edges)
    return np.stack([src, dst], axis=1), typ


class _Padded:
    """Padded batches of same-shaped graphs: a template per batch size, features replaced."""

    n_nodes: int
    n_edge_types = 1
    edge_type_names = None

    def __init__(self, dev, edge_index=None, edge_type=None):
        self.dev = dev
        self._e = np.zeros((0, 2), dtype=np.int64) if edge_index is None else edge_index
        self._t = np.zeros(0, dtype=np.int64) if edge_type is None else edge_type
        self._templates = {}

    def _template(self, B):
        t = self._templates.get(B)
        if t is None:
            E, P = len(self._e), self.n_nodes
            t = GraphBatch(torch.zeros(B, P, 1, dtype=torch.bool), torch.full((B,), P),
                           torch.from_numpy(np.tile(self._e, (B, 1))), torch.from_numpy(np.tile(self._t, B)),
                           torch.full((B,), E), None, self.n_edge_types, 1, self.edge_type_names)
            t = self._templates[B] = t.pad(P, E).to(self.dev)
        return t

    def features(self, xb):
        raise NotImplementedError

    def __call__(self, xb):
        """Image planes (B, Z, 32, 32) on the device -> the padded GraphBatch of these images."""
        g = copy.copy(self._template(xb.shape[0]))
        g.node_features = self.features(xb)
        g.full_nodes = True                    # every graph has all its nodes (set on the host)
        return g


class GridInput(_Padded):
    kind = "grid"
    patch, stride, grid = PATCH, STRIDE, GRID
    n_nodes = GRID * GRID
    n_edge_types = len(EDGE_TYPES)
    edge_type_names = EDGE_TYPES
    planes = 24

    def __init__(self, dev, edges="grid", seed=0):
        super().__init__(dev, *grid_edges(edges, seed))
        self.edges = edges

    def features(self, xb):
        return GraphBatch.from_images(xb, PATCH, STRIDE).node_features

    def feature_names(self):
        """Names of the 412 node features, in the order of ``from_images`` (channel-major)."""
        names = [f"{'RGB'[p // 8]}>={p % 8 + 1}[{ky},{kx}]" for p in range(24)
                 for ky in range(PATCH) for kx in range(PATCH)]
        return names + [f"row>{k}" for k in range(GRID - 1)] + [f"col>{k}" for k in range(GRID - 1)]


class PaperInput(_Padded):
    kind = "paper_graph"
    patch, stride, grid = PAPER["patch"], PAPER["stride"], PAPER["grid"]
    n_nodes = PAPER["grid"] ** 2
    planes = 27

    def __init__(self, dev):
        super().__init__(dev)
        g = self.grid
        n = torch.arange(g * g, device=dev)
        pos = torch.zeros(g * g, 2 * g, dtype=torch.bool, device=dev)
        pos[n, torch.div(n, g, rounding_mode="floor")] = True          # row symbol
        pos[n, g + n % g] = True                                       # column symbol
        self.pos = pos

    def features(self, xb):
        f = xb.half() if xb.is_cuda else xb.float()
        u = torch.nn.functional.unfold(f, self.patch, stride=self.stride).transpose(1, 2) > 0.5
        return torch.cat([u, self.pos.unsqueeze(0).expand(xb.shape[0], -1, -1)], dim=2)

    def feature_names(self):
        k = self.patch
        planes = [f"thr{'RGB'[c]}" for c in range(3)] + [f"{'RGB'[p // 8]}>={p % 8 + 1}" for p in range(24)]
        names = [f"{p}[{ky},{kx}]" for p in planes for ky in range(k) for kx in range(k)]
        return names + [f"row={i}" for i in range(self.grid)] + [f"col={i}" for i in range(self.grid)]


class ConvInput:
    """The CoTM baseline of arm P: the threshold planes as a Boolean image tensor."""
    kind = "paper_cotm"
    patch, stride, grid = PAPER["patch"], PAPER["stride"], PAPER["grid"]
    n_nodes = PAPER["grid"] ** 2
    planes = 3

    def __init__(self, dev):
        self.dev = dev

    def __call__(self, xb):
        return xb


def make_input(kind, dev, edges="grid", seed=0):
    if kind == "grid":
        return GridInput(dev, edges, seed)
    return PaperInput(dev) if kind == "paper_graph" else ConvInput(dev)
