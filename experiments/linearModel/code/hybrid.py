"""Hybrid model of spec §4–§5, §8 and §15.2: frozen Boolean gates times affine outputs.

``S_k(x) = beta_k + sum_j C_kj(x) (b_kj + w_kj^T x)`` with every coefficient ``delta * q``,
``q`` an integer state in ``[-Q, Q]``.

Coordinate layout per output (spec §5.2): global intercept first, then for each clause its
local intercept (if ``clause_bias``) followed by its ``d`` slopes (if ``slopes``) in ascending
feature order. ``A1`` uses ``slopes=False``; ``A2`` uses a :class:`~.adapter.ConstantGate` with
``clause_bias=False``; ``A3`` uses both.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from .adapter import ConstantGate, gates_from_state

GLOBAL_BIAS, CLAUSE_BIAS, SLOPE = 0, 1, 2
FORMAT_VERSION = 1


@dataclass(frozen=True)
class Layout:
    K: int
    M: int
    d: int
    clause_bias: bool = True
    slopes: bool = True

    @property
    def per_clause(self) -> int:
        return int(self.clause_bias) + self.d * int(self.slopes)

    @property
    def D(self) -> int:
        """``D_k`` (same for every output)."""
        return 1 + self.M * self.per_clause

    @property
    def D_total(self) -> int:
        return self.K * self.D

    def descriptor(self, r: int) -> Tuple[int, int, int]:
        """``(kind, clause_id, feature_id)`` of coordinate ``r`` (``-1`` where absent)."""
        if r == 0:
            return GLOBAL_BIAS, -1, -1
        j, w = divmod(r - 1, self.per_clause)
        if self.clause_bias and w == 0:
            return CLAUSE_BIAS, j, -1
        return SLOPE, j, w - int(self.clause_bias)

    def kinds(self) -> np.ndarray:
        """``(D,)`` int8 coordinate kinds, ``(D,)`` clause ids, ``(D,)`` feature ids."""
        kinds = np.empty(self.D, np.int8)
        cl = np.full(self.D, -1, np.int32)
        ft = np.full(self.D, -1, np.int32)
        for r in range(self.D):
            kinds[r], cl[r], ft[r] = self.descriptor(r)
        return np.stack([kinds.astype(np.int32), cl, ft])

    # ---- vectorised unpacking -------------------------------------------------
    def split(self, q: np.ndarray):
        """``q`` ``(K, D)`` -> ``beta (K,)``, ``bias (K, M)`` or None, ``W (K, M, d)`` or None."""
        beta = q[:, 0]
        body = q[:, 1:].reshape(self.K, self.M, self.per_clause)
        bias = body[..., 0] if self.clause_bias else None
        W = body[..., int(self.clause_bias):] if self.slopes else None
        return beta, bias, W

    def join(self, beta, bias, W) -> np.ndarray:
        body = np.zeros((self.K, self.M, self.per_clause), dtype=np.result_type(beta))
        if self.clause_bias:
            body[..., 0] = bias
        if self.slopes:
            body[..., int(self.clause_bias):] = W
        return np.concatenate([np.asarray(beta).reshape(self.K, 1), body.reshape(self.K, -1)], 1)


def gates_by_output(G: np.ndarray, layout: Layout) -> np.ndarray:
    """``(N, K*M)`` bank-ordered gates -> ``(K, N, M)``."""
    N = G.shape[0]
    return np.ascontiguousarray(G.reshape(N, layout.K, layout.M).transpose(1, 0, 2))


# ------------------------------------------------------------------------- prediction paths
def scores_direct(X: np.ndarray, G: np.ndarray, layout: Layout, q: np.ndarray, delta: float) -> np.ndarray:
    """Spec §8.1 PREDICT_SCORES (vectorised over rules): float64 ``(N, K)``."""
    X = np.asarray(X, dtype=np.float64)
    Gk = gates_by_output(G, layout).astype(np.float64)        # (K, N, M)
    theta = delta * q.astype(np.float64)
    beta, bias, W = layout.split(theta)
    S = np.empty((X.shape[0], layout.K))
    for k in range(layout.K):
        local = np.zeros((X.shape[0], layout.M))
        if bias is not None:
            local += bias[k][None, :]
        if W is not None:
            local += X @ W[k].T
        S[:, k] = beta[k] + (Gk[k] * local).sum(1)            # inactive rules contribute 0
    return S


def design_column(X: np.ndarray, G: np.ndarray, layout: Layout, k: int, r: int) -> np.ndarray:
    """Spec §8.2 DESIGN_COLUMN."""
    kind, j, f = layout.descriptor(r)
    if kind == GLOBAL_BIAS:
        return np.ones(X.shape[0])
    g = G[:, k * layout.M + j].astype(np.float64)
    if kind == CLAUSE_BIAS:
        return g
    return g * X[:, f]


def scores_from_states(X: np.ndarray, G: np.ndarray, layout: Layout, q: np.ndarray, delta: float) -> np.ndarray:
    """Spec §8.2 SCORES_FROM_STATES (column path; slow, for verification)."""
    S = np.zeros((X.shape[0], layout.K))
    for k in range(layout.K):
        for r in np.flatnonzero(q[k]):
            S[:, k] += delta * q[k, r] * design_column(X, G, layout, k, r)
    return S


# ------------------------------------------------------------------------- A^T v in block form
def design_T(X: np.ndarray, Gk: np.ndarray, V: np.ndarray, layout: Layout) -> np.ndarray:
    """``A_k^T V[:, k]`` for every output -> ``(K, D)``; ``Gk`` from :func:`gates_by_output`."""
    out = np.zeros((layout.K, layout.D))
    for k in range(layout.K):
        v = V[:, k]
        gv = Gk[k].astype(np.float64) * v[:, None]            # (N, M)
        bias = gv.sum(0) if layout.clause_bias else None
        W = (X.T @ gv).T if layout.slopes else None           # (M, d)
        out[k] = _join1(layout, v.sum(), bias, W)
    return out


def _join1(layout: Layout, beta, bias, W) -> np.ndarray:
    body = np.zeros((layout.M, layout.per_clause))
    if layout.clause_bias:
        body[:, 0] = bias
    if layout.slopes:
        body[:, int(layout.clause_bias):] = W
    return np.concatenate([[beta], body.reshape(-1)])


def column_sq_norms(X: np.ndarray, Gk: np.ndarray, layout: Layout) -> np.ndarray:
    """``a^T a`` for every coordinate -> ``(K, D)``."""
    X2 = X * X
    out = np.zeros((layout.K, layout.D))
    for k in range(layout.K):
        g = Gk[k].astype(np.float64)
        out[k] = _join1(layout, float(X.shape[0]), g.sum(0), (X2.T @ g).T if layout.slopes else None)
    return out


def objective(R: np.ndarray, q: np.ndarray, delta: float, lam: float) -> float:
    """Spec §10.1 OBJECTIVE: ``||R||^2 / (2N) + lambda/2 ||delta q||^2``."""
    th = delta * q.astype(np.float64)
    return float((R * R).sum() / (2 * R.shape[0]) + lam / 2 * (th * th).sum())


def validation_key(S: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Spec §10.1 VALIDATION_KEY: ``(-accuracy, mse)``; argmax ties -> lowest index."""
    acc = float((S.argmax(1) == y).mean())          # np.argmax returns the first maximum
    Y = np.zeros_like(S)
    Y[np.arange(len(y)), y] = 1.0
    return (-acc, float(((S - Y) ** 2).mean()))


# ------------------------------------------------------------------------- model record
@dataclass
class HybridConfig:
    delta: float
    Q: int
    lam: float


class HybridModel:
    """Spec §5.2 ``HybridModel``: gates + layout + integer states."""

    def __init__(self, gates, layout: Layout, cfg: HybridConfig, states: np.ndarray = None,
                 metadata: dict = None):
        if cfg.delta <= 0 or cfg.Q < 1 or cfg.lam < 0:
            raise ValueError("need delta > 0, Q >= 1, lambda >= 0")
        if gates.K != layout.K or gates.M != layout.M:
            raise ValueError("gate banks and layout disagree")
        self.gates, self.layout, self.cfg = gates, layout, cfg
        if states is None:
            states = np.zeros((layout.K, layout.D), np.int16)
        self.states = self._check_states(states)
        self.metadata = dict(metadata or {})

    def _check_states(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q)
        if q.shape != (self.layout.K, self.layout.D) or not np.issubdtype(q.dtype, np.integer):
            raise ValueError(f"bad states {q.shape} {q.dtype}")
        if np.abs(q.astype(np.int64)).max(initial=0) > self.cfg.Q:
            raise ValueError("state out of bounds")
        return q.astype(np.int16)

    def gate_values(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.gates(X, device=device)

    def predict_scores(self, X: np.ndarray, G: np.ndarray = None) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.layout.d or not np.isfinite(X).all():
            raise ValueError("invalid input")
        G = self.gate_values(X) if G is None else G
        return scores_direct(X, G, self.layout, self.states, self.cfg.delta)

    def predict(self, X: np.ndarray, G: np.ndarray = None) -> np.ndarray:
        return self.predict_scores(X, G).argmax(1)

    # ---- persistence (spec §15.2) --------------------------------------------
    def checksum(self) -> str:
        h = hashlib.sha256(np.ascontiguousarray(self.states).tobytes())
        h.update(json.dumps([asdict(self.layout), asdict(self.cfg), self.gates.fingerprint]).encode())
        return h.hexdigest()

    def save(self, path) -> int:
        """Write a versioned ``.npz``; returns its size in bytes."""
        gst = self.gates.state()
        arrays = {"states": self.states}
        scalars = {}
        for kk, v in gst.items():
            if isinstance(v, np.ndarray):
                arrays["gates/" + kk] = v
            else:
                scalars[kk] = v
        meta = dict(format_version=FORMAT_VERSION, task="CLASSIFICATION", input_dimension=self.layout.d,
                    output_labels=list(range(self.layout.K)), numeric_preprocessor="pixel/255",
                    layout=asdict(self.layout), cfg=asdict(self.cfg), gates_scalars=scalars,
                    gate_fingerprint=self.gates.fingerprint, checksum=self.checksum(),
                    metadata=self.metadata)
        arrays["meta"] = np.frombuffer(json.dumps(meta, default=_jsonable).encode(), np.uint8)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        return len(data)

    @classmethod
    def load(cls, path, device: str = "cpu") -> HybridModel:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta"]).decode())
        if meta["format_version"] != FORMAT_VERSION:
            raise RuntimeError("unknown format version")
        gst = dict(meta["gates_scalars"])
        for kk in z.files:
            if kk.startswith("gates/"):
                gst[kk[len("gates/"):]] = z[kk]
        if gst.get("weight", "missing") is None or "weight" not in gst:
            gst["weight"] = None
        gates = gates_from_state(gst, device=device)
        if gates.fingerprint != meta["gate_fingerprint"]:
            raise RuntimeError("gate fingerprint mismatch")
        m = cls(gates, Layout(**meta["layout"]), HybridConfig(**meta["cfg"]), z["states"], meta["metadata"])
        if m.checksum() != meta["checksum"]:
            raise RuntimeError("artifact checksum mismatch")
        return m


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def a2_model(K: int, d: int, cfg: HybridConfig) -> HybridModel:
    return HybridModel(ConstantGate(K), Layout(K, 1, d, clause_bias=False, slopes=True), cfg)


def summary(layout: Layout) -> Dict[str, int]:
    return dict(K=layout.K, M=layout.M, d=layout.d, D_k=layout.D, D_total=layout.D_total)
