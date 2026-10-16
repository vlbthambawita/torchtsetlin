"""Spec-literal reference optimizer (spec §9–§10): lazy design columns over the active rows of
each clause, ``g = a^T R / N + lambda theta`` recomputed for every visited coordinate from the
latest residual, ``BEST_ONE_STATE_MOVE``, residual update ``R += delta m a``.

Outputs are separable (spec §10.2), so each output's coordinate loop runs in its own numba
thread with its own pre-drawn permutation; the decisions are identical to running the outputs
one after another.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .hybrid import Layout


@njit(cache=True, inline="always")
def best_move(state, g, h, delta, Q, line):
    """Spec §10.1 BEST_ONE_STATE_MOVE (``line=False``), or the A3-L integer line search:
    the integer minimiser of the 1-D quadratic, ``clip(round(-g / (delta h)))`` (D5)."""
    if not line:
        best_m = 0
        best_c = 0.0
        for m in (-1, 1):
            p = np.int64(state) + m
            if p < -Q or p > Q:
                continue
            s = delta * m
            c = s * g + (s * s / 2.0) * h
            if c < best_c:
                best_m = m
                best_c = c
        return best_m, best_c
    if h <= 0.0:
        return 0, 0.0
    m = np.int64(np.floor(-g / (delta * h) + 0.5))
    lo = -Q - np.int64(state)
    hi = Q - np.int64(state)
    if m < lo:
        m = lo
    if m > hi:
        m = hi
    if m == 0:
        return 0, 0.0
    s = delta * m
    return m, s * g + (s * s / 2.0) * h


@njit(cache=True)
def _sweep_one(X, indptr, rows, kinds, clid, ftid, perm, q, R, h, delta, Q, lam, tol, line):
    N = R.shape[0]
    accepted = 0
    predicted = 0.0
    for t in range(perm.shape[0]):
        r = perm[t]
        kind = kinds[r]
        j = clid[r]
        f = ftid[r]
        dot = 0.0
        if kind == 0:
            for i in range(N):
                dot += R[i]
        elif kind == 1:
            for p in range(indptr[j], indptr[j + 1]):
                dot += R[rows[p]]
        else:
            for p in range(indptr[j], indptr[j + 1]):
                i = rows[p]
                dot += X[i, f] * R[i]
        g = dot / N + lam * (delta * q[r])
        m, change = best_move(q[r], g, h[r], delta, Q, line)
        if m != 0 and change < -tol:
            q[r] = q[r] + m
            s = delta * m
            if kind == 0:
                for i in range(N):
                    R[i] += s
            elif kind == 1:
                for p in range(indptr[j], indptr[j + 1]):
                    R[rows[p]] += s
            else:
                for p in range(indptr[j], indptr[j + 1]):
                    i = rows[p]
                    R[i] += s * X[i, f]
            predicted += change
            accepted += 1
    return accepted, predicted


@njit(cache=True, parallel=True)
def _sweep_all(X, indptr, rows, kinds, clid, ftid, perms, q, R, h, delta, Q, lam, tol, line):
    K = q.shape[0]
    acc = np.zeros(K, np.int64)
    pred = np.zeros(K)
    for k in prange(K):
        Rk = R[:, k].copy()
        a, p = _sweep_one(X, indptr[k], rows[k], kinds, clid, ftid, perms[k], q[k], Rk, h[k],
                          delta, Q, lam, tol, line)
        R[:, k] = Rk
        acc[k] = a
        pred[k] = p
    return acc, pred


class RefEngine:
    name = "ref"

    def __init__(self, X: np.ndarray, Gk: np.ndarray, layout: Layout, h: np.ndarray):
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.layout = layout
        K, N, M = Gk.shape
        self.kinds, self.clid, self.ftid = (np.ascontiguousarray(a) for a in layout.kinds())
        # CSR of active rows per (output, clause): column j of output k touches only these rows
        self.indptr = np.zeros((K, M + 1), np.int64)
        rows = np.zeros((K, max(1, int(Gk.sum(axis=(1, 2)).max()))), np.int64)
        for k in range(K):
            off = 0
            for j in range(M):
                act = np.flatnonzero(Gk[k, :, j])
                rows[k, off:off + len(act)] = act
                off += len(act)
                self.indptr[k, j + 1] = off
        self.rows = rows
        self.h = np.ascontiguousarray(h, dtype=np.float64)

    def start(self, R: np.ndarray, q: np.ndarray, delta: float, lam: float) -> None:
        self.R = np.ascontiguousarray(R.copy())

    def sweep(self, perms, q, delta, Q, lam, tol, line):
        return _sweep_all(self.X, self.indptr, self.rows, self.kinds, self.clid, self.ftid,
                          perms, q, self.R, self.h, float(delta), int(Q), float(lam), float(tol), bool(line))

    def resync(self, R: np.ndarray, g_rebuilt: np.ndarray) -> float:
        """Install the rebuilt residual; return the max drift of the maintained residual."""
        drift = float(np.abs(self.R - R).max())
        self.R = np.ascontiguousarray(R.copy())
        return drift
