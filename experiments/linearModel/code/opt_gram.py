"""Gram-form exact optimizer (PLAN §5.2, D4).

With ``Hraw_k = A_k^T A_k / N`` and the maintained gradient ``g_k = A_k^T R_k / N + lambda theta_k``,
a move ``m`` on coordinate ``r`` changes ``g_k`` by ``delta m (Hraw_k[r, :] + lambda e_r)`` — exact,
so each decision is O(1) and each accepted move one O(D_k) axpy. Same decisions as the reference
in exact arithmetic; rounding can differ (checked in gate G4). The outer loop rebuilds ``g`` from
the rebuilt residual once per sweep and logs the drift.
"""

from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

from .hybrid import Layout
from .opt_ref import best_move


def design_rows(Xc: np.ndarray, gc: np.ndarray, layout: Layout) -> np.ndarray:
    """Dense rows of ``A_k`` for a chunk: ``Xc (n, d)``, ``gc (n, M)`` -> ``(n, D)``."""
    n = Xc.shape[0]
    parts = []
    if layout.clause_bias:
        parts.append(np.ones((n, 1)))
    if layout.slopes:
        parts.append(Xc)
    phi = np.concatenate(parts, 1)                                   # (n, per)
    body = gc.astype(np.float64)[:, :, None] * phi[:, None, :]       # (n, M, per)
    return np.concatenate([np.ones((n, 1)), body.reshape(n, -1)], 1)


def build_gram(X: np.ndarray, Gk: np.ndarray, layout: Layout, chunk: int = 2048,
               out: np.ndarray = None, blocked: bool = True) -> np.ndarray:
    """``Hraw (K, D, D) = A_k^T A_k / N`` in float64.

    ``blocked``: block ``(j, j')`` is ``Z_S^T Z_S / N`` over the rows ``S`` where clauses ``j`` and
    ``j'`` both fire (``Z = [1, x]`` restricted to the layout's per-clause features), so the work
    scales with gate co-activation instead of ``N`` (PLAN §5.2). Otherwise dense row chunks."""
    if blocked:
        return _build_gram_blocked(X, Gk, layout, out)
    K, N, _ = Gk.shape
    D = layout.D
    H = np.zeros((K, D, D)) if out is None else out
    for k in range(K):
        Hk = H[k]
        Hk[...] = 0.0
        for i in range(0, N, chunk):
            A = design_rows(X[i:i + chunk], Gk[k, i:i + chunk], layout)
            Hk += A.T @ A
        Hk /= N
    return H


@njit(cache=True, parallel=True)
def _sweep_all(H, g, perms, q, h, delta, Q, lam, tol, line):
    K, D = q.shape
    acc = np.zeros(K, np.int64)
    pred = np.zeros(K)
    for k in prange(K):
        a = 0
        p = 0.0
        for t in range(perms.shape[1]):
            r = perms[k, t]
            m, change = best_move(q[k, r], g[k, r], h[k, r], delta, Q, line)
            if m != 0 and change < -tol:
                q[k, r] = q[k, r] + m
                s = delta * m
                for c in range(D):
                    g[k, c] += s * H[k, r, c]
                g[k, r] += s * lam
                p += change
                a += 1
        acc[k] = a
        pred[k] = p
    return acc, pred


class GramEngine:
    name = "gram"

    def __init__(self, X: np.ndarray, Gk: np.ndarray, layout: Layout, h: np.ndarray, H: np.ndarray = None):
        t0 = time.perf_counter()
        self.H = build_gram(X, Gk, layout) if H is None else H
        self.build_seconds = time.perf_counter() - t0
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.layout = layout

    def start(self, R: np.ndarray, q: np.ndarray, delta: float, lam: float, g0: np.ndarray = None) -> None:
        self.g = np.ascontiguousarray(g0.copy())

    def sweep(self, perms, q, delta, Q, lam, tol, line):
        return _sweep_all(self.H, self.g, perms, q, self.h, float(delta), int(Q), float(lam), float(tol), bool(line))

    def resync(self, R: np.ndarray, g_rebuilt: np.ndarray) -> float:
        drift = float(np.abs(self.g - g_rebuilt).max())
        self.g = np.ascontiguousarray(g_rebuilt.copy())
        return drift


def _build_gram_blocked(X: np.ndarray, Gk: np.ndarray, layout: Layout, out: np.ndarray = None) -> np.ndarray:
    K, N, M = Gk.shape
    D, per = layout.D, layout.per_clause
    H = np.zeros((K, D, D)) if out is None else out
    parts = []
    if layout.clause_bias:
        parts.append(np.ones((N, 1)))
    if layout.slopes:
        parts.append(X)
    Z = np.ascontiguousarray(np.concatenate(parts, 1))              # (N, per)
    for k in range(K):
        Hk = H[k]
        Hk[...] = 0.0
        Hk[0, 0] = N
        for j in range(M):
            Sj = np.flatnonzero(Gk[k, :, j])
            if len(Sj) == 0:
                continue
            Zj = Z[Sj]
            a = 1 + j * per
            Hk[0, a:a + per] = Zj.sum(0)
            Hk[a:a + per, 0] = Hk[0, a:a + per]
            Gsub = Gk[k][Sj]                                          # (|S_j|, M)
            for j2 in range(j, M):
                Zjj = Zj[Gsub[:, j2]] if j2 != j else Zj
                if Zjj.shape[0] == 0:
                    continue
                b = 1 + j2 * per
                blk = Zjj.T @ Zjj
                Hk[a:a + per, b:b + per] = blk
                if j2 != j:
                    Hk[b:b + per, a:a + per] = blk.T
        Hk /= N
    return H
