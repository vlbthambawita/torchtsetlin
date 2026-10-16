"""Pure-Python transcription of spec §10.1–§10.3 on an explicit dense design matrix.

Slow; only for tiny verification fixtures. It is the ground truth the numba engines are
compared against (same permutations -> same accepted moves).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def objective(A: List[np.ndarray], Y: np.ndarray, q: List[np.ndarray], delta: float, lam: float) -> float:
    N = Y.shape[0]
    data = sum(float(((Ak @ (delta * qk) - Y[:, k]) ** 2).sum()) for k, (Ak, qk) in enumerate(zip(A, q)))
    pen = sum(float(((delta * qk) ** 2).sum()) for qk in q)
    return data / (2 * N) + lam / 2 * pen


def best_one_state_move(state: int, g: float, h: float, delta: float, Q: int) -> Tuple[int, float]:
    best_move, best_change = 0, 0.0
    for m in (-1, +1):
        proposed = int(state) + m
        if proposed < -Q or proposed > Q:
            continue
        s = delta * m
        change = s * g + (s * s / 2) * h
        if change < best_change:
            best_move, best_change = m, change
    return best_move, best_change


def best_line_move(state: int, g: float, h: float, delta: float, Q: int) -> Tuple[int, float]:
    if h <= 0:
        return 0, 0.0
    m = int(math.floor(-g / (delta * h) + 0.5))
    m = max(-Q - int(state), min(Q - int(state), m))
    if m == 0:
        return 0, 0.0
    s = delta * m
    return m, s * g + (s * s / 2) * h


def fit(A: List[np.ndarray], Y: np.ndarray, delta: float, Q: int, lam: float, accept_tol: float,
        perms_per_sweep: List[List[np.ndarray]], line: bool = False, check_each_move: bool = True):
    """Returns ``(q, per-sweep list of accepted (k, r, m) moves, stop_reason)``."""
    N = Y.shape[0]
    K = len(A)
    q = [np.zeros(Ak.shape[1], np.int64) for Ak in A]
    R = -Y.astype(np.float64).copy()
    h = [np.array([Ak[:, r] @ Ak[:, r] / N + lam for r in range(Ak.shape[1])]) for Ak in A]
    log = []
    stop = "SWEEP_BUDGET"
    for perms in perms_per_sweep:
        accepted = []
        for k in range(K):
            for r in perms[k]:
                a = A[k][:, r]
                theta = delta * q[k][r]
                g = float(a @ R[:, k]) / N + lam * theta
                m, change = (best_line_move if line else best_one_state_move)(q[k][r], g, h[k][r], delta, Q)
                if m != 0 and change < -accept_tol:
                    if check_each_move:
                        before = objective(A, Y, q, delta, lam)
                    q[k][r] += m
                    R[:, k] = R[:, k] + delta * m * a
                    if check_each_move:
                        after = objective(A, Y, q, delta, lam)
                        assert after < before, (before, after)
                        assert abs((after - before) - change) <= 1e-12 * max(1.0, abs(before))
                    accepted.append((k, int(r), int(m)))
        R = np.stack([A[k] @ (delta * q[k]) for k in range(K)], 1) - Y
        log.append(accepted)
        if not accepted:
            stop = "NO_ACCEPTED_MOVES"
            break
    return q, log, stop


def audit(A: List[np.ndarray], Y: np.ndarray, q: List[np.ndarray], delta: float, Q: int, lam: float,
          accept_tol: float) -> dict:
    N = Y.shape[0]
    smallest, n_imp = 0.0, 0
    for k, Ak in enumerate(A):
        R = Ak @ (delta * q[k]) - Y[:, k]
        for r in range(Ak.shape[1]):
            a = Ak[:, r]
            g = float(a @ R) / N + lam * delta * q[k][r]
            h = float(a @ a) / N + lam
            _, c = best_one_state_move(q[k][r], g, h, delta, Q)
            smallest = min(smallest, c)
            n_imp += c < -accept_tol
    return dict(stationary_within_tolerance=n_imp == 0, improving_coordinate_count=int(n_imp),
                best_available_objective_change=smallest)
