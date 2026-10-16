"""Continuous ridge reference (spec §14.3; arms A4, A1c, A2c) and the §13.3 constants.

Solves ``(A_k^T A_k / N + lambda I) theta_k = A_k^T Y_k / N`` per output by Cholesky of the same
Gram matrix the Gram-form optimizer uses.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.linalg as sla

from .hybrid import Layout, design_T, gates_by_output


def ridge_solve(X: np.ndarray, G: np.ndarray, Y: np.ndarray, layout: Layout, lam: float,
                gram: Optional[np.ndarray] = None, with_mu: bool = False) -> Dict:
    from .opt_gram import build_gram
    Gk = gates_by_output(G, layout)
    N = X.shape[0]
    H = build_gram(X, Gk, layout) if gram is None else gram
    b = design_T(X, Gk, Y, layout) / N
    theta = np.zeros((layout.K, layout.D))
    mu = np.full(layout.K, np.nan)
    for k in range(layout.K):
        Hk = H[k] + lam * np.eye(layout.D)
        try:
            c, low = sla.cho_factor(Hk, lower=True, check_finite=False)
            theta[k] = sla.cho_solve((c, low), b[k], check_finite=False)
        except np.linalg.LinAlgError:
            # singular only when lambda = 0 (redundant intercepts): minimum-norm solution
            theta[k] = np.linalg.lstsq(Hk, b[k], rcond=None)[0]
        if with_mu:
            mu[k] = sla.eigvalsh(Hk, subset_by_index=[0, 0], check_finite=False)[0]
    return dict(theta=theta, mu=mu)


def quantization_bound(H: np.ndarray, lam: float, delta: float, mu: np.ndarray, eps: float = 0.0) -> float:
    """Spec §13.3: ``sum_k sum_r (delta H_rr / 2 + eps / delta)^2 / (2 mu_k)``."""
    tot = 0.0
    for k in range(H.shape[0]):
        hrr = np.diagonal(H[k]) + lam
        tot += float(((delta * hrr / 2 + eps / delta) ** 2).sum() / (2 * mu[k]))
    return tot
