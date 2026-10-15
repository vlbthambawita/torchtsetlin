"""Synthetic explainability tasks with known per-feature contributions (Phase 4).

Modelled on the three XAI-Bench families used in the paper (Liu et al., 2021): Gaussian
Linear, Gaussian Non-Linear Additive and Gaussian Piecewise Constant, with feature correlation
``rho``. This is a self-contained re-implementation, not the XAI-Bench package (D10): the label
is ``y = 1[f(x) + eps > median]`` with ``f(x) = sum_m g_m(x_m)``, so for independent features
(``rho = 0``) the exact interventional Shapley value of feature ``m`` on ``f`` is
``g_m(x_m) - E[g_m(x_m)]`` -- the ground truth the attributions are scored against.

Metrics (all per example, then averaged):

- ``gt_corr``: Pearson correlation of the attribution vector with the ground-truth
  contributions (the paper's "Shapley correlation", with exact instead of estimated Shapley).
- ``faithfulness``: Pearson correlation of the attribution with the change of the *full model*
  output when each feature is set to its baseline (XAI-Bench faithfulness with a fixed baseline).
- ``infidelity`` (Yeh et al., 2019): ``E_I[(I . phi - (f(x) - f(x - I)))^2]`` with
  ``I ~ N(0, sigma^2)``, sigma = 0.1 in standardized units, 32 draws per example.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from .data import TabularSplits, synthetic_splits

TASKS = ("linear", "nonlinear", "piecewise")


def _g(kind: str, X: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    d = X.shape[1]
    if kind == "linear":
        beta = rng.uniform(-1, 1, size=d)
        return X * beta, {"beta": beta.tolist()}
    if kind == "nonlinear":
        funcs = [np.sin, np.abs, lambda v: v ** 2 / 2, np.tanh, lambda v: np.exp(-v ** 2)]
        sel = rng.integers(len(funcs), size=d)
        a = rng.uniform(0.5, 1.5, size=d) * rng.choice([-1, 1], size=d)
        G = np.stack([a[m] * funcs[sel[m]](X[:, m]) for m in range(d)], 1)
        return G, {"func": sel.tolist(), "scale": a.tolist()}
    if kind == "piecewise":
        cuts = np.sort(rng.normal(size=(d, 3)), 1)
        vals = rng.uniform(-1, 1, size=(d, 4))
        G = np.stack([vals[m][np.searchsorted(cuts[m], X[:, m])] for m in range(d)], 1)
        return G, {"cuts": cuts.tolist(), "values": vals.tolist()}
    raise ValueError(kind)


def make_task(spec: str, seed: int, n: int = 10000, d: int = 8) -> Tuple[TabularSplits, np.ndarray]:
    """``spec`` = ``<kind>`` or ``<kind>_rho<r>`` (e.g. ``linear_rho0.5``).

    Returns the splits and the ground-truth contribution matrix of the *test* split, expressed
    in the same units as the logit of a calibrated model up to a positive scale.
    """
    kind, rho = spec, 0.0
    if "_rho" in spec:
        kind, r = spec.split("_rho")
        rho = float(r)
    rng = np.random.default_rng(1000 + seed)
    cov = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    G, params = _g(kind, X, rng)
    f = G.sum(1)
    noise = rng.normal(scale=0.1 * f.std(), size=n)
    y = (f + noise > np.median(f)).astype(np.int64)
    S = synthetic_splits(X, y, f"xai_{spec}", seed)
    te = np.asarray(S.info.pop("test_index"))
    gt = G[te] - G.mean(0, keepdims=True)
    S.info.update({"task": kind, "rho": rho, "params": params})
    S.gt_test = gt  # type: ignore[attr-defined]
    return S, gt


def _rowcorr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 1e-12
    return float(np.mean((a * b).sum(1)[ok] / den[ok])) if ok.any() else float("nan")


@torch.no_grad()
def xai_metrics(ex, X: torch.Tensor, phi: np.ndarray, gt: np.ndarray, seed: int = 0,
                sigma: float = 0.1, n_draws: int = 32) -> Dict[str, float]:
    from .explain import removal_attributions
    delta = removal_attributions(ex, X, use_local=False)
    z0 = ex.full(X)[:, 0]
    gen = torch.Generator(device=X.device).manual_seed(seed)
    phit = torch.as_tensor(phi, device=X.device, dtype=torch.float32)
    errs = []
    for _ in range(n_draws):
        Ipert = torch.randn(X.shape, generator=gen, device=X.device) * sigma
        lhs = (Ipert * phit).sum(1)
        rhs = z0 - ex.full(X - Ipert)[:, 0]
        errs.append(((lhs - rhs) ** 2).cpu().numpy())
    return {"gt_corr": _rowcorr(phi, gt), "faithfulness": _rowcorr(phi, delta),
            "infidelity": float(np.mean(errs))}
