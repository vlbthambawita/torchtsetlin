"""Phase 5 / gate G5: 1-D regression with known fixed interval gates (spec §12.2, §13.4).

``f(x) = sin(3x)`` on ``[-1, 1]``; gates are intervals of width ``h`` written as conjunctions of
thermometer bits ``[x > t_lo] AND NOT [x > t_hi]``. Compares the finite-state fit and the
continuous least-squares fit (same gates) against the existence bound ``M h^2 / 8 + delta``.
Writes ``results/phase5_regression.json``.
"""

from __future__ import annotations

import json

import numpy as np

from .data import RESULTS
from .fit import FitConfig, fit_coefficients
from .hybrid import HybridConfig, HybridModel, Layout, scores_direct
from .ridge import ridge_solve

M_CURV = 9.0          # max |f''| of sin(3x)


class IntervalGates:
    kind = "interval"

    def __init__(self, h: float):
        self.t = np.arange(-1.0 + h, 1.0 - 1e-12, h)          # interior thermometer thresholds
        self.K, self.M = 1, len(self.t) + 1
        self.fingerprint = f"interval-h{h}"

    def __call__(self, X, device=None):
        B = X[:, :1] > self.t[None, :]                           # thermometer bits
        lo = np.concatenate([np.ones((len(X), 1), bool), B], 1)  # [x > t_lo]  (first: always)
        hi = np.concatenate([B, np.zeros((len(X), 1), bool)], 1)  # [x > t_hi]  (last: never)
        return lo & ~hi                                          # conjunction of two literals


def run(N: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = lambda x: np.sin(3 * x)  # noqa: E731
    Xtr = rng.uniform(-1, 1, (N, 1))
    Xva = rng.uniform(-1, 1, (500, 1))
    grid = np.linspace(-1, 1, 20001)[:, None]
    out = []
    for h in (1.0, 0.5, 0.25):
        gates = IntervalGates(h)
        lay = Layout(1, gates.M, 1)
        cfg = HybridConfig(delta=1 / 64, Q=256, lam=0.0)
        Gtr, Gva, Gg = gates(Xtr), gates(Xva), gates(grid)
        assert (Gtr.sum(1) == 1).all() and (Gg.sum(1) == 1).all()
        cont = ridge_solve(Xtr, Gtr, f(Xtr), lay, 0.0)["theta"]
        S_c = scores_direct(grid, Gg, lay, cont, 1.0)[:, 0]
        err_c = float(np.abs(S_c - f(grid[:, 0])).max())
        # representability: the continuous fit rounded to the grid (not an optimizer result)
        q_round = np.clip(np.round(cont / cfg.delta), -cfg.Q, cfg.Q).astype(np.int16)
        err_r = float(np.abs(scores_direct(grid, Gg, lay, q_round, cfg.delta)[:, 0] - f(grid[:, 0])).max())
        bound = M_CURV * h * h / 8
        rec = dict(h=h, n_gates=gates.M, D=lay.D, sup_err_continuous=err_c, sup_err_rounded_continuous=err_r,
                   taylor_bound=bound, bound_plus_delta=bound + cfg.delta,
                   continuous_within_bound=err_c <= bound, rounded_within_bound_plus_delta=err_r <= bound + cfg.delta)
        for name, line in (("one_state", False), ("line", True)):
            res = fit_coefficients(HybridModel(gates, lay, cfg), Xtr, f(Xtr), Xva, f(Xva),
                                   FitConfig(max_sweeps=20000, engine="gram", seed=seed, line=line),
                                   Gtr=Gtr, Gva=Gva)
            S_q = scores_direct(grid, Gg, lay, res.terminal_states, cfg.delta)[:, 0]
            err_q = float(np.abs(S_q - f(grid[:, 0])).max())
            rec[name] = dict(sup_err=err_q, gap_to_continuous=err_q - err_c,
                             max_abs_coef_diff=float(np.abs(res.terminal_states * cfg.delta - cont).max()),
                             train_J=res.trace[-1]["J"], stop_reason=res.stop_reason, sweeps=res.terminal_sweep,
                             stationary=res.terminal_audit["stationary_within_tolerance"],
                             within_bound_plus_delta=err_q <= bound + cfg.delta)
        R_c = scores_direct(Xtr, Gtr, lay, cont, 1.0) - f(Xtr)
        rec["continuous_train_J"] = float((R_c ** 2).sum() / (2 * N))
        out.append(rec)
        print(json.dumps(rec))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "phase5_regression.json").write_text(json.dumps(dict(task="sin3x", N=N, seed=seed, rows=out), indent=1))
    return out


if __name__ == "__main__":
    run()
