"""Phases 3–4 / gates G3, G4: loss-change identity, descent, fixtures, stop reporting, integrity,
split isolation, engine equivalence, ridge."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.linearModel.code import adapter as A
from experiments.linearModel.code import data as D
from experiments.linearModel.code import hybrid as H
from experiments.linearModel.code import spec_literal as SL
from experiments.linearModel.code.fit import FitConfig, IntegrityError, fit_coefficients, gradient
from experiments.linearModel.code.opt_gram import build_gram, design_rows
from experiments.linearModel.code.ridge import ridge_solve

from .conftest import random_problem


class ArrayGates:
    """Fixed gate matrix as a gate set (tests only)."""
    kind = "array"

    def __init__(self, Xref, G, K, M):
        self.Xref, self.G, self.K, self.M = Xref, G, K, M
        self.fingerprint = "array"

    def __call__(self, X, device=None):
        assert X is self.Xref
        return self.G


def dense_A(X, G, lay):
    Gk = H.gates_by_output(G, lay)
    return [design_rows(X, Gk[k], lay) for k in range(lay.K)]


def test_loss_change_identity():
    rng = np.random.default_rng(0)
    for lam in (0.0, 0.3):
        X, G, Y, _, lay = random_problem(rng, N=9, d=3, K=2, M=2)
        delta, Q = 0.25, 5
        q = rng.integers(-4, 5, (lay.K, lay.D)).astype(np.int16)
        R = H.scores_direct(X, G, lay, q, delta) - Y
        J0 = H.objective(R, q, delta, lam)
        g = gradient(X, H.gates_by_output(G, lay), R, q, lay, delta, lam)
        h = H.column_sq_norms(X, H.gates_by_output(G, lay), lay) / 9 + lam
        for k in range(lay.K):
            for r in range(lay.D):
                for m in (-1, 1):
                    if abs(int(q[k, r]) + m) > Q:
                        continue
                    q2 = q.copy()
                    q2[k, r] += m
                    J1 = H.objective(H.scores_direct(X, G, lay, q2, delta) - Y, q2, delta, lam)
                    s = delta * m
                    assert (J1 - J0) == pytest.approx(s * g[k, r] + s * s / 2 * h[k, r], abs=1e-12)


def test_simple_fixture_reaches_2():
    A1 = [np.array([[1.0], [2.0]])]
    Y = np.array([[2.0], [4.0]])
    q, log, stop = SL.fit(A1, Y, 0.5, 4, 0.0, 1e-12, [[np.array([0])]] * 20)
    assert q[0][0] * 0.5 == 2.0 and stop == "NO_ACCEPTED_MOVES"


def test_coordinate_trap():
    A2 = [np.array([[1.0, 1.0], [0.0, 0.1]])]
    Y = np.array([[0.0], [0.1]])
    q, log, stop = SL.fit(A2, Y, 1.0, 2, 0.0, 1e-12, [[np.array([0, 1])]] * 5)
    assert q[0].tolist() == [0, 0] and stop == "NO_ACCEPTED_MOVES"
    au = SL.audit(A2, Y, q, 1.0, 2, 0.0, 1e-12)
    assert au["stationary_within_tolerance"]
    assert SL.objective(A2, Y, [np.array([-1, 1])], 1.0, 0.0) < SL.objective(A2, Y, q, 1.0, 0.0)


def _fit(X, G, Y, y, lay, delta, Q, lam, engine, sweeps, seed=0, line=False, tol=1e-12):
    model = H.HybridModel(ArrayGates(X, G, lay.K, lay.M), lay, H.HybridConfig(delta, Q, lam))
    return fit_coefficients(model, X, Y, X, y, FitConfig(max_sweeps=sweeps, accept_tol=tol, seed=seed,
                                                          engine=engine, line=line),
                            Gtr=G, Gva=G, record_moves=True)


def _moves(res):
    out = []
    for perms, qb, qa in res.moves:
        sweep = []
        for k in range(perms.shape[0]):
            # every coordinate is visited once per sweep, so the changed states in perm order
            # are the accepted moves in order
            for r in perms[k]:
                if qa[k, r] != qb[k, r]:
                    sweep.append((k, int(r), int(qa[k, r]) - int(qb[k, r])))
        out.append(sweep)
    return out


@pytest.mark.parametrize("line", [False, True])
def test_engines_match_spec_literal(line):
    rng = np.random.default_rng(5)
    for trial in range(6):
        X, G, Y, y, lay = random_problem(rng, N=30, d=3, K=2, M=2)
        delta, Q, lam = 0.125, 6, (0.0, 1e-2)[trial % 2]
        res = {e: _fit(X, G, Y, y, lay, delta, Q, lam, e, 15, seed=trial, line=line) for e in ("ref", "gram")}
        perms = [[p[k] for k in range(lay.K)] for p, _, _ in res["ref"].moves]
        q, log, stop = SL.fit(dense_A(X, G, lay), Y, delta, Q, lam, 1e-12, perms, line=line)
        assert sum(map(len, log)) > 10
        assert _moves(res["ref"]) == log
        assert _moves(res["gram"]) == log
        assert np.array_equal(res["ref"].terminal_states, np.stack(q))
        au = SL.audit(dense_A(X, G, lay), Y, q, delta, Q, lam, 1e-12)
        assert au["improving_coordinate_count"] == res["ref"].terminal_audit["improving_coordinate_count"]


def test_residual_maintenance_and_descent():
    rng = np.random.default_rng(6)
    X, G, Y, y, lay = random_problem(rng, N=60, d=4, K=3, M=3)
    res = _fit(X, G, Y, y, lay, 1 / 16, 64, 1e-3, "ref", 30)
    for a, b in zip(res.trace, res.trace[1:]):
        assert (b["J"] < a["J"]) if b["accepted"] else (b["J"] == pytest.approx(a["J"], abs=1e-15))
    assert sum(t["accepted"] for t in res.trace) > 50
    assert max(t["drift"] for t in res.trace) <= 1e-12
    res_g = _fit(X, G, Y, y, lay, 1 / 16, 64, 1e-3, "gram", 30)
    assert max(t["drift"] for t in res_g.trace) <= 1e-12


def test_stop_reporting():
    rng = np.random.default_rng(7)
    X, G, Y, y, lay = random_problem(rng, N=50, d=3, K=2, M=2)
    r = _fit(X, G, Y, y, lay, 1 / 64, 256, 1e-3, "ref", 2)
    assert r.stop_reason == "SWEEP_BUDGET" and not r.terminal_audit["stationary_within_tolerance"]
    assert set(r.selected_audit) == set(r.terminal_audit)
    r = _fit(X, G, Y, y, lay, 0.25, 8, 1e-3, "ref", 500)
    assert r.stop_reason == "NO_ACCEPTED_MOVES" and r.terminal_audit["stationary_within_tolerance"]


def test_bounds_never_exceeded():
    rng = np.random.default_rng(8)
    X, G, Y, y, lay = random_problem(rng, N=50, d=3, K=2, M=2)
    Y = Y * 50.0                               # pushes coefficients to the bound
    for e in ("ref", "gram"):
        for line in (False, True):
            r = _fit(X, G, Y, y, lay, 0.25, 3, 0.0, e, 50, line=line)
            assert np.abs(r.terminal_states.astype(int)).max() == 3


def test_fingerprint_mismatch_aborts():
    rng = np.random.default_rng(9)
    X, G, Y, y, lay = random_problem(rng, N=50, d=3, K=2, M=2)
    G = G.copy()
    model = H.HybridModel(ArrayGates(X, G, lay.K, lay.M), lay, H.HybridConfig(1 / 64, 256, 1e-3))

    def mutate(_):
        G[0, 0] = not G[0, 0]
    with pytest.raises(IntegrityError):
        fit_coefficients(model, X, Y, X, y, FitConfig(max_sweeps=3), Gtr=G, Gva=G, progress=mutate)


def test_split_isolation():
    with pytest.raises(PermissionError):
        D.load_split("split_seed0", "test")
    tr = D.read_manifest("split_seed0")
    assert not set(tr["train"]) & set(tr["valid"])


def test_ridge_vs_lstsq():
    rng = np.random.default_rng(10)
    for lam in (1e-3, 0.1):
        X, G, Y, _, lay = random_problem(rng, N=40, d=3, K=2, M=2)
        sol = ridge_solve(X, G, Y, lay, lam, with_mu=True)
        for k, Ak in enumerate(dense_A(X, G, lay)):
            N, Dk = Ak.shape
            Aaug = np.vstack([Ak / np.sqrt(N), np.sqrt(lam) * np.eye(Dk)])
            yaug = np.concatenate([Y[:, k] / np.sqrt(N), np.zeros(Dk)])
            ref = np.linalg.lstsq(Aaug, yaug, rcond=None)[0]
            assert np.allclose(sol["theta"][k], ref, atol=1e-9)
            assert sol["mu"][k] == pytest.approx(np.linalg.eigvalsh(Ak.T @ Ak / N + lam * np.eye(Dk)).min(), rel=1e-8)


def test_gram_matches_dense():
    rng = np.random.default_rng(11)
    X, G, Y, _, lay = random_problem(rng, N=70, d=4, K=2, M=3)
    for cb, sl in ((True, True), (True, False), (False, True)):
        lay = H.Layout(2, 3, 4, cb, sl)
        Gk = H.gates_by_output(G, lay)
        Hd = build_gram(X, Gk, lay, chunk=16, blocked=False)
        Hb = build_gram(X, Gk, lay)
        for k, Ak in enumerate(dense_A(X, G, lay)):
            assert np.allclose(Hd[k], Ak.T @ Ak / 70, atol=1e-13)
            assert np.allclose(Hb[k], Ak.T @ Ak / 70, atol=1e-13)
            assert np.array_equal(Hb[k], Hb[k].T)


def test_hybrid_fit_on_real_gates(small_tm, integ):
    """End-to-end smoke on MNIST integration data with real frozen rules (both engines)."""
    tr, va = integ
    rules = A.freeze_rules(small_tm, 0.5, tr.X)
    lay = H.Layout(10, rules.M, 784)
    out = {}
    for e in ("ref", "gram"):
        m = H.HybridModel(rules, lay, H.HybridConfig(1 / 64, 256, 1e-3))
        out[e] = fit_coefficients(m, tr.X[:500], tr.Y[:500], va.X[:300], va.y[:300],
                                  FitConfig(max_sweeps=3, engine=e, seed=1))
    assert out["ref"].trace[-1]["J"] < out["ref"].trace[0]["J"]
    assert out["ref"].trace[-1]["J"] == pytest.approx(out["gram"].trace[-1]["J"], rel=1e-9)
