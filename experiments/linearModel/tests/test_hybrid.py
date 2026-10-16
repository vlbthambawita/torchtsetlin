"""Phase 2 / gate G2: state decoding, contributions, prediction paths, persistence."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.linearModel.code import adapter as A
from experiments.linearModel.code import hybrid as H
from experiments.linearModel.code.opt_ref import best_move

from .conftest import random_problem


def _rules(include, threshold, K=1, M=None):
    include = np.asarray(include, bool)
    M = include.shape[0] // K if M is None else M
    return A.FrozenRules(K=K, M=M, n_features=include.shape[1] // 2, threshold=threshold, include=include,
                         empty_clause_output=np.zeros(include.shape[0], np.uint8),
                         polarity=np.ones(include.shape[0], np.int8), weight=None, source_identity="fixture")


def test_worked_example_12_1():
    inc = np.zeros((1, 6), bool)
    inc[0, [0, 1, 3 + 2]] = True                    # x1 > .5 AND x2 > .5 AND NOT(x3 > .5)
    lay = H.Layout(1, 1, 3)
    cfg = H.HybridConfig(delta=0.1, Q=100, lam=0.0)
    q = lay.join(np.array([0]), np.array([[-1]]), np.array([[[6, 3, -2]]])).astype(np.int16)
    m = H.HybridModel(_rules(inc, 0.5), lay, cfg, q)
    x = np.array([[0.9, 0.8, 0.1]])
    assert m.predict_scores(x)[0, 0] == pytest.approx(0.66, abs=1e-12)
    assert m.predict_scores(np.array([[0.6, 0.6, 0.4]]))[0, 0] == pytest.approx(0.36, abs=1e-12)
    # the one-state move on the second slope: DeltaJ = -0.024 (N = 1, lambda = 0)
    g, h = 0.8 * (0.66 - 1), 0.8 ** 2
    mv, change = best_move(3, g, h, 0.1, 100, False)
    assert mv == 1 and change == pytest.approx(-0.024, abs=1e-12)
    q2 = q.copy()
    q2[0, 1 + 1 + 1] += 1
    m2 = H.HybridModel(m.gates, lay, cfg, q2)
    assert m2.predict_scores(x)[0, 0] == pytest.approx(0.74, abs=1e-12)
    assert 0.5 * (0.74 - 1) ** 2 - 0.5 * (0.66 - 1) ** 2 == pytest.approx(change, abs=1e-12)


def test_regression_fixture_12_2():
    inc = np.array([[False, True], [True, False]])  # C1 = NOT[x > 0], C2 = [x > 0]
    lay = H.Layout(1, 2, 1)
    q = lay.join(np.array([0]), np.array([[2, 2]]), np.array([[[4], [-1]]])).astype(np.int16)
    m = H.HybridModel(_rules(inc, 0.0), lay, H.HybridConfig(0.5, 8, 0.0), q)
    out = m.predict_scores(np.array([[-0.5], [0.0], [0.5]]))[:, 0]
    assert np.allclose(out, [0.0, 1.0, 0.75], atol=1e-12)


def test_state_decoding_and_bounds():
    lay = H.Layout(1, 1, 2)
    cfg = H.HybridConfig(0.25, 3, 0.0)
    gates = _rules(np.zeros((1, 4), bool), 0.5)
    for q in (-3, 0, 2):
        st = np.full((1, lay.D), q, np.int16)
        assert np.all(H.HybridModel(gates, lay, cfg, st).states * cfg.delta == q * 0.25)
    with pytest.raises(ValueError):
        H.HybridModel(gates, lay, cfg, np.full((1, lay.D), 4, np.int16))
    # outward move at a bound is unavailable, inward is allowed
    assert best_move(3, -10.0, 1.0, 0.25, 3, False)[0] == 0
    assert best_move(-3, 10.0, 1.0, 0.25, 3, False)[0] == 0
    assert best_move(3, 10.0, 1.0, 0.25, 3, False)[0] == -1
    assert best_move(3, -10.0, 1.0, 0.25, 3, True)[0] == 0
    assert best_move(32767, -10.0, 1.0, 0.25, 32767, False)[0] == 0      # no int16 wraparound


def test_inactive_overlap_no_active():
    rng = np.random.default_rng(1)
    X = rng.random((5, 2))
    G = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0]], bool)
    lay = H.Layout(1, 2, 2)
    q = rng.integers(-5, 6, (1, lay.D)).astype(np.int16)
    S = H.scores_direct(X, G, lay, q, 0.5)[:, 0]
    beta, bias, W = lay.split(q * 0.5)
    c = [bias[0, j] + X @ W[0, j] for j in range(2)]
    assert S[0] == pytest.approx(beta[0]) and S[4] == pytest.approx(beta[0])      # no active clause
    assert S[1] == pytest.approx(beta[0] + c[0][1])                               # inactive -> 0
    assert S[3] == pytest.approx(beta[0] + c[0][3] + c[1][3])                     # overlap adds


def test_no_clauses_in_bank():
    lay = H.Layout(3, 0, 4)
    gates = A.FrozenRules(K=3, M=0, n_features=4, threshold=0.5, include=np.zeros((0, 8), bool),
                          empty_clause_output=np.zeros(0, np.uint8), polarity=np.zeros(0, np.int8),
                          weight=None, source_identity="empty")
    q = np.array([[1], [-2], [3]], np.int16)
    m = H.HybridModel(gates, lay, H.HybridConfig(0.5, 4, 0.0), q)
    S = m.predict_scores(np.random.default_rng(0).random((6, 4)))
    assert np.allclose(S, [0.5, -1.0, 1.5])


@pytest.mark.parametrize("cb,sl", [(True, True), (True, False), (False, True)])
def test_direct_vs_column(cb, sl):
    rng = np.random.default_rng(2)
    for _ in range(100):
        X, G, _, _, lay = random_problem(rng, N=12, d=3, K=3, M=2, clause_bias=cb, slopes=sl)
        q = rng.integers(-7, 8, (lay.K, lay.D)).astype(np.int16)
        assert np.abs(H.scores_direct(X, G, lay, q, 1 / 64) - H.scores_from_states(X, G, lay, q, 1 / 64)).max() <= 1e-12


def test_design_T_and_norms_match_columns():
    rng = np.random.default_rng(3)
    X, G, Y, _, lay = random_problem(rng, N=15, d=3, K=2, M=3)
    Gk = H.gates_by_output(G, lay)
    V = rng.standard_normal((15, 2))
    AtV, nrm = H.design_T(X, Gk, V, lay), H.column_sq_norms(X, Gk, lay)
    for k in range(2):
        for r in range(lay.D):
            a = H.design_column(X, G, lay, k, r)
            assert AtV[k, r] == pytest.approx(a @ V[:, k], abs=1e-12)
            assert nrm[k, r] == pytest.approx(a @ a, abs=1e-12)


def test_argmax_ties_lowest_index():
    S = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    assert S.argmax(1).tolist() == [0, 1, 0]
    assert H.validation_key(S, np.array([0, 1, 0]))[0] == -1.0


def test_save_load_fidelity(tmp_path, small_tm, small_conv, integ):
    tr, va = integ
    rng = np.random.default_rng(4)
    rules = A.freeze_rules(small_tm, 0.5, tr.X[:200])
    for gates in (rules, A.LibraryGates(small_conv, 0.5), A.ConstantGate(10)):
        lay = H.Layout(10, gates.M, 784, clause_bias=not isinstance(gates, A.ConstantGate))
        q = rng.integers(-256, 257, (lay.K, lay.D)).astype(np.int16)
        m = H.HybridModel(gates, lay, H.HybridConfig(1 / 64, 256, 1e-3), q, metadata={"arm": "t"})
        p = tmp_path / f"{gates.kind}.npz"
        nbytes = m.save(p)
        assert nbytes == p.stat().st_size
        m2 = H.HybridModel.load(p)
        assert np.array_equal(m2.states, m.states) and m2.states.dtype == np.int16
        assert m2.gates.fingerprint == m.gates.fingerprint and m2.checksum() == m.checksum()
        assert np.array_equal(m2.gate_values(va.X), m.gate_values(va.X))
        assert np.array_equal(m2.predict_scores(va.X), m.predict_scores(va.X))   # bit-identical


def test_random_rules_save_load(tmp_path, small_tm, integ):
    from experiments.linearModel.code.gates_random import random_rules_like
    tr, va = integ
    src = A.freeze_rules(small_tm, 0.5, tr.X)
    rnd = random_rules_like(src, tr.X, seed=0, device="cpu")
    assert (rnd.include.sum(1) == src.include.sum(1)).all()
    rep = rnd.match_report                                  # D12: rates must actually be matched
    assert abs(rep["achieved_mean_rate"] - rep["target_mean_rate"]) <= 0.2 * rep["target_mean_rate"]
    lay = H.Layout(10, rnd.M, 784)
    m = H.HybridModel(rnd, lay, H.HybridConfig(1 / 64, 256, 1e-3))
    m.save(tmp_path / "r.npz")
    m2 = H.HybridModel.load(tmp_path / "r.npz")
    assert m2.gates.kind == "random_rules" and np.array_equal(m2.gate_values(va.X), m.gate_values(va.X))
