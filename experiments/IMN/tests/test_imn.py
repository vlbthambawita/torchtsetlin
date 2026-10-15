"""Correctness gates for the IMN x TM programme (run: ``pytest experiments/IMN/tests -q``)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from experiments.IMN.code import models as M
from experiments.IMN.code import train_nn
from experiments.IMN.code.arms import run_arm
from experiments.IMN.code.booleanize import TabularBooleanizer
from experiments.IMN.code.data import synthetic_splits
from experiments.IMN.code.explain import Explainable, local_fidelity, removal_attributions
from experiments.IMN.code.tm_source import basis_from_tm, random_basis, tm_config, train_tm

import torchtsetlin as tt

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _toy(n=1200, d=5, n_classes=2, seed=0):
    g = np.random.default_rng(seed)
    X = g.normal(size=(n, d))
    if n_classes == 2:
        y = ((X[:, 0] > 0) ^ (X[:, 1] > 0.3)).astype(np.int64)
    else:
        y = (np.digitize(X[:, 0] + 0.5 * X[:, 1], [-0.7, 0.7])).astype(np.int64)
    return synthetic_splits(X, y, f"toy{n_classes}", seed)


@pytest.fixture(scope="module", params=[2, 3])
def trained(request):
    S = _toy(n_classes=request.param)
    cfg = tm_config({"n_clauses": 20, "epochs": 5, "patience": 5, "n_bits": 6})
    tm, boo, meta = train_tm(S, cfg, seed=0, device="cpu", log=lambda *_: None)
    return S, tm, boo


def test_booleanizer_bits_and_binary_columns():
    X = np.stack([np.arange(100.0), np.r_[np.zeros(50), np.ones(50)]], 1)
    b = TabularBooleanizer(4).fit(X)
    assert (b.col == 1).sum() == 1 and np.isclose(b.thr[b.col == 1][0], 0.5)
    xb = b.transform(X).numpy()
    c0 = xb[:, b.col == 0]
    assert (np.diff(c0.astype(int), axis=0) >= 0).all()  # thermometer: monotone in the value
    assert (c0[:, :-1] >= c0[:, 1:]).all()                # and ordered across thresholds


def test_preprocessing_is_fitted_on_train_only():
    S = _toy()
    assert np.allclose(S.X_train.mean(0), 0, atol=1e-5)
    assert not np.allclose(S.X_test.mean(0), 0, atol=1e-5)


def test_clause_basis_reproduces_tm_votes(trained):
    S, tm, boo = trained
    basis = basis_from_tm(tm, boo, S.n_classes)
    X = torch.as_tensor(S.X_test)
    xb = boo.transform(X)
    with torch.no_grad():
        v = tm._votes(tm.evaluate_clauses(xb), clamp=False)
    got = basis.evaluate(X) @ basis.tm_weight
    want = (v[:, 1:2] - v[:, :1]) if S.n_classes == 2 else v
    assert torch.allclose(got, want)
    assert basis.K <= tm.n_clauses_total and bool(basis.include.any(1).all())


@pytest.mark.parametrize("arm", ["imn_tm", "imn_hybrid", "clause_lr"])
def test_tm_initialised_models_start_as_the_tm(trained, arm):
    S, tm, boo = trained
    basis = basis_from_tm(tm, boo, S.n_classes)
    n_out = 1 if S.n_classes == 2 else S.n_classes
    model = M.build(arm, S.n_features, n_out, K=basis.K, clause_init=basis.tm_weight / basis.T).eval()
    X = torch.as_tensor(S.X_test)
    C = basis.evaluate(X)
    z, _ = model(X, C)
    assert torch.allclose(z, (C @ basis.tm_weight) / basis.T, atol=1e-5)


def test_imn_removal_attribution_equals_w_times_x():
    S = _toy()
    model = M.build("imn", S.n_features, 1).eval()
    X = torch.as_tensor(S.X_test[:64])
    z, parts = model(X)
    W = parts["w"]
    ex = Explainable(full=lambda Xq: model(Xq)[0],
                     local=lambda Xm, rows: torch.einsum("bp,bpk->bk", model.local_features(Xm, None), W[rows]))
    phi = removal_attributions(ex, X)
    assert np.allclose(phi, (W[:, :-1, 0] * X).detach().numpy(), atol=1e-5)


def test_random_basis_fires_and_matches_lengths(trained):
    S, tm, boo = trained
    basis = basis_from_tm(tm, boo, S.n_classes)
    rb = random_basis(basis, S.X_train, seed=1)
    assert torch.equal(rb.lengths(), basis.lengths().clamp_max(boo.n_bool))
    fires = rb.evaluate(torch.as_tensor(S.X_train)).sum(0)
    assert bool((fires > 0).all())
    inc = rb.include
    F = boo.n_bool
    assert not bool((inc[:, :F] & inc[:, F:]).any())  # no contradictory clause


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arm", ["imn", "imn_tm", "imn_hybrid", "tm_gated", "clause_lr", "tabresnet"])
def test_arm_end_to_end_learns_xor(arm, device, tmp_path, monkeypatch):
    from experiments.IMN.code import tm_source
    monkeypatch.setattr(tm_source, "CKPT_DIR", tmp_path)
    S = _toy()
    res = run_arm(arm, S, 0, device, nn_over={"epochs": 30, "batch_size": 128},
                  tm_over={"n_clauses": 20, "epochs": 10, "n_bits": 6}, log=lambda *_: None)
    from experiments.IMN.code.metrics import classification_metrics
    m = classification_metrics(S.y_test, res.proba["test"], 2)
    assert m["auroc"] > 0.8, (arm, m)          # XOR: a plain linear model sits near 0.5
    if res.explain.local is not None:
        X = torch.as_tensor(S.X_test[:50], device=device)
        rows = torch.arange(50, device=device)
        # the local model of each example reproduces the full model's output at that example
        assert torch.allclose(res.explain.local(X, rows), res.explain.full(X), atol=1e-4)
        fid = local_fidelity(res.explain, X, S.y_test[:50], k_per_class=(5,), n_anchor=10)
        assert 0.0 <= fid["local_acc_k5"] <= 1.0


def test_tm_arm_and_cache_roundtrip(tmp_path, monkeypatch):
    from experiments.IMN.code import tm_source
    monkeypatch.setattr(tm_source, "CKPT_DIR", tmp_path)
    S = _toy()
    over = {"n_clauses": 20, "epochs": 5, "n_bits": 6}
    r1 = run_arm("tm", S, 0, "cpu", tm_over=over, log=lambda *_: None)
    r2 = run_arm("tm", S, 0, "cpu", tm_over=over, log=lambda *_: None)  # from cache
    assert np.allclose(r1.proba["test"], r2.proba["test"])
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_lr_schedule_snapshots_match_official_formula():
    S = _toy(n=400)
    model = M.build("imn", S.n_features, 1)
    b = train_nn.Batches(torch.as_tensor(S.X_train), torch.as_tensor(S.y_train), None)
    out = train_nn.fit(model, b, train_nn.nn_config({"epochs": 14, "batch_size": 64}), log=lambda *_: None)
    steps = out["steps"]
    assert out["T0"] == steps // 7 and len(out["snapshots"]) == 3


def test_library_untouched():
    import subprocess
    from pathlib import Path
    root = Path(tt.__file__).resolve().parents[2]
    diff = subprocess.run(["git", "diff", "--stat", "HEAD", "--", "src/"], cwd=root,
                          capture_output=True, text=True).stdout
    assert diff.strip() == "", "C1: src/torchtsetlin must stay read-only in this programme"


def test_firing_matched_random_basis(trained):
    from experiments.IMN.code.tm_source import random_basis_firing_matched
    S, tm, boo = trained
    basis = basis_from_tm(tm, boo, S.n_classes)
    rb = random_basis_firing_matched(basis, S.X_train, seed=0)
    X = torch.as_tensor(S.X_train)
    r_learned = basis.evaluate(X).mean(0)
    r_rand = rb.evaluate(X).mean(0)
    assert rb.K == basis.K
    assert float((r_rand - r_learned).abs().mean()) < 0.1  # rates matched up to one literal step
    F = boo.n_bool
    assert not bool((rb.include[:, :F] & rb.include[:, F:]).any())


def test_split_l1_penalty_reduces_to_single_weight():
    S = _toy(n=200)
    model = M.build("imn_hybrid", S.n_features, 1, K=3)
    x = torch.as_tensor(S.X_train[:8])
    c = torch.ones(8, 3)
    _, parts = model(x, c)
    a = train_nn._l1(model, parts, x, c, "all", 0.3)
    b = train_nn._l1(model, parts, x, c, "all", 0.3, 0.3)
    assert torch.allclose(a, b)
    assert not torch.allclose(a, train_nn._l1(model, parts, x, c, "all", 0.3, 3.0))


@pytest.mark.parametrize("arm", ["imn_tm_xc", "imn_tm_randfr"])
def test_ablation_arms_run(arm, tmp_path, monkeypatch):
    from experiments.IMN.code import tm_source
    monkeypatch.setattr(tm_source, "CKPT_DIR", tmp_path)
    S = _toy()
    res = run_arm(arm, S, 0, "cpu", nn_over={"epochs": 20, "batch_size": 128},
                  tm_over={"n_clauses": 20, "epochs": 10, "n_bits": 6}, log=lambda *_: None)
    X = torch.as_tensor(S.X_test[:20])
    assert torch.allclose(res.explain.local(X, torch.arange(20)), res.explain.full(X), atol=1e-4)


def test_oversized_model_is_recorded_as_skipped(tmp_path, monkeypatch):
    import json

    from experiments.IMN.code import arms, run
    monkeypatch.setattr(arms, "MAX_PARAMS", 10)
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "load_splits", lambda d, s: _toy(n=200))
    p = run.run_one("t", "x", "imn", 0, "cpu", {"epochs": 1}, {}, log=lambda *_: None)
    assert json.loads(p.read_text())["status"] == "skipped"


# ------------------------------------------------------------------------- MNIST study
def test_fixed_threshold_booleanizer_skips_constant_and_one_sided_columns():
    X = np.array([[0.0, 0.0, 0.1], [0.0, 0.5, 0.2], [0.0, 1.0, 0.25]], dtype=np.float32)
    boo = TabularBooleanizer(10, thresholds=[0.3]).fit(X)
    assert boo.col.tolist() == [1] and boo.thr.tolist() == pytest.approx([0.3])
    st = TabularBooleanizer.from_state(boo.state())
    assert st.thresholds == [0.3] and torch.equal(st.transform(X), boo.transform(X))


def test_clause_credit_sums_to_clause_logit(trained):
    from experiments.IMN.code.mnist import clause_credit
    S, tm, boo = trained
    basis = basis_from_tm(tm, boo, S.n_classes)
    X = torch.as_tensor(S.X_test[:50])
    Wk = torch.randn(50, basis.K)
    psi = clause_credit(basis, X, Wk)
    assert torch.allclose(psi.sum(1), (basis.evaluate(X) * Wk).sum(1), atol=1e-4)


def test_deletion_curves_start_at_the_prediction():
    from experiments.IMN.code.mnist import FRACS, deletion
    g = torch.Generator().manual_seed(0)
    X = torch.rand(30, 12, generator=g) * (torch.rand(30, 12, generator=g) > 0.5)
    Wl = torch.randn(12, 4, generator=g)
    full = lambda x: x @ Wl  # noqa: E731
    tgt = full(X).argmax(1)
    cur, auc, ret, rauc = deletion(full, X, torch.randn(30, 12, generator=g), tgt)
    assert ret[:, 0].min() == 1 and cur.shape == (30, len(FRACS))
    # everything deleted -> all-zero input, same logits for every image
    assert np.allclose(cur[:, -1], torch.softmax(full(torch.zeros(1, 12)), 1)[0, tgt].numpy())
