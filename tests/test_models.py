import io

import pytest
import torch

import torchtsetlin as tt
from torchtsetlin.data import make_noisy_xor, make_shapes


def _train_xor(model, device, epochs=40, bs=10, n=2000):
    x, y = make_noisy_xor(n, noise=0.4, seed=0, device=device)
    xt, yt = make_noisy_xor(2000, noise=0.0, seed=1, device=device)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            model.update(x[perm[i : i + bs]], y[perm[i : i + bs]])
    model.eval()
    return tt.metrics.accuracy(model(xt), yt)


def test_construction_and_repr():
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9)
    assert m.ta_state.shape == (40, 24)
    assert "TsetlinMachine" in repr(m)
    assert m.state_summary()["empty_clauses"] == 40  # boundary init -> all excluded
    with pytest.raises(ValueError):
        tt.TsetlinMachine(12, 1, 20, T=15)
    with pytest.raises(ValueError):
        tt.TsetlinMachine(12, 2, 20, T=0)


def test_update_returns_votes_and_shapes(device):
    m = tt.TsetlinMachine(12, 3, 10, T=5).to(device)
    x = torch.randint(0, 2, (7, 12), device=device).bool()
    y = torch.randint(0, 3, (7,), device=device)
    v = m.update(x, y)
    assert v.shape == (7, 3) and v.device.type == device.type
    assert m(x).shape == (7, 3)
    assert m.predict(x).shape == (7,)
    p = m.predict_proba(x)
    assert torch.allclose(p.sum(1), torch.ones(7, device=device), atol=1e-5)
    assert m.confidence(x).shape == (7,)
    assert m.evaluate_clauses(x).shape == (7, 30)


def test_accepts_numpy_and_lists():
    import numpy as np

    m = tt.TsetlinMachine(3, 2, 4, T=2)
    m.update(np.array([[1, 0, 1], [0, 1, 0]]), [0, 1])
    m.update([[True, False, True]], np.array([1]))
    assert m([[1, 0, 1]]).shape == (1, 2)


def test_lazy_init_and_device_move(device):
    m = tt.TsetlinMachine(None, 2, 4, T=2).to(device)
    assert not m.is_initialized
    with pytest.raises(RuntimeError):
        m.state_summary()
    m.update(torch.ones(2, 5, dtype=torch.bool, device=device), torch.tensor([0, 1], device=device))
    assert m.is_initialized and m.n_features == 5 and m.ta_state.device.type == device.type
    with pytest.raises(ValueError):
        m.update(torch.ones(2, 6, dtype=torch.bool, device=device), torch.tensor([0, 1], device=device))


def test_labels_validated():
    m = tt.TsetlinMachine(3, 2, 4, T=2)
    with pytest.raises(ValueError):
        m.update(torch.ones(2, 3, dtype=torch.bool), torch.tensor([0, 2]))
    with pytest.raises(ValueError):
        m.update(torch.ones(2, 3, dtype=torch.bool), torch.tensor([0]))


def test_state_dict_roundtrip(device):
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, weighted=True).to(device)
    x, y = make_noisy_xor(200, device=device)
    m.update(x, y)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = tt.TsetlinMachine(None, 2, 20, T=15, s=3.9, weighted=True)
    m2.load_state_dict(torch.load(buf, map_location="cpu"))
    m.eval()
    m2.eval()
    assert torch.equal(m(x).cpu(), m2(x.cpu()))
    assert torch.equal(m.weights.cpu(), m2.weights)


def test_learns_noisy_xor_batched(device):
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50).to(device)
    acc = _train_xor(m, device, epochs=60, bs=10)
    assert acc > 0.85, acc
    rules = m.rules()
    assert rules and all(r.startswith("IF ") for r in rules)


def test_learns_noisy_xor_sequential():
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50, feedback_mode="sequential")
    acc = _train_xor(m, torch.device("cpu"), epochs=25, bs=50, n=1000)
    assert acc > 0.8, acc


def test_train_eval_mode_empty_clause_semantics():
    m = tt.TsetlinMachine(4, 2, 3, T=2)  # 2 positive + 1 negative clause per class
    x = torch.zeros(1, 4, dtype=torch.bool)
    lit = m._encode(x)
    m.train()
    assert bool(m.clause_outputs(lit).all())  # empty clauses are True while learning
    assert m(x).tolist() == [[1.0, 1.0]]  # 2 - 1 votes per class
    m.eval()
    assert not bool(m.clause_outputs(lit).any())  # ... and False when predicting
    assert bool((m(x) == 0).all())


def test_weighted_and_options(device):
    m = tt.TsetlinMachine(
        12, 2, 20, T=15, s=3.9, weighted=True, max_included_literals=4, drop_clause_p=0.2,
        drop_literal_p=0.1, focused_negative_sampling=True, init="random",
    ).to(device)
    x, y = make_noisy_xor(500, device=device)
    for _ in range(5):
        m.update(x, y)
    assert bool((m.weights >= 0).all())
    assert m.clause_info(0)["weight"] >= 0
    m.drop_granularity = "epoch"
    m.resample_dropout()
    m.update(x, y)


def test_coalesced_multiclass_and_multilabel(device):
    m = tt.CoalescedTsetlinMachine(12, 3, 30, T=10, s=3.9).to(device)
    x = torch.randint(0, 2, (50, 12), device=device).bool()
    y = torch.randint(0, 3, (50,), device=device)
    v = m.update(x, y)
    assert v.shape == (50, 3) and m.predict(x).shape == (50,)
    assert m.rules() is not None
    ml = tt.CoalescedTsetlinMachine(12, 2, 60, T=20, s=3.9, multi_label=True, n_states=50).to(device)
    xm = torch.randint(0, 2, (2000, 12), device=device).bool()
    ym = torch.stack([xm[:, 0] ^ xm[:, 1], xm[:, 2] ^ xm[:, 3]], dim=1)
    for _ in range(10):
        perm = torch.randperm(2000, device=device)
        for i in range(0, 2000, 10):
            ml.update(xm[perm[i : i + 10]], ym[perm[i : i + 10]])
    ml.eval()
    pred = ml.predict(xm)
    assert pred.shape == (2000, 2) and pred.dtype == torch.bool
    assert float((pred == ym).float().mean()) > 0.9
    x = xm[:50]
    assert ml.predict_proba(x).shape == (50, 2)


def test_regression(device):
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 2, (2000, 6), generator=g).bool().to(device)
    y = x[:, :3].float().sum(1) / 3
    m = tt.RegressionTsetlinMachine(6, 100, T=50, s=3.0, y_range=(0, 1), n_states=50).to(device)
    for _ in range(15):
        perm = torch.randperm(2000, device=device)
        for i in range(0, 2000, 20):
            m.update(x[perm[i : i + 20]], y[perm[i : i + 20]])
    m.eval()
    pred = m.predict(x)
    assert pred.shape == (2000,)
    assert float((pred - y).abs().mean()) < 0.1
    assert m.rules()


def test_regression_infers_range_with_warning():
    m = tt.RegressionTsetlinMachine(4, 10, T=10)
    with pytest.warns(UserWarning):
        m.update(torch.ones(3, 4, dtype=torch.bool), torch.tensor([1.0, 3.0, 2.0]))
    assert m.y_min == 1.0 and m.y_max == 3.0


def test_conv_shapes_and_learning(device):
    # Circle-vs-cross is translation invariant: position literals are distractors here, so the
    # learning check runs without them (see docs/concepts/convolution.md).
    x, y = make_shapes(2000, size=8, seed=0, device=device)
    xt, yt = make_shapes(500, size=8, seed=1, device=device)
    m = tt.ConvTsetlinMachine(2, 40, T=20, s=3.9, patch_size=3, n_states=50, position_encoding=False).to(device)
    assert not m.is_initialized
    v = m.update(x[:16], y[:16])
    assert v.shape == (16, 2)
    assert m.n_features == 9 and m.input_shape == (1, 8, 8)
    for _ in range(15):
        perm = torch.randperm(2000, device=device)
        for i in range(0, 2000, 16):
            m.update(x[perm[i : i + 16]], y[perm[i : i + 16]])
    m.eval()
    assert tt.metrics.accuracy(m(xt), yt) > 0.9
    assert m.clause_patch(0).shape == (1, 3, 3)
    assert m(x[0]).shape == (1, 2)  # single image
    with pytest.raises(ValueError):
        m(torch.ones(2, 1, 9, 9, dtype=torch.bool, device=device))


def test_conv_position_encoding_structure(device):
    x, y = make_shapes(64, size=8, seed=0, device=device)
    m = tt.ConvTsetlinMachine(2, 8, T=4, s=3.9, patch_size=3, n_states=50).to(device)
    m.update(x, y)
    assert m.n_features == 9 + 5 + 5
    names = m.default_feature_names()
    assert names[:2] == ["p[0,0,0]", "p[0,0,1]"] and names[9] == "y>0" and names[-1] == "x>4"
    r = m.clause_region(0)
    assert set(r) == {"y", "x"}
    # lo > hi is legal: a contradictory position constraint (the clause never matches)
    assert all(0 <= v <= 5 for v in (*r["y"], *r["x"]))
    # position bits: patch (row 2, col 3) -> y>0,y>1 True ; x>0..x>2 True
    bits = m._position_bits(6, 6, m.device)
    row = bits[2 * 6 + 3].int().tolist()
    assert row == [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]


def test_conv_variants_construct(device):
    x = torch.randint(0, 2, (8, 2, 6, 6), device=device).bool()
    c = tt.ConvCoalescedTsetlinMachine(3, 12, T=5, patch_size=2, stride=2).to(device)
    assert c.update(x, torch.randint(0, 3, (8,), device=device)).shape == (8, 3)
    r = tt.ConvRegressionTsetlinMachine(12, T=5, patch_size=2, y_range=(0, 1)).to(device)
    assert r.update(x, torch.rand(8, device=device)).shape == (8, 1)
    s = tt.Conv1dTsetlinMachine(2, 8, T=4, kernel_size=3).to(device)
    seq = torch.randint(0, 2, (5, 10), device=device).bool()
    assert s.update(seq, torch.randint(0, 2, (5,), device=device)).shape == (5, 2)
    assert s.input_shape == (1, 1, 10)
    nopos = tt.ConvTsetlinMachine(2, 4, T=2, patch_size=2, position_encoding=False, input_shape=(2, 6, 6))
    assert nopos.n_features == 8


def test_chunking_matches_unchunked():
    torch.manual_seed(0)
    x, y = make_noisy_xor(300)
    m1 = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9)
    m2 = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, max_chunk_elements=2000)
    m2.load_state_dict(m1.state_dict())
    assert torch.equal(m1(x), m2(x))
    assert m2._chunk_size(x) < 300


def test_parameters_warns():
    m = tt.TsetlinMachine(4, 2, 4, T=2)
    with pytest.warns(UserWarning):
        assert list(m.parameters()) == []
