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


def _ref_patch_feats(img, kh, kw, sh, sw, Py, Px):
    """Chapter 4 sec 4.2/4.5: patch pixels (z, row, col) then the position thermometers."""
    Z = img.shape[0]
    out = []
    for y in range(Py):
        for x in range(Px):
            feats = [bool(img[z, y * sh + i, x * sw + j])
                     for z in range(Z) for i in range(kh) for j in range(kw)]
            out.append(feats + [y >= k for k in range(1, Py)] + [x >= k for k in range(1, Px)])
    return out


def _ref_match(feats, include, Fn, empty_value):
    if not include.any():
        return empty_value
    return not any((include[k] and not feats[k]) or (include[Fn + k] and feats[k])
                   for k in range(Fn))


def test_conv_evaluation_is_or_over_patches(device):
    """A clause is True for the image iff it matches at least one patch (Ch. 4 sec 4.3)."""
    m = tt.ConvTsetlinMachine(2, 6, T=4, s=3.9, patch_size=(3, 2), stride=(2, 1),
                              n_states=8, input_shape=(2, 7, 6)).to(device)
    m.ta_state.copy_(torch.randint(0, 2 * m.n_states, m.ta_state.shape,
                                   dtype=m.ta_state.dtype, device=device))
    m.ta_state[torch.rand(m.ta_state.shape, device=device) < 0.9] = m.n_states - 1
    m._refresh_include()
    x = torch.randint(0, 2, (5, 2, 7, 6), device=device).bool()
    Py, Px = m._grid(7, 6)
    assert m.n_features == 2 * 3 * 2 + (Py - 1) + (Px - 1)
    literals = m._encode(x)
    inc, Fn = m.included_mask(), int(m.n_features)
    for empty_value in (True, False):
        out, _ = m._evaluate(literals, empty_value)  # the fused kernel when predicting on CUDA
        dense, matches = m._evaluate_dense(literals, empty_value)
        assert torch.equal(out, dense)
        assert matches.shape == (5, Py * Px, m.n_clauses_total)
        for b in range(5):
            ref = _ref_patch_feats(x[b], 3, 2, 2, 1, Py, Px)
            # patch features and their negations
            for p, feats in enumerate(ref):
                assert literals[b, p, :Fn].bool().tolist() == feats
                assert literals[b, p, Fn:].bool().tolist() == [not f for f in feats]
            for j in range(m.n_clauses_total):
                want = any(_ref_match(f, inc[j], Fn, empty_value) for f in ref)
                assert bool(out[b, j]) is want


def test_conv_feedback_uses_a_random_matching_patch(device):
    """Ch. 4 sec 4.4: Recognize/Reject update from a uniformly random *matching* patch;
    Erase needs no patch at all."""
    from torchtsetlin.models.base import FeedbackAccumulator

    m = tt.ConvTsetlinMachine(2, 6, T=4, s=3.9, patch_size=3, n_states=8,
                              input_shape=(1, 6, 6)).to(device)
    m.ta_state.copy_(torch.randint(0, 2 * m.n_states, m.ta_state.shape,
                                   dtype=m.ta_state.dtype, device=device))
    m.ta_state[torch.rand(m.ta_state.shape, device=device) < 0.92] = m.n_states - 1
    m._refresh_include()
    x = torch.randint(0, 2, (7, 1, 6, 6), device=device).bool()
    literals = m._encode(x)
    clause_out, matches = m._evaluate(literals, True)
    C, L, cd = m.n_clauses_total, m.n_literals, m.compute_dtype
    fire = clause_out.to(cd)
    nofire = (~clause_out).to(cd)
    inc = m.included_mask().to(cd)
    for _ in range(20):
        acc = FeedbackAccumulator(C, L, x.device, cd)
        m._feedback_counts(literals, matches, fire, nofire, fire, acc)
        # a matching patch makes every *included* literal True, so none may be counted False
        assert float((acc.n_false * inc).sum()) == 0.0
        assert float((acc.n2 * inc).sum()) == 0.0
        # one Type Ia event per matching clause, one Type Ib event per non-matching one
        assert torch.equal(acc.n_true + acc.n_false, fire.sum(0).unsqueeze(1).expand(C, L))
        assert torch.equal(acc.n_ib, nofire.sum(0))

    # the draw is uniform over the matching patches of the clause
    P = matches.shape[1]
    b, j, mp = next(
        (b, j, mp)
        for b in range(7)
        for j in range(C)
        for mp in [[p for p in range(P) if matches[b, p, j]]]
        if len(mp) >= 3 and len({tuple(literals[b, p].tolist()) for p in mp}) == len(mp)
    )
    sel = torch.zeros(7, C, dtype=cd, device=x.device)
    sel[b, j] = 1
    zero = torch.zeros(7, C, dtype=cd, device=x.device)
    counts = dict.fromkeys(mp, 0)
    n_draws = 600
    for _ in range(n_draws):
        acc = FeedbackAccumulator(C, L, x.device, cd)
        m._feedback_counts(literals, matches, sel, zero, zero, acc)
        used = [p for p in mp if torch.equal(literals[b, p], acc.n_true[j])]
        assert len(used) == 1, "feedback used a patch the clause does not match"
        counts[used[0]] += 1
    assert all(c > n_draws / (4 * len(mp)) for c in counts.values()), counts


def test_conv_prediction_draws_no_random_patches(device):
    """Only learning picks a patch, so prediction must be deterministic and RNG-free
    (drawing per prediction cost ~5x on MNIST-sized inputs)."""
    m = tt.ConvTsetlinMachine(3, 20, T=8, s=3.9, patch_size=3, input_shape=(1, 8, 8)).to(device)
    x = torch.randint(0, 2, (16, 1, 8, 8), device=device).bool()
    m.update(x, torch.randint(0, 3, (16,), device=device))
    m.eval()
    rng_state = (
        (lambda: torch.cuda.get_rng_state(device)) if device.type == "cuda"
        else torch.random.get_rng_state
    )
    before = rng_state()
    first = m(x)
    assert torch.equal(rng_state(), before), "forward() consumed randomness"
    assert torch.equal(m(x), first)
    before = rng_state()
    assert torch.equal(m.evaluate_clauses(x), m.evaluate_clauses(x))
    assert torch.equal(rng_state(), before), "evaluate_clauses() consumed randomness"


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


def test_batched_classifier_respects_max_included_literals(device):
    """Regression (issue #1): under budget_mode="hard" the budget holds for batched feedback."""
    x = torch.rand(2000, 40, device=device) > 0.5
    y = (x[:, 0] ^ x[:, 1]).long()
    m = tt.TsetlinMachine(40, 2, 100, T=15, s=10.0, max_included_literals=6, budget_mode="hard").to(device)
    for _ in range(3):
        for i in range(0, 2000, 50):
            m.update(x[i:i + 50], y[i:i + 50])
    assert int(m.include_count.max()) <= 6


def _trained_small_T(device, cls=None, **kw):
    """A classifier with T far below its vote capacity, so class sums saturate."""
    cls = cls or tt.TsetlinMachine
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(1000, noise=0.1)
    m = cls(n_features=12, n_classes=2, n_clauses=40, T=2, s=3.9, **kw).to(device)
    for _ in range(5):
        for i in range(0, 1000, 25):
            m.update(x[i:i + 25].to(device), y[i:i + 25].to(device))
    m.eval()
    return m, x.to(device), y.to(device)


def test_prediction_votes_are_not_clamped(device):
    # Learning clamps the vote sum to [-T, T]; prediction must not (MC-1): with every winning
    # class saturated at +T, the argmax would be decided by class index instead of by the vote.
    m, x, _ = _trained_small_T(device)
    votes = m(x)
    assert float(votes.abs().max()) > m.T
    with torch.no_grad():
        xb = m._prepare(x)
        out, _ = m._evaluate(m._encode(xb), empty_value=False)
        clamped = m._votes(out, clamp=True)
    assert torch.equal(votes.clamp(-m.T, m.T), clamped)
    assert torch.equal(m.predict(x), votes.argmax(dim=1))


def test_predict_proba_and_confidence_still_use_the_vote_margin(device):
    m, x, _ = _trained_small_T(device)
    v = m(x).clamp(-m.T, m.T)
    p = 0.5 * (1.0 + v / m.T)
    assert torch.allclose(m.predict_proba(x), p / p.sum(dim=1, keepdim=True).clamp_min(1e-12))
    assert float(m.confidence(x).max()) <= 1.0


def test_coalesced_prediction_votes_are_not_clamped(device):
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(1000, noise=0.1)
    m = tt.CoalescedTsetlinMachine(n_features=12, n_outputs=2, n_clauses=40, T=2, s=3.9).to(device)
    for _ in range(5):
        for i in range(0, 1000, 25):
            m.update(x[i:i + 25].to(device), y[i:i + 25].to(device))
    m.eval()
    votes = m(x.to(device))
    assert float(votes.abs().max()) > m.T
    probs = m.predict_proba(x.to(device))
    assert float(probs.min()) >= 0.0


def test_regression_prediction_stays_in_range(device):
    torch.manual_seed(0)
    x = torch.randint(0, 2, (500, 8))
    y = x[:, :3].sum(dim=1).float()
    m = tt.RegressionTsetlinMachine(n_features=8, n_clauses=40, T=5, s=3.0, y_range=(0.0, 3.0)).to(device)
    for _ in range(3):
        m.update(x.to(device), y.to(device))
    m.eval()
    votes = m(x.to(device))
    assert float(votes.min()) >= 0.0 and float(votes.max()) <= m.T
    pred = m.predict(x.to(device))
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 3.0


def test_clause_weights_never_drop_below_one(device):
    # Reference implementations (pyTsetlinMachine, tmu) keep integer clause weights >= 1.
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(2000, noise=0.4, device=device)
    clf = tt.TsetlinMachine(n_features=12, n_classes=2, n_clauses=20, T=15, s=3.9,
                            weighted=True).to(device)
    reg = tt.RegressionTsetlinMachine(n_features=12, n_clauses=20, T=15, s=3.9, weighted=True,
                                      y_range=(0.0, 1.0)).to(device)
    for _ in range(20):
        clf.update(x, y)
        reg.update(x, y.float())
    assert int(clf.weights.min()) >= 1
    assert int(reg.weights.min()) >= 1


def test_regression_feedback_probability_is_squared_error():
    # pyTsetlinMachine / tmu: feedback with probability ((v* - v) / T)**2.
    torch.manual_seed(0)
    m = tt.RegressionTsetlinMachine(n_features=4, n_clauses=20_000, T=10, s=3.9, y_range=(0.0, 1.0))
    votes = torch.tensor([[2.0], [8.0]])
    y = torch.tensor([0.7, 0.3])  # v* = 7 and 3: errors +0.5 (Type I) and -0.5 (Type II)
    type_i, type_ii, _ = m._select_feedback(votes, y, None)
    assert abs(type_i[0].float().mean().item() - 0.25) < 0.02
    assert abs(type_ii[1].float().mean().item() - 0.25) < 0.02
    assert not bool(type_i[1].any()) and not bool(type_ii[0].any())


@pytest.mark.parametrize("mode", ["batch", "sequential"])
def test_include_caches_track_the_automata(device, mode):
    # the sparse commit refreshes include / include_count incrementally; they must always
    # equal a full recomputation from ta_state
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(400, noise=0.2, device=device)
    for m in (tt.TsetlinMachine(12, 2, 10, 15, 3.9, feedback_mode=mode, max_included_literals=3),
              tt.CoalescedTsetlinMachine(12, 2, 20, 15, 3.9, feedback_mode=mode),
              tt.ConvTsetlinMachine(2, 10, 15, 3.9, patch_size=2, input_shape=(1, 3, 4),
                                    feedback_mode=mode)):
        m = m.to(device)
        xin = x.view(-1, 1, 3, 4) if isinstance(m, tt.ConvTsetlinMachine) else x
        for i in range(0, 400, 50):
            m.update(xin[i:i + 50], y[i:i + 50])
            inc = m.ta_state >= m.n_states
            assert torch.equal(m.include, inc.to(m.include.dtype))
            assert torch.equal(m.include_count, inc.sum(dim=1).to(m.include_count.dtype))


def test_accumulator_reuse_recovers_after_a_failed_update(device):
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(200, device=device)
    a = tt.TsetlinMachine(12, 2, 10, 15, 3.9).to(device)
    b = tt.TsetlinMachine(12, 2, 10, 15, 3.9).to(device)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(1)
    a.update(x[:50], y[:50])
    torch.manual_seed(1)
    b.update(x[:50], y[:50])
    assert torch.equal(a.ta_state, b.ta_state)
    real = a._feedback_counts

    def boom(*args):
        real(*args)  # counts land in the reused buffers, then the update fails
        raise RuntimeError("boom")

    a._feedback_counts = boom
    with pytest.raises(RuntimeError):
        a.update(x[50:100], y[50:100])
    del a._feedback_counts
    torch.manual_seed(5)
    a.update(x[100:150], y[100:150])
    torch.manual_seed(5)
    b.update(x[100:150], y[100:150])
    assert torch.equal(a.ta_state, b.ta_state)


def test_max_weight_caps_clause_weights(device):
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(2000, noise=0.0, device=device)
    clf = tt.TsetlinMachine(12, 2, 10, 15, 3.9, weighted=True, max_weight=3).to(device)
    co = tt.CoalescedTsetlinMachine(12, 2, 20, 15, 3.9, max_weight=2).to(device)
    for _ in range(10):
        for i in range(0, 2000, 50):  # one 2000-example batch learns nothing (batch fidelity)
            clf.update(x[i:i + 50], y[i:i + 50])
            co.update(x[i:i + 50], y[i:i + 50])
    assert 1 <= int(clf.weights.min()) and int(clf.weights.max()) <= 3
    assert int(clf.weights.max()) == 3  # the cap is reached on clean XOR
    assert int(co.weights.abs().max()) <= 2
    with pytest.raises(ValueError):
        tt.TsetlinMachine(12, 2, 10, 15, 3.9, weighted=True, max_weight=0)


def test_clamp_prediction_votes_constructor_flag(device):
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(500, device=device)
    m = tt.TsetlinMachine(12, 2, 10, 2, 3.9, clamp_prediction_votes=True).to(device)
    for _ in range(5):
        m.update(x, y)
    m.eval()
    assert float(m(x).abs().max()) <= 2.0
    assert tt.TsetlinMachine.clamp_prediction_votes is False  # class default untouched
    assert tt.TsetlinMachine(12, 2, 10, 2, 3.9).clamp_prediction_votes is False


def test_default_state_dtype_is_8_bit(device):
    m = tt.TsetlinMachine(12, 2, 10, 15, 3.9).to(device)
    assert m.ta_state.dtype == torch.uint8
    assert tt.TsetlinMachine(12, 2, 10, 15, 3.9, n_states=1000).ta_state.dtype == torch.int16
    # an int32 checkpoint (0.4.0 and earlier) loads into the 8-bit model unchanged
    old = tt.TsetlinMachine(12, 2, 10, 15, 3.9, state_dtype=torch.int32).to(device)
    x, y = tt.data.make_noisy_xor(500, device=device)
    old.update(x, y)
    m.load_state_dict(old.state_dict())
    assert torch.equal(m.ta_state.long(), old.ta_state.long())
    assert torch.equal(m.include, old.include)


def test_learns_noisy_xor_with_compiled_cpu_kernels(compiled):
    # G8: the CPU commit through the torch.compile kernels learns and keeps the caches exact
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50, max_included_literals=6)
    acc = _train_xor(m, torch.device("cpu"), epochs=40, bs=10)
    assert compiled._forget_rows_kernel in compiled._KERNELS
    assert acc > 0.85, acc
    inc = m.ta_state >= m.n_states
    assert torch.equal(m.include, inc.to(m.include.dtype))
    assert torch.equal(m.include_count, inc.sum(dim=1).to(m.include_count.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused sequential mode is CUDA-only")
def test_fused_sequential_mode_learns_and_tracks_caches(compiled, monkeypatch):
    # G9: sequential mode on CUDA commits each example with one fused kernel
    dev = torch.device("cuda")
    calls = []
    real = compiled.commit_single_example
    monkeypatch.setattr(compiled, "commit_single_example",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    torch.manual_seed(0)
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50, feedback_mode="sequential",
                          weighted=True, max_included_literals=6).to(dev)
    m.cuda_graphs = False  # count the Python-level calls (a CUDA graph replays without them)
    assert m._fuse_single_examples()
    acc = _train_xor(m, dev, epochs=10, bs=100, n=1000)
    assert acc > 0.8, acc
    assert len(calls) == 10 * 1000
    ml = tt.CoalescedTsetlinMachine(12, 2, 60, T=20, s=3.9, multi_label=True, n_states=50,
                                    feedback_mode="sequential").to(dev)
    xm = torch.randint(0, 2, (2000, 12), device=dev).bool()
    ym = torch.stack([xm[:, 0] ^ xm[:, 1], xm[:, 2] ^ xm[:, 3]], dim=1)
    for _ in range(5):
        ml.update(xm, ym)
    ml.eval()
    assert float((ml.predict(xm) == ym).float().mean()) > 0.9
    for model in (m, ml):
        inc = model.ta_state >= model.n_states
        assert torch.equal(model.include, inc.to(model.include.dtype))
        assert torch.equal(model.include_count, inc.sum(dim=1).to(model.include_count.dtype))
    x = torch.randint(0, 2, (500, 6)).bool().to(dev)
    y = x[:, :3].float().sum(1) / 3
    r = tt.RegressionTsetlinMachine(6, 100, T=50, s=3.0, y_range=(0, 1), n_states=50,
                                    feedback_mode="sequential").to(dev)
    for _ in range(4):
        r.update(x, y)
    r.eval()
    assert float((r.predict(x) - y).abs().mean()) < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only include dtype")
def test_cuda_include_cache_is_half_precision_and_predicts_like_cpu():
    torch.manual_seed(0)
    x, y = tt.data.make_noisy_xor(1000, noise=0.1)
    for m in (tt.TsetlinMachine(12, 2, 20, 15, 3.9),
              tt.ConvTsetlinMachine(2, 10, 15, 3.9, patch_size=2, input_shape=(1, 3, 4))):
        xin = x.view(-1, 1, 3, 4) if isinstance(m, tt.ConvTsetlinMachine) else x
        for i in range(0, 1000, 50):
            m.update(xin[i:i + 50], y[i:i + 50])
        m.eval()
        ref = m(xin)
        assert m.include.dtype == m.compute_dtype
        m.cuda()
        assert m.include.dtype == torch.float16
        assert torch.equal(m(xin.cuda()).cpu(), ref)
        m.cpu()
        assert m.include.dtype == m.compute_dtype


def test_conv_batch_is_evaluated_in_one_chunk():
    # the chunk size follows the largest real intermediate (the (P, C) block); counting the
    # worst-case patch draw split a batch of 16 MNIST images into 4 chunks
    m = tt.ConvTsetlinMachine(10, 2000, 5000, 5.0, patch_size=10, input_shape=(1, 28, 28))
    xb = torch.zeros(16, 1, 28, 28, dtype=torch.bool)
    assert m._chunk_size(xb) == 16
    assert m._event_block(361) * (2 * 361 + 2 * m.n_literals) <= m.max_chunk_elements


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused commits are CUDA-only")
@pytest.mark.parametrize("triton_path", [True, False])
def test_fused_cuda_commits_learn_and_track_caches(compiled, monkeypatch, triton_path):
    # batched CUDA updates commit through the Triton kernel (or, without it, F.commit_dense);
    # sequential conv models through F.commit_single_example with one literal row per clause
    from torchtsetlin import _triton

    if triton_path and not _triton.available(torch.device("cuda")):
        pytest.skip("Triton is unavailable")
    if not triton_path:
        monkeypatch.setattr(_triton, "available", lambda device: False)
    dev = torch.device("cuda")
    fused, dense, single = [], [], []
    real_fused, real_dense = _triton.fused_commit, compiled.commit_dense
    real_single = compiled.commit_single_example
    monkeypatch.setattr(_triton, "fused_commit", lambda *a, **k: fused.append(1) or real_fused(*a, **k))
    monkeypatch.setattr(compiled, "commit_dense", lambda *a, **k: dense.append(1) or real_dense(*a, **k))
    monkeypatch.setattr(compiled, "commit_single_example",
                        lambda *a, **k: single.append(1) or real_single(*a, **k))
    torch.manual_seed(0)
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50, max_included_literals=6).to(dev)
    m.cuda_graphs = False  # count the Python-level calls (a CUDA graph replays without them)
    acc = _train_xor(m, dev, epochs=40, bs=10)  # eager and fused: ~0.93 over 6 seeds
    assert acc > 0.85, acc
    assert (fused and not dense) if triton_path else (dense and not fused)
    x, y = tt.data.make_2d_noisy_xor(2000, device=dev)
    c = tt.ConvTsetlinMachine(2, 20, 30, 3.9, patch_size=2, position_encoding=False,
                              input_shape=tuple(x.shape[1:]), feedback_mode="sequential").to(dev)
    c.cuda_graphs = False
    assert c._fuse_single_examples(len(x))
    for _ in range(10):
        c.update(x, y)
    assert len(single) == 10 * len(x)
    c.eval()
    assert float((c.predict(x) == y).float().mean()) > 0.9
    for model in (m, c):
        inc = model.ta_state >= model.n_states
        assert torch.equal(model.include, inc.to(model.include.dtype))
        assert torch.equal(model.include_count, inc.sum(dim=1).to(model.include_count.dtype))
