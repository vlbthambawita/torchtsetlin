"""3D convolutional Tsetlin machines. The test ids (T1...) refer to ``plans/conv3d.md``."""

import itertools

import pytest
import torch

import torchtsetlin as tt


def _rand_volume(shape, device, p=0.5):
    # Built on the device on purpose (see CLAUDE.md: host-built tensors hide async copies).
    return torch.rand(shape, device=device) < p


# ---- T1: geometry ------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape,k,s",
    [((5, 6, 7), (2, 3, 4), (1, 2, 1)), ((4, 4, 4), (4, 4, 4), (1, 1, 1)), ((9, 8, 7), (3, 3, 2), (2, 3, 2))],
)
def test_grid_and_feature_count(shape, k, s):
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=k, stride=s)
    grid = m._grid(*shape)
    assert grid == tuple((n - kk) // ss + 1 for n, kk, ss in zip(shape, k, s))
    Z = 3
    assert m.conv_features((Z, *shape)) == Z * k[0] * k[1] * k[2] + sum(g - 1 for g in grid)
    m_nopos = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=k, stride=s, position_encoding=False)
    assert m_nopos.conv_features((Z, *shape)) == Z * k[0] * k[1] * k[2]


def test_volume_smaller_than_window_raises():
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=(3, 3, 3))
    with pytest.raises(ValueError, match="smaller than patch"):
        m.update(torch.zeros(1, 1, 2, 5, 5, dtype=torch.bool), [0])
    with pytest.raises(ValueError):
        tt.Conv3dTsetlinMachine(2, 4, T=3, input_shape=(1, 4, 4))
    with pytest.raises(ValueError):
        tt.Conv3dTsetlinMachine(2, 4, T=3, patches_per_chunk=0)


# ---- T2: encoder -------------------------------------------------------------------
def test_encode_matches_explicit_slicing(device):
    Z, D, H, W = 2, 5, 6, 7
    k, s = (2, 3, 4), (1, 2, 1)
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=k, stride=s, position_encoding=False).to(device)
    x = _rand_volume((3, Z, D, H, W), device)
    lit = m._encode(m._prepare(x))
    Pd, Ph, Pw = m._grid(D, H, W)
    Fn = Z * k[0] * k[1] * k[2]
    assert lit.shape == (3, Pd * Ph * Pw, 2 * Fn)
    for u, v, w in itertools.product(range(Pd), range(Ph), range(Pw)):
        ref = x[:, :, u * s[0] : u * s[0] + k[0], v * s[1] : v * s[1] + k[1], w * s[2] : w * s[2] + k[2]]
        p = (u * Ph + v) * Pw + w
        assert torch.equal(lit[:, p, :Fn].bool(), ref.reshape(3, -1))
        assert torch.equal(lit[:, p, Fn:].bool(), ~ref.reshape(3, -1))


# ---- T3: exact reduction to 2D --------------------------------------------------------
@pytest.mark.parametrize("position_encoding", [True, False])
def test_depth_one_is_bitwise_the_2d_model(device, position_encoding):
    kw = dict(T=8, s=4.0, position_encoding=position_encoding, n_states=50)
    m2 = tt.ConvTsetlinMachine(3, 6, patch_size=(3, 4), stride=(2, 1), **kw).to(device)
    m3 = tt.Conv3dTsetlinMachine(3, 6, patch_size=(1, 3, 4), stride=(1, 2, 1), **kw).to(device)
    x2 = _rand_volume((40, 2, 9, 8), device, p=0.4)
    x3 = x2.unsqueeze(2)
    y = torch.randint(0, 3, (40,), device=device)

    assert torch.equal(m3._encode(m3._prepare(x3)), m2._encode(m2._prepare(x2)))
    for model, x in ((m2, x2), (m3, x3)):
        torch.manual_seed(123)
        for i in range(0, 40, 8):
            model.update(x[i : i + 8], y[i : i + 8])
    assert torch.equal(m2.ta_state, m3.ta_state)
    # Feature names differ only by the (always 0) depth offset of the voxel.
    names2 = [n.replace(",", ",0,", 1) if n.startswith("p[") else n for n in m2.default_feature_names()]
    assert m3.default_feature_names() == names2


# ---- T4: position bits ---------------------------------------------------------------
def test_position_bits_decode_to_coordinates(device):
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=(2, 2, 3), stride=(1, 2, 1)).to(device)
    Pd, Ph, Pw = m._grid(6, 7, 8)
    bits = m._position_bits(Pd, Ph, Pw, torch.device(device))
    assert bits.shape == (Pd * Ph * Pw, (Pd - 1) + (Ph - 1) + (Pw - 1))
    for p, (u, v, w) in enumerate(itertools.product(range(Pd), range(Ph), range(Pw))):
        row = bits[p]
        bd, bh, bw = row[: Pd - 1], row[Pd - 1 : Pd - 1 + Ph - 1], row[Pd - 1 + Ph - 1 :]
        # A thermometer is a run of ones followed by zeros; its length is the coordinate.
        for b, c in ((bd, u), (bh, v), (bw, w)):
            assert int(b.sum()) == c
            assert bool(b[:c].all()) and not bool(b[c:].any())
    assert m._position_bits(Pd, Ph, Pw, torch.device(device)) is bits  # cached


# ---- T5: position literals express exactly boxes ---------------------------------------
def test_clause_region_round_trip_and_match_box(device):
    shape = (1, 6, 7, 8)
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=(2, 2, 3), input_shape=shape).to(device)
    Pd, Ph, Pw = m.output_grid()
    Fn = m.n_features
    n_vox = 1 * 2 * 2 * 3
    box = {"d": (1, 3), "y": (0, 2), "x": (2, Pw - 1)}
    offsets = {"d": n_vox, "y": n_vox + Pd - 1, "x": n_vox + Pd - 1 + Ph - 1}
    state = torch.full_like(m.ta_state, m.n_states - 1)
    for a, (lo, hi) in box.items():
        n_bits = {"d": Pd, "y": Ph, "x": Pw}[a] - 1
        if lo > 0:
            state[0, offsets[a] + lo - 1] = m.n_states  # coord > lo-1
        if hi < n_bits:
            state[0, Fn + offsets[a] + hi] = m.n_states  # NOT coord > hi
    m.ta_state.copy_(state)
    m._refresh_include()
    assert m.clause_region(0) == box
    m.eval()
    hits = m.patch_clause_outputs(_rand_volume((2, *shape), device))[..., 0]  # (B, Pd, Ph, Pw)
    want = torch.zeros(Pd, Ph, Pw, dtype=torch.bool, device=device)
    want[box["d"][0] : box["d"][1] + 1, box["y"][0] : box["y"][1] + 1, box["x"][0] : box["x"][1] + 1] = True
    assert torch.equal(hits[0], want) and torch.equal(hits[1], want)


# ---- T6: patch-blocked evaluation and draw ---------------------------------------------
@pytest.mark.parametrize("patches_per_chunk", [None, 3])
def test_random_matching_patch_is_uniform(device, patches_per_chunk):
    m = tt.Conv3dTsetlinMachine(
        2, 3, T=3, patch_size=3, position_encoding=False, input_shape=(1, 3, 3, 3),
        patches_per_chunk=patches_per_chunk,
    ).to(device)
    P, N, C, L = 10, 4000, m.n_clauses_total, m.n_literals
    lit = torch.zeros(1, P, L, device=device)
    lit[0, torch.arange(P), torch.arange(P)] = 1.0  # feature p == "this is patch p"
    lit = lit.expand(N, P, L).contiguous()
    matched = [1, 4, 5, 8]
    matches = torch.zeros(N, P, C, dtype=torch.bool, device=device)
    matches[:, matched, 0] = True
    sel = torch.zeros(N, C, device=device)
    sel[:, 0] = 1.0
    acc = m._new_accumulator()
    zero = torch.zeros_like(sel)
    m._feedback_counts(lit, matches, sel, zero, zero, acc)
    counts = acc.n_true[0, :P].cpu()
    assert int(counts.sum()) == N
    others = [p for p in range(P) if p not in matched]
    assert float(counts[others].sum()) == 0.0
    exp = N / len(matched)
    chi2 = float(((counts[matched] - exp) ** 2 / exp).sum())
    assert chi2 < 16.27  # chi^2(3), p = 0.001


def test_blocked_forward_equals_unblocked(device):
    kw = dict(T=10, s=5.0, patch_size=3, n_states=50)
    x, y = tt.data.make_3d_shapes(64, size=(5, 7, 7), seed=0, device=device)
    full = tt.Conv3dTsetlinMachine(2, 8, **kw).to(device)
    for i in range(0, 64, 16):
        full.update(x[i : i + 16], y[i : i + 16])
    blocked = tt.Conv3dTsetlinMachine(2, 8, patches_per_chunk=7, **kw).to(device)
    blocked.load_state_dict(full.state_dict())
    for mode in ("train", "eval"):
        getattr(full, mode)()
        getattr(blocked, mode)()
        assert torch.equal(full(x), blocked(x))
        assert torch.equal(full.patch_clause_outputs(x), blocked.patch_clause_outputs(x))


def test_single_block_path_is_the_unblocked_path(device):
    kw = dict(T=10, s=5.0, patch_size=3, n_states=50)
    x, y = tt.data.make_3d_shapes(48, size=(5, 7, 7), seed=0, device=device)
    states = []
    for ppc in (None, 10**6):
        m = tt.Conv3dTsetlinMachine(2, 8, patches_per_chunk=ppc, **kw).to(device)
        torch.manual_seed(7)
        for i in range(0, 48, 16):
            m.update(x[i : i + 16], y[i : i + 16])
        states.append(m.ta_state.clone())
    assert torch.equal(states[0], states[1])


def test_auto_patch_blocking_under_a_small_budget(device):
    x, y = tt.data.make_3d_shapes(32, size=(6, 8, 8), seed=0, device=device)
    m = tt.Conv3dTsetlinMachine(
        2, 10, T=10, s=5.0, patch_size=3, position_encoding=False, max_chunk_elements=12_000
    ).to(device)
    m.update(x, y)  # lazy init happens here
    P = m._patches_per_example(x)
    assert m._patch_block(P) < P
    assert m._chunk_elements_per_example(x) <= 3 * m._patch_block(P) * m.n_clauses_total + P * m.n_clauses_total // 4 + 2 * m.n_clauses_total * m.n_literals + P * m.n_literals
    ref = tt.Conv3dTsetlinMachine(2, 10, T=10, s=5.0, patch_size=3, position_encoding=False).to(device)
    ref.load_state_dict(m.state_dict())
    ref.eval()
    m.eval()
    assert torch.equal(ref(x), m(x))


# ---- T7: learning ------------------------------------------------------------------------
def test_learns_3d_shapes_where_a_flat_machine_cannot(device):
    size = (8, 12, 12)
    xtr, ytr = tt.data.make_3d_shapes(1000, size=size, seed=0, device=device)
    xte, yte = tt.data.make_3d_shapes(500, size=size, seed=1, device=device)
    kw = dict(T=15, s=10.0, n_states=100)

    def fit(model, a, b):
        tt.seed_everything(0)
        trainer = tt.Trainer(model, device=device, batch_size=16, verbose=False)
        trainer.fit((a, ytr), epochs=8)
        return trainer.test((b, yte))["test_accuracy"]

    acc3d = fit(tt.Conv3dTsetlinMachine(2, 20, patch_size=3, position_encoding=False, **kw), xtr, xte)
    flat = fit(tt.TsetlinMachine(xtr[0].numel(), 2, 20, **kw), xtr.flatten(1), xte.flatten(1))
    assert acc3d >= 0.95
    assert flat < acc3d - 0.1


# ---- T8: API parity ------------------------------------------------------------------------
def test_input_coercion(device):
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=2, input_shape=(1, 4, 5, 5)).to(device)
    v = _rand_volume((4, 5, 5), device)
    assert m(v).shape == (1, 2)  # (D, H, W)
    assert m(v.unsqueeze(0)).shape == (1, 2)  # (Z, D, H, W) == input_shape
    assert m(torch.stack([v, v, v])).shape == (3, 2)  # (B, D, H, W)
    with pytest.raises(ValueError, match="Expected input shape"):
        m(_rand_volume((2, 1, 4, 6, 5), device))
    with pytest.raises(ValueError, match=r"\(B, Z, D, H, W\)"):
        m(torch.zeros(5, 5, dtype=torch.bool, device=device))


def test_empty_clause_semantics_follow_mode(device):
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=2, input_shape=(1, 3, 4, 4)).to(device)
    x = _rand_volume((2, 1, 3, 4, 4), device)
    m.train()
    assert bool(m.patch_clause_outputs(x).all())
    m.eval()
    assert not bool(m.patch_clause_outputs(x).any())
    assert not bool(m.evaluate_clauses(x).any())


def test_state_dict_round_trip_lazy_and_interpretation(device):
    x, y = tt.data.make_3d_shapes(64, size=(5, 7, 7), seed=0, device=device)
    m = tt.Conv3dTsetlinMachine(2, 8, T=10, s=5.0, patch_size=3, n_states=50).to(device)
    m.update(x, y)
    m.eval()
    lazy = tt.Conv3dTsetlinMachine(2, 8, T=10, s=5.0, patch_size=3, n_states=50).to(device)
    lazy.load_state_dict(m.state_dict())
    lazy.eval()
    assert torch.equal(lazy(x), m(x))
    assert m.clause_patch(0).shape == (1, 3, 3, 3)
    assert set(m.clause_region(0)) == {"d", "y", "x"}
    names = m.default_feature_names()
    assert len(names) == m.n_features and names[0] == "p[0,0,0,0]" and "d>0" in names
    exp = tt.interpret.explain(m, x[0])
    assert exp.prediction in (0, 1)
    assert "patch_size=(3, 3, 3)" in repr(m)


def test_coalesced_and_regression_variants(device):
    x, y = tt.data.make_3d_shapes(32, size=(5, 6, 6), seed=0, device=device)
    co = tt.Conv3dCoalescedTsetlinMachine(2, 10, T=8, patch_size=3).to(device)
    assert co.update(x, y).shape == (32, 2)
    reg = tt.Conv3dRegressionTsetlinMachine(10, T=8, patch_size=3, y_range=(0.0, 1.0)).to(device)
    assert reg.update(x, y.float()).shape[0] == 32
    reg.eval()
    assert reg.predict(x).shape[0] == 32


# ---- T9: translation invariance -------------------------------------------------------------
def test_translation_by_stride_multiples_leaves_output_unchanged(device):
    m = tt.Conv3dTsetlinMachine(
        2, 12, T=6, patch_size=3, position_encoding=False, init="random", input_shape=(1, 7, 8, 8)
    ).to(device)
    torch.manual_seed(1)
    m.ta_state.copy_(torch.randint(m.n_states - 2, m.n_states + 2, m.ta_state.shape, device=device))
    m._refresh_include()
    m.eval()
    blob = _rand_volume((1, 3, 4, 4), device)
    a = torch.zeros(1, 1, 7, 8, 8, dtype=torch.bool, device=device)
    b = torch.zeros_like(a)
    a[0, :, 0:3, 1:5, 0:4] = blob
    b[0, :, 3:6, 2:6, 4:8] = blob  # shifted by (3, 1, 4)
    assert torch.equal(m.evaluate_clauses(a), m.evaluate_clauses(b))
    assert torch.equal(m(a), m(b))


def test_plot_3d_clauses():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    m = tt.Conv3dTsetlinMachine(2, 4, T=3, patch_size=(2, 3, 3), input_shape=(1, 4, 5, 5))
    fig = tt.viz.plot_conv_clauses(m, range(4))
    assert len(fig.axes) == 4
    tt.viz.plot_conv_clause(m, 0, depth=1)
