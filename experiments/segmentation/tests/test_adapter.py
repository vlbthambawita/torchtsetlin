"""Identity adapter parity, sampler validity and ordering checks (PLAN H1)."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import tm  # noqa: E402
import torchtsetlin as tt  # noqa: E402


def _data(n=6, size=16):
    x, y = tt.data.make_segmentation_shapes(n, size=size, seed=3)
    return x, y


def _fit(model, x, y, epochs=2):
    for _ in range(epochs):
        for i in range(0, len(x), 3):
            model.update(x[i : i + 3], y[i : i + 3])
    return model


def test_library_order_is_bit_identical():
    x, y = _data()
    kw = dict(n_clauses=40, T=10, s=4.0, patch_size=3, patches_per_commit=32)
    tt.seed_everything(0)
    ref = _fit(tt.SegmentationTsetlinMachine(3, **kw), x, y)
    tt.seed_everything(0)
    m = tm.ExpSegTM(3, **kw)
    m.init_order("library")
    _fit(m, x, y)
    assert torch.equal(ref.ta_state, m.ta_state)


def test_raster_order_matches_library_without_void():
    # with no void pixels the valid-raster order is the library order
    x, y = _data()
    kw = dict(n_clauses=40, T=10, s=4.0, patch_size=3, patches_per_commit=32)
    tt.seed_everything(0)
    ref = _fit(tt.SegmentationTsetlinMachine(3, **kw), x, y)
    tt.seed_everything(0)
    m = tm.ExpSegTM(3, **kw)
    m.init_order("raster")
    _fit(m, x, y)
    assert torch.equal(ref.ta_state, m.ta_state)


def test_orders_are_permutations_of_valid_pixels():
    _, y = _data(4)
    y = y.clone()
    y[:, 0, :] = 255
    m = tm.ExpSegTM(3, n_clauses=10, T=5, patch_size=3)
    for order in ("raster", "perm", "disperse"):
        m.init_order(order)
        idx = m._order_indices(y)
        valid = torch.nonzero((y != 255).reshape(-1)).squeeze(1)
        assert torch.equal(idx.sort().values, valid), order
    m.init_order("disperse")
    idx = m._order_indices(y)
    # consecutive pixels in the dispersed order are far apart or in different images
    P = 16 * 16
    a, b = idx[:-1], idx[1:]
    same_img = (a // P) == (b // P)
    dist = ((a % P) // 16 - (b % P) // 16).abs().maximum(((a % P) % 16 - (b % P) % 16).abs())
    assert bool(((~same_img) | (dist >= 8)).float().mean() > 0.95)


def test_sampler_never_draws_void_and_respects_mixture():
    _, y = _data(8, 32)
    y = y.clone()
    y[:, :4, :] = 255
    g = torch.Generator().manual_seed(0)
    idx = tm.sample_pixels(y, 4000, 0.5, 0.0, g, 3)
    flat = y.view(8, -1)
    lab = torch.gather(flat, 1, idx)
    assert not bool((lab == 255).any())
    frac = torch.stack([(lab == k).float().mean() for k in range(3)])
    assert frac[1] > 0.5 / 3 and frac[2] > 0.5 / 3  # a/|present| floor per class
    band = tm.boundary_band(y)
    idx = tm.sample_pixels(y, 2000, 0.0, 1.0, g, 3)
    assert bool(torch.gather(band.view(8, -1), 1, idx).all())


def test_coarse_labels_and_oof_folds_exclude_heldout():
    import hierarchy

    y = torch.zeros(2, 8, 8, dtype=torch.uint8)
    y[0, :4, :4] = 1
    y[0, 0, 0] = 255
    y[1] = 255
    c = hierarchy.coarse_labels(y, 4, 2)
    assert c[0, 0, 0] == 1 and c[0, 1, 1] == 0 and bool((c[1] == 255).all())
    groups = ["a", "a", "b", "c", "c", "d", "e", "f"]
    fold = hierarchy._folds(groups, 3, 0)
    for g in set(groups):  # a group never straddles folds
        assert len({fold[i] for i, gg in enumerate(groups) if gg == g}) == 1


def test_gathered_path_equals_full_encoding():
    torch.manual_seed(0)
    x = torch.rand(3, 5, 9, 11) > 0.5
    y = torch.randint(0, 3, (3, 9, 11))
    y[0, 0, :] = 255
    for order in ("perm", "disperse"):
        tt.seed_everything(1)
        a = tm.ExpSegTM(3, n_clauses=20, T=5, patch_size=1, patches_per_commit=7, ignore_index=255)
        a.init_order(order, seed=3)
        a.max_chunk_elements = 2**40
        a.update(x, y)
        tt.seed_everything(1)
        b = tm.ExpSegTM(3, n_clauses=20, T=5, patch_size=1, patches_per_commit=7, ignore_index=255)
        b.init_order(order, seed=3)
        b.max_chunk_elements = 2**40
        b.gather = True
        b.update(x, y)
        assert torch.equal(a.ta_state, b.ta_state), order
        xb = b._prepare(x)
        idx = torch.arange(3 * 99)[torch.randperm(3 * 99)[:50]]
        assert torch.equal(b._gathered_literals(xb, idx), b._encode(xb)[idx])
