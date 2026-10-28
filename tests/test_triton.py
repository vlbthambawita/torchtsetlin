"""The Triton kernels (fused commit, fused convolutional evaluation) and CUDA-graph updates.

Each kernel is checked against the eager code it replaces: exactly where the math is exact
(clause evaluation, caches), in distribution where it samples (commit, patch draw).
"""

import pytest
import torch

import torchtsetlin as tt
from torchtsetlin import _triton as K
from torchtsetlin import functional as F

CUDA = torch.device("cuda")
needs_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and K.available(CUDA)), reason="needs CUDA and Triton"
)


def test_pack_bits_layout():
    x = torch.zeros(2, 40, dtype=torch.bool)
    x[0, [0, 31, 32, 39]] = True
    w = K.pack_bits(x)
    assert w.shape == (2, 2) and w.dtype == torch.int32
    assert w[0, 0].item() == 1 - 2**31 and w[0, 1].item() == 1 + 2**7
    assert w[1].tolist() == [0, 0]


@needs_triton
def test_conv_eval_is_exact_and_draws_matching_patches():
    g = torch.Generator(device=CUDA).manual_seed(0)
    B, P, L, C = 7, 50, 70, 300
    lit = (torch.rand(B, P, L, device=CUDA, generator=g) < 0.5).float()
    inc = (torch.rand(C, L, device=CUDA, generator=g) < 0.03).float()
    inc[:20] = 0  # empty clauses match every patch while learning
    ref = ((1 - lit).reshape(B * P, L) @ inc.T == 0).view(B, P, C)
    for draw in (False, True):
        out, p_i, p_ii = K.conv_eval(K.pack_bits(lit.reshape(B * P, L) == 0), K.pack_bits(inc), B, P, draw)
        assert torch.equal(out, ref.any(1))
    b = torch.arange(B, device=CUDA)[:, None].expand(B, C)
    c = torch.arange(C, device=CUDA)[None].expand(B, C)
    assert bool(ref[b, p_i.long(), c][out].all()) and bool(ref[b, p_ii.long(), c][out].all())


@needs_triton
def test_conv_patch_draws_are_uniform_and_independent():
    lit = torch.ones(8, 4, device=CUDA)
    lit[:, 0] = 0
    lit[[1, 3, 4, 6, 7], 0] = 1  # the clause (literal 0) matches 5 of 8 patches
    inc = torch.zeros(1, 4, device=CUDA)
    inc[0, 0] = 1
    neg, incb = K.pack_bits(lit == 0), K.pack_bits(inc)
    n = 4000
    joint = torch.zeros(8, 8, device=CUDA)
    for _ in range(n):
        _, a, b = K.conv_eval(neg, incb, 1, 8, draw=True)
        joint[a.long(), b.long()] += 1
    for marg in (joint.sum(1), joint.sum(0)):
        freq = marg / n
        assert float(freq[[0, 2, 5]].sum()) == 0.0  # never a non-matching patch
        assert float((freq[[1, 3, 4, 6, 7]] - 0.2).abs().max()) < 0.03
    cells = joint[[1, 3, 4, 6, 7]][:, [1, 3, 4, 6, 7]] / n  # Type Ia and II drawn independently
    assert float((cells - 0.04).abs().max()) < 0.015


def _events(conv, g, B=24, C=40, L=70, P=5):
    lit = (torch.rand(B, P, L, device=CUDA, generator=g) < 0.5).float()
    sel = torch.rand(B, C, device=CUDA, generator=g)
    fire = torch.rand(B, C, device=CUDA, generator=g) < 0.6
    si, sib = (fire & (sel < 0.4)).float(), (~fire & (sel < 0.4)).float()
    sii = (fire & (sel > 0.7)).float()
    p_i = torch.randint(0, P, (B, C), device=CUDA, dtype=torch.int32, generator=g)
    p_ii = torch.randint(0, P, (B, C), device=CUDA, dtype=torch.int32, generator=g)
    ar = torch.arange(B, device=CUDA)[:, None]
    r_i, r_ii = lit[ar, p_i.long()], lit[ar, p_ii.long()]  # (B, C, L) rows of each event
    if not conv:
        lit, r_i, r_ii = lit[:, :1], lit[:, :1].expand(B, C, L), lit[:, :1].expand(B, C, L)
    counts = ((si[:, :, None] * r_i).sum(0), (si[:, :, None] * (1 - r_i)).sum(0), sib.sum(0),
              (sii[:, :, None] * (1 - r_ii)).sum(0))
    lit_in = lit if conv else lit[:, 0]
    return lit_in, si, sib, sii, (p_i if conv else None), (p_ii if conv else None), counts


@needs_triton
@pytest.mark.parametrize("conv", [False, True])
@pytest.mark.parametrize("boost,budget,dropout,s", [(True, None, False, 4.0), (False, 3, True, 4.0),
                                                    (False, None, False, 1.5)])
def test_fused_commit_matches_apply_feedback(conv, boost, budget, dropout, s):
    g = torch.Generator(device=CUDA).manual_seed(1)
    N, reps = 16, 300
    lit, si, sib, sii, p_i, p_ii, (nt, nf, nib, n2) = _events(conv, g)
    C, L = nt.shape
    state0 = torch.randint(N - 4, N + 4, (C, L), device=CUDA, dtype=torch.uint8, generator=g)
    act = (torch.rand(L, device=CUDA, generator=g) < 0.8).float() if dropout else None
    count0 = (state0 >= N).sum(1)
    kw = dict(n_states=N, s=s, boost_true_positive=boost, max_included_literals=budget, literal_active=act)
    da, db = [], []
    for _ in range(reps):
        a = state0.clone()
        F.apply_feedback(a, nt.clone(), nf.clone(), nib.clone(), n2.clone(), include_count=count0,
                         generator=g, **kw)
        b = state0.clone()
        inc = torch.zeros(C, L, device=CUDA, dtype=torch.half)
        bits = torch.zeros(C, K.n_words(L), dtype=torch.int32, device=CUDA)
        cnt = count0.clone()
        assert K.fused_commit(b, inc, bits, cnt, lit, si, sii, p_i, p_ii, si.sum(0), nib,
                              max_events=si.shape[0], **kw)
        assert torch.equal(inc, (b >= N).half()) and torch.equal(cnt, (b >= N).sum(1))
        assert torch.equal(bits, K.pack_bits(b >= N))
        da.append((a.int() - state0.int()).float())
        db.append((b.int() - state0.int()).float())
    da, db = torch.stack(da), torch.stack(db)
    se = ((da.var(0) + db.var(0)) / reps).sqrt().clamp(min=1e-3)
    assert ((da.mean(0) - db.mean(0)).abs() / se).max().item() < 5.5
    assert abs(da.mean().item() - db.mean().item()) < 0.03
    assert abs(da.var(0).mean().item() - db.var(0).mean().item()) < 0.05 * da.var(0).mean().item() + 0.02


@needs_triton
def test_fused_commit_declines_counts_beyond_its_table():
    state = torch.zeros(4, 6, dtype=torch.uint8, device=CUDA)
    z = torch.zeros(3, 4, device=CUDA)
    assert not K.fused_commit(state, state.half(), torch.zeros(4, 1, dtype=torch.int32, device=CUDA),
                              torch.zeros(4, dtype=torch.long, device=CUDA), torch.zeros(3, 6, device=CUDA),
                              z, z, None, None, z[0], z[0], n_states=8, s=4.0,
                              boost_true_positive=True, max_included_literals=None,
                              literal_active=None, max_events=K.TABLE_N)


def _models():
    return [
        ("flat", tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True, max_included_literals=6), False),
        ("coalesced", tt.CoalescedTsetlinMachine(12, 2, 30, 15, 3.9, focused_negative_sampling=True), False),
        ("regression", tt.RegressionTsetlinMachine(12, 40, 20, 3.0, y_range=(0, 1)), False),
        ("conv", tt.ConvTsetlinMachine(2, 20, 30, 3.9, patch_size=2, input_shape=(1, 3, 4)), True),
    ]


@needs_triton
@pytest.mark.parametrize("seq", [False, True])
def test_updates_never_synchronise_with_the_host(compiled, seq):
    # (the `compiled` fixture makes test-sized models take the fused sequential path, which is
    # the sync-free one; small models otherwise use the eager row-sparse commit)
    x, y = tt.data.make_noisy_xor(500, noise=0.1, device=CUDA)
    for name, m, conv in _models():
        m = m.to(CUDA)
        m.cuda_graphs = False
        m.feedback_mode = "sequential" if seq else "batch"
        xin = x.view(-1, 1, 3, 4) if conv else x
        yin = y.float() if name == "regression" else y
        for i in range(0, 200, 50):  # compile / warm up
            m.update(xin[i:i + 50], yin[i:i + 50])
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            m.update(xin[200:250], yin[200:250])
        finally:
            torch.cuda.set_sync_debug_mode(0)


@needs_triton
@pytest.mark.parametrize("seq", [False, True])
def test_cuda_graph_updates_learn_and_draw_fresh_randomness(seq):
    x, y = tt.data.make_noisy_xor(3000, noise=0.2, device=CUDA)
    xt, yt = tt.data.make_noisy_xor(1000, noise=0.0, seed=1, device=CUDA)
    torch.manual_seed(0)
    m = tt.TsetlinMachine(12, 2, 20, 15, 3.9, feedback_mode="sequential" if seq else "batch").to(CUDA)
    bs = 500 if seq else 50
    for _ in range(8):
        for i in range(0, 3000, bs):
            m.update(x[i:i + bs], y[i:i + bs])
    graphs = [v for v in m.__dict__["_graphs"].values() if not isinstance(v, str)]
    assert graphs, "no update was replayed from a CUDA graph"
    m.eval()
    assert float((m.predict(xt) == yt).float().mean()) > 0.9
    m.train()
    # replaying the same update from the same state twice must not repeat the random draws
    state = m.ta_state.clone()
    m.update(x[:bs], y[:bs])
    first = m.ta_state.clone()
    m.ta_state.copy_(state)
    m.include.copy_((state >= m.n_states).to(m.include.dtype))
    m.include_count.copy_((state >= m.n_states).sum(1))
    m.update(x[:bs], y[:bs])
    assert not torch.equal(first, m.ta_state)
    inc = m.ta_state >= m.n_states
    assert torch.equal(m.include, inc.to(m.include.dtype))
    assert torch.equal(m.include_bits(), K.pack_bits(inc))


@needs_triton
def test_cuda_graphs_are_recaptured_when_the_model_changes():
    x, y = tt.data.make_noisy_xor(400, device=CUDA)
    m = tt.TsetlinMachine(12, 2, 20, 15, 3.9).to(CUDA)
    for _ in range(5):
        m.update(x[:50], y[:50])
    keys = set(m.__dict__["_graphs"])
    m.s = 5.0  # a Python constant inside the graph
    m.update(x[:50], y[:50])
    assert set(m.__dict__["_graphs"]) - keys
    keys = set(m.__dict__["_graphs"])
    other = tt.TsetlinMachine(12, 2, 20, 15, 3.9).to(CUDA)
    m.load_state_dict(other.state_dict())  # refreshes the include cache (new buffers)
    m.update(x[:50], y[:50])
    assert set(m.__dict__["_graphs"]) - keys


@needs_triton
def test_fused_paths_are_used_and_keep_the_caches_exact():
    calls = {"commit": 0, "eval": 0}
    real_commit, real_eval = K.fused_commit, K.conv_eval

    def commit(*a, **k):
        calls["commit"] += 1
        return real_commit(*a, **k)

    def evaluate(*a, **k):
        calls["eval"] += 1
        return real_eval(*a, **k)

    K.fused_commit, K.conv_eval = commit, evaluate
    try:
        # the 2-D Noisy XOR benchmark's model (B2; fused and eager both reach 0.99 on its data
        # after 60 epochs of batch 32); here a short run only has to learn clearly
        x, y = tt.data.make_2d_noisy_xor(2500, device=CUDA)
        m = tt.ConvTsetlinMachine(2, 40, 60, 3.9, patch_size=2, input_shape=tuple(x.shape[1:]),
                                  boost_true_positive=False).to(CUDA)
        m.cuda_graphs = False  # count the Python-level calls (a graph replays without them)
        for _ in range(30):
            perm = torch.randperm(2500, device=CUDA)
            for i in range(0, 2500, 50):
                m.update(x[perm[i:i + 50]], y[perm[i:i + 50]])
        m.eval()
        assert float((m.predict(x) == y).float().mean()) > 0.7
    finally:
        K.fused_commit, K.conv_eval = real_commit, real_eval
    assert calls["commit"] == 30 * 50 and calls["eval"] > 30 * 50
    inc = m.ta_state >= m.n_states
    assert torch.equal(m.include, inc.to(m.include.dtype))
    assert torch.equal(m.include_count, inc.sum(1))
    assert torch.equal(m.include_bits(), K.pack_bits(inc))
