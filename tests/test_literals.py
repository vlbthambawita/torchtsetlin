"""``literals="positive" | "negated"``: a clause may only include one kind of literal (issue #23).

A switched-off literal starts excluded and never receives feedback, on every update path and for
every model family; the default ``"both"`` is the model from before the option existed.
"""

import pytest
import torch

import torchtsetlin as tt
from torchtsetlin import _triton as K
from torchtsetlin.data.graphs import GraphBatch

CUDA = torch.device("cuda")
needs_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and K.available(CUDA)), reason="needs CUDA and Triton"
)
KINDS = ["positive", "negated"]


def _graphs(n, F, gen):
    feats, edges, types = [], [], []
    for _ in range(n):
        k = int(torch.randint(2, 6, (1,), generator=gen))
        feats.append(torch.rand(k, F, generator=gen) < 0.4)
        E = int(torch.randint(1, 3 * k, (1,), generator=gen))
        edges.append(torch.randint(0, k, (E, 2), generator=gen))
        types.append(torch.randint(0, 2, (E,), generator=gen))
    return GraphBatch.from_graphs(feats, edges, types, n_edge_types=2)


def _family(name, literals, device, **kw):
    """(model, inputs, targets) of one model family on a small random task."""
    gen = torch.Generator().manual_seed(3)
    kw = dict(kw, literals=literals)
    x, y = tt.data.make_noisy_xor(400, noise=0.1, device=device)
    if name == "flat":
        m = tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True, max_included_literals=6, **kw)
    elif name == "coalesced":
        m = tt.CoalescedTsetlinMachine(12, 2, 30, 15, 3.9, **kw)
    elif name == "regression":
        m, y = tt.RegressionTsetlinMachine(12, 40, 20, 3.0, y_range=(0, 1), **kw), y.float()
    elif name == "conv":
        m, x = tt.ConvTsetlinMachine(2, 20, 30, 3.9, patch_size=2, input_shape=(1, 3, 4), **kw), x.view(-1, 1, 3, 4)
    elif name == "conv3d":
        m = tt.Conv3dTsetlinMachine(2, 20, 30, 3.9, patch_size=(2, 2, 2), input_shape=(1, 2, 2, 3), **kw)
        x = x.view(-1, 1, 2, 2, 3)
    elif name == "segmentation":
        x, y = tt.data.make_segmentation_shapes(24, size=16, seed=0, device=device)
        m = tt.SegmentationTsetlinMachine(3, n_clauses=20, T=10, s=4.0, patch_size=3, input_shape=(1, 16, 16), **kw)
    elif name == "graph":
        x = _graphs(60, 6, gen).to(device)
        y = torch.randint(0, 2, (60,), generator=gen).to(device)
        m = tt.GraphTsetlinMachine(6, 2, 16, T=8, s=3.0, depth=2, message_size=8, message_bits=2,
                                   n_edge_types=2, **kw)
    else:  # pragma: no cover
        raise KeyError(name)
    return m.to(device), x, y


FAMILIES = ["flat", "coalesced", "regression", "conv", "conv3d", "segmentation", "graph"]


def _halves(m):
    inc, Fn = m.included_mask(), int(m.n_features)
    return inc[:, :Fn], inc[:, Fn:]


def _off_on(m):
    pos, neg = _halves(m)
    return (neg, pos) if m.literals == "positive" else (pos, neg)


def _train(m, x, y, steps=12, batch=20, sequential=False):
    n = len(y)
    for i in range(steps):
        j = torch.arange(i * batch, (i + 1) * batch) % n
        m.update(x[j.to(y.device)] if not isinstance(x, GraphBatch) else x[j], y[j.to(y.device)],
                 sequential=sequential)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("literals", KINDS)
@pytest.mark.parametrize("sequential", [False, True])
def test_switched_off_literals_are_never_included(device, family, literals, sequential):
    m, x, y = _family(family, literals, device, init="random")
    off, on = _off_on(m)
    assert not off.any(), "init='random' included a switched-off literal"
    assert on.any(), "init='random' should include some allowed literals"
    start = m.ta_state.clone()
    _train(m, x, y, sequential=sequential and family != "segmentation")
    off, on = _off_on(m)
    assert not off.any()
    assert on.any()
    assert (m.ta_state != start).any(), "nothing was learned"
    # the switched-off automata did not move at all: they get no feedback of any type
    blocked = (m.literal_allowed == 0)
    assert torch.equal(m.ta_state[:, blocked], start[:, blocked])


@pytest.mark.parametrize("family", ["flat", "conv"])
@pytest.mark.parametrize("literals", KINDS)
def test_compiled_kernels_respect_the_restriction(device, compiled, family, literals):
    m, x, y = _family(family, literals, device)
    for seq in (False, True):
        _train(m, x, y, steps=8, sequential=seq)
    off, on = _off_on(m)
    assert not off.any() and on.any()


def test_both_is_the_default_and_changes_nothing(device):
    states = []
    for kw in ({}, {"literals": "both"}):
        tt.seed_everything(1)
        m = tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True, **kw).to(device)
        x, y = tt.data.make_noisy_xor(400, noise=0.1, device=device)
        _train(m, x, y)
        assert m.literal_allowed is None and m._literal_active is None
        assert "literal_allowed" not in m.state_dict()
        states.append((m.ta_state.clone(), m.weights.clone()))
    assert torch.equal(states[0][0], states[1][0]) and torch.equal(states[0][1], states[1][1])
    pos, neg = _halves(m)
    assert pos.any() and neg.any()


def test_invalid_value_is_rejected():
    with pytest.raises(ValueError, match="literals"):
        tt.TsetlinMachine(4, 2, 4, 5, literals="none")


def _one_hot(n, device, seed):
    """Two positions, each holding one of four symbols (one-hot, 8 features). The class says
    whether position 0 holds symbol 0 or 1 (two "synonyms" of one concept). With negated
    literals a clause finds it by elimination (NOT s2 AND NOT s3), with positive literals one
    clause per synonym does; position 1 is noise."""
    g = torch.Generator().manual_seed(seed)
    sym = torch.randint(0, 4, (n, 2), generator=g)
    x = torch.zeros(n, 8, dtype=torch.bool)
    x[torch.arange(n), sym[:, 0]] = True
    x[torch.arange(n), 4 + sym[:, 1]] = True
    return x.to(device), (sym[:, 0] < 2).long().to(device)


@pytest.mark.parametrize("literals", KINDS)
def test_the_allowed_kind_learns(device, literals):
    x, y = _one_hot(2000, device, 0)
    xt, yt = _one_hot(1000, device, 1)
    tt.seed_everything(0)
    m = tt.TsetlinMachine(8, 2, 10, 10, 3.0, literals=literals).to(device)
    for _ in range(15):
        for i in range(0, 2000, 25):
            m.update(x[i:i + 25], y[i:i + 25])
    m.eval()
    assert float((m.predict(xt) == yt).float().mean()) > 0.97
    off, on = _off_on(m)
    assert not off.any()
    if literals == "negated":  # a positive clause of class 1 rules out symbols 2 and 3
        cl = (m.clause_class == 1) & (m.clause_polarity > 0)
        assert on[cl][:, 2:4].all(dim=1).any()


def test_literal_dropout_is_combined_with_the_restriction(device):
    m, x, y = _family("flat", "negated", device, drop_literal_p=0.5)
    seen = []
    for i in range(12):
        m.update(x[i * 20:(i + 1) * 20], y[i * 20:(i + 1) * 20])
        act = m._literal_active
        assert bool((act <= m.literal_allowed).all()), "dropout switched a blocked literal on"
        seen.append(act.clone())
    assert any(not torch.equal(a, m.literal_allowed) for a in seen), "no literal was ever dropped"
    off, on = _off_on(m)
    assert not off.any() and on.any()


def test_restriction_follows_the_model_across_devices_and_resets(device):
    m = tt.TsetlinMachine(12, 2, 20, 15, 3.9, literals="positive")
    assert m._literal_active is m.literal_allowed
    m = m.to(device)
    assert m.literal_allowed.device.type == device.type
    assert m._literal_active is m.literal_allowed
    x, y = tt.data.make_noisy_xor(400, noise=0.1, device=device)
    _train(m, x, y)
    off, on = _off_on(m)
    assert not off.any() and on.any()
    m.reset_parameters()
    assert not m.included_mask().any() and m._literal_active is m.literal_allowed
    _train(m, x, y)
    assert not _off_on(m)[0].any()
    assert "literals='positive'" in repr(m)


def test_lazy_models_build_the_mask_on_the_first_batch(device):
    m = tt.TsetlinMachine(None, 2, 20, 15, 3.9, literals="negated").to(device)
    assert m.literal_allowed is None
    x, y = tt.data.make_noisy_xor(400, noise=0.1, device=device)
    _train(m, x, y)
    assert m.literal_allowed.shape == (24,) and m.literal_allowed.device.type == device.type
    off, on = _off_on(m)
    assert not off.any() and on.any()


def test_save_and_load(device):
    m, x, y = _family("flat", "negated", device)
    _train(m, x, y)
    sd = m.state_dict()
    assert "literal_allowed" not in sd  # the checkpoint format is unchanged
    same = tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True, max_included_literals=6, literals="negated")
    same.load_state_dict(sd)
    assert torch.equal(same.ta_state.cpu(), m.ta_state.cpu())
    lazy = tt.TsetlinMachine(None, 2, 20, 15, 3.9, weighted=True, literals="negated")
    lazy.load_state_dict(sd)
    assert lazy.literal_allowed is not None and torch.equal(lazy.ta_state.cpu(), m.ta_state.cpu())
    m.eval(), same.eval()
    assert torch.equal(same.to(device).predict(x), m.predict(x))
    # a state trained with another setting does not fit silently
    other = tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True, literals="positive")
    with pytest.raises(ValueError, match="switches off"):
        other.load_state_dict(sd)
    both = tt.TsetlinMachine(12, 2, 20, 15, 3.9, weighted=True)
    both.load_state_dict(sd)  # an unrestricted model accepts any state


@needs_triton
@pytest.mark.parametrize("seq", [False, True])
def test_restricted_updates_stay_sync_free(compiled, seq):
    for family in ("flat", "coalesced", "regression", "conv"):
        m, x, y = _family(family, "negated", CUDA)
        m.cuda_graphs = False
        m.feedback_mode = "sequential" if seq else "batch"
        for i in range(0, 200, 50):  # compile / warm up
            m.update(x[i:i + 50], y[i:i + 50])
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            m.update(x[200:250], y[200:250])
        finally:
            torch.cuda.set_sync_debug_mode(0)
        assert not _off_on(m)[0].any()


@needs_triton
@pytest.mark.parametrize("seq", [False, True])
def test_restricted_updates_are_replayed_as_cuda_graphs(seq):
    x, y = _one_hot(3000, CUDA, 0)
    xt, yt = _one_hot(1000, CUDA, 1)
    tt.seed_everything(0)
    m = tt.TsetlinMachine(8, 2, 20, 10, 3.0, literals="negated",
                          feedback_mode="sequential" if seq else "batch").to(CUDA)
    bs = 500 if seq else 50
    for _ in range(8):
        for i in range(0, 3000, bs):
            m.update(x[i:i + bs], y[i:i + bs])
    graphs = [v for v in m.__dict__["_graphs"].values() if not isinstance(v, str)]
    assert graphs, "no update was replayed from a CUDA graph"
    assert not _off_on(m)[0].any()
    m.eval()
    assert float((m.predict(xt) == yt).float().mean()) > 0.97
