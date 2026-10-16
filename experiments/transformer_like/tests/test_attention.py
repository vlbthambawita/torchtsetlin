"""Spec tests 14-23 and the §15.3 aggregation fixture."""
import math

import numpy as np
import pytest
import torch

from experiments.transformer_like.code import attention as A
from experiments.transformer_like.code import attention_ref as R


def _rand(B=6, n=16, d=10, seed=0, p_active=0.7):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, n, d, generator=g)
    k = torch.randn(B, n, d, generator=g)
    v = (torch.rand(B, n, 55, generator=g) < 0.4).float()
    act = torch.rand(B, n, generator=g) < p_active
    if B >= 2:
        act[0] = False          # an all-masked image
        act[1] = False
        act[1, 3] = True        # a single active token
    return q, k, v, act


def test_spec_fixture():
    q = torch.tensor([[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    v = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    act = torch.ones(1, 3, dtype=torch.bool)
    alpha = A.attention_weights(act, "qk", q, k, 1.0)
    ctx = A.aggregate(alpha, v)[0, 0]
    a = math.exp(1 / math.sqrt(2))
    assert torch.allclose(ctx, torch.tensor([2 * a, 1 + a]) / (2 * a + 1), atol=1e-6)
    assert torch.allclose(ctx, torch.tensor([0.802224, 0.598888]), atol=1e-6)
    assert A.quantize(ctx).tolist() == [[True, True, True], [True, True, False]]


@pytest.mark.parametrize("mode", A.MODES)
def test_rows_sum_and_masks(mode):  # spec 14, 15, 26
    q, k, _, act = _rand()
    alpha = A.attention_weights(act, mode, q, k, 0.7)
    allowed = act.unsqueeze(-1) & act.unsqueeze(-2)
    assert torch.isfinite(alpha).all()
    assert (alpha[~allowed] == 0).all()
    s = alpha.sum(-1)
    assert torch.allclose(s[act], torch.ones_like(s[act]), atol=1e-6)
    assert (s[~act] == 0).all() and (alpha[0] == 0).all()


def test_shift_invariance():  # spec 16
    q, k, _, act = _rand()
    allowed = act.unsqueeze(-1) & act.unsqueeze(-2)
    S = A.compatibility(q, k, 1.0)
    assert torch.allclose(A.masked_softmax(S, allowed), A.masked_softmax(S + 37.5, allowed), atol=1e-6)


def test_orientation():  # spec 17
    n = 4
    q = torch.zeros(1, n, 2)
    q[0, 0] = torch.tensor([10.0, 0.0])
    k = torch.zeros(1, n, 2)
    k[0, 2] = torch.tensor([10.0, 0.0])
    act = torch.ones(1, n, dtype=torch.bool)
    alpha = A.attention_weights(act, "qk", q, k, 1.0)
    assert alpha[0, 0].argmax() == 2  # receiver row 0 attends to sender column 2
    assert torch.allclose(alpha[0, 1], torch.full((n,), 0.25))


def test_equal_keys_uniform():  # spec 18
    q, _, _, act = _rand()
    k = torch.ones_like(q)
    a = A.attention_weights(act, "qk", q, k, 1.0)
    u = A.attention_weights(act, "uniform")
    assert torch.allclose(a, u, atol=1e-6)


def test_constant_scores_normalize_to_zero():  # spec 19
    u = torch.full((3, 16, 10), 0.4)
    assert (A.normalize_votes(u) == 0).all()
    x = torch.randn(4, 10, dtype=torch.float64)
    assert np.allclose(A.normalize_votes(x).numpy(), R.normalize(x.numpy()), atol=1e-12)


@pytest.mark.parametrize("mode", A.MODES)
def test_context_range_and_reference(mode):  # spec 20, 23
    q, k, v, act = _rand(seed=3)
    ctx = A.aggregate(A.attention_weights(act, mode, q, k, 0.5), v)
    assert (ctx >= -1e-6).all() and (ctx <= 1 + 1e-6).all()
    ref = R.context(R.attention(q.double().numpy(), k.double().numpy(), act.numpy(), 0.5, mode), v.double().numpy())
    assert np.allclose(ctx.numpy(), ref, atol=1e-5)


def test_joint_sender_permutation():  # spec 21
    q, k, v, act = _rand(B=1, seed=4, p_active=1.0)
    act[:] = True
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(1))
    c0 = A.aggregate(A.attention_weights(act, "qk", q, k), v)
    kp, vp = k[:, perm], v[:, perm]
    c1 = A.aggregate(A.attention_weights(act[:, perm], "qk", q, kp), vp)
    # receivers unchanged; senders jointly permuted (with their mask) -> same context
    assert torch.allclose(c0, c1, atol=1e-6)
    c2 = A.aggregate(A.attention_weights(act, "qk", q, kp), v)  # K alone permuted
    assert not torch.allclose(c0, c2, atol=1e-3)


def test_slot_permutation_equivariance():  # spec 22
    q, k, v, act = _rand(B=2, seed=5)
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(2))
    c0 = A.aggregate(A.attention_weights(act, "qk", q, k), v)
    c1 = A.aggregate(A.attention_weights(act[:, perm], "qk", q[:, perm], k[:, perm]), v[:, perm])
    assert torch.allclose(c0[:, perm], c1, atol=1e-6)


def test_shuffled_keys_rule():
    k = torch.arange(16, dtype=torch.float32).view(1, 16, 1).repeat(3, 1, 1)
    act = torch.zeros(3, 16, dtype=torch.bool)
    act[0, [2, 5, 9]] = True
    act[1, 7] = True  # single active -> identity
    out = A.shuffle_keys(k, act)
    assert out[0, [2, 5, 9], 0].tolist() == [5.0, 9.0, 2.0]
    assert torch.equal(out[1], k[1]) and torch.equal(out[2], k[2])
    assert np.array_equal(R.shuffle_keys(k.numpy(), act.numpy()), out.numpy())


def test_quantize_boundaries():  # spec 24
    x = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.2499999])
    assert A.quantize(x).int().tolist() == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]]
    assert A.quantize(torch.tensor([0.60])).int().tolist() == [[1, 1, 0]]
    d = A.quantize(torch.tensor([-0.5, -0.1, 0.0, 0.5]), A.DIRECT_THRESHOLDS).int().tolist()
    assert d == [[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
