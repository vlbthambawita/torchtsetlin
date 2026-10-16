"""Spec tests 25, 26, 28, 29, 33."""
import numpy as np
import pytest
import torch

from experiments.transformer_like.code import attention as A
from experiments.transformer_like.code import attention_ref as R
from experiments.transformer_like.code.features import (N_FEATURES, cache_key, transform, transform_batch)
from experiments.transformer_like.code.tokenization import encode_images


@pytest.mark.parametrize("mode", ["qk", "uniform", "self_only", "shuffled_keys", "direct"])
def test_feature_counts_and_order(mode, smoke, tiny_encoders):  # spec 25, 26
    q, k, _ = tiny_encoders
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:32])
    f, it = transform_batch(u8, mode, q, k, 1.0, intermediates=True)
    assert f.shape == (32, N_FEATURES[mode]) and f.dtype == torch.bool
    bits, tokens, active = encode_images(u8)
    assert torch.equal(f[:, :784], bits.reshape(32, -1))
    if mode == "direct":
        rest = f[:, 784:].view(32, 2, 16, 10, 3)
        assert not rest[:, :, ~active[0]][0].any() if (~active[0]).any() else True
        assert torch.equal(rest[:, 0], A.quantize(it["uq"], A.DIRECT_THRESHOLDS) & active[..., None, None])
    else:
        ctx_bits = f[:, 784:].view(32, 16, 55, 3)
        assert not ctx_bits[~active].any()  # inactive receivers -> zero context bits
        assert torch.equal(ctx_bits, A.quantize(it["context"]))


def test_batch_independence(smoke, tiny_encoders):  # spec 28
    q, k, _ = tiny_encoders
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:40])
    _, a = transform_batch(u8, "qk", q, k, 1.0, intermediates=True)
    _, b = transform_batch(u8[7:8], "qk", q, k, 1.0, intermediates=True)
    assert torch.allclose(a["context"][7:8], b["context"], atol=1e-6)
    f1 = transform(tr.images[:40], "qk", q, k, 1.0, batch=40)
    f2 = transform(tr.images[:40], "qk", q, k, 1.0, batch=3)
    assert torch.equal(f1, f2)


def test_reference_bits_away_from_thresholds(smoke, tiny_encoders):  # spec 29
    q, k, _ = tiny_encoders
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:64])
    _, it = transform_batch(u8, "qk", q, k, 0.5, intermediates=True)
    _, tokens, active = encode_images(u8)
    qn = R.normalize(it["uq"].double().numpy())
    kn = R.normalize(it["uk"].double().numpy())
    ref = R.context(R.attention(qn, kn, active.numpy(), 0.5), tokens.double().numpy())
    main = it["context"].numpy()
    assert np.allclose(main, ref, atol=1e-5)
    far = np.ones_like(ref, dtype=bool)
    for t in A.CONTEXT_THRESHOLDS:
        far &= np.abs(ref - t) > 1e-5
    assert np.array_equal(R.quantize(ref)[far], A.quantize(torch.as_tensor(main)).numpy()[far])


def test_cache_key_invalidation(tiny_encoders, smoke):  # spec 33
    q, k, _ = tiny_encoders
    tr, _ = smoke
    base = cache_key(tr.ids, "qk", 1.0, q, k)
    assert base == cache_key(tr.ids, "qk", 1.0, q, k)
    assert base != cache_key(tr.ids, "qk", 0.5, q, k)
    assert base != cache_key(tr.ids, "shuffled_keys", 1.0, q, k)
    assert base != cache_key(tr.ids[:-1], "qk", 1.0, q, k)
    assert base != cache_key(tr.ids, "qk", 1.0, k, q)  # different encoder checkpoints
    assert cache_key(tr.ids, "uniform", 1.0, q, k) == cache_key(tr.ids, "uniform", 0.5, None, None)


def test_uniform_and_self_only_never_call_encoders(smoke):
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:8])
    for mode in ("uniform", "self_only"):
        assert transform_batch(u8, mode, None, None).shape == (8, 3424)


def test_uniform_context_exact_at_thresholds():
    """D19: k/m values that equal a threshold quantize as the exact rational would."""
    import torch
    from experiments.transformer_like.code.tokenization import encode_images
    u8 = torch.zeros(1, 28, 28, dtype=torch.uint8)
    for tok in range(4):  # four active tokens in row 0, pixel 0 on in exactly one of them
        u8[0, 0, 7 * tok + 1] = 255
    u8[0, 0, 0] = 255
    f = transform_batch(u8, "uniform")
    ctx = f[0, 784:].view(16, 55, 3)
    assert ctx[0, 0].tolist() == [True, False, False]  # 1/4 >= 0.25 exactly
