"""Spec tests 2-5 and the B1-s7 token identity (PLAN §2.2)."""
import torch

import torchtsetlin as tt
from experiments.transformer_like.code.tokenization import (encode_images, patches, position_bits,
                                                            unpatch)


def test_patch_round_trip(smoke):  # spec 2
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:64])
    bits, tokens, _ = encode_images(u8)
    assert torch.equal(unpatch(tokens[..., :49]), bits)


def test_single_pixel_location():  # spec 3
    for r, c in [(0, 0), (6, 6), (7, 0), (13, 20), (27, 27), (10, 3)]:
        u8 = torch.zeros(1, 28, 28, dtype=torch.uint8)
        u8[0, r, c] = 200
        _, tokens, active = encode_images(u8)
        tok = 4 * (r // 7) + c // 7
        assert active[0].nonzero().flatten().tolist() == [tok]
        assert tokens[0, tok, :49].nonzero().flatten().tolist() == [7 * (r % 7) + c % 7]


def test_position_bits():  # spec 4
    p = position_bits()
    assert p[0].int().tolist() == [0] * 6
    assert p[15].int().tolist() == [1] * 6
    assert p[4 * 1 + 2].int().tolist() == [1, 0, 0, 1, 1, 0]


def test_blank_patch_inactive():  # spec 5
    u8 = torch.zeros(2, 28, 28, dtype=torch.uint8)
    u8[1, 27, 27] = 255
    _, tokens, active = encode_images(u8)
    assert not active[0].any() and tokens[0, 15, 49:].all()
    assert active[1].sum() == 1


def test_stride7_conv_sees_spec_tokens(smoke):
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:16])
    bits, tokens, _ = encode_images(u8)
    m = tt.ConvTsetlinMachine(10, 2, T=2, patch_size=7, stride=7, input_shape=(1, 28, 28))
    lit = m._encode(bits.unsqueeze(1))
    assert m.n_features == 55
    assert torch.equal(lit[..., :55].bool(), tokens)
