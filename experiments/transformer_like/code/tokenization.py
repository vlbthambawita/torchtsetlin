"""Binary preprocessing and 7x7 patch tokens (spec §6).

- ``binarize``: ``x = 1[u >= 128]`` on uint8 intensities.
- ``patches``: 16 non-overlapping 7x7 patches, token ``i = 4 r + c``, row-major pixels.
- ``POSITION_BITS``: ``[r>=1, r>=2, r>=3, c>=1, c>=2, c>=3]`` (identical to the library's
  ``_ConvMixin._position_bits`` for a 4x4 grid; PLAN §2.2).
- activity uses the 49 pixel bits only.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

PIXEL_THRESHOLD = 128
GRID = 4
PATCH = 7
N_TOKENS = GRID * GRID
N_PIX = PATCH * PATCH
N_POS = 6
TOKEN_DIM = N_PIX + N_POS  # 55


def binarize(u8: Tensor) -> Tensor:
    if u8.dtype != torch.uint8:
        raise TypeError("binarize expects uint8 intensities")
    return u8 >= PIXEL_THRESHOLD


def position_bits(device=None) -> Tensor:
    r = torch.arange(GRID, device=device).repeat_interleave(GRID)
    c = torch.arange(GRID, device=device).repeat(GRID)
    k = torch.arange(1, GRID, device=device)
    return torch.cat([r[:, None] >= k[None], c[:, None] >= k[None]], dim=1)  # (16, 6) bool


def patches(bits: Tensor) -> Tensor:
    """``(B, 28, 28)`` -> ``(B, 16, 49)``."""
    B = bits.shape[0]
    return bits.reshape(B, GRID, PATCH, GRID, PATCH).permute(0, 1, 3, 2, 4).reshape(B, N_TOKENS, N_PIX)


def unpatch(p: Tensor) -> Tensor:
    """Inverse of :func:`patches`."""
    B = p.shape[0]
    return p.reshape(B, GRID, GRID, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(B, 28, 28)


def encode_images(u8: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """uint8 images -> (image bits (B,28,28), tokens (B,16,55), active (B,16)), all bool."""
    bits = binarize(u8)
    pix = patches(bits)
    active = pix.any(dim=-1)
    pos = position_bits(u8.device).unsqueeze(0).expand(pix.shape[0], -1, -1)
    return bits, torch.cat([pix, pos], dim=-1), active
