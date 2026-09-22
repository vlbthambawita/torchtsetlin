"""HOG Booleanization for the TM Composites / Toolbox HOG specialist (LG-008).

**Specification supplied by the tm-theorist from the reference scripts**, not inferred:

```python
cv2.HOGDescriptor((32, 32),   # winSize   — the whole image: this specialist is FLAT
                  (12, 12),   # blockSize
                  (4, 4),     # blockStride
                  (4, 4),     # cellSize
                  18,         # nbins            (non-default)
                  1,          # derivAperture
                  -1.0,       # winSigma
                  0,          # histogramNormType
                  0.2,        # L2HysThreshold
                  True,       # gammaCorrection
                  64,         # nlevels
                  True)       # signedGradient   (non-default)
bits = hog.compute(img_uint8_3channel) >= 0.1     # 5 832 features
```

Two things in there are easy to get wrong and both are load-bearing: ``nbins=18`` with
``signedGradient=True`` (the usual defaults are 9 and unsigned), and the descriptor is computed
on the **3-channel uint8 image — not greyscaled**; OpenCV takes the largest-magnitude channel
per pixel internally.

**This specialist is flat, not convolutional.** ``winSize`` is the whole 32x32 image, so there
is exactly one window and the descriptor is a 5 832-element vector with no spatial layout left
to convolve over. It is emitted here as ``(N, 5832, 1, 1)`` so that the existing convolutional
builder expresses it exactly with ``patch_size=1``: one patch, ``n_features = 5832``, and the
thermometer-style position bits vanish because ``Py = Px = 1``. That is a flat Tsetlin machine
written in the conv model's shape, which keeps one code path instead of two.

An earlier version of this file implemented HOG in pure torch against my own reading of
Dalal & Triggs (8 unsigned orientations, 4x4 cells, 2x2 L2-Hys blocks, thermometer-4, emitted
on a 7x7 block grid). That reading was **wrong in every parameter that matters** and produced a
128-plane convolutional map rather than a flat 5 832-bit vector. It is gone; the lesson is in
`AUDIT.md` A22.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor

# The reference configuration, verbatim. Changing any of these makes a different encoder, so
# they are named constants rather than call-site literals.
WIN_SIZE: Tuple[int, int] = (32, 32)
BLOCK_SIZE: Tuple[int, int] = (12, 12)
BLOCK_STRIDE: Tuple[int, int] = (4, 4)
CELL_SIZE: Tuple[int, int] = (4, 4)
NBINS = 18
DERIV_APERTURE = 1
WIN_SIGMA = -1.0
HIST_NORM_TYPE = 0
L2HYS_THRESHOLD = 0.2
GAMMA_CORRECTION = True
NLEVELS = 64
SIGNED_GRADIENT = True
THRESHOLD = 0.1
N_FEATURES = 5832


def _descriptor():
    import cv2

    return cv2.HOGDescriptor(WIN_SIZE, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS,
                             DERIV_APERTURE, WIN_SIGMA, HIST_NORM_TYPE, L2HYS_THRESHOLD,
                             GAMMA_CORRECTION, NLEVELS, SIGNED_GRADIENT)


def hog_features(x_u8: Tensor) -> Tensor:
    """``(N, 3, 32, 32)`` uint8 -> real HOG descriptors ``(N, 5832)`` float32."""
    hog = _descriptor()
    imgs = x_u8.permute(0, 2, 3, 1).contiguous().numpy()      # (N, 32, 32, 3) uint8
    out = np.empty((imgs.shape[0], N_FEATURES), dtype=np.float32)
    for i in range(imgs.shape[0]):
        out[i] = hog.compute(np.ascontiguousarray(imgs[i])).reshape(-1)
    return torch.from_numpy(out)


def hog_boolean(x_u8: Tensor, threshold: float = THRESHOLD, chunk: int = 10000) -> Tensor:
    """``(N, 3, 32, 32)`` uint8 -> Boolean ``(N, 5832, 1, 1)``.

    The trailing singleton spatial dimensions are what let the convolutional builder express
    this flat specialist with ``patch_size=1`` (see the module docstring).
    """
    outs = []
    for i in range(0, x_u8.shape[0], chunk):
        f = hog_features(x_u8[i : i + chunk])
        outs.append((f >= threshold).unsqueeze(-1).unsqueeze(-1))
    return torch.cat(outs)


def make(threshold: float = THRESHOLD):
    """Registry-shaped callable for ``data.BOOLEANIZATIONS``."""

    def enc(x_u8: Tensor) -> Tensor:
        return hog_boolean(x_u8, threshold=threshold)

    return enc


if __name__ == "__main__":
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as data_mod

    raw = data_mod.raw_cifar10()["xtr"][:2000]
    t0 = time.time()
    f = hog_features(raw)
    dt = time.time() - t0
    b = hog_boolean(raw)
    print(f"features {tuple(f.shape)} (expected (2000, {N_FEATURES}))  "
          f"range [{float(f.min()):.3f}, {float(f.max()):.3f}]")
    print(f"boolean  {tuple(b.shape)} {b.dtype}  density {b.float().mean():.4f}")
    print(f"encode rate {2000 / dt:.0f} img/s -> full 60 000 in ~{60000 / (2000 / dt):.0f}s")
