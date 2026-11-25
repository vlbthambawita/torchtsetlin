"""Booleanization of RGB uint8 images or crops (PLAN_KVASIR §7), on the device of the input.

    T4     thermometer, 4 levels per channel          Z = 12
    T8     thermometer, 8 levels per channel          Z = 24
    G8     8 bit planes of the grey image (MSB first) Z = 8
    AG     adaptive Gaussian threshold, block 11      Z = 3
    T4+AG  planes of T4 and AG                        Z = 15

``planes`` returns ``(B, Z, H, W)`` Boolean. ``pyramid`` adds each pixel's neighbourhood at
several OR-pooled resolutions (the library's ``PyramidEncoder``); ``position_bits`` are the
thermometer coordinates used by the mask head.
"""

import torch

from torchtsetlin.data.encoders import AdaptiveThresholdEncoder, PyramidEncoder

Z = {"T4": 12, "T8": 24, "G8": 8, "AG": 3, "T4+AG": 15}
DEPTH = {"T8": (3, 8), "G8": (1, 8)}                 # (channels, levels) for 3D members
_AG = {}


def _thermo(x, n):
    thr = (torch.arange(1, n + 1, device=x.device, dtype=torch.float32) * (255.0 / (n + 1))).view(1, 1, n, 1, 1)
    B, C, H, W = x.shape
    return (x.float().unsqueeze(2) >= thr).reshape(B, C * n, H, W)


def _adaptive(x):
    enc = _AG.get(x.device)
    if enc is None:
        enc = _AG[x.device] = AdaptiveThresholdEncoder(block_size=11, C=2.0, method="gaussian").to(x.device)
    return enc(x.float())


def planes(x, enc):
    """``x``: ``(B, 3, H, W)`` uint8 (or float holding 0..255). Returns Boolean planes."""
    if enc == "T4":
        return _thermo(x, 4)
    if enc == "T8":
        return _thermo(x, 8)
    if enc == "G8":
        xf = x.float()
        g = (0.299 * xf[:, 0] + 0.587 * xf[:, 1] + 0.114 * xf[:, 2]).round().clamp_(0, 255).long()
        sh = torch.arange(7, -1, -1, device=x.device).view(1, 8, 1, 1)
        return ((g.unsqueeze(1) >> sh) & 1).bool()
    if enc == "AG":
        return _adaptive(x)
    if enc == "T4+AG":
        return torch.cat([_thermo(x, 4), _adaptive(x)], 1)
    raise ValueError(f"unknown encoding {enc!r}")


def volume(x, enc):
    """``(B, C, D, H, W)`` for the 3D members: the levels of an encoding as the depth axis."""
    C, D = DEPTH[enc]
    p = planes(x, enc)
    return p.reshape(p.shape[0], C, D, p.shape[2], p.shape[3])


def pyramid(p, scales, patch):
    """``(B, Z, H, W)`` -> ``(B, Z * patch**2 * len(scales), H, W)``. ``H`` and ``W`` must be
    multiples of the largest scale (library gap MR-G10)."""
    assert p.shape[-1] % max(scales) == 0 and p.shape[-2] % max(scales) == 0, (p.shape, scales)
    return PyramidEncoder(scales=tuple(scales), patch=int(patch))(p)


def pyramid_size(enc, scales, patch):
    return Z[enc] * patch * patch * len(scales)


def position_bits(h, w, device):
    """``(h - 1 + w - 1, h, w)`` thermometer coordinates: bit ``t`` of the y block is ``y > t``."""
    ys = torch.arange(h, device=device).view(1, h, 1) > torch.arange(h - 1, device=device).view(h - 1, 1, 1)
    xs = torch.arange(w, device=device).view(1, 1, w) > torch.arange(w - 1, device=device).view(w - 1, 1, 1)
    return torch.cat([ys.expand(h - 1, h, w), xs.expand(w - 1, h, w)], 0)
