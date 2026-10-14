"""Boolean feature planes for the dense TM arms (PLAN H2 and baselines B1/B2).

Every spec is a plain dict so it can be stored in run records. ``encode(x, spec)`` maps a
``uint8`` image batch ``(B, C, H, W)`` (on any device) to Boolean planes ``(B, Z, H, W)``
and ``feature_names(spec, C)`` returns one provenance string per plane.

Spec keys
---------
``bits``       thermometer levels per input channel (thresholds k/(bits+1), k=1..bits).
``pyr``        ``None`` or ``{"scales": [1,2,4], "mode": "or"|"and"|"occ", "t": [..], "patch": 3}``.
               At each scale the planes are pooled (``occ``: mean then ``>= t``), a
               ``patch x patch`` neighbourhood is taken *at that level* and upsampled
               (nearest) to full resolution — the library ``PyramidEncoder`` layout. Scale 1
               is always the raw planes (occupancy thresholds collapse there: deduplicated).
``contrast``   ``None`` or ``{"windows": [9, 25], "t": [-0.1, 0, 0.1]}``: ``I - mean_W(I) >= t``
               on [0,1] intensities per channel, added as extra planes.
``extra_bits`` capacity control: additional plain thermometer levels (finer thresholds).
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.nn import functional as F


def thermometer(xf: torch.Tensor, bits: int) -> torch.Tensor:
    """``(B, C, H, W)`` floats in [0,1] -> ``(B, C*bits, H, W)`` bool, channel-major."""
    thr = torch.arange(1, bits + 1, device=xf.device, dtype=xf.dtype) / (bits + 1)
    B, C, H, W = xf.shape
    return (xf.unsqueeze(2) >= thr.view(1, 1, -1, 1, 1)).reshape(B, C * bits, H, W)


def _neigh(f: torch.Tensor, patch: int) -> torch.Tensor:
    """``(B, Z, h, w)`` float -> ``(B, Z*patch*patch, h, w)`` with zero padding."""
    if patch == 1:
        return f
    pad = patch // 2
    B, Z, h, w = f.shape
    u = F.unfold(F.pad(f, (pad, pad, pad, pad)), kernel_size=patch)  # (B, Z*p*p, h*w)
    return u.view(B, Z * patch * patch, h, w)


def _pool(planes: torch.Tensor, s: int, mode: str, t: float = 0.5) -> torch.Tensor:
    f = planes.float()
    if s == 1:
        return f
    if mode == "or":
        return F.max_pool2d(f, s, ceil_mode=True)
    if mode == "and":
        return -F.max_pool2d(-f, s, ceil_mode=True)
    if mode == "occ":
        return (F.avg_pool2d(f, s, ceil_mode=True) >= t - 1e-6).float()
    raise ValueError(mode)


def pyramid(planes: torch.Tensor, scales, mode: str, patch: int, ts=(0.5,)) -> torch.Tensor:
    H, W = planes.shape[-2:]
    parts = []
    for s in scales:
        variants = [None] if (mode != "occ" or s == 1) else list(ts)
        for t in variants:
            p = _neigh(_pool(planes, s, mode, 0.5 if t is None else t), patch)
            if p.shape[-2:] != (H, W):
                p = F.interpolate(p, scale_factor=s, mode="nearest")[..., :H, :W]
            parts.append(p > 0.5)
    return torch.cat(parts, 1)


def contrast(xf: torch.Tensor, windows, ts) -> torch.Tensor:
    parts = []
    for w in windows:
        m = F.avg_pool2d(F.pad(xf, (w // 2,) * 4, mode="replicate"), w, stride=1)
        d = xf - m
        for t in ts:
            parts.append(d >= t)
    return torch.cat(parts, 1)


def encode(x: torch.Tensor, spec: Dict) -> torch.Tensor:
    xf = x.float() / 255.0
    planes = thermometer(xf, spec["bits"])
    if spec.get("extra_bits"):
        # finer thermometer levels placed between the base ones (capacity control)
        nb = spec["bits"] + spec["extra_bits"]
        fine = thermometer(xf, nb)
        planes = torch.cat([planes, fine], 1)
    if spec.get("contrast"):
        c = spec["contrast"]
        planes = torch.cat([planes, contrast(xf, c["windows"], c["t"])], 1)
    if spec.get("pyr"):
        p = spec["pyr"]
        planes = pyramid(planes, p["scales"], p["mode"], p.get("patch", 3), p.get("t", (0.5,)))
    return planes.contiguous()


def n_planes(spec: Dict, C: int) -> int:
    return int(encode(torch.zeros(1, C, 16, 16, dtype=torch.uint8), spec).shape[1])


def feature_names(spec: Dict, C: int) -> List[str]:
    """Provenance names of the encoded planes (scale, channel, statistic, threshold, offset)."""
    base = [f"ch{c}>={k}/{spec['bits'] + 1}" for c in range(C) for k in range(1, spec["bits"] + 1)]
    if spec.get("extra_bits"):
        nb = spec["bits"] + spec["extra_bits"]
        base += [f"ch{c}>={k}/{nb + 1}" for c in range(C) for k in range(1, nb + 1)]
    if spec.get("contrast"):
        cc = spec["contrast"]
        base += [f"ch{c}-mean{w}>={t:+g}" for w in cc["windows"] for t in cc["t"] for c in range(C)]
    if not spec.get("pyr"):
        return base
    p = spec["pyr"]
    k = p.get("patch", 3)
    offs = [(i - k // 2, j - k // 2) for i in range(k) for j in range(k)]
    names = []
    for s in p["scales"]:
        variants = [None] if (p["mode"] != "occ" or s == 1) else list(p.get("t", (0.5,)))
        for t in variants:
            stat = "raw" if s == 1 else (p["mode"] if t is None else f"occ>={t:g}")
            names += [f"s{s}:{stat}({b})@{dy:+d},{dx:+d}" for b in base for dy, dx in offs]
    return names
