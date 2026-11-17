"""Booleanizations of the members (plan §3). Thermometers and bit planes run on the GPU from the
(augmented) uint8 images; the adaptive thresholds are OpenCV code in ``augment.cpu_encode``.

    therm8  colour thermometer, 8 levels per channel: x >= k*255/9, k = 1..8 -> (N, 24, 32, 32),
            plane = channel * 8 + level (Toolbox ``color_thermometer``; cifar10_batch_clauses)
    bit8    8 bit planes per colour, MSB first -> (N, 3, 8, 32, 32) for the 3D convolution
    agauss / amean  per-channel adaptive Gaussian / mean threshold -> (N, 3, 32, 32)
"""

import numpy as np
import torch
from augment import CPU_ENCS, cpu_encode

_LEVELS = torch.tensor(np.arange(1, 9) * 255 / 9, dtype=torch.float64)


def gpu_encode(enc, x):
    """x: uint8 (N, 32, 32, 3) on any device -> bool."""
    n = x.shape[0]
    if enc == "therm8":
        bits = x.unsqueeze(-1).to(torch.float64) >= _LEVELS.to(x.device)          # (N, 32, 32, 3, 8)
        return bits.reshape(n, 32, 32, 24).permute(0, 3, 1, 2).contiguous()
    if enc == "bit8":
        shifts = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)
        bits = (x.unsqueeze(-1) >> shifts) & 1                                     # (N, 32, 32, 3, 8)
        return bits.to(torch.bool).permute(0, 3, 4, 1, 2).contiguous()
    raise KeyError(enc)


def encode_fixed(enc, x_np, device):
    """Un-augmented images (validation, test, panel) -> bool tensor on ``device``."""
    if enc in CPU_ENCS:
        return torch.from_numpy(cpu_encode(enc, x_np)).to(device)
    xt = torch.from_numpy(x_np).to(device)
    return torch.cat([gpu_encode(enc, xt[i:i + 2000]) for i in range(0, len(xt), 2000)])
