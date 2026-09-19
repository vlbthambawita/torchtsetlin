"""CIFAR-10 loading and Booleanization for the MCTM experiments."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

import torchtsetlin as tt

CACHE = os.environ.get("MCTM_CACHE", "/work/vajira/DL2026/torchtsetlin/experiments/mctm/.cache")
ROOT = os.environ.get("MCTM_DATA", "/work/vajira/DL2026/torchtsetlin/experiments/mctm/.data")


def load_cifar10(n_bits: int = 4, device="cuda", max_train: Optional[int] = None,
                 max_test: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Boolean CIFAR-10: (N, 3*n_bits, 32, 32) bool on ``device`` plus int64 labels.

    Per-channel thermometer encoding (``ColorThermometerEncoder``). Cached on disk as bool
    tensors so repeated runs pay the encode once.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"cifar10_therm{n_bits}.pt")
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
    else:
        from torchvision.datasets import CIFAR10

        enc = tt.data.ColorThermometerEncoder(n_bits=n_bits, value_range=(0.0, 1.0))
        xtr, ytr = tt.data.load_torchvision_boolean(CIFAR10, ROOT, train=True, encoder=enc)
        xte, yte = tt.data.load_torchvision_boolean(CIFAR10, ROOT, train=False, encoder=enc)
        blob = {"xtr": xtr.cpu(), "ytr": ytr.cpu(), "xte": xte.cpu(), "yte": yte.cpu()}
        torch.save(blob, path)
    xtr, ytr, xte, yte = blob["xtr"], blob["ytr"], blob["xte"], blob["yte"]
    if max_train:
        xtr, ytr = xtr[:max_train], ytr[:max_train]
    if max_test:
        xte, yte = xte[:max_test], yte[:max_test]
    return xtr.to(device), ytr.to(device), xte.to(device), yte.to(device)


if __name__ == "__main__":
    for nb in (4, 8):
        xtr, ytr, xte, yte = load_cifar10(n_bits=nb, device="cpu")
        print(f"n_bits={nb}: train {tuple(xtr.shape)} {xtr.dtype}, test {tuple(xte.shape)}, "
              f"density {xtr.float().mean():.3f}, classes {int(ytr.max())+1}")
