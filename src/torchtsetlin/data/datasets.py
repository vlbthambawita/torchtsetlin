"""Ready-made Boolean datasets and helpers to wrap your own data."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, TensorDataset

from ..utils import as_bool_tensor, as_long_tensor

__all__ = [
    "BooleanTensorDataset",
    "TransformDataset",
    "make_xor",
    "make_noisy_xor",
    "make_parity",
    "make_2d_noisy_xor",
    "make_shapes",
    "load_mnist_boolean",
    "load_torchvision_boolean",
    "to_boolean_tensors",
]


class BooleanTensorDataset(TensorDataset):
    """``TensorDataset`` that coerces features to ``torch.bool`` and targets to ``long``
    (or float for regression). Keeps everything on one device for fast GPU epochs."""

    def __init__(self, x, y, device=None, regression: bool = False) -> None:
        xb = as_bool_tensor(x, device=torch.device(device) if device else None)
        if regression:
            yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device=xb.device)
        else:
            yt = as_long_tensor(y, device=xb.device)
        super().__init__(xb, yt)

    @property
    def x(self) -> Tensor:
        return self.tensors[0]

    @property
    def y(self) -> Tensor:
        return self.tensors[1]


class TransformDataset(Dataset):
    """Apply a (booleanizing) transform lazily to the samples of another dataset."""

    def __init__(self, base: Dataset, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        self.base = base
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


def to_boolean_tensors(x, y=None, device=None) -> Tuple[Tensor, Optional[Tensor]]:
    """Coerce ``x`` to a Boolean tensor and ``y`` to a long tensor on ``device``."""
    dev = torch.device(device) if device else None
    xb = as_bool_tensor(x, device=dev)
    yb = as_long_tensor(y, device=dev) if y is not None else None
    return xb, yb


# --------------------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------------------
def make_xor(n: int, n_features: int = 2, seed: Optional[int] = 0, device=None) -> Tuple[Tensor, Tensor]:
    """Random Boolean vectors labelled by ``x0 XOR x1`` (other features are distractors)."""
    return make_noisy_xor(n, n_features=n_features, noise=0.0, seed=seed, device=device)


def make_noisy_xor(
    n: int, n_features: int = 12, noise: float = 0.4, seed: Optional[int] = 0, device=None
) -> Tuple[Tensor, Tensor]:
    """The *Noisy XOR* benchmark of the original Tsetlin machine paper: 12 Boolean features
    of which only the first two matter (``y = x0 XOR x1``); a fraction ``noise`` of the
    training labels are flipped."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, 2, (n, n_features), generator=g).bool()
    y = (x[:, 0] ^ x[:, 1]).long()
    if noise > 0:
        flip = torch.rand(n, generator=g) < noise
        y = torch.where(flip, 1 - y, y)
    dev = torch.device(device) if device else None
    return (x.to(dev), y.to(dev)) if dev else (x, y)


def make_parity(n: int, n_bits: int = 4, n_features: int = 8, seed: Optional[int] = 0, device=None) -> Tuple[Tensor, Tensor]:
    """Parity of the first ``n_bits`` features (harder than XOR)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.randint(0, 2, (n, n_features), generator=g).bool()
    y = (x[:, :n_bits].long().sum(dim=1) % 2).long()
    dev = torch.device(device) if device else None
    return (x.to(dev), y.to(dev)) if dev else (x, y)


def make_2d_noisy_xor(
    n: int, size: int = 4, noise: float = 0.0, seed: Optional[int] = 0, device=None, max_tries: int = 200
) -> Tuple[Tensor, Tensor]:
    """2D Noisy XOR of the Convolutional TM paper: a ``size x size`` binary image whose
    pixels are random, except for one ``2 x 2`` patch (at a random position) that carries
    the class pattern.

    Class ``1`` patterns are the diagonals ``[[1,0],[0,1]]`` / ``[[0,1],[1,0]]``, class ``0``
    patterns are the horizontal bars ``[[1,1],[0,0]]`` / ``[[0,0],[1,1]]``. Backgrounds are
    resampled so that no *other* window contains any of the four patterns (otherwise the
    label would be ambiguous). A fraction ``noise`` of the labels is flipped.

    Returns images ``(n, 1, size, size)`` and labels ``(n,)``.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    patterns = torch.tensor(
        [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=torch.bool
    )  # first two -> class 0, last two -> class 1
    y = torch.randint(0, 2, (n,), generator=g)
    variant = torch.randint(0, 2, (n,), generator=g)
    which = y * 2 + variant
    py = torch.randint(0, size - 1, (n,), generator=g)
    px = torch.randint(0, size - 1, (n,), generator=g)
    x = torch.randint(0, 2, (n, 1, size, size), generator=g).bool()
    ar = torch.arange(n)

    def stamp(imgs: Tensor, idx: Tensor) -> None:
        pat = patterns[which[idx]]  # (k, 2, 2)
        for dy in range(2):
            for dx in range(2):
                imgs[idx, 0, py[idx] + dy, px[idx] + dx] = pat[:, dy, dx]

    def ambiguous(imgs: Tensor) -> Tensor:
        win = imgs[:, 0].unfold(1, 2, 1).unfold(2, 2, 1)  # (n, size-1, size-1, 2, 2)
        hit = torch.zeros(imgs.shape[0], size - 1, size - 1, dtype=torch.bool)
        for pat in patterns:
            hit |= (win == pat).all(dim=-1).all(dim=-1)
        hit[ar, py, px] = False  # the placed patch is allowed
        return hit.any(dim=(1, 2))

    stamp(x, ar)
    bad = ambiguous(x)
    tries = 0
    while bool(bad.any()) and tries < max_tries:
        idx = torch.nonzero(bad).flatten()
        x[idx] = torch.randint(0, 2, (idx.numel(), 1, size, size), generator=g).bool()
        stamp(x, idx)
        bad = ambiguous(x)
        tries += 1
    if noise > 0:
        flip = torch.rand(n, generator=g) < noise
        y = torch.where(flip, 1 - y, y)
    dev = torch.device(device) if device else None
    return (x.to(dev), y.long().to(dev)) if dev else (x, y.long())


def make_shapes(n: int, size: int = 8, seed: Optional[int] = 0, device=None) -> Tuple[Tensor, Tensor]:
    """Toy image classification: a hollow ``3x3`` circle (class 0) or a ``3x3`` cross (class 1)
    at a random position in a ``size x size`` image (Chapter 4 of the Tsetlin machine book).
    Returns ``(n, 1, size, size)`` Boolean images and labels."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.zeros(n, 1, size, size, dtype=torch.bool)
    y = torch.randint(0, 2, (n,), generator=g)
    circle = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.bool)
    cross = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.bool)
    py = torch.randint(0, size - 2, (n,), generator=g)
    px = torch.randint(0, size - 2, (n,), generator=g)
    for i in range(n):
        x[i, 0, py[i] : py[i] + 3, px[i] : px[i] + 3] = cross if y[i] == 1 else circle
    dev = torch.device(device) if device else None
    return (x.to(dev), y.long().to(dev)) if dev else (x, y.long())


# --------------------------------------------------------------------------------------
# torchvision wrappers
# --------------------------------------------------------------------------------------
def load_torchvision_boolean(
    dataset_cls,
    root: str,
    train: bool,
    encoder: Optional[Callable] = None,
    threshold: float = 0.3,
    device=None,
    download: bool = True,
    max_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Load a torchvision dataset fully into memory and booleanize it.

    ``encoder`` receives a float tensor ``(N, C, H, W)`` scaled to ``[0, 1]`` and must return
    a Boolean tensor. By default pixels are thresholded at ``threshold`` (0.3, as in the
    original MNIST experiments). Returns ``(x_bool, y_long)`` on ``device``.
    """
    ds = dataset_cls(root=root, train=train, download=download, **kwargs)
    data = ds.data
    if not isinstance(data, Tensor):
        data = torch.as_tensor(np.asarray(data))
    if data.dim() == 3:  # (N, H, W)
        data = data.unsqueeze(1)
    elif data.dim() == 4 and data.shape[-1] in (1, 3):  # (N, H, W, C) -> (N, C, H, W)
        data = data.permute(0, 3, 1, 2)
    targets = torch.as_tensor(np.asarray(ds.targets)).long()
    if max_samples is not None:
        data, targets = data[:max_samples], targets[:max_samples]
    x = data.float() / 255.0
    if device is not None:
        x, targets = x.to(device), targets.to(device)
    xb = encoder(x) if encoder is not None else (x > threshold)
    return xb, targets


def load_mnist_boolean(
    root: str = "./data",
    train: bool = True,
    threshold: float = 0.3,
    encoder: Optional[Callable] = None,
    device=None,
    flatten: bool = False,
    max_samples: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """MNIST as Boolean tensors ``(N, 1, 28, 28)`` (or ``(N, 784)`` with ``flatten=True``).
    Requires ``torchvision``."""
    try:
        from torchvision.datasets import MNIST
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision is required for load_mnist_boolean") from e
    x, y = load_torchvision_boolean(
        MNIST, root, train, encoder=encoder, threshold=threshold, device=device, max_samples=max_samples
    )
    if flatten:
        x = x.reshape(x.shape[0], -1)
    return x, y
