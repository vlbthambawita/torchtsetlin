"""Ready-made Boolean datasets and helpers to wrap your own data."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

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
    "make_segmentation_shapes",
    "make_scenes",
    "balanced_class_probabilities",
    "SCENE_CLASSES",
    "load_segmentation_boolean",
    "load_voc_segmentation_boolean",
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
# Dense prediction (semantic segmentation)
# --------------------------------------------------------------------------------------
SCENE_CLASSES = ("sky", "building", "tree", "road")


def make_segmentation_shapes(
    n: int, size: int = 16, seed: Optional[int] = 0, device=None
) -> Tuple[Tensor, Tensor]:
    """Toy dense-prediction task: label every pixel ``background`` / ``disc`` / ``ring``.

    Two solid discs and two rings of radius 2-3 are stamped into a ``size x size`` Boolean
    image. Disc interiors and ring interiors look identical on their own, so the classes are
    only separable from a pixel's *neighbourhood* — which is exactly what a dense Tsetlin
    machine gets to see, and what a per-pixel lookup cannot solve.

    Returns ``(n, 1, size, size)`` Boolean images and ``(n, size, size)`` long label maps
    with classes ``0 = background, 1 = disc, 2 = ring``.
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    x = torch.zeros(n, 1, size, size, dtype=torch.bool)
    y = torch.zeros(n, size, size, dtype=torch.long)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    for i in range(n):
        for cls in (1, 2):
            for _ in range(2):
                r = int(torch.randint(2, 4, (1,), generator=g))
                cy = int(torch.randint(r + 1, size - r - 1, (1,), generator=g))
                cx = int(torch.randint(r + 1, size - r - 1, (1,), generator=g))
                d2 = (yy - cy) ** 2 + (xx - cx) ** 2
                solid = d2 <= r * r
                shape = solid if cls == 1 else (solid & (d2 > (r - 1) ** 2))
                x[i, 0] |= shape
                y[i][solid] = cls
    dev = torch.device(device) if device else None
    return (x.to(dev), y.to(dev)) if dev else (x, y)


def make_scenes(
    n: int,
    size: Union[int, Tuple[int, int]] = 32,
    seed: Optional[int] = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Tensor, Tensor]:
    """Synthetic CamVid-like street scenes for semantic segmentation.

    Generates the properties that make CamVid hard rather than the photographs themselves
    (which are not redistributable): a wavy **horizon**, a large contiguous **sky**,
    **buildings** standing on the horizon, **trees**, and a **road**, with per-class texture
    (window grids, foliage noise, lane markings) and a global illumination jitter so that the
    classes are not separable by colour alone. Class frequencies are heavily imbalanced, as
    in CamVid.

    Args:
        n: number of scenes.
        size: ``H`` or ``(H, W)``.
        seed: RNG seed (``None`` = unseeded).
        device: device for the returned tensors.

    Returns:
        ``(rgb, labels)`` with ``rgb`` float ``(n, 3, H, W)`` in ``[0, 1]`` and ``labels``
        long ``(n, H, W)`` indexing :data:`SCENE_CLASSES`
        (``0 = sky, 1 = building, 2 = tree, 3 = road``).

    Booleanize the images before handing them to a model, for example with
    :class:`~torchtsetlin.data.ColorThermometerEncoder`, which keeps the ordering between
    intensity levels that a plain threshold would throw away.
    """
    SKY, BUILDING, TREE, ROAD = 0, 1, 2, 3
    H, W = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    if H < 12 or W < 12:
        raise ValueError("scenes need to be at least 12x12")
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    rgb = torch.zeros(n, 3, H, W)
    lab = torch.full((n, H, W), ROAD, dtype=torch.long)
    xs = torch.arange(W).float()
    rows = torch.arange(H).unsqueeze(1)
    base = {
        SKY: (120, 160, 215),
        BUILDING: (150, 145, 135),
        TREE: (70, 120, 65),
        ROAD: (95, 95, 100),
    }
    two_pi = 2 * math.pi

    def rint(lo: int, hi: int) -> int:
        return int(torch.randint(lo, max(lo + 1, hi), (1,), generator=g))

    for i in range(n):
        h0 = rint(H // 3, H // 2)
        amp = float(torch.rand(1, generator=g)) * 2.0
        phase = float(torch.rand(1, generator=g)) * two_pi
        horizon = (h0 + amp * torch.sin(xs / W * two_pi + phase)).round().long().clamp(2, H - 4)
        lab[i] = torch.where(rows < horizon.unsqueeze(0), SKY, ROAD)

        for cls, count, wlo, whi, hlo, hhi in (
            (BUILDING, rint(1, 4), 3, 9, 4, 12),
            (TREE, rint(0, 3), 3, 6, 4, 9),
        ):
            for _ in range(count):
                bw = rint(wlo, whi)
                bx = rint(0, max(1, W - bw))
                bh = rint(hlo, hhi)
                top = max(0, int(horizon[bx : bx + bw].min()) - bh)
                lab[i, top : int(horizon[bx : bx + bw].max()), bx : bx + bw] = cls

        for c, col in base.items():
            m = lab[i] == c
            for ch in range(3):
                rgb[i, ch][m] = col[ch]
        win = (torch.arange(H).view(-1, 1) % 3 == 0) & (torch.arange(W).view(1, -1) % 3 == 0)
        rgb[i][:, (lab[i] == BUILDING) & win] -= 45  # windows
        foliage = lab[i] == TREE
        if bool(foliage.any()):
            rgb[i][:, foliage] += torch.randn(3, int(foliage.sum()), generator=g) * 28
        lane = (
            (lab[i] == ROAD)
            & (torch.arange(W).view(1, -1) % 7 < 2)
            & (torch.arange(H).view(-1, 1) > H * 0.75)
        )
        rgb[i][:, lane] += 40  # lane markings
        rgb[i] += torch.randn(3, H, W, generator=g) * 10
        rgb[i] += float(torch.randn(1, generator=g)) * 18  # illumination
    rgb = rgb.clamp(0, 255) / 255.0
    dev = torch.device(device) if device else None
    return (rgb.to(dev), lab.to(dev)) if dev else (rgb, lab)


def balanced_class_probabilities(
    labels: Tensor,
    n_classes: Optional[int] = None,
    ignore_index: Optional[int] = None,
    floor: float = 0.0,
) -> Tensor:
    """Per-class feedback probabilities that equalise how often each class is learned from.

    Dense targets are dominated by whichever class covers the most pixels, and a Tsetlin
    machine has no loss to reweight — the only lever is *which pixels produce feedback*.
    This returns ``p_k = min_j freq_j / freq_k`` (so the rarest class keeps probability 1),
    ready to pass as ``class_feedback_p`` to
    :class:`~torchtsetlin.models.SegmentationTsetlinMachine`.

    Args:
        labels: any tensor of integer labels (a label map or a flat vector).
        n_classes: number of classes (inferred from the maximum label if omitted).
        ignore_index: label to exclude from the frequency count.
        floor: lower bound on the returned probabilities; raising it (e.g. ``0.05``) keeps a
            very frequent class from being starved of feedback altogether.
    """
    y = labels.reshape(-1).long()
    if ignore_index is not None:
        y = y[y != int(ignore_index)]
    K = int(n_classes) if n_classes is not None else (int(y.max()) + 1 if y.numel() else 1)
    counts = torch.bincount(y, minlength=K).to(torch.float64)[:K]
    present = counts[counts > 0]
    if present.numel() == 0:
        return torch.ones(K, dtype=torch.float32, device=labels.device)
    p = torch.where(counts > 0, present.min() / counts.clamp_min(1), torch.ones_like(counts))
    return p.clamp(min=float(floor), max=1.0).to(torch.float32).to(labels.device)


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


def load_segmentation_boolean(
    dataset: Dataset,
    encoder: Optional[Callable] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    n_bits: int = 4,
    device: Optional[Union[str, torch.device]] = None,
    max_samples: Optional[int] = None,
    ignore_index: int = 255,
    remap: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Load a torchvision *segmentation* dataset into memory as Boolean planes + label maps.

    Segmentation datasets yield ``(image, mask)`` PIL pairs rather than the packed tensors
    :func:`load_torchvision_boolean` expects, and the mask must be resized with **nearest**
    interpolation — bilinear would invent class indices that do not exist. This handles both.

    Args:
        dataset: an instantiated torchvision segmentation dataset, e.g.
            ``VOCSegmentation(root, image_set="train", download=True)`` or
            ``Cityscapes(root, split="train", target_type="semantic")``.
        encoder: maps a float image batch ``(N, 3, H, W)`` in ``[0, 1]`` to Boolean planes.
            Defaults to a :class:`~torchtsetlin.data.ColorThermometerEncoder` with ``n_bits``
            levels per channel, which keeps the ordering between intensities.
        size: ``H`` or ``(H, W)`` to resize to. Dense Tsetlin machines evaluate one patch per
            pixel, so full-resolution frames are expensive — 64-128 px is a sane starting
            point. ``None`` keeps the native size (all images must then already match).
        n_bits: thermometer levels per channel for the default encoder.
        device: device for the returned tensors.
        max_samples: stop after this many images.
        ignore_index: mask value to keep as the void class. Pass the same value as the
            model's ``ignore_index``.
        remap: optional ``{original_label: new_label}`` mapping applied to the masks, for
            collapsing a dataset's classes into a coarser set.

    Returns:
        ``(x_bool, labels)`` with ``x_bool`` ``(N, Z, H, W)`` and ``labels`` ``(N, H, W)``.
    """
    try:
        from torchvision.transforms import functional as VF
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision is required for load_segmentation_boolean") from e

    hw = None if size is None else ((size, size) if isinstance(size, int) else tuple(size))
    n = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    imgs, masks = [], []
    for i in range(n):
        img, mask = dataset[i][0], dataset[i][1]
        if hw is not None:
            img = VF.resize(img, list(hw))
            mask = VF.resize(mask, list(hw), interpolation=VF.InterpolationMode.NEAREST)
        img_t = img if isinstance(img, Tensor) else VF.pil_to_tensor(img)
        mask_t = mask if isinstance(mask, Tensor) else VF.pil_to_tensor(mask)
        imgs.append(img_t.float() / 255.0)
        masks.append(mask_t.squeeze(0).long())
    x = torch.stack(imgs)
    y = torch.stack(masks)
    if remap:
        out = torch.full_like(y, int(ignore_index))
        for src, dst in remap.items():
            out[y == int(src)] = int(dst)
        y = out
    if device is not None:
        x, y = x.to(device), y.to(device)
    if encoder is None:
        from .encoders import ColorThermometerEncoder

        encoder = ColorThermometerEncoder(n_bits=n_bits, value_range=(0.0, 1.0))
        if device is not None:
            encoder = encoder.to(device)
    return encoder(x), y


def load_voc_segmentation_boolean(
    root: str = "./data",
    image_set: str = "train",
    size: Union[int, Tuple[int, int]] = 96,
    n_bits: int = 4,
    device=None,
    download: bool = True,
    max_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Pascal VOC 2012 segmentation as Boolean planes + label maps (21 classes + void).

    Void pixels keep the value ``255``, so pass ``ignore_index=255`` to the model. Requires
    ``torchvision``.
    """
    try:
        from torchvision.datasets import VOCSegmentation
    except ImportError as e:  # pragma: no cover
        raise ImportError("torchvision is required for load_voc_segmentation_boolean") from e
    ds = VOCSegmentation(root=root, image_set=image_set, download=download, **kwargs)
    return load_segmentation_boolean(
        ds, size=size, n_bits=n_bits, device=device, max_samples=max_samples, ignore_index=255
    )
