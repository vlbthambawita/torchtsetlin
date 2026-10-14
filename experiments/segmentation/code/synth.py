"""D0/D1 synthetic datasets as uint8 image tensors (PLAN section 4).

All generators return ``x`` uint8 ``(n, C, H, W)`` and ``y`` uint8 ``(n, H, W)``. Splits use
independent seeds: train ``s``, val ``s + 1``, final ``s + 2``.
"""

from __future__ import annotations

from typing import Tuple

import torch

import torchtsetlin as tt

SEEDS = {"train": 101, "val": 102, "final": 103}
SIZES = {"train": 1000, "val": 200, "final": 200}


def scenes(split: str, size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    rgb, lab = tt.data.make_scenes(SIZES[split], size, seed=SEEDS[split])
    return (rgb * 255).round().to(torch.uint8), lab.to(torch.uint8)


def shapes(split: str, size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y = tt.data.make_segmentation_shapes(SIZES[split], size=size, seed=SEEDS[split])
    return x.to(torch.uint8) * 255, y.to(torch.uint8)


def longrange(split: str, size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """D1 long-range ambiguity: every image holds two identical 8x8 textured squares; a
    distant cue (a 4x4 bright marker in the top-left *or* bottom-right corner region) decides
    whether both squares are class 1 or class 2. Square interiors and their local
    neighbourhoods are pixel-identical across the two classes (collision checked below), so
    a local-only predictor is at chance on square pixels. Pairs (same layout, flipped cue)
    are generated consecutively and stay in one split."""
    n = SIZES[split]
    g = torch.Generator().manual_seed(SEEDS[split] + 1000)
    x = torch.zeros(n, 1, size, size)
    y = torch.zeros(n, size, size, dtype=torch.uint8)
    tex = (torch.rand(8, 8, generator=g) > 0.5).float() * 0.5 + 0.4  # fixed square texture
    for i in range(0, n, 2):
        bg = torch.rand(size, size, generator=g) * 0.25
        boxes = []
        while len(boxes) < 2:
            cy, cx = int(torch.randint(16, size - 16, (1,), generator=g)), int(torch.randint(16, size - 16, (1,), generator=g))
            if all(abs(cy - a) > 10 or abs(cx - b) > 10 for a, b in boxes):
                boxes.append((cy, cx))
        for j, cls in ((i, 1), (i + 1, 2)):
            if j >= n:
                break
            im = bg.clone()
            for cy, cx in boxes:
                im[cy - 4 : cy + 4, cx - 4 : cx + 4] = tex
                y[j, cy - 4 : cy + 4, cx - 4 : cx + 4] = cls
            if cls == 1:
                im[2:6, 2:6] = 1.0
            else:
                im[size - 6 : size - 2, size - 6 : size - 2] = 1.0
            x[j, 0] = im
    return (x * 255).round().to(torch.uint8), y


def load(name: str, split: str):
    fn = {"scenes": scenes, "shapes": shapes, "longrange": longrange}[name]
    x, y = fn(split)
    return {"x": x, "y": y, "ids": list(range(len(x)))}
