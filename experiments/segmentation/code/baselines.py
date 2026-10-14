"""CNN and trivial baselines (PLAN B0, B4, B5)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

IGNORE = 255


class UNet(nn.Module):
    """Compact U-Net: ``levels`` resolutions, two 3x3 conv+BN+ReLU per level, skip connections."""

    def __init__(self, in_ch: int, K: int, width: int = 16, levels: int = 4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            )

        ch = [width * 2**i for i in range(levels)]
        self.down = nn.ModuleList([block(in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(levels - 1, 0, -1)])
        self.dec = nn.ModuleList([block(2 * ch[i - 1], ch[i - 1]) for i in range(levels - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], K, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


class LocalCNN(nn.Module):
    """Receptive field ``rf x rf`` (one spatial conv) followed by 1x1 layers — same RF as B1."""

    def __init__(self, in_ch: int, K: int, width: int = 64, rf: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, rf, padding=rf // 2), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 1), nn.ReLU(inplace=True),
            nn.Conv2d(width, K, 1),
        )

    def forward(self, x):
        return self.net(x)


class PixelLinear(nn.Module):
    """B0 per-pixel linear classifier on colour/intensity (a 1x1 convolution)."""

    def __init__(self, in_ch: int, K: int):
        super().__init__()
        self.lin = nn.Conv2d(in_ch, K, 1)

    def forward(self, x):
        return self.lin(x)


def masked_loss(logits, y, K: int, dice_w: float = 0.0):
    ce = F.cross_entropy(logits, y, ignore_index=IGNORE)
    if dice_w <= 0:
        return ce
    valid = (y != IGNORE).unsqueeze(1).float()
    yy = torch.where(y == IGNORE, torch.zeros_like(y), y)
    oh = F.one_hot(yy, K).permute(0, 3, 1, 2).float() * valid
    p = logits.softmax(1) * valid
    inter = (p * oh).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return ce + dice_w * dice
