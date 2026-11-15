"""Training-time augmentation on uint8 pixels, before Booleanization (plans/fmnist-bitplanes-augmentation.md §2).

Booleanizing first and shifting bits would be equivalent for a shift, but a pixel's bit planes must
move together and no interpolation may create values that are not uint8 codes, so augmentation
works on the raw images and the encoding runs afterwards on the GPU (``encode_gpu``; checked bit
for bit against ``common.encode`` in ``check_augment.py``).

Policies (``common.AUGMENTS``): a random translation by (dy, dx) in [-shift, shift]^2 with zero
padding (black = Fashion-MNIST background), then a horizontal flip with probability ``flip_p``.
The draws come from a dedicated CPU generator (seed 2000 + seed), one vector per epoch, so the
augmentation stream is reproducible and independent of the batch order (1000 + seed) and of the
model's own randomness.
"""

import torch


class Augmenter:
    def __init__(self, raw_u8, shift, flip_p, gen):
        """raw_u8: (N, H, W) uint8 tensor on the training device."""
        self.n, self.h, self.w = raw_u8.shape
        self.shift, self.flip_p, self.gen = int(shift), float(flip_p), gen
        s = self.shift
        self.padded = torch.nn.functional.pad(raw_u8, (s, s, s, s))           # (N, H+2s, W+2s)
        dev = raw_u8.device
        self.ar_h = torch.arange(self.h, device=dev)
        self.ar_w = torch.arange(self.w, device=dev)

    def draw_epoch(self):
        """Per-image offsets and flips for one epoch, from the CPU generator (moved to the device)."""
        dev = self.padded.device
        off = torch.randint(0, 2 * self.shift + 1, (self.n, 2), generator=self.gen)
        flip = torch.rand(self.n, generator=self.gen) < self.flip_p
        self.off, self.flip = off.to(dev), flip.to(dev)

    def batch(self, idx):
        """Augmented uint8 images (B, H, W) for training indices ``idx`` (a device tensor)."""
        oy, ox = self.off[idx, 0], self.off[idx, 1]
        rows = oy[:, None] + self.ar_h                                          # (B, H)
        cols = ox[:, None] + torch.where(self.flip[idx, None], self.w - 1 - self.ar_w, self.ar_w)
        return self.padded[idx[:, None, None], rows[:, :, None], cols[:, None, :]]


def encode_gpu(enc, x_u8):
    """uint8 (B, H, W) -> bool planes (B, 8, H, W), MSB first; same as common.encode for bit8/gray8."""
    if enc == "gray8":
        x_u8 = x_u8 ^ (x_u8 >> 1)
    elif enc != "bit8":
        raise ValueError(f"augmentation supports bit8/gray8 only, not {enc}")
    sh = torch.arange(7, -1, -1, device=x_u8.device, dtype=torch.uint8)
    return ((x_u8[:, None] >> sh[None, :, None, None]) & 1).bool()
