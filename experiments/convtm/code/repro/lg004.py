"""LG-004 reproduction: the default chunk budget serialises large convolutional models.

Prints, for a batch of 50 CIFAR-10-shaped Boolean images, how many examples per chunk the
library actually processes as a function of clause count and ``max_chunk_elements``. A chunk
size below the batch size means the mini-batch is executed as several sequential passes; a
chunk size of 1 means the batch is fully serialised. Nothing about the *result* changes --
only the throughput.

    python code/repro/lg004.py            # CPU only, no data, seconds
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchtsetlin as tt  # noqa: E402

BATCH = 50
SHAPES = {"therm4": (12, 32, 32), "therm8": (24, 32, 32)}


def chunk_for(n_clauses: int, patch: int, bl: str, max_chunk: int) -> int:
    Z, H, W = SHAPES[bl]
    m = tt.ConvTsetlinMachine(10, max(2, n_clauses // 10), 0.8 * n_clauses / 10, 10.0,
                              patch_size=patch, stride=1, position_encoding=True,
                              input_shape=(Z, H, W), max_chunk_elements=max_chunk)
    xb = torch.zeros(BATCH, Z, H, W, dtype=torch.bool)
    return m._chunk_size(xb)


if __name__ == "__main__":
    default = 2 ** 27
    for bl in ("therm4",):
        for patch in (4, 10):
            print(f"\n{bl}, {patch}x{patch} patches, batch {BATCH}, "
                  f"default max_chunk_elements = 2**27 = {default}")
            print(f"  {'clauses':>8} | " + " | ".join(
                f"2**{e:<2d}" for e in (27, 29, 31, 33)))
            for C in (200, 640, 2000, 4000, 8000, 16000, 32000):
                cells = [f"{chunk_for(C, patch, bl, 2 ** e):>4d}" for e in (27, 29, 31, 33)]
                print(f"  {C:>8d} | " + " | ".join(f"{c:>4s}" for c in cells))
    print("\nchunk size < batch size  => the mini-batch is executed as several sequential "
          "passes\nchunk size == 1          => fully serialised")
