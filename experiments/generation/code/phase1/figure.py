"""Render a grid of generated digits next to real ones."""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-digit", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)

    d = torch.load(a.samples, map_location="cpu", weights_only=False)
    s, dig = d["samples"], d["digits"]
    real = C.load_mnist(dev)["test"]
    rx, ry = real[0].cpu(), real[1].cpu()

    n = a.per_digit
    fig, axes = plt.subplots(10, 2 * n + 1, figsize=(2 * n + 1, 10.5))
    for r in range(10):
        gi = (dig == r).nonzero().flatten()[:n]
        ri = (ry == r).nonzero().flatten()[:n]
        for c in range(n):
            axes[r, c].imshow(s[gi[c]], cmap="gray_r", vmin=0, vmax=1)
            axes[r, n + 1 + c].imshow(rx[ri[c]], cmap="gray_r", vmin=0, vmax=1)
        axes[r, n].axis("off")
        for c in range(2 * n + 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(False)
    axes[0, n // 2].set_title("generated", fontsize=11)
    axes[0, n + 1 + n // 2].set_title("real", fontsize=11)
    fig.suptitle("Autoregressive Tsetlin machine — binarised MNIST", fontsize=12)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
