"""Minimal segmentation Tsetlin machine on Kvasir-SEG: prints train and validation mIoU per epoch.

    python simple_train.py [epochs] [device]        e.g.  python simple_train.py 20 cuda:0

Needs the image cache of this folder (./setup.sh or python code/data.py builds it).
"""
import sys
from pathlib import Path

import torch
import torchtsetlin as tt

SIZE = 64            # image size: 64, 128 or 256 (256 is about 16 times slower per epoch)
CLAUSES = 400        # clauses per class
PATCH = 3            # each pixel is classified from its PATCH x PATCH neighbourhood
BATCH = 8            # images per update
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DEVICE = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"

# data: uint8 images (N, 3, S, S) and masks (N, S, S) with 0 = background, 1 = polyp
d = torch.load(Path(__file__).parent / ".cache" / "data" / f"kvasir_{SIZE}.pt", weights_only=False)
encoder = tt.data.ColorThermometerEncoder(n_bits=4, value_range=(0.0, 1.0))      # 4 Boolean planes per colour
x = encoder(d["x"].float() / 255.0).bool()
y = d["y"].long()
x_train, y_train = x[d["split"]["train"]], y[d["split"]["train"]]
x_val, y_val = x[d["split"]["val"]], y[d["split"]["val"]]

model = tt.SegmentationTsetlinMachine(2, n_clauses=CLAUSES, T=CLAUSES // 5, s=10.0, patch_size=PATCH,
                                      position_encoding=8, weighted=True, n_states=100).to(DEVICE)


def miou(xs, ys):
    """mIoU of the current model on a whole split (no learning)."""
    model.eval()
    pred = torch.cat([model.predict(xs[i:i + 32].to(DEVICE)).cpu() for i in range(0, len(xs), 32)])
    model.train()
    return float(tt.metrics.mean_iou(pred, ys, n_classes=2))


for epoch in range(1, EPOCHS + 1):
    order = torch.randperm(len(x_train))
    for i in range(0, len(order), BATCH):
        idx = order[i:i + BATCH]
        model.update(x_train[idx].to(DEVICE), y_train[idx].to(DEVICE))
    print(f"epoch {epoch:3d}   train mIoU {miou(x_train, y_train):.3f}   val mIoU {miou(x_val, y_val):.3f}", flush=True)
