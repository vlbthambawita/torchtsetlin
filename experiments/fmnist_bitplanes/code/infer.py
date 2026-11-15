"""Reload a saved best-validation model and run inference (plans/fmnist-bitplanes-augmentation.md §5).

1. Rebuild the model from the file's hp, load the state, predict the test set in eval() mode and
   assert the accuracy equals the one stored at save time (load/rebuild check).
2. Confusion matrix, per-class accuracy.
3. Shift test (QA4): test accuracy for every shift (dy, dx) in [-R, R]^2 (zero padding, as in
   training) and for the horizontally flipped test set.
Writes results/inference/<arm>/<tag>.json.

Usage: python code/infer.py models/aug/<tag>.pt [--device cuda:1] [--radius 3] [--no-shift]
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from augment import encode_gpu
from common import RESULTS, load_data, new_record, write_json
from train import build_model, class_sums


def load_model(path, dev):
    st = torch.load(path, map_location="cpu", weights_only=False)
    hp = st["hp"]
    model = build_model(hp["arch"], hp["planes"], hp["clauses_per_class"], hp["T"], hp["s"], hp["kd"])
    model.load_state_dict(st["state_dict"])
    return model.to(dev).eval(), st


def to_model_input(arch, x):                          # x: (N, P, 28, 28) bool on the device
    if arch == "flat":
        return x.reshape(len(x), -1)
    if arch == "conv3":
        return x.unsqueeze(1)
    return x


def shifted(x_u8, dy, dx):
    """Zero-padded translation of uint8 images (N, H, W): content moves by (dy, dx)."""
    out = torch.zeros_like(x_u8)
    H, W = x_u8.shape[1:]
    out[:, max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = \
        x_u8[:, max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--no-shift", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=1000)
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, st = load_model(args.model, dev)
    hp, arch, enc = st["hp"], st["hp"]["arch"], st["hp"]["enc"]
    d = load_data(enc)
    n_te = hp["data"]["n_test"]
    yte = d["y_test"][:n_te]
    xte = torch.from_numpy(np.ascontiguousarray(d["x_test"][:n_te])).to(dev).bool()

    def acc(x):
        p = class_sums(model, to_model_input(arch, x), args.eval_batch).argmax(1).cpu().numpy()
        return p, 100 * float((p == yte).mean())

    pred, test_acc = acc(xte)
    model.eval()                                      # class_sums switches back to train()
    ok = abs(test_acc - st["test_acc"]) < 1e-9
    print(f"{st['tag']}: epoch {st['epoch']}, test {test_acc:.2f} (saved {st['test_acc']:.2f}) "
          f"{'reproduced' if ok else 'MISMATCH'}", flush=True)
    cm = np.zeros((10, 10), np.int64)
    np.add.at(cm, (yte, pred), 1)
    rec = new_record("inference", device=dev, tag=st["tag"], arm=st["arm"], model=str(args.model),
                     epoch=st["epoch"], hp=hp, test_acc=test_acc, saved_test_acc=st["test_acc"],
                     reproduced=ok, confusion=cm.tolist(),
                     test_acc_per_class=(100 * np.diag(cm) / cm.sum(1)).tolist())
    if not args.no_shift:
        if enc not in ("bit8", "gray8"):
            raise SystemExit("shift test needs bit8/gray8 (GPU re-encoding)")
        raw = torch.from_numpy(np.ascontiguousarray(d["raw_test"][:n_te])).to(dev)
        R = args.radius
        grid = np.zeros((2 * R + 1, 2 * R + 1))
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                grid[dy + R, dx + R] = acc(encode_gpu(enc, shifted(raw, dy, dx)))[1]
        flip_acc = acc(encode_gpu(enc, raw.flip(-1)))[1]
        rec.update(shift_radius=R, shift_grid=grid.tolist(), flip_test_acc=flip_acc,
                   shift_mean_by_distance={str(k): float(np.mean([grid[dy + R, dx + R]
                                           for dy in range(-R, R + 1) for dx in range(-R, R + 1)
                                           if max(abs(dy), abs(dx)) == k])) for k in range(R + 1)})
        print("  accuracy by shift distance (Chebyshev):", {k: round(v, 2) for k, v in rec["shift_mean_by_distance"].items()},
              f" flipped: {flip_acc:.2f}", flush=True)
    write_json(rec, RESULTS / "inference" / st["arm"] / f"{st['tag']}.json")
    if not ok:
        raise SystemExit("reloaded model does not reproduce its saved test accuracy")


if __name__ == "__main__":
    main()
