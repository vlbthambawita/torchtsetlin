"""Phase 1 shared machinery: data, causal contexts, calibration, metrics.

Design decisions carried over from gate G0 (see ../../FINDINGS_G0.md):
  * batch <= 50                       (Finding 5: batched-feedback cliff)
  * s = 40, T ~ C/32                  (Findings 1 and 3)
  * assert empty_clauses == 0         (Finding 4: detects every collapse)
  * report analytic AND isotonic      (Finding 2)
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]          # experiments/generation
REPO = ROOT.parents[1]                              # repo root
OUT = ROOT / "results" / "phase1"
H = W = 28
NPIX = H * W


# --------------------------------------------------------------------------- data
def load_mnist(device, threshold=0.3, seed=0):
    """Binarised MNIST split by IMAGE into 50k train / 10k val / 10k test."""
    from torchtsetlin.data import load_mnist_boolean

    xtr, ytr = load_mnist_boolean(root=str(REPO / "data"), train=True, threshold=threshold)
    xte, yte = load_mnist_boolean(root=str(REPO / "data"), train=False, threshold=threshold)
    xtr = xtr.reshape(-1, H, W).to(device)
    xte = xte.reshape(-1, H, W).to(device)
    ytr, yte = ytr.to(device), yte.to(device)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(xtr.shape[0], generator=g).to(device)
    tr, va = perm[:50000], perm[50000:]
    return {
        "train": (xtr[tr], ytr[tr]),
        "val": (xtr[va], ytr[va]),
        "test": (xte, yte),
    }


def thermometer(v, n):
    """v >= k for k = 1..n  ->  (B, n) bool."""
    ks = torch.arange(1, n + 1, device=v.device)
    return v.unsqueeze(1) >= ks


# --------------------------------------------------------------------------- contexts
class WindowContext:
    """Causal neighbourhood: `rows_above` full rows above, `left` pixels to the left.

    Nothing at or after the target position can enter the context — enforced by
    construction (only strictly-earlier raster positions are addressed) and checked by
    `selftest_no_leakage`.
    """

    name = "window"

    def __init__(self, rows_above=2, left=4, n_digits=10):
        self.rows_above, self.left, self.n_digits = rows_above, left, n_digits
        self.n_window = rows_above * W + left
        self.n_features = self.n_window + (H - 1) + (W - 1) + n_digits

    def pad(self, imgs):
        n = imgs.shape[0]
        p = torch.zeros(n, H + self.rows_above, W + self.left,
                        dtype=torch.bool, device=imgs.device)
        p[:, self.rows_above:, self.left:] = imgs
        return p

    def encode(self, padded, idx, pos, digit):
        r, c = pos // W, pos % W
        ra, le = self.rows_above, self.left
        parts = []
        for k in range(1, ra + 1):                       # full rows above
            parts.append(padded[idx, r + ra - k, le:le + W])
        if le:                                           # pixels to the left, same row
            cols = c.unsqueeze(1) + torch.arange(le, device=pos.device)
            parts.append(padded[idx.unsqueeze(1), (r + ra).unsqueeze(1), cols])
        parts.append(thermometer(r, H - 1))
        parts.append(thermometer(c, W - 1))
        parts.append(torch.nn.functional.one_hot(digit, self.n_digits).bool())
        return torch.cat(parts, dim=1)


class CanvasContext:
    """The whole partial canvas plus the `observed` mask — the original design, with
    VALIDITY.md E1 applied (the redundant 784-wide position one-hot is dropped, since
    `observed` already IS the thermometer of the position under raster order)."""

    name = "canvas"

    def __init__(self, n_digits=10):
        self.n_digits = n_digits
        self.n_features = NPIX + NPIX + n_digits

    def pad(self, imgs):
        return imgs.reshape(imgs.shape[0], NPIX)

    def encode(self, flat, idx, pos, digit):
        ar = torch.arange(NPIX, device=pos.device)
        observed = ar.unsqueeze(0) < pos.unsqueeze(1)          # (B, 784)
        known = flat[idx] & observed
        return torch.cat([known, observed,
                          torch.nn.functional.one_hot(digit, self.n_digits).bool()], dim=1)


def selftest_no_leakage(ctx, device, n=256, seed=0):
    """A context must not change when pixels at or after the target position change."""
    g = torch.Generator(device=device).manual_seed(seed)
    imgs = torch.randint(0, 2, (n, H, W), generator=g, device=device).bool()
    digit = torch.randint(0, 10, (n,), generator=g, device=device)
    pos = torch.randint(0, NPIX, (n,), generator=g, device=device)
    idx = torch.arange(n, device=device)

    a = ctx.encode(ctx.pad(imgs), idx, pos, digit)
    flipped = imgs.reshape(n, NPIX).clone()
    ar = torch.arange(NPIX, device=device)
    future = ar.unsqueeze(0) >= pos.unsqueeze(1)               # target and everything after
    flipped[future] = ~flipped[future]
    b = ctx.encode(ctx.pad(flipped.reshape(n, H, W)), idx, pos, digit)
    assert torch.equal(a, b), f"{ctx.name}: context depends on the target or future pixels"

    # and it must actually depend on the past
    past = ~future
    flipped2 = imgs.reshape(n, NPIX).clone()
    flipped2[past] = ~flipped2[past]
    c = ctx.encode(ctx.pad(flipped2.reshape(n, H, W)), idx, pos, digit)
    assert not torch.equal(a, c), f"{ctx.name}: context ignores the past entirely"
    return True


# --------------------------------------------------------------------------- calibration
def fit_isotonic(v, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        v.detach().cpu().numpy().astype("float64"), y.detach().cpu().numpy().astype("float64"))

    def f(x):
        import numpy as np
        return torch.as_tensor(
            np.clip(iso.predict(x.detach().cpu().numpy().astype("float64")), 1e-6, 1 - 1e-6),
            dtype=torch.float32, device=x.device)
    return f


def analytic_p(v, T):
    return ((v + T) / (2.0 * T)).clamp(1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------- records
def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(REPO), text=True).strip()
    except Exception:
        return "unknown"


def write_record(arm, config, result, wall, subdir="phase1"):
    d = ROOT / "results" / subdir
    d.mkdir(parents=True, exist_ok=True)
    rec = {"arm": arm, "config": config, "result": result, "wall_s": round(wall, 2),
           "git_sha": git_sha(), "torch": torch.__version__, "host": platform.node(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / f"{arm}__{int(time.time()*1000)}.json").write_text(json.dumps(rec, indent=2))
    return rec
