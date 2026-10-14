"""Frozen validation stress sets (PLAN section 4): every severity is reported, none selected.

    python code/stress.py RUN_ID [RUN_ID ...] [--split val] [--device cuda:1]

Photometric shifts change only the image; the geometric "zoom-out" shift shrinks image and
mask together and marks newly uncovered pixels as ignore. Screen-grid metrics.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402
import record  # noqa: E402
import run as R  # noqa: E402

IGNORE = 255


def _photo(fn):
    def t(x, y):
        xf = x.float() / 255.0
        return (fn(xf).clamp(0, 1) * 255).round().to(torch.uint8), y
    return t


def _noise(sigma):
    def fn(xf):
        g = torch.Generator().manual_seed(4242)
        return xf + sigma * torch.randn(xf.shape, generator=g)
    return fn


def _bias(xf):  # smooth multiplicative ramp 0.8 -> 1.2 across the image (MRI bias field)
    W = xf.shape[-1]
    ramp = torch.linspace(0.8, 1.2, W).view(1, 1, 1, W)
    return xf * ramp


def _zoom(f):
    def t(x, y):
        N, C, H, W = x.shape
        h, w = round(H * f), round(W * f)
        xs = F.interpolate(x.float(), size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        ys = F.interpolate(y.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
        xo = torch.zeros(N, C, H, W)
        yo = torch.full((N, H, W), IGNORE, dtype=torch.float32)
        t0, l0 = (H - h) // 2, (W - w) // 2
        xo[:, :, t0 : t0 + h, l0 : l0 + w] = xs
        yo[:, t0 : t0 + h, l0 : l0 + w] = ys
        return xo.round().clamp(0, 255).to(torch.uint8), yo.to(torch.uint8)
    return t


STRESS = {
    "kvasir": {
        "bright+0.1": _photo(lambda x: x + 0.1), "bright-0.1": _photo(lambda x: x - 0.1),
        "gamma0.7": _photo(lambda x: x.clamp_min(0) ** 0.7), "gamma1.4": _photo(lambda x: x.clamp_min(0) ** 1.4),
        "noise0.05": _photo(_noise(0.05)), "zoom0.75": _zoom(0.75),
    },
    "acdc": {
        "bias0.8-1.2": _photo(_bias), "noise0.05": _photo(_noise(0.05)),
        "gamma0.7": _photo(lambda x: x.clamp_min(0) ** 0.7), "gamma1.4": _photo(lambda x: x.clamp_min(0) ** 1.4),
        "zoom0.75": _zoom(0.75),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    for rid in a.runs:
        rec = evalio.load_run(rid)
        ds = rec["config"]["dataset"]
        rows = []
        for name, tf in [("clean", None)] + list(STRESS[ds].items()):
            cfg, d, scores = evalio.predict(rec, a.split, a.device, tf)
            summ, st = R.eval_screen(scores, d, R.K_OF[ds])
            rows.append({"stress": name, "summary": summ, "stats": st})
            print(rid, name, round(summ["miou"], 4), flush=True)
        record.write_record({"run_id": f"STRESS_{rid}_{a.split}", "status": "completed", "source_run": rid,
                             "split": a.split, "env": record.environment(a.device), "rows": rows,
                             "written_at": time.time()})


if __name__ == "__main__":
    main()
