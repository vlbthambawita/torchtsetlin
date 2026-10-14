"""Tables for the report, generated only from run records (PLAN section 9).

    python code/analyze.py            # writes report/tables.md and report/summary.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
import record  # noqa: E402

OUT = os.path.join(record.HERE, "report")


def arm_of(rid):
    return re.sub(r"_s\d+$", "", rid)


def fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def ms(vals, d=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}"
    return f"{np.mean(vals):.{d}f} ± {np.std(vals, ddof=1):.{d}f}"


def group(recs):
    g = defaultdict(list)
    for rid, r in recs.items():
        g[arm_of(rid)].append(r)
    return g


def get(r, key, native=False):
    src = r.get("val_native") if native and r.get("val_native") else r.get("val_screen") or {}
    return src.get(key)


def table(recs, cols, native=False, title=""):
    lines = [f"### {title}", "", "| arm | n | " + " | ".join(c for c, _ in cols) + " |", "|---|---|" + "---|" * len(cols)]
    g = group(recs)
    for arm in sorted(g, key=lambda a: -np.mean([get(r, "miou", native) or 0 for r in g[a]])):
        rs = [r for r in g[arm] if r["status"] == "completed"]
        fails = [r["status"] for r in g[arm] if r["status"] != "completed"]
        cells = []
        for _, fn in cols:
            cells.append(ms([fn(r) for r in rs]))
        lines.append(f"| {arm} | {len(rs)}{' (+' + ','.join(fails) + ')' if fails else ''} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def screen_cols(K):
    cols = [("mIoU", lambda r: get(r, "miou"))]
    cols += [(f"IoU{k}", (lambda k: lambda r: (get(r, "iou") or [None] * K)[k])(k)) for k in range(K)]
    cols += [("BF1 t2 (fg)", lambda r: get(r, "bf1_t2_fg_mean")), ("epoch", lambda r: r.get("selected_epoch")),
             ("train min", lambda r: (r.get("train_time_s") or 0) / 60)]
    return cols


def native_cols(ds):
    cols = [("mIoU", lambda r: get(r, "miou", True)), ("fg Dice (pooled)", lambda r: get(r, "fg_dice", True)),
            ("mean img fg Dice", lambda r: get(r, "mean_img_fg_dice", True)),
            ("BF1 norm-tol (fg)", lambda r: get(r, "bf1_tn_fg_mean", True)), ("BF1 t2 (fg)", lambda r: get(r, "bf1_t2_fg_mean", True))]
    if ds == "acdc":
        cols += [("3D Dice RV", lambda r: get(r, "vol_dice_RV", True)), ("3D Dice MYO", lambda r: get(r, "vol_dice_MYO", True)),
                 ("3D Dice LV", lambda r: get(r, "vol_dice_LV", True)), ("HD95 mean mm", lambda r: _hd(r))]
    cols += [("epoch", lambda r: r.get("selected_epoch")), ("train min", lambda r: (r.get("train_time_s") or 0) / 60)]
    return cols


def _hd(r):
    v = [get(r, f"vol_hd95_{c}", True) for c in ("RV", "MYO", "LV")]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def paired(recs_a, recs_b, K, metric="miou", native=False):
    """Bootstrap B - A over matched seeds."""
    key = "val_native_stats" if native else "val_screen_stats"
    sa = {r["config"]["seed"]: r[key] for r in recs_a if r.get(key)}
    sb = {r["config"]["seed"]: r[key] for r in recs_b if r.get(key)}
    common = sorted(set(sa) & set(sb))
    if not common:
        return None
    return M.paired_bootstrap([sa[s] for s in common], [sb[s] for s in common], K, metric)


def main():
    os.makedirs(OUT, exist_ok=True)
    allr = record.load_records()
    parts = []
    for phase in ("E0", "E1", "E2", "E4"):
        for ds, K in (("scenes", 4), ("shapes", 3), ("longrange", 3), ("kvasir", 2), ("acdc", 4)):
            rs = {k: v for k, v in allr.items() if k.startswith(f"{phase}_{ds}_")}
            if not rs:
                continue
            parts.append(table(rs, screen_cols(K), False, f"{phase} {ds} — validation, model grid"))
            if ds in ("kvasir", "acdc") and any(r.get("val_native") for r in rs.values()):
                parts.append(table({k: v for k, v in rs.items() if v.get("val_native")}, native_cols(ds), True,
                                   f"{phase} {ds} — validation, native grid"))
    with open(os.path.join(OUT, "tables.md"), "w") as f:
        f.write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
