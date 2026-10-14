"""E3: H4 Potts / H5 diffusion / controls on cached validation unaries, at the native grid.

    python code/refine.py RUN_ID [--split val] [--device cuda:1] [--arms grid|frozen] [--params JSON]

Scores of the frozen model are restored to the native grid (crop letterbox, bilinear), then
refined with the native image as the edge source; the result is evaluated with the same
evaluator as ``run.eval_native``. TM unaries: ``-v/T`` (normalised votes; T from the run).
CNN unaries: ``-log_softmax(logits)`` (``T=1`` on log-probabilities). Writes
``results/E3_refine_<RUN_ID>[_<split>].json`` with one row per arm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import metrics as M  # noqa: E402
import record  # noqa: E402
import run as R  # noqa: E402
import spatial as SP  # noqa: E402


def native_items(cfg, d, split):
    """Yield (id, group, native image (C,H,W) float [0,1], native gt (H,W), geom index list, vol info)."""
    if cfg["dataset"] == "kvasir":
        for i, stem in enumerate(d["ids"]):
            img = np.asarray(Image.open(os.path.join(data.KVASIR_ROOT, "images", stem + ".jpg")).convert("RGB"), np.float32) / 255
            yield stem, d["groups"][i], torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(data.load_kvasir_mask(stem)), i
        return
    man = data.acdc_manifest()
    i = 0
    for pid in man["splits"][split]:
        for fr in man["patients"][pid]["frames"]:
            v, m, _ = data.load_acdc_volume(fr)
            for z in range(v.shape[2]):
                assert int(d["slice"][i]) == z
                yield f"{pid}_{fr['phase']}_{z}", pid, torch.from_numpy(v[:, :, z][None].copy()), torch.from_numpy(m[:, :, z].astype(np.int64)), i
                i += 1


def unaries(cfg, scores, rec):
    if cfg["family"] in ("tm", "coalesced"):
        return scores.float(), float(rec["model_info"]["T"])
    return torch.log_softmax(scores.float(), 0), 1.0


def arms_grid(dataset):
    arms = [("raw", {}), ("majority3", {})]
    for lam in (0.1, 0.3, 1.0):
        for sig in (0.05, 0.15):
            arms.append((f"potts_l{lam}_s{sig}", {"lam": lam, "sigma": sig}))
        arms.append((f"potts_unw_l{lam}", {"lam": lam, "sigma": None}))
    for lam in (0.1, 1.0, 10.0):
        for temp in (0.5, 1.0, 2.0):
            arms.append((f"diff_l{lam}_t{temp}", {"lam": lam, "sigma": 0.15, "temp": temp, "method": "cg"}))
        arms.append((f"diff_unw_l{lam}", {"lam": lam, "sigma": None, "temp": 1.0, "method": "cg"}))
    arms.append(("diff_l1.0_t1.0_explicit100", {"lam": 1.0, "sigma": 0.15, "temp": 1.0, "method": "explicit"}))
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--votes", default=None, help="override votes file (final split)")
    ap.add_argument("--arms", default=None, help="JSON list of [name, params] (default: grid)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--grid", default="native", choices=["native", "model"],
                    help="model: refine on the 128-px letterboxed grid, then restore scores (H5: Q; "
                         "H4/majority: one-hot labels) bilinearly and take argmax on the native grid")
    a = ap.parse_args()
    with open(os.path.join(record.RESULTS, a.run_id + ".json")) as f:
        rec = json.load(f)
    cfg = rec["config"]
    K = R.K_OF[cfg["dataset"]]
    if a.split == "final":
        os.environ["SEG_UNSEAL"] = "1"
    d = R.load_data(cfg, a.split)
    vpath = a.votes or os.path.join(record.HERE, rec["val_votes"])
    scores = torch.load(vpath)
    arms = json.loads(a.arms) if a.arms else arms_grid(cfg["dataset"])
    S = cfg["size"]
    evs = {n: M.SegEvaluator(K, tols=(1, 2, "n")) for n, _ in arms}
    stats = {n: {"time_s": 0.0, "changed": 0, "valid": 0, "new_errors": 0, "fixed_errors": 0, "nonconverged": 0} for n, _ in arms}
    vol_preds = {n: {} for n, _ in arms}
    dev = a.device
    for id_, grp, img, gt, i in native_items(cfg, d, a.split):
        if a.grid == "model":
            U, T = unaries(cfg, scores[i].float(), rec)
            im = (d["x"][i].float() / 255.0).unsqueeze(0).to(dev)
            valid = (d["y"][i] != 255).unsqueeze(0).to(dev)
        else:
            s = data.restore_scores(scores[i].float(), d["geom"][i])
            U, T = unaries(cfg, s, rec)
            im = img.unsqueeze(0).to(dev)
            valid = torch.ones(1, *gt.shape, dtype=torch.bool, device=dev)
        U = U.unsqueeze(0).to(dev)

        def to_native(lbl=None, Q=None):
            if a.grid == "native":
                return lbl if Q is None else Q.argmax(1)
            sc = Q[0] if Q is not None else torch.nn.functional.one_hot(lbl[0], K).permute(2, 0, 1).float()
            return data.restore_scores(sc.cpu(), d["geom"][i]).argmax(0).unsqueeze(0).to(dev)

        raw = to_native(Q=U) if a.grid == "model" else U.argmax(1)
        ntol = max(1, round(2 * max(gt.shape) / S))
        for name, p in arms:
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            if name == "raw":
                lbl = raw
            elif name.startswith("majority"):
                lbl = to_native(SP.majority_filter(U, valid, 3))
            elif name.startswith("potts"):
                lbl = to_native(SP.potts_icm(U, im, valid, T, p["lam"], p["sigma"], 10)[0])
            else:
                Q, info = SP.diffuse(U, im, valid, T, p["lam"], p["sigma"], p["temp"], p["method"], 100, 1e-4)
                lbl = to_native(Q=Q)
                stats[name]["nonconverged"] += int(not info["converged"])
            torch.cuda.synchronize(dev)
            stats[name]["time_s"] += time.perf_counter() - t
            g = gt.to(dev)
            stats[name]["changed"] += int((lbl != raw).sum())
            stats[name]["valid"] += int(gt.numel())
            stats[name]["new_errors"] += int(((lbl != raw) & (raw[0] == g) & (lbl[0] != g)).sum())
            stats[name]["fixed_errors"] += int(((lbl != raw) & (raw[0] != g) & (lbl[0] == g)).sum())
            lc = lbl[0].cpu()
            evs[name].add(lc, gt, id_, ntol)
            if cfg["dataset"] == "acdc":
                vol_preds[name].setdefault(id_.rsplit("_", 1)[0], []).append(lc.numpy().astype(np.uint8))
    rows = []
    gts = data.acdc_native_masks(a.split) if cfg["dataset"] == "acdc" else None
    for name, p in arms:
        st = evs[name].stats()
        st["groups"] = [x.split("_")[0] for x in st["ids"]] if cfg["dataset"] == "acdc" else d["groups"]
        row = {"arm": name, "params": p, "summary": evs[name].summary(), "stats": st,
               "changed_pixel_rate": stats[name]["changed"] / max(1, stats[name]["valid"]),
               "new_errors": stats[name]["new_errors"], "fixed_errors": stats[name]["fixed_errors"],
               "nonconverged": stats[name]["nonconverged"],
               "ms_per_image": 1000 * stats[name]["time_s"] / len(st["ids"])}
        if cfg["dataset"] == "acdc":
            vols = []
            for vi, v in enumerate(d["vols"]):
                pred = np.stack(vol_preds[name][f"{v['patient']}_{v['phase']}"], 2)
                vols.append({"patient": v["patient"], "phase": v["phase"], "group": v["group"],
                             **M.acdc_volume_metrics(pred, gts[vi], v["spacing"])})
            row["volumes"] = vols
            fg = [np.mean([r[k] for k in ("dice_RV", "dice_MYO", "dice_LV") if r[k] is not None]) for r in vols]
            row["summary"]["vol_fg_dice"] = float(np.mean(fg))
            for key in ("hd95_RV", "hd95_MYO", "hd95_LV"):
                vals = [r[key] for r in vols if r[key] is not None]
                row["summary"][f"vol_{key}"] = float(np.mean(vals)) if vals else None
        s = row["summary"]
        print(f"{name:28s} mIoU={s['miou']:.4f} fgDice={s.get('fg_dice') or 0:.4f} bf1n={s.get('bf1_tn_fg_mean') or 0:.4f} "
              f"chg={row['changed_pixel_rate']:.4f} ms={row['ms_per_image']:.1f}", flush=True)
        rows.append(row)
    rid = f"E3_refine_{a.run_id}{'_' + a.split if a.split != 'val' else ''}{'_modelgrid' if a.grid == 'model' else ''}{a.tag}"
    record.write_record({"run_id": rid, "status": "completed", "source_run": a.run_id, "split": a.split,
                         "env": record.environment(dev), "rows": rows})


if __name__ == "__main__":
    main()
