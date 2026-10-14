"""Generate report/REPORT.md and report/fig/*.png from run records only.

    python code/report.py

Every number printed in the report is read from ``results/*.json`` here; prose that states a
direction ("better", "worse") is conditioned on those numbers, so a different set of
records produces a different report rather than a wrong one.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import record  # noqa: E402
import stats as ST  # noqa: E402

REP = os.path.join(record.HERE, "report")
FIG = os.path.join(REP, "fig")
R = record.load_records()
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
K_OF = {"kvasir": 2, "acdc": 4, "scenes": 4, "shapes": 3, "longrange": 3}


# ------------------------------------------------------------------ helpers
def runs(prefix, status="completed"):
    """Records of one arm (all seeds), latest completed attempt per seed for re-run suffixes."""
    out = {}
    for rid, r in R.items():
        if re.fullmatch(re.escape(prefix) + r"_s\d+(_r\d+)?", rid) and r.get("status") == status:
            out[r["config"]["seed"]] = r
    return [out[k] for k in sorted(out)]


def v(r, key, src="val_native"):
    d = r.get(src) or {}
    return d.get(key)


def agg(rs, fn):
    vals = [fn(r) for r in rs]
    vals = [x for x in vals if x is not None]
    if not vals:
        return None, None, 0
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else None), len(vals)


def f3(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def ms(rs, fn, d=3):
    m, s, n = agg(rs, fn)
    if m is None:
        return "–"
    return f"{m:.{d}f}" + (f" ± {s:.{d}f}" if s is not None else "")


def sgn(x, d=3):
    return "–" if x is None else f"{x:+.{d}f}"


def ci(c, d=3):
    if not c or c.get("n_seeds", 0) == 0:
        return "–"
    return f"{c['estimate']:+.{d}f} [{c['ci95'][0]:+.{d}f}, {c['ci95'][1]:+.{d}f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def screen_miou(r):
    return v(r, "miou", "val_screen")


def nat_miou(r):
    return v(r, "miou")


def train_min(r):
    return (r.get("train_time_s") or 0) / 60


def pixels_fed(r):
    h = r.get("history") or []
    return h[-1]["pixels_fed"] if h and "pixels_fed" in h[-1] else None


def fig_paths(name):
    return os.path.join(FIG, name + ".png"), f"fig/{name}.png"


# ------------------------------------------------------------------ figures
def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "figure.dpi": 150})
    return plt


def fig_curves():
    """Validation screen mIoU per epoch: all-pixel vs sampled (Kvasir, wide OR), the oscillation."""
    plt = plt_setup()
    series = [("E2_kvasir_CTX_or_wide", "all pixels, raster, q=128", GRAY),
              ("E1_kvasir_S_or_wide", "sampled 512 px/img, q=128", BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    have = False
    for arm, lab, col in series:
        rs = runs(arm)
        if not rs:
            continue
        h = rs[0]["history"]
        x = np.cumsum([e["pixels_fed"] - (h[i - 1]["pixels_fed"] if i else 0) for i, e in enumerate(h)]) / 1e6
        ax.plot(x, [e["screen_miou"] for e in h], color=col, lw=2, marker="o", ms=4, label=lab)
        have = True
    floor = runs("E0_kvasir_B0_majority")
    if floor:
        ax.axhline(screen_miou(floor[0]), color=INK2, lw=1, ls="--")
        ax.text(0, screen_miou(floor[0]) + 0.005, "all-background floor", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("training pixels fed (millions, log scale)")
    ax.set_ylabel("validation mIoU (128-px grid)")
    ax.set_title("Kvasir, wide OR pyramid: per-epoch validation mIoU", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(top=ax.get_ylim()[1] + 0.03)
    fig.tight_layout()
    p, rel = fig_paths("kvasir_curves")
    if have:
        fig.savefig(p)
    plt.close(fig)
    return rel if have else None


def fig_factorial(ds):
    plt = plt_setup()
    arms = [("F_base_B2", "base: tuned B2"), ("F_A_sampled", "A: + sampled pixels"), ("F_B_wide", "B: + wide OR context"),
            ("F_AB_sampled_wide", "A+B (finalist)"), ("F_unet", "U-Net (B4)")]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ys, labels = [], []
    for i, (a, lab) in enumerate(arms):
        fin = [R.get(f"FINAL_{r['run_id']}") for r in runs(f"E4_{ds}_{a}")]
        vals = [f["final_native"]["miou"] for f in fin if f]
        if not vals:
            continue
        y = len(labels)
        col = BLUE if a == "F_AB_sampled_wide" else (ORANGE if a == "F_unet" else GRAY)
        ax.scatter(vals, [y] * len(vals), color=col, s=22, zorder=3, edgecolor="white", linewidth=1)
        ax.plot([np.mean(vals)] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        labels.append(lab)
        ys.append(y)
    if not labels:
        plt.close(fig)
        return None
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlabel("sealed final split, native-grid mIoU (dots: seeds; bar: mean)")
    ax.set_title(f"{ds.upper() if ds == 'acdc' else 'Kvasir-SEG'}: frozen E4 comparison", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    p, rel = fig_paths(f"{ds}_final")
    fig.savefig(p)
    plt.close(fig)
    return rel


def fig_qualitative(ds="kvasir", n=4):
    """Fixed IDs (first n final-split IDs in manifest order): image, GT, base, A+B, U-Net (seed 0)."""
    import torch
    from PIL import Image

    import data

    rows = []
    srcs = []
    for a in ("F_base_B2", "F_AB_sampled_wide", "F_unet"):
        rs = runs(f"E4_{ds}_{a}")
        f = R.get(f"FINAL_{rs[0]['run_id']}") if rs else None
        if not f:
            return None
        srcs.append(torch.load(os.path.join(record.HERE, f["final_votes"])))
    if ds != "kvasir":
        return None
    os.environ["SEG_UNSEAL"] = "1"
    d = data.load_split("kvasir", 128, "final")
    plt = plt_setup()
    fig, axs = plt.subplots(n, 5, figsize=(7.5, 1.6 * n))
    for i in range(n):
        stem = d["ids"][i]
        img = np.asarray(Image.open(os.path.join(data.KVASIR_ROOT, "images", stem + ".jpg")).convert("RGB"))
        gt = data.load_kvasir_mask(stem)
        preds = [data.restore_scores(s[i].float(), d["geom"][i]).argmax(0).numpy() for s in srcs]
        for j, (im, t) in enumerate([(img, "image"), (gt, "ground truth"), (preds[0], "base B2 (raw TM)"),
                                     (preds[1], "A+B (raw TM)"), (preds[2], "U-Net")]):
            ax = axs[i, j]
            ax.imshow(im if j == 0 else im, cmap=None if j == 0 else "gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(t, fontsize=8, color=INK)
        axs[i, 0].set_ylabel(stem[:8], fontsize=7, color=INK2)
    fig.tight_layout()
    p, rel = fig_paths("kvasir_qualitative")
    fig.savefig(p)
    plt.close(fig)
    return rel


# ------------------------------------------------------------------ sections
def sec_diagnosis():
    rows = []
    for arm, lab in (("E0_kvasir_B0_majority", "B0 all-background"), ("E2_kvasir_B5_local_bool_t0", "B5 local CNN, 3×3 RF, B1 planes"),
                     ("E0_kvasir_B1", "B1 dense TM, 3×3 patch"), ("E2_kvasir_B2_t07", "B2 dense TM, pyramid (1,2,4)"),
                     ("E2_kvasir_CTX_or_wide", "dense TM, wide OR pyramid (1…32)"),
                     ("E2_kvasir_CTX_linear_or_wide", "linear, wide OR planes"),
                     ("E1_kvasir_S_or_wide", "dense TM, wide OR, sampled pixels"),
                     ("E2_kvasir_CTX_unet_bool_or_wide", "U-Net on wide OR Boolean planes"),
                     ("E2_kvasir_B4_t11", "U-Net on RGB (best of 12)")):
        rs = runs(arm)
        if rs:
            rows.append([lab, f"`{arm}`", ms(rs, screen_miou), ms(rs, nat_miou)])
    dp = R.get("DIAG_pixels_or_wide", {}).get("result", {})
    txt = table(["model", "record", "val mIoU (128 grid)", "val mIoU (native)"], rows)
    if dp:
        tm_best = max((max(e["val"]["miou"] for e in h["history"]) for k, h in dp.items() if k.startswith("tm_")), default=None)
        txt += (f"\nPixel-level check on the same wide-OR Boolean vectors (`DIAG_pixels_or_wide`, {dp['n_train']} sampled "
                f"training pixels, {dp['n_features']} features): logistic regression val mIoU {f3(dp['logreg']['val']['miou'])}, "
                f"gradient boosting {f3(dp['hgb']['val']['miou'])} (train {f3(dp['hgb']['train']['miou'])}), "
                f"flat TM best epoch {f3(tm_best)}.\n")
    ties = [R[k]["result"] for k in R if k.startswith("DIAG_ties_")]
    if ties:
        txt += (f"\nVote clipping at prediction time changes {f3(min(t['pred_changed_rate'] for t in ties))}–"
                f"{f3(max(t['pred_changed_rate'] for t in ties))} of pixel decisions; |mIoU(clipped) − mIoU(unclipped)| ≤ "
                f"{f3(max(abs(t['clipped_miou'] - t['unclipped_miou']) for t in ties))} (`DIAG_ties_*`).\n")
    return txt


def sec_h1_scenes():
    rows = []
    for order in ("raster", "perm", "disperse"):
        row = [order]
        for q in (32, 128, 512):
            row.append(ms(runs(f"E1_scenes_H1_{order}_q{q}"), screen_miou))
        rows.append(row)
    t = table(["order (8-image chunk)", "q = 32", "q = 128", "q = 512"], rows)
    rows2 = []
    for a, b in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.25)):
        rs = runs(f"E1_scenes_H1_sample_a{a}_b{b}")
        rows2.append([f"sample 256 px/img, a={a}, b={b}", ms(rs, screen_miou), ms(rs, train_min, 1)])
    for arm, lab in (("E1_scenes_H1_perm_q128", "all pixels, perm, q=128"), ("E1_scenes_H1_classfb_balanced", "all pixels + class_feedback_p (balanced)")):
        rs = runs(arm)
        rows2.append([lab, ms(rs, screen_miou), ms(rs, train_min, 1)])
    return t + "\n" + table(["H1 sampling arm (D0 scenes, 3 seeds)", "val mIoU", "train min (contended)"], rows2)


def sec_h1_kvasir():
    rows = []
    arms = [("E1_kvasir_H1_raster_q128", "all pixels, raster, q=128 (B2 features)"),
            ("E1_kvasir_H1_perm_q128", "all pixels, perm within 1 image, q=128"),
            ("E1_kvasir_H1_raster_q512", "all pixels, raster, q=512"),
            ("E1_kvasir_H1_perm_q512", "all pixels, perm within 1 image, q=512"),
            ("E2_kvasir_CTX_or_wide", "all pixels, raster, q=128 (wide OR)"),
            ("E1_kvasir_H1x_xperm8_q128_r3", "all pixels, perm across 8 images, q=128 (wide OR)"),
            ("E1_kvasir_H1x_disperse8_q128_r3", "all pixels, dispersed across 8 images (wide OR)"),
            ("E1_kvasir_H1x_samp512_a0.0_b0.0_q32", "sampled 512/img, a=b=0, q=32, 16-img pool"),
            ("E1_kvasir_H1x_samp512_a0.0_b0.0_q32_1img", "sampled 512/img, a=b=0, q=32, 1-img pool"),
            ("E1_kvasir_H1x_samp512_a0.0_b0.0_q128_r2", "sampled 512/img, a=b=0, q=128"),
            ("E1_kvasir_H1x_samp512_a0.25_b0.25_q32_r2", "sampled 512/img, a=b=0.25, q=32"),
            ("E1_kvasir_H1x_samp512_a0.25_b0.25_q128_r2", "sampled 512/img, a=b=0.25, q=128"),
            ("E1_kvasir_H1x_samp512_a0.5_b0.0_q32_r2", "sampled 512/img, class mix a=0.5, q=32"),
            ("E1_kvasir_H1x_samp512_a0.0_b0.5_q32_r2", "sampled 512/img, boundary mix b=0.5, q=32"),
            ("E1_kvasir_H1x_samp2048_a0.0_b0.0_q128_r2", "sampled 2048/img, a=b=0, q=128")]
    for arm, lab in arms:
        rs = runs(arm)
        if not rs:
            failed = [r for k, r in R.items() if k.startswith(arm) and r.get("status") != "completed"]
            rows.append([lab, f"`{arm}`", "not completed" + (f" ({failed[0]['status']})" if failed else ""), "", "", ""])
            continue
        r = rs[0]
        rows.append([lab, f"`{arm}`", f3(screen_miou(r)), f3(nat_miou(r)),
                     f3((pixels_fed(r) or 0) / 1e6, 1), f3(train_min(r), 0)])
    return table(["arm (Kvasir, 1 seed)", "record", "val mIoU 128", "val mIoU native", "pixels fed (M)", "train min"], rows)


def sec_h2():
    rows = []
    for nm, lab in (("H1_perm_q128", "OR pyramid (1,2,4) [B2 features]"), ("H2_and", "AND pyramid"), ("H2_occ_maj", "occupancy ≥ 0.5"),
                    ("H2_occ3", "occupancy {0.25, 0.5, 0.75}"), ("H2_occ3_bytes", "occupancy ×3, clauses cut to equal 2FC"),
                    ("H2_contrast", "OR pyramid + contrast literals"), ("H2_contrast_bytes", "… contrast, clauses cut to equal 2FC"),
                    ("H2_patch5", "single 5×5 patch (no pyramid)"), ("H2_extrabits_p3", "3×3 patch + extra thermometer levels (= B2 F)"),
                    ("H2_linear_pyr_or", "linear on OR pyramid planes"), ("H2_linear_occ3", "linear on occupancy planes"),
                    ("H2_linear_contrast", "linear on OR + contrast planes")):
        rs = runs(f"E1_scenes_{nm}")
        rows.append([lab, ms(rs, screen_miou), str(len(rs))])
    t = table(["D0 scenes arm (perm, q=128)", "val mIoU", "seeds"], rows)
    rows = []
    for nm, lab in (("S_pyr124", "OR (1,2,4) [B2 features]"), ("S_or_wide", "OR (1…32)"), ("S_and_wide", "AND (1…32)"),
                    ("S_occmaj_wide", "occupancy ≥ 0.5 (1…32), same F as OR"), ("S_occ3_wide", "occupancy ×3 (1…32)"),
                    ("S_or_wide_contrast", "OR (1…32) + contrast"), ("S_occmaj_wide_contrast", "occupancy (1…32) + contrast")):
        rs = runs(f"E1_kvasir_{nm}")
        rows.append([lab, ms(rs, screen_miou), ms(rs, nat_miou), str(rs[0]["model_info"]["n_features"]) if rs else "–"])
    t += "\n" + table(["Kvasir arm (sampled recipe, 1 seed)", "val mIoU 128", "val mIoU native", "F"], rows)
    rows = []
    for nm, lab in (("B1", "B1 3×3 patch"), ("B2_pyr_or", "B2 OR (1,2,4)"), ("CTX_or_wide", "OR (1…32)"), ("CTX_and_wide", "AND (1…32)"),
                    ("CTX_occmaj_wide", "occupancy ≥ 0.5 (1…32)"), ("CTX_occ3_wide", "occupancy ×3 (1…32)"),
                    ("CTX_linear_or_wide", "linear, OR (1…32)"), ("CTX_linear_occmaj_wide", "linear, occupancy (1…32)"),
                    ("H3_stack_oof", "H3 stack, out-of-fold coarse votes"), ("H3_stack_insample", "H3, in-sample coarse votes (mismatch ablation)"),
                    ("H3_stack_bilinear", "H3, bilinear-upsampled votes"), ("H3_stack_shuffled", "H3, shuffled coarse maps (control)"),
                    ("H3_stack_ORACLE", "H3, ground-truth coarse maps (ORACLE diagnostic)"),
                    ("CTX_unet_bool_or_wide", "U-Net on OR (1…32) planes"), ("B4_unet16", "U-Net on raw image")):
        rs = runs(f"E2_longrange_{nm}")
        rows.append([lab, ms(rs, screen_miou), str(len(rs))])
    t2 = table(["D1 long-range arm (all pixels, raster)", "val mIoU", "seeds"], rows)
    return t, t2


def sec_h3_kvasir():
    rows = []
    for arm, lab in (("E1_kvasir_S_pyr124", "fine only (B2 features), sampled"), ("E1_kvasir_S_or_wide", "fixed wide OR pyramid, sampled"),
                     ("E1_kvasir_S_H3_oof", "H3: fine + OOF coarse votes, sampled"), ("E1_kvasir_S_H3_shuffled", "H3 shuffled coarse maps, sampled"),
                     ("E1_kvasir_S_H3_ORACLE", "H3 ORACLE coarse maps, sampled"),
                     ("E2_kvasir_H3_stack_oof", "H3 OOF, all pixels (fine = B1 patch)"), ("E2_kvasir_H3_stack_insample", "H3 in-sample, all pixels"),
                     ("E2_kvasir_H3_stack_bilinear", "H3 bilinear, all pixels"), ("E2_kvasir_H3_stack_shuffled", "H3 shuffled, all pixels")):
        rs = runs(arm)
        if not rs:
            continue
        si = rs[0].get("stack_info", {})
        rows.append([lab, f3(nat_miou(rs[0])), f3(si.get("train_feature_coarse_miou")), f3(si.get("val_deploy_coarse_miou")),
                     f3(si.get("bit_rate_train_oof")), f3(si.get("bit_rate_val_deploy"))])
    return table(["Kvasir H3 arm (1 seed)", "val mIoU native", "coarse mIoU on train features", "coarse mIoU val (deploy)",
                  "vote-bit rate train", "vote-bit rate val"], rows)


def sec_refine():
    rows = []
    for rid, lab in (("E3_refine_E1_kvasir_H1x_samp512_a0.25_b0.25_q128_r2_s0_modelgrid", "Kvasir TM sampled wide"),
                     ("E3_refine_E2_kvasir_CTX_or_wide_s0_modelgrid", "Kvasir TM all-pixel wide"),
                     ("E3_refine_E2_kvasir_B4_t11_s0_modelgrid", "Kvasir U-Net"),
                     ("E3_refine_E4_acdc_S_or_wide_a0.25_b0.25_s0_modelgrid", "ACDC TM sampled wide"),
                     ("E3_refine_E4_acdc_B4_t04_s0_modelgrid", "ACDC U-Net")):
        r = R.get(rid)
        if not r:
            continue
        by = {x["arm"]: x for x in r["rows"]}
        raw = by["raw"]["summary"]
        for arm in ("majority3", "potts_l1.0_s0.15", "potts_unw_l1.0", "diff_l1.0_t1.0", "diff_l10.0_t1.0", "diff_unw_l10.0", "diff_l1.0_t1.0_explicit100"):
            if arm not in by:
                continue
            s = by[arm]["summary"]
            rows.append([lab, arm, f3(raw["miou"]), sgn(s["miou"] - raw["miou"]),
                         sgn((s.get("bf1_tn_fg_mean") or 0) - (raw.get("bf1_tn_fg_mean") or 0)),
                         f3(by[arm]["changed_pixel_rate"], 4), f3(by[arm]["ms_per_image"], 0)])
    return table(["unaries", "refinement", "raw mIoU", "Δ mIoU", "Δ BF1 (norm-tol, fg)", "changed px", "ms/img"], rows)


def final_stats(ds, arm):
    out = []
    for r in runs(f"E4_{ds}_{arm}"):
        f = R.get(f"FINAL_{r['run_id']}")
        if f:
            st = dict(f["final_native_stats"])
            out.append(st)
    return out


def val_stats(ds, arm):
    return [r["val_native_stats"] for r in runs(f"E4_{ds}_{arm}") if r.get("val_native_stats")]


def sec_e4(ds):
    K = K_OF[ds]
    arms = [("F_base_B2", "base: tuned B2 (all pixels, raster, pyramid 1–4)"), ("F_A_sampled", "A: base + sampled pixels"),
            ("F_B_wide", "B: base + wide OR pyramid (1–32)"), ("F_AB_sampled_wide", "A+B: finalist"), ("F_unet", "U-Net (B4, tuned)")]
    rows = []
    for a, lab in arms:
        rs = runs(f"E4_{ds}_{a}")
        fins = [R.get(f"FINAL_{r['run_id']}") for r in rs]
        fins = [f for f in fins if f]

        def fv(key):
            vals = [f["final_native"].get(key) for f in fins]
            vals = [x for x in vals if x is not None]
            if not vals:
                return "–"
            return f"{np.mean(vals):.3f}" + (f" ± {np.std(vals, ddof=1):.3f}" if len(vals) > 1 else "")

        row = [lab, f"{len(rs)}/{len(fins)}", ms(rs, nat_miou), fv("miou"), fv("fg_dice"), fv("mean_img_fg_dice"), fv("bf1_tn_fg_mean")]
        if ds == "acdc":
            row += [fv("vol_dice_RV"), fv("vol_dice_MYO"), fv("vol_dice_LV"), fv("vol_fg_dice")]
        row += [ms(rs, train_min, 0)]
        rows.append(row)
    hdr = ["arm", "seeds val/final", "val mIoU", "FINAL mIoU", "FINAL fg Dice (pooled)", "FINAL mean img fg Dice", "FINAL BF1 norm-tol fg"]
    if ds == "acdc":
        hdr += ["3D Dice RV", "3D Dice MYO", "3D Dice LV", "3D fg Dice"]
    hdr += ["train min (contended)"]
    txt = table(hdr, rows)
    # paired contrasts on the sealed split
    S = {a: final_stats(ds, a) for a, _ in arms}
    res = {}
    if all(S[a] for a in ("F_base_B2", "F_AB_sampled_wide")):
        res["AB_vs_base"] = ST.contrast([S["F_base_B2"], S["F_AB_sampled_wide"]], [-1, 1], K, "miou")
        res["AB_vs_base_fgdice"] = ST.contrast([S["F_base_B2"], S["F_AB_sampled_wide"]], [-1, 1], K, "fg_dice")
        res["AB_vs_base_bf1"] = ST.contrast([S["F_base_B2"], S["F_AB_sampled_wide"]], [-1, 1], K, "bf1_n_fg", n_boot=500)
        if S["F_A_sampled"] and S["F_B_wide"]:
            res["A_vs_base"] = ST.contrast([S["F_base_B2"], S["F_A_sampled"]], [-1, 1], K, "miou")
            res["B_vs_base"] = ST.contrast([S["F_base_B2"], S["F_B_wide"]], [-1, 1], K, "miou")
            res["interaction"] = ST.contrast([S["F_AB_sampled_wide"], S["F_A_sampled"], S["F_B_wide"], S["F_base_B2"]],
                                             [1, -1, -1, 1], K, "miou")
            res["AB_vs_A"] = ST.contrast([S["F_A_sampled"], S["F_AB_sampled_wide"]], [-1, 1], K, "miou")
            res["AB_vs_B"] = ST.contrast([S["F_B_wide"], S["F_AB_sampled_wide"]], [-1, 1], K, "miou")
        if S["F_unet"]:
            res["unet_vs_AB"] = ST.contrast([S["F_AB_sampled_wide"], S["F_unet"]], [-1, 1], K, "miou")
    rows = [[k, ci(c), str(c.get("n_seeds")), str(c.get("n_groups"))] for k, c in res.items()]
    txt += "\n" + table(["paired contrast on the sealed split (Δ, 95% hierarchical bootstrap)", "estimate [CI]", "seeds", "groups"], rows)
    return txt, res


def sec_acdc_volumes():
    rows = []
    for a, lab in (("F_base_B2", "base"), ("F_AB_sampled_wide", "A+B"), ("F_unet", "U-Net")):
        fins = [R.get(f"FINAL_{r['run_id']}") for r in runs(f"E4_acdc_{a}")]
        fins = [f for f in fins if f]
        if not fins:
            continue
        for phase in ("ED", "ES"):
            cells = [lab, phase]
            for c in ("RV", "MYO", "LV"):
                d = [np.mean([x[f"dice_{c}"] for x in f["final_volumes"] if x["phase"] == phase and x[f"dice_{c}"] is not None]) for f in fins]
                h = [np.mean([x[f"hd95_{c}"] for x in f["final_volumes"] if x["phase"] == phase and x[f"hd95_{c}"] is not None] or [np.nan]) for f in fins]
                nf = np.mean([sum(x[f"hd95_{c}"] is None for x in f["final_volumes"] if x["phase"] == phase) for f in fins])
                cells.append(f"{np.mean(d):.3f} / {np.nanmean(h):.1f} mm" + (f" ({nf:.1f} undef.)" if nf else ""))
            rows.append(cells)
    return table(["arm", "phase", "RV Dice / HD95", "MYO Dice / HD95", "LV Dice / HD95"], rows)


def sec_cost():
    rows = []
    bench = [r for k, r in R.items() if k.startswith("BENCH_latency")]
    lat = {}
    for b in bench:
        for row in b["rows"]:
            lat[row["run_id"]] = row
    for ds in ("kvasir", "acdc"):
        for a in ("F_base_B2", "F_A_sampled", "F_B_wide", "F_AB_sampled_wide", "F_unet"):
            rs = runs(f"E4_{ds}_{a}")
            if not rs:
                continue
            r0 = rs[0]
            mi = r0.get("model_info", {})
            L = lat.get(r0["run_id"], {})
            rows.append([ds, a, f3((pixels_fed(r0) or 0) / 1e6, 1) if r0["config"]["family"] == "tm" else "–",
                         ms(rs, train_min, 0), mi.get("automaton_states", mi.get("params", "–")),
                         f3(r0["checkpoint"]["bytes"] / 2**20, 1), f3(r0.get("gpu_peak_alloc_mb"), 0),
                         f3(1000 * L["b1"]["median_s"], 1) if L else "–", f3(L.get("images_per_s_b16"), 0) if L else "–"])
    return table(["dataset", "arm", "pixels fed, seed 0 (M)", "train min (contended)", "automaton states / CNN params",
                  "checkpoint MiB", "peak GPU MiB (train)", "latency b=1 ms (idle)", "images/s b=16 (idle)"], rows)


def sec_profile():
    prof = [r for k, r in sorted(R.items()) if k.startswith("BENCH_profile")]
    if not prof:
        return ""
    rows = []
    for x in prof[-1]["rows"]:
        if x["status"] != "completed":
            rows.append([x["name"], x["size"], x["status"], "", "", ""])
            continue
        rows.append([x["name"], x["size"], f3(x["train_8img"]["median_s"], 3), f3(1000 * x["infer_1img"]["median_s"], 1),
                     f3(1000 * x["infer_8img"]["median_s"], 1), f3(x["gpu_peak_alloc_mb"], 0)])
    return table(["model (all-pixel TM, q=128; RGB, 2 classes)", "px", "train step, 8 images (s)", "inference 1 image (ms)",
                  "inference 8 images (ms)", "peak GPU MiB"], rows)


def sec_stress():
    rows = []
    for k, r in sorted(R.items()):
        if not k.startswith("STRESS_E4_"):
            continue
        clean = r["rows"][0]["summary"]["miou"]
        for row in r["rows"][1:]:
            rows.append([r["source_run"], row["stress"], f3(clean), sgn(row["summary"]["miou"] - clean)])
    return table(["model (seed 0)", "stress (validation)", "clean mIoU", "Δ mIoU"], rows)


def sec_explain():
    """Top clauses of the Kvasir finalist (seed 0) decoded with feature provenance."""
    import torch

    import evalio
    import features

    rs = runs("E4_kvasir_F_AB_sampled_wide")
    if not rs:
        return ""
    rec = rs[0]
    os.environ.setdefault("SEG_UNSEAL", "0")
    d = {"x": torch.zeros(1, 3, 128, 128, dtype=torch.uint8)}
    cfg, m = evalio.load_model(rec, "cpu", d)
    names = features.feature_names(cfg["spec"], 3)
    assert len(names) == m.n_features
    inc = m.included_mask()
    w = m.weights if hasattr(m, "weights") else None
    lines = []
    C = m.n_clauses
    for cls in (1,):
        for pol in (1,):
            sel = range(cls * C, cls * C + C // 2) if pol else range(cls * C + C // 2, (cls + 1) * C)
            ws = [(int(w[j]) if w is not None else 1, j) for j in sel]
            for wt, j in sorted(ws, reverse=True)[:3]:
                lits = []
                for f in torch.nonzero(inc[j][: m.n_features]).flatten().tolist():
                    lits.append(names[f])
                for f in torch.nonzero(inc[j][m.n_features:]).flatten().tolist():
                    lits.append("NOT " + names[f])
                lines.append(f"- weight {wt}, {len(lits)} literals: " + " AND ".join(lits[:12]) + (" AND …" if len(lits) > 12 else ""))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ main
def main():
    os.makedirs(FIG, exist_ok=True)
    parts = {}
    parts["diag"] = sec_diagnosis()
    parts["h1s"] = sec_h1_scenes()
    parts["h1k"] = sec_h1_kvasir()
    parts["h2s"], parts["d1"] = sec_h2()
    parts["h3k"] = sec_h3_kvasir()
    parts["refine"] = sec_refine()
    parts["e4k"], resk = sec_e4("kvasir")
    parts["e4a"], resa = sec_e4("acdc")
    parts["acdcvol"] = sec_acdc_volumes()
    parts["cost"] = sec_cost()
    parts["stress"] = sec_stress()
    parts["profile"] = sec_profile()
    try:
        parts["explain"] = sec_explain()
    except Exception as e:  # pragma: no cover
        parts["explain"] = f"(clause decoding unavailable: {e})\n"
    figs = {"curves": fig_curves(), "fk": fig_factorial("kvasir"), "fa": fig_factorial("acdc")}
    try:
        figs["qual"] = fig_qualitative()
    except Exception as e:  # pragma: no cover
        figs["qual"] = None
        print("qualitative figure failed:", e)
    with open(os.path.join(REP, "parts.json"), "w") as f:
        json.dump({"parts": parts, "figs": figs, "res_kvasir": resk, "res_acdc": resa}, f, indent=1)
    import report_text

    md = report_text.render(parts, figs, resk, resa, R)
    with open(os.path.join(REP, "REPORT.md"), "w") as f:
        f.write(md)
    print("wrote report/REPORT.md")


if __name__ == "__main__":
    main()
