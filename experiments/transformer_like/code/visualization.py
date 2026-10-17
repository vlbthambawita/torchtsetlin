"""Report figures (spec §16 figures 1-8), PNG via matplotlib, generated from records and saved
checkpoints only. Figure image ids follow a documented deterministic rule (see ``pick_examples``).
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .common import ART, EXP  # noqa: E402

FIG = EXP / "report" / "figures"
# reference categorical palette (light), fixed order; text never wears series colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = "Blues"
ARM_COLOR = {"B0": SERIES[0], "B1": SERIES[1], "N0": SERIES[2], "B0M": SERIES[6], "B1s7": SERIES[3],
             "A2": SERIES[4], "A0": SERIES[7], "A1": SERIES[5], "A3": "#8a8984", "A2M": "#b4b2ab"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}"


def read_predictions(path: Path) -> Dict[str, np.ndarray]:
    with gzip.open(path, "rt") as f:
        a = np.loadtxt(f, delimiter=",", skiprows=1)
    a = a[np.argsort(a[:, 0])]
    return {"ids": a[:, 0].astype(int), "y": a[:, 1].astype(int), "pred": a[:, 2].astype(int), "scores": a[:, 3:]}


def fig_architecture() -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 2.2]})
    def box(ax, x, y, w, h, text, fc="#f3f2ef"):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                                       fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=INK)
    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    a = axes[0]
    box(a, 0.1, 0.8, 0.8, 0.12, "28x28 image -> u >= 128")
    box(a, 0.1, 0.5, 0.8, 0.12, "784 bits")
    box(a, 0.1, 0.2, 0.8, 0.12, "flat TM (B0)\n10 x M clauses", fc="#dce9f9")
    arrow(a, 0.5, 0.8, 0.5, 0.62); arrow(a, 0.5, 0.5, 0.5, 0.32)
    a.set_title("B0: flat TM", fontsize=9, color=INK)
    b = axes[1]
    box(b, 0.02, 0.84, 0.3, 0.1, "image -> u >= 128")
    box(b, 0.02, 0.62, 0.3, 0.12, "16 tokens: 7x7 pixels\n+ 6 position bits")
    box(b, 0.38, 0.74, 0.18, 0.1, "Query TM\n10 scores", fc="#dbf2e9")
    box(b, 0.38, 0.58, 0.18, 0.1, "Key TM\n10 scores", fc="#dbf2e9")
    box(b, 0.62, 0.62, 0.36, 0.16, "normalise Q, K;\nsoftmax(q.k / (sqrt(10) t))\nover active tokens", fc="#f9e3d9")
    box(b, 0.62, 0.38, 0.36, 0.14, "context o_i = sum_j alpha_ij v_j\n(values = token bits)")
    box(b, 0.62, 0.16, 0.36, 0.12, "thresholds 0.25/0.5/0.75\n-> 2640 bits")
    box(b, 0.02, 0.16, 0.48, 0.12, "head TM on [784 image bits ;\n2640 context bits]", fc="#dce9f9")
    arrow(b, 0.17, 0.84, 0.17, 0.74); arrow(b, 0.32, 0.68, 0.38, 0.79); arrow(b, 0.32, 0.66, 0.38, 0.63)
    arrow(b, 0.56, 0.79, 0.62, 0.72); arrow(b, 0.56, 0.63, 0.62, 0.67); arrow(b, 0.8, 0.62, 0.8, 0.52)
    arrow(b, 0.8, 0.38, 0.8, 0.28); arrow(b, 0.62, 0.22, 0.5, 0.22); arrow(b, 0.1, 0.62, 0.1, 0.28)
    b.set_title("N0: TMQKAttention (Q/K frozen, head trained)", fontsize=9, color=INK)
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0.1, 1); ax.axis("off")
    return _save(fig, "fig1_architecture.png")


def _grid(ax, img, title):
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
    for k in range(1, 4):
        ax.axhline(7 * k - 0.5, color=SERIES[1], lw=0.8)
        ax.axvline(7 * k - 0.5, color=SERIES[1], lw=0.8)
    ax.set_xticks([3 + 7 * c for c in range(4)], [f"c{c}" for c in range(4)], fontsize=7)
    ax.set_yticks([3 + 7 * r for r in range(4)], [f"r{r}" for r in range(4)], fontsize=7)
    ax.set_title(title, fontsize=8, color=INK)


def fig_patch_grid(u8: np.ndarray, image_id: int) -> str:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    _grid(ax, u8 >= 128, f"test image {image_id}: 4x4 grid of 7x7 tokens")
    for r in range(4):
        for c in range(4):
            ax.text(7 * c + 0.3, 7 * r + 1.2, str(4 * r + c), fontsize=6, color=SERIES[0])
    return _save(fig, "fig2_patch_grid.png")


@torch.no_grad()
def model_intermediates(model, u8: np.ndarray):
    from .features import transform_batch
    _, it = transform_batch(torch.as_tensor(u8[None]).to(model.device), model.mode, model.q, model.k,
                            model.temperature, intermediates=True)
    return {k: v[0].cpu().numpy() for k, v in it.items()}


def fig_attention(model, examples: List[Dict]) -> List[str]:
    out = []
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2))
    fig.subplots_adjust(wspace=0.45)
    for ax, ex in zip(np.atleast_1d(axes), examples):
        it = model_intermediates(model, ex["u8"])
        im = ax.imshow(it["alpha"], cmap=SEQ, vmin=0, vmax=1)
        ax.set_xlabel("sender j", fontsize=8, color=INK2)
        ax.set_ylabel("receiver i", fontsize=8, color=INK2)
        ax.set_xticks(range(0, 16, 3)); ax.set_yticks(range(0, 16, 3)); ax.tick_params(labelsize=7)
        ax.set_title(f"id {ex['id']} ({ex['label']}): {ex['note']}", fontsize=8, color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="alpha")
    out.append(_save(fig, "fig3_attention_heatmaps.png"))
    # fig 4: sender overlays for the two most-active query tokens of each example
    fig, axes = plt.subplots(n, 2, figsize=(5.2, 2.6 * n))
    axes = np.atleast_2d(axes)
    for row, ex in enumerate(examples):
        it = model_intermediates(model, ex["u8"])
        act = np.flatnonzero(it["active"])
        pix = (ex["u8"] >= 128).reshape(4, 7, 4, 7).transpose(0, 2, 1, 3).reshape(16, 49).sum(1)
        qs = act[np.argsort(-pix[act], kind="stable")][:2]
        for col in range(2):
            ax = axes[row, col]
            if col >= len(qs):
                ax.axis("off"); continue
            qi = int(qs[col])
            w = it["alpha"][qi].reshape(4, 4)
            heat = np.kron(w, np.ones((7, 7)))
            ax.imshow(ex["u8"] >= 128, cmap="gray_r", vmin=0, vmax=1)
            ax.imshow(heat, cmap=SEQ, alpha=0.55, vmin=0, vmax=max(w.max(), 1e-6))
            r, c = divmod(qi, 4)
            ax.add_patch(matplotlib.patches.Rectangle((7 * c - 0.5, 7 * r - 0.5), 7, 7, fill=False, ec=SERIES[1], lw=1.5))
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"id {ex['id']}: query token {qi}", fontsize=7.5, color=INK)
    out.append(_save(fig, "fig4_sender_overlays.png"))
    # fig 5: Q/K digit-score profiles for the strongest-attended pair of the first example
    ex = examples[0]
    it = model_intermediates(model, ex["u8"])
    a = it["alpha"].copy()
    np.fill_diagonal(a, 0)
    i, j = np.unravel_index(np.argmax(a), a.shape)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(10)
    ax.bar(x - 0.2, it["uq"][i], 0.38, color=SERIES[0], label=f"Q of receiver {i}")
    ax.bar(x + 0.2, it["uk"][j], 0.38, color=SERIES[1], label=f"K of sender {j}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(x); ax.set_xlabel("digit coordinate", fontsize=8, color=INK2)
    ax.set_ylabel("u = clip(votes / T)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"id {ex['id']} (label {ex['label']}): alpha[{i},{j}] = {it['alpha'][i, j]:.2f}", fontsize=8, color=INK)
    _style(ax)
    out.append(_save(fig, "fig5_qk_profiles.png"))
    return out


def fig_curves(histories: Dict[str, List[Dict]], extra_offsets: Dict[str, float]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for arm, hist in histories.items():
        ep = [h["epoch"] for h in hist]
        acc = [100 * h["val_accuracy"] for h in hist]
        t = [(h["cum_train_s"] + extra_offsets.get(arm, 0.0)) / 60 for h in hist]
        c = ARM_COLOR.get(arm, INK2)
        axes[0].plot(ep, acc, color=c, lw=2, label=arm)
        axes[1].plot(t, acc, color=c, lw=2, label=arm)
        axes[1].text(t[-1], acc[-1], f" {arm}", fontsize=7, color=INK2, va="center")
    axes[0].set_xlabel("epoch", fontsize=8, color=INK2)
    axes[1].set_xlabel("cumulative training time incl. encoders + features (min)", fontsize=8, color=INK2)
    axes[0].set_ylabel("validation accuracy (%)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        _style(ax)
    fig.suptitle("Selected configurations, final seed 23", fontsize=9, color=INK)
    return _save(fig, "fig6_validation_curves.png")


def fig_pareto(points: List[Dict]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    placed = {"bytes_mb": [], "latency_ms": []}
    for p in points:
        c = ARM_COLOR.get(p["arm"], INK2)
        for ax, key in zip(axes, ("bytes_mb", "latency_ms")):
            if p.get(key) is None:
                continue
            ax.errorbar(p[key], p["acc"], yerr=p["sd"], fmt="o", ms=6, color=c, ecolor=c, elinewidth=1,
                        mec="white", mew=1.5)
            # stagger labels of points that (nearly) coincide
            near = sum(1 for x, y in placed[key] if abs(np.log(x / p[key])) < 0.15 and abs(y - p["acc"]) < 0.6)
            placed[key].append((p[key], p["acc"]))
            ax.annotate(p["arm"], (p[key], p["acc"]), xytext=(5, 3 - 10 * near), textcoords="offset points",
                        fontsize=7, color=INK2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("persistent automaton state (MB, int32 as stored)", fontsize=8, color=INK2)
    axes[1].set_xlabel("batch-1 latency, raw image to digit (ms, GPU)", fontsize=8, color=INK2)
    axes[0].set_ylabel("test accuracy (%), mean +/- SD over 5 seeds", fontsize=8, color=INK2)
    for ax in axes:
        _style(ax)
    return _save(fig, "fig7_accuracy_vs_cost.png")


def fig_confusions(cms: Dict[str, np.ndarray]) -> str:
    fig, axes = plt.subplots(1, len(cms), figsize=(3.3 * len(cms), 3.2))
    for ax, (arm, cm) in zip(np.atleast_1d(axes), cms.items()):
        off = cm.astype(float).copy()
        np.fill_diagonal(off, np.nan)
        im = ax.imshow(off, cmap="Oranges")
        ax.set_title(f"{arm}: off-diagonal errors (seed 23)", fontsize=8, color=INK)
        ax.set_xlabel("predicted", fontsize=7, color=INK2); ax.set_ylabel("true", fontsize=7, color=INK2)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, shrink=0.7)
    return _save(fig, "fig8a_confusion.png")


def fig_disagreements(examples: List[Dict]) -> str:
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 2.3))
    for ax, ex in zip(np.atleast_1d(axes), examples):
        ax.imshow(ex["u8"] >= 128, cmap="gray_r")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"id {ex['id']} y={ex['label']}\n" + " ".join(f"{k}:{v}" for k, v in ex["preds"].items()),
                     fontsize=6.5, color=INK)
    return _save(fig, "fig8b_disagreements.png")
