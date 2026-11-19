"""Two summary figures of the grid arm: batch size x clauses -> accuracy and GPU time.

  figures/summary_heatmaps.png   per model: test accuracy at the best-validation epoch (left) and
                                 GPU hours to that epoch (right), one cell per (clauses, batch)
  figures/summary_pareto.png     per model: test error against GPU hours, coloured by batch size,
                                 with the Pareto front (no other run is both faster and better)

Every number comes from results/grid/*.json (seed 0, T scaled).
Usage: python code/summary_fig.py            # figures/*.png
       python code/summary_fig.py --paper    # also the paper's figures/batch_clauses_heatmaps.pdf
                                             # and generated/batch_summary_numbers.tex (R{bcs/...} macros)
"""

import argparse

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

PROG = Path(__file__).resolve().parents[1]
PAPER = Path("/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE = LinearSegmentedColormap.from_list("blue", BLUES)
ORANGE = LinearSegmentedColormap.from_list("orange", ["#fde6d9", "#f6b08f", "#eb6834", "#b8491d", "#7a2d0e"])
MODELS = [("flat", "Flat TM"), ("conv", "Conv 2D TM"), ("conv3", "Conv 3D TM (bit planes)")]
ACC_FLOOR = {"flat": 96.0, "conv": 98.4, "conv3": 96.5}      # colour scale floor; values stay printed
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def load():
    out = {}
    for f in sorted((PROG / "results" / "grid").glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("status") == "done":
            h = r["hp"]
            out[(h["model"], h["clauses_per_class"], h["batch_size"])] = dict(
                version=r["env"]["torchtsetlin"],
                acc=r["test_acc_at_best"], last=r["test_acc_last"], hours=r["gpu_s_to_best"] / 3600,
                epoch=r["best_epoch"])
    return out


def axes_of(d, model):
    return (sorted({c for m, c, _ in d if m == model}), sorted({b for m, _, b in d if m == model}))


def ink_on(rgba):
    r, g, b = rgba[:3]
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def heat(ax, grid, cmap, norm, fmt, cs, bs, title):
    ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(cs)):
        for j in range(len(bs)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8, color=ink_on(cmap(norm(v))))
    ax.set_xticks(range(len(bs)), [str(b) for b in bs])
    ax.set_yticks(range(len(cs)), [f"{c // 1000}k" for c in cs])
    ax.set_xticks(np.arange(-0.5, len(bs)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cs)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=6)


def heatmaps(d, paper=False):
    fig, axs = plt.subplots(3, 2, figsize=(11.5, 9.0 if paper else 9.6), height_ratios=[7, 7, 4.3])
    for row, (model, name) in enumerate(MODELS):
        cs, bs = axes_of(d, model)
        acc = np.array([[d.get((model, c, b), {}).get("acc", np.nan) for b in bs] for c in cs])
        hrs = np.array([[d.get((model, c, b), {}).get("hours", np.nan) for b in bs] for c in cs])
        lo = ACC_FLOOR[model]
        heat(axs[row, 0], acc, BLUE, Normalize(lo, np.nanmax(acc)), lambda v: f"{v:.2f}", cs, bs,
             f"{name}: test accuracy (%)   [colour scale starts at {lo:g}]")
        heat(axs[row, 1], hrs, ORANGE, LogNorm(0.01, 15), lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}", cs, bs,
             f"{name}: GPU hours to the best epoch")
        axs[row, 0].set_ylabel("clauses per class")
    for ax in axs[2]:
        ax.set_xlabel("batch size")
    if paper:                                          # the caption carries title and notes
        fig.tight_layout(h_pad=1.6, w_pad=2.0)
        fig.savefig(PAPER / "figures" / "batch_clauses_heatmaps.pdf")
        plt.close(fig)
        return
    fig.suptitle("MNIST: batch size x clauses, accuracy and training time", x=0.012, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.012, 0.012, "Seed 0, T scaled with clauses, early stopping on validation accuracy (patience 25). "
             "Accuracy = test accuracy at the best-validation epoch. Darker = higher.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.965), h_pad=1.6, w_pad=2.0)
    fig.savefig(PROG / "figures" / "summary_heatmaps.png", dpi=160)
    plt.close(fig)


def write_macros(d):
    """\\R{bcs/<model>/...} values quoted in the paper's text (no number is typed there)."""
    def acc(model, bs):
        return [v["acc"] for (m, _, b), v in d.items() if m == model and b in bs]

    def hours(model, c, b):
        h = d[(model, c, b)]["hours"]
        return f"{h:.2f}" if h < 10 else f"{h:.1f}"
    versions = sorted({v["version"] for v in d.values()})
    assert len(versions) == 1, f"grid measured on several library versions: {versions}"
    m = {"runs": str(len(d)), "version": versions[0]}
    for model, groups in (("flat", {"small": (1, 16, 32, 64, 128, 256, 512), "b1024": (1024,)}),
                          ("conv", {"small": (1, 16, 32, 64), "b256": (256,), "b1024": (1024,)}),
                          ("conv3", {"small": (1, 32), "b256": (256,), "b1024": (1024,)})):
        for g, bs in groups.items():
            m[f"{model}/{g}/min"], m[f"{model}/{g}/max"] = f"{min(acc(model, bs)):.2f}", f"{max(acc(model, bs)):.2f}"
    # final-epoch reading (the cross-library protocol): largest gap to the best-epoch value
    for model, bs in (("conv", None), ("conv3", None), ("flat", (1, 16, 32, 64, 128, 256, 512))):
        gaps = [abs(v["last"] - v["acc"]) for (mm, _, b), v in d.items() if mm == model and (bs is None or b in bs)]
        m[f"{model}/lastgap"] = f"{max(gaps):.2f}"
    last = [v["last"] for (mm, _, b), v in d.items() if mm == "flat" and b == 1024]
    m["flat/b1024/lastmin"], m["flat/b1024/lastmax"] = f"{min(last):.2f}", f"{max(last):.2f}"
    cmax = {"flat": 64000, "conv": 64000, "conv3": 8000}
    for model, bs in (("flat", (1, 16)), ("conv", (16, 1024)), ("conv3", (32, 1024))):
        for b in bs:
            m[f"{model}/hours/b{b}"] = hours(model, cmax[model], b)
    lines = ["% generated by experiments/mnist_batch_clauses/code/summary_fig.py from results/grid/*.json; do not edit",
             "\\providecommand{\\R}[1]{\\ifcsname R@#1\\endcsname\\csname R@#1\\endcsname"
             "\\else\\textcolor{red}{??#1}\\fi}"]
    lines += [f"\\expandafter\\def\\csname R@bcs/{k}\\endcsname{{{v}}}" for k, v in sorted(m.items())]
    (PAPER / "generated" / "batch_summary_numbers.tex").write_text("\n".join(lines) + "\n")


def pareto(pts):
    """Indices of the runs that no other run beats on both time and error."""
    order = sorted(range(len(pts)), key=lambda i: (pts[i][0], pts[i][1]))
    front, best = [], np.inf
    for i in order:
        if pts[i][1] < best:
            front.append(i)
            best = pts[i][1]
    return front


def pareto_fig(d):
    all_bs = sorted({b for _, _, b in d})
    col = {b: BLUE(0.28 + 0.72 * k / (len(all_bs) - 1)) for k, b in enumerate(all_bs)}
    ymax = {"flat": 4.5, "conv": 1.7, "conv3": 3.6}
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9))
    for ax, (model, name) in zip(axs, MODELS):
        every = [k for k in d if k[0] == model]
        keys = [k for k in every if 100 - d[k]["acc"] <= ymax[model]]      # front: the plotted runs
        off = len(every) - len(keys)
        pts = [(d[k]["hours"], 100 - d[k]["acc"]) for k in keys]
        for k, (h, e) in zip(keys, pts):
            ax.scatter(h, e, s=18 + 14 * np.log2(k[1] / 1000), color=col[k[2]], edgecolor=SURFACE,
                       linewidth=0.9, zorder=3)
        f = pareto(pts)
        ax.step([pts[i][0] for i in f], [pts[i][1] for i in f], where="post", color=INK2, linewidth=1.2, zorder=2)
        last = None                                    # label the front where the labels have room
        for i in [f[0]] + f[1:-1] + ([f[-1]] if len(f) > 1 else []):
            here = np.log10(pts[i])
            far = last is None or abs(here[0] - last[0]) > 0.42 or abs(here[1] - last[1]) > 0.09
            if far or i == f[-1]:
                ax.annotate(f"b{keys[i][2]} · {keys[i][1] // 1000}k", pts[i], xytext=(-6, -12),
                            textcoords="offset points", ha="right", fontsize=7.5, color=INK)
                last = here
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(top=ymax[model])
        ax.set_xlim(left=min(h for h, _ in pts) / 3.2)              # room for the front labels
        if max(h for h, _ in pts) / min(h for h, _ in pts) < 100:
            ax.set_xticks([0.3, 1, 3])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(name + (f"   ({off} runs above the axis)" if off else ""), loc="left", fontsize=10, color=INK)
        ax.set_xlabel("GPU hours to the best epoch (log)")
    axs[0].set_ylabel("test error, % (log; lower is better)")
    hb = [plt.Line2D([], [], marker="o", linestyle="", color=col[b], markersize=7, label=str(b)) for b in all_bs]
    hc = [plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=np.sqrt(18 + 14 * np.log2(c / 1000)),
                     label=f"{c // 1000}k") for c in (1000, 8000, 64000)]
    hl = [plt.Line2D([], [], color=INK2, linewidth=1.2, label="Pareto front (label: batch · clauses)")]
    fig.legend(handles=hb, title="batch size", ncol=len(hb), loc="lower left", bbox_to_anchor=(0.03, -0.01),
               frameon=False, fontsize=8, title_fontsize=8, columnspacing=1.0, handletextpad=0.1)
    fig.legend(handles=hc + hl, title="clauses per class", ncol=4, loc="lower right", bbox_to_anchor=(0.99, -0.01),
               frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle("MNIST: what accuracy does each GPU hour buy?  Lower-left is better", x=0.012, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(PROG / "figures" / "summary_pareto.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args()
    data = load()
    if args.paper:
        heatmaps(data, paper=True)
        write_macros(data)
        print("paper: figures/batch_clauses_heatmaps.pdf, generated/batch_summary_numbers.tex")
    (PROG / "figures").mkdir(exist_ok=True)
    heatmaps(data)
    pareto_fig(data)
    print(f"{len(data)} runs -> figures/summary_heatmaps.png, figures/summary_pareto.png")
