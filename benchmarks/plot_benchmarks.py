#!/usr/bin/env python
"""Turn ``bench_device.py`` results into the figures and tables used in ``docs/benchmarks.md``.

Every figure is rendered twice — once for the light and once for the dark documentation
theme (``<name>.png`` / ``<name>_dark.png``), which MkDocs Material switches between with
the ``#only-light`` / ``#only-dark`` image suffixes.

Usage::

    python benchmarks/plot_benchmarks.py \\
        --results benchmarks/results/cpu_vs_gpu.json \\
        --out docs/assets/benchmarks
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

# --------------------------------------------------------------------------------- theme
# Validated categorical slots (see the data-visualisation palette): slot 1 blue,
# 2 orange, 3 aqua, 4 yellow, 5 magenta, 6 green, 7 violet.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        ink3="#77756f",
        grid="#e6e5e1",
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        ink3="#96958c",
        grid="#33332f",
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"],
    ),
}

DEVICE_SLOT = {"cpu": 0, "cuda": 1}  # CPU = blue, GPU = orange, everywhere


def device_label(rec: Dict, short: bool = True) -> str:
    if rec["device"].startswith("cpu"):
        return "CPU" if short else f"CPU — {rec['device_name']}"
    name = rec["device_name"].replace("NVIDIA GeForce ", "")
    return f"GPU ({name})" if short else f"GPU — {rec['device_name']}"


def color_for(theme: Dict, rec_or_device) -> str:
    dev = rec_or_device if isinstance(rec_or_device, str) else rec_or_device["device"]
    return theme["series"][DEVICE_SLOT["cpu" if dev.startswith("cpu") else "cuda"]]


def style(theme: Dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["grid"],
        "axes.labelcolor": theme["ink2"],
        "axes.titlecolor": theme["ink"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": theme["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "text.color": theme["ink"],
        "xtick.color": theme["ink2"],
        "ytick.color": theme["ink2"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 140,
    })


def sig3(v: float) -> str:
    """Three significant figures, without trailing zeros: 899.637 -> 900, 4.4571 -> 4.46."""
    return f"{float(f'{v:.3g}'):g}"


def thousands(v: float, _pos: int = 0) -> str:
    if v >= 1_000_000:
        return f"{sig3(v / 1e6)}M"
    if v >= 1000:
        return f"{sig3(v / 1000)}k"
    return sig3(v)


def tidy(ax, *, xlog: bool = False, ylog: bool = False, yfmt: bool = True) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if xlog:
        ax.set_xscale("log", base=2)
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if yfmt:
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.grid(True, which="major", axis="both")
    ax.tick_params(length=0)


def end_label(ax, x, y, text, color, theme, dy=1.06, dpts=0.0, ha="left",
              fontsize=8.5) -> None:
    """Direct label at the end of a line — in ink, with the coloured marker beside it."""
    ax.annotate(text, (x, y * dy if ax.get_yscale() == "log" else y),
                xytext=(4, dpts), textcoords="offset points", color=theme["ink2"],
                fontsize=fontsize, ha=ha, va="center", clip_on=False)
    ax.plot([x], [y], marker="o", markersize=6, color=color,
            markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=5, clip_on=False)


def end_labels(ax, items, theme, **kw) -> None:
    """Direct labels for several lines, stacked in points when two ends nearly coincide."""
    items = sorted(items, key=lambda it: it[1])
    offs = [(1.06, 0.0)] * len(items)
    for i in range(1, len(items)):
        if items[i][1] < items[i - 1][1] * 1.4:  # within ~40% — the labels would collide
            offs[i - 1], offs[i] = (1.0, -7.0), (1.0, 7.0)
    for (x, y, text, color), (dy, dpts) in zip(items, offs):
        end_label(ax, x, y, text, color, theme, dy=dy, dpts=dpts, **kw)


# ---------------------------------------------------------------------------- data access
def load(path: str) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def sel(records: Sequence[Dict], **crit) -> List[Dict]:
    out = []
    for r in records:
        ok = True
        for k, v in crit.items():
            got = r["extra"].get(k) if k not in r else r[k]
            if got != v:
                ok = False
                break
        if ok:
            out.append(r)
    return out


def series(records: Sequence[Dict], xkey: str, **crit) -> List[Dict]:
    return sorted(sel(records, **crit), key=lambda r: r[xkey])


def devices_of(records: Sequence[Dict]) -> List[str]:
    order = {"cpu": 0}
    return sorted({r["device"] for r in records}, key=lambda d: order.get(d, 1))


# -------------------------------------------------------------------------------- figures
def fig_batch(recs, theme, env):
    """Throughput vs batch size, for a large and a tiny machine (small multiples)."""
    scales = [
        ("batch", "MNIST scale — 784 features, 5 000 clauses"),
        ("small", "Noisy-XOR scale — 12 features, 20 clauses"),
    ]
    phases = [("train", "training (update)"), ("infer", "inference (forward)")]
    scales = [s for s in scales if sel(recs, suite=s[0])]
    if not scales:
        return None
    fig, axes = plt.subplots(len(scales), 2, figsize=(9.2, 3.3 * len(scales)), squeeze=False)
    for i, (suite, scale_title) in enumerate(scales):
        for j, (phase, phase_title) in enumerate(phases):
            ax = axes[i][j]
            if not sel(recs, suite=suite, phase=phase):
                ax.set_visible(False)
                continue
            ends = []
            for dev in devices_of(recs):
                pts = series(recs, "batch_size", suite=suite, phase=phase, device=dev)
                if not pts:
                    continue
                c = color_for(theme, dev)
                xs = [p["batch_size"] for p in pts]
                ys = [p["examples_per_s"] for p in pts]
                ax.plot(xs, ys, color=c, marker="o", label=device_label(pts[0]),
                        markeredgecolor=theme["surface"], markeredgewidth=1.2)
                ends.append((xs[-1], ys[-1], thousands(ys[-1]), c))
            end_labels(ax, ends, theme)
            tidy(ax, xlog=True, ylog=True)
            ax.set_title(f"{scale_title}\n{phase_title}", loc="left", color=theme["ink"])
            if i == len(scales) - 1:
                ax.set_xlabel("batch size (examples per update call)")
            if j == 0:
                ax.set_ylabel("examples / s")
            ax.set_xlim(right=max(p["batch_size"] for p in sel(recs, suite=suite)) * 2.6)
    legend_ax = next((a for row in axes for a in row if a.get_visible()), axes[0][0])
    legend_ax.legend(loc="upper left", labelcolor=theme["ink2"])
    fig.suptitle("Throughput vs batch size — CPU vs GPU", x=0.012, ha="left",
                 fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def fig_speedup(recs, theme, env):
    """GPU-over-CPU speedup, one line per model scale."""
    scales = [("batch", "MNIST scale (5 000 clauses)", 6), ("small", "Noisy-XOR scale (20 clauses)", 2)]
    phases = [("train", "training (update)"), ("infer", "inference (forward)")]
    if not any(sel(recs, suite=s) for s, _, _ in scales):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for j, (phase, phase_title) in enumerate(phases):
        ax = axes[j]
        ax.axhline(1.0, color=theme["ink3"], linewidth=1.0, zorder=1)
        ax.annotate("1× — parity", (1, 1.0), xytext=(2, 5), textcoords="offset points",
                    color=theme["ink3"], fontsize=8)
        ends = []
        for suite, title, slot in scales:
            cpu = {r["batch_size"]: r["examples_per_s"]
                   for r in sel(recs, suite=suite, phase=phase, device="cpu")}
            gpu = [r for r in series(recs, "batch_size", suite=suite, phase=phase)
                   if not r["device"].startswith("cpu")]
            if not cpu or not gpu:
                continue
            xs = [r["batch_size"] for r in gpu if r["batch_size"] in cpu]
            ys = [r["examples_per_s"] / cpu[r["batch_size"]] for r in gpu if r["batch_size"] in cpu]
            c = theme["series"][slot]
            ax.plot(xs, ys, color=c, marker="o", label=title,
                    markeredgecolor=theme["surface"], markeredgewidth=1.2)
            ends.append((xs[-1], ys[-1],
                         f"{ys[-1]:.0f}×" if ys[-1] >= 10 else f"{ys[-1]:.1f}×", c))
        end_labels(ax, ends, theme)
        tidy(ax, xlog=True, ylog=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
        ax.set_title(phase_title, loc="left", color=theme["ink"])
        ax.set_xlabel("batch size")
        ax.set_xlim(right=5000 * 2.2)
    axes[0].set_ylabel("GPU throughput / CPU throughput")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2,
                   labelcolor=theme["ink2"])
    fig.suptitle("How much the GPU buys you", x=0.012, ha="left", fontsize=13,
                 color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_model_size(recs, theme, env):
    """Throughput vs clause budget and vs feature count."""
    cols = [
        ("clauses", "n_clauses_per_class", "clauses per class (784 features, batch 256)"),
        ("features", "n_features", "Boolean features (500 clauses/class, batch 256)"),
    ]
    rows = [("train", "training (update)"), ("infer", "inference (forward)")]
    cols = [c for c in cols if sel(recs, suite=c[0])]
    if not cols:
        return None
    fig, axes = plt.subplots(2, len(cols), figsize=(4.6 * len(cols), 6.6), squeeze=False)
    for i, (phase, phase_title) in enumerate(rows):
        for j, (suite, xkey, xlabel) in enumerate(cols):
            ax = axes[i][j]
            if not sel(recs, suite=suite, phase=phase):
                ax.set_visible(False)
                continue
            ends = []
            for dev in devices_of(recs):
                pts = series(recs, xkey, suite=suite, phase=phase, device=dev)
                if not pts:
                    continue
                c = color_for(theme, dev)
                xs = [p[xkey] for p in pts]
                ys = [p["examples_per_s"] for p in pts]
                ax.plot(xs, ys, color=c, marker="o", label=device_label(pts[0]),
                        markeredgecolor=theme["surface"], markeredgewidth=1.2)
                ends.append((xs[-1], ys[-1], thousands(ys[-1]), c))
            end_labels(ax, ends, theme)
            tidy(ax, xlog=True, ylog=True)
            ax.set_title(phase_title, loc="left", color=theme["ink"])
            ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel("examples / s")
            ax.set_xlim(right=max(p[xkey] for p in sel(recs, suite=suite)) * 2.4)
    legend_ax = next((a for row in axes for a in row if a.get_visible()), axes[0][0])
    legend_ax.legend(loc="lower left", labelcolor=theme["ink2"])
    fig.suptitle("Throughput vs model size — CPU vs GPU", x=0.012, ha="left", fontsize=13,
                 color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def _grouped_barh(ax, labels, groups, theme, *, gap_px=2, height=0.38):
    """Horizontal grouped bars; ``groups`` is a list of (label, colour, values)."""
    n = len(groups)
    ys = list(range(len(labels)))
    bars = []
    for k, (glabel, color, values) in enumerate(groups):
        offs = (k - (n - 1) / 2) * height
        pos = [y - offs for y in ys]
        bars.append(ax.barh(pos, values, height=height * 0.92, color=color, label=glabel,
                            zorder=3))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5, color=theme["ink2"])
    ax.invert_yaxis()
    return bars


def fig_model_zoo(recs, theme, env):
    """Per-model throughput at a common batch size."""
    order = [r["model"] for r in sel(recs, suite="models", phase="train", device="cpu")]
    if not order:
        seen = []
        for r in sel(recs, suite="models", phase="train"):
            if r["model"] not in seen:
                seen.append(r["model"])
        order = seen
    if not order:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for j, (phase, title) in enumerate([("train", "training (update)"),
                                        ("infer", "inference (forward)")]):
        ax = axes[j]
        groups = []
        for dev in devices_of(recs):
            by_model = {r["model"]: r["examples_per_s"]
                        for r in sel(recs, suite="models", phase=phase, device=dev)}
            if not by_model:
                continue
            ref = sel(recs, suite="models", phase=phase, device=dev)[0]
            groups.append((device_label(ref), color_for(theme, dev),
                           [by_model.get(m, 0.0) for m in order]))
        labels = [m.replace("TsetlinMachine", "TM").replace(" (weighted)", "\n(weighted)")
                  for m in order]
        _grouped_barh(ax, labels, groups, theme)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(length=0)
        ax.set_xlabel("examples / s")
        ax.set_title(title, loc="left", color=theme["ink"])
        hi = max(max(g[2]) for g in groups)
        ax.set_xlim(right=hi * 4.0)
        for k, (_, _, values) in enumerate(groups):
            offs = (k - (len(groups) - 1) / 2) * 0.38
            for y, v in enumerate(values):
                ax.annotate(thousands(v), (v, y - offs), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.5,
                            color=theme["ink2"])
        if j == 1:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
                      labelcolor=theme["ink2"])
    fig.suptitle("Model zoo — 784 Boolean features, batch 128", x=0.012, ha="left",
                 fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


STAGE_GROUPS = [
    ("clause evaluation", ["encode", "evaluate"]),
    ("votes + feedback selection", ["votes+select"]),
    ("feedback counts (matmuls)", ["feedback_counts"]),
    ("apply_feedback (commit)", ["apply_feedback"]),
    ("bookkeeping (alloc, targets, include cache)",
     ["accumulator_alloc", "coerce_targets", "refresh_include"]),
]


def _stage_ms(recs, device) -> Dict[str, float]:
    st = {r["extra"]["stage"]: r["sec_per_batch"] * 1e3
          for r in sel(recs, suite="phases", phase="stage", device=device)}
    if "apply_feedback" not in st and "commit" in st:
        # Results from before the stage was timed directly: apply_feedback is the commit
        # minus the two things bench_device.py measures separately.
        st["apply_feedback"] = max(
            st["commit"] - st.get("refresh_include", 0.0) - st.get("coerce_targets", 0.0), 0.0)
    return st


def fig_phases(recs, theme, env):
    """Where the time of one update goes, per device."""
    devs = devices_of([r for r in recs if r["suite"] == "phases"])
    if not devs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    labels, totals = [], []
    for dev in devs:
        st = _stage_ms(recs, dev)
        vals = [sum(st.get(k, 0.0) for k in keys) for _, keys in STAGE_GROUPS]
        total = sum(vals)
        totals.append(total)
        ref = sel(recs, suite="phases", phase="stage", device=dev)[0]
        measured = sel(recs, suite="phases", phase="update_total", device=dev)
        shown = measured[0]["sec_per_batch"] * 1e3 if measured else total
        labels.append(f"{device_label(ref)}\n{shown:,.1f} ms / update")
        y = len(labels) - 1
        left = 0.0
        # 2px surface gap between segments (never a border).
        gap = 0.4
        for slot, (name, _keys) in enumerate(STAGE_GROUPS):
            share = 100.0 * vals[slot] / total
            ax.barh([y], [max(share - gap, 0.02)], left=left, height=0.44,
                    color=theme["series"][slot], zorder=3,
                    label=name if y == 0 else None)
            if share > 7:
                ax.annotate(f"{share:.0f}%", (left + share / 2, y), ha="center", va="center",
                            fontsize=8, color="#ffffff", weight="medium", zorder=4)
            left += share
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5, color=theme["ink2"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of one update() call")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}%"))
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title("cost share per stage", loc="left", color=theme["ink"])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=2, labelcolor=theme["ink2"])

    ax2 = axes[1]
    groups = []
    for dev in devs:
        st = _stage_ms(recs, dev)
        vals = [sum(st.get(k, 0.0) for k in keys) for _, keys in STAGE_GROUPS]
        vals.append(st.get("torch.binomial (C x 2F)", 0.0))
        ref = sel(recs, suite="phases", phase="stage", device=dev)[0]
        groups.append((device_label(ref), color_for(theme, dev), vals))
    short = ["clause eval", "votes + selection", "feedback counts", "apply_feedback",
             "bookkeeping", "torch.binomial"]
    _grouped_barh(ax2, short, groups, theme)
    ax2.set_xscale("log")
    ax2.set_xlabel("milliseconds per update (batch 256)")
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    for spine in ("top", "right", "left"):
        ax2.spines[spine].set_visible(False)
    ax2.grid(True, axis="x")
    ax2.grid(False, axis="y")
    ax2.tick_params(length=0)
    ax2.set_title("absolute cost per stage", loc="left", color=theme["ink"])
    hi = max(max(g[2]) for g in groups)
    ax2.set_xlim(right=hi * 12)
    for k, (_, _, values) in enumerate(groups):
        offs = (k - (len(groups) - 1) / 2) * 0.38
        for y, v in enumerate(values):
            ax2.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}", (v, y - offs), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=7.5,
                         color=theme["ink2"])
    ax2.legend(loc="upper left", bbox_to_anchor=(0.0, -0.32), ncol=2, labelcolor=theme["ink2"])
    fig.suptitle("Where an update goes — 5 000 clauses, 784 features, batch 256",
                 x=0.012, ha="left", fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_threads(recs, theme, env):
    """CPU thread scaling, with the GPU as a reference line."""
    if not sel(recs, suite="threads"):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for j, (phase, title) in enumerate([("train", "training (update)"),
                                        ("infer", "inference (forward)")]):
        ax = axes[j]
        cpu = series(recs, "threads", suite="threads", phase=phase, device="cpu")
        if cpu:
            c = color_for(theme, "cpu")
            xs = [p["threads"] for p in cpu]
            ys = [p["examples_per_s"] for p in cpu]
            ax.plot(xs, ys, color=c, marker="o", label="CPU",
                    markeredgecolor=theme["surface"], markeredgewidth=1.2)
            end_label(ax, xs[-1], ys[-1], thousands(ys[-1]), c, theme)
            ideal = [ys[0] * (x / xs[0]) for x in xs]
            ax.plot(xs, ideal, color=theme["ink3"], linewidth=1.0, zorder=1,
                    label="linear scaling from 1 thread")
        gpu = [r for r in sel(recs, suite="threads", phase=phase) if r["device"] != "cpu"]
        if gpu:
            g = color_for(theme, gpu[0]["device"])
            ax.axhline(gpu[0]["examples_per_s"], color=g, linewidth=2.0)
            ax.annotate(f"{device_label(gpu[0])} — {thousands(gpu[0]['examples_per_s'])}",
                        (1, gpu[0]["examples_per_s"]), xytext=(2, 5),
                        textcoords="offset points", color=theme["ink2"], fontsize=8.5)
        tidy(ax, xlog=True, ylog=True)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.set_xlabel("CPU threads (torch.set_num_threads)")
        ax.set_title(title, loc="left", color=theme["ink"])
        if j == 0:
            ax.set_ylabel("examples / s")
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.18), ncol=2,
                      labelcolor=theme["ink2"])
        ax.set_xlim(0.85, 48)
    fig.suptitle("CPU thread scaling — 5 000 clauses, 784 features, batch 256",
                 x=0.012, ha="left", fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_feedback(recs, theme, env):
    """Batched vs sequential feedback, per device."""
    scales = [("xor", "Noisy-XOR scale (20 clauses, 12 features)"),
              ("mnist-1k", "MNIST scale (1 000 clauses, 784 features)")]
    scales = [s for s in scales
              if sel(recs, suite="feedback", phase="train", scale=s[0])]
    if not scales:
        return None
    fig, axes = plt.subplots(1, len(scales), figsize=(4.8 * len(scales), 4.0), squeeze=False)
    axes = axes[0]
    for j, (scale, title) in enumerate(scales):
        ax = axes[j]
        rows = sel(recs, suite="feedback", phase="train", scale=scale)
        combos = sorted({(r["extra"]["feedback_mode"], r["batch_size"]) for r in rows},
                        key=lambda t: (t[0] != "batch", t[1]))
        labels = [f"{'batched' if m == 'batch' else 'sequential'}, B={b}" for m, b in combos]
        groups = []
        for dev in devices_of(rows):
            vals = []
            for mode, bs in combos:
                hit = [r for r in rows if r["device"] == dev
                       and r["extra"]["feedback_mode"] == mode and r["batch_size"] == bs]
                vals.append(hit[0]["examples_per_s"] if hit else 0.0)
            ref = [r for r in rows if r["device"] == dev][0]
            groups.append((device_label(ref), color_for(theme, dev), vals))
        _grouped_barh(ax, labels, groups, theme)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(length=0)
        ax.set_xlabel("examples / s (training)")
        ax.set_title(title, loc="left", color=theme["ink"])
        hi = max(max(g[2]) for g in groups)
        ax.set_xlim(left=max(1.0, min(min(v for v in g[2] if v > 0) for g in groups) / 3),
                    right=hi * 6)
        for k, (_, _, values) in enumerate(groups):
            offs = (k - (len(groups) - 1) / 2) * 0.38
            for y, v in enumerate(values):
                if v > 0:
                    ax.annotate(thousands(v), (v, y - offs), xytext=(4, 0),
                                textcoords="offset points", va="center", fontsize=7.5,
                                color=theme["ink2"])
        if j == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
                      labelcolor=theme["ink2"])
    fig.suptitle("Batched vs sequential feedback", x=0.012, ha="left", fontsize=13,
                 color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_mnist(recs, theme, env):
    """End-to-end MNIST: seconds per epoch and the accuracy that came with it."""
    rows = sel(recs, suite="mnist", phase="epoch")
    if not rows:
        return None
    order = [r["model"] for r in rows if r["device"] == "cpu"] or [rows[0]["model"]]
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    groups = []
    for dev in devices_of(rows):
        vals, accs = [], []
        for m in order:
            hit = [r for r in rows if r["device"] == dev and r["model"] == m]
            vals.append(hit[0]["sec_per_batch"] if hit else 0.0)
            accs.append(hit[0]["extra"].get("val_accuracy") if hit else None)
        ref = [r for r in rows if r["device"] == dev][0]
        groups.append((device_label(ref), color_for(theme, dev), vals, accs))
    labels = [m.replace(" ", "\n", 1) for m in order]
    _grouped_barh(ax, labels, [(g[0], g[1], g[2]) for g in groups], theme)
    ax.set_xscale("log")
    ax.set_xlabel("seconds per epoch (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(length=0)
    hi = max(max(g[2]) for g in groups)
    lo = min(min(v for v in g[2] if v > 0) for g in groups)
    ax.set_xlim(lo / 2.5, hi * 9)
    for k, (_, _, vals, accs) in enumerate(groups):
        offs = (k - (len(groups) - 1) / 2) * 0.38
        for y, (v, a) in enumerate(zip(vals, accs)):
            if v <= 0:
                continue
            txt = f"{v:.1f} s" + (f"  ·  {a:.1%} test acc." if a == a and a is not None else "")
            ax.annotate(txt, (v, y - offs), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=theme["ink2"])
    # speedup callout per model
    for y, m in enumerate(order):
        c = [r for r in rows if r["device"] == "cpu" and r["model"] == m]
        g = [r for r in rows if r["device"] != "cpu" and r["model"] == m]
        if c and g:
            ax.annotate(f"{c[0]['sec_per_batch'] / g[0]['sec_per_batch']:.0f}× faster",
                        (hi * 6.0, y), ha="right", va="center", fontsize=8.5,
                        color=theme["ink"], weight="medium")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2, labelcolor=theme["ink2"])
    fig.suptitle("End-to-end MNIST — wall clock per epoch", x=0.012, ha="left", fontsize=13,
                 color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def fig_segmentation(recs, theme, env):
    """Dense per-pixel throughput: pixels per second, training and inference."""
    rows = sel(recs, suite="segmentation", phase="train")
    if not rows:
        return None
    order = [r["model"] for r in rows if r["device"] == "cpu"] or [r["model"] for r in rows]
    seen, labels = set(), []
    for m in order:
        if m not in seen:
            seen.add(m)
            labels.append(m)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, phase, title in zip(axes, ("train", "infer"), ("training", "inference")):
        groups = []
        for dev in devices_of(sel(recs, suite="segmentation", phase=phase)):
            vals = []
            for m in labels:
                hit = [r for r in sel(recs, suite="segmentation", phase=phase, model=m)
                       if r["device"] == dev]
                vals.append(hit[0]["extra"].get("pixels_per_s", 0.0) if hit else 0.0)
            ref = [r for r in sel(recs, suite="segmentation", phase=phase) if r["device"] == dev][0]
            groups.append((device_label(ref), color_for(theme, dev), vals))
        _grouped_barh(ax, labels, groups, theme)
        ax.set_xscale("log")
        ax.set_xlabel(f"pixels per second — {title} (log scale)")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(length=0)
        hi = max((max(g[2]) for g in groups), default=1.0)
        ax.set_xlim(right=hi * 4)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2, labelcolor=theme["ink2"])
    fig.suptitle("Dense segmentation — pixels per second", x=0.012, ha="left", fontsize=13,
                 color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


FIGURES = {
    "throughput-vs-batch": fig_batch,
    "gpu-speedup": fig_speedup,
    "throughput-vs-model-size": fig_model_size,
    "model-zoo": fig_model_zoo,
    "update-cost-breakdown": fig_phases,
    "cpu-threads": fig_threads,
    "feedback-mode": fig_feedback,
    "mnist-epoch": fig_mnist,
    "segmentation-throughput": fig_segmentation,
}


# --------------------------------------------------------------------------------- tables
def md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def num(v: Optional[float], digits: int = 0) -> str:
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def speed_rows(recs, suite, xkey, xname) -> str:
    xs = sorted({r[xkey] for r in sel(recs, suite=suite)})
    rows = []
    for x in xs:
        row = [num(x)]
        for phase in ("train", "infer"):
            cpu = sel(recs, suite=suite, phase=phase, device="cpu", **{xkey: x})
            gpu = [r for r in sel(recs, suite=suite, phase=phase, **{xkey: x})
                   if r["device"] != "cpu"]
            c = cpu[0]["examples_per_s"] if cpu else None
            g = gpu[0]["examples_per_s"] if gpu else None
            row += [num(c), num(g), f"{g / c:,.1f}×" if c and g else "—"]
        rows.append(row)
    return md_table([xname, "CPU train ex/s", "GPU train ex/s", "train speedup",
                     "CPU infer ex/s", "GPU infer ex/s", "infer speedup"], rows)


def table_models(recs) -> str:
    rows = []
    for m in [r["model"] for r in sel(recs, suite="models", phase="train", device="cpu")]:
        row = [m]
        for phase in ("train", "infer"):
            cpu = sel(recs, suite="models", phase=phase, device="cpu", model=m)
            gpu = [r for r in sel(recs, suite="models", phase=phase, model=m)
                   if r["device"] != "cpu"]
            c = cpu[0]["examples_per_s"] if cpu else None
            g = gpu[0]["examples_per_s"] if gpu else None
            row += [num(c), num(g), f"{g / c:,.0f}×" if c and g else "—"]
        gpu = [r for r in sel(recs, suite="models", phase="train", model=m)
               if r["device"] != "cpu"]
        row.append(num(gpu[0]["peak_mem_mb"], 0) + " MB" if gpu else "—")
        rows.append(row)
    return md_table(["model", "CPU train ex/s", "GPU train ex/s", "train speedup",
                     "CPU infer ex/s", "GPU infer ex/s", "infer speedup",
                     "GPU peak memory"], rows)


STAGE_ROWS = ["encode", "evaluate", "votes+select", "feedback_counts", "accumulator_alloc",
              "coerce_targets", "apply_feedback", "refresh_include", "commit",
              "torch.binomial (C x 2F)"]


def table_phases(recs) -> str:
    devs = devices_of([r for r in recs if r["suite"] == "phases"])
    if not devs:
        return ""
    rows = []
    for st in STAGE_ROWS:
        row = [st]
        empty = True
        for dev in devs:
            v = _stage_ms(recs, dev).get(st)
            row.append(f"{v:,.3f}" if v is not None else "—")
            empty = empty and v is None
        if not empty:
            rows.append(row)
    totals = ["**update() total**"]
    for dev in devs:
        tot = sel(recs, suite="phases", phase="update_total", device=dev)
        totals.append(f"**{tot[0]['sec_per_batch'] * 1e3:,.3f}**" if tot else "—")
    rows.append(totals)
    return md_table(["stage"] + [f"{d} (ms)" for d in devs], rows)


def table_threads(recs) -> str:
    rows = []
    for r in series(recs, "threads", suite="threads", phase="train", device="cpu"):
        inf = sel(recs, suite="threads", phase="infer", device="cpu", threads=r["threads"])
        rows.append([r["threads"], num(r["examples_per_s"]),
                     num(inf[0]["examples_per_s"]) if inf else "—"])
    return md_table(["threads", "train ex/s", "infer ex/s"], rows)


def table_feedback(recs) -> str:
    rows = []
    for r in sorted(sel(recs, suite="feedback", phase="train"),
                    key=lambda r: (r["extra"]["scale"], r["extra"]["feedback_mode"],
                                   r["batch_size"], r["device"])):
        rows.append([r["extra"]["scale"], r["extra"]["feedback_mode"], r["batch_size"],
                     r["device"], num(r["examples_per_s"]),
                     f"{r['sec_per_batch'] * 1e3:,.2f}"])
    return md_table(["scale", "feedback_mode", "batch", "device", "examples/s",
                     "ms / update call"], rows)


def table_transfer(recs) -> str:
    rows = []
    for r in sorted(sel(recs, suite="transfer", phase="train"),
                    key=lambda r: (r["device"], r["batch_size"],
                                   not r["extra"]["data_resident_on_device"])):
        rows.append([r["device"], r["batch_size"],
                     "device" if r["extra"]["data_resident_on_device"] else "host",
                     num(r["examples_per_s"]), f"{r['sec_per_batch'] * 1e3:,.2f}"])
    return md_table(["device", "batch", "dataset lives on", "examples/s", "ms / update"], rows)


def table_mnist(recs) -> str:
    rows = []
    for r in sorted(sel(recs, suite="mnist", phase="epoch"),
                    key=lambda r: (r["model"], r["device"])):
        e = r["extra"]
        rows.append([r["model"], r["device"], f"{e['train_examples']:,}", e["epochs"],
                     f"{r['sec_per_batch']:,.2f}", num(r["examples_per_s"]),
                     f"{e['val_accuracy']:.4f}",
                     num(r["peak_mem_mb"], 0) + " MB" if r["peak_mem_mb"] else "—"])
    return md_table(["model", "device", "train examples", "epochs", "s / epoch",
                     "examples/s", "test accuracy", "GPU peak memory"], rows)


def table_segmentation(recs) -> str:
    rows = []
    seen = set()
    for r in sel(recs, suite="segmentation", phase="train"):
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        row = [r["model"], f"{r['extra'].get('patches_per_commit') or 'all'}"]
        for phase in ("train", "infer"):
            cpu = [q for q in sel(recs, suite="segmentation", phase=phase, model=r["model"])
                   if q["device"] == "cpu"]
            gpu = [q for q in sel(recs, suite="segmentation", phase=phase, model=r["model"])
                   if q["device"] != "cpu"]
            c = cpu[0]["extra"].get("pixels_per_s") if cpu else None
            g = gpu[0]["extra"].get("pixels_per_s") if gpu else None
            row += [num(c), num(g), f"{g / c:,.0f}\u00d7" if c and g else "\u2014"]
        rows.append(row)
    return md_table(["configuration", "patches/commit", "CPU train px/s", "GPU train px/s",
                     "train speedup", "CPU infer px/s", "GPU infer px/s", "infer speedup"], rows)


TABLES = [
    ("batch", "Batch size — MNIST scale (784 features, 500 clauses/class)",
     lambda r: speed_rows(r, "batch", "batch_size", "batch")),
    ("small", "Batch size — Noisy-XOR scale (12 features, 10 clauses/class)",
     lambda r: speed_rows(r, "small", "batch_size", "batch")),
    ("clauses", "Clause budget (784 features, batch 256)",
     lambda r: speed_rows(r, "clauses", "n_clauses_per_class", "clauses/class")),
    ("features", "Feature count (500 clauses/class, batch 256)",
     lambda r: speed_rows(r, "features", "n_features", "features")),
    ("models", "Model zoo (784 features, batch 128)", table_models),
    ("phases", "Update cost breakdown (5 000 clauses, 784 features, batch 256)", table_phases),
    ("threads", "CPU thread scaling (5 000 clauses, 784 features, batch 256)", table_threads),
    ("feedback", "Batched vs sequential feedback", table_feedback),
    ("transfer", "Host- vs device-resident data (training, 784 features, 500 clauses/class)",
     table_transfer),
    ("segmentation", "Dense segmentation (4 planes, 4 classes, batch 8 images)",
     table_segmentation),
    ("mnist", "End-to-end MNIST", table_mnist),
]


def build_tables(recs) -> Dict[str, str]:
    """Render every table; a suite that was not run yields an empty string."""
    out: Dict[str, str] = {}
    for slug, _heading, fn in TABLES:
        try:
            md = fn(recs)
        except (IndexError, KeyError, ValueError, ZeroDivisionError, TypeError):
            md = ""
        out[slug] = md if md.count("\n") > 2 else ""  # header + separator only = no data
    return out


def write_tables(recs, env, path: str) -> None:
    parts = [f"<!-- generated by benchmarks/plot_benchmarks.py — {env.get('timestamp', '')} -->\n"]
    tables = build_tables(recs)
    for slug, heading, _fn in TABLES:
        if tables[slug]:
            parts.append(f"## {heading}\n")
            parts.append(tables[slug])
    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- doc page
def one(recs, **crit) -> Optional[Dict]:
    hits = sel(recs, **crit)
    return hits[0] if hits else None


def gpu_of(recs, **crit) -> Optional[Dict]:
    hits = [r for r in sel(recs, **crit) if not r["device"].startswith("cpu")]
    return hits[0] if hits else None


def xps(rec: Optional[Dict]) -> Optional[float]:
    return rec["examples_per_s"] if rec else None


def ratio(a: Optional[float], b: Optional[float]) -> str:
    return f"{a / b:,.0f}×" if a and b else "—"


def prose_batch(recs, env) -> str:
    big = sorted({r["batch_size"] for r in sel(recs, suite="batch", phase="train")})
    if not big:
        return ""
    ref = 256 if 256 in big else big[-1]
    c = xps(one(recs, suite="batch", phase="train", device="cpu", batch_size=ref))
    g = xps(gpu_of(recs, suite="batch", phase="train", batch_size=ref))
    g1 = xps(gpu_of(recs, suite="batch", phase="train", batch_size=big[0]))
    gmax = xps(gpu_of(recs, suite="batch", phase="train", batch_size=big[-1]))
    out = [
        f"One `update()` call touches every automaton in the machine, so its cost depends far "
        f"more on the size of the machine than on how many examples are in the batch. Feeding "
        f"the GPU more examples per call is therefore nearly free: training throughput rises "
        f"from {num(g1)} examples/s at batch {big[0]} to {num(gmax)} at batch {big[-1]}. "
        f"At batch {ref} the same model trains {num(g)} examples/s on the GPU against "
        f"{num(c)} on the CPU ({ratio(g, c)})."
    ]
    small = series(recs, "batch_size", suite="small", phase="train")
    if small:
        cross = None
        for bs in sorted({r["batch_size"] for r in small}):
            cc = xps(one(recs, suite="small", phase="train", device="cpu", batch_size=bs))
            gg = xps(gpu_of(recs, suite="small", phase="train", batch_size=bs))
            if cc and gg and gg >= cc:
                cross = bs
                break
        tail = (f"only past batch {cross} does the GPU pull ahead"
                if cross else "the GPU never catches up over the range measured here")
        out.append(
            f"The Noisy-XOR scale machine in the bottom row is the counter-example: 20 clauses "
            f"over 12 features is less work than a kernel launch, so {tail}. Small machines "
            f"belong on the CPU."
        )
    return "\n\n".join(out)


def prose_speedup(recs, env) -> str:
    pts = [(r["batch_size"],
            xps(r) / xps(one(recs, suite="batch", phase="train", device="cpu",
                             batch_size=r["batch_size"]) or {"examples_per_s": 0}) or 0)
           for r in series(recs, "batch_size", suite="batch", phase="train")
           if not r["device"].startswith("cpu")
           and xps(one(recs, suite="batch", phase="train", device="cpu",
                       batch_size=r["batch_size"]))]
    if not pts:
        return ""
    bs, best = max(pts, key=lambda t: t[1])
    last_bs, last = pts[-1]
    trend = (f"and falls back to {last:,.0f}× at batch {last_bs}, where the CPU finally "
             f"amortises its own vectorisation over a long batch"
             if last < best * 0.8 else "and holds from there")
    inf = [xps(r) / xps(one(recs, suite="batch", phase="infer", device="cpu",
                            batch_size=r["batch_size"]))
           for r in series(recs, "batch_size", suite="batch", phase="infer")
           if not r["device"].startswith("cpu")
           and xps(one(recs, suite="batch", phase="infer", device="cpu",
                       batch_size=r["batch_size"]))]
    inf_bit = (f" Inference behaves differently again: the CPU forward pass is already cheap, so "
               f"the gap there tops out at {max(inf):,.0f}× — worth having, but not the same "
               f"kind of difference." if inf else "")
    return (f"The speedup is not one number. It grows with batch size because the CPU pays the "
            f"per-clause cost serially while the GPU hides it, peaks at about {best:,.0f}× "
            f"around batch {bs}, {trend}.{inf_bit}")


def prose_model_size(recs, env) -> str:
    cs = sorted({r["n_clauses_per_class"] for r in sel(recs, suite="clauses", phase="train")})
    fs = sorted({r["n_features"] for r in sel(recs, suite="features", phase="train")})
    bits = []
    if len(cs) >= 2:
        lo = ratio(xps(gpu_of(recs, suite="clauses", phase="train", n_clauses_per_class=cs[0])),
                   xps(one(recs, suite="clauses", phase="train", device="cpu",
                           n_clauses_per_class=cs[0])))
        hi = ratio(xps(gpu_of(recs, suite="clauses", phase="train", n_clauses_per_class=cs[-1])),
                   xps(one(recs, suite="clauses", phase="train", device="cpu",
                           n_clauses_per_class=cs[-1])))
        bits.append(f"{lo} at {cs[0]} clauses/class against {hi} at {cs[-1]}")
    if len(fs) >= 2:
        lo = ratio(xps(gpu_of(recs, suite="features", phase="train", n_features=fs[0])),
                   xps(one(recs, suite="features", phase="train", device="cpu",
                           n_features=fs[0])))
        hi = ratio(xps(gpu_of(recs, suite="features", phase="train", n_features=fs[-1])),
                   xps(one(recs, suite="features", phase="train", device="cpu",
                           n_features=fs[-1])))
        bits.append(f"{lo} at {fs[0]} features against {hi} at {fs[-1]}")
    if not bits:
        return ""
    return (f"Both axes cost the same thing — the update is `n_clauses × 2 × n_features` "
            f"element-wise work — so both curves fall roughly as `1/x`. The CPU falls off "
            f"immediately; the GPU stays flat until the machine is big enough to fill it, which "
            f"is why the gap *widens* with model size: {'; '.join(bits)}. Big machines are "
            f"exactly the ones worth moving to a GPU.")


def prose_zoo(recs, env) -> str:
    flat = xps(gpu_of(recs, suite="models", phase="train", model="TsetlinMachine"))
    conv = xps(gpu_of(recs, suite="models", phase="train", model="ConvTsetlinMachine"))
    if not flat:
        return ""
    tail = ""
    if conv:
        tail = (f" The convolutional models are the outlier at {ratio(flat, conv)} the cost of "
                f"the flat machine per example — a 10×10 patch over a 28×28 image is 361 "
                f"patches, and every patch is evaluated against every clause.")
    return (f"Weighted and coalesced variants cost about what the plain machine costs: they "
            f"change the output layer and the feedback policy, not the per-clause work that "
            f"dominates.{tail}")


def prose_phases(recs, env) -> str:
    devs = devices_of([r for r in recs if r["suite"] == "phases"])
    if not devs:
        return ""
    bits, draws = [], []
    for dev in devs:
        st = _stage_ms(recs, dev)
        tot = one(recs, suite="phases", phase="update_total", device=dev)
        commit = st.get("apply_feedback", st.get("commit"))
        binom = st.get("torch.binomial (C x 2F)")
        if not tot or not commit:
            continue
        total_ms = tot["sec_per_batch"] * 1e3
        label = "CPU" if dev.startswith("cpu") else "GPU"
        bits.append(f"on the {label} it is {100.0 * commit / total_ms:.0f} % of the "
                    f"{total_ms:,.1f} ms call")
        if binom:
            draws.append(f"{binom:,.2f} ms on the {label}")
    if not bits:
        return ""
    draw_bit = (f" A standalone `torch.binomial` draw of the same `(C, 2F)` shape costs "
                f"{' and '.join(draws)} — the single most expensive op in the commit."
                if draws else "")
    return (f"`apply_feedback` — the stochastic increment/decrement of the whole "
            f"`(n_clauses, 2F)` automaton state — is the expensive part of an update: "
            f"{'; '.join(bits)}.{draw_bit} Everything before the commit (encoding, clause "
            f"evaluation, the vote sums and the feedback count matmuls) is comparatively cheap. "
            f"That is also why batching pays: a batched update draws once per commit, a "
            f"sequential one draws once per example.")


def prose_threads(recs, env) -> str:
    tr = series(recs, "threads", suite="threads", phase="train", device="cpu")
    inf = series(recs, "threads", suite="threads", phase="infer", device="cpu")
    if len(tr) < 2:
        return ""
    t_gain = xps(tr[-1]) / xps(tr[0])
    i_gain = xps(inf[-1]) / xps(inf[0]) if len(inf) >= 2 else None
    inf_bit = (f" Inference scales much better ({i_gain:,.1f}× over the same range): the forward "
               f"pass is one big matmul-shaped reduction." if i_gain else "")
    return (f"More CPU threads buy surprisingly little in training: {t_gain:,.1f}× going from "
            f"{tr[0]['threads']} to {tr[-1]['threads']} threads. The batched update is a chain "
            f"of large element-wise ops over the state, and the binomial draw that dominates it "
            f"does not thread.{inf_bit} If a CPU run is too slow, the lever is the clause budget "
            f"or the batch size, not `torch.set_num_threads`.")


def prose_feedback(recs, env) -> str:
    rows = sel(recs, suite="feedback", phase="train")
    if not rows:
        return ""
    gaps = []
    for scale in ("xor", "mnist-1k"):
        for dev in devices_of(rows):
            b = one(recs, suite="feedback", phase="train", device=dev, scale=scale,
                    feedback_mode="batch", batch_size=32)
            s = one(recs, suite="feedback", phase="train", device=dev, scale=scale,
                    feedback_mode="sequential", batch_size=32)
            if b and s and xps(s):
                gaps.append(xps(b) / xps(s))
    if not gaps:
        return ""
    span = (f"{min(gaps):,.0f}×" if max(gaps) - min(gaps) < 1
            else f"{min(gaps):,.0f}-{max(gaps):,.0f}×")
    return (f"`feedback_mode=\"sequential\"` is the exact per-example algorithm: it commits once "
            f"per example, so a mini-batch of 32 pays 32 commits instead of one. Measured at "
            f"batch 32 that is {span} slower than the batched update, depending on the model "
            f"scale and the device. Batching is the default for that "
            f"reason — but it is an approximation, and its fidelity degrades as the batch grows. "
            f"See [Batched vs sequential feedback](concepts/batching.md) for the accuracy side of "
            f"this trade.")


def prose_transfer(recs, env) -> str:
    rows = sel(recs, suite="transfer", phase="train")
    if not rows:
        return ""
    worst = 0.0
    for r in rows:
        if r["extra"].get("data_resident_on_device"):
            continue
        res = one(recs, suite="transfer", phase="train", device=r["device"],
                  batch_size=r["batch_size"], data_resident_on_device=True)
        if res and xps(res):
            worst = max(worst, 100.0 * (1.0 - xps(r) / xps(res)))
    cost = "under 1 %" if worst < 1 else f"at most {worst:.0f} %"
    return (f"Keeping the Boolean dataset in host memory and copying each mini-batch across costs "
            f"{cost} of throughput here — the update is long enough to hide the "
            f"transfer. So stream a dataset that does not fit in VRAM without worrying, but do "
            f"hand the *model* a device tensor: see "
            f"[Choosing a device](getting-started/devices.md) for the one copy that does bite "
            f"(Booleanization encoders keep their thresholds on the CPU).")


def prose_mnist(recs, env) -> str:
    rows = sel(recs, suite="mnist", phase="epoch")
    if not rows:
        return ""
    bits = []
    for m in sorted({r["model"] for r in rows}):
        c = one(recs, suite="mnist", phase="epoch", device="cpu", model=m)
        g = gpu_of(recs, suite="mnist", phase="epoch", model=m)
        if c and g:
            bits.append(f"**{m}** — {g['sec_per_batch']:,.1f} s/epoch on the GPU vs "
                        f"{c['sec_per_batch']:,.0f} s on the CPU "
                        f"({c['sec_per_batch'] / g['sec_per_batch']:,.0f}×), "
                        f"{g['extra']['val_accuracy']:.1%} test accuracy")
        elif g or c:
            r = g or c
            bits.append(f"**{m}** — {r['sec_per_batch']:,.1f} s/epoch on {r['device']}, "
                        f"{r['extra']['val_accuracy']:.1%} test accuracy")
    return ("Synthetic micro-benchmarks flatter whichever device wins the micro-benchmark, so "
            "here is the whole thing end to end, `Trainer.fit` included:\n\n"
            + "\n".join(f"- {b}" for b in bits)
            + "\n\nAccuracy matches across devices to within run-to-run noise, as it should — "
              "only the wall clock moves. These are short runs at a modest clause budget; see "
              "[Benchmarks](benchmarks.md) for what longer runs reach.")


def prose_segmentation(recs, env) -> str:
    rows = sel(recs, suite="segmentation", phase="train")
    if not rows:
        return ""
    base = [r for r in rows if r["model"] == "dense 3x3 32px"]
    inf = sel(recs, suite="segmentation", phase="infer", model="dense 3x3 32px")
    ppc_none = [r for r in rows if r["model"] == "dense 3x3 32px ppc=None"]
    bits = []
    if base and inf:
        b, i = base[0], inf[0]
        bits.append(
            f"A 3\u00d73 dense model on 32\u00d732 inputs runs at "
            f"{b['extra']['pixels_per_s']:,.0f} px/s training and "
            f"{i['extra']['pixels_per_s']:,.0f} px/s predicting on {device_label(b)} \u2014 "
            f"prediction is roughly {i['extra']['pixels_per_s'] / b['extra']['pixels_per_s']:,.0f}\u00d7 "
            "cheaper, because it never draws feedback or touches the automata."
        )
    if base and ppc_none:
        bits.append(
            f"`patches_per_commit` is the price of fidelity: committing once per batch reaches "
            f"{ppc_none[0]['extra']['pixels_per_s']:,.0f} px/s against "
            f"{base[0]['extra']['pixels_per_s']:,.0f} at the default 128, a "
            f"{ppc_none[0]['extra']['pixels_per_s'] / base[0]['extra']['pixels_per_s']:,.1f}\u00d7 "
            "speedup that costs real accuracy (see the segmentation guide's ablation)."
        )
    return ("Dense prediction evaluates one patch per **pixel**, so throughput is quoted in "
            "pixels per second rather than examples per second \u2014 one 32\u00d732 image is "
            "1 024 examples.\n\n" + "\n".join(f"- {b}" for b in bits))


# (figure, heading, prose, [table slugs])
PAGE_SECTIONS = [
    ("throughput-vs-batch", "Throughput vs batch size", prose_batch, ["batch", "small"]),
    ("gpu-speedup", "How much the GPU buys you", prose_speedup, []),
    ("throughput-vs-model-size", "Throughput vs model size", prose_model_size,
     ["clauses", "features"]),
    ("model-zoo", "The model zoo", prose_zoo, ["models"]),
    ("update-cost-breakdown", "Where an update goes", prose_phases, ["phases"]),
    ("cpu-threads", "CPU thread scaling", prose_threads, ["threads"]),
    ("feedback-mode", "Batched vs sequential feedback", prose_feedback, ["feedback"]),
    (None, "Where the data lives", prose_transfer, ["transfer"]),
    ("segmentation-throughput", "Dense segmentation", prose_segmentation, ["segmentation"]),
    ("mnist-epoch", "End to end: MNIST", prose_mnist, ["mnist"]),
]


def env_table(env: Dict) -> str:
    rows = [["CPU", f"{env.get('cpu', '?')} ({env.get('cpu_count', '?')} logical cores, "
                    f"torch using {env.get('torch_threads', '?')} threads)"]]
    for i, gpu in enumerate(env.get("gpus", [])):
        rows.append([f"GPU {i}", gpu])
    rows += [
        ["PyTorch", f"{env.get('torch', '?')}" +
                    (f" / CUDA {env['cuda']}" if env.get("cuda") else "")],
        ["torchtsetlin", env.get("torchtsetlin", "?")],
        ["Python", env.get("python", "?")],
        ["measured", env.get("timestamp", "?")],
    ]
    return md_table(["", ""], rows)


def write_page(recs, env, path: str, fig_dir: str, results: str, figures: Sequence[str]) -> None:
    """Assemble the CPU-vs-GPU documentation page from the figures and tables."""
    rel = os.path.relpath(fig_dir, os.path.dirname(path) or ".")
    tables = build_tables(recs)
    out = [
        "# CPU vs GPU",
        "",
        f"<!-- generated by benchmarks/plot_benchmarks.py from {results} — do not edit by hand; "
        f"re-run ./benchmarks/run_benchmarks.sh -->",
        "",
        "Every number on this page comes from one run of `benchmarks/run_benchmarks.sh` on a "
        "single machine:",
        "",
        env_table(env),
        "",
        "!!! note \"How to read these\"",
        "",
        "    Each point is the median of three timed blocks of steady-state calls, after a "
        "warm-up, with the device synchronised once per block. The micro-benchmarks use "
        "synthetic Boolean data at 20 % density, resident on the device unless stated "
        "otherwise. Reproduce — or replace these numbers with your own hardware's — with:",
        "",
        "    ```bash",
        "    ./benchmarks/run_benchmarks.sh          # ~20-25 min, rewrites this page",
        "    ```",
        "",
    ]
    for fig, heading, prose, slugs in PAGE_SECTIONS:
        body = prose(recs, env)
        has_tables = [s for s in slugs if tables.get(s)]
        has_fig = fig in figures
        if not body and not has_tables and not has_fig:
            continue
        out += [f"## {heading}", ""]
        if has_fig:
            out += [
                f"![{heading}]({rel}/{fig}.png#only-light)",
                f"![{heading}]({rel}/{fig}_dark.png#only-dark)",
                "",
            ]
        if body:
            out += [body, ""]
        for slug in has_tables:
            heading_for = dict((s, h) for s, h, _ in TABLES)[slug]
            out += [f"??? abstract \"{heading_for}\"", ""]
            out += ["    " + line if line else "" for line in tables[slug].splitlines()]
            out += [""]
    out += [
        "## Reproducing",
        "",
        "```bash",
        "./benchmarks/run_benchmarks.sh                  # every suite, cpu + cuda:0",
        "./benchmarks/run_benchmarks.sh --quick          # ~1 min smoke test",
        "./benchmarks/run_benchmarks.sh -s batch models  # a subset",
        "./benchmarks/run_benchmarks.sh --plots-only     # re-render from existing results",
        "```",
        "",
        f"The raw records live in `{results}`; `benchmarks/plot_benchmarks.py` turns them into "
        f"the figures above and writes this page.",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results/cpu_vs_gpu.json")
    ap.add_argument("--out", default="docs/assets/benchmarks")
    ap.add_argument("--tables", default="benchmarks/results/tables.md")
    ap.add_argument("--page", default="docs/cpu-vs-gpu.md",
                    help="documentation page to (re)generate")
    ap.add_argument("--no-page", action="store_true", help="do not write the documentation page")
    ap.add_argument("--figures", nargs="*", default=None, choices=list(FIGURES))
    args = ap.parse_args()

    payload = load(args.results)
    recs, env = payload["records"], payload["environment"]
    os.makedirs(args.out, exist_ok=True)
    names = args.figures or list(FIGURES)
    written = []
    for name in names:
        ok = True
        for mode, theme in THEMES.items():
            style(theme)
            fig = FIGURES[name](recs, theme, env)
            if fig is None:
                print(f"skipped {name} ({mode}) — no data")
                ok = False
                continue
            suffix = "" if mode == "light" else "_dark"
            path = os.path.join(args.out, f"{name}{suffix}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
            plt.close(fig)
            print(f"wrote {path}")
        if ok:
            written.append(name)
    write_tables(recs, env, args.tables)
    if not args.no_page:
        os.makedirs(os.path.dirname(args.page) or ".", exist_ok=True)
        write_page(recs, env, args.page, args.out, args.results, written)


if __name__ == "__main__":
    main()
