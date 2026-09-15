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


def thousands(v: float, _pos: int = 0) -> str:
    if v >= 1_000_000:
        return f"{v / 1e6:g}M"
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


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


def end_label(ax, x, y, text, color, theme, dy=1.06, ha="left", fontsize=8.5) -> None:
    """Direct label at the end of a line — in ink, with the coloured marker beside it."""
    ax.annotate(text, (x, y * dy if ax.get_yscale() == "log" else y),
                xytext=(4, 0), textcoords="offset points", color=theme["ink2"],
                fontsize=fontsize, ha=ha, va="center", clip_on=False)
    ax.plot([x], [y], marker="o", markersize=6, color=color,
            markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=5, clip_on=False)


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
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    for i, (suite, scale_title) in enumerate(scales):
        for j, (phase, phase_title) in enumerate(phases):
            ax = axes[i][j]
            for dev in devices_of(recs):
                pts = series(recs, "batch_size", suite=suite, phase=phase, device=dev)
                if not pts:
                    continue
                c = color_for(theme, dev)
                xs = [p["batch_size"] for p in pts]
                ys = [p["examples_per_s"] for p in pts]
                ax.plot(xs, ys, color=c, marker="o", label=device_label(pts[0]),
                        markeredgecolor=theme["surface"], markeredgewidth=1.2)
                end_label(ax, xs[-1], ys[-1], thousands(ys[-1]), c, theme)
            tidy(ax, xlog=True, ylog=True)
            ax.set_title(f"{scale_title}\n{phase_title}", loc="left", color=theme["ink"])
            if i == 1:
                ax.set_xlabel("batch size (examples per update call)")
            if j == 0:
                ax.set_ylabel("examples / s")
            ax.set_xlim(right=max(p["batch_size"] for p in sel(recs, suite=suite)) * 2.6)
    axes[0][0].legend(loc="upper left", labelcolor=theme["ink2"])
    fig.suptitle("Throughput vs batch size — CPU vs GPU", x=0.012, ha="left",
                 fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return fig


def fig_speedup(recs, theme, env):
    """GPU-over-CPU speedup, one line per model scale."""
    scales = [("batch", "MNIST scale (5 000 clauses)", 6), ("small", "Noisy-XOR scale (20 clauses)", 2)]
    phases = [("train", "training (update)"), ("infer", "inference (forward)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for j, (phase, phase_title) in enumerate(phases):
        ax = axes[j]
        ax.axhline(1.0, color=theme["ink3"], linewidth=1.0, zorder=1)
        ax.annotate("1× — parity", (1, 1.0), xytext=(2, 5), textcoords="offset points",
                    color=theme["ink3"], fontsize=8)
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
            end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.0f}×" if ys[-1] >= 10 else f"{ys[-1]:.1f}×",
                      c, theme)
        tidy(ax, xlog=True, ylog=True)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}×"))
        ax.set_title(phase_title, loc="left", color=theme["ink"])
        ax.set_xlabel("batch size")
        ax.set_xlim(right=5000 * 2.2)
    axes[0].set_ylabel("GPU throughput / CPU throughput")
    axes[0].legend(loc="upper left", labelcolor=theme["ink2"])
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
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for i, (phase, phase_title) in enumerate(rows):
        for j, (suite, xkey, xlabel) in enumerate(cols):
            ax = axes[i][j]
            for dev in devices_of(recs):
                pts = series(recs, xkey, suite=suite, phase=phase, device=dev)
                if not pts:
                    continue
                c = color_for(theme, dev)
                xs = [p[xkey] for p in pts]
                ys = [p["examples_per_s"] for p in pts]
                ax.plot(xs, ys, color=c, marker="o", label=device_label(pts[0]),
                        markeredgecolor=theme["surface"], markeredgewidth=1.2)
                end_label(ax, xs[-1], ys[-1], thousands(ys[-1]), c, theme)
            tidy(ax, xlog=True, ylog=True)
            ax.set_title(phase_title, loc="left", color=theme["ink"])
            ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel("examples / s")
            ax.set_xlim(right=max(p[xkey] for p in sel(recs, suite=suite)) * 2.4)
    axes[0][0].legend(loc="lower left", labelcolor=theme["ink2"])
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
            ax.legend(loc="lower right", labelcolor=theme["ink2"])
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
    return {r["extra"]["stage"]: r["sec_per_batch"] * 1e3
            for r in sel(recs, suite="phases", phase="stage", device=device)}


def fig_phases(recs, theme, env):
    """Where the time of one update goes, per device."""
    devs = devices_of([r for r in recs if r["suite"] == "phases"])
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
        labels.append(f"{device_label(ref)}\n{total:.1f} ms / update")
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
    ax2.legend(loc="lower right", labelcolor=theme["ink2"])
    fig.suptitle("Where an update goes — 5 000 clauses, 784 features, batch 256",
                 x=0.012, ha="left", fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_threads(recs, theme, env):
    """CPU thread scaling, with the GPU as a reference line."""
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
            ax.legend(loc="lower right", labelcolor=theme["ink2"])
        ax.set_xlim(0.85, 48)
    fig.suptitle("CPU thread scaling — 5 000 clauses, 784 features, batch 256",
                 x=0.012, ha="left", fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def fig_feedback(recs, theme, env):
    """Batched vs sequential feedback, per device."""
    scales = [("xor", "Noisy-XOR scale (20 clauses, 12 features)"),
              ("mnist-1k", "MNIST scale (1 000 clauses, 784 features)")]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
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
            ax.legend(loc="lower right", labelcolor=theme["ink2"])
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
    ax.legend(loc="lower right", labelcolor=theme["ink2"])
    fig.suptitle("End-to-end MNIST — wall clock per epoch", x=0.012, ha="left", fontsize=13,
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


def write_tables(recs, env, path: str) -> None:
    parts = [f"<!-- generated by benchmarks/plot_benchmarks.py — {env.get('timestamp', '')} -->\n"]
    parts.append("## batch size (MNIST scale: 784 features, 500 clauses/class)\n")
    parts.append(speed_rows(recs, "batch", "batch_size", "batch"))
    parts.append("\n## batch size (Noisy-XOR scale: 12 features, 10 clauses/class)\n")
    parts.append(speed_rows(recs, "small", "batch_size", "batch"))
    parts.append("\n## clause budget (784 features, batch 256)\n")
    parts.append(speed_rows(recs, "clauses", "n_clauses_per_class", "clauses/class"))
    parts.append("\n## feature count (500 clauses/class, batch 256)\n")
    parts.append(speed_rows(recs, "features", "n_features", "features"))

    parts.append("\n## model zoo (784 features, batch 128)\n")
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
    parts.append(md_table(["model", "CPU train ex/s", "GPU train ex/s", "train speedup",
                           "CPU infer ex/s", "GPU infer ex/s", "infer speedup",
                           "GPU peak memory"], rows))

    parts.append("\n## update cost breakdown (5 000 clauses, 784 features, batch 256)\n")
    stages = ["encode", "evaluate", "votes+select", "feedback_counts", "accumulator_alloc",
              "coerce_targets", "apply_feedback", "refresh_include", "commit",
              "torch.binomial (C x 2F)"]
    devs = devices_of([r for r in recs if r["suite"] == "phases"])
    rows = []
    for st in stages:
        row = [st]
        for dev in devs:
            v = _stage_ms(recs, dev).get(st)
            row.append(f"{v:,.3f}" if v is not None else "—")
        rows.append(row)
    totals = ["**update() total**"]
    for dev in devs:
        tot = sel(recs, suite="phases", phase="update_total", device=dev)
        totals.append(f"**{tot[0]['sec_per_batch'] * 1e3:,.3f}**" if tot else "—")
    rows.append(totals)
    parts.append(md_table(["stage"] + [f"{d} (ms)" for d in devs], rows))

    parts.append("\n## CPU thread scaling (5 000 clauses, 784 features, batch 256)\n")
    rows = []
    for r in series(recs, "threads", suite="threads", phase="train", device="cpu"):
        inf = sel(recs, suite="threads", phase="infer", device="cpu", threads=r["threads"])
        rows.append([r["threads"], num(r["examples_per_s"]),
                     num(inf[0]["examples_per_s"]) if inf else "—"])
    parts.append(md_table(["threads", "train ex/s", "infer ex/s"], rows))

    parts.append("\n## batched vs sequential feedback\n")
    rows = []
    for r in sorted(sel(recs, suite="feedback", phase="train"),
                    key=lambda r: (r["extra"]["scale"], r["extra"]["feedback_mode"],
                                   r["batch_size"], r["device"])):
        rows.append([r["extra"]["scale"], r["extra"]["feedback_mode"], r["batch_size"],
                     r["device"], num(r["examples_per_s"]),
                     f"{r['sec_per_batch'] * 1e3:,.2f}"])
    parts.append(md_table(["scale", "feedback_mode", "batch", "device", "examples/s",
                           "ms / update call"], rows))

    parts.append("\n## host-resident vs device-resident data (training, 784 features, "
                 "500 clauses/class)\n")
    rows = []
    for r in sorted(sel(recs, suite="transfer", phase="train"),
                    key=lambda r: (r["device"], r["batch_size"],
                                   not r["extra"]["data_resident_on_device"])):
        rows.append([r["device"], r["batch_size"],
                     "device" if r["extra"]["data_resident_on_device"] else "host",
                     num(r["examples_per_s"]), f"{r['sec_per_batch'] * 1e3:,.2f}"])
    parts.append(md_table(["device", "batch", "dataset lives on", "examples/s",
                           "ms / update"], rows))

    parts.append("\n## end-to-end MNIST\n")
    rows = []
    for r in sorted(sel(recs, suite="mnist", phase="epoch"), key=lambda r: (r["model"], r["device"])):
        e = r["extra"]
        rows.append([r["model"], r["device"], f"{e['train_examples']:,}", e["epochs"],
                     f"{r['sec_per_batch']:,.2f}", num(r["examples_per_s"]),
                     f"{e['val_accuracy']:.4f}",
                     num(r["peak_mem_mb"], 0) + " MB" if r["peak_mem_mb"] else "—"])
    parts.append(md_table(["model", "device", "train examples", "epochs", "s / epoch",
                           "examples/s", "test accuracy", "GPU peak memory"], rows))

    with open(path, "w") as fh:
        fh.write("\n".join(parts))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/results/cpu_vs_gpu.json")
    ap.add_argument("--out", default="docs/assets/benchmarks")
    ap.add_argument("--tables", default="benchmarks/results/tables.md")
    ap.add_argument("--figures", nargs="*", default=None, choices=list(FIGURES))
    args = ap.parse_args()

    payload = load(args.results)
    recs, env = payload["records"], payload["environment"]
    os.makedirs(args.out, exist_ok=True)
    names = args.figures or list(FIGURES)
    for name in names:
        for mode, theme in THEMES.items():
            style(theme)
            fig = FIGURES[name](recs, theme, env)
            if fig is None:
                print(f"skipped {name} ({mode}) — no data")
                continue
            suffix = "" if mode == "light" else "_dark"
            path = os.path.join(args.out, f"{name}{suffix}.png")
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
            plt.close(fig)
            print(f"wrote {path}")
    write_tables(recs, env, args.tables)


if __name__ == "__main__":
    main()
