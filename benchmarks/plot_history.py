#!/usr/bin/env python
"""Compare the benchmark results of every torchtsetlin version side by side.

``run_benchmarks.sh`` keeps one results file per library version under
``benchmarks/results/versions/<version>/cpu_vs_gpu.json``. This script reads all of them and
writes the "Benchmark history" documentation page: the machines each version was measured
on, a headline figure (speed relative to the oldest version), a headline table, and one
collapsible table per suite with a column per version.

A configuration is matched across versions by its settings (suite, phase, device kind,
model, batch size, sizes, threads and the string/boolean settings in ``extra``), never by
position, so a suite added in a later version simply starts its row there.

Usage::

    python benchmarks/plot_history.py                       # defaults below
    python benchmarks/plot_history.py --versions benchmarks/results/versions \\
        --page docs/benchmark-history.md --out docs/assets/benchmarks
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from plot_benchmarks import TABLES, THEMES, md_table, sig3, style, tidy  # noqa: E402

# settings that identify a configuration; everything else in a record is a measurement
CONFIG_FIELDS = ("model", "batch_size", "n_features", "n_clauses_per_class", "n_classes",
                 "threads")
# integer/list settings kept in ``extra`` (floats there are always measurements)
CONFIG_EXTRA = ("stage", "scale", "feedback_mode", "data_resident_on_device", "threads_axis",
                "patch", "image_px", "patches_per_commit", "volume", "patches_per_chunk",
                "train_examples", "val_examples", "epochs")
FIELD_NAMES = {"batch_size": "batch", "n_features": "features",
               "n_clauses_per_class": "clauses/class", "n_classes": "classes",
               "threads": "threads"}
SUITE_HEADINGS = {slug: heading for slug, heading, _fn in TABLES}

# (label, suite, phase, criteria) — one line per metric in the headline figure and table
HEADLINES = [
    ("Flat, batch 32, train", "batch", "train", {"model": "flat", "batch_size": 32}),
    ("Flat, batch 256, train", "batch", "train", {"model": "flat", "batch_size": 256}),
    ("Flat, batch 4096, predict", "batch", "infer", {"model": "flat", "batch_size": 4096}),
    ("Conv 2D, train", "models", "train", {"model": "ConvTsetlinMachine"}),
    ("Sequential feedback, train", "feedback", "train",
     {"model": "mnist-1k/sequential", "batch_size": 32}),
    ("MNIST epoch, flat", "mnist", "epoch", {"model": "TsetlinMachine 500/class"}),
]


# ---------------------------------------------------------------------------------- data
def version_key(v: str) -> Tuple:
    """0.10.0 sorts after 0.9.1; a suffix such as ``.dev1`` sorts before the release."""
    nums = [int(n) for n in re.findall(r"\d+", v.split("+")[0])[:3]]
    return tuple(nums) + ((0,) if re.search(r"[a-z]", v) else (1,))


def load_versions(root: str) -> List[Tuple[str, Dict]]:
    """[(version, payload)], oldest first, without records carried over from older versions.

    Before results were kept per version, a run of a single suite merged into the previous
    version's file, so 0.3.0's file holds 0.2.0's numbers for every suite but conv3d. A
    record whose timings equal an older version's record exactly was not re-measured; it is
    dropped here so the older measurement is not credited to the newer version.
    """
    out = []
    for path in glob.glob(os.path.join(root, "*", "cpu_vs_gpu.json")):
        with open(path) as fh:
            payload = json.load(fh)
        version = payload.get("environment", {}).get("torchtsetlin") or \
            os.path.basename(os.path.dirname(path))
        out.append((version, payload))
    out.sort(key=lambda vp: version_key(vp[0]))
    seen = set()
    for _v, payload in out:
        fresh = []
        for rec in payload.get("records", []):
            sig = (config_key(rec), json.dumps(rec.get("reps_sec_per_batch")))
            if sig not in seen:
                fresh.append(rec)
        seen |= {(config_key(r), json.dumps(r.get("reps_sec_per_batch")))
                 for r in payload.get("records", [])}
        payload["records"] = fresh
    return out


def previous(pts: Dict[int, Tuple[float, bool]], last: int) -> Optional[int]:
    """The newest version before ``last`` that measured the configuration."""
    earlier = [i for i in pts if i < last]
    return max(earlier) if earlier else None


def device_kind(rec: Dict) -> str:
    return "CPU" if rec["device"].startswith("cpu") else "GPU"


def config_key(rec: Dict) -> Tuple:
    extra = rec.get("extra") or {}
    parts = [rec["suite"], rec["phase"], device_kind(rec)]
    parts += [rec.get(f) for f in CONFIG_FIELDS]
    for f in CONFIG_EXTRA:
        v = extra.get(f)
        parts.append(tuple(v) if isinstance(v, list) else v)
    return tuple(parts)


def measure(rec: Dict) -> Tuple[Optional[float], str, bool]:
    """(value, unit, higher_is_better) for one record."""
    if rec["phase"] in ("stage", "update_total"):
        return rec["sec_per_batch"] * 1e3, "ms", False
    if rec["phase"] == "epoch":
        return rec["sec_per_batch"], "s/epoch", False
    return rec.get("examples_per_s"), "ex/s", True


def index(payload: Dict) -> Dict[Tuple, Dict]:
    out: Dict[Tuple, Dict] = {}
    for rec in payload.get("records", []):
        out.setdefault(config_key(rec), rec)  # several GPUs: the first one measured
    return out


def speed_factor(new: Optional[float], old: Optional[float], higher: bool) -> Optional[float]:
    """How many times faster ``new`` is than ``old`` (> 1 = faster)."""
    if not new or not old:
        return None
    return new / old if higher else old / new


def fmt_value(v: Optional[float], unit: str) -> str:
    if v is None:
        return "—"
    if unit == "ex/s":
        return f"{v:,.0f}" if v >= 100 else sig3(v)
    return f"{v:,.1f}" if v >= 100 else sig3(v)


def fmt_factor(f: Optional[float]) -> str:
    if f is None:
        return "—"
    if 0.95 <= f <= 1.05:
        return "≈ same"
    return f"**{sig3(f)}× faster**" if f > 1 else f"{sig3(1 / f)}× slower"


# ------------------------------------------------------------------------------- figures
def headline_series(versions, label_crit) -> Dict[str, List[Tuple[int, float, bool]]]:
    """device -> [(version index, value, higher_is_better)] for one headline metric."""
    _label, suite, phase, crit = label_crit
    out: Dict[str, List[Tuple[int, float, bool]]] = {}
    for i, (_v, payload) in enumerate(versions):
        for rec in payload.get("records", []):
            if rec["suite"] != suite or rec["phase"] != phase:
                continue
            if any(rec.get(k, (rec.get("extra") or {}).get(k)) != val for k, val in crit.items()):
                continue
            dev = device_kind(rec)
            if any(j == i for j, _, _ in out.get(dev, [])):
                continue
            value, _unit, higher = measure(rec)
            if value:
                out.setdefault(dev, []).append((i, value, higher))
    return out


def spread_labels(ax, items, theme, min_gap: float = 0.065) -> None:
    """Label line ends at the right, pushed apart so no two overlap.

    ``min_gap`` is a fraction of the axis height (the y axis is logarithmic).
    """
    lo, hi = ax.get_ylim()
    min_gap *= math.log10(hi / lo)
    items = sorted(items, key=lambda it: it[1])
    ys = [math.log10(it[1]) for it in items]
    for i in range(1, len(ys)):            # push up past the label below
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = (ys[-1] - math.log10(items[-1][1])) / 2
    ys = [y - shift for y in ys]           # centre the stack on the data
    for (x, y, text, color), ly in zip(items, ys):
        ax.plot([x], [y], marker="o", markersize=6, color=color,
                markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=5, clip_on=False)
        ax.annotate(text, xy=(x, y), xytext=(x + 0.25, 10 ** ly), textcoords="data",
                    color=theme["ink2"], fontsize=8.5, va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=theme["ink3"], linewidth=0.6,
                                    shrinkA=0, shrinkB=4))


def fig_history(versions, theme):
    """Speed of each headline metric relative to the oldest version that measured it."""
    if len(versions) < 2:
        return None
    names = [v for v, _ in versions]
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 7.6), sharex=True)
    for ax, dev in zip(axes, ("CPU", "GPU")):
        items, lo, hi = [], 1.0, 1.0
        for slot, hl in enumerate(HEADLINES):
            pts = headline_series(versions, hl).get(dev, [])
            if len(pts) < 2:
                continue
            base = pts[0]
            xs = [i for i, _, _ in pts]
            ys = [speed_factor(v, base[1], h) for _, v, h in pts]
            lo, hi = min(lo, *ys), max(hi, *ys)
            color = theme["series"][slot % len(theme["series"])]
            ax.plot(xs, ys, color=color, marker="o", markeredgecolor=theme["surface"],
                    markeredgewidth=1.2)
            items.append((xs[-1], ys[-1], f"{hl[0]} — {sig3(ys[-1])}×", color))
        ax.axhline(1.0, color=theme["ink3"], linewidth=1.0, zorder=1)
        tidy(ax, ylog=True, yfmt=False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{sig3(v)}×"))
        ax.set_ylim(lo / 1.5, hi * 1.5)
        ax.set_xlim(-0.3, len(names) - 1 + 0.3)
        ax.set_title(dev, loc="left", color=theme["ink"])
        ax.set_ylabel("speed vs oldest version")
        if items:
            spread_labels(ax, items, theme)
    axes[-1].set_xticks(range(len(names)))
    axes[-1].set_xticklabels(names)
    axes[-1].set_xlabel("torchtsetlin version")
    fig.suptitle("Speed by library version (same benchmark, higher is faster)",
                 x=0.012, ha="left", fontsize=13, color=theme["ink"], weight="medium")
    fig.tight_layout(rect=(0, 0, 0.72, 0.95))
    return fig


# -------------------------------------------------------------------------------- tables
def env_rows(versions) -> str:
    rows = []
    for v, payload in versions:
        env = payload.get("environment", {})
        rows.append([v, env.get("timestamp", "?")[:10],
                     f"{env.get('torch', '?')}" + (f" / CUDA {env['cuda']}"
                                                   if env.get("cuda") else ""),
                     env.get("cpu", "?"), (env.get("gpus") or ["—"])[0],
                     str(env.get("torch_threads", "?")), suites_measured(payload)])
    return md_table(["version", "measured", "PyTorch", "CPU", "GPU (cuda:0)", "threads",
                     "suites measured"], rows)


def suites_measured(payload: Dict) -> str:
    got = list(dict.fromkeys(r["suite"] for r in payload.get("records", [])))
    every = list(dict.fromkeys(slug for slug, _h, _f in TABLES))
    if all(s in got for s in every):
        return "all"
    if len(got) <= len(every) // 2:
        return ", ".join(got)
    return "all but " + ", ".join(s for s in every if s not in got)


def same_machine(versions) -> bool:
    sig = {(p["environment"].get("cpu"), (p["environment"].get("gpus") or [None])[0])
           for _v, p in versions}
    return len(sig) == 1


def headline_table(versions) -> str:
    names = [v for v, _ in versions]
    rows = []
    for hl in HEADLINES:
        label, suite, phase, _crit = hl
        series = headline_series(versions, hl)
        for dev in ("CPU", "GPU"):
            pts = {i: (val, h) for i, val, h in series.get(dev, [])}
            if not pts:
                continue
            unit = "s/epoch" if phase == "epoch" else "ex/s"
            cells = [fmt_value(pts[i][0], unit) if i in pts else "—" for i in range(len(names))]
            last = len(names) - 1
            prev = previous(pts, last)
            vs_prev = (speed_factor(pts[last][0], pts[prev][0], pts[last][1])
                       if last in pts and prev is not None else None)
            first = min(pts)
            vs_first = (speed_factor(pts[last][0], pts[first][0], pts[last][1])
                        if last in pts and first != last else None)
            rows.append([label, dev, unit] + cells + [fmt_factor(vs_prev), fmt_factor(vs_first)])
    return md_table(["metric", "device", "unit"] + names +
                    [f"{names[-1]} vs previous", f"{names[-1]} vs oldest"], rows)


def row_label(rec: Dict, varying: Sequence[str]) -> str:
    extra = rec.get("extra") or {}
    bits = [str(rec.get("model"))] if "model" in varying else []
    for f in varying:
        if f == "model":
            continue
        v = rec.get(f, extra.get(f))
        if v is None:
            continue
        if isinstance(v, bool):
            if f == "data_resident_on_device":
                bits.append("on device" if v else "host → device")
            continue
        if f in ("stage", "scale", "feedback_mode"):
            continue  # already in the model name
        bits.append(f"{FIELD_NAMES.get(f, f.replace('_', ' '))} {v}")
    if "stage" in varying and extra.get("stage"):
        bits = [extra["stage"]] + [b for b in bits if b != str(rec.get("model"))]
    return ", ".join(bits) if bits else str(rec.get("model"))


def suite_tables(versions, suite: str) -> List[Tuple[str, str]]:
    """[(phase title, markdown table)] comparing every configuration of one suite."""
    names = [v for v, _ in versions]
    indexed = [index(p) for _v, p in versions]
    keys: List[Tuple] = []
    for idx in reversed(indexed):  # newest version's order first, older-only rows after
        keys += [k for k in idx if k[0] == suite and k not in keys]
    if not keys:
        return []
    recs = [idx[k] for k in keys for idx in indexed if k in idx]
    # label a row by the model plus the settings that vary *within* that model
    by_model: Dict[str, List[Dict]] = {}
    for r in recs:
        by_model.setdefault(str(r.get("model")), []).append(r)
    varying = ["model"] if len(by_model) > 1 else []
    varying += [f for f in CONFIG_FIELDS[1:] + CONFIG_EXTRA
                if any(len({json.dumps(r.get(f, (r.get("extra") or {}).get(f))) for r in rs}) > 1
                       for rs in by_model.values())]
    out = []
    phase_titles = {"train": "training", "infer": "inference", "stage": "update stages",
                    "update_total": "whole update", "epoch": "end to end"}
    for phase in dict.fromkeys(k[1] for k in keys):
        rows = []
        unit = "ex/s"
        for dev in ("CPU", "GPU"):
            for k in keys:
                if k[1] != phase or k[2] != dev:
                    continue
                cells, pts = [], {}
                for i, idx in enumerate(indexed):
                    rec = idx.get(k)
                    if rec is None:
                        cells.append("—")
                        continue
                    val, unit, higher = measure(rec)
                    pts[i] = (val, higher)
                    cells.append(fmt_value(val, unit))
                some = next(idx[k] for idx in indexed if k in idx)
                last = len(names) - 1
                prev = previous(pts, last)
                vs_prev = (speed_factor(pts[last][0], pts[prev][0], pts[last][1])
                           if last in pts and prev is not None else None)
                rows.append([row_label(some, varying), dev] + cells + [fmt_factor(vs_prev)])
        title = f"{phase_titles.get(phase, phase)} ({unit})"
        out.append((title, md_table(["configuration", "device"] + names +
                                    [f"{names[-1]} vs previous"], rows)))
    # end-to-end MNIST also has an accuracy to compare
    if suite == "mnist":
        rows = []
        for k in keys:
            cells = []
            for idx in indexed:
                rec = idx.get(k)
                acc = (rec or {}).get("extra", {}).get("val_accuracy")
                cells.append(f"{acc:.2%}" if acc is not None else "—")
            some = next(idx[k] for idx in indexed if k in idx)
            rows.append([row_label(some, varying), k[2]] + cells)
        out.append(("test accuracy", md_table(["configuration", "device"] + names, rows)))
    return out


# ---------------------------------------------------------------------------------- page
def write_page(versions, path: str, fig_dir: str, root: str, has_fig: bool) -> None:
    names = [v for v, _ in versions]
    rel = os.path.relpath(fig_dir, os.path.dirname(path) or ".")
    out = [
        "# Benchmark history",
        "",
        f"<!-- generated by benchmarks/plot_history.py from {root}/*/cpu_vs_gpu.json — do not "
        f"edit by hand; re-run ./benchmarks/run_benchmarks.sh -->",
        "",
        f"The same benchmark suites, run on every torchtsetlin release that has results "
        f"({', '.join(names)}), side by side. [CPU vs GPU](cpu-vs-gpu.md) explains each suite "
        f"in detail for the newest version, {names[-1]}; this page shows how the numbers moved "
        f"between versions.",
        "",
    ]
    if same_machine(versions):
        out += ["Every version was measured on the same machine, so the columns are directly "
                "comparable (run-to-run noise is a few percent):", ""]
    else:
        out += ["!!! warning \"Different machines\"", "",
                "    Not every version was measured on the same hardware. Compare columns "
                "only where the CPU and GPU below match.", ""]
    out += [env_rows(versions), ""]
    out += ["## Headline numbers", ""]
    if has_fig:
        out += ["![Speed by version](" + rel + "/version-history.png#only-light)",
                "![Speed by version](" + rel + "/version-history_dark.png#only-dark)", ""]
    out += ["Throughput in examples per second (MNIST: seconds per epoch, lower is better). "
            "The last two columns give the speed of the newest version relative to the "
            "previous version that measured the configuration and to the oldest one.", "",
            headline_table(versions), ""]
    out += ["## Every configuration", "",
            "One table per suite and phase, with a column per version. A dash means that "
            "version did not measure the configuration (the suite did not exist yet, or that "
            "release re-ran only some suites — see *suites measured* above).", ""]
    suites = list(dict.fromkeys(slug for slug, _h, _f in TABLES))
    for suite in suites:
        tables = suite_tables(versions, suite)
        if not tables:
            continue
        out += [f"### {SUITE_HEADINGS.get(suite, suite)}", ""]
        for title, md in tables:
            out += [f"??? abstract \"{title}\"", ""]
            out += ["    " + line if line else "" for line in md.splitlines()]
            out += [""]
    out += [
        "## Adding a version",
        "",
        "```bash",
        "pip install -e .                    # the version to measure",
        "./benchmarks/run_benchmarks.sh      # writes benchmarks/results/versions/<version>/",
        "```",
        "",
        "Each run writes to the folder of the installed library version, so the results of "
        "older versions are never overwritten. The driver then regenerates "
        "[CPU vs GPU](cpu-vs-gpu.md) for the newest version and this page for all of them; "
        "`python benchmarks/plot_history.py` rebuilds this page alone.",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--versions", default="benchmarks/results/versions",
                    help="directory with one <version>/cpu_vs_gpu.json per version")
    ap.add_argument("--out", default="docs/assets/benchmarks", help="figure directory")
    ap.add_argument("--page", default="docs/benchmark-history.md")
    args = ap.parse_args()

    versions = load_versions(args.versions)
    if not versions:
        raise SystemExit(f"no results under {args.versions}/*/cpu_vs_gpu.json")
    os.makedirs(args.out, exist_ok=True)
    has_fig = True
    for mode, theme in THEMES.items():
        style(theme)
        fig = fig_history(versions, theme)
        if fig is None:
            has_fig = False
            break
        suffix = "" if mode == "light" else "_dark"
        path = os.path.join(args.out, f"version-history{suffix}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=theme["surface"])
        plt.close(fig)
        print(f"wrote {path}")
    os.makedirs(os.path.dirname(args.page) or ".", exist_ok=True)
    write_page(versions, args.page, args.out, args.versions, has_fig)


if __name__ == "__main__":
    main()
