"""Turn the P0 raw measurements into the two artefacts the rest of the programme reads.

* ``results/calibration_throughput.json`` -- every calibration cell from every GPU, merged,
  plus the derived scheduling answers (largest arm that fits the 10 GB card, projected cost
  of the literature-scale configurations).
* ``results/calibration_seednoise.json`` -- the seed-noise band: mean +- sd over the 5 seeds
  of ``ctm-small``, on validation and on test, under the full protocol.

Also prints the markdown tables that go into ``AUDIT.md`` and the ROUND-1 position paper.
Constraint C4: nothing downstream may hand-type any of these numbers.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as stats
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")


def load_cells() -> List[dict]:
    cells: List[dict] = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "calib_*.json"))):
        with open(p) as f:
            for c in json.load(f):
                c["source"] = os.path.basename(p)
                cells.append(c)
    return cells


def short_gpu(name: str) -> str:
    return name.replace("NVIDIA GeForce ", "")


def throughput_table(cells: List[dict]) -> str:
    rows = ["| GPU | clauses | patch | bool | automata | s/epoch (45k) | 30 ep | peak alloc MB "
            "| peak reserved MB | infer img/s |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(cells, key=lambda c: (short_gpu(c["gpu"]), c["booleanization"],
                                          c["n_clauses"], c["patch"])):
        g = short_gpu(c["gpu"])
        if c.get("oom"):
            rows.append(f"| {g} | {c['n_clauses']} | {c['patch']}x{c['patch']} | "
                        f"{c['booleanization']} | - | **OOM** | - | - | - | - |")
            continue
        sec = c["epoch_s_full_scale"]
        star = "*" if c.get("extrapolated") else ""
        rows.append(
            f"| {g} | {c['n_clauses']} | {c['patch']}x{c['patch']} | {c['booleanization']} "
            f"| {c['n_automata']/1e6:.1f}M | {sec:.1f}{star} | {sec*30/60:.0f} min "
            f"| {c['peak_total_mb']:.0f} | {c['peak_reserved_mb']:.0f} "
            f"| {c['infer_img_per_s']:.0f} |")
    rows.append("")
    rows.append("`*` = extrapolated from a fixed number of mini-batches (probe mode), not a "
                "full measured epoch.")
    return "\n".join(rows)


def fits(cells: List[dict], limit_mb: float, headroom: float = 0.90) -> Dict[str, object]:
    """Largest measured configuration that fits under ``limit_mb`` with headroom, by automata."""
    ok = [c for c in cells if not c.get("oom")
          and c["peak_reserved_mb"] <= limit_mb * headroom]
    if not ok:
        return {}
    best = max(ok, key=lambda c: c["n_automata"])
    return {"n_clauses": best["n_clauses"], "patch": best["patch"],
            "booleanization": best["booleanization"], "n_automata": best["n_automata"],
            "peak_reserved_mb": best["peak_reserved_mb"],
            "epoch_s_full_scale": best["epoch_s_full_scale"], "gpu": best["gpu"],
            "limit_mb": limit_mb, "headroom": headroom}


def seed_band(arm: str = "ctm-small") -> Dict[str, object]:
    recs = []
    for p in sorted(glob.glob(os.path.join(RESULTS, f"{arm}_seed*.json"))):
        with open(p) as f:
            recs.append(json.load(f))
    if len(recs) < 2:
        return {"arm": arm, "n_seeds": len(recs), "note": "not enough seeds yet"}
    test = [r["test_acc"] for r in recs]
    val = [r["val_acc"] for r in recs]
    sel = [r["selected_epoch"] for r in recs]
    hashes = {r["data"]["split_hash"] for r in recs}
    out = {
        "arm": arm, "n_seeds": len(recs), "seeds": [r["seed"] for r in recs],
        "epochs": recs[0]["hp"]["epochs"], "hp": recs[0]["hp"],
        "data": recs[0]["data"], "split_hashes_agree": len(hashes) == 1,
        "split_hash": sorted(hashes)[0],
        "test_acc": {"values": test, "mean": stats.mean(test), "sd": stats.stdev(test),
                     "min": min(test), "max": max(test), "range": max(test) - min(test)},
        "val_acc": {"values": val, "mean": stats.mean(val), "sd": stats.stdev(val),
                    "min": min(val), "max": max(val), "range": max(val) - min(val)},
        "selected_epoch": {"values": sel, "mean": stats.mean(sel)},
        "wall_s": {"mean": stats.mean([r["wall_s"] for r in recs])},
        # The operational band. A difference between two 3-seed means smaller than
        # 2 sd_seed / sqrt(3) is inside the noise of this harness and is not a difference.
        "band_1sd": stats.stdev(test),
        "band_2sd": 2 * stats.stdev(test),
        "min_detectable_diff_3seeds": 2 * stats.stdev(test) * (2 / 3) ** 0.5,
        "git_sha": recs[0]["git_sha"], "env": recs[0]["env"],
        "chunk_size_at_batch": recs[0]["hp"].get("chunk_size_at_batch"),
        "max_chunk_elements": recs[0]["hp"].get("max_chunk_elements"),
        "epoch_s": {"mean": stats.mean([e["epoch_s"] for r in recs for e in r["curve"]])},
    }
    # How much of the band is epoch-selection noise rather than seed-to-seed learning
    # variability? Test at a fixed epoch is deliberately not recorded (that is the leakage
    # guard), so the decomposition is done on validation: the spread of val at the LAST epoch
    # is selection-free, and comparing it to the spread of val at the SELECTED epoch shows
    # what selection adds. Also: how flat is the curve the selector is choosing on?
    last = [r["curve"][-1]["val_acc"] for r in recs]
    flat = []
    for r in recs:
        vv = [e["val_acc"] for e in r["curve"]]
        b = max(vv)
        near = [i + 1 for i, x in enumerate(vv) if b - x <= 0.005]
        flat.append({"seed": r["seed"], "n_epochs_within_0.5pp_of_best": len(near),
                     "first_such_epoch": near[0], "selected_epoch": r["selected_epoch"],
                     "n_epochs": len(vv)})
    out["selection_noise"] = {
        "val_at_last_epoch": {"values": last, "mean": stats.mean(last),
                              "sd": stats.stdev(last)},
        "val_at_selected_epoch_sd": stats.stdev(val),
        "plateau": flat,
        "note": "sd(val at a fixed epoch) is selection-free; the difference from "
                "sd(val at the selected epoch) is what argmax selection adds on a flat curve",
    }
    # M2 across seeds: the statistic that decides the P5 counting-pool branch.
    mcs = [r["diagnostics"].get("match_count") for r in recs]
    if all(mcs):
        def agg(k):
            v = [m[k] for m in mcs]
            return {"mean": stats.mean(v), "sd": stats.stdev(v) if len(v) > 1 else 0.0,
                    "values": v}
        out["match_count"] = {
            "definition": mcs[0]["definition"], "split": mcs[0]["split"],
            "n_images": mcs[0]["n_images"], "n_patches": mcs[0]["n_patches"],
            **{k: agg(k) for k in ("fire_frac", "median_given_firing", "mean_given_firing",
                                   "p95_given_firing", "frac_ge5_given_firing",
                                   "frac_eq1_given_firing", "mean_all_pairs")},
        }
    # The other per-run diagnostics, across seeds -- the density signature of the family.
    out["diagnostics_across_seeds"] = {
        "clause_len_median": [r["diagnostics"]["clause_len"]["median"] for r in recs],
        "negation_fraction": [r["diagnostics"]["negation_fraction"] for r in recs],
        "firing_rate_median": [r["diagnostics"]["firing_rate"]["median"] for r in recs],
        "dead_frac": [r["diagnostics"]["firing_rate"]["dead_frac"] for r in recs],
        "clauses_used_frac": [r["diagnostics"]["clauses_used_frac"] for r in recs],
    }
    return out


def chunk_table(cells: List[dict]) -> str:
    """Only the cells that were swept over max_chunk_elements (LG-004)."""
    keys = {(c["n_clauses"], c["patch"], c["booleanization"]) for c in cells}
    swept = [k for k in keys
             if len({c.get("max_chunk_elements", 2 ** 27) for c in cells
                     if (c["n_clauses"], c["patch"], c["booleanization"]) == k}) > 1]
    if not swept:
        return ""
    rows = ["| clauses | patch | bool | max_chunk_elements | chunk/50 | s/epoch | speedup | "
            "peak alloc MB | peak reserved MB |", "|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(swept):
        grp = sorted([c for c in cells
                      if (c["n_clauses"], c["patch"], c["booleanization"]) == k
                      and not c.get("oom")], key=lambda c: c.get("max_chunk_elements", 2 ** 27))
        if not grp:
            continue
        base = grp[0]["epoch_s_full_scale"]
        for c in grp:
            e = int(c.get("max_chunk_elements", 2 ** 27)).bit_length() - 1
            rows.append(f"| {c['n_clauses']} | {c['patch']}x{c['patch']} | "
                        f"{c['booleanization']} | 2**{e} | {c.get('chunk_size', '-')} | "
                        f"{c['epoch_s_full_scale']:.1f} | "
                        f"{base / c['epoch_s_full_scale']:.2f}x | "
                        f"{c['peak_total_mb']:.0f} | {c['peak_reserved_mb']:.0f} |")
    return "\n".join(rows)


def main() -> None:
    cells = load_cells()
    if cells:
        merged = {"cells": cells,
                  "fits_10gb": fits(cells, 10240.0), "fits_24gb": fits(cells, 24576.0)}
        with open(os.path.join(RESULTS, "calibration_throughput.json"), "w") as f:
            json.dump(merged, f, indent=1)
        tbl = throughput_table(cells)
        ctbl = chunk_table(cells)
        with open(os.path.join(RESULTS, "tables_p0.md"), "w") as f:
            f.write("### Throughput and memory, all measured cells\n\n" + tbl
                    + "\n\n### Chunk-budget sweep (LG-004)\n\n" + ctbl + "\n")
        print(tbl)
        print("\n" + ctbl)
        print("\nlargest measured config under 10 GB:", json.dumps(merged["fits_10gb"]))
        print("largest measured config under 24 GB:", json.dumps(merged["fits_24gb"]))
    band = seed_band()
    with open(os.path.join(RESULTS, "calibration_seednoise.json"), "w") as f:
        json.dump(band, f, indent=1)
    if "test_acc" in band:
        t, v = band["test_acc"], band["val_acc"]
        print(f"\nseed band ({band['arm']}, {band['n_seeds']} seeds, {band['epochs']} epochs)")
        print(f"  test {t['mean']*100:.2f} +- {t['sd']*100:.2f} pp   "
              f"(range {t['min']*100:.2f}-{t['max']*100:.2f}, spread {t['range']*100:.2f} pp)")
        print(f"  val  {v['mean']*100:.2f} +- {v['sd']*100:.2f} pp   "
              f"selected epochs {band['selected_epoch']['values']}")
        print(f"  1 sd = {band['band_1sd']*100:.2f} pp;  2 sd = {band['band_2sd']*100:.2f} pp;  "
              f"min detectable diff between two 3-seed means = "
              f"{band['min_detectable_diff_3seeds']*100:.2f} pp")
        if "match_count" in band:
            m = band["match_count"]
            print(f"  M2 |M_j| ({m['split']}, {m['n_images']} imgs, P={m['n_patches']}): "
                  f"fire_frac {m['fire_frac']['mean']:.4f}, "
                  f"median|firing {m['median_given_firing']['mean']:.1f}, "
                  f"mean|firing {m['mean_given_firing']['mean']:.1f}, "
                  f"P(>=5|firing) {m['frac_ge5_given_firing']['mean']:.3f} "
                  f"+- {m['frac_ge5_given_firing']['sd']:.3f}, "
                  f"P(=1|firing) {m['frac_eq1_given_firing']['mean']:.3f}, "
                  f"mean_all {m['mean_all_pairs']['mean']:.2f}")
    else:
        print("\nseed band:", json.dumps(band, indent=1))


if __name__ == "__main__":
    main()
