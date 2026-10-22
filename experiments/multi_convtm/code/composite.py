"""TM Composite (Granmo 2023, Eqs. 7-8) over saved specialist class sums.

alpha_t = max_{d,i} c^i_{t,d} - min_{d,i} c^i_{t,d}  over the evaluated set (here: the 10 000 test
images, exactly as the reference CIFAR10Team.py does), and y_hat_d = argmax_i sum_t c^i_{t,d} / alpha_t.

Writes results/reference_check.json (the authors' released class sums) and results/composite.json
(our specialists, every seed, every epoch they share).

Usage: python code/composite.py
"""

import glob
import json
import pickle
from pathlib import Path

import numpy as np
from common import CACHE, CIFAR10, REFERENCE, RESULTS, new_record, write_json

ROWS = [  # Table 2 of the paper, in its order; values are the CIFAR-10 column
    ("10x10 Thresholding", ["T10"], 57.0),
    ("3x3 Color Thermometers", ["C3"], 62.1),
    ("4x4 Color Thermometers", ["C4"], 62.7),
    ("Histogram of Gradients", ["HoG"], 63.5),
    ("10x10T + HoG", ["T10", "HoG"], 68.0),
    ("10x10T + 3x3C", ["T10", "C3"], 68.6),
    ("10x10T + 4x4C", ["T10", "C4"], 68.7),
    ("10x10T + 3x3C + 4x4C", ["T10", "C3", "C4"], 71.7),
    ("10x10T + 3x3C + 4x4C + HoG", ["T10", "C3", "C4", "HoG"], 75.1),
]
REF_FILES = {"T10": "CIFAR10AdaptiveThresholding_{e}_2000_500_10.0_10_32_1.txt",
             "C3": "CIFAR10ColorThermometers_{e}_2000_1500_2.5_3_8_32_1.txt",
             "C4": "CIFAR10ColorThermometers_{e}_2000_1500_2.5_4_8_32_1.txt",
             "HoG": "CIFAR10HistogramOfGradients_{e}_2000_50_10.0_0_32_0.txt"}


def test_labels():
    with open(CIFAR10 / "test_batch", "rb") as fh:
        return np.asarray(pickle.load(fh, encoding="bytes")[b"labels"], dtype=np.int64)


def composite_acc(sums, members, y):
    votes = sum(sums[m] / (sums[m].max() - sums[m].min()) for m in members)
    return float((votes.argmax(1) == y).mean() * 100)


def table(sums, y):
    return {name: composite_acc(sums, members, y) for name, members, _ in ROWS}


def reference_check(y):
    out = {}
    for e in (9, 99):
        sums = {k: np.loadtxt(REFERENCE / f.format(e=e), delimiter=",") for k, f in REF_FILES.items()}
        out[f"epoch{e}"] = table(sums, y)
    rec = new_record("reference_check", source=(REFERENCE / "SOURCE").read_text().strip(),
                     source_commit=(REFERENCE / "SOURCE_COMMIT").read_text().strip(),
                     note="Authors' released test-set class sums; file suffix _99_ is epoch+1 = 99 in "
                          "their scripts. Recomputes Table 2 (CIFAR-10) with Eqs. 7-8.",
                     paper={name: p for name, _, p in ROWS}, accuracy=out,
                     max_abs_diff_epoch99=max(abs(out["epoch99"][n] - p) for n, _, p in ROWS))
    write_json(rec, RESULTS / "reference_check.json")
    return rec


def ours(y, variant=None):
    runs = {}
    for f in sorted(glob.glob(str(RESULTS / "*_seed*.json"))):
        r = json.loads(Path(f).read_text())
        if r.get("kind") == "specialist" and r.get("variant") == variant:
            runs.setdefault(r["seed"], {})[r["spec"]] = r
    per_seed = {}
    for seed, specs in sorted(runs.items()):
        sfx = f"_{variant}" if variant else ""
        dirs = {k: CACHE / "class_sums" / f"{k}_seed{seed}{sfx}" for k in specs}
        epochs = sorted(set.intersection(*[{int(Path(p).stem[1:]) for p in glob.glob(str(d / "e*.npy"))}
                                           for d in dirs.values()])) if dirs else []
        curve = []
        for e in epochs:
            sums = {k: np.load(d / f"e{e:03d}.npy") for k, d in dirs.items()}
            curve.append({"epoch": e, **{name: composite_acc(sums, m, y)
                                         for name, m, _ in ROWS if all(k in sums for k in m)}})
        per_seed[seed] = {"specs": {k: {"status": r["status"], "epochs_done": len(r["curve"])}
                                    for k, r in specs.items()},
                          "curve": curve,
                          "epoch100": next((c for c in curve if c["epoch"] == 100), None),
                          "epoch99": next((c for c in curve if c["epoch"] == 99), None)}
    summary = {}
    for name, members, paper in ROWS:
        vals = [s["epoch100"][name] for s in per_seed.values() if s["epoch100"] and name in s["epoch100"]]
        summary[name] = {"members": members, "paper": paper, "n_seeds": len(vals), "values": vals,
                         "mean": float(np.mean(vals)) if vals else None,
                         "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    rec = new_record("composite", variant=variant, alpha="max-min of the specialist's class sums over the "
                     "10 000 test images (as CIFAR10Team.py)", summary_epoch100=summary, per_seed=per_seed)
    write_json(rec, RESULTS / (f"composite_{variant}.json" if variant else "composite.json"))
    return rec


def tmu(y, run=0):
    """Table 2 from our tmu runs of the authors' scripts (their own per-epoch class-sum files)."""
    pattern = {"T10": "CIFAR10AdaptiveThresholding_{e}_2000_*.txt", "C3": "CIFAR10ColorThermometers_{e}_2000_*_3_8_*.txt",
               "C4": "CIFAR10ColorThermometers_{e}_2000_*_4_8_*.txt", "HoG": "CIFAR10HistogramOfGradients_{e}_2000_*.txt"}
    dirs = {k: CACHE / "tmu_runs" / f"tmu_{k}_run{run}" for k in pattern}

    def load(k, e):
        hits = sorted(dirs[k].glob(pattern[k].format(e=e)))
        return np.loadtxt(hits[0], delimiter=",") if hits else None

    per_epoch = {}
    for e in (9, 99, 100):
        sums = {k: load(k, e) for k in pattern}
        sums = {k: v for k, v in sums.items() if v is not None}
        per_epoch[f"epoch{e}"] = {name: composite_acc(sums, m, y) for name, m, _ in ROWS
                                  if all(k in sums for k in m)}
    rec = new_record("tmu_composite", run=run, source="our tmu runs of the authors' scripts "
                     "(reference_code/), their own per-epoch class-sum files", accuracy=per_epoch,
                     paper={name: p for name, _, p in ROWS})
    write_json(rec, RESULTS / f"tmu_composite_run{run}.json")
    return rec


if __name__ == "__main__":
    y = test_labels()
    ref = reference_check(y)
    print("reference (authors' sums, epoch 99):",
          {k: round(v, 2) for k, v in ref["accuracy"]["epoch99"].items()})
    ours(y, "softbudget")
    t = tmu(y)
    print("tmu (authors' code, our run), epoch 100:", {k: round(v, 2) for k, v in t["accuracy"]["epoch100"].items()})
    c = ours(y)
    for name, s in c["summary_epoch100"].items():
        print(f"{name:30s} paper {s['paper']:5.1f}  ours {s['mean'] if s['mean'] is None else round(s['mean'], 2)}"
              f" (n={s['n_seeds']})")
