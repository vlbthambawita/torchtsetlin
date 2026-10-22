"""Measure what tmu's 3x3 thermometer specialist actually learns: included literals per clause
and clause weights after each epoch, plus test accuracy. Our measurement script (not the
authors'), but the model and encoding are exactly CIFAR10ColorThermometerScoring.py's defaults.

Run in the tmu environment (see run_tmu_reference.py for the env it needs):
    python code/tmu_clause_stats.py --epochs 5 --out results/tmu_C3_clausestats.json
"""
import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "reference_code", "shim"))
from keras.datasets import cifar10  # noqa: E402  (the shim: same arrays as keras)
from tmu.models.classification.vanilla_classifier import TMClassifier  # noqa: E402


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    q = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return {"mean": float(a.mean()), "q0": q[0], "q5": q[1], "q25": q[2], "q50": q[3], "q75": q[4],
            "q95": q[5], "q100": q[6], "frac_gt_32": float((a > 32).mean()), "frac_eq_0": float((a == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patch_size", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = 8
    (xtr0, ytr), (xte0, yte) = cifar10.load_data()

    def therm(x):
        b = np.empty((*x.shape, res), dtype=np.uint8)
        for z in range(res):
            b[..., z] = x >= (z + 1) * 255 / (res + 1)
        return b.reshape(x.shape[0], x.shape[1], x.shape[2], 3 * res).astype(np.uint32)

    xtr, xte = therm(xtr0), therm(xte0)
    ytr, yte = ytr.reshape(-1).astype(np.uint32), yte.reshape(-1).astype(np.uint32)
    tm = TMClassifier(number_of_clauses=2000, T=1500, s=2.5, max_included_literals=32, platform="GPU",
                      weighted_clauses=True, patch_dim=(args.patch_size, args.patch_size))
    out = {"model": "tmu TMClassifier, CIFAR10ColorThermometerScoring.py defaults", "patch": args.patch_size,
           "epochs": []}
    for e in range(1, args.epochs + 1):
        t0 = time.time()
        tm.fit(xtr, ytr)
        acc = float((tm.predict(xte) == yte).mean() * 100)
        sizes, wpos, wneg = [], [], []
        for c in range(10):
            cb = tm.clause_banks[c]
            sizes += [int(cb.number_of_include_actions(j)) for j in range(cb.number_of_clauses)]
            w = tm.weight_banks[c].get_weights()
            wpos += list(w[w > 0])
            wneg += list(-w[w < 0])
        rec = {"epoch": e, "test_acc": acc, "seconds": time.time() - t0, "included_literals": stats(sizes),
               "weight_magnitude_positive": stats(wpos), "weight_magnitude_negative": stats(wneg)}
        out["epochs"].append(rec)
        print(json.dumps({"epoch": e, "acc": round(acc, 2), "size_q50": rec["included_literals"]["q50"],
                          "size_mean": round(rec["included_literals"]["mean"], 1),
                          "frac_gt_32": round(rec["included_literals"]["frac_gt_32"], 3)}), flush=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
