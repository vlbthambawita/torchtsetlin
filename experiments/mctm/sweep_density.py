"""E0: layer-1 density diagnostic -- run first, it sizes everything else.

The MCTM premise is that layer-2 clauses can form useful conjunctions over layer-1 clause
maps. A conjunction of j bits each firing with probability p is satisfied with probability
~p^j, so if the maps are sparse the second layer cannot learn anything but negated
(near-constant) literals. This sweep measures, across layer-1 settings:

  * median per-channel firing rate of the clause maps at OR-pool 1, 2 and 4,
  * the fraction of dead channels,
  * the median clause size, and
  * layer 1's own test accuracy (what the extra density costs).

It also documents an interaction found while building this: ``max_included_literals`` gates
only Type Ia feedback, while Type II pushes excluded literals towards inclusion ungated.
Under ``feedback_mode="batch"`` with 841 patches per image, a single commit aggregates
enough Type II events to push many literals across the include boundary at once, so the
clause-size budget stops binding as batch size grows.
"""
from __future__ import annotations

import argparse, json, os

import torch

from cifar import load_cifar10
from mctm import conv_clause_maps, firing_stats, or_pool
from run import RESULTS, accuracy, make_conv, now, P1, S1, C1, T1

# (label, s, max_included_literals, batch_size, feedback_mode, subset_scale)
# Sequential feedback commits once per example, which on 841 patches is ~1000x slower than
# batched; those rows are diagnostics of the clause-size budget, not accuracy numbers, so
# they run on a fraction of the data (subset_scale) and their accuracy is not comparable.
CONFIGS = [
    ("s=1.5",              1.5,  None, 50, "batch",      1.0),
    ("s=2",                2.0,  None, 50, "batch",      1.0),
    ("s=3",                3.0,  None, 50, "batch",      1.0),
    ("s=5",                5.0,  None, 50, "batch",      1.0),
    ("s=10",              10.0,  None, 50, "batch",      1.0),
    ("s=10, b=8, bs=50",  10.0,     8, 50, "batch",      1.0),
    ("s=10, b=8, bs=5",   10.0,     8,  5, "batch",      1.0),
    ("s=10, b=8, seq",    10.0,     8,  5, "sequential", 0.1),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--subset", type=int, default=20000)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", type=int, default=4000, help="images used for firing stats")
    a = ap.parse_args()

    xtr, ytr, xte, yte = load_cifar10(n_bits=a.n_bits, device="cuda", max_train=a.subset)
    in_shape = tuple(xtr.shape[1:])
    out = {"epochs": a.epochs, "subset": a.subset, "n_bits": a.n_bits,
           "in_shape": list(in_shape), "C1": C1, "P1": P1, "T1": T1, "rows": []}

    for label, s, max_inc, bs, mode, scale in CONFIGS:
        t0 = now()
        n_use = max(bs, int(xtr.shape[0] * scale))
        m = make_conv(C1, T1, s, P1, S1, in_shape, max_inc=max_inc, seed=a.seed)
        m.feedback_mode = mode
        m.train()
        for _ in range(a.epochs):
            perm = torch.randperm(n_use, device=xtr.device)
            for i in range(0, n_use, bs):
                idx = perm[i : i + bs]
                m.update(xtr[idx], ytr[idx])
        m.eval()
        acc = accuracy(m, xte, yte)
        raw = conv_clause_maps(m, xtr[: a.probe])
        fire = {}
        for k in (1, 2, 4):
            st = firing_stats(or_pool(raw, k))
            fire[f"pool{k}"] = {kk: vv for kk, vv in st.items() if kk != "per_channel"}
            if k == 2:
                per_channel = st["per_channel"]
        row = {"label": label, "s": s, "max_included_literals": max_inc,
               "batch_size": bs, "feedback_mode": mode, "n_train_used": n_use,
               "comparable_acc": scale == 1.0, "l1_test_acc": acc,
               "include_count": {"min": int(m.include_count.min()),
                                 "median": int(m.include_count.median()),
                                 "max": int(m.include_count.max()),
                                 "mean": float(m.include_count.float().mean())},
               "firing": fire, "per_channel_pool2": per_channel, "wall_s": now() - t0}
        out["rows"].append(row)
        print(f"{label:<18} acc {acc:.4f}  includes ~{row['include_count']['median']:>4}"
              f"  firing p1 {fire['pool1']['median']:.5f}"
              f"  p2 {fire['pool2']['median']:.5f}"
              f"  p4 {fire['pool4']['median']:.5f}"
              f"  dead(p2) {fire['pool2']['frac_dead']:.2f}  ({row['wall_s']:.0f}s)", flush=True)
        del m, raw
        torch.cuda.empty_cache()

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "density_sweep.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
