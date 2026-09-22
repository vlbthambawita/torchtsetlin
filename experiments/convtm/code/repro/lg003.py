"""LG-003 reproduction: the clause-size budget does not bind for convolutional models
under batched feedback.

Trains the same convolutional TM with ``max_included_literals=8`` at batch 50, batch 5 and
in sequential mode, on a small CIFAR-10 subset, and prints the resulting median clause size.
The budget is supposed to cap it at 8.

    CUDA_VISIBLE_DEVICES=1 python code/repro/lg003.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as data_mod  # noqa: E402
from diagnostics import clause_composition  # noqa: E402

import torchtsetlin as tt  # noqa: E402

BUDGETS = (8, 16, 32, 64)   # SOTA recipes (TM Composites, GraphTM) use 32


def run(batch_size: int, sequential: bool, xtr, ytr, budget: int, epochs: int = 2) -> dict:
    tt.seed_everything(0)
    m = tt.ConvTsetlinMachine(10, 20, 16.0, 10.0, patch_size=4, stride=1,
                              position_encoding=True, input_shape=tuple(xtr.shape[1:]),
                              max_included_literals=budget).to(xtr.device)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(xtr.shape[0], device=xtr.device)
        for i in range(0, xtr.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            m.update(xtr[idx], ytr[idx], sequential=sequential or None)
    return clause_composition(m)


if __name__ == "__main__":
    import json
    ds = data_mod.load_boolean("therm4", device="cuda", splits=("train",), subset=2000)
    xtr, ytr = ds["xtr"], ds["ytr"]
    print("Is the clause-size overshoot PROPORTIONAL or ABSOLUTE? The SOTA recipes use "
          "budget 32;\nif the overshoot is proportional the budget is unusable at every "
          "scale, if absolute it is\nimmaterial at 32.\n")
    print(f"  {'budget':>6} {'feedback':>11} {'median':>7} {'mean':>7} {'p95':>7} "
          f"{'over abs':>9} {'over %':>7}  verdict")
    out = []
    for budget in BUDGETS:
        for label, bs, seq in (("batch 50", 50, False), ("batch 5", 5, False),
                               ("sequential", 1, True)):
            if seq and budget not in (8, 32):
                continue          # the sequential control is only needed at two points
            c = run(bs, seq, xtr, ytr, budget)["clause_len"]
            over, pct = c["median"] - budget, 100.0 * (c["median"] - budget) / budget
            print(f"  {budget:>6d} {label:>11s} {c['median']:>7.1f} {c['mean']:>7.1f} "
                  f"{c['p95']:>7.1f} {over:>+9.1f} {pct:>+6.0f}%  "
                  f"{'HELD' if c['median'] <= budget else 'EXCEEDED'}")
            out.append({"budget": budget, "feedback": label, **c})
    with open(os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), os.pardir, "results", "lg003_budget_sweep.json"),
            "w") as f:
        json.dump(out, f, indent=1)
