"""E6: the three section-7.3 repairs on the two tasks whose ceilings are known in advance.

E2 of the first report established the useful property of these tasks: with a hand-built
oracle layer 1 the stack clears a bound a single layer provably cannot, and with the greedily
trained layer 1 it does not. The gap between those two numbers is exactly what a better
training scheme has to close, and it is measured rather than argued.

That makes this the right place to judge idea C in particular. On CIFAR-10 the credit path is
starved: a layer-2 clause carrying ~1200 literals almost never matches *because of* a positive
channel literal, so there is almost nothing to propagate. Here layer-2 clauses are small and
match often, so the mechanism is actually exercised.

Arms are the E2 arms plus four new layer-1 modes:

  auto     idea A -- layer 1 pretrained as a patch autoencoder (no labels)
  calib    idea B -- greedy layer 1, then calibrated to a target firing rate
  credit   idea C -- layer 1 initialised random, then trained through layer 2
  credit-task      idea C starting from the greedy layer 1 instead
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from ideas import CreditStack, _probe_literals, calibrate_density, pretrain_autoencoder_l1
from mctm import clause_composition, conv_clause_maps, firing_stats, or_pool, randomize_clauses
from run import RESULTS, accuracy, now
from synthetic import (
    C2,
    L2_PATCH,
    POOL,
    T2,
    build_l1,
    build_oracle_l1,
    make_task,
    train,
)

import torchtsetlin as tt


def _fire(l1, x, pool=POOL) -> dict:
    return {k: v for k, v in firing_stats(or_pool(conv_clause_maps(l1, x), pool)).items()
            if k != "per_channel"}


def make_l1(mode, in_shape, xtr, ytr, xte, yte, s, seed, epochs, bs, device, rec, calib):
    """Layer 1 by objective. Everything but ``oracle`` is a real training scheme."""
    if mode.startswith("oracle"):
        n_cl = int(mode[6:]) if len(mode) > 6 else 2
        return build_oracle_l1(in_shape, device, n_clauses=n_cl, seed=seed)
    l1 = build_l1(in_shape, s, seed, device)
    if mode == "random":
        randomize_clauses(l1, n_include=3, seed=seed)
    elif mode == "auto":
        rec["ae"] = pretrain_autoencoder_l1(l1, xtr, n_patches=200_000, epochs=4, s=s,
                                            batch_size=200, seed=seed)
    else:                                   # greedy: layer 1 trained on the task itself
        train(l1, xtr, ytr, xte, yte, epochs, bs, "L1", rec["log_l1"])
        rec["l1_test_acc"] = rec["log_l1"][-1]["test_acc"]
    l1.eval()
    rec["l1_firing_pre"] = _fire(l1, xtr[:4000])
    if calib > 0:
        rec["calibration"] = calibrate_density(l1, xtr, calib, seed=seed, n_patches=40000)
        rec["l1_firing_post"] = _fire(l1, xtr[:4000])
    return l1


def run(task, arm, seed, epochs, n_train, n_test, s, bs, device, l1_mode, calib,
        credit="none", calib_every=0, warmup=0, credit_rate=1.0) -> dict:
    xtr, ytr = make_task(n_train, task, seed=seed, device=device)
    xte, yte = make_task(n_test, task, seed=seed + 500, device=device)
    in_shape = tuple(xtr.shape[1:])
    rec = {"task": task, "arm": arm, "l1_mode": l1_mode, "calib": calib, "credit": credit,
           "seed": seed, "epochs": epochs, "s": s, "warmup": warmup,
           "credit_rate": credit_rate,
           "n_train": n_train, "n_test": n_test,
           "input_shape": list(in_shape), "log": [], "log_l1": []}

    l1 = make_l1(l1_mode, in_shape, xtr, ytr, xte, yte, s, seed, epochs, bs, device, rec, calib)
    rec["l1_clauses"] = int(l1.n_clauses_total)
    maps0 = or_pool(conv_clause_maps(l1, xtr[:64]), POOL)
    k = L2_PATCH[task] or int(maps0.shape[2])
    rec["l2_patch"] = k
    rec["feature_shape"] = list(maps0.shape[1:])
    tt.seed_everything(seed + 7)
    head = tt.ConvCoalescedTsetlinMachine(2, C2, T2, s, patch_size=k, stride=1,
                                          position_encoding=False,
                                          input_shape=tuple(maps0.shape[1:])).to(device)
    if credit == "none":
        htr = or_pool(conv_clause_maps(l1, xtr), POOL)
        hte = or_pool(conv_clause_maps(l1, xte), POOL)
        rec["l1_firing"] = _fire(l1, xtr[:4000])
        train(head, htr, ytr, hte, yte, epochs, bs, f"{task}:{arm}", rec["log"])
    else:
        stack = CreditStack(l1, head, pool=POOL, type_ib=(credit != "ia"),
                            ib_ratio=(1.0 if credit == "bal" else 0.0),
                            credit_rate=credit_rate)
        probe = _probe_literals(l1, xtr, 40000, seed=seed) if calib_every else None
        if warmup:
            htr = or_pool(conv_clause_maps(l1, xtr), POOL)
            hte = or_pool(conv_clause_maps(l1, xte), POOL)
            train(head, htr, ytr, hte, yte, warmup, bs, f"{task}:{arm}:warmup", rec["log"])
            del htr, hte
        stack.train()
        for ep in range(warmup, epochs):
            t0 = now()
            perm = torch.randperm(n_train, device=device)
            nb = 0
            for i in range(0, n_train, bs):
                stack.update(xtr[perm[i : i + bs]], ytr[perm[i : i + bs]])
                nb += 1
                if calib_every and nb % calib_every == 0:
                    calibrate_density(l1, xtr, calib, seed=seed, lits=probe, max_rounds=40)
            stack.eval()
            te = accuracy(stack, xte, yte)
            stack.train()
            fs = _fire(l1, xtr[:2000])
            rec["log"].append({"epoch": ep + 1, "test_acc": te, "epoch_s": now() - t0,
                               "l1_fire_median": fs["median"],
                               "l1_size_median": float(l1.include_count.float().median()),
                               "credit": dict(stack.stats)})
            if (ep + 1) % 5 == 0 or ep == epochs - 1:
                print(f"  [{task}:{arm}] epoch {ep+1}/{epochs} test {te:.4f} "
                      f"L1 size {float(l1.include_count.float().median()):.0f} "
                      f"fire {fs['median']*100:.2f}% events "
                      f"{stack.stats['kept_i']}/{stack.stats['kept_ib']}/{stack.stats['kept_ii']}",
                      flush=True)
        rec["credit_stats"] = dict(stack.stats)
        rec["l1_firing"] = _fire(l1, xtr[:4000])
    rec["composition"] = clause_composition(head)
    rec["composition_l1"] = clause_composition(l1)
    rec["n_clauses_total"] = int(l1.n_clauses_total) + int(head.n_clauses_total)
    rec["final_test_acc"] = rec["log"][-1]["test_acc"]
    rec["best_test_acc"] = max(r["test_acc"] for r in rec["log"])
    return rec


# (name, l1_mode, calib, credit, calib_every, warmup, credit_rate)
ARMS = {
    "auto":           ("auto",   0.0,  "none", 0,  0,  1.0),
    "auto-calib":     ("auto",   0.05, "none", 0,  0,  1.0),
    "calib":          ("task",   0.05, "none", 0,  0,  1.0),
    "random":         ("random", 0.0,  "none", 0,  0,  1.0),
    "credit-ia":      ("random", 0.0,  "ia",   0,  0,  1.0),
    "credit":         ("random", 0.0,  "bal",  0,  0,  1.0),
    "credit-warm":    ("task",   0.0,  "bal",  0,  10, 1.0),
    "credit-warm-p1": ("task",   0.0,  "bal",  0,  10, 0.1),
    "credit-warm-p01":("task",   0.0,  "bal",  0,  10, 0.01),
    "credit-cal":     ("task",   0.05, "bal",  40, 10, 0.1),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["xor", "near"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--s", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out = []
    path = a.out or os.path.join(RESULTS, "synthetic_ideas.json")
    for task in a.tasks:
        for arm in a.arms:
            mode, calib, credit, every, warm, crate = ARMS[arm]
            for seed in a.seeds:
                print(f"=== {task} / {arm} (L1={mode} calib={calib} credit={credit}) "
                      f"/ seed {seed} ===", flush=True)
                rec = run(task, arm, seed, a.epochs, a.n_train, a.n_test, a.s,
                          a.batch_size, a.device, mode, calib, credit, every, warm, crate)
                out.append(rec)
                print(f"  -> best {rec['best_test_acc']:.4f}", flush=True)
                with open(path, "w") as f:      # checkpoint after every run
                    json.dump(out, f, indent=1)
    print(f"-> {path}")
