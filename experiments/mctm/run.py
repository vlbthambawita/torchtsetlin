"""MCTM experiment driver for CIFAR-10.

Every arm answers one question from the review of the MCTM proposal:

  single-l1        what layer 1 alone achieves (the greedy ceiling it is trained against)
  single-matched   same total clause budget in ONE layer  -> does depth beat width?
  single-rf        one layer with patch = the stack's effective receptive field (9x9)
                   -> is any gain just a bigger receptive field?
  flat-stack       layer-1 bits, globally OR-pooled, into a FLAT TM
                   -> does keeping the spatial map earn its cost over the trivial stack?
  mctm             the proposal: frozen layer 1 -> spatial maps -> OR-pool -> conv layer 2
  mctm-random      random layer-1 clauses, same layer 2
                   -> does the supervised local objective buy anything?
  mctm-nopool      pool=1 ablation -> is OR-pooling load-bearing (density control)?
  mctm-nopos       position_encoding=False at layer 2
"""
from __future__ import annotations

import argparse, json, os, time
from typing import Optional

import torch

import torchtsetlin as tt
from cifar import load_cifar10
from mctm import (MCTM, clause_composition, conv_clause_maps, effective_receptive_field,
                  firing_stats, or_pool, randomize_clauses)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ckpt")

# ---- shared hyper-parameters -------------------------------------------------------
P1, S1, C1 = 4, 1, 128          # layer 1: 4x4 patches, stride 1, 128 shared clauses
POOL = 2                        # OR-pool between layers
P2, S2, C2 = 3, 1, 512          # layer 2: 3x3 patches over the pooled maps
T1, SPEC1 = 128.0, 10.0
T2, SPEC2 = 512.0, 10.0


def now() -> float:
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@torch.no_grad()
def accuracy(model, x, y, batch_size: int = 256) -> float:
    was = model.training
    model.eval()
    correct = 0
    for i in range(0, x.shape[0], batch_size):
        v = model(x[i : i + batch_size])
        correct += int((v.argmax(dim=1) == y[i : i + batch_size]).sum())
    model.train(was)
    return correct / x.shape[0]


def train_model(model, xtr, ytr, xte, yte, epochs: int, batch_size: int, tag: str, log: list):
    model.train()
    n = xtr.shape[0]
    for ep in range(epochs):
        t0 = now()
        perm = torch.randperm(n, device=xtr.device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            model.update(xtr[idx], ytr[idx])
        dt = now() - t0
        tr = accuracy(model, xtr[:10000], ytr[:10000])
        te = accuracy(model, xte, yte)
        log.append({"epoch": ep + 1, "train_acc": tr, "test_acc": te, "epoch_s": dt})
        print(f"  [{tag}] epoch {ep+1}/{epochs}  train {tr:.4f}  test {te:.4f}  ({dt:.1f}s)", flush=True)
    return log


def make_conv(n_clauses, T, s, patch, stride, in_shape, pos=True, max_inc=None, seed=0):
    tt.seed_everything(seed)
    return tt.ConvCoalescedTsetlinMachine(
        n_outputs=10, n_clauses=n_clauses, T=T, s=s,
        patch_size=patch, stride=stride, position_encoding=pos,
        input_shape=in_shape, max_included_literals=max_inc,
    ).to("cuda")


@torch.no_grad()
def materialize(model, x, pool, batch_size=256):
    outs = []
    was = model.training
    model.eval()
    for i in range(0, x.shape[0], batch_size):
        outs.append(or_pool(conv_clause_maps(model, x[i : i + batch_size]), pool))
    model.train(was)
    return torch.cat(outs, dim=0)


def run(arm: str, seed: int, epochs: int, epochs_l1: int, subset: Optional[int], n_bits: int,
        batch_size: int, max_inc1: Optional[int], s1: float,
        pool_override: Optional[int] = None, s2: float = None) -> dict:
    dev = "cuda"
    xtr, ytr, xte, yte = load_cifar10(n_bits=n_bits, device=dev, max_train=subset)
    in_shape = tuple(xtr.shape[1:])
    rec = {"arm": arm, "seed": seed, "epochs": epochs, "epochs_l1": epochs_l1,
           "n_bits": n_bits, "n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]),
           "batch_size": batch_size, "input_shape": list(in_shape),
           "hp": {"P1": P1, "S1": S1, "C1": C1, "POOL": POOL, "P2": P2, "S2": S2, "C2": C2,
                  "T1": T1, "s1": s1, "T2": T2, "s2": s2, "max_inc1": max_inc1},
           "log": [], "log_l1": []}
    t_start = now()

    # ---------------- single-layer baselines ----------------
    if arm in ("single-l1", "single-matched", "single-rf"):
        if arm == "single-l1":
            n_cl, patch, T = C1, P1, T1
        elif arm == "single-matched":
            n_cl, patch, T = C1 + C2, P1, T2
        else:
            rf, _ = effective_receptive_field([P1, P2], [S1, S2], [POOL, 1])
            n_cl, patch, T = C1 + C2, rf, T2
            rec["rf_patch"] = rf
        m = make_conv(n_cl, T, s1, patch, S1, in_shape, max_inc=max_inc1, seed=seed)
        train_model(m, xtr, ytr, xte, yte, epochs, batch_size, arm, rec["log"])
        rec["n_clauses_total"] = int(m.n_clauses_total)
        rec["n_automata"] = int(m.ta_state.numel())
        rec["composition"] = clause_composition(m)
        rec["final_test_acc"] = rec["log"][-1]["test_acc"]
        rec["best_test_acc"] = max(r["test_acc"] for r in rec["log"])
        rec["wall_s"] = now() - t_start
        return rec

    # ---------------- everything else needs layer 1 ----------------
    # mctm / mctm-nopool / mctm-nopos / flat-stack all share one layer 1 per (seed, config),
    # so it is trained once and cached rather than retrained four times.
    l1 = make_conv(C1, T1, s1, P1, S1, in_shape, max_inc=max_inc1, seed=seed)
    if arm == "mctm-random":
        randomize_clauses(l1, n_include=3, seed=seed)
        rec["l1"] = "random"
    else:
        os.makedirs(CKPT, exist_ok=True)
        key = f"l1_seed{seed}_inc{max_inc1}_s{s1}_ep{epochs_l1}_n{xtr.shape[0]}_b{n_bits}.pt"
        path = os.path.join(CKPT, key)
        if os.path.exists(path):
            blob = torch.load(path, map_location=xtr.device)
            l1.load_state_dict(blob["state"])
            rec["log_l1"] = blob["log"]
            rec["l1_cached"] = True
            print(f"  [{arm}:L1] loaded cache {key}", flush=True)
        else:
            train_model(l1, xtr, ytr, xte, yte, epochs_l1, batch_size, f"{arm}:L1", rec["log_l1"])
            torch.save({"state": l1.state_dict(), "log": rec["log_l1"]}, path)
            rec["l1_cached"] = False
        rec["l1"] = "trained"
        rec["l1_test_acc"] = rec["log_l1"][-1]["test_acc"]
    l1.eval()

    pool = 1 if arm == "mctm-nopool" else (pool_override or POOL)
    rec["pool"] = pool

    rec["l1_include_count"] = {
        "min": int(l1.include_count.min()), "median": int(l1.include_count.median()),
        "max": int(l1.include_count.max()), "mean": float(l1.include_count.float().mean())}

    if arm == "flat-stack":
        htr = torch.cat([l1.evaluate_clauses(xtr[i:i+512]) for i in range(0, xtr.shape[0], 512)])
        hte = torch.cat([l1.evaluate_clauses(xte[i:i+512]) for i in range(0, xte.shape[0], 512)])
        rec["feature_shape"] = list(htr.shape[1:])
        tt.seed_everything(seed)
        head = tt.CoalescedTsetlinMachine(None, 10, C2, T2, s2).to(dev)
        train_model(head, htr, ytr, hte, yte, epochs, batch_size, arm, rec["log"])
    else:
        t_tf = now()
        htr = materialize(l1, xtr, pool)
        hte = materialize(l1, xte, pool)
        rec["transform_s"] = now() - t_tf
        rec["feature_shape"] = list(htr.shape[1:])
        rec["l1_firing"] = firing_stats(htr[:5000])
        print(f"  [{arm}] maps {tuple(htr.shape)}  median firing "
              f"{rec['l1_firing']['median']:.4f}  dead {rec['l1_firing']['frac_dead']:.2f}", flush=True)
        pos2 = arm != "mctm-nopos"
        head = make_conv(C2, T2, s2, P2, S2, tuple(htr.shape[1:]), pos=pos2, seed=seed + 1000)
        train_model(head, htr, ytr, hte, yte, epochs, batch_size, arm, rec["log"])

    rec["composition"] = clause_composition(head)
    rec["composition_l1"] = clause_composition(l1)
    if arm != "flat-stack":
        rf, _ = effective_receptive_field([P1, P2], [S1, S2], [pool, 1])
        rec["rf"] = rf  # pool 4 buys density but a 15x15 RF, unlike the 9x9 single-rf baseline
    print(f"  [{arm}] layer-2 clause size {rec['composition']['size_median']:.0f}, "
          f"negation fraction {rec['composition']['neg_fraction_mean']:.3f}, "
          f"empty {rec['composition']['frac_empty']:.2f}", flush=True)

    stack = MCTM([l1], [pool], head)
    rec["n_clauses_total"] = stack.n_clauses_total()
    rec["n_automata"] = stack.n_automata()
    rec["final_test_acc"] = rec["log"][-1]["test_acc"]
    rec["best_test_acc"] = max(r["test_acc"] for r in rec["log"])
    rec["wall_s"] = now() - t_start
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--epochs-l1", type=int, default=10)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-inc1", type=int, default=None, help="layer-1 clause size budget")
    ap.add_argument("--s1", type=float, default=SPEC1)
    ap.add_argument("--pool", type=int, default=None, help="override the OR-pool size")
    ap.add_argument("--s2", type=float, default=SPEC2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rec = run(a.arm, a.seed, a.epochs, a.epochs_l1, a.subset, a.n_bits, a.batch_size,
              a.max_inc1, a.s1, a.pool, a.s2)
    rec["tag"] = a.tag
    out = a.out or os.path.join(RESULTS, f"{a.arm}{a.tag}_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"-> {out}  best_test_acc={rec['best_test_acc']:.4f}  wall={rec['wall_s']:.0f}s")
