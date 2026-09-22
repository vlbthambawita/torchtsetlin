"""E3-E5: the three repairs proposed in section 7.3 of the first report, on CIFAR-10.

The first report concluded that the MCTM architecture is sound and its greedy supervised
training is not, and named three ways out. This driver runs them. Every arm keeps the
architecture of the first report fixed -- 4x4 layer-1 patches, 128 shared clauses, 2x2 OR
pool, 3x3 layer-2 patches, 512 clauses -- so the numbers are directly comparable to
``results/<arm>_seed<k>.json`` from E1.

  --l1 greedy   layer 1 trained to classify, as in E1 (loaded from the E1 checkpoint cache)
  --l1 auto     idea A: layer 1 pretrained as a patch autoencoder, no labels
  --l1 random   random clauses, the E1 control that beat both

  --calib R     idea B: calibrate layer 1 to a target per-patch firing rate R before freezing
  --calib-every N   re-run the controller every N batches during credit training
  --calib-l2 R  run the controller on layer 2 as well, to a target match rate

  --credit MODE idea C: ia   propagate Type Ia + Type II only (the literal proposal)
                        ib   + Type Ib, blamed on the literal that blocked the match
                        bal  + Type Ib capped at the Type Ia rate
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from cifar import load_cifar10
from ideas import (
    CreditStack,
    _probe_literals,
    calibrate_density,
    pretrain_autoencoder_l1,
)
from mctm import (
    MCTM,
    clause_composition,
    conv_clause_maps,
    firing_stats,
    or_pool,
    randomize_clauses,
)
from run import (
    C1,
    C2,
    CKPT,
    P1,
    P2,
    POOL,
    RESULTS,
    S1,
    S2,
    T1,
    T2,
    accuracy,
    make_conv,
    materialize,
    now,
    train_model,
)

import torchtsetlin as tt


# ------------------------------------------------------------------ layer 1 construction
def build_l1(kind: str, in_shape, xtr, ytr, xte, yte, *, seed: int, s1: float, epochs_l1: int,
             n_bits: int, batch_size: int, ae_patches: int, ae_epochs: int, rec: dict):
    """The frozen (or initial) first layer, by objective."""
    l1 = make_conv(C1, T1, s1, P1, S1, in_shape, seed=seed)
    os.makedirs(CKPT, exist_ok=True)
    if kind == "random":
        randomize_clauses(l1, n_include=3, seed=seed)
        return l1
    if kind == "auto":
        key = f"ae_seed{seed}_s{s1}_ep{ae_epochs}_p{ae_patches}_n{xtr.shape[0]}_b{n_bits}.pt"
        path = os.path.join(CKPT, key)
        if os.path.exists(path):
            blob = torch.load(path, map_location=xtr.device)
            l1.load_state_dict(blob["state"])
            rec["ae"] = blob["ae"]
            rec["ae_cached"] = True
            print(f"  [auto:L1] loaded cache {key}", flush=True)
        else:
            rec["ae"] = pretrain_autoencoder_l1(l1, xtr, n_patches=ae_patches,
                                                epochs=ae_epochs, s=s1, seed=seed)
            torch.save({"state": l1.state_dict(), "ae": rec["ae"]}, path)
            rec["ae_cached"] = False
        return l1
    # greedy: the E1 layer 1, from the same cache key run.py writes
    key = f"l1_seed{seed}_incNone_s{s1}_ep{epochs_l1}_n{xtr.shape[0]}_b{n_bits}.pt"
    path = os.path.join(CKPT, key)
    if os.path.exists(path):
        blob = torch.load(path, map_location=xtr.device)
        l1.load_state_dict(blob["state"])
        rec["log_l1"] = blob["log"]
        rec["l1_cached"] = True
        print(f"  [greedy:L1] loaded cache {key}", flush=True)
    else:
        train_model(l1, xtr, ytr, xte, yte, epochs_l1, batch_size, "greedy:L1", rec["log_l1"])
        torch.save({"state": l1.state_dict(), "log": rec["log_l1"]}, path)
        rec["l1_cached"] = False
    rec["l1_test_acc"] = rec["log_l1"][-1]["test_acc"]
    return l1


def l1_report(l1, xtr, pool: int, tag: str) -> dict:
    """Firing statistics and clause composition of a feature layer, as E0 measured them.

    ``or_density`` is the mean value of the *globally* OR-pooled bit -- what the flat-stack
    readout sees. A channel firing on p of 841 patch positions is globally on with
    probability ~1-(1-p)^841, so a feature layer dense enough to be useful to a spatial second
    layer is simultaneously saturated for the global-OR readout. The two numbers have to be
    read together.
    """
    maps = or_pool(conv_clause_maps(l1, xtr[:5000]), pool)
    fs = firing_stats(maps)
    orb = float(l1.evaluate_clauses(xtr[:5000]).float().mean())
    print(f"  [{tag}] L1 size {float(l1.include_count.float().median()):.0f}  median firing "
          f"{fs['median']*100:.3f}%  dead {fs['frac_dead']:.3f}  global-OR {orb:.3f}",
          flush=True)
    return {"firing": fs, "or_density": orb, "composition": clause_composition(l1),
            "include_count": {"min": int(l1.include_count.min()),
                              "median": int(l1.include_count.median()),
                              "max": int(l1.include_count.max())}}


# ------------------------------------------------------------------ the run
def run(arm: str, a) -> dict:
    dev = "cuda"
    xtr, ytr, xte, yte = load_cifar10(n_bits=a.n_bits, device=dev, max_train=a.subset)
    in_shape = tuple(xtr.shape[1:])
    rec = {"arm": arm, "l1_kind": a.l1, "calib": a.calib, "max_size": a.max_size,
           "calib_l2": a.calib_l2,
           "credit": a.credit, "credit_rate": a.credit_rate,
           "calib_every": a.calib_every, "warmup": a.warmup,
           "seed": a.seed,
           "epochs": a.epochs, "epochs_l1": a.epochs_l1, "n_bits": a.n_bits,
           "n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]),
           "batch_size": a.batch_size, "input_shape": list(in_shape),
           "hp": {"P1": P1, "S1": S1, "C1": C1, "POOL": POOL, "P2": P2, "S2": S2, "C2": C2,
                  "T1": T1, "s1": a.s1, "T2": T2, "s2": a.s2},
           "log": [], "log_l1": []}
    t_start = now()

    l1 = build_l1(a.l1, in_shape, xtr, ytr, xte, yte, seed=a.seed, s1=a.s1,
                  epochs_l1=a.epochs_l1, n_bits=a.n_bits, batch_size=a.batch_size,
                  ae_patches=a.ae_patches, ae_epochs=a.ae_epochs, rec=rec)
    l1.eval()
    rec["l1_before"] = l1_report(l1, xtr, POOL, f"{arm}:pre")

    # ---- idea B: calibrate the frozen representation to a target firing rate ----------
    probe = None
    if a.calib > 0 or a.calib_every:
        probe = _probe_literals(l1, xtr, a.probe_patches, seed=a.seed)
    if a.calib > 0:
        t0 = now()
        rec["calibration"] = calibrate_density(l1, xtr, a.calib, band=a.calib_band,
                                               max_size=a.max_size, seed=a.seed, lits=probe)
        rec["calibration"]["seconds"] = now() - t0
        print(f"  [{arm}] calibrated to {a.calib:.3f}: raw median "
              f"{rec['calibration']['rate_median']*100:.3f}%  in band "
              f"{rec['calibration']['frac_in_band']:.2f}  size "
              f"{rec['calibration']['size_median']:.0f}  ({rec['calibration']['seconds']:.1f}s)",
              flush=True)
        rec["l1_after"] = l1_report(l1, xtr, POOL, f"{arm}:post")

    # ---- flat readout: how much class information survives in the layer-1 bits -------
    if a.head == "flat":
        htr = torch.cat([l1.evaluate_clauses(xtr[i:i+512]) for i in range(0, xtr.shape[0], 512)])
        hte = torch.cat([l1.evaluate_clauses(xte[i:i+512]) for i in range(0, xte.shape[0], 512)])
        rec["feature_shape"] = list(htr.shape[1:])
        tt.seed_everything(a.seed)
        head = tt.CoalescedTsetlinMachine(None, 10, C2, T2, a.s2).to(dev)
        train_model(head, htr, ytr, hte, yte, a.epochs, a.batch_size, arm, rec["log"])
        rec["composition"] = clause_composition(head)
        rec["n_clauses_total"] = C1 + C2
        rec["n_automata"] = int(l1.ta_state.numel()) + int(head.ta_state.numel())
    elif a.credit == "none":
        # ---- frozen stack, exactly the E1 pipeline -----------------------------------
        t_tf = now()
        htr, hte = materialize(l1, xtr, POOL), materialize(l1, xte, POOL)
        rec["transform_s"] = now() - t_tf
        rec["feature_shape"] = list(htr.shape[1:])
        rec["l1_firing"] = firing_stats(htr[:5000])
        head = make_conv(C2, T2, a.s2, P2, S2, tuple(htr.shape[1:]), seed=a.seed + 1000)
        train_model(head, htr, ytr, hte, yte, a.epochs, a.batch_size, arm, rec["log"])
        rec["composition"] = clause_composition(head)
        rec["composition_l1"] = clause_composition(l1)
        stack = MCTM([l1], [POOL], head)
        rec["n_clauses_total"] = stack.n_clauses_total()
        rec["n_automata"] = stack.n_automata()
    else:
        # ---- idea C: joint training with credit flowing down --------------------------
        maps0 = or_pool(conv_clause_maps(l1, xtr[:64]), POOL)
        rec["feature_shape"] = list(maps0.shape[1:])
        head = make_conv(C2, T2, a.s2, P2, S2, tuple(maps0.shape[1:]), seed=a.seed + 1000)
        stack = CreditStack(l1, head, pool=POOL, type_ib=(a.credit != "ia"),
                            ib_ratio=(1.0 if a.credit == "bal" else 0.0),
                            credit_rate=a.credit_rate)
        rec["l1_firing"] = firing_stats(or_pool(conv_clause_maps(l1, xtr[:5000]), POOL))
        probe2 = None
        n = xtr.shape[0]
        if a.warmup:
            # Greedy pretraining followed by end-to-end fine-tuning, which is how layer-wise
            # pretraining is actually used. It also gives the credit path its best case: an
            # untrained layer 2 has no included positive channel literals, so there is nothing
            # to propagate through until it has some.
            htr, hte = materialize(l1, xtr, POOL), materialize(l1, xte, POOL)
            train_model(head, htr, ytr, hte, yte, a.warmup, a.batch_size,
                        f"{arm}:warmup", rec["log"])
            rec["warmup_acc"] = rec["log"][-1]["test_acc"]
            del htr, hte
            torch.cuda.empty_cache()
        stack.train()
        for ep in range(a.warmup, a.epochs):
            t0 = now()
            perm = torch.randperm(n, device=dev)
            nb = 0
            for i in range(0, n, a.batch_size):
                idx = perm[i : i + a.batch_size]
                stack.update(xtr[idx], ytr[idx])
                nb += 1
                if a.calib_every and nb % a.calib_every == 0:
                    calibrate_density(l1, xtr, a.calib if a.calib > 0 else a.calib_target,
                                      band=a.calib_band, seed=a.seed, lits=probe,
                                      max_rounds=a.calib_rounds)
                    if a.calib_l2 > 0:
                        if probe2 is None:
                            probe2 = _probe_literals(
                                head, or_pool(conv_clause_maps(l1, xtr[:2000]), POOL),
                                a.probe_patches, seed=a.seed)
                        calibrate_density(head, None, a.calib_l2, band=a.calib_band,
                                          seed=a.seed, lits=probe2, max_rounds=a.calib_rounds)
            dt = now() - t0
            stack.eval()
            te = accuracy(stack, xte, yte)
            tr = accuracy(stack, xtr[:10000], ytr[:10000])
            stack.train()
            fs = firing_stats(or_pool(conv_clause_maps(l1, xtr[:2000]), POOL))
            rec["log"].append({"epoch": ep + 1, "train_acc": tr, "test_acc": te, "epoch_s": dt,
                               "l1_fire_median": fs["median"], "l1_dead": fs["frac_dead"],
                               "l1_size_median": float(l1.include_count.float().median()),
                               "credit": dict(stack.stats)})
            print(f"  [{arm}] epoch {ep+1}/{a.epochs}  train {tr:.4f}  test {te:.4f}  "
                  f"L1 size {float(l1.include_count.float().median()):.0f} fire "
                  f"{fs['median']*100:.3f}%  events Ia/Ib/II "
                  f"{stack.stats['kept_i']}/{stack.stats['kept_ib']}/{stack.stats['kept_ii']}"
                  f"  ({dt:.1f}s)", flush=True)
        rec["credit_stats"] = dict(stack.stats)
        rec["composition"] = clause_composition(head)
        rec["composition_l1"] = clause_composition(l1)
        rec["l1_after"] = l1_report(l1, xtr, POOL, f"{arm}:post")
        rec["n_clauses_total"] = stack.n_clauses_total()
        rec["n_automata"] = stack.n_automata()

    rec["final_test_acc"] = rec["log"][-1]["test_acc"]
    rec["best_test_acc"] = max(r["test_acc"] for r in rec["log"])
    rec["wall_s"] = now() - t_start
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="name of the output record")
    ap.add_argument("--l1", default="greedy", choices=["greedy", "auto", "random"])
    ap.add_argument("--head", default="conv", choices=["conv", "flat"])
    ap.add_argument("--calib", type=float, default=0.0, help="target layer-1 firing rate")
    ap.add_argument("--calib-l2", type=float, default=0.0)
    ap.add_argument("--calib-band", type=float, default=2.0)
    ap.add_argument("--max-size", type=int, default=None,
                    help="hard cap on layer-1 clause size, enforced by the controller")
    ap.add_argument("--calib-every", type=int, default=0, help="batches between controller runs")
    ap.add_argument("--calib-target", type=float, default=0.02)
    ap.add_argument("--calib-rounds", type=int, default=40)
    ap.add_argument("--credit", default="none", choices=["none", "ia", "ib", "bal"])
    ap.add_argument("--credit-rate", type=float, default=1.0,
                    help="fraction of credit events actually applied (the step size)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="epochs of frozen-layer-1 training before credit is switched on")
    ap.add_argument("--probe-patches", type=int, default=20000)
    ap.add_argument("--ae-patches", type=int, default=300000)
    ap.add_argument("--ae-epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--epochs-l1", type=int, default=30)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--n-bits", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--s1", type=float, default=10.0)
    ap.add_argument("--s2", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rec = run(a.arm, a)
    out = a.out or os.path.join(RESULTS, f"{a.arm}_seed{a.seed}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"-> {out}  best_test_acc={rec['best_test_acc']:.4f}  wall={rec['wall_s']:.0f}s")
