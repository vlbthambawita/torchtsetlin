"""torchtsetlin_plans/experiments/transformer_concepts/PLAN_HTM.md Part H1: tolerant clauses (E-H1a equivalence, E-H1b noise, E-H1c CIFAR-10 flat).

    python -m experiments.transformer_concepts.code.htm.exp_h1 jobs        # jobs_htm_h1a.json, jobs_htm_h1b_search.json
    python -m experiments.transformer_concepts.code.htm.exp_h1 freeze      # results/htm/frozen_h1b.json
    python -m experiments.transformer_concepts.code.htm.exp_h1 final      # jobs_htm_h1b_final.json
    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.htm.exp_h1 work jobs.json [...]

Records: results/htm/h1a/, h1b_search/, h1b_final/, h1c/. Accuracy of E-H1a is on validation
(an equivalence check, no claim about the task); E-H1b finals and E-H1c use the test set.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

import torchtsetlin as tt
from experiments.transformer_concepts.code import runner
from experiments.transformer_concepts.code.htm.tolerant import TolerantTM
from experiments.transformer_concepts.code.pairs import pair_tokens

R = runner.RESULTS / "htm"
ROOT = runner.ROOT


# ----------------------------------------------------------------------------- data
def noisy_xor(n, seed, label_noise):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (n, 12), generator=g).bool()
    y = (x[:, 0] ^ x[:, 1]).long()
    if label_noise:
        flip = torch.rand(n, generator=g) < label_noise
        y = torch.where(flip, 1 - y, y)
    return x, y


def prototypes(n, eta, seed, F=64, K=4, per_class=3, split_seed=0):
    """K classes, each a mixture of `per_class` random prototypes of F bits; i.i.d. bit flips eta.
    The prototypes are fixed by split_seed (same for every split); `seed` draws the examples."""
    gp = torch.Generator().manual_seed(1000 + split_seed)
    protos = torch.randint(0, 2, (K * per_class, F), generator=gp).bool()
    g = torch.Generator().manual_seed(seed)
    which = torch.randint(0, K * per_class, (n,), generator=g)
    x = protos[which] ^ (torch.rand(n, F, generator=g) < eta)
    return x, which // per_class, protos


def bitflip(x, eta, seed):
    if eta == 0:
        return x
    g = torch.Generator().manual_seed(seed)
    return x ^ (torch.rand(x.shape, generator=g) < eta)


SPLIT = {"train": (20000, 11), "valid": (5000, 12), "test": (10000, 13)}


def dataset(task, split, eta=0.0):
    """Returns (x Boolean, features ready for the model, y, n_classes, kind)."""
    n, sd = SPLIT[split]
    if task == "xor":
        x, y = noisy_xor(n if split != "train" else 5000, sd, 0.4 if split == "train" else 0.0)
        return x, y, 2, "flat"
    if task == "proto":
        x, y, _ = prototypes(n, eta, sd)
        return x, y, 4, "flat"
    if task in ("T5", "T4"):
        xt, y, spec = runner.make_data(task, 16, split)
        cols = pair_tokens(xt, spec.V, window=2, last_bit=True)  # (n, F, P)
        return bitflip(cols, eta, sd), y, spec.n_classes, "pair"
    if task == "T7":
        xt, y, spec = runner.make_data(task, 16, split)
        x = torch.nn.functional.one_hot(xt, spec.V).bool().reshape(len(xt), -1)
        return bitflip(x, eta, sd), y, spec.n_classes, "flat"
    if task == "cifar":
        from experiments.transformer_concepts.code import cifar
        D = cifar.load("cpu")
        key = {"train": ("xtr", "ytr"), "valid": ("xva", "yva"), "test": ("xte", "yte")}[split]
        return D[key[0]].reshape(len(D[key[1]]), -1), D[key[1]], 10, "flat"
    raise ValueError(task)


def bayes_proto(eta, split="test"):
    """Nearest-prototype (maximum-likelihood mixture) accuracy for the proto task."""
    n, sd = SPLIT[split]
    x, y, protos = prototypes(n, eta, sd)
    d = (x[:, None, :] != protos[None]).sum(2)
    return float(((d.argmin(1) // 3) == y).float().mean())


# ----------------------------------------------------------------------------- one run
def build(impl, kind, K, M, T, s, delta, seed, device):
    if impl == "mine":
        return TolerantTM(K, M, T, s, delta, weighted=True, device=device, seed=seed)
    torch.manual_seed(seed)
    if kind == "flat":
        return tt.TsetlinMachine(None, K, M, T, s, weighted=True).to(device)
    return tt.Conv1dTsetlinMachine(K, M, T, s, kernel_size=1, position_encoding=False, weighted=True).to(device)


def predict(model, x, device, bs=256):
    if isinstance(model, TolerantTM):
        return model.predict(x).cpu()
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(x), bs):
            out.append(model.predict(x[k:k + bs].to(device)).cpu())
    model.train()
    return torch.cat(out)


def n_included(model):
    if isinstance(model, TolerantTM):
        return model.n_included()
    return float((model.ta_state >= model.n_states).sum(1).float().mean())


def run(stage, key, task, impl, delta, s, seed, M=100, T=25, epochs=20, eta=0.0, test=False,
        device="cuda", batch=32):
    path = R / stage / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    xtr, ytr, K, kind = dataset(task, "train", eta)
    xva, yva, _, _ = dataset(task, "valid", eta)
    model = build(impl, kind, K, M, T, s, delta, seed, device)
    g = torch.Generator().manual_seed(seed)
    curve, lits, best, best_ep, t_train = [], [], -1.0, -1, 0.0
    best_state = None
    xtr_d, ytr_d = xtr.to(device), ytr.to(device)
    for ep in range(epochs):
        perm = torch.randperm(len(xtr), generator=g).to(device)
        t0 = time.perf_counter()
        for k in range(0, len(xtr), batch):
            idx = perm[k:k + batch]
            model.update(xtr_d[idx], ytr_d[idx])
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        acc = float((predict(model, xva, device) == yva).float().mean())
        curve.append(round(acc, 5))
        lits.append(round(n_included(model), 3))
        if acc > best:
            best, best_ep = acc, ep
            best_state = (model.state.clone(), model.w.clone()) if impl == "mine" else \
                {k: v.clone() for k, v in model.state_dict().items()}
    rec = dict(stage=stage, key=key, task=task, impl=impl, delta=delta, s=s, seed=seed, M=M, T=T, eta=eta,
               epochs=epochs, batch=batch, n_train=len(xtr), val_curve=curve, included_per_clause=lits,
               best_val=best, best_epoch=best_ep, final_val=curve[-1], train_seconds=round(t_train, 1),
               gpu=torch.cuda.get_device_name(), torchtsetlin=tt.__version__)
    if test:
        if impl == "mine":
            model.state, model.w = best_state
        else:
            model.load_state_dict(best_state)
        xte, yte, _, _ = dataset(task, "test", eta)
        pred = predict(model, xte, device)
        runner.log_test_access(f"htm/{stage}/{key}", len(xte))
        rec["test_acc"] = float((pred == yte).float().mean())
        if task == "proto":
            rec["bayes_test_acc"] = bayes_proto(eta)
        if impl == "mine":  # firing on noisy positives vs H1.3, per learned clause length
            rec["clause_lengths"] = np.bincount(model.clause_lengths().cpu().numpy(), minlength=1).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


# ----------------------------------------------------------------------------- job lists
H1A = [("xor", dict(M=10, T=15, s=3.9, epochs=30)), ("T5", dict(M=100, T=25, s=5.0, epochs=20)),
       ("T4", dict(M=100, T=25, s=40.0, epochs=20))]
H1B_TASKS = ("proto", "T7")
ETAS = (0.0, 0.02, 0.05, 0.1)
DELTAS = (0, 1, 2, 4)
S_GRID = (5.0, 15.0, 40.0)
FINAL_SEEDS = (23, 29, 31, 37, 41)


def jobs():
    a = []
    for task, kw in H1A:
        for impl in ("lib", "mine"):
            for seed in range(10):
                a.append(dict(stage="h1a", key=f"{task}_{impl}_seed{seed}", task=task, impl=impl, delta=0,
                              seed=seed, **kw))
    (ROOT / "jobs_htm_h1a.json").write_text(json.dumps(a, indent=1))
    b = []
    for task in H1B_TASKS:
        for eta in ETAS:
            for d in DELTAS:
                for s in S_GRID:
                    b.append(dict(stage="h1b_search", key=f"{task}_eta{eta}_d{d}_s{int(s)}", task=task, impl="mine",
                                  delta=d, s=s, seed=7, eta=eta, epochs=20))
    (ROOT / "jobs_htm_h1b_search.json").write_text(json.dumps(b, indent=1))
    print(len(a), len(b))


def freeze():
    p = R / "frozen_h1b.json"
    if p.exists():
        raise SystemExit("freeze exists")
    chosen = {}
    for task in H1B_TASKS:
        for eta in ETAS:
            for d in DELTAS:
                recs = [json.loads((R / "h1b_search" / f"{task}_eta{eta}_d{d}_s{int(s)}.json").read_text()) for s in S_GRID]
                b = max(recs, key=lambda r: (r["best_val"], -r["s"]))
                chosen[f"{task}|{eta}|{d}"] = dict(s=b["s"], best_val=b["best_val"])
    p.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), chosen=chosen,
                                 rule="best validation, seed 7, ties -> smaller s"), indent=1))
    print(len(chosen), "settings frozen")


def final():
    fz = json.loads((R / "frozen_h1b.json").read_text())["chosen"]
    out = []
    for k, c in fz.items():
        task, eta, d = k.split("|")
        for seed in FINAL_SEEDS:
            out.append(dict(stage="h1b_final", key=f"{task}_eta{eta}_d{d}_seed{seed}", task=task, impl="mine",
                            delta=int(d), s=c["s"], seed=seed, eta=float(eta), epochs=20, test=True))
    (ROOT / "jobs_htm_h1b_final.json").write_text(json.dumps(out, indent=1))
    print(len(out))


ETAS_HARD = (0.2, 0.3, 0.35)  # D24: the first grid sat at the ceiling (Bayes 100 %)


def jobs_hard():
    b = [dict(stage="h1b_search", key=f"proto_eta{eta}_d{d}_s{int(s)}", task="proto", impl="mine", delta=d, s=s,
              seed=7, eta=eta, epochs=20) for eta in ETAS_HARD for d in DELTAS for s in S_GRID]
    (ROOT / "jobs_htm_h1b_hard_search.json").write_text(json.dumps(b, indent=1))
    print(len(b))


def freeze_hard():
    p = R / "frozen_h1b_hard.json"
    if p.exists():
        raise SystemExit("freeze exists")
    chosen = {}
    for eta in ETAS_HARD:
        for d in DELTAS:
            recs = [json.loads((R / "h1b_search" / f"proto_eta{eta}_d{d}_s{int(s)}.json").read_text()) for s in S_GRID]
            b = max(recs, key=lambda r: (r["best_val"], -r["s"]))
            chosen[f"proto|{eta}|{d}"] = dict(s=b["s"], best_val=b["best_val"])
    p.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), chosen=chosen,
                                 rule="best validation, seed 7, ties -> smaller s"), indent=1))
    out = []
    for k, c in chosen.items():
        task, eta, d = k.split("|")
        for seed in FINAL_SEEDS:
            out.append(dict(stage="h1b_final", key=f"{task}_eta{eta}_d{d}_seed{seed}", task=task, impl="mine",
                            delta=int(d), s=c["s"], seed=seed, eta=float(eta), epochs=20, test=True))
    (ROOT / "jobs_htm_h1b_hard_final.json").write_text(json.dumps(out, indent=1))
    print(len(chosen), "settings frozen,", len(out), "final jobs")


def work(paths):
    for path in paths:
        for job in json.loads(open(path).read()):
            d = R / job["stage"]
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{job['key']}.json").exists():
                continue
            try:
                fd = os.open(d / f"{job['key']}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            t0 = time.time()
            try:
                r = run(**job)
                (d / f"{job['key']}.lock").unlink(missing_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] {job['stage']}/{job['key']} val={r['best_val']:.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)
            except Exception:
                (d / f"{job['key']}.err").write_text(traceback.format_exc())
                print("FAILED", job["key"], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "work":
        work(sys.argv[2:])
    else:
        {"jobs": jobs, "freeze": freeze, "final": final, "jobs_hard": jobs_hard, "freeze_hard": freeze_hard}[cmd]()
