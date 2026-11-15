"""Implementability smoke (not a result): tiny runs of the baselines and the pair TM.

Checks that every arm of torchtsetlin_plans/experiments/transformer_concepts/PLAN.md section 3 runs on the unchanged library and measures seconds
per epoch for the cost estimate. Writes results/smoke.json. Never reads a test split (there is
none here: the data are generated).

Run: python -m experiments.transformer_concepts.code.smoke --device cuda
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from experiments.transformer_concepts.code import tasks
from experiments.transformer_concepts.code.pairs import pair_tokens, token_columns

import torchtsetlin as tt


def make(task, N, L, g):
    if task == "T1_ordered_pair":
        return tasks.ordered_pair(N, L, 8, g), 8, 2
    if task == "T4_needle_d4":
        return tasks.needle(N, L, 8, 4, 4, g), 8, 4
    if task == "T2_assoc_recall":
        return tasks.associative_recall(N, L, 8, 4, g), 12, 4
    raise ValueError(task)


def arms(V, L, n_cls, M, T, s):
    kw = dict(weighted=True)
    flat = lambda x: torch.eye(V, dtype=torch.bool)[x].reshape(x.shape[0], -1)  # noqa: E731
    return {
        "flat": (lambda: tt.TsetlinMachine(None, n_cls, M, T, s, **kw), flat),
        "conv_k1_nopos": (lambda: tt.Conv1dTsetlinMachine(n_cls, M, T, s, kernel_size=1,
                                                          position_encoding=False, **kw),
                          lambda x: token_columns(x, V, "none")),
        "conv_k1_therm": (lambda: tt.Conv1dTsetlinMachine(n_cls, M, T, s, kernel_size=1,
                                                          position_encoding=False, **kw),
                          lambda x: token_columns(x, V, "thermometer")),
        "pair_self_only": (lambda: tt.Conv1dTsetlinMachine(n_cls, M, T, s, kernel_size=1,
                                                           position_encoding=False, **kw),
                           lambda x: pair_tokens(x, V, window=2, last_bit=True, self_only=True)),
        "pair_w2": (lambda: tt.Conv1dTsetlinMachine(n_cls, M, T, s, kernel_size=1,
                                                    position_encoding=False, **kw),
                    lambda x: pair_tokens(x, V, window=2, last_bit=True)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1000)
    args = ap.parse_args()
    dev = torch.device(args.device)
    M, T, s, B = 40, 10, 5.0, 32
    recs = []
    for task, L in (("T1_ordered_pair", 12), ("T4_needle_d4", 12), ("T2_assoc_recall", 11)):
        g = torch.Generator().manual_seed(12345)
        (x, y), V, n_cls = make(task, args.n_train + args.n_val, L, g)
        for name, (build, feat) in arms(V, L, n_cls, M, T, s).items():
            torch.manual_seed(0)
            X = feat(x).to(dev)
            Y = y.to(dev)
            xtr, ytr, xva, yva = X[: args.n_train], Y[: args.n_train], X[args.n_train:], Y[args.n_train:]
            model = build().to(dev)
            hist, t_train = [], 0.0
            for _ in range(args.epochs):
                model.train()
                perm = torch.randperm(args.n_train, device=dev)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for k in range(0, args.n_train, B):
                    idx = perm[k:k + B]
                    model.update(xtr[idx], ytr[idx])
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_train += time.perf_counter() - t0
                model.eval()
                with torch.no_grad():
                    acc = (model.predict(xva) == yva).float().mean().item()
                hist.append(round(acc, 4))
            rec = dict(task=task, L=L, V=V, arm=name, features=int(X.shape[1]),
                       pairs_or_tokens=int(X.shape[-1]) if X.dim() == 3 else None,
                       M=M, T=T, s=s, batch=B, epochs=args.epochs, n_train=args.n_train,
                       n_val=args.n_val, val_acc_per_epoch=hist,
                       train_s_per_epoch=round(t_train / args.epochs, 3), device=str(dev))
            print(json.dumps(rec))
            recs.append(rec)
    out = dict(kind="smoke", note="implementability check, not a result", torchtsetlin=tt.__version__,
               torch=torch.__version__, python=platform.python_version(),
               gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, records=recs)
    path = Path(__file__).resolve().parents[1] / "results" / "smoke.json"
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
