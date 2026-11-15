"""Smoke diagnostic: is T4 unlearned in ``smoke.py`` because of the features or the settings?

Checks a hand-set clause on the pair features (all 5,000 generated examples), then trains the
pair TM for 20 epochs at two ``s`` and two ``(M, T)`` settings. Writes results/smoke_t4_s.json.
Not a result: one seed, generated validation data, no search.

Run: python -m experiments.transformer_concepts.code.smoke_t4_s
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from experiments.transformer_concepts.code import tasks
from experiments.transformer_concepts.code.pairs import pair_feature_names, pair_tokens

import torchtsetlin as tt


def main() -> None:
    g = torch.Generator().manual_seed(12345)
    L, V = 12, 8
    x, y = tasks.needle(5000, L, V, 4, 4, g)
    cols = pair_tokens(x, V, window=2, last_bit=True)
    names = pair_feature_names(L, V, 2, False, True)
    scores = []
    for c in range(4):
        lit = [names.index("q=0"), names.index(f"k0={c + 1}"), names.index("rel>3")]
        hit = cols[:, lit, :].all(1) & ~cols[:, names.index("rel>4"), :]  # rel == 4
        scores.append(hit.any(1))
    sc = torch.stack(scores, 1)
    rec = dict(kind="smoke_diagnostic", task="T4_needle_d4", L=L, V=V, arm="pair_w2",
               hand_clause_acc_all_5000=(sc.float().argmax(1) == y).float().mean().item(), runs=[])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = cols.to(dev), y.to(dev)
    for s in (5.0, 15.0):
        for M, T in ((40, 10), (100, 25)):
            torch.manual_seed(0)
            m = tt.Conv1dTsetlinMachine(4, M, T, s, kernel_size=1, position_encoding=False,
                                        weighted=True).to(dev)
            hist = []
            for _ in range(20):
                m.train()
                p = torch.randperm(4000, device=dev)
                for k in range(0, 4000, 32):
                    m.update(X[p[k:k + 32]], Y[p[k:k + 32]])
                m.eval()
                hist.append(round((m.predict(X[4000:]) == Y[4000:]).float().mean().item(), 3))
            rec["runs"].append(dict(s=s, M=M, T=T, batch=32, n_train=4000, n_val=1000,
                                    val_acc_per_epoch=hist))
    path = Path(__file__).resolve().parents[1] / "results" / "smoke_t4_s.json"
    path.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec)[:400])


if __name__ == "__main__":
    main()
