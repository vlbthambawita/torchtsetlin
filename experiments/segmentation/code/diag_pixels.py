"""Diagnostic: are the dense Boolean features informative, and can a TM fit them?

Samples balanced-at-random pixels from Kvasir 128 train/val, encodes them with a feature spec,
and fits logistic regression, histogram gradient boosting and a flat TsetlinMachine on the
*same* Boolean vectors. Metric: pixel IoU on the polyp class and balanced accuracy, train & val.

    python code/diag_pixels.py --spec or_wide --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
import features  # noqa: E402
import record  # noqa: E402
import torchtsetlin as tt  # noqa: E402

SPECS = {
    "local": ({"bits": 4}, 3),
    "pyr124": ({"bits": 4, "pyr": {"scales": [1, 2, 4], "mode": "or", "patch": 3}}, 1),
    "or_wide": ({"bits": 4, "pyr": {"scales": [1, 2, 4, 8, 16, 32], "mode": "or", "patch": 3}}, 1),
    "occmaj_wide": ({"bits": 4, "pyr": {"scales": [1, 2, 4, 8, 16, 32], "mode": "occ", "t": [0.5], "patch": 3}}, 1),
}


def sample(split, spec, patch, n, seed, dev):
    d = data.load_split("kvasir", 128, split)
    g = torch.Generator().manual_seed(seed)
    X, Y = [], []
    per = n // len(d["x"]) + 1
    for i in range(0, len(d["x"]), 16):
        planes = features.encode(d["x"][i : i + 16].to(dev), spec).float()
        if patch > 1:
            B, Z, H, W = planes.shape
            planes = F.unfold(planes, patch, padding=patch // 2).view(B, Z * patch * patch, H, W)
        y = d["y"][i : i + 16]
        for b in range(planes.shape[0]):
            valid = torch.nonzero((y[b] != 255).view(-1)).squeeze(1)
            idx = valid[torch.randint(len(valid), (per,), generator=g)]
            X.append(planes[b].view(planes.shape[1], -1)[:, idx.to(dev)].t().bool().cpu())
            Y.append(y[b].view(-1)[idx].long())
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def scores(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    tp = ((pred == 1) & (y == 1)).sum()
    iou1 = tp / max(1, ((pred == 1) | (y == 1)).sum())
    iou0 = ((pred == 0) & (y == 0)).sum() / max(1, ((pred == 0) | (y == 0)).sum())
    bacc = 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())
    return {"miou": float((iou0 + iou1) / 2), "iou_fg": float(iou1), "bal_acc": float(bacc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="or_wide")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=60000)
    a = ap.parse_args()
    spec, patch = SPECS[a.spec]
    Xtr, ytr = sample("train", spec, patch, a.n, 1, a.device)
    Xva, yva = sample("val", spec, patch, a.n // 3, 2, a.device)
    out = {"spec": a.spec, "n_features": int(Xtr.shape[1]), "n_train": len(ytr), "fg_rate_train": float(ytr.float().mean())}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    for name, clf in (("logreg", LogisticRegression(max_iter=2000, C=1.0)),
                      ("hgb", HistGradientBoostingClassifier(max_iter=300))):
        t = time.time()
        clf.fit(Xtr.numpy().astype(np.float32), ytr.numpy())
        out[name] = {"train": scores(clf.predict(Xtr.numpy().astype(np.float32)), ytr),
                     "val": scores(clf.predict(Xva.numpy().astype(np.float32)), yva), "s": time.time() - t}
        print(name, out[name], flush=True)
    for C, T, s, bs in ((600, 120, 20.0, 64), (600, 60, 10.0, 1024), (2000, 400, 20.0, 64)):
        tt.seed_everything(0)
        m = tt.TsetlinMachine(Xtr.shape[1], 2, n_clauses=C, T=T, s=s, weighted=True, n_states=100).to(a.device)
        hist = []
        t = time.time()
        for ep in range(10):
            m.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(perm), bs):
                j = perm[i : i + bs]
                m.update(Xtr[j].to(a.device), ytr[j].to(a.device))
            m.eval()
            ptr = m.predict(Xtr.to(a.device)).cpu()
            pva = m.predict(Xva.to(a.device)).cpu()
            hist.append({"epoch": ep + 1, "train": scores(ptr, ytr), "val": scores(pva, yva)})
            print(f"TM C={C} T={T} s={s} bs={bs}", hist[-1], flush=True)
        out[f"tm_C{C}_T{T}_s{s}_bs{bs}"] = {"history": hist, "s": time.time() - t}
    record.write_record({"run_id": f"DIAG_pixels_{a.spec}", "status": "completed", "env": record.environment(a.device), "result": out})


if __name__ == "__main__":
    main()
