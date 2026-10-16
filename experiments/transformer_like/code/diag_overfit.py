"""Diagnostic (spec §18 'N0 training high, validation low'): train/val accuracy and clause size per
epoch for the N0 head vs the flat baseline. Train/valid only; selects nothing."""
import json
import sys

import torch

from . import data as D
from .common import ART, RESULTS, provenance, seed_torch, write_json
from .features import cached_transform
from .run import load_encoders
from .tm_adapter import TMAdapter


def main(device="cuda"):
    tr, va = D.load_split(D.MAIN, "train"), D.load_split(D.MAIN, "valid")
    q, k, _ = load_encoders(ART / "encoders/seed17.pt", device)
    ytr, yva = torch.as_tensor(tr.y).to(device), torch.as_tensor(va.y).to(device)
    out = {}
    for name, mode, M, T, s in [("N0_M500_s3_T250_t0.5", "qk", 500, 250, 3.0), ("A1_M500_s3_T250", "self_only", 500, 250, 3.0),
                                 ("B0_M500_s3_T250", "image_flat", 500, 250, 3.0)]:
        Xtr, _, _ = cached_transform(tr.images, tr.ids, mode, q, k, 0.5, device)
        Xva, _, _ = cached_transform(va.images, va.ids, mode, q, k, 0.5, device)
        seed_torch(17, "head")
        h = TMAdapter(Xtr.shape[1], M, T, s, device=device)
        rows = []
        for ep in range(6):
            h.fit_epoch(Xtr, ytr)
            tra = float((h.votes(Xtr[:10000]).argmax(1) == ytr[:10000]).double().mean())
            vaa = float((h.votes(Xva).argmax(1) == yva).double().mean())
            sm = h.state_manifest()
            img_lits = h.model.include[:, :784].sum(1) + h.model.include[:, h.model.n_features:h.model.n_features + 784].sum(1)
            rows.append({"epoch": ep + 1, "train_acc_10k": tra, "val_acc": vaa, "mean_clause_size": sm["mean_clause_size"],
                         "mean_image_literals": float(img_lits.float().mean())})
            print(name, rows[-1], flush=True)
        out[name] = rows
    write_json(RESULTS / "diagnostics" / "overfit_seed17.json", {"rows": out, **provenance(device)})


if __name__ == "__main__":
    main()
