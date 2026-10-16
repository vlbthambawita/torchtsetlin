"""Source TMs (spec §7, PLAN Phase 7): train a flat or convolutional TM on the training partition,
select the epoch by validation accuracy (ties -> earlier epoch), save the checkpoint + record.

    python -m experiments.linearModel.code.source_tm --kind flat --M 20 --T 20 --s 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import torchtsetlin as tt

from .adapter import git_rev, snapshot_identity, src_diff_stat
from .data import EXP, load_split, read_manifest

ART = EXP / "artifacts" / "source"
RES = EXP / "results" / "source"


def source_id(kind, M, T, s, seed, threshold, manifest, weighted=False, epochs=30):
    w = "_w" if weighted else ""
    return f"{kind}{w}_M{M}_T{T:g}_s{s:g}_thr{threshold:g}_seed{seed}_{manifest}_e{epochs}"


def build(kind: str, M: int, T: float, s: float, weighted: bool = False, patch: int = 10):
    if kind == "flat":
        return tt.TsetlinMachine(784, 10, n_clauses=M, T=T, s=s, weighted=weighted)
    if kind == "conv":
        return tt.ConvTsetlinMachine(10, M, T=T, s=s, patch_size=patch, stride=1, position_encoding=True,
                                     input_shape=(1, 28, 28), weighted=weighted)
    raise ValueError(kind)


def bool_view(X: np.ndarray, threshold: float, kind: str, device) -> torch.Tensor:
    B = torch.as_tensor(X, dtype=torch.float32) > threshold          # == tt.data.Binarizer(threshold)
    if kind == "conv":
        B = B.reshape(-1, 1, 28, 28)
    return B.to(device)


@torch.no_grad()
def accuracy(model, B: torch.Tensor, y: torch.Tensor, bs: int = 2048) -> float:
    model.eval()
    pred = torch.cat([model.predict(B[i:i + bs]) for i in range(0, B.shape[0], bs)])
    return float((pred.cpu() == y.cpu()).float().mean())


def train(kind: str, M: int, T: float, s: float, seed: int, threshold: float = 0.5,
          manifest: str = "split_seed0", epochs: int = 30, batch: int = 32, weighted: bool = False,
          device: str = "cuda", save: bool = True, patience: int = None, log=print) -> dict:
    sid = source_id(kind, M, T, s, seed, threshold, manifest, weighted, epochs)
    tr, va = load_split(manifest, "train"), load_split(manifest, "valid")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(kind, M, T, s, weighted).to(device)
    Btr, Bva = bool_view(tr.X, threshold, kind, device), bool_view(va.X, threshold, kind, device)
    ytr, yva = torch.as_tensor(tr.y, device=device), torch.as_tensor(va.y)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hist, best, best_state, best_epoch = [], -1.0, None, 0
    t_fit = 0.0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(ytr), generator=gen).to(device)
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            model.update(Btr[idx], ytr[idx])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fit += time.perf_counter() - t0
        acc = accuracy(model, Bva, yva)
        hist.append(dict(epoch=ep, val_acc=acc, seconds=time.perf_counter() - t0))
        if acc > best:                                        # strict: ties keep the earlier epoch
            best, best_epoch = acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"[{sid}] epoch {ep} val_acc {acc:.4f} best {best:.4f}@{best_epoch}")
        if patience and ep - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    rec = dict(source_id=sid, kind=kind, M=M, T=T, s=s, seed=seed, threshold=threshold, weighted=weighted,
               manifest=manifest, split_sha=read_manifest(manifest)["sha256"], epochs=epochs, batch=batch,
               feedback_mode=model.feedback_mode, n_states=model.n_states, best_epoch=best_epoch,
               val_acc=best, history=hist, fit_seconds=t_fit, identity=snapshot_identity(model),
               n_included_literals=int(model.included_mask().sum()),
               n_empty_clauses=int((model.included_mask().sum(1) == 0).sum()),
               device=torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
               git_rev=git_rev(), src_diff=src_diff_stat(), torchtsetlin=tt.__version__)
    if kind == "conv":
        rec.update(patch=list(model.patch_size), stride=list(model.stride), position_encoding=model.position_encoding)
    if save:
        ART.mkdir(parents=True, exist_ok=True)
        RES.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state=best_state, record=rec), ART / f"{sid}.pt")
        (RES / f"{sid}.json").write_text(json.dumps(rec, indent=1))
    return dict(model=model, record=rec)


def load(sid: str, device: str = "cpu"):
    ck = torch.load(ART / f"{sid}.pt", map_location=device)
    r = ck["record"]
    m = build(r["kind"], r["M"], r["T"], r["s"], r["weighted"])
    m.load_state_dict(ck["state"])
    m.to(device).eval()
    if snapshot_identity(m) != r["identity"]:
        raise RuntimeError(f"source checkpoint {sid} identity mismatch")
    return m, r


def exists(sid: str) -> bool:
    return (RES / f"{sid}.json").exists() and (ART / f"{sid}.pt").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="flat")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--T", type=float, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--manifest", default="split_seed0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = train(a.kind, a.M, a.T, a.s, a.seed, a.threshold, a.manifest, a.epochs, weighted=a.weighted, device=a.device)
    print(json.dumps({k: v for k, v in out["record"].items() if k != "history"}))


if __name__ == "__main__":
    main()
