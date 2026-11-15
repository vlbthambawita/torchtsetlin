"""CIFAR-10: normal Tsetlin machines vs the pair-clause attention TM (REPORT_CIFAR10.md).

Data: the thermometer-4 Booleanization cached by ``experiments/convtm`` (12 x 32 x 32 bits:
3 channels x 4 thresholds), read-only, with the same stratified 45,000 / 5,000 split
(seed 1234, 500 images per class to validation) so that numbers sit next to the convtm ones.
The official 10,000-image test set is read once per final run and logged.

Arms (all class-wise, 200 clauses per class = 2,000 total, T = 80, unweighted, soft literal
budget 32, batch 32, the ``ctm-small`` configuration of experiments/convtm):

- ``flat``      ``TsetlinMachine`` on all 12,288 bits.
- ``ctm``       ``ConvTsetlinMachine``, 4 x 4 patches, stride 1, position encoding (841 patches).
- ``pair``      attention TM. Tokens = the 64 non-overlapping 4 x 4 patches (192 bits each).
                One column per ordered token pair (i, j), 4,096 columns:
                [token_i | token_j | therm(dy) | therm(dx) | therm(row_i) | therm(col_i)],
                dy, dx = relative offset of the sender j from the receiver i (14 bits each),
                row_i, col_i = receiver grid position (7 bits each). F = 426.
                ``Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)`` ORs each clause
                over pairs: a clause is Q(token_i) AND K(token_j) AND R(dy, dx) AND A(i).
- ``pair_local`` same columns, senders restricted to |dy|, |dx| <= 1 (576 pairs).
- ``pair_self``  same columns, self pairs only (i = j, 64 columns): the no-attention ablation
                 (a non-overlapping patch TM with the same token and position bits).
- ``qkv``        explicit Q/K/V attention: depth-2 graph TM over the 64 patch tokens (see
                 ``TokenGraphs``); ``qkv_d1`` is its depth-1 (no message passing) control.

Usage:
    python -m experiments.transformer_concepts.code.cifar time            # throughput probe
    python -m experiments.transformer_concepts.code.cifar run ARM S SEED [--epochs E] [--test]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import torchtsetlin as tt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results" / "cifar"
CACHE = ROOT.parents[0] / "convtm" / ".cache" / "cifar10_therm4.pt"
SPLIT_SEED, N_VAL, N_CLASSES = 1234, 5000, 10
G, PS = 8, 4  # token grid 8 x 8 of 4 x 4 patches

ARMS_TIMED = None
COMMON = dict(n_per_class=200, T=80.0, max_included_literals=32, batch=32)


# ----------------------------------------------------------------------------- data
def load(device):
    d = torch.load(CACHE, weights_only=False)
    y = d["ytr"]
    g = torch.Generator().manual_seed(SPLIT_SEED)  # same rule as experiments/convtm/code/data.py
    val, trn = [], []
    for c in range(N_CLASSES):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        val.append(idx[:N_VAL // N_CLASSES])
        trn.append(idx[N_VAL // N_CLASSES:])
    val, trn = torch.cat(val).sort().values, torch.cat(trn).sort().values
    return dict(xtr=d["xtr"][trn].to(device), ytr=y[trn].to(device),
                xva=d["xtr"][val].to(device), yva=y[val].to(device),
                xte=d["xte"], yte=d["yte"])


def tokens(x):
    """(B, 12, 32, 32) -> (B, 64, 192): non-overlapping 4 x 4 patches, row-major."""
    B, Z = x.shape[:2]
    t = x.view(B, Z, G, PS, G, PS).permute(0, 2, 4, 1, 3, 5)
    return t.reshape(B, G * G, Z * PS * PS)


def _therm(v, lo, hi):
    return v.unsqueeze(-1) > torch.arange(lo, hi, device=v.device)


class PairCols:
    """Gathers pair columns (B, F, P) for a set of (receiver, sender) token pairs."""

    def __init__(self, mode, device):
        ii, jj = torch.meshgrid(torch.arange(G * G), torch.arange(G * G), indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        ri, ci, rj, cj = ii // G, ii % G, jj // G, jj % G
        dy, dx = rj - ri, cj - ci
        keep = {"pair": torch.ones_like(ii, dtype=torch.bool),
                "pair_local": (dy.abs() <= 1) & (dx.abs() <= 1),
                "pair_self": ii == jj}[mode]
        ii, jj, dy, dx, ri, ci = (t[keep] for t in (ii, jj, dy, dx, ri, ci))
        pos = torch.cat([_therm(dy, -(G - 1), G - 1), _therm(dx, -(G - 1), G - 1),
                         _therm(ri, 0, G - 1), _therm(ci, 0, G - 1)], 1)  # (P, 42)
        self.ii, self.jj = ii.to(device), jj.to(device)
        self.pos = pos.t().contiguous().to(device)  # (42, P)
        self.P = ii.numel()

    def __call__(self, x):
        t = tokens(x)  # (B, 64, 192)
        B = t.shape[0]
        q = t[:, self.ii].transpose(1, 2)
        k = t[:, self.jj].transpose(1, 2)
        return torch.cat([q, k, self.pos.unsqueeze(0).expand(B, -1, -1)], 1)


N_EDGE_TYPES = 16


class TokenGraphs:
    """Q/K/V attention as a depth-2 graph TM (arm ``qkv``). One node per 4 x 4 patch
    (192 patch bits + 14 thermometer position bits), an edge j -> i for every pair j != i,
    typed by the direction of j from i (8 compass sectors) x near (Chebyshev distance 1) / far.

    Layer 1 (``H^0`` part of every clause) runs on every node; its matches at node j are
    the *values* V_j, sent as hashed message bits bound to the edge type (the position part of
    the *key*). The inbox of node i is the OR over its senders (hard attention). Layer 2 is
    the clause's AND of its own-node literals (the *query* Q_i) and inbox literals (which
    values arrived, from which direction), so one clause can read several senders at once.
    Depth 1 (arm ``qkv_d1``) is the same machine without message passing."""

    def __init__(self, device):
        ii, jj = torch.meshgrid(torch.arange(G * G), torch.arange(G * G), indexing="ij")
        ii, jj = ii.reshape(-1), jj.reshape(-1)
        keep = ii != jj
        dst, src = ii[keep], jj[keep]
        dy, dx = src // G - dst // G, src % G - dst % G
        sector = {(-1, -1): 0, (-1, 0): 1, (-1, 1): 2, (0, 1): 3, (1, 1): 4, (1, 0): 5, (1, -1): 6, (0, -1): 7}
        sec = torch.tensor([sector[(int(a), int(b))] for a, b in zip(dy.sign(), dx.sign())])
        far = (torch.maximum(dy.abs(), dx.abs()) > 1).long()
        self.edges = torch.stack([src, dst], 1).to(device)  # (E, 2) source -> destination
        self.etype = (sec * 2 + far).to(device)
        self.E = self.edges.shape[0]
        r = torch.arange(G * G)
        self.pos = torch.cat([_therm(r // G, 0, G - 1), _therm(r % G, 0, G - 1)], 1).to(device)  # (64, 14)
        self.P = G * G

    def __call__(self, x):
        t = tokens(x)
        B = t.shape[0]
        feats = torch.cat([t, self.pos.unsqueeze(0).expand(B, -1, -1)], 2)
        gb = tt.data.GraphBatch(feats, torch.full((B,), G * G, device=x.device), self.edges.repeat(B, 1),
                                self.etype.repeat(B), torch.full((B,), self.E, device=x.device),
                                n_edge_types=N_EDGE_TYPES).pad(G * G, self.E)
        gb.full_nodes = True
        return gb


def build(arm, s, device):
    kw = dict(max_included_literals=COMMON["max_included_literals"])
    M, T = COMMON["n_per_class"], COMMON["T"]
    if arm in ("qkv", "qkv_d1"):
        # coalesced: 2,000 shared clauses with per-class weights (the graph TM's output layer)
        m = tt.GraphTsetlinMachine(None, N_CLASSES, M * N_CLASSES, T, s, depth=2 if arm == "qkv" else 1,
                                   message_size=1024, message_bits=2, n_edge_types=N_EDGE_TYPES, **kw)
        return m.to(device), TokenGraphs(device)
    if arm == "flat":
        return tt.TsetlinMachine(12 * 32 * 32, N_CLASSES, M, T, s, **kw).to(device), lambda x: x.reshape(x.shape[0], -1)
    if arm == "ctm":
        m = tt.ConvTsetlinMachine(N_CLASSES, M, T, s, patch_size=4, stride=1, position_encoding=True,
                                  input_shape=(12, 32, 32), **kw)
        return m.to(device), lambda x: x
    pc = PairCols(arm, device)
    m = tt.Conv1dTsetlinMachine(N_CLASSES, M, T, s, kernel_size=1, position_encoding=False, **kw)
    return m.to(device), pc


@torch.no_grad()
def predict(model, feats, x, bs=128):
    model.eval()
    out = []
    for k in range(0, x.shape[0], bs):
        out.append(model.predict(feats(x[k:k + bs].to(model.ta_state.device))))
    return torch.cat(out)


def run(arm, s, seed, epochs=30, test=False, n_train=None, tag="", device="cuda"):
    key = f"{arm}_s{s:g}_seed{seed}_e{epochs}{('_n' + str(n_train)) if n_train else ''}{tag}"
    path = RESULTS / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    RESULTS.mkdir(parents=True, exist_ok=True)
    D = load(device)
    xtr, ytr = D["xtr"], D["ytr"]
    if n_train:
        xtr, ytr = xtr[:n_train], ytr[:n_train]
    tt.seed_everything(seed)
    model, feats = build(arm, s, device)
    g = torch.Generator().manual_seed(seed)
    curve, best, best_ep, best_state, t_train, t_all = [], -1.0, -1, None, 0.0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for k in range(0, len(xtr), COMMON["batch"]):
            idx = perm[k:k + COMMON["batch"]]
            model.update(feats(xtr[idx]), ytr[idx])
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        acc = (predict(model, feats, D["xva"]) == D["yva"]).float().mean().item()
        curve.append(round(acc, 5))
        print(f"{key} epoch {ep} val {acc:.4f} train_s {t_train:.0f}", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    rec = dict(key=key, arm=arm, s=s, seed=seed, epochs=epochs, **COMMON, n_train=len(xtr),
               n_valid=len(D["xva"]), booleanization="therm4 (experiments/convtm cache)",
               split=f"stratified seed {SPLIT_SEED}, 45000/5000", val_curve=curve, best_val=best,
               best_epoch=best_ep, train_seconds=round(t_train, 1), wall_seconds=round(time.time() - t_all, 1),
               gpu=torch.cuda.get_device_name(), torchtsetlin=tt.__version__,
               n_columns=getattr(feats, "P", None), n_clauses_total=int(model.n_clauses_total))
    if test:
        pred = predict(model, feats, D["xte"]).cpu()
        with open(ROOT / "results" / "test_access_log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "record": f"cifar/{key}",
                                "n": len(pred)}) + "\n")
        rec["test_acc"] = (pred == D["yte"]).float().mean().item()
        rec["test_acc_per_class"] = [((pred == D["yte"]) & (D["yte"] == c)).sum().item() / (D["yte"] == c).sum().item()
                                     for c in range(N_CLASSES)]
        np.save(RESULTS / f"{key}.pred.npy", pred.numpy().astype(np.int8))
        torch.save(model.state_dict(), RESULTS / f"{key}.pt")
    path.write_text(json.dumps(rec, indent=1))
    return rec


def time_probe(device="cuda"):
    D = load(device)
    out = {}
    for arm in (ARMS_TIMED or ("flat", "ctm", "pair_self", "pair_local", "pair")):
        tt.seed_everything(0)
        model, feats = build(arm, 10.0, device)
        model.train()
        for k in range(8):  # warm-up incl. CUDA-graph capture
            model.update(feats(D["xtr"][k * 32:(k + 1) * 32]), D["ytr"][k * 32:(k + 1) * 32])
        torch.cuda.synchronize()
        t0, n = time.perf_counter(), 0
        for k in range(8, 68):
            model.update(feats(D["xtr"][k * 32:(k + 1) * 32]), D["ytr"][k * 32:(k + 1) * 32])
            n += 32
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        t1 = time.perf_counter()
        predict(model, feats, D["xva"][:1024])
        torch.cuda.synchronize()
        out[arm] = dict(train_img_per_s=round(n / dt, 1), epoch_min=round(45000 / (n / dt) / 60, 2),
                        predict_img_per_s=round(1024 / (time.perf_counter() - t1), 1))
        print(arm, out[arm], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / ("timing_probe.json" if ARMS_TIMED is None else "timing_probe_qkv.json")).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["time", "run"])
    ap.add_argument("arm", nargs="?")
    ap.add_argument("s", nargs="?", type=float)
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.cmd == "time":
        time_probe()
    else:
        r = run(a.arm, a.s, a.seed, a.epochs, a.test, a.n_train)
        print(json.dumps({k: r[k] for k in ("key", "best_val", "best_epoch", "train_seconds")} |
                         ({"test_acc": r["test_acc"]} if "test_acc" in r else {})))


if __name__ == "__main__":
    main()
