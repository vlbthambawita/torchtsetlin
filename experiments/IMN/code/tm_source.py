"""Source Tsetlin machine (the ``tm`` arm) and the frozen clause basis built from it.

Uses only the public ``torchtsetlin`` API (``src/`` is read-only for this programme, C1).
The same trained TM is cached per (dataset, seed, TM config) and shared by every arm that
builds on its clauses (``tm``, ``clause_lr``, ``imn_tm``, ``imn_hybrid``, ``tm_gated``), so the
comparisons between those arms are on *identical* clauses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

import torchtsetlin as tt
from torchtsetlin import functional as TF

from .booleanize import TabularBooleanizer
from .data import EXP_DIR, TabularSplits
from .metrics import classification_metrics

CKPT_DIR = EXP_DIR / ".ckpt" / "tm"

TM_DEFAULTS: Dict = {
    "n_bits": 10,              # thermometer thresholds per column
    "n_clauses": 200,          # per class (half positive, half negative polarity)
    "T_ratio": 0.5,            # T = T_ratio * n_clauses
    "s": 5.0,
    "weighted": True,
    "n_states": 128,
    "max_included_literals": None,
    "batch_size": 32,          # batch feedback: 10-50 matches sequential (docs/benchmarks.md)
    "epochs": 60,
    "patience": 15,            # epochs without validation AUROC improvement
    "prob_temperature": 1.0,   # softmax(votes / T / temperature) for AUROC / log-loss
}


def tm_config(overrides: Optional[Dict] = None) -> Dict:
    cfg = dict(TM_DEFAULTS)
    cfg.update(overrides or {})
    return cfg


def _cfg_hash(cfg: Dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def tm_scores(tm: tt.TsetlinMachine, xb: Tensor, temperature: float = 1.0) -> np.ndarray:
    """Class probabilities from clamped vote sums: ``softmax(v / (T * temperature))``."""
    tm.eval()
    votes = tm(xb)
    return TF.predict_proba_from_votes(votes, tm.T, temperature).cpu().numpy()


def train_tm(splits: TabularSplits, cfg: Dict, seed: int, device: str, log=print):
    """Train a TM with validation-AUROC early stopping. Returns (tm, booleanizer, history)."""
    tt.seed_everything(seed)
    boo = TabularBooleanizer(cfg["n_bits"]).fit(splits.X_train)
    dev = torch.device(device)
    xtr = boo.transform(torch.as_tensor(splits.X_train, device=dev))
    xva = boo.transform(torch.as_tensor(splits.X_valid, device=dev))
    ytr = torch.as_tensor(splits.y_train, device=dev)
    T = max(1.0, round(cfg["T_ratio"] * cfg["n_clauses"]))
    tm = tt.TsetlinMachine(
        boo.n_bool, splits.n_classes, n_clauses=cfg["n_clauses"], T=T, s=cfg["s"],
        weighted=cfg["weighted"], n_states=cfg["n_states"],
        max_included_literals=cfg["max_included_literals"],
    ).to(dev)
    best, best_state, best_epoch, hist = -np.inf, None, -1, []
    bs, n = cfg["batch_size"], xtr.shape[0]
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        tm.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            tm.update(xtr[idx], ytr[idx])
        p = tm_scores(tm, xva, cfg["prob_temperature"])
        m = classification_metrics(splits.y_valid, p, splits.n_classes)
        hist.append({"epoch": epoch, "valid_auroc": m["auroc"], "valid_acc": m["accuracy"],
                     "time": time.time() - t0})
        if m["auroc"] > best + 1e-6:
            best, best_epoch = m["auroc"], epoch
            best_state = copy.deepcopy(tm.state_dict())
        if epoch - best_epoch >= cfg["patience"]:
            break
    tm.load_state_dict(best_state)
    tm.eval()
    log(f"  tm: best valid AUROC {best:.4f} at epoch {best_epoch} ({time.time() - t0:.1f}s)")
    return tm, boo, {"history": hist, "best_epoch": best_epoch, "train_time": time.time() - t0, "T": T}


def _load_tm(path, splits: TabularSplits, cfg: Dict, device: str, fp: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["data_fingerprint"] != fp:
        return None
    boo = TabularBooleanizer.from_state(ck["booleanizer"])
    tm = tt.TsetlinMachine(boo.n_bool, splits.n_classes, n_clauses=cfg["n_clauses"],
                           T=ck["meta"]["T"], s=cfg["s"], weighted=cfg["weighted"],
                           n_states=cfg["n_states"],
                           max_included_literals=cfg["max_included_literals"]).to(device)
    tm.load_state_dict(ck["state_dict"])
    tm.eval()
    return tm, boo, ck["meta"]


def get_tm(splits: TabularSplits, cfg: Dict, seed: int, device: str, log=print):
    """Load the cached source TM for (dataset, split seed, cfg) or train and cache it.

    Concurrent workers serialise on a per-key file lock, so a source TM is trained exactly
    once and every clause arm of that (dataset, seed, cfg) uses the identical clauses; the
    checkpoint is written to a temporary file and renamed into place (atomic).
    """
    import fcntl
    import os
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{splits.name}_d{splits.dataset_id}_s{seed}_{_cfg_hash(cfg)}"
    path = CKPT_DIR / f"{key}.pt"
    fp = splits.fingerprint()
    with open(CKPT_DIR / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            got = _load_tm(path, splits, cfg, device, fp)
            if got is not None:
                return (*got, key)
            log(f"  tm cache {path.name} has a different data fingerprint; retraining")
        tm, boo, meta = train_tm(splits, cfg, seed, device, log)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        torch.save({"state_dict": tm.state_dict(), "booleanizer": boo.state(), "meta": meta,
                    "cfg": cfg, "data_fingerprint": fp}, tmp)
        os.replace(tmp, path)
        return tm, boo, meta, key


# ---------------------------------------------------------------------------- clause basis
@dataclass
class ClauseBasis:
    """A frozen set of conjunctions over Booleanized features.

    ``include``: ``(K, 2F)`` bool (first F = ``x_b``, last F = ``NOT x_b``);
    ``tm_weight``: ``(K, n_out)`` float, the source TM's signed vote of each clause on each
    output (``n_out = 1`` for binary tasks: vote for class 1 minus vote for class 0), used to
    initialise the clause-weighting models at the TM; ``T`` the TM margin.
    """

    include: Tensor
    tm_weight: Tensor
    T: float
    booleanizer: TabularBooleanizer
    source: str
    n_raw_clauses: int

    @property
    def K(self) -> int:
        return int(self.include.shape[0])

    def to(self, device) -> ClauseBasis:
        self.include = self.include.to(device)
        self.tm_weight = self.tm_weight.to(device)
        self._incf = self.include.float()
        self._cnt = self.include.sum(1)
        return self

    def evaluate_bool(self, xb: Tensor) -> Tensor:
        """``(B, n_bool)`` bool -> ``(B, K)`` float clause outputs (empty clauses are False)."""
        lits = TF.to_literals(xb)
        return TF.clause_outputs(lits, self._incf, self._cnt, empty_value=False).float()

    def evaluate(self, X: Tensor, chunk: int = 8192) -> Tensor:
        """Preprocessed float features ``(B, d)`` -> ``(B, K)`` clause outputs."""
        # wide inputs (e.g. 4k features -> 40k bits): keep each chunk's literal matrix ~30M elements
        chunk = max(1, min(chunk, 30_000_000 // (2 * self.booleanizer.n_bool + self.K)))
        outs = [self.evaluate_bool(self.booleanizer.transform(X[i:i + chunk]))
                for i in range(0, X.shape[0], chunk)]
        return torch.cat(outs, 0)

    def lengths(self) -> Tensor:
        return self.include.sum(1)

    def describe(self, j: int, feature_names) -> str:
        names = self.booleanizer.bit_names(feature_names)
        F = len(names)
        inc = self.include[j].nonzero().flatten().tolist()
        lits = [names[i] if i < F else f"NOT {names[i - F]}" for i in inc]
        return " AND ".join(lits) if lits else "<empty>"


def basis_from_tm(tm: tt.TsetlinMachine, boo: TabularBooleanizer, n_classes: int) -> ClauseBasis:
    """Drop empty clauses, merge exact duplicates (summing their signed TM votes)."""
    inc = tm.included_mask().bool()
    C = inc.shape[0]
    signed = tm._signed_weights().float()  # polarity * weight (read-only use of a helper)
    cls = tm.clause_class
    if n_classes == 2:
        w = (signed * torch.where(cls == 1, 1.0, -1.0).to(signed)).unsqueeze(1)
    else:
        w = torch.zeros(C, n_classes, device=inc.device)
        w[torch.arange(C, device=inc.device), cls] = signed
    keep = inc.any(1)
    inc, w = inc[keep], w[keep]
    uniq, inv = torch.unique(inc, dim=0, return_inverse=True)
    wm = torch.zeros(uniq.shape[0], w.shape[1], device=w.device).index_add_(0, inv, w)
    return ClauseBasis(uniq, wm, float(tm.T), boo, "tm", C).to(inc.device)


def random_basis(like: ClauseBasis, X_train: np.ndarray, seed: int) -> ClauseBasis:
    """Control basis (``imn_tm_rand``): data-anchored random conjunctions.

    For each learned clause of length L, pick a random training example and L random
    Boolean features, and include the literal (``x_b`` or ``NOT x_b``) that is *true* for
    that example. Every random clause therefore fires on at least one training example and
    the length distribution matches the learned basis; what differs is that the literals
    were not chosen by TM feedback.
    """
    g = np.random.default_rng(seed)
    boo = like.booleanizer
    xb = boo.transform(X_train).numpy()
    F = xb.shape[1]
    lengths = like.lengths().cpu().numpy()
    inc = np.zeros((len(lengths), 2 * F), dtype=bool)
    for j, L in enumerate(lengths):
        r = xb[g.integers(len(xb))]
        bits = g.choice(F, size=min(int(L), F), replace=False)
        inc[j, bits[r[bits]]] = True
        inc[j, F + bits[~r[bits]]] = True
    dev = like.include.device
    return ClauseBasis(torch.as_tensor(inc, device=dev), torch.zeros_like(like.tm_weight),
                       like.T, boo, "random", len(lengths)).to(dev)


def literal_columns(basis: ClauseBasis) -> np.ndarray:
    """Source column of each of the 2F literals."""
    c = basis.booleanizer.col
    return np.concatenate([c, c])



def random_basis_firing_matched(like: ClauseBasis, X_train: np.ndarray, seed: int,
                                n_probe: int = 5000, max_len: int = 64) -> ClauseBasis:
    """Control basis (``imn_tm_randfr``): data-anchored random conjunctions whose *training
    firing rate* matches the corresponding learned clause.

    For learned clause ``j`` with firing rate ``r_j`` (on up to ``n_probe`` training rows), pick
    a random anchor row and add its true literals in random order until the rate drops to
    ``<= r_j`` (or ``max_len`` literals). Firing patterns, not literal counts, are matched.
    """
    g = np.random.default_rng(seed)
    boo = like.booleanizer
    xb = boo.transform(X_train).numpy()
    if len(xb) > n_probe:
        xb = xb[g.choice(len(xb), n_probe, replace=False)]
    F = xb.shape[1]
    target = like.evaluate_bool(torch.as_tensor(xb, device=like.include.device)).mean(0).cpu().numpy()
    inc = np.zeros((like.K, 2 * F), dtype=bool)
    for j, r_j in enumerate(target):
        r = xb[g.integers(len(xb))]
        fire = np.ones(len(xb), dtype=bool)
        for b in g.permutation(F)[:max_len]:
            if fire.mean() <= r_j:
                break
            if r[b]:
                inc[j, b] = True
                fire &= xb[:, b]
            else:
                inc[j, F + b] = True
                fire &= ~xb[:, b]
    dev = like.include.device
    return ClauseBasis(torch.as_tensor(inc, device=dev), torch.zeros_like(like.tm_weight),
                       like.T, boo, "random_fr", like.n_raw_clauses).to(dev)
