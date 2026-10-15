"""Run one arm on one split: returns probabilities on every split plus an :class:`Explainable`."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from . import models as M
from . import train_nn
from .data import TabularSplits
from .explain import Explainable
from .tm_source import (
    ClauseBasis,
    basis_from_tm,
    get_tm,
    random_basis,
    random_basis_firing_matched,
    tm_config,
    tm_scores,
)

NEURAL_ARMS = ("tabresnet", "imn", "imn_tm", "imn_tm_rand", "imn_hybrid", "tm_gated", "clause_lr",
               "imn_tm_xc", "imn_tm_randfr")
# a run whose model would exceed this many parameters is recorded as skipped (Tier 2, D24)
MAX_PARAMS = 40_000_000


class SkipArm(Exception):
    """The configuration is infeasible under the preregistered budget; recorded, not run."""
SKLEARN_ARMS = ("lr", "dt", "rf", "gbdt")
ALL_ARMS = SKLEARN_ARMS + ("tm",) + NEURAL_ARMS[:7]  # 'all' = the 12 main arms


@dataclass
class ArmResult:
    proba: Dict[str, np.ndarray]            # split -> (N,) or (N, C)
    explain: Optional[Explainable]           # bound to X_test[explain_rows] (local models), or full model
    info: Dict = field(default_factory=dict)


def _sk_model(arm: str, n_classes: int, seed: int, params: Optional[Dict] = None):
    params = dict(params or {})
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    if arm == "lr":
        return LogisticRegression(**{"max_iter": 1000, **params}, random_state=seed), "sklearn.LogisticRegression"
    if arm == "dt":
        return DecisionTreeClassifier(**params, random_state=seed), "sklearn.DecisionTree"
    if arm == "rf":
        return (RandomForestClassifier(**{"n_estimators": 500, **params}, n_jobs=int(__import__("os").environ.get("OMP_NUM_THREADS", 8)), random_state=seed),
                "sklearn.RandomForest")
    try:  # CatBoost as in the paper when installed; sklearn's HGB otherwise (D8)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, random_seed=seed, verbose=False, thread_count=int(__import__("os").environ.get("OMP_NUM_THREADS", 8))), "catboost"
    except ImportError:
        return HistGradientBoostingClassifier(**params, random_state=seed), "sklearn.HistGradientBoosting"


def run_sklearn(arm: str, S: TabularSplits, seed: int, device: str, params: Optional[Dict] = None) -> ArmResult:
    """``params``: estimator hyper-parameters (passed through ``--nn`` for sklearn arms)."""
    model, impl = _sk_model(arm, S.n_classes, seed, params)
    t0 = time.time()
    model.fit(S.X_train, S.y_train)
    train_time = time.time() - t0

    def proba(X):
        p = model.predict_proba(X)
        return p[:, 1] if S.n_classes == 2 else p

    t1 = time.time()
    out = {k: proba(getattr(S, f"X_{k}")) for k in ("train", "valid", "test")}
    infer = time.time() - t1

    def full(X: Tensor) -> Tensor:
        p = np.clip(model.predict_proba(X.cpu().numpy()), 1e-7, 1 - 1e-7)
        z = np.log(p[:, 1:2] / p[:, :1]) if S.n_classes == 2 else np.log(p)
        return torch.as_tensor(z, dtype=torch.float32, device=X.device)

    return ArmResult(out, Explainable(full), {"impl": impl, "params": params or {}, "train_time": train_time,
                                              "inference_time": infer})


def run_tm(S: TabularSplits, seed: int, device: str, tm_over: Optional[Dict], log=print) -> ArmResult:
    cfg = tm_config(tm_over)
    tm, boo, meta, key = get_tm(S, cfg, seed, device, log)
    dev = torch.device(device)
    out = {}
    t1 = time.time()
    rows = max(1, min(8192, 30_000_000 // (2 * boo.n_bool)))  # bound the literal matrix on wide inputs
    for k in ("train", "valid", "test"):
        Xk = torch.as_tensor(getattr(S, f"X_{k}"), device=dev)
        p = np.concatenate([tm_scores(tm, boo.transform(Xk[i:i + rows]), cfg["prob_temperature"])
                            for i in range(0, Xk.shape[0], rows)])
        out[k] = p[:, 1] if S.n_classes == 2 else p
    infer = time.time() - t1

    def full(X: Tensor) -> Tensor:
        v = torch.cat([tm(boo.transform(X[i:i + rows])) for i in range(0, X.shape[0], rows)]) / tm.T
        return (v[:, 1:2] - v[:, :1]) if S.n_classes == 2 else v

    basis = basis_from_tm(tm, boo, S.n_classes)
    info = {"tm_key": key, "tm_cfg": cfg, "train_time": meta["train_time"], "best_epoch": meta["best_epoch"],
            "T": meta["T"], "n_bool": boo.n_bool, "n_clauses_raw": basis.n_raw_clauses,
            "n_clauses_basis": basis.K, "clause_len_mean": float(basis.lengths().float().mean()),
            "inference_time": infer}
    return ArmResult(out, Explainable(full), info)


def run_neural(arm: str, S: TabularSplits, seed: int, device: str, nn_over: Optional[Dict],
               tm_over: Optional[Dict], log=print, explain_rows: Optional[np.ndarray] = None) -> ArmResult:
    from torchtsetlin import seed_everything
    cfg = train_nn.nn_config(nn_over)
    dev = torch.device(device)
    X = {k: torch.as_tensor(getattr(S, f"X_{k}"), device=dev) for k in ("train", "valid", "test")}
    y_tr = torch.as_tensor(S.y_train, device=dev)
    n_out = 1 if S.n_classes == 2 else S.n_classes
    info: Dict = {"nn_cfg": cfg}
    basis: Optional[ClauseBasis] = None
    C = {k: None for k in X}
    clause_init = None
    if M.needs_clauses(arm):
        tcfg = tm_config(tm_over)
        tm, boo, meta, key = get_tm(S, tcfg, seed, device, log)
        basis = basis_from_tm(tm, boo, S.n_classes)
        if arm == "imn_tm_rand":
            basis = random_basis(basis, S.X_train, seed)
        elif arm == "imn_tm_randfr":
            basis = random_basis_firing_matched(basis, S.X_train, seed)
        C = {k: basis.evaluate(X[k]).bool() for k in X}
        if cfg["tm_init"] and basis.source == "tm" and arm != "tm_gated":
            clause_init = basis.tm_weight * (cfg["tm_init_scale"] / basis.T)
        info.update({"tm_key": key, "tm_cfg": tcfg, "basis": basis.source, "n_clauses_basis": basis.K,
                     "clause_len_mean": float(basis.lengths().float().mean()),
                     "train_clause_firing_rate": float(C["train"].float().mean()),
                     "train_examples_no_active_clause": float((~C["train"].any(1)).float().mean())})
    seed_everything(seed)
    model = M.build(arm, S.n_features, n_out, K=0 if basis is None else basis.K, cfg=cfg,
                    clause_init=clause_init)
    info["n_params"] = int(sum(p.numel() for p in model.parameters()))
    if info["n_params"] > MAX_PARAMS:
        raise SkipArm(f"{info['n_params']} parameters > MAX_PARAMS={MAX_PARAMS}")
    model = model.to(dev)
    fit = train_nn.fit(model, train_nn.Batches(X["train"], y_tr, C["train"]), cfg, log)
    snaps = fit["snapshots"]
    info.update({"train_time": fit["train_time"], "epochs": fit["epochs"], "steps": fit["steps"],
                 "n_snapshots": len(snaps), "history": fit["history"]})
    out = {}
    t1 = time.time()
    for k in X:
        z = train_nn.ensemble_forward(model, snaps, X[k], C[k])
        out[k] = train_nn.logits_to_proba(z)
    info["inference_time"] = time.time() - t1

    def clause_fn(Xq: Tensor) -> Tensor:
        return basis.evaluate(Xq)

    def full(Xq: Tensor) -> Tensor:
        return train_nn.ensemble_forward(model, snaps, Xq, None if basis is None else clause_fn(Xq))

    if not getattr(model, "interpretable", False):
        return ArmResult(out, Explainable(full), info)

    # generated local models only for the explained test rows (memory: N x P x n_out)
    if explain_rows is None:
        explain_rows = np.arange(len(S.X_test))
    er = torch.as_tensor(explain_rows, device=dev)
    _, W_test = train_nn.ensemble_forward(model, snaps, X["test"][er], None if C["test"] is None else C["test"][er],
                                          return_w=True)
    uses_local_clauses = model.K > 0

    # rows per sub-batch so the gathered (b, P, K) weights stay ~64 MB (wide TMs with many
    # classes otherwise OOM inside the removal-attribution loop)
    local_chunk = max(1, int(16_000_000 // (W_test.shape[1] * W_test.shape[2])))

    def local(Xmod: Tensor, rows: Tensor) -> Tensor:
        out_l = []
        for i in range(0, Xmod.shape[0], local_chunk):
            xm, r = Xmod[i:i + local_chunk], rows[i:i + local_chunk]
            c = clause_fn(xm) if uses_local_clauses else None
            out_l.append(torch.einsum("bp,bpk->bk", model.local_features(xm, c), W_test[r]))
        return torch.cat(out_l)

    def parts(rows: Tensor):
        return model.split(W_test[rows])

    info["basis_obj"] = basis
    return ArmResult(out, Explainable(full, local, parts, clause_fn if basis is not None else None), info)


def run_arm(arm: str, S: TabularSplits, seed: int, device: str, nn_over=None, tm_over=None, log=print,
            explain_rows: Optional[np.ndarray] = None) -> ArmResult:
    """``explain_rows``: test rows whose local models are kept (positions index the returned
    ``Explainable``'s ``local``/``parts``); default all test rows."""
    if arm in SKLEARN_ARMS:
        return run_sklearn(arm, S, seed, device, nn_over)
    if arm == "tm":
        return run_tm(S, seed, device, tm_over, log)
    if arm in NEURAL_ARMS:
        return run_neural(arm, S, seed, device, nn_over, tm_over, log, explain_rows)
    raise ValueError(f"unknown arm {arm!r}; choose from {ALL_ARMS}")
