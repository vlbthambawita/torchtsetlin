"""MNIST study (``mnist/PLAN.md``): accuracy and explainability of IMN, IMN-TM and references.

Datasets (``--dataset``):

- ``mnist``: official 60k/10k split from ``<repo>/data/MNIST/raw`` (torchvision, no download),
  10k validation images held out of the training set by split seed, pixels / 255 (MD1).
- ``mnist_decoy``: the same with a one-hot "barcode" on row 0, columns 9-18 (always black in
  MNIST). Train/valid light the pixel of the true class with probability 0.9, else a random
  class's. The test split is *consistent* (true class); ``X_test_swapped`` lights a different
  random class (``swap_class``) and ``X_test_clean`` has no barcode.

One run writes ``results/<stage>/<name>/<arm>__s<seed>.json`` (scalars, rule C2) and an ``.npz``
next to it (showcase maps, per-image deletion AUCs) for the report's figures and paired tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from . import models as M
from . import train_nn
from .data import TabularSplits
from .explain import Explainable, _target, local_fidelity, local_model_summary, removal_attributions
from .metrics import classification_metrics

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"

N_VALID = 10_000
BARCODE = np.arange(9, 19)          # row 0, columns 9..18 -> flat pixel indices 9..18
DECOY_P = 0.9                       # probability that the barcode shows the true class
FRACS = np.linspace(0.0, 1.0, 21)   # deletion fractions of the ink pixels
IG_STEPS = 64
N_TOP_CLAUSES = 8

_MEMO: Dict = {}


# ------------------------------------------------------------------------------------- data
def _raw_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "raw" not in _MEMO:
        from torchvision.datasets import MNIST
        root = str(REPO / "data")
        tr, te = MNIST(root, train=True, download=False), MNIST(root, train=False, download=False)
        f = lambda d: (d.data.numpy().reshape(len(d), -1).astype(np.float32) / 255.0,  # noqa: E731
                       d.targets.numpy().astype(np.int64))
        _MEMO["raw"] = (*f(tr), *f(te))
    return _MEMO["raw"]


def _barcode(X: np.ndarray, cls: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[:, BARCODE] = 0.0
    X[np.arange(len(X)), BARCODE[cls]] = 1.0
    return X


def showcase_rows(y_test: np.ndarray) -> np.ndarray:
    """First test image of each class (fixed for every seed and arm)."""
    return np.asarray([int(np.flatnonzero(y_test == c)[0]) for c in range(10)])


def load_mnist(dataset: str, seed: int) -> TabularSplits:
    key = (dataset, seed)
    if key in _MEMO:
        return _MEMO[key]
    from sklearn.model_selection import train_test_split
    Xtr, ytr, Xte, yte = _raw_mnist()
    i_tr, i_va = train_test_split(np.arange(len(Xtr)), test_size=N_VALID, random_state=seed, stratify=ytr)
    names = [f"p{r}_{c}" for r in range(28) for c in range(28)]
    S = TabularSplits(dataset_id=554, name=dataset, X_train=Xtr[i_tr], y_train=ytr[i_tr],
                      X_valid=Xtr[i_va], y_valid=ytr[i_va], X_test=Xte, y_test=yte,
                      feature_names=names, categorical=[False] * 784, n_classes=10, split_seed=seed,
                      info={"source": "torchvision MNIST raw (official split)", "scale": "x/255"})
    if dataset == "mnist_decoy":
        g = np.random.default_rng(10_000 + seed)

        def noisy(y):
            c = y.copy()
            flip = g.random(len(y)) > DECOY_P
            c[flip] = g.integers(0, 10, flip.sum())
            return c
        S.X_train = _barcode(S.X_train, noisy(S.y_train))
        S.X_valid = _barcode(S.X_valid, noisy(S.y_valid))
        S.X_test_clean = Xte
        S.swap_class = (yte + g.integers(1, 10, len(yte))) % 10   # always != true class
        S.X_test_swapped = _barcode(Xte, S.swap_class)
        S.X_test = _barcode(Xte, yte)
    elif dataset != "mnist":
        raise ValueError(dataset)
    _MEMO[key] = S
    return S


# ------------------------------------------------------------------------ explanation binding
def bind_neural(model, snaps, basis, X: Tensor) -> Tuple[Explainable, Optional[Tensor]]:
    """Explainable whose local models are those generated for ``X`` (positions 0..N-1); the
    same construction as ``arms.run_neural``. Returns it and the generated weights."""
    clause_fn = None if basis is None else basis.evaluate

    def full(Xq: Tensor) -> Tensor:
        return train_nn.ensemble_forward(model, snaps, Xq, None if basis is None else clause_fn(Xq))

    if not getattr(model, "interpretable", False):
        return Explainable(full), None
    C = None if basis is None else clause_fn(X)
    _, W = train_nn.ensemble_forward(model, snaps, X, C, return_w=True)
    uses_clauses = model.K > 0
    local_chunk = max(1, int(16_000_000 // (W.shape[1] * W.shape[2])))

    def local(Xmod: Tensor, rows: Tensor) -> Tensor:
        out = []
        for i in range(0, Xmod.shape[0], local_chunk):
            xm, r = Xmod[i:i + local_chunk], rows[i:i + local_chunk]
            c = clause_fn(xm) if uses_clauses else None
            out.append(torch.einsum("bp,bpk->bk", model.local_features(xm, c), W[r]))
        return torch.cat(out)

    return Explainable(full, local, lambda rows: model.split(W[rows]), clause_fn), W


def literal_pixel_matrix(basis, d: int) -> Tensor:
    """``(K, d)`` count of clause ``j``'s literals that read pixel ``m`` (either polarity)."""
    col = torch.as_tensor(basis.booleanizer.col, device=basis.include.device)
    F = col.numel()
    inc = basis.include.float()
    A = torch.zeros(basis.K, d, device=inc.device)
    A.index_add_(1, col, inc[:, :F])
    A.index_add_(1, col, inc[:, F:])
    return A


@torch.no_grad()
def clause_credit(basis, X: Tensor, Wk: Tensor) -> Tensor:
    """``psi (N, d)``: each active clause spreads its weight ``Wk[n, j]`` evenly over its literals'
    pixels; ``psi.sum(1) == (Wk * c).sum(1)`` exactly."""
    c = basis.evaluate(X)
    A = literal_pixel_matrix(basis, X.shape[1])
    A = A / A.sum(1, keepdim=True).clamp_min(1)
    return (c * Wk) @ A


def integrated_gradients(model, snaps, X: Tensor, tgt: Tensor, steps: int = IG_STEPS,
                         rows_per_pass: int = 64) -> Tensor:
    """IG with a black baseline (midpoint Riemann sum), averaged over the snapshot ensemble."""
    alphas = (torch.arange(steps, device=X.device, dtype=X.dtype) + 0.5) / steps
    out = torch.zeros_like(X)
    for sd in snaps:
        model.load_state_dict(sd)
        model.eval()
        for i in range(0, X.shape[0], rows_per_pass):
            x, t = X[i:i + rows_per_pass], tgt[i:i + rows_per_pass]
            b = x.shape[0]
            xa = (alphas[:, None, None] * x[None]).reshape(steps * b, -1).requires_grad_(True)
            z = model(xa, None)[0]
            sel = z.gather(1, t.repeat(steps)[:, None]).sum()
            g, = torch.autograd.grad(sel, xa)
            out[i:i + b] += g.view(steps, b, -1).mean(0) * x
    return out / len(snaps)


# ------------------------------------------------------------------------------------ metrics
@torch.no_grad()
def deletion(full, X: Tensor, attr: Tensor, tgt: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MoRF pixel deletion: zero the top fraction ``f`` of the *ink* pixels ranked by ``attr``.
    Returns per-image probability-of-``tgt`` curves ``(N, len(FRACS))`` and their AUCs, and the
    scale-free *retention* curves (1 while the prediction is still ``tgt``) and their AUCs."""
    ink = X > 0
    key = torch.where(ink, attr, torch.full_like(attr, -float("inf")))
    order = key.argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(X.shape[1], device=X.device).expand_as(order))
    n_ink = ink.sum(1, keepdim=True)
    curves, keep = [], []
    for f in FRACS:
        drop = (rank < torch.round(f * n_ink)) & ink
        z = full(X.masked_fill(drop, 0.0))
        curves.append(torch.softmax(z, 1).gather(1, tgt[:, None])[:, 0])
        keep.append((z.argmax(1) == tgt).float())
    cur = torch.stack(curves, 1).cpu().numpy()
    ret = torch.stack(keep, 1).cpu().numpy()
    return cur, np.trapezoid(cur, FRACS, axis=1), ret, np.trapezoid(ret, FRACS, axis=1)


def _spearman_rows(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    from scipy.stats import spearmanr
    rs = []
    for i in range(len(a)):
        m = mask[i]
        if m.sum() >= 3 and np.ptp(a[i, m]) > 0 and np.ptp(b[i, m]) > 0:
            rs.append(spearmanr(a[i, m], b[i, m]).statistic)
    return float(np.mean(rs)) if rs else float("nan")


def _pred_metrics(full, X: Tensor, y: np.ndarray) -> Dict[str, float]:
    with torch.no_grad():
        p = torch.softmax(full(X), 1).cpu().numpy()
    return classification_metrics(y, p, 10)


# ------------------------------------------------------------------------------ explanations
def explain_all(arm: str, res, S: TabularSplits, X: Tensor, seed: int) -> Dict[str, Tensor]:
    """Every explanation of ``arm`` for the images ``X`` (``(N, 784)``); keys are method names,
    plus ``"_tgt"`` (explained class), ``"_ex"`` (the bound Explainable), ``"_W"``."""
    info = res.info
    model, snaps, basis = info.get("_model"), info.get("_snaps"), info.get("basis_obj")
    if model is not None:
        ex, W = bind_neural(model, snaps, basis, X)
    else:
        ex, W = res.explain, None
    with torch.no_grad():
        tgt = _target(ex.full(X))
    out: Dict = {"_tgt": tgt, "_ex": ex, "_W": W}
    out["removal"] = torch.as_tensor(removal_attributions(ex, X, use_local=ex.local is not None),
                                     device=X.device)
    rows = torch.arange(X.shape[0], device=X.device)
    if arm == "imn":
        w = ex.parts(rows)["linear"]                                   # (N, d, 10)
        out["weights"] = w.gather(2, tgt[:, None, None].expand(-1, w.shape[1], 1))[..., 0]
    if basis is not None and arm in ("imn_tm", "clause_lr", "tm"):
        if arm == "tm":
            Wk = (basis.tm_weight / basis.T)[:, tgt].T                  # (N, K) global votes
        else:
            Wc = ex.parts(rows)["clauses"]                              # (N, K, 10)
            Wk = Wc.gather(2, tgt[:, None, None].expand(-1, Wc.shape[1], 1))[..., 0]
        out["credit"] = clause_credit(basis, X, Wk)
        out["_Wk"] = Wk
    if arm == "tabresnet":
        out["ig"] = integrated_gradients(model, snaps, X, tgt)
    return out


def top_clauses(basis, X: Tensor, Wk: Tensor, n: int = N_TOP_CLAUSES) -> Dict[str, np.ndarray]:
    """For each image, the ``n`` active clauses with the largest ``|W|``: weights, lengths and
    their literals as pixel masks (positive = ``x >= t`` required, negative = ``x < t``)."""
    c = basis.evaluate(X)
    score = (c * Wk).abs()
    idx = score.topk(min(n, basis.K), 1).indices                       # (N, n)
    col = torch.as_tensor(basis.booleanizer.col, device=X.device)
    F = col.numel()
    pos = torch.zeros(X.shape[0], idx.shape[1], X.shape[1], device=X.device)
    neg = torch.zeros_like(pos)
    inc = basis.include[idx].float()                                    # (N, n, 2F)
    pos.index_add_(2, col, inc[..., :F])
    neg.index_add_(2, col, inc[..., F:])
    return {"clause_idx": idx.cpu().numpy(), "clause_w": Wk.gather(1, idx).cpu().numpy(),
            "clause_active": c.gather(1, idx).cpu().numpy(), "clause_len": basis.lengths()[idx].cpu().numpy(),
            "clause_pos": (pos > 0).cpu().numpy(), "clause_neg": (neg > 0).cpu().numpy()}


def random_model_explanations(arm: str, res, S: TabularSplits, X: Tensor, seed: int) -> Dict[str, Tensor]:
    """Model-randomisation sanity check: same architecture, untrained random parameters (and,
    for ``imn_tm``, random data-anchored clauses of matched lengths)."""
    from torchtsetlin import seed_everything

    from .tm_source import random_basis
    info = res.info
    basis = info.get("basis_obj")
    if arm == "imn_tm":
        basis = random_basis(basis, S.X_train, seed + 777)
    seed_everything(seed + 777)
    model = M.build(arm, S.n_features, 10, K=0 if basis is None else basis.K, cfg=info["nn_cfg"],
                    clause_init=None).to(X.device)
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    fake = type("R", (), {"info": {"_model": model, "_snaps": [sd], "basis_obj": basis, "nn_cfg": info["nn_cfg"]},
                          "explain": None})()
    return explain_all(arm, fake, S, X, seed)


def mnist_explain(arm: str, res, S: TabularSplits, device: str, rows: np.ndarray, seed: int):
    """Scalars for the record and arrays for the ``.npz``."""
    dev = torch.device(device)
    t0 = time.time()
    show = showcase_rows(S.y_test)
    X = torch.as_tensor(S.X_test[rows], device=dev)
    Xs = torch.as_tensor(S.X_test[show], device=dev)
    n = X.shape[0]
    E = explain_all(arm, res, S, torch.cat([X, Xs]), seed)
    tgt, ex = E["_tgt"], E["_ex"]
    methods = [k for k in E if not k.startswith("_")]
    out: Dict = {"n_explained": int(n), "methods": methods}
    arrays: Dict[str, np.ndarray] = {"rows": rows, "showcase": show, "fracs": FRACS,
                                     "showcase_pred": tgt[n:].cpu().numpy()}
    full = ex.full
    for m in methods:
        A = E[m]
        arrays[f"show_{m}"] = A[n:].cpu().numpy()
        arrays[f"first_{m}"] = A[:min(10, n)].cpu().numpy()   # pairs with show_random_* (sanity)
        cur, auc, ret, rauc = deletion(full, X, A[:n], tgt[:n])
        rnd = torch.rand(A[:n].shape, generator=torch.Generator(device=dev).manual_seed(seed), device=dev)
        cur_r, auc_r, ret_r, rauc_r = deletion(full, X, rnd, tgt[:n])
        arrays[f"del_auc_{m}"], arrays["del_auc_random"] = auc, auc_r
        arrays[f"ret_auc_{m}"], arrays["ret_auc_random"] = rauc, rauc_r
        out[f"deletion_{m}"] = {"auc": float(auc.mean()), "auc_random": float(auc_r.mean()),
                                "gain": float((auc_r - auc).mean()), "curve": cur.mean(0).tolist(),
                                "curve_random": cur_r.mean(0).tolist(),
                                "ret_auc": float(rauc.mean()), "ret_auc_random": float(rauc_r.mean()),
                                "ret_gain": float((rauc_r - rauc).mean()), "ret_curve": ret.mean(0).tolist(),
                                "ret_curve_random": ret_r.mean(0).tolist()}
    if "credit" in E:
        z = full(X)
        b = 0.0
        if ex.parts is not None:
            bias = ex.parts(torch.arange(n, device=dev))["bias"]
            b = bias.gather(1, tgt[:n, None])[:, 0]
        out["credit_completeness_err"] = float((E["credit"][:n].sum(1) - (z.gather(1, tgt[:n, None])[:, 0] - b)).abs().mean()) \
            if arm != "tm" else None
        out.update({f"credit_{k}": v for k, v in _credit_stats(E["credit"][:n], X).items()})
        arrays.update({f"show_{k}": v for k, v in top_clauses(res.info["basis_obj"], Xs, E["_Wk"][n:]).items()})
    if "ig" in E:
        z = full(X).gather(1, tgt[:n, None])[:, 0]
        z0 = full(torch.zeros_like(X)).gather(1, tgt[:n, None])[:, 0]
        out["ig_completeness_err"] = float((E["ig"][:n].sum(1) - (z - z0)).abs().mean())
    # readability and local validity (local-model arms)
    if ex.local is not None:
        Xn = X
        phi_full = removal_attributions(ex, Xn, use_local=False)
        phi = E["removal"][:n].cpu().numpy()
        num = (phi * phi_full).sum(1)
        den = np.linalg.norm(phi, axis=1) * np.linalg.norm(phi_full, axis=1)
        ok = den > 1e-12
        out["local_vs_full_removal_corr"] = float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")
        basis = res.info.get("basis_obj")
        sub = Explainable(ex.full, ex.local, ex.parts, ex.clause_fn)
        out.update(local_model_summary(sub, Xn, None if basis is None else basis.lengths()))
        out.update(local_fidelity(sub, Xn, S.y_test[rows], k_per_class=(10, 25, 50),
                                  n_anchor=min(200, n), seed=seed))
    ink = (X > 0).cpu().numpy()
    out["ink_pixels_mean"] = float(ink.sum(1).mean())
    for m in methods:
        a = np.abs(E[m][:n].cpu().numpy())
        out[f"terms90_{m}"] = float(_terms(a, 0.9).mean())
    # model-randomisation sanity check
    if arm in ("imn", "imn_tm", "tabresnet"):
        R = random_model_explanations(arm, res, S, X, seed)
        for m in methods:
            if m in R:
                out[f"sanity_spearman_{m}"] = _spearman_rows(np.abs(E[m][:n].cpu().numpy()),
                                                             np.abs(R[m].cpu().numpy()), ink)
                arrays[f"show_random_{m}"] = R[m][:min(10, n)].cpu().numpy()
    # decoy ground truth
    if hasattr(S, "X_test_swapped"):
        out.update(_decoy_block(arm, res, S, dev, rows, seed, E, n, arrays))
    out["explain_time"] = time.time() - t0
    return out, arrays


def _terms(a: np.ndarray, mass: float) -> np.ndarray:
    s = np.sort(a, 1)[:, ::-1]
    tot = s.sum(1, keepdims=True)
    cum = np.cumsum(s, 1) / np.maximum(tot, 1e-12)
    return (cum < mass).sum(1) + 1


def _credit_stats(psi: Tensor, X: Tensor) -> Dict[str, float]:
    a = psi.abs()
    on_bg = (a * (X <= 0)).sum(1) / a.sum(1).clamp_min(1e-12)
    return {"background_share": float(on_bg.mean())}


def _decoy_block(arm, res, S, dev, rows, seed, E, n, arrays) -> Dict:
    full = E["_ex"].full
    out = {"acc_consistent": _pred_metrics(full, torch.as_tensor(S.X_test, device=dev), S.y_test)["accuracy"],
           "acc_clean": _pred_metrics(full, torch.as_tensor(S.X_test_clean, device=dev), S.y_test)["accuracy"],
           "acc_swapped": _pred_metrics(full, torch.as_tensor(S.X_test_swapped, device=dev), S.y_test)["accuracy"]}
    with torch.no_grad():
        pred_sw = full(torch.as_tensor(S.X_test_swapped, device=dev)).argmax(1).cpu().numpy()
    out["reliance"] = float((pred_sw == S.swap_class).mean())
    # share of |attribution| on the barcode, consistent images
    for m in [k for k in E if not k.startswith("_")]:
        a = np.abs(E[m][:n].cpu().numpy())
        out[f"barcode_share_{m}"] = float((a[:, BARCODE].sum(1) / np.maximum(a.sum(1), 1e-12)).mean())
    # ground truth on swapped images whose prediction follows the barcode
    Xw = torch.as_tensor(S.X_test_swapped[rows], device=dev)
    Ew = explain_all(arm, res, S, Xw, seed)
    pred = Ew["_tgt"].cpu().numpy()
    follow = pred == S.swap_class[rows]
    lit = BARCODE[S.swap_class[rows]]
    out["n_follow"] = int(follow.sum())
    for m in [k for k in Ew if not k.startswith("_")]:
        a = Ew[m].cpu().numpy()
        top = np.argsort(-a, 1)[:, :10]
        hit = (top == lit[:, None]).any(1)
        out[f"barcode_hit10_{m}"] = float(hit[follow].mean()) if follow.any() else float("nan")
        out[f"barcode_hit10_notfollow_{m}"] = float(hit[~follow].mean()) if (~follow).any() else float("nan")
        arrays[f"swap_{m}"] = a[:10]
        arrays[f"swapall_{m}"] = a.astype(np.float16)  # every explained swapped image, for the figure
    arrays["swap_rows"], arrays["swap_pred"], arrays["swap_class"] = rows[:10], pred[:10], S.swap_class[rows][:10]
    arrays["swapall_rows"], arrays["swapall_pred"] = rows, pred
    return out


# ---------------------------------------------------------------------------------- driver
def run_mnist_one(stage: str, dataset: str, arm: str, seed: int, device: str, nn_over: dict,
                  tm_over: dict, tag: str = "", force: bool = False, n_explain: int = 500,
                  log=print) -> Path:
    from .arms import run_arm
    from .run import explain_rows_for, provenance
    S = load_mnist(dataset, seed)
    path = RESULTS / stage / f"554_{dataset}" / (f"{arm}__s{seed}" + (f"__{tag}" if tag else "") + ".json")
    if path.exists() and not force:
        log(f"skip {path.relative_to(EXP_DIR)} (exists)")
        return path
    log(f"[{stage}] {dataset} arm={arm} seed={seed} nn={nn_over} tm={tm_over}")
    t0 = time.time()
    res = run_arm(arm, S, seed, device, nn_over, tm_over, log, explain_rows=np.arange(1))
    metrics = {k: classification_metrics(getattr(S, f"y_{k}"), p, S.n_classes) for k, p in res.proba.items()}
    xinfo, arrays = {}, {}
    if n_explain > 0:
        rows = explain_rows_for(S, n_explain, seed)
        xinfo, arrays = mnist_explain(arm, res, S, device, rows, seed)
    info = {k: v for k, v in res.info.items() if k != "basis_obj" and not k.startswith("_")}
    rec = {"stage": stage, "dataset_id": 554, "dataset": dataset, "arm": arm, "seed": seed, "tag": tag,
           "status": "ok", "nn_overrides": nn_over, "tm_overrides": tm_over,
           "data": {"n_train": len(S.X_train), "n_valid": len(S.X_valid), "n_test": len(S.X_test),
                    "n_features": S.n_features, "n_classes": 10, "fingerprint": S.fingerprint()},
           "metrics": metrics, "info": info, "explain": xinfo, "wall_time": time.time() - t0,
           "provenance": provenance()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.write_text(json.dumps(rec, indent=1, default=float))
    log(f"  -> test acc {metrics['test']['accuracy']:.4f} (valid {metrics['valid']['accuracy']:.4f}) "
        f"[{time.time() - t0:.0f}s] {path.relative_to(EXP_DIR)}")
    return path


def selected_config(kind: str) -> Dict:
    """The M0 selection (written by ``mnist_select``): ``kind`` in {"tm", "<arm>"}."""
    sel = json.loads((RESULTS / "mnist_pilot" / "selection.json").read_text())
    return sel[kind]


PILOT_TM = [{"thresholds": th, "n_clauses": c, "s": s_, "epochs": 30, "patience": 5}
            for th in ([0.3], [0.25, 0.5, 0.75]) for c in (200, 500) for s_ in (5.0, 10.0)]
PILOT_STEPS = (50_000, 200_000)
NN_BASE = {"imn": {}, "tabresnet": {}, "imn_tm": {"l1_mode": "prior", "weight_norm": 1.0},
           "clause_lr": {"l1_mode": "prior", "weight_norm": 1.0}}


def pilot_tag(tm: Dict) -> str:
    return f"th{len(tm['thresholds'])}_c{tm['n_clauses']}_s{int(tm['s'])}"


def _records(stage: str, dataset: str = "mnist") -> List[Dict]:
    return [json.loads(p.read_text()) for p in sorted((RESULTS / stage / f"554_{dataset}").glob("*.json"))]


def select(kind: str) -> None:
    """M0 selection on **validation accuracy** (seed 0), written to results/mnist_pilot/selection.json."""
    path = RESULTS / "mnist_pilot" / "selection.json"
    sel = json.loads(path.read_text()) if path.exists() else {}
    recs = [r for r in _records("mnist_pilot") if r["status"] == "ok"]
    if kind == "tm":
        tms = [r for r in recs if r["arm"] == "tm"]
        assert len(tms) == len(PILOT_TM), f"{len(tms)}/{len(PILOT_TM)} TM pilot records"
        best = max(tms, key=lambda r: r["metrics"]["valid"]["accuracy"])
        sel["tm"] = best["tm_overrides"]
        sel["tm_candidates"] = {r["tag"]: r["metrics"]["valid"]["accuracy"] for r in tms}
    else:
        for arm in ("tabresnet", "imn", "imn_tm"):
            rs = [r for r in recs if r["arm"] == arm]
            assert len(rs) == len(PILOT_STEPS), f"{arm}: {len(rs)}/{len(PILOT_STEPS)} records"
            best = max(rs, key=lambda r: r["metrics"]["valid"]["accuracy"])
            sel[arm] = best["nn_overrides"]
            sel[f"{arm}_candidates"] = {r["tag"]: r["metrics"]["valid"]["accuracy"] for r in rs}
        # clause_lr has no hypernetwork; it gets imn_tm's budget
        sel["clause_lr"] = {**NN_BASE["clause_lr"], "max_steps": sel["imn_tm"]["max_steps"]}
    path.write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


def make_jobs(which: str) -> None:
    """Write ``jobs/mnist_<which>.jsonl`` (``pilot_tm``, ``pilot_nn``, ``main``, ``decoy``)."""
    J = []
    job = lambda stage, ds, arm, seed, nn, tm, tag="": {"stage": stage, "dataset": ds, "arm": arm,  # noqa: E731
                                                       "seed": seed, "nn": nn, "tm": tm, "tag": tag}
    if which == "pilot_tm":
        J = [job("mnist_pilot", "mnist", "tm", 0, {}, t, pilot_tag(t)) for t in PILOT_TM]
        J += [job("mnist_pilot", "mnist", a, 0, {**NN_BASE[a], "max_steps": k}, {}, f"k{k // 1000}")
              for k in PILOT_STEPS for a in ("tabresnet", "imn")]
    elif which == "pilot_nn":
        tm = selected_config("tm")
        J = [job("mnist_pilot", "mnist", "imn_tm", 0, {**NN_BASE["imn_tm"], "max_steps": k}, tm, f"k{k // 1000}")
             for k in PILOT_STEPS]
    elif which in ("main", "decoy"):
        tm = selected_config("tm")
        arms = ("imn_tm", "imn", "tabresnet", "clause_lr", "tm", "lr") if which == "main" else \
            ("imn_tm", "imn", "tabresnet", "tm", "lr")
        seeds = range(5) if which == "main" else range(3)
        ds = "mnist" if which == "main" else "mnist_decoy"
        for s_ in seeds:
            for a in arms:
                nn = selected_config(a) if a in NN_BASE else {}
                J.append(job(which if which == "main" else "mnist_decoy", ds, a, s_, nn,
                             tm if a in ("imn_tm", "clause_lr", "tm") else {}))
        for j in J:
            if j["stage"] == "main":
                j["stage"] = "mnist"
    path = EXP_DIR / "jobs" / f"mnist_{which}.jsonl"
    path.write_text("".join(json.dumps(j) + "\n" for j in J))
    print(f"wrote {len(J)} jobs -> {path.relative_to(EXP_DIR)}")


__all__: List[str] = ["load_mnist", "run_mnist_one", "mnist_explain", "selected_config"]

if __name__ == "__main__":
    import sys
    cmd, arg = sys.argv[1], sys.argv[2]
    {"select": select, "jobs": make_jobs}[cmd](arg)
