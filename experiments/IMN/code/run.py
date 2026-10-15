"""One (stage, dataset, arm, seed) -> one ``results/<stage>/<dataset>/<arm>__s<seed>[__tag].json``.

Usage (from the repository root)::

    python -m experiments.IMN.code.run --stage pilot --dataset 31 --arm imn_tm --seed 0
    python -m experiments.IMN.code.run --stage pilot --dataset 31 --arm all --seed 0 1 2
    python -m experiments.IMN.code.run --stage hpo --dataset 31 --arm imn_tm --seed 0 \
        --nn '{"weight_norm": 0.01}' --tm '{"n_clauses": 400}' --tag t07

Existing records are skipped unless ``--force``. Numbers in reports come only from these
records (rule C2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .arms import ALL_ARMS, SkipArm, run_arm
from .explain import local_fidelity, local_model_summary, removal_attributions
from .metrics import classification_metrics

EXP_DIR = Path(__file__).resolve().parents[1]
REPO = EXP_DIR.parents[1]
RESULTS = EXP_DIR / "results"


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:  # noqa: BLE001
            return None
    src_diff = git("diff", "HEAD", "--", "src/") or ""
    import torchtsetlin
    return {"git_rev": git("rev-parse", "HEAD"), "src_dirty": bool(src_diff),
            "src_diff_sha": hashlib.sha256(src_diff.encode()).hexdigest()[:16],
            "torchtsetlin": torchtsetlin.__version__, "torch": torch.__version__,
            "python": platform.python_version(), "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def load_splits(dataset: str, seed: int) -> D.TabularSplits:
    if dataset.startswith("xai:"):
        from .xai import make_task
        return make_task(dataset[4:], seed)[0]
    return D.load(int(dataset), seed)


def record_path(stage: str, S: D.TabularSplits, arm: str, seed: int, tag: str) -> Path:
    name = f"{S.dataset_id}_{S.name}" if S.dataset_id >= 0 else S.name
    return RESULTS / stage / name / (f"{arm}__s{seed}" + (f"__{tag}" if tag else "") + ".json")


def explain_rows_for(S: D.TabularSplits, n_explain: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    n = len(S.X_test)
    return np.sort(g.choice(n, size=min(max(n_explain, 1), n), replace=False))


def explain_block(res, S: D.TabularSplits, device: str, rows: np.ndarray, seed: int) -> dict:
    """``res.explain``'s local models are bound to ``X_test[rows]`` (positions 0..len(rows)-1)."""
    if res.explain is None or len(rows) == 0:
        return {}
    dev = torch.device(device)
    Xte = torch.as_tensor(S.X_test, device=dev)
    Xs = Xte[torch.as_tensor(rows, device=dev)]
    ex = res.explain
    t0 = time.time()
    out: dict = {"n_explained": int(len(rows))}
    if ex.local is not None:
        phi = removal_attributions(ex, Xs, use_local=True)
        phi_full = removal_attributions(ex, Xs, use_local=False)
        num = (phi * phi_full).sum(1)
        den = np.linalg.norm(phi, axis=1) * np.linalg.norm(phi_full, axis=1)
        ok = den > 1e-12
        out["local_vs_full_removal_corr"] = float(np.mean(num[ok] / den[ok])) if ok.any() else float("nan")
        basis = res.info.get("basis_obj")
        out.update(local_model_summary(ex, Xs, None if basis is None else basis.lengths()))
        out.update(local_fidelity(ex, Xs, S.y_test[rows], k_per_class=(10, 25, 50),
                                  n_anchor=min(200, len(rows)), seed=seed))
    else:
        phi = removal_attributions(ex, Xs, use_local=False)
    sub = ex
    if hasattr(S, "gt_test"):  # synthetic XAI task: score against the known contributions
        from .xai import xai_metrics
        out.update(xai_metrics(sub, Xs, phi, S.gt_test[rows], seed=seed))
    imp = np.abs(phi).mean(0)
    order = np.argsort(-imp)
    out["global_importance_top10"] = [[S.feature_names[i], float(imp[i])] for i in order[:10]]
    # paper Fig. 6 analogue: test AUROC after zeroing the top-k globally important features
    # (inference-time removal on the full model, no retraining)
    from .metrics import classification_metrics
    from .train_nn import logits_to_proba
    with torch.no_grad():
        base_auc = classification_metrics(S.y_test, logits_to_proba(ex.full(Xte)), S.n_classes)["auroc"]
        drops = []
        for k in range(1, min(5, S.n_features) + 1):
            Xk = Xte.clone()
            Xk[:, torch.as_tensor(order[:k].copy(), device=dev)] = 0.0
            auc = classification_metrics(S.y_test, logits_to_proba(ex.full(Xk)), S.n_classes)["auroc"]
            drops.append(float(base_auc - auc))
    out["topk_removal_auroc_drop"] = drops
    out["attribution_time"] = time.time() - t0
    return out


def run_one(stage: str, dataset: str, arm: str, seed: int, device: str, nn_over: dict, tm_over: dict,
            tag: str = "", force: bool = False, n_explain: int = 500, log=print) -> Path:
    S = load_splits(dataset, seed)
    path = record_path(stage, S, arm, seed, tag)
    if path.exists() and not force:
        log(f"skip {path.relative_to(EXP_DIR)} (exists)")
        return path
    log(f"[{stage}] {S.name} ({S.dataset_id}) arm={arm} seed={seed} "
        f"N={len(S.X_train)}/{len(S.X_valid)}/{len(S.X_test)} d={S.n_features} C={S.n_classes}")
    t0 = time.time()
    rows = explain_rows_for(S, n_explain, seed) if n_explain > 0 else np.zeros(0, dtype=np.int64)
    try:
        res = run_arm(arm, S, seed, device, nn_over, tm_over, log, explain_rows=rows if len(rows) else None)
    except SkipArm as e:  # infeasible under the preregistered budget: record it, do not drop it
        rec = {"stage": stage, "dataset_id": S.dataset_id, "dataset": S.name, "arm": arm, "seed": seed,
               "tag": tag, "status": "skipped", "reason": str(e), "nn_overrides": nn_over,
               "tm_overrides": tm_over, "provenance": provenance()}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=1))
        log(f"  -> SKIPPED ({e})")
        return path
    metrics = {k: classification_metrics(getattr(S, f"y_{k}"), p, S.n_classes) for k, p in res.proba.items()}
    xinfo = explain_block(res, S, device, rows, seed)
    info = {k: v for k, v in res.info.items() if k != "basis_obj"}
    rec = {"stage": stage, "dataset_id": S.dataset_id, "dataset": S.name, "arm": arm, "seed": seed, "tag": tag,
           "status": "ok", "nn_overrides": nn_over, "tm_overrides": tm_over,
           "data": {"n_train": len(S.X_train), "n_valid": len(S.X_valid), "n_test": len(S.X_test),
                    "n_features": S.n_features, "n_classes": S.n_classes, "fingerprint": S.fingerprint()},
           "metrics": metrics, "info": info, "explain": xinfo, "wall_time": time.time() - t0,
           "provenance": provenance()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=float))
    log(f"  -> test AUROC {metrics['test']['auroc']:.4f} acc {metrics['test']['accuracy']:.4f} "
        f"(valid {metrics['valid']['auroc']:.4f}) [{time.time() - t0:.0f}s] {path.relative_to(EXP_DIR)}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dataset", nargs="+", required=True, help="OpenML ids, 'tier1', 'tier2' or xai:<task>")
    ap.add_argument("--arm", nargs="+", required=True)
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--nn", default="{}", help="JSON overrides of train_nn.NN_DEFAULTS")
    ap.add_argument("--tm", default="{}", help="JSON overrides of tm_source.TM_DEFAULTS")
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-explain", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ds = []
    for d in a.dataset:
        ds += [str(i) for i in (D.TIER1 if d == "tier1" else D.TIER2 if d == "tier2" else [d])]
    arms = list(ALL_ARMS) if a.arm == ["all"] else a.arm
    failures = 0
    for d in ds:
        for seed in a.seed:
            for arm in arms:
                try:
                    run_one(a.stage, d, arm, seed, a.device, json.loads(a.nn), json.loads(a.tm), a.tag,
                            a.force, a.n_explain)
                except Exception:  # noqa: BLE001  one failing config must not stop a sweep
                    failures += 1
                    traceback.print_exc()
                    print(f"FAILED {d} {arm} seed={seed}", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
