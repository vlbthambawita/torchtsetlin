"""Phase 3: equal-budget random-search HPO (PLAN.md section 6, DECISIONS.md D11/D20-D22).

    python -m experiments.IMN.code.hpo trials --name hpo_a --arms lr dt rf gbdt tm tabresnet imn
    python -m experiments.IMN.code.hpo trials --name hpo_b --arms clause_lr imn_tm imn_tm_rand imn_hybrid tm_gated
        # stage-b trials use, per dataset, the source TM selected for the ``tm`` arm in stage a
    python -m experiments.IMN.code.hpo final  --name hpo_final  # winners x seeds 0-4 -> stage hpo_final

Trials run on split seed 0 (stage ``hpo``, tag ``t<k>``); the winner of each (arm, dataset) is
the trial with the best **validation** AUROC (C3); it is re-run on seeds 0-4 (stage
``hpo_final``, empty tag) and those records are what the report uses.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from . import data as D
from .jobs import JOB_DIR, _COST_HINT
from .report import RESULTS

N_TRIALS = 20
CLAUSE_ARMS = ("clause_lr", "imn_tm", "imn_tm_rand", "imn_hybrid", "tm_gated")
NEURAL = ("tabresnet", "imn") + CLAUSE_ARMS


def _loguni(g, lo, hi):
    return float(np.exp(g.uniform(np.log(lo), np.log(hi))))


def sample(arm: str, g: np.random.Generator) -> Dict:
    """One configuration: ``{"nn": ..., "tm": ...}`` (sklearn params travel in ``nn``)."""
    if arm == "lr":  # paper Table 12
        pen = g.choice(["l2", None])
        p = {"C": float(g.uniform(1e-5, 5)), "max_iter": int(g.integers(50, 501)),
             "fit_intercept": bool(g.integers(2))}
        p["penalty"] = None if pen is None else str(pen)
        return {"nn": p, "tm": {}}
    if arm in ("dt", "rf"):  # Tables 13 / 15
        p = {"criterion": str(g.choice(["gini", "entropy"])), "max_depth": int(g.integers(1, 22)),
             "min_samples_split": int(g.integers(2, 12)), "max_leaf_nodes": int(g.integers(3, 27))}
        if arm == "dt":
            p["splitter"] = str(g.choice(["best", "random"]))
        else:
            p["n_estimators"] = int(g.integers(100, 4001))
        return {"nn": p, "tm": {}}
    if arm == "gbdt":  # Table 14 mapped onto HistGradientBoosting (D21)
        return {"nn": {"learning_rate": _loguni(g, 1e-5, 1), "max_iter": int(g.integers(100, 4001)),
                       "l2_regularization": _loguni(g, 1, 10)}, "tm": {}}
    if arm == "tm":
        mil = g.choice([0, 8, 16, 32])
        return {"nn": {}, "tm": {"n_clauses": int(g.choice([50, 100, 200, 400, 800])),
                                 "T_ratio": float(g.choice([0.25, 0.5, 1.0, 2.0])),
                                 "s": float(g.choice([2, 3, 5, 8, 12, 20])),
                                 "n_bits": int(g.choice([5, 10, 20])),
                                 "max_included_literals": None if mil == 0 else int(mil)}}
    if arm in NEURAL:  # paper Table 11 (+ L1 weight / mode for interpretable arms)
        p = {"epochs": int(g.integers(10, 501)), "lr": _loguni(g, 1e-5, 1e-1),
             "batch_size": int(g.choice([32, 64, 128, 256, 512])), "weight_decay": _loguni(g, 1e-5, 1e-1),
             "dropout": float(g.uniform(0, 0.5))}
        if arm == "imn":
            p["weight_norm"] = _loguni(g, 1e-5, 1)
        elif arm in CLAUSE_ARMS:
            p["weight_norm"] = _loguni(g, 1e-5, 10)
            if arm not in ("tm_gated",):
                p["l1_mode"] = str(g.choice(["all", "active", "prior"]))
        return {"nn": p, "tm": {}}
    raise ValueError(arm)


def _records(stage: str) -> List[dict]:
    out = []
    for p in (RESULTS / stage).glob("**/*.json"):
        r = json.loads(p.read_text())
        if r.get("status", "ok") == "ok":
            out.append(r)
    return out


def winners(stage: str = "hpo") -> Dict:
    """``{(arm, dataset_id): record}`` with the best seed-0 validation AUROC."""
    best: Dict = {}
    for r in _records(stage):
        k = (r["arm"], str(r["dataset_id"]))
        v = r["metrics"]["valid"]["auroc"]
        if k not in best or v > best[k]["metrics"]["valid"]["auroc"]:
            best[k] = r
    return best


def _datasets(spec: List[str]) -> List[str]:
    ds = []
    for d in spec:
        ds += [str(i) for i in (D.TIER1 if d == "tier1" else [d])]
    return sorted(ds, key=lambda d: -_COST_HINT.get(int(d), 0))


def cmd_trials(a) -> None:
    tm_win = {}
    if any(arm in CLAUSE_ARMS for arm in a.arms):
        w = winners()
        tm_win = {d: w[("tm", d)]["tm_overrides"] for (arm, d) in w if arm == "tm"}
    jobs = []
    for d in _datasets(a.dataset):
        for arm in a.arms:
            # stable per-(arm, dataset) stream (Python's hash() is salted per process)
            g = np.random.default_rng(int.from_bytes(f"{arm}|{d}".encode(), "little") % 2**32)
            for t in range(a.trials):
                cfg = sample(arm, g)
                if arm in CLAUSE_ARMS:
                    if d not in tm_win:
                        raise SystemExit(f"no tm winner for dataset {d}; finish stage-a trials first")
                    cfg["tm"] = tm_win[d]
                jobs.append({"stage": "hpo", "dataset": d, "arm": arm, "seed": 0, "nn": cfg["nn"],
                             "tm": cfg["tm"], "tag": f"t{t:02d}"})
    JOB_DIR.mkdir(exist_ok=True)
    (JOB_DIR / f"{a.name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"wrote {len(jobs)} trial jobs -> jobs/{a.name}.jsonl")


def cmd_final(a) -> None:
    w = winners()
    jobs, table = [], {}
    for (arm, d), r in sorted(w.items(), key=lambda kv: -_COST_HINT.get(int(kv[0][1]), 0)):
        table[f"{arm}|{d}"] = {"tag": r["tag"], "valid_auroc": r["metrics"]["valid"]["auroc"],
                               "nn": r["nn_overrides"], "tm": r["tm_overrides"]}
        for s in a.seeds:
            jobs.append({"stage": "hpo_final", "dataset": d, "arm": arm, "seed": s,
                         "nn": r["nn_overrides"], "tm": r["tm_overrides"], "tag": ""})
    (RESULTS / "hpo").mkdir(parents=True, exist_ok=True)
    (RESULTS / "hpo" / "winners.json").write_text(json.dumps(table, indent=1))
    (JOB_DIR / f"{a.name}.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(table)} winners -> results/hpo/winners.json; {len(jobs)} jobs -> jobs/{a.name}.jsonl")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trials")
    t.add_argument("--name", required=True)
    t.add_argument("--arms", nargs="+", required=True)
    t.add_argument("--dataset", nargs="+", default=["tier1"])
    t.add_argument("--trials", type=int, default=N_TRIALS)
    f = sub.add_parser("final")
    f.add_argument("--name", required=True)
    f.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args(argv)
    {"trials": cmd_trials, "final": cmd_final}[a.cmd](a)


if __name__ == "__main__":
    main()
