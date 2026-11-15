"""PLAN_HTM.md E-H3a and E-H3b: job lists, freeze, finals.

    python -m experiments.transformer_concepts.code.htm.exp_h3 jobs     # jobs_htm_h3.json (E-H3a + E-H3b search)
    python -m experiments.transformer_concepts.code.htm.exp_h3 freeze   # results/htm/frozen_h3.json
    python -m experiments.transformer_concepts.code.htm.exp_h3 final    # jobs_htm_h3_final.json
    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.htm.worker jobs_htm_h3.json

Protocol as the main programme (exp_main.py): weighted, M = 100 per class, T = 25, batch 32,
<= 40 epochs, patience 8; s searched over {5, 15, 40} at L = 32 (T2: 33) with seed 7 on
validation; finals at L in {16, 32} (T2: {17, 33}), seeds 23, 29, 31, 37, 41, test set.
E-H3a (pair TM on T3, very high s, validation only): s in {80, 160, 320}, M in {100, 300}
(T = M / 4), L = 16, seeds 1-3.
"""
from __future__ import annotations

import json
import sys
import time

from experiments.transformer_concepts.code.runner import RESULTS, ROOT

S_GRID = (5.0, 15.0, 40.0)
SEARCH_SEED = 7
FINAL_SEEDS = (23, 29, 31, 37, 41)
H3B_ARMS = {"T3": ("H_T3_seen", "H_T3_comp", "H_T3_seen_shuf", "H_T3_comp_shuf"),
            "T2": ("H_T2_mem", "H_T2_mem_shuf")}


def length(task, L):
    return L + 1 if task == "T2" else L


def jobs():
    out = []
    for s in (80.0, 160.0, 320.0):
        for M in (100, 300):
            for seed in (1, 2, 3):
                out.append(dict(stage="htm/eh3a", key=f"T3_L16_A2_s{int(s)}_M{M}_seed{seed}", task="T3", L=16,
                                arm_name="A2_pair_w2", s=s, seed=seed, M=M, T=M // 4, device="cuda"))
    for task, arms in H3B_ARMS.items():
        for arm in arms:
            for s in S_GRID:
                LL = length(task, 32)
                out.append(dict(stage="htm/h3_search", key=f"{task}_L{LL}_{arm}_s{int(s)}", task=task, L=LL,
                                arm_name=arm, s=s, seed=SEARCH_SEED, device="cuda"))
    out.sort(key=lambda j: j["stage"] != "htm/h3_search")  # cheap register arms first
    (ROOT / "jobs_htm_h3.json").write_text(json.dumps(out, indent=1))
    print(len(out), "jobs")


def freeze():
    chosen = {}
    for task, arms in H3B_ARMS.items():
        for arm in arms:
            recs = []
            for s in S_GRID:
                p = RESULTS / "htm/h3_search" / f"{task}_L{length(task, 32)}_{arm}_s{int(s)}.json"
                recs.append(json.loads(p.read_text()))
            best = max(recs, key=lambda r: (r["best_val"], -r["s"]))  # ties -> smaller s
            chosen[f"{task}|{arm}"] = dict(s=best["s"], best_val=best["best_val"],
                                           all={str(r["s"]): r["best_val"] for r in recs})
    rec = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), rule="best validation at L=32 (T2: 33), seed 7; "
               "ties -> smaller s", chosen=chosen, final_seeds=FINAL_SEEDS)
    p = RESULTS / "htm" / "frozen_h3.json"
    if p.exists():
        raise SystemExit(f"{p} exists: the freeze is written once")
    p.write_text(json.dumps(rec, indent=1))
    print(json.dumps(chosen, indent=1))


def final():
    fz = json.loads((RESULTS / "htm" / "frozen_h3.json").read_text())
    out = []
    for key, c in fz["chosen"].items():
        task, arm = key.split("|")
        for L in (16, 32):
            LL = length(task, L)
            for seed in FINAL_SEEDS:
                out.append(dict(stage="htm/h3_final", key=f"{task}_L{LL}_{arm}_seed{seed}", task=task, L=LL,
                                arm_name=arm, s=c["s"], seed=seed, test=True, device="cuda"))
    (ROOT / "jobs_htm_h3_final.json").write_text(json.dumps(out, indent=1))
    print(len(out), "jobs")


if __name__ == "__main__":
    {"jobs": jobs, "freeze": freeze, "final": final}[sys.argv[1]]()
