"""PROOF_PLAN Part III: the main comparison (PLAN.md section 3-5), search -> freeze -> finals.

    python -m experiments.transformer_concepts.code.exp_main search   # jobs_search.json
    python -m experiments.transformer_concepts.code.exp_main freeze   # results/frozen_experiment.json
    python -m experiments.transformer_concepts.code.exp_main final    # jobs_final.json (needs the freeze)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

from experiments.transformer_concepts.code.runner import RESULTS, ROOT

TASKS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
COMMON = ("B0_flat", "B1_conv_k1", "B2_conv_k3_pe", "B4_graph_chain", "A1_pair_w1", "A2_pair_w2",
          "A3_pair_bucket", "A4_graph_complete", "A2_self", "A2_norel", "A2_abs", "A2_shuf")
P_ARMS = ("P_onehot", "P_therm", "P_binary", "P_gray")
S_GRID = (5.0, 15.0, 40.0)
SEARCH_SEED = 7
FINAL_SEEDS = (23, 29, 31, 37, 41)
CURVE_SIZES = (1000, 4000, 16000)
CURVE_SEEDS = (23, 29, 31)


def arms_for(task):
    a = list(COMMON)
    if task == "T4":
        a.append("B3_conv_kd1")
    if task in ("T1", "T4", "T6"):
        a += list(P_ARMS)
    return a


def length(task, L):
    return L + 1 if task == "T2" else L


def search_jobs():
    out = []
    for task in TASKS:
        for arm in arms_for(task):
            for s in S_GRID:
                LL = length(task, 32)
                out.append(dict(stage="search", key=f"{task}_L{LL}_{arm}_s{int(s)}", task=task, L=LL,
                                arm_name=arm, s=s, seed=SEARCH_SEED, device="cuda"))
    # cheap arms first, pair/graph arms (slow at L=32) interleaved after
    out.sort(key=lambda j: (j["arm_name"].startswith(("A", "B4")), j["task"]))
    return out


def code_hash():
    h = hashlib.sha256()
    for f in sorted((ROOT / "code").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()


def freeze():
    chosen, missing = {}, []
    for task in TASKS:
        for arm in arms_for(task):
            best = None
            for s in S_GRID:
                p = RESULTS / "search" / f"{task}_L{length(task, 32)}_{arm}_s{int(s)}.json"
                if not p.exists():
                    missing.append(p.name)
                    continue
                r = json.loads(p.read_text())
                if best is None or r["best_val"] > best[1] + 1e-12:
                    best = (s, r["best_val"])
            if best is not None:
                chosen[f"{task}|{arm}"] = dict(s=best[0], search_val=best[1])
    rec = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"), s_grid=S_GRID, search_seed=SEARCH_SEED,
               final_seeds=FINAL_SEEDS, lengths={t: [length(t, 16), length(t, 32)] for t in TASKS},
               chosen=chosen, missing_search_records=missing, code_sha256=code_hash(),
               rule="best validation accuracy at L=32 (T2: 33), 1 seed; ties -> smaller s")
    (RESULTS / "frozen_experiment.json").write_text(json.dumps(rec, indent=1))
    print(f"frozen {len(chosen)} (task, arm) settings; {len(missing)} search records missing")


def final_jobs():
    fz = json.loads((RESULTS / "frozen_experiment.json").read_text())
    out = []
    for seed in FINAL_SEEDS:
        for task in TASKS:
            for L in (16, 32):
                LL = length(task, L)
                for arm in arms_for(task):
                    c = fz["chosen"].get(f"{task}|{arm}")
                    if c is None:
                        continue
                    out.append(dict(stage="final", key=f"{task}_L{LL}_{arm}_seed{seed}", task=task, L=LL,
                                    arm_name=arm, s=c["s"], seed=seed, test=True, device="cuda",
                                    save_model=arm == "A2_pair_w2"))
    for seed in CURVE_SEEDS:
        for n in CURVE_SIZES:
            for arm in ("A2_pair_w2", "B0_flat", "A2_self"):
                c = fz["chosen"][f"T2|{arm}"]
                out.append(dict(stage="curves", key=f"T2_L17_{arm}_n{n}_seed{seed}", task="T2", L=17,
                                arm_name=arm, s=c["s"], seed=seed, n_train=n, test=True, device="cuda"))
    # seeds outer: if the night runs out, every (task, arm) has the same number of seeds
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "search":
        j = search_jobs()
        (ROOT / "jobs_search.json").write_text(json.dumps(j))
        print(len(j), "search jobs")
    elif cmd == "freeze":
        freeze()
    elif cmd == "final":
        j = final_jobs()
        (ROOT / "jobs_final.json").write_text(json.dumps(j))
        print(len(j), "final jobs")
