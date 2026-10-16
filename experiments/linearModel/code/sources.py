"""Enumerate and run the Phase 7 source-TM jobs on a small GPU worker pool.

    python -m experiments.linearModel.code.sources grid      # seed-0 hyperparameter grids
    python -m experiments.linearModel.code.sources seeds     # seeds 1-2 (+ thr 0.3) at the selected (T, s)
    python -m experiments.linearModel.code.sources status
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .data import EXP
from .source_tm import RES, exists, source_id

LOGS = EXP / "logs" / "source"


def grid_jobs():
    jobs = []
    for kind in ("flat", "conv"):
        for M in (2, 10, 20):
            for T in sorted({max(2, M // 2), max(2, M), 2 * M}):
                for s in (3, 5, 10):
                    jobs.append(dict(kind=kind, M=M, T=T, s=s, seed=0, weighted=kind == "conv"))
    for M in (50, 100, 500):
        for T in (25, 50, 100):
            for s in (5, 10):
                jobs.append(dict(kind="flat", M=M, T=T, s=s, seed=0))
    for T in (50, 100, 200):
        jobs.append(dict(kind="conv", M=200, T=T, s=10, seed=0, weighted=True))
    return jobs


def selected(kind, M, threshold=0.5):
    """Seed-0 (T, s) with the best validation accuracy (ties: listed first = smaller T, s)."""
    best = None
    w = "_w" if kind == "conv" else ""                        # D10: conv sources are weighted
    for p in sorted(RES.glob(f"{kind}{w}_M{M}_T*_thr{threshold:g}_seed0_split_seed0_e30.json")):
        r = json.loads(p.read_text())
        key = (-r["val_acc"], r["T"], r["s"])
        if best is None or key < best[0]:
            best = (key, r)
    return best[1] if best else None


def seed_jobs():
    jobs = []
    for kind, Ms in (("flat", (2, 10, 20, 50, 100, 500)), ("conv", (2, 10, 20, 200))):
        for M in Ms:
            r = selected(kind, M)
            if r is None:
                continue
            for seed in (1, 2):
                jobs.append(dict(kind=kind, M=M, T=r["T"], s=r["s"], seed=seed, weighted=r["weighted"]))
            if kind == "flat" and M == 20:
                for seed in (0, 1, 2):
                    jobs.append(dict(kind=kind, M=M, T=r["T"], s=r["s"], seed=seed, threshold=0.3))
    return jobs


def run(jobs, per_gpu=4, gpus=(0, 1)):
    LOGS.mkdir(parents=True, exist_ok=True)
    sid_of = lambda j: source_id(j["kind"], j["M"], j["T"], j["s"], j["seed"], j.get("threshold", 0.5),  # noqa: E731
                                 "split_seed0", j.get("weighted", False))
    todo = [j for j in jobs if not exists(sid_of(j))]
    # big jobs first so they do not trail at the end
    todo.sort(key=lambda j: -(j["M"] * (30 if j["kind"] == "conv" else 1)))
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    print(f"{len(todo)} jobs", flush=True)
    while todo or running:
        for slot in [s for s in range(len(slots)) if s not in running]:
            if not todo:
                break
            j = todo.pop(0)
            sid = sid_of(j)
            cmd = [sys.executable, "-m", "experiments.linearModel.code.source_tm", "--kind", j["kind"], "--M", str(j["M"]),
                   "--T", str(j["T"]), "--s", str(j["s"]), "--seed", str(j["seed"]),
                   "--threshold", str(j.get("threshold", 0.5))] + (["--weighted"] if j.get("weighted") else [])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slots[slot]), OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
            running[slot] = (subprocess.Popen(cmd, env=env, stdout=open(LOGS / f"{sid}.log", "w"), stderr=subprocess.STDOUT), sid)
            print(f"start {sid} on gpu {slots[slot]}", flush=True)
        time.sleep(5)
        for slot, (p, sid) in list(running.items()):
            if p.poll() is not None:
                print(f"{'done' if p.returncode == 0 else 'FAILED'} {sid}", flush=True)
                del running[slot]


def status():
    for kind, Ms in (("flat", (2, 10, 20, 50, 100, 500)), ("conv", (2, 10, 20, 200))):
        for M in Ms:
            r = selected(kind, M)
            n = len(list(RES.glob(f"{kind}{'_w' if kind == 'conv' else ''}_M{M}_T*_seed0_split_seed0_e30.json")))
            print(kind, M, n, "runs;", "best:" if r else "", r and (r["T"], r["s"], round(r["val_acc"], 4), r["best_epoch"]))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "grid":
        run(grid_jobs())
    elif cmd == "grid-conv":
        run([j for j in grid_jobs() if j["kind"] == "conv"], per_gpu=int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    elif cmd == "seeds":
        kinds = sys.argv[2:] or ["flat", "conv"]
        run([j for j in seed_jobs() if j["kind"] in kinds], per_gpu=2)
    elif cmd == "status":
        status()
