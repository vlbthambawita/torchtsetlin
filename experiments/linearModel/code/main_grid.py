"""Phase 7 orchestration: build run groups from the selected source TMs and execute them.

    python -m experiments.linearModel.code.main_grid seed0          # full hyperparameter grids, seed 0
    python -m experiments.linearModel.code.main_grid select         # -> configs/final.json (frozen before test)
    python -m experiments.linearModel.code.main_grid seeds          # selected configs for seeds 1-2 (+ thr 0.3)

Groups are written to ``configs/main_<stage>.json`` and run by worker subprocesses
(``--parallel`` groups at a time; each M = 20 group needs ~27 GB RAM).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .data import EXP
from .source_tm import RES as SRC_RES
from .source_tm import source_id
from .sources import selected

CONFIGS = EXP / "configs"
RESULTS = EXP / "results"
LAMS = (1e-4, 1e-3, 1e-2)
DQ = ((1 / 16, 64), (1 / 64, 256), (1 / 256, 1024))
MS = (2, 10, 20)
SWEEPS = 2000


def int_cfgs(arm, seed=0, lams=LAMS, dqs=DQ):
    return [dict(arm=arm, engine="gram", delta=d, Q=Q, lam=lam, max_sweeps=SWEEPS, seed=seed, log_every=100)
            for lam in lams for d, Q in dqs]


def ridge_cfgs(arm, lams=LAMS):
    return [dict(arm=arm, lam=lam) for lam in lams]


def a4q_cfgs(lams=LAMS, dqs=DQ):
    return [dict(arm="A4q", lam=lam, delta=d, Q=Q) for lam in lams for d, Q in dqs]


def gated_configs(seed=0, flat=True):
    # ordered so the Gram cache is reused: A1 family, then the full-layout family
    c = int_cfgs("A1", seed) + ridge_cfgs("A1c") + int_cfgs("A3", seed) + int_cfgs("A3L", seed)
    c += ridge_cfgs("A4") + a4q_cfgs()
    return c


def seed0_groups():
    groups = []
    for kind in ("flat", "conv"):
        for M in MS:
            r = selected(kind, M)
            if r is None:
                print(f"missing source {kind} M={M}", file=sys.stderr)
                continue
            sid = r["source_id"]
            groups.append(dict(name=f"{kind}_M{M}_seed0", out="main", manifest="split_seed0", gates="source",
                               source=sid, arms=["A0"], configs=gated_configs()))
            if kind == "flat":
                for draw in range(3):
                    groups.append(dict(name=f"rand{draw}_M{M}_seed0", out="main", manifest="split_seed0",
                                       gates=f"random:{draw}", source=sid, arms=[], configs=int_cfgs("A5")))
    groups.append(dict(name="global_seed0", out="main", manifest="split_seed0", gates="constant", source=None,
                       arms=[], configs=int_cfgs("A2", 0) + ridge_cfgs("A2c")))
    # reference sources: A0 only
    for kind, Ms in (("flat", (50, 100, 500)), ("conv", (200,))):
        for M in Ms:
            r = selected(kind, M)
            if r is not None:
                groups.append(dict(name=f"{kind}_M{M}_seed0_ref", out="main", manifest="split_seed0", gates="source",
                                   source=r["source_id"], arms=["A0"], configs=[]))
    return groups


def run_groups(groups, stage, parallel=2, threads=16):
    CONFIGS.mkdir(exist_ok=True)
    (EXP / "logs" / "main").mkdir(parents=True, exist_ok=True)
    pending = list(groups)
    running = []
    gpu = 0
    while pending or running:
        while pending and len(running) < parallel:
            g = pending.pop(0)
            p = CONFIGS / f"main_{stage}_{g['name']}.json"
            p.write_text(json.dumps([g], indent=1))
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), NUMBA_NUM_THREADS=str(threads),
                       CUDA_VISIBLE_DEVICES=str(gpu % 2), PYTHONUNBUFFERED="1")
            gpu += 1
            log = open(EXP / "logs" / "main" / f"{g['name']}.log", "a")
            proc = subprocess.Popen([sys.executable, "-m", "experiments.linearModel.code.run", "--spec", str(p)],
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((proc, g["name"]))
            print(f"start {g['name']}", flush=True)
        time.sleep(10)
        for proc, name in list(running):
            if proc.poll() is not None:
                print(f"{'done' if proc.returncode == 0 else 'FAILED'} {name}", flush=True)
                running.remove((proc, name))


# ------------------------------------------------------------------------- selection (seed 0)
def _records(stage_dir="main"):
    for p in sorted((RESULTS / stage_dir).glob("*.json")):
        yield json.loads(p.read_text())


def _cfg_key(c):
    """Hyperparameter identity (no seed)."""
    return (c["arm"], c.get("lam"), c.get("delta"), c.get("Q"))


def _grid_rank(c):
    """Position in the preregistered grid (ties -> earlier)."""
    lam = LAMS.index(c["lam"]) if c.get("lam") in LAMS else 0
    dq = [d for d, _ in DQ].index(c["delta"]) if c.get("delta") in [d for d, _ in DQ] else 0
    return (lam, dq)


def select():
    """Each (family, M, arm) picks its configuration by the spec §10.1 validation key on seed 0.
    A5 is read at the A3-selected configuration (PLAN). Writes ``configs/final.json``."""
    best = {}
    a5 = {}
    for r in _records():
        g = r["group"]
        if not (g.endswith("_seed0") or g.endswith("_seed0_ref")):    # refs: A0 only, no hyperparameters
            continue
        fam, M = g.split("_")[0], (int(g.split("_")[1][1:]) if g.split("_")[1].startswith("M") else 0)
        arm = r["arm"]
        if arm == "A0":
            best[(fam, M, "A0")] = dict(run_id=r["run_id"], group=g, config=None, val_acc=r["val"]["acc"])
            continue
        c = r["config"]
        if arm == "A5":
            a5.setdefault((M, _cfg_key(dict(c, arm="A3"))), []).append(r)
            continue
        key = (-r["val"]["acc"], r["val"]["mse"], _grid_rank(c))
        k = (fam, M, arm)
        if k not in best or key < best[k]["key"]:
            best[k] = dict(run_id=r["run_id"], group=g, config=c, val_acc=r["val"]["acc"], key=key)
    for (fam, M, arm), b in list(best.items()):
        if fam == "flat" and arm == "A3":
            rs = a5.get((M, _cfg_key(b["config"])), [])
            for r in rs:
                best[("rand" + r["group"][4], M, "A5")] = dict(run_id=r["run_id"], group=r["group"], config=r["config"],
                                                                 val_acc=r["val"]["acc"], key=None)
    out = [dict(family=f, M=M, arm=a, **{k: v for k, v in b.items() if k != "key"}) for (f, M, a), b in sorted(best.items())]
    CONFIGS.mkdir(exist_ok=True)
    (CONFIGS / "final.json").write_text(json.dumps(out, indent=1))
    for o in out:
        print(o["family"], o["M"], o["arm"], round(o["val_acc"], 4), o["config"] and {k: o["config"].get(k) for k in ("lam", "delta", "Q")})
    return out


def seeds_groups(seeds=(1, 2)):
    """Selected configurations re-run for further source seeds (and the thr-0.3 check)."""
    final = json.loads((CONFIGS / "final.json").read_text())
    sel = {(o["family"], o["M"], o["arm"]): o["config"] for o in final}
    groups = []

    def cfgs(fam, M, arms, seed):
        out = []
        for a in arms:
            c = sel.get((fam, M, a))
            if c is not None:
                out.append(dict(c, seed=seed) if "engine" in c else dict(c))
        return out

    for kind in ("flat", "conv"):
        for M in MS:
            r0 = selected(kind, M)
            if r0 is None:
                continue
            for seed in seeds:
                sid = source_id(kind, M, r0["T"], r0["s"], seed, 0.5, "split_seed0", r0["weighted"])
                if not (SRC_RES / f"{sid}.json").exists():
                    print("missing", sid, file=sys.stderr)
                    continue
                arms = ["A1", "A1c", "A3", "A3L", "A4", "A4q"]
                groups.append(dict(name=f"{kind}_M{M}_seed{seed}", out="main", manifest="split_seed0", gates="source",
                                   source=sid, arms=["A0"], configs=cfgs(kind, M, arms, seed)))
                if kind == "flat":
                    a3 = sel[("flat", M, "A3")]
                    for draw in range(3):
                        groups.append(dict(name=f"rand{draw}_M{M}_seed{seed}", out="main", manifest="split_seed0",
                                           gates=f"random:{draw}", source=sid, arms=[],
                                           configs=[dict(a3, arm="A5", seed=seed)]))
                if kind == "flat" and M == 20:
                    pass
        # thr 0.3 secondary check (D1): flat M = 20, seeds 0-2, A0/A1/A3
    r0 = selected("flat", 20)
    for seed in (0,) + tuple(seeds):
        sid = source_id("flat", 20, r0["T"], r0["s"], seed, 0.3, "split_seed0", False)
        if (SRC_RES / f"{sid}.json").exists():
            groups.append(dict(name=f"thr03_M20_seed{seed}", out="main", manifest="split_seed0", gates="source",
                               source=sid, arms=["A0"], configs=cfgs("flat", 20, ["A1", "A3"], seed)))
    for seed in seeds:
        groups.append(dict(name=f"global_seed{seed}", out="main", manifest="split_seed0", gates="constant", source=None,
                           arms=[], configs=[dict(sel[("global", 0, "A2")], seed=seed)]))
    for kind, Ms in (("flat", (50, 100, 500)), ("conv", (200,))):
        for M in Ms:
            r0 = selected(kind, M)
            for seed in seeds:
                sid = r0 and source_id(kind, M, r0["T"], r0["s"], seed, 0.5, "split_seed0", r0["weighted"])
                if sid and (SRC_RES / f"{sid}.json").exists():
                    groups.append(dict(name=f"{kind}_M{M}_seed{seed}_ref", out="main", manifest="split_seed0",
                                       gates="source", source=sid, arms=["A0"], configs=[]))
    return groups


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "seed0":
        gs = seed0_groups()
        only = sys.argv[2:] if len(sys.argv) > 2 else None
        if only:
            gs = [g for g in gs if any(g["name"].startswith(o) for o in only)]
        run_groups(gs, "seed0", parallel=int(os.environ.get("GROUPS_PARALLEL", 2)))
    elif cmd == "select":
        select()
    elif cmd == "seeds":
        gs = seeds_groups()
        only = sys.argv[2:] if len(sys.argv) > 2 else None
        if only:
            gs = [g for g in gs if any(g["name"].startswith(o) for o in only)]
        run_groups(gs, "seeds")
    elif cmd == "list":
        for g in seed0_groups():
            print(g["name"], g["source"], len(g["configs"]))
