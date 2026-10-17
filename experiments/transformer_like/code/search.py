"""Search ledger, GPU job queue, selection/freeze and the five-seed benchmark (PLAN §6, spec §11).

Jobs are subprocesses of ``code.run`` pinned to one GPU each (``CUDA_VISIBLE_DEVICES``). A job
whose record already exists with ``status == completed`` is skipped (resumable). A job may wait
for an encoder checkpoint to exist.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from . import data as D
from .common import ART, EXP, FROZEN, LOGS, REPO, RESULTS, git_sha, read_json, resolve_T, src_diff, write_json
from .resources import matched_M

TUNING_SEED = 17
FINAL_SEEDS = [23, 41, 59, 83, 101]
S1_SEEDS = [23, 41, 59]
BASE_GRID = {"M": [500, 1000, 2000], "s": [3.0, 5.0], "T_ratio": [0.25, 0.5]}
N0_GRID = {"M": [250, 500, 1000], "s": [3.0, 5.0], "temperature": [0.5, 1.0]}
BUDGET_RATIOS = {"s": [3.0, 5.0], "T_ratio": [0.25, 0.5]}
ENC_REL = "artifacts/encoders/seed{seed}{tag}.pt"


# ---------------------------------------------------------------------- jobs
def trial_spec(stage: str, arm: str, seed: int, head: Dict, temperature: float = 1.0, **kw) -> Dict:
    spec = {"stage": stage, "arm": arm, "seed": seed, "head": head, "temperature": temperature,
            "manifest": kw.pop("manifest", D.MAIN), "max_epochs": 50, "patience": 10, "batch_size": 32}
    if arm in ("N0", "A2", "A2M", "A3"):
        spec["encoders"] = ENC_REL.format(seed=seed, tag=kw.pop("enc_tag", ""))
    spec.update(kw)
    from .run import run_id_of
    spec["run_id"] = run_id_of(spec)
    return spec


def search_specs() -> List[Dict]:
    specs = []
    for arm in ("B0", "B1", "B1s7"):
        for M, s, r in itertools.product(BASE_GRID["M"], BASE_GRID["s"], BASE_GRID["T_ratio"]):
            specs.append(trial_spec("search", arm, TUNING_SEED, {"M": M, "s": s, "T_ratio": r}))
    for M, s, t in itertools.product(N0_GRID["M"], N0_GRID["s"], N0_GRID["temperature"]):
        specs.append(trial_spec("search", "N0", TUNING_SEED, {"M": M, "s": s, "T_ratio": 0.5}, t))
    return specs


def _done(stage: str, run_id: str) -> bool:
    p = RESULTS / stage / f"{run_id}.json"
    return p.exists() and read_json(p).get("status") == "completed"


def _cost(job: Dict) -> float:
    if job["kind"] == "encoders":
        return 1e9  # first: other jobs wait for them
    s = job["spec"]
    M = s["head"]["M"]
    arm = s["arm"]
    lit = {"B0": 1568, "B0M": 1568, "B1": 182 * 484, "B1s7": 110 * 16, "A2": 3488, "A2M": 3488}.get(arm, 6848)
    return M * lit


def run_queue(jobs: List[Dict], gpus: List[int], per_gpu: int, tag: str) -> List[Dict]:
    """Run jobs on ``per_gpu`` slots per GPU. Encoder jobs gate the trials that need them."""
    LOGS.mkdir(parents=True, exist_ok=True)
    pending = sorted(jobs, key=_cost, reverse=True)
    slots = {(g, i): None for g in gpus for i in range(per_gpu)}
    finished = []
    qlog = open(LOGS / f"queue_{tag}.log", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        qlog.write(line + "\n")
        qlog.flush()

    say(f"queue {tag}: {len(pending)} jobs on {gpus} x {per_gpu}")
    while pending or any(v is not None for v in slots.values()):
        for key, cur in list(slots.items()):
            if cur is not None and cur[0].poll() is not None:
                proc, job, t0 = cur
                say(f"rc={proc.returncode} {job['id']} on gpu{key[0]} ({(time.time() - t0) / 60:.1f} min)")
                finished.append({**job, "rc": proc.returncode})
                slots[key] = None
        for key in [k for k, v in slots.items() if v is None]:
            ready = [j for j in pending if all((REPO / EXP.relative_to(REPO) / n).exists() for n in j.get("needs", []))]
            if not ready:
                break
            # B1 stride-1 jobs prefer the larger GPU 0
            job = ready[0]
            if key[0] != gpus[0]:
                small = [j for j in ready if not (j["kind"] == "trial" and j["spec"]["arm"] == "B1"
                                                  and j["spec"]["head"]["M"] >= 2000)]
                if small:
                    job = small[0]
            pending.remove(job)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(key[0]), "PYTHONUNBUFFERED": "1"}
            log = open(LOGS / f"{job['id']}.log", "w")
            proc = subprocess.Popen(job["cmd"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            slots[key] = (proc, job, time.time())
            say(f"start {job['id']} on gpu{key[0]} ({len(pending)} pending)")
        if pending and all(v is None for v in slots.values()):
            missing = {n for j in pending for n in j.get("needs", [])}
            say(f"deadlock: {len(pending)} jobs wait for {missing}")
            break
        time.sleep(5)
    say(f"queue {tag} finished: {sum(j['rc'] == 0 for j in finished)}/{len(finished)} rc=0")
    return finished


def encoder_job(seed: int, tag: str = "", manifest: str = D.MAIN, cfg: Optional[Dict] = None) -> Dict:
    cmd = [sys.executable, "-m", "experiments.transformer_like.code.run", "--encoders", "--seed", str(seed),
           "--tag", tag, "--manifest", manifest, "--device", "cuda"]
    if cfg:
        cmd += ["--encoder-cfg", json.dumps(cfg)]
    return {"kind": "encoders", "id": f"encoders_seed{seed}{tag}", "cmd": cmd}


def trial_job(spec: Dict) -> Dict:
    p = RESULTS / "specs" / spec["stage"] / f"{spec['run_id']}.json"
    write_json(p, spec)
    job = {"kind": "trial", "id": f"{spec['stage']}__{spec['run_id']}", "spec": spec,
           "cmd": [sys.executable, "-m", "experiments.transformer_like.code.run", "--spec", str(p), "--device", "cuda"]}
    if "encoders" in spec:
        job["needs"] = [spec["encoders"]]
    return job


def jobs_for(specs: List[Dict], encoder_seeds: List[int], enc_tag: str = "", manifest: str = D.MAIN,
             enc_cfg: Optional[Dict] = None) -> List[Dict]:
    jobs = []
    for seed in encoder_seeds:
        rec = RESULTS / "encoders" / f"seed{seed}{enc_tag}.json"
        if not (rec.exists() and (ART / "encoders" / f"seed{seed}{enc_tag}.pt").exists()):
            jobs.append(encoder_job(seed, enc_tag, manifest, enc_cfg))
    jobs += [trial_job(s) for s in specs if not _done(s["stage"], s["run_id"])]
    return jobs


# ---------------------------------------------------------------------- probe
def probe(device: str = "cuda") -> None:
    """Phase 2 cost probe: 2 epochs of the largest configurations (records in results/probe)."""
    from .run import run_trial
    specs = [trial_spec("probe", "B0", TUNING_SEED, {"M": 2000, "s": 5.0, "T_ratio": 0.5}, max_epochs=2),
             trial_spec("probe", "B1s7", TUNING_SEED, {"M": 2000, "s": 5.0, "T_ratio": 0.5}, max_epochs=2),
             trial_spec("probe", "N0", TUNING_SEED, {"M": 1000, "s": 5.0, "T_ratio": 0.5}, 1.0, max_epochs=2),
             trial_spec("probe", "B1", TUNING_SEED, {"M": 2000, "s": 5.0, "T_ratio": 0.5}, max_epochs=2)]
    for s in specs:
        r = run_trial(s, device)
        if r["status"] == "completed":
            print(s["run_id"], [round(h["epoch_train_s"], 1) for h in r["fit"]["history"]], "s/epoch",
                  f"peak {r.get('peak_cuda_gb', 0):.2f} GB")


# ---------------------------------------------------------------------- search + freeze
def run_search(gpus: List[int], per_gpu: int) -> None:
    run_queue(jobs_for(search_specs(), [TUNING_SEED]), gpus, per_gpu, "search")
    b0m = b0m_specs()
    if b0m:
        run_queue(jobs_for(b0m, []), gpus, per_gpu, "search_b0m")
    # D22: equal-grid secondary (N0 head T/M = 0.25), then B0-M re-matched to its selection
    run_queue(jobs_for(s3_search_specs(), []), gpus, per_gpu, "search_s3")
    b0m3 = b0m_specs(s3=True)
    if b0m3:
        run_queue(jobs_for(b0m3, []), gpus, per_gpu, "search_s3_b0m")


def s3_search_specs() -> List[Dict]:
    return [trial_spec("search_s3", "N0", TUNING_SEED, {"M": M, "s": s, "T_ratio": 0.25}, t)
            for M, s, t in itertools.product(N0_GRID["M"], N0_GRID["s"], N0_GRID["temperature"])]


def select_s3_n0() -> Dict:
    recs = [r for st in ("search", "search_s3") for r in _records(st, "N0") if r["status"] == "completed"]
    recs.sort(key=lambda r: (-r["valid"]["accuracy"], r["sizes"]["stored_bytes"], r["fit"]["best_epoch"], r["run_id"]))
    return recs[0]


def _records(stage: str, arm: str) -> List[Dict]:
    out = []
    for p in sorted((RESULTS / stage).glob(f"{arm}_*.json")):
        r = read_json(p)
        if r["arm"] == arm:
            out.append(r)
    return out


def select(arm: str, stage: str = "search") -> Dict:
    """Spec §11 key: highest val accuracy, smaller persistent bytes, earlier best epoch, config id."""
    recs = [r for r in _records(stage, arm) if r["status"] == "completed"]
    if not recs:
        raise RuntimeError(f"no completed {arm} trials")
    recs.sort(key=lambda r: (-r["valid"]["accuracy"], r["sizes"]["stored_bytes"], r["fit"]["best_epoch"], r["run_id"]))
    return recs[0]


def b0m_specs(s3: bool = False) -> List[Dict]:
    try:
        n0 = select_s3_n0() if s3 else select("N0")
    except (RuntimeError, IndexError):
        return []
    M = matched_M(n0["sizes"]["automata"]["total"], 784)
    stage = "search_s3" if s3 else "search"
    if s3 and M == matched_M(select("N0")["sizes"]["automata"]["total"], 784):
        return []  # same budget as the primary: the primary B0-M selection is reused
    return [trial_spec(stage, "B0M", TUNING_SEED, {"M": M, "s": s, "T_ratio": r})
            for s, r in itertools.product(BUDGET_RATIOS["s"], BUDGET_RATIOS["T_ratio"])]


def freeze() -> Dict:
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists; the protocol is frozen")
    ledger = {}
    for arm in ("B0", "B1", "B1s7", "N0", "B0M"):
        recs = _records("search", arm)
        ledger[arm] = [{"run_id": r["run_id"], "status": r["status"],
                        "val_accuracy": r.get("valid", {}).get("accuracy"),
                        "best_epoch": r.get("fit", {}).get("best_epoch"),
                        "stored_bytes": r.get("sizes", {}).get("stored_bytes")} for r in recs]
    sel = {arm: select(arm) for arm in ("B0", "B1", "B1s7", "N0", "B0M")}
    n0 = sel["N0"]
    n0_total = n0["sizes"]["automata"]["total"]
    h = n0["spec"]["head"]
    configs = {arm: {"head": {k: r["spec"]["head"][k] for k in ("M", "s", "T_ratio")},
                     "temperature": r["spec"].get("temperature", 1.0), "selected_run": r["run_id"],
                     "val_accuracy": r["valid"]["accuracy"]} for arm, r in sel.items()}
    n0cfg = configs["N0"]
    for arm in ("A0", "A1", "A3", "A2"):
        configs[arm] = {"head": dict(n0cfg["head"]), "temperature": n0cfg["temperature"], "selected_run": None}
    # A2-M: head sized so that A2's total (its head + the same Q/K) matches N0's total
    enc = n0["sizes"]["automata"]["q"] + n0["sizes"]["automata"]["k"]
    a2m = matched_M(n0_total - enc, 1744)
    configs["A2M"] = {"head": {"M": a2m, "s": h["s"], "T_ratio": h["T_ratio"]},
                      "temperature": n0cfg["temperature"], "selected_run": None}
    # D22: equal-grid secondary
    n3 = select_s3_n0()
    h3 = n3["spec"]["head"]
    s3 = {"selected_N0_run": n3["run_id"], "same_as_primary": n3["run_id"] == n0["run_id"], "configs": {}}
    if not s3["same_as_primary"]:
        t3 = n3["spec"].get("temperature", 1.0)
        base3 = {"M": h3["M"], "s": h3["s"], "T_ratio": h3["T_ratio"]}
        for arm in ("N0", "A0", "A1", "A3", "A2"):
            s3["configs"][arm] = {"head": dict(base3), "temperature": t3}
        n3_total = n3["sizes"]["automata"]["total"]
        enc3 = n3["sizes"]["automata"]["q"] + n3["sizes"]["automata"]["k"]
        s3["configs"]["A2M"] = {"head": {**base3, "M": matched_M(n3_total - enc3, 1744)}, "temperature": t3}
        b3 = [r for r in _records("search_s3", "B0M") if r["status"] == "completed"]
        if b3:
            b3.sort(key=lambda r: (-r["valid"]["accuracy"], r["sizes"]["stored_bytes"], r["fit"]["best_epoch"], r["run_id"]))
            s3["configs"]["B0M"] = {"head": {k: b3[0]["spec"]["head"][k] for k in ("M", "s", "T_ratio")},
                                   "temperature": 1.0, "selected_run": b3[0]["run_id"]}
        s3["N0_total_automata"] = n3_total
        s3["val_accuracy"] = n3["valid"]["accuracy"]
        s3["ledger"] = [{"run_id": r["run_id"], "val_accuracy": r["valid"]["accuracy"]}
                        for r in _records("search_s3", "N0") + _records("search_s3", "B0M") if r["status"] == "completed"]
    frozen = {
        "experiment": "tm_qk_mnist_v1",
        "s3_equal_grid": s3, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_sha": git_sha(),
        "src_diff": src_diff(), "split": D.MAIN, "split_sha": D.read_manifest(D.MAIN)["sha256"],
        "tuning_seed": TUNING_SEED, "final_seeds": FINAL_SEEDS, "s1_seeds": S1_SEEDS,
        "checkpoint_rule": "earliest epoch with best validation accuracy; stop after 10 epochs without a strictly higher value; max 50",
        "batch_size": 32, "feedback_mode": "batch",
        "selection_key": "max val accuracy, min stored bytes, min best epoch, run_id",
        "memory_matching": {"N0_total_automata": n0_total, "B0M_M": sel["B0M"]["spec"]["head"]["M"], "A2M_M": a2m,
                            "B0M_total_automata": sel["B0M"]["sizes"]["automata"]["total"],
                            "A2M_total_automata": a2m * 10 * 2 * 1744 + enc},
        "configs": configs, "search_ledger": ledger,
        "test_procedure": "one pass per (arm, seed) inside the final run; timing afterwards, serially, on one GPU",
    }
    write_json(FROZEN, frozen)
    print(json.dumps(configs, indent=1))
    return frozen


# ---------------------------------------------------------------------- final benchmark
FINAL_ARMS = ("B0", "B1", "B1s7", "N0", "A0", "A1", "A2", "A2M", "A3", "B0M")


def final_specs(fr: Dict) -> List[Dict]:
    specs = []
    for seed in fr["final_seeds"]:
        for arm in FINAL_ARMS:
            c = fr["configs"][arm]
            specs.append(trial_spec("final", arm, seed, dict(c["head"]), c["temperature"], test=True,
                                    interventions=(arm == "N0")))
    return specs


def s1_specs(fr: Dict) -> List[Dict]:
    specs = []
    for seed in fr["s1_seeds"]:
        for arm in ("B0", "B1", "N0", "A1", "A2", "B0M"):  # A1/A2/B0M added by D23
            c = fr["configs"][arm]
            specs.append(trial_spec("secondary_s1", arm, seed, {**c["head"], "weighted": True}, c["temperature"],
                                    test=True, interventions=(arm == "N0")))
    return specs


def s2_specs(fr: Dict) -> List[Dict]:
    """B0 at its selected config on the 5k subset, batch-32 vs sequential feedback, 10 epochs."""
    c = fr["configs"]["B0"]
    out = []
    for fm in ("batch", "sequential"):
        out.append(trial_spec("secondary_s2", "B0", TUNING_SEED, {**c["head"], "feedback_mode": fm},
                              manifest=D.S2, max_epochs=10, patience=10))
    return out


def s3_final_specs(fr: Dict) -> List[Dict]:
    s3 = fr.get("s3_equal_grid", {})
    specs = []
    for seed in fr["final_seeds"]:
        for arm, c in s3.get("configs", {}).items():
            specs.append(trial_spec("secondary_s3", arm, seed, dict(c["head"]), c["temperature"], test=True,
                                    interventions=(arm == "N0")))
    return specs


def run_benchmark(manifest: str, gpus: List[int], per_gpu: int, stage: Optional[str] = None) -> None:
    if not Path(manifest).exists():
        raise SystemExit("benchmark refuses an unfrozen experiment: run freeze-experiment first")
    fr = read_json(manifest)
    if stage in (None, "final"):
        run_queue(jobs_for(final_specs(fr), fr["final_seeds"]), gpus, per_gpu, "final")
    if stage in (None, "secondary"):
        jobs = jobs_for(s3_final_specs(fr), [])
        jobs += jobs_for(s1_specs(fr), [])
        jobs += jobs_for(s2_specs(fr), [])
        run_queue(jobs, gpus, per_gpu, "secondary")
