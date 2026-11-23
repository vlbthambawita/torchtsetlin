"""Run the programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu<K>.sh`` (one per GPU / tmux session). Each runner repeatedly takes the first
job that is not finished and not claimed, in this order (plan §8):

  1. pricing   9 short timings (teams M, S, L at B = 1, 32, 1024) -> cost table, job ordering
  2. pilot     team M, B = 32, 40 epochs, the three augmentation policies; when all three are
               done, results/pilot/choice.json is written (best validation accuracy; within
               0.5 pp the stronger policy wins)
  3. grid      teams M and S x 7 batch sizes with the chosen policy, longest first
  4. ext       team M at B = 1 and 4        (only with CJT_PHASES=...,ext)
  5. large     team L at CJT_LARGE_BATCHES  (only with CJT_PHASES=...,large)
  6. cprice    clause sweep, 30 timed updates per size -> s/epoch and peak memory
  7. clauses   team M, B = 64, chosen policy, 16 000 ... 128 000 clauses per class and member
               (run_clauses_gpu<K>.sh; CJT_CLAUSES, CJT_CLAUSE_BATCH, CJT_T_MODE, CJT_TIME_CAP_H)
  8. lprice    longer training + drop clause: 30 timed updates per model -> s/epoch, peak memory
  9. long      team M, B = 64, geo+col, T scaled with the clauses, early stopping with patience 75
               and NO epoch limit (run_long_gpu<K>.sh). Stage 1: 32 000 clauses at drop_clause_p
               0, 0.25, 0.1, 0.5. Stage 2: 64 000 clauses at p = 0 (first free GPU) and at p*
               (when stage 1 is done: results/long/choice.json = the p > 0 with the best
               validation accuracy, within 0.5 pp the smaller p). CJT_LONG_PSTAR=0.25 fixes p* at
               once, CJT_LONG_BIG=none skips stage 2, CJT_LONG_PATIENCE=150 raises the patience
               and continues finished runs from their kept last checkpoint.

Phases: ``--phases`` or CJT_PHASES (default ``pricing,pilot,grid``).

A job is claimed by atomically creating ``.cache/claims/<phase>/<tag>/`` (with owner.json). A
finished job is skipped. A crashed job leaves ``<tag>.failed-gpu<K>`` so this GPU does not retry
it (another GPU may). Restarting a runner removes its own stale claims (dead pid on this host),
so interrupted runs resume from their checkpoints.

Stop gracefully: ``touch STOP`` in the programme folder (runners finish their current job and exit).

Usage: python code/queue_runner.py --gpu 0                    # through run_gpu0.sh
       python code/queue_runner.py --gpu 0 --phases smoke     # smoke test only (setup.sh)
       python code/queue_runner.py --gpu 0 --dry-run          # what this GPU would run next
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time

import torch
from common import (
    CACHE,
    CLAUSE_SWEEP,
    FINAL,
    LOGS,
    LONG,
    PILOT,
    POLICIES,
    PROG,
    RESULTS,
    est_cost,
    jobs,
    long_choice,
    long_patience,
    policy_choice,
    read_json,
    status_of,
    write_json,
)

SMOKE_ARGS = ["--subset", "2000", "--min-epochs", "1", "--max-epochs", "1", "--viz-every", "1"]
SMOKE_WANDB = {"M_b16_full_s0"}
CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("CJT_DEVICE", "cuda:0")
INTERRUPTED = {143, -15, 130, -2, 137, -9}          # SIGTERM / SIGINT / SIGKILL: not a failure
HOST = socket.gethostname()
ORDER = ["pricing", "pilot", "grid", "ext", "large", "cprice", "clauses", "lprice", "long"]
LONGEST_FIRST = ("grid", "ext", "large", "clauses")
NEEDS_POLICY = ("grid", "ext", "large", "cprice", "clauses")


def ordered_jobs(phases):
    out = []
    for ph in ORDER:
        if ph not in phases:
            continue
        js = jobs(ph)
        out += sorted(js, key=est_cost, reverse=True) if ph in LONGEST_FIRST else js
    return out


def waits_for_pilot(phases):
    return policy_choice() is None and any(p in phases for p in NEEDS_POLICY)


def choose_policy():
    """Once every pilot run is done: best validation accuracy; policies within ``tie_pp`` of the
    best count as tied and the strongest of them wins (POLICIES is ordered weak -> strong)."""
    if policy_choice() is not None:
        return
    recs = {j["policy"]: read_json(RESULTS / "pilot" / f"{j['tag']}.json") for j in jobs("pilot")}
    if any(r is None or r.get("status") != "done" for r in recs.values()):
        return
    val = {p: r["best_val_acc"] for p, r in recs.items()}
    top = max(val.values())
    pol = [p for p in POLICIES if val[p] >= top - PILOT["tie_pp"]][-1]
    write_json({"policy": pol, "best_val_acc": val, "test_acc_at_best": {p: r["test_acc_at_best"] for p, r in recs.items()},
                "rule": f"max validation accuracy (team {PILOT['team']}, B={PILOT['batch']}, {PILOT['epochs']} epochs); "
                        f"within {PILOT['tie_pp']} pp the stronger policy wins",
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, RESULTS / "pilot" / "choice.json")
    print(f"== pilot choice: augmentation policy {pol} (val {val})", flush=True)


def choose_drop():
    """Once the four stage-1 runs of ``long`` are done: p* = the p > 0 with the best validation
    accuracy; values within ``tie_pp`` of it are tied and the smallest wins. If no p > 0 comes
    within ``tie_pp`` of p = 0, drop clause did not help at this size and stage 2 still runs the
    fallback p (the question at the larger size stays open)."""
    if long_choice() is not None:
        return
    recs = {j["drop"]: read_json(RESULTS / "long" / f"{j['tag']}.json")
            for j in jobs("long") if j["clauses"] == LONG["clauses"]}
    if any(r is None or r.get("status") != "done" for r in recs.values()):
        return
    val = {p: r["best_val_acc"] for p, r in recs.items()}
    pos = {p: v for p, v in val.items() if p > 0}
    helped = max(pos.values()) >= val[0.0] - LONG["tie_pp"]
    p = min(q for q, v in pos.items() if v >= max(pos.values()) - LONG["tie_pp"]) if helped else LONG["fallback_drop"]
    write_json({"drop_clause_p": p, "helped_at_stage1": helped, "best_val_acc": {f"{q:g}": v for q, v in val.items()},
                "test_acc_at_best": {f"{q:g}": r["test_acc_at_best"] for q, r in recs.items()},
                "rule": f"p > 0 with the highest best-validation accuracy at {LONG['clauses']} clauses; within "
                        f"{LONG['tie_pp']} pp the smaller p wins; if none is within {LONG['tie_pp']} pp of p = 0, "
                        f"the fallback p = {LONG['fallback_drop']}",
                "written": time.strftime("%Y-%m-%dT%H:%M:%S")}, RESULTS / "long" / "choice.json")
    print(f"== long: stage 2 runs drop_clause_p = {p:g} (stage-1 validation accuracy {val})", flush=True)


def continuable(job):
    """A finished ``long`` run that a larger CJT_LONG_PATIENCE lets train on: it stopped by
    patience, its last checkpoint was kept, and the new patience is larger than the old one."""
    if job["phase"] != "long":
        return False
    r = read_json(RESULTS / "long" / f"{job['tag']}.json") or {}
    return (str(r.get("stop_reason", "")).startswith("no validation improvement")
            and long_patience() > r["hp"]["early_stopping"]["patience"]
            and (CACHE / "ckpt" / "long" / f"{job['tag']}.final.pt").exists())


def stop_reason():
    return "STOP file present" if (PROG / "STOP").exists() else None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_stale_claims():
    for d in CLAIMS.glob("*/*"):
        if not d.is_dir():
            continue
        o = read_json(d / "owner.json")
        if o and o.get("host") == HOST and not any(alive(int(o.get(k) or -1)) for k in ("pid", "child")):
            shutil.rmtree(d, ignore_errors=True)
            print(f"== released stale claim {d.parent.name}/{d.name} (pid {o.get('pid')} gone)", flush=True)


def claim(job, gpu):
    d = CLAIMS / job["phase"] / job["tag"]
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        return None
    write_json({"host": HOST, "pid": os.getpid(), "gpu": gpu, "started": time.strftime("%c")}, d / "owner.json")
    return d


def command(job):
    cmd = [sys.executable, "train.py", "--phase", job["phase"], "--team", job["team"], "--batch-size",
           str(job["batch"]), "--policy", job["policy"], "--seed", str(job["seed"]), "--device", DEVICE]
    if job.get("clauses"):                            # clause sweep: larger members, own tag suffix
        cmd += ["--clauses", str(job["clauses"]), "--t-mode", job["t_mode"],
                "--tag-suffix", job["tag"].split(f"_s{job['seed']}", 1)[1]]
    if job["phase"] == "clauses":                     # long epochs: checkpoint each one, longer cap
        cmd += ["--ckpt-every", "1", "--time-cap-h", os.environ.get("CJT_TIME_CAP_H") or str(CLAUSE_SWEEP["time_cap_h"])]
    if job.get("drop"):
        cmd += ["--drop-clause-p", f"{job['drop']:g}"]
    if job["phase"] == "long":                        # patience only: no epoch limit, safety time cap
        cmd += ["--patience", str(long_patience()), "--min-epochs", str(LONG["min_epochs"]), "--max-epochs", "0",
                "--ckpt-every", "1", "--time-cap-h", os.environ.get("CJT_TIME_CAP_H") or str(LONG["time_cap_h"]),
                "--train-eval", str(LONG["train_eval"]), "--keep-final", "--continue-final"]
    if job["phase"] == "lprice":
        cmd += ["--calibrate", str(LONG["price_updates"]), "--no-wandb"]
    if job["phase"] == "cprice":
        cmd += ["--calibrate", str(CLAUSE_SWEEP["price_updates"]), "--no-wandb"]
    if job["phase"] == "pricing":
        cmd += ["--calibrate", "300", "--no-wandb"]
    if job["phase"] == "smoke":
        cmd += SMOKE_ARGS
        if job["tag"] not in SMOKE_WANDB:
            cmd.append("--no-wandb")
    return cmd


def gpu_name(slot, cvd):
    try:
        return torch.cuda.get_device_name(0)
    except RuntimeError as e:
        try:
            smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e2:  # noqa: BLE001
            smi = f"nvidia-smi failed: {e2}"
        raise SystemExit(
            f"GPU slot {slot}: PyTorch sees no GPU with CUDA_VISIBLE_DEVICES={cvd} ({e}).\n"
            f"  outer CUDA_VISIBLE_DEVICES (before run_gpu.sh): {os.environ.get('CJT_OUTER_CVD') or 'unset'}\n"
            f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}\n"
            f"  nvidia-smi -L:\n    " + smi.replace("\n", "\n    ")) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="GPU slot 0-3 (run_gpu.sh sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--phases", default=os.environ.get("CJT_PHASES") or "pricing,pilot,grid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bad = [p for p in phases if p not in ORDER + ["smoke"]]
    if bad:
        raise SystemExit(f"unknown phase(s) {bad}; choose from {ORDER + ['smoke']}")
    smoke = phases == ["smoke"]

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if DEVICE != "cpu" and (not cvd or "," in cvd):
        raise SystemExit(f"start me through run_gpu{args.gpu}.sh: it gives this runner exactly one GPU "
                         f"(CUDA_VISIBLE_DEVICES is {cvd or 'unset'})")
    name = gpu_name(args.gpu, cvd) if DEVICE != "cpu" else "cpu (harness test)"
    os.environ["CJT_GPU"] = str(args.gpu)
    me = f"gpu{args.gpu}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    clear_stale_claims()
    print(f"== {me}: {name} on {HOST}, phases {','.join(phases)}", flush=True)

    idle = False
    while True:
        if not smoke:
            choose_policy()
            if "long" in phases:
                choose_drop()
            why = stop_reason()
            if why:
                print(f"== {me}: stopping ({why})", flush=True)
                return
        todo = jobs("smoke") if smoke else ordered_jobs(phases)
        pending, waiting = [], 0
        for j in todo:
            if status_of(j["phase"], j["tag"]) in FINAL and not continuable(j):
                continue
            if (CLAIMS / j["phase"] / f"{j['tag']}.failed-{me}").exists():
                continue
            if (CLAIMS / j["phase"] / j["tag"]).exists():
                waiting += 1
                continue
            pending.append(j)
        if not smoke and not pending and waits_for_pilot(phases):
            pilot_left = [j for j in jobs("pilot") if status_of("pilot", j["tag"]) not in FINAL]
            if "pilot" not in phases and pilot_left:
                print(f"== {me}: no augmentation policy yet (results/pilot/choice.json): run the pilot phase first",
                      flush=True)
                return
            if any((CLAIMS / "pilot" / j["tag"]).exists() for j in pilot_left):
                waiting += 1                      # the grid waits for the pilot running elsewhere
            elif pilot_left:
                print(f"== {me}: pilot run(s) failed on this GPU ({[j['tag'] for j in pilot_left]}); "
                      f"see logs/runs/", flush=True)
                return
        if args.dry_run:
            for j in pending[:20]:
                print(j["phase"], j["tag"], f"est {est_cost(j):.0f} s/epoch")
            print(f"... {len(pending)} runnable now, {waiting} claimed elsewhere or waiting"
                  + ("; grid follows the pilot" if waits_for_pilot(phases) else ""))
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            if waiting == 0:
                print(f"== {me}: nothing left to do", flush=True)
                return
            if not idle:
                idle = True
                print(f"== {me}: {waiting} job(s) running elsewhere or waiting for the pilot; polling", flush=True)
            time.sleep(60)
            continue
        idle = False
        log = LOGS / "runs" / f"{job['phase']}__{job['tag']}.log"
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['phase']} {job['tag']}", flush=True)
        with open(log, "a") as fh:
            p = subprocess.Popen(command(job), cwd=PROG / "code", stdout=fh, stderr=subprocess.STDOUT)
            owner = CLAIMS / job["phase"] / job["tag"] / "owner.json"
            write_json(dict(read_json(owner) or {}, child=p.pid), owner)
            try:
                rc = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                rc = p.wait()
        st = status_of(job["phase"], job["tag"])
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['tag']} exit {rc} status {st}", flush=True)
        if rc in INTERRUPTED:                   # stopped by the user: resumes on the next start
            shutil.rmtree(CLAIMS / job["phase"] / job["tag"], ignore_errors=True)
            print(f"== {me}: interrupted, exiting", flush=True)
            return
        if rc != 0 and (st not in FINAL or continuable(job)):
            (CLAIMS / job["phase"] / f"{job['tag']}.failed-{me}").touch()
        shutil.rmtree(CLAIMS / job["phase"] / job["tag"], ignore_errors=True)


if __name__ == "__main__":
    main()
