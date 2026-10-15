"""A file-based job queue so several workers per GPU can share one sweep.

    # write a job list (one JSON object per line)
    python -m experiments.IMN.code.jobs make --name main --stage main --dataset tier1 \
        --arm lr dt rf gbdt tm tabresnet imn tm_gated clause_lr --seed 0 1 2 3 4
    python -m experiments.IMN.code.jobs make --name main --append --stage main --dataset tier1 \
        --arm imn_tm imn_tm_rand imn_hybrid --seed 0 1 2 3 4 --nn '{"l1_mode": "prior"}'

    # start workers (each claims the next unclaimed job; records that exist are skipped)
    CUDA_VISIBLE_DEVICES=0 python -m experiments.IMN.code.jobs work --name main --worker 0

A job is claimed by creating ``.claims/<name>/<job-id>`` with ``O_EXCL`` (atomic). A worker that
dies leaves its claim behind; delete ``.claims/<name>`` and restart the workers to retry --
finished jobs are skipped because their records exist. Failures are written into the claim file
and the worker moves on. Jobs run in file order; ``make`` puts the expensive datasets first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from . import data as D

EXP_DIR = Path(__file__).resolve().parents[1]
JOB_DIR = EXP_DIR / "jobs"
CLAIM_DIR = EXP_DIR / ".claims"

# rough relative cost (training rows) used only to order jobs: big datasets first
_COST_HINT = {1590: 29000, 41027: 27000, 41146: 3000, 41143: 1800, 1489: 3200, 3: 1900,
              40984: 1400, 1067: 1300, 31: 600, 54: 500, 1464: 450, 40981: 400}


def job_id(j: dict) -> str:
    return hashlib.sha1(json.dumps(j, sort_keys=True).encode()).hexdigest()[:16]


def make(a) -> None:
    ds = []
    for d in a.dataset:
        ds += [str(i) for i in (D.TIER1 if d == "tier1" else D.TIER2 if d == "tier2" else [d])]
    ds.sort(key=lambda d: -_COST_HINT.get(int(d), 0) if d.lstrip("-").isdigit() else 0)
    jobs = [{"stage": a.stage, "dataset": d, "arm": arm, "seed": s, "nn": json.loads(a.nn),
             "tm": json.loads(a.tm), "tag": a.tag}
            for d in ds for s in a.seed for arm in a.arm]
    JOB_DIR.mkdir(exist_ok=True)
    path = JOB_DIR / f"{a.name}.jsonl"
    with open(path, "a" if a.append else "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"{'appended' if a.append else 'wrote'} {len(jobs)} jobs -> {path.relative_to(EXP_DIR)}")


def _claim(name: str, jid: str) -> bool:
    d = CLAIM_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(d / jid, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"pid={os.getpid()} start={time.strftime('%H:%M:%S')}\n".encode())
    os.close(fd)
    return True


def work(a) -> None:
    """Drain each named queue in turn (``--name a b c``)."""
    for name in a.name:
        _work_one(name, a)


def _work_one(name: str, a) -> None:
    import torch

    from .run import run_one
    a = argparse.Namespace(**{**vars(a), "name": name})
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = JOB_DIR / f"{a.name}.jsonl"
    if not path.exists():
        print(f"worker {a.worker}: no job file {path.name}, skipping", flush=True)
        return
    done = failed = 0
    while True:  # re-read the file so jobs appended later are picked up
        jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        todo = [(job_id(j), j) for j in jobs if not (CLAIM_DIR / a.name / job_id(j)).exists()]
        if not todo:
            break
        progressed = False
        for jid, j in todo:
            if not _claim(a.name, jid):
                continue
            progressed = True
            try:
                run_one(j["stage"], j["dataset"], j["arm"], j["seed"], device, j["nn"], j["tm"],
                        j.get("tag", ""), False, a.n_explain)
                done += 1
            except Exception:  # noqa: BLE001
                failed += 1
                tb = traceback.format_exc()
                print(f"FAILED {j}\n{tb}", file=sys.stderr, flush=True)
                with open(CLAIM_DIR / a.name / jid, "a") as f:
                    f.write("FAILED\n" + tb)
            break  # re-scan after every job so concurrent claims are respected in order
        if not progressed:
            break
    print(f"worker {a.worker}: {a.name}: {done} jobs run, {failed} failed", flush=True)


def reclaim(a) -> None:
    """Delete claims whose worker process is gone and that did not fail, so the jobs run again
    (jobs that finished are then skipped because their record exists)."""
    n = 0
    for name in a.name:
        for c in (CLAIM_DIR / name).glob("*") if (CLAIM_DIR / name).exists() else []:
            txt = c.read_text()
            if "FAILED" in txt:
                continue
            try:
                pid = int(txt.split("pid=")[1].split()[0])
                os.kill(pid, 0)
                continue  # still running
            except (ProcessLookupError, IndexError, ValueError):
                c.unlink()
                n += 1
    print(f"released {n} stale claims")


def status(a) -> None:
    path = JOB_DIR / f"{a.name}.jsonl"
    jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    claims = CLAIM_DIR / a.name
    n_claim = sum((claims / job_id(j)).exists() for j in jobs)
    n_fail = sum((claims / job_id(j)).exists() and "FAILED" in (claims / job_id(j)).read_text() for j in jobs)
    print(f"{a.name}: {len(jobs)} jobs, {n_claim} claimed (incl. running), {n_fail} failed")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--name", required=True)
    m.add_argument("--stage", required=True)
    m.add_argument("--dataset", nargs="+", required=True)
    m.add_argument("--arm", nargs="+", required=True)
    m.add_argument("--seed", nargs="+", type=int, default=[0])
    m.add_argument("--nn", default="{}")
    m.add_argument("--tm", default="{}")
    m.add_argument("--tag", default="")
    m.add_argument("--append", action="store_true")
    w = sub.add_parser("work")
    w.add_argument("--name", nargs="+", required=True)
    w.add_argument("--worker", type=int, default=0)
    w.add_argument("--device", default=None)
    w.add_argument("--n-explain", type=int, default=500)
    r = sub.add_parser("reclaim")
    r.add_argument("--name", nargs="+", required=True)
    s = sub.add_parser("status")
    s.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    {"make": make, "work": work, "status": status, "reclaim": reclaim}[a.cmd](a)


if __name__ == "__main__":
    main()
