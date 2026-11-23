"""Run the programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu<K>.sh`` (one per GPU / tmux session). Each runner repeatedly takes the first
job that is not finished and not claimed, in this order (plan §5, §8):

  1. price   every cell for 3 epochs (seed 0) -> seconds per epoch, used to order the jobs
  2. p1      port and batching: arm ``neg`` at seq, b8, b32, b128, b512, all seeds, longest
             first (with CG_REF=1 also arm ``ref``, the reference library)
  3. p2      the standard TM: arm ``both`` at seq and b32, and at 1 000 and 10 000 clauses per class

Phases: ``--phases`` or CG_PHASES (default ``price,p1,p2``). CG_SEEDS=0 runs one seed, CG_REF=1
adds the reference-library runs (off by default), CG_BOTH_MAX_EPOCHS=500 shortens phase 2.

A job is claimed by atomically creating ``.cache/claims/<phase>/<tag>/`` (with owner.json). A
finished job is skipped. A crashed job leaves ``<tag>.failed-<runner>`` so this runner does not
retry it (another may). Restarting a runner removes its own stale claims (dead pid on this host),
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

from common import (
    CACHE,
    FINAL,
    LOGS,
    PROG,
    both_max_epochs,
    est_cost,
    jobs,
    read_json,
    status_of,
    write_json,
)

CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("CG_DEVICE", "cuda:0")
INTERRUPTED = {143, -15, 130, -2, 137, -9}          # SIGTERM / SIGINT / SIGKILL: not a failure
HOST = socket.gethostname()
ORDER = ["price", "p1", "p2"]
LONGEST_FIRST = ("p1", "p2")
SMOKE_ARGS = ["--subset", "4000", "--max-epochs", "3"]
REF_BASE = CACHE / "ref"


def ordered_jobs(phases):
    out = []
    for ph in ORDER:
        if ph in phases:
            js = jobs(ph)
            out += sorted(js, key=est_cost, reverse=True) if ph in LONGEST_FIRST else js
    return out


def ref_python():
    return REF_BASE / "venv" / "bin" / "python"


def ref_env():
    """The reference library compiles its kernels with nvcc at start-up."""
    e = dict(os.environ)
    root = REF_BASE / "cuda_root"
    if root.exists():
        r = root.read_text().strip()
        e["PATH"] = f"{r}/bin:{e.get('PATH', '')}"
        e["LD_LIBRARY_PATH"] = f"{r}/lib64:{e.get('LD_LIBRARY_PATH', '')}"
    return e


def command(job):
    """(argv, environment) of one job."""
    if job["arm"] == "ref":
        cmd = [str(ref_python()), "run_ref.py", "--phase", job["phase"], "--seed", str(job["seed"])]
        env = ref_env()
    else:
        cmd = [sys.executable, "train.py", "--phase", job["phase"], "--arm", job["arm"], "--feedback", job["feedback"],
               "--seed", str(job["seed"]), "--clauses", str(job["clauses"]), "--T", str(job["T"]), "--device", DEVICE]
        env = None
        if job["arm"] == "both" and job["phase"] == "p2":
            cmd += ["--max-epochs", str(both_max_epochs())]
    if job["phase"] == "smoke":
        cmd += SMOKE_ARGS
    return cmd, env


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


def claim(job, me):
    d = CLAIMS / job["phase"] / job["tag"]
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        return None
    write_json({"host": HOST, "pid": os.getpid(), "runner": me, "started": time.strftime("%c")}, d / "owner.json")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="GPU slot 0-3 (run_gpu.sh sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--sub", default="", help="second runner on the same GPU (CG_SLOTS=2): b")
    ap.add_argument("--phases", default=os.environ.get("CG_PHASES") or "price,p1,p2")
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
    os.environ["CG_GPU"] = str(args.gpu)
    me = f"gpu{args.gpu}{args.sub}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    clear_stale_claims()
    have_ref = ref_python().exists()
    print(f"== {me} on {HOST} (CUDA_VISIBLE_DEVICES={cvd}), phases {','.join(phases)}"
          + ("" if have_ref else "; no reference environment: `ref` jobs are skipped"), flush=True)

    idle = False
    while True:
        if not smoke:
            why = stop_reason()
            if why:
                print(f"== {me}: stopping ({why})", flush=True)
                return
        todo = jobs("smoke") if smoke else ordered_jobs(phases)
        pending, waiting = [], 0
        for j in todo:
            if status_of(j["phase"], j["tag"]) in FINAL:
                continue
            if j["arm"] == "ref" and not have_ref:
                continue
            if (CLAIMS / j["phase"] / f"{j['tag']}.failed-{me}").exists():
                continue
            if (CLAIMS / j["phase"] / j["tag"]).exists():
                waiting += 1
                continue
            pending.append(j)
        # the long phases are ordered by the pricing records: wait until pricing is complete
        if not smoke and "price" in phases and any(j["phase"] == "price" for j in todo
                                                   if status_of(j["phase"], j["tag"]) not in FINAL
                                                   and not (j["arm"] == "ref" and not have_ref)):
            pending = [j for j in pending if j["phase"] == "price"]
            waiting = max(waiting, 0 if pending else 1)
        if args.dry_run:
            for j in pending[:30]:
                print(j["phase"], j["tag"], f"est {est_cost(j):.1f} s/epoch")
            print(f"... {len(pending)} runnable now, {waiting} claimed elsewhere")
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            if waiting == 0:
                print(f"== {me}: nothing left to do", flush=True)
                return
            if not idle:
                idle = True
                print(f"== {me}: {waiting} job(s) running elsewhere; polling", flush=True)
            time.sleep(20)
            continue
        idle = False
        log = LOGS / "runs" / f"{job['phase']}__{job['tag']}.log"
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['phase']} {job['tag']}", flush=True)
        cmd, env = command(job)
        with open(log, "a") as fh:
            p = subprocess.Popen(cmd, cwd=PROG / "code", stdout=fh, stderr=subprocess.STDOUT, env=env)
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
        if rc != 0 and st not in FINAL:
            (CLAIMS / job["phase"] / f"{job['tag']}.failed-{me}").touch()
        shutil.rmtree(CLAIMS / job["phase"] / job["tag"], ignore_errors=True)


if __name__ == "__main__":
    main()
