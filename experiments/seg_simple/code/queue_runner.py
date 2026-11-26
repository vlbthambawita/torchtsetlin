"""Run the programme on ONE GPU, sharing one job list with the runners of the other GPUs.

Started by ``run_gpu<K>.sh`` (one per GPU / tmux session). Each runner repeatedly takes the first
job of its phases that is not done, not claimed by another runner, and whose ``needs`` are done
(``jobs.py``): a team job waits for its members.

Phases: ``--phases`` or SS_PHASES, e.g. ``price``, then ``main``, ``side``, ``wide``, ``price3d``,
``conv3d``. Use the same value in every session.

A job is claimed by atomically creating ``.cache/claims/<tag>/`` (with owner.json). A crashed job
leaves ``<tag>.failed-<runner>`` so this runner does not retry it (another may). Restarting a
runner removes its own stale claims (dead pid on this host); interrupted runs resume from their
checkpoints. ``touch STOP`` (./stop.sh): training stops after the current epoch, runners exit.

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

import jobs as JOBS
from common import CACHE, LOGS, PROG, is_done, read_json, stop_requested, write_json

CLAIMS = CACHE / "claims"
DEVICE = os.environ.get("SS_DEVICE", "cuda:0")
INTERRUPTED = {143, -15, 130, -2, 137, -9}           # stopped by the user: not a failure
HOST = socket.gethostname()
SCRIPT = {"train": "train.py", "team": "evals.py"}


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def clear_stale_claims():
    for d in CLAIMS.glob("*"):
        if not d.is_dir():
            continue
        o = read_json(d / "owner.json")
        if o and o.get("host") == HOST and not any(alive(int(o.get(k) or -1)) for k in ("pid", "child")):
            shutil.rmtree(d, ignore_errors=True)
            print(f"== released stale claim {d.name} (pid {o.get('pid')} gone)", flush=True)


def retry_own_failures(me):
    """A restart tries again what failed on this runner before (the cause may have been fixed)."""
    old = sorted(CLAIMS.glob(f"*.failed-{me}"))
    for f in old:
        f.unlink(missing_ok=True)
    if old:
        print(f"== {me}: retrying {len(old)} job(s) that failed here before, e.g. {old[0].name.rsplit('.failed-', 1)[0]}",
              flush=True)


def summary(js, me):
    done = sum(1 for j in js if is_done(j["tag"]))
    failed = sorted({f.name.rsplit(".failed-", 1)[0] for f in CLAIMS.glob("*.failed-*")} & {j["tag"] for j in js})
    failed = [t for t in failed if not is_done(t)]
    line = f"{done} of {len(js)} listed job(s) done"
    if failed:
        line += f", {len(failed)} FAILED (logs/runs/<tag>.log), e.g. {failed[0]}"
    return line


def claim(job, me):
    d = CLAIMS / job["tag"]
    d.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(d)
    except FileExistsError:
        return None
    write_json({"host": HOST, "pid": os.getpid(), "runner": me, "started": time.strftime("%c")}, d / "owner.json")
    return d


def todo(phases, smoke):
    if smoke:
        return JOBS.smoke_jobs(), []
    return JOBS.all_jobs(phases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="GPU slot 0-3 (run_gpu.sh sets CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--sub", default="", help="further runners on the same GPU (run_gpu.sh): b, c, d")
    ap.add_argument("--phases", default=os.environ.get("SS_PHASES") or "price")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    bad = [p for p in phases if p not in JOBS.PHASES + ["smoke"]]
    if bad:
        raise SystemExit(f"unknown phase(s) {bad}; choose from {JOBS.PHASES + ['smoke']}")
    smoke = phases == ["smoke"]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if DEVICE != "cpu" and (not cvd or "," in cvd) and not args.dry_run:
        raise SystemExit(f"start me through run_gpu{args.gpu}.sh: it gives this runner exactly one GPU "
                         f"(CUDA_VISIBLE_DEVICES is {cvd or 'unset'})")
    me = f"gpu{args.gpu}{args.sub}"
    (LOGS / "runs").mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        clear_stale_claims()
        retry_own_failures(me)
        print(f"== {me} on {HOST} (CUDA_VISIBLE_DEVICES={cvd}), phases {','.join(phases)}", flush=True)

    idle = False
    while True:
        if stop_requested() and not smoke:
            print(f"== {me}: stopping (STOP file present)", flush=True)
            return
        js, blocked = todo(phases, smoke)
        pending, claimed, waiting = [], 0, []
        for j in js:
            if is_done(j["tag"]) or (CLAIMS / f"{j['tag']}.failed-{me}").exists():
                continue
            if (CLAIMS / j["tag"]).exists():
                claimed += 1
            elif JOBS.runnable(j):
                pending.append(j)
            else:
                waiting.append(j)
        if args.dry_run:
            for j in pending[:40]:
                print(f"{j['phase']:6s} {j['kind']:8s} {j['tag']}")
            print(f"... {len(pending)} runnable now, {claimed} running elsewhere, {len(waiting)} waiting for other jobs")
            for b in blocked:
                print(f"    later: {b}")
            return
        job = next((j for j in pending if claim(j, me)), None)
        if job is None:
            if claimed == 0:
                if waiting:
                    print(f"== {me}: {len(waiting)} job(s) wait for runs that failed or belong to a phase that is not "
                          f"open, e.g. {waiting[0]['tag']} needs "
                          f"{[t for t in waiting[0]['needs'] if not is_done(t)][:2]}", flush=True)
                for b in sorted(set(blocked)):
                    print(f"== {me}: later: {b}", flush=True)
                print(f"== {me}: nothing left to do in {','.join(phases)}: {summary(js, me)}", flush=True)
                return
            if not idle:
                idle = True
                print(f"== {me}: {claimed} job(s) running elsewhere; polling", flush=True)
            time.sleep(5 if smoke else 20)
            continue
        idle = False
        log = LOGS / "runs" / f"{job['tag']}.log"
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['phase']} {job['tag']}", flush=True)
        cmd = [sys.executable, SCRIPT[job["kind"]], "--tag", job["tag"], "--device", DEVICE]
        with open(log, "a") as fh:
            p = subprocess.Popen(cmd, cwd=PROG / "code", stdout=fh, stderr=subprocess.STDOUT)
            owner = CLAIMS / job["tag"] / "owner.json"
            write_json(dict(read_json(owner) or {}, child=p.pid), owner)
            try:
                rc = p.wait()
            except KeyboardInterrupt:
                p.terminate()
                rc = p.wait()
        done = is_done(job["tag"])
        print(f"== {me} {time.strftime('%m-%d %H:%M')}: {job['tag']} exit {rc} {'done' if done else 'NOT done'}", flush=True)
        if rc in INTERRUPTED:
            shutil.rmtree(CLAIMS / job["tag"], ignore_errors=True)
            print(f"== {me}: interrupted, exiting (the run resumes on the next start)", flush=True)
            return
        if rc != 0 and not done:
            (CLAIMS / f"{job['tag']}.failed-{me}").touch()
            print(f"== {me}: {job['tag']} FAILED, see {log}", flush=True)
        shutil.rmtree(CLAIMS / job["tag"], ignore_errors=True)


if __name__ == "__main__":
    main()
