"""Run a list of configs with a fixed number of worker processes per GPU.

    python code/jobq.py configs/e1.json --slots cuda:0=5,cuda:1=4

Skips configs whose record already exists, so it can be re-run after an interruption. A
phase cap in device-hours (``--cap``) stops *launching* once the summed wall time of this
phase's completed records plus the running jobs' elapsed time reaches it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "code"))
import run as R  # noqa: E402


def rid(cfg):
    c = {**R.DEFAULTS, **cfg}
    return cfg.get("run_id") or f"{c['phase']}_{c['dataset']}_{c['name']}_s{c['seed']}"


def phase_hours(phase):
    tot = 0.0
    d = os.path.join(HERE, "results")
    for f in os.listdir(d):
        if f.startswith(phase + "_") and f.endswith(".json"):
            with open(os.path.join(d, f)) as fh:
                tot += json.load(fh).get("wall_s", 0.0)
    return tot / 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--slots", default="cuda:0=5,cuda:1=4")
    ap.add_argument("--cap", type=float, default=None, help="device-hour cap for the phase")
    a = ap.parse_args()
    with open(a.file) as f:
        cfgs = json.load(f)
    slots = {}
    for s in a.slots.split(","):
        d, n = s.split("=")
        slots[d] = int(n)
    running = {d: [] for d in slots}
    def busy(c):  # running under another queue: its log was written in the last 20 minutes
        lp = os.path.join(HERE, "logs", rid(c) + ".log")
        return os.path.exists(lp) and time.time() - os.path.getmtime(lp) < 1200

    todo = [i for i, c in enumerate(cfgs)
            if not os.path.exists(os.path.join(HERE, "results", rid(c) + ".json")) and not busy(c)]
    print(f"{len(todo)} of {len(cfgs)} to run", flush=True)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    while todo or any(running.values()):
        for d in slots:
            running[d] = [(p, i, t) for p, i, t in running[d] if p.poll() is None]
        if a.cap is not None and todo:
            phase = {**R.DEFAULTS, **cfgs[todo[0]]}["phase"]
            used = phase_hours(phase) + sum(time.time() - t for d in running for _, _, t in running[d]) / 3600
            if used >= a.cap:
                print(f"cap {a.cap} h reached ({used:.2f} h); not launching {len(todo)} jobs", flush=True)
                todo = []
        for d, n in slots.items():
            while todo and len(running[d]) < n:
                i = todo.pop(0)
                if os.path.exists(os.path.join(HERE, "results", rid(cfgs[i]) + ".json")) or busy(cfgs[i]):
                    continue
                log = open(os.path.join(HERE, "logs", rid(cfgs[i]) + ".log"), "w")
                p = subprocess.Popen([sys.executable, os.path.join(HERE, "code", "run.py"), "--file", a.file,
                                      "--index", str(i), "--device", d], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
                running[d].append((p, i, time.time()))
                print(time.strftime("%H:%M:%S"), "launch", d, rid(cfgs[i]), flush=True)
        time.sleep(5)
    print("queue done", flush=True)


if __name__ == "__main__":
    main()
