"""Watchdog: verify Block 2's reproducibility cells the moment they land, and stop the queue
if they do not match.

    setsid nohup python code/repro_guard.py > logs/repro_guard.log 2>&1 < /dev/null &

`ctm-small-T80-s10-{unc,b32}` re-run the pre-existing `ctm-small-T80-{unc,b32}` cells at the
same seeds on the same card (`--s 10.0` is the arm default, so the configuration is byte-for-byte
the one already measured). Five harness files changed on 2026-09-21 -- `diagnostics.py`,
`arms.py`, `data.py`, `run_arm.py`, `record.py` -- and the new diagnostics were measured
**RNG-neutral** on both devices (AUDIT A24). That licenses a tolerance of **zero**, not the
1.00 pp seed band: the re-run must return the same numbers to the digit.

**Why this is a program and not a note to myself.** The first cell lands about six minutes into
the wave, at roughly 22:56, and the orchestrator's instruction is that nothing else may run if it
fails. A check that depends on somebody being awake at 22:56 is not a check. This one:

* compares only quantities that MUST be identical -- the full `train_acc` / `val_acc` curve,
  `selected_epoch`, `val_acc`, `test_acc`, and `test_predictions_sha`, which is a sha1 over all
  10 000 test predictions and is the sharpest single discriminator available;
* ignores quantities that legitimately differ -- `wall_s`, `epoch_s`, `started`, `argv`, the
  label, `hp.task` (a key that did not exist before today) and the new diagnostic fields;
* on mismatch, **terminates the launcher and the queue** (SIGTERM, launcher first so wave 2
  cannot start), writes `logs/REPRO_FAILURE.md`, and leaves the partial records in place for
  inspection. It never touches GPU 0's ladder or the CNN queue: it verifies the lock owner's
  command line names this job file before signalling anything.

Exit codes: 0 = both pairs verified identical · 2 = MISMATCH, queue stopped · 3 = timed out
waiting · 4 = the queue died on its own before producing the cells.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import signal
import subprocess
import time
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
JOBFILE = "p3_block2_s_sweep.txt"

# (re-run label, the record it must reproduce) -- checked for every seed that appears.
PAIRS = [("ctm-small-T80-s10-unc", "ctm-small-T80-unc"),
         ("ctm-small-T80-s10-b32", "ctm-small-T80-b32")]
SEEDS = (0, 1, 2)
DEADLINE_S = 6 * 3600
POLL_S = 20


def _p(label: str, seed: int) -> str:
    return os.path.join(RESULTS, f"{label}_seed{seed}.json")


def _log(msg: str) -> None:
    print(f"[repro-guard {dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def compare(new: dict, old: dict) -> List[str]:
    """`[]` means bit-identical on every quantity that must not have moved."""
    bad: List[str] = []
    for k in ("selected_epoch", "test_predictions_sha"):
        if new.get(k) != old.get(k):
            bad.append(f"{k}: {new.get(k)!r} != {old.get(k)!r}")
    for k in ("val_acc", "test_acc"):
        a, b = float(new.get(k, -1)), float(old.get(k, -2))
        if a != b:
            bad.append(f"{k}: {a!r} != {b!r}  (delta {100*(a-b):+.4f} pp)")
    ca, cb = new.get("curve", []), old.get("curve", [])
    if len(ca) != len(cb):
        bad.append(f"curve length: {len(ca)} != {len(cb)}")
    else:
        for i, (x, y) in enumerate(zip(ca, cb)):
            for k in ("train_acc", "val_acc"):
                if float(x[k]) != float(y[k]):
                    bad.append(f"curve[{i}].{k}: {x[k]!r} != {y[k]!r}")
    # A budget-matching cross-check, free: the hyperparameters that are not the sweep's axes.
    for k in ("n_clauses", "patch", "stride", "T", "s", "batch_size", "booleanization",
              "epochs", "max_included_literals", "feedback_mode", "weighted",
              "n_clauses_per_class", "vote_capacity", "chunk_size_at_batch"):
        if new.get("hp", {}).get(k) != old.get("hp", {}).get(k):
            bad.append(f"hp.{k}: {new.get('hp', {}).get(k)!r} != {old.get('hp', {}).get(k)!r}")
    if (new.get("env") or {}).get("gpu") != (old.get("env") or {}).get("gpu"):
        bad.append(f"env.gpu: {(new.get('env') or {}).get('gpu')!r} != "
                   f"{(old.get('env') or {}).get('gpu')!r}  (DR-001: device+seed is the key)")
    return bad


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def lock_owner() -> Optional[int]:
    lock = os.path.join(LOGS, ".gpu1.lock")
    try:
        pid = int(open(lock).read().strip())
    except (OSError, ValueError):
        return None
    return pid if os.path.exists(f"/proc/{pid}") else None


def stop_queue() -> List[str]:
    """SIGTERM the launcher, then the queue, then the running job. Nothing else."""
    killed = []
    # 1. the launcher shell, first, so wave 2 cannot start when the queue exits.
    try:
        out = subprocess.check_output(["pgrep", "-f", "launch_p3_block2.sh"]).decode().split()
        for pid in out:
            os.kill(int(pid), signal.SIGTERM)
            killed.append(f"launcher pid {pid}")
    except (subprocess.CalledProcessError, ProcessLookupError, OSError):
        pass
    # 2. the queue that owns the GPU-1 lock -- but only if it is OUR job file. GPU 0's ladder
    #    and the CNN queue must never be touched by this.
    pid = lock_owner()
    if pid is not None:
        cmd = _cmdline(pid)
        if JOBFILE in cmd:
            try:
                kids = subprocess.check_output(["pgrep", "-P", str(pid)]).decode().split()
            except (subprocess.CalledProcessError, OSError):
                kids = []
            for k in kids:
                try:
                    os.kill(int(k), signal.SIGTERM)
                    killed.append(f"run_arm pid {k}")
                except (ProcessLookupError, OSError):
                    pass
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(f"queue pid {pid}")
            except (ProcessLookupError, OSError):
                pass
        else:
            killed.append(f"NOT killed: gpu1 lock pid {pid} is not {JOBFILE} ({cmd[:120]})")
    return killed


def write_failure(report: List[Tuple[str, int, List[str]]], killed: List[str]) -> str:
    path = os.path.join(LOGS, "REPRO_FAILURE.md")
    with open(path, "w") as f:
        f.write("# REPRODUCIBILITY CHECK FAILED — the queue was stopped\n\n")
        f.write(f"**{dt.datetime.now().isoformat(timespec='seconds')}**, `code/repro_guard.py`.\n\n")
        f.write("`ctm-small-T80-s10-{unc,b32}` re-ran `ctm-small-T80-{unc,b32}` at the same "
                "seeds on the same card, under a configuration that differs in **nothing** "
                "(`--s 10.0` is the arm default). The new diagnostics were measured RNG-neutral "
                "on both devices (AUDIT A24), so the tolerance here is **zero**, not the seed "
                "band. They did not match.\n\n")
        f.write("**What this means**: one of the five harness files changed on 2026-09-21 — "
                "`diagnostics.py`, `arms.py`, `data.py`, `run_arm.py`, `record.py` — perturbs "
                "training. **Every arm run after 2026-09-21 is suspect**, and so is any "
                "comparison that mixes arms from before and after.\n\n")
        f.write("**Do not restart the queue.** Diagnose first; the partial records are left in "
                "place.\n\n## Differences\n\n")
        for label, seed, bad in report:
            f.write(f"### `{label}` seed {seed}\n\n")
            for m in bad:
                f.write(f"* {m}\n")
            f.write("\n")
        f.write("## Processes signalled\n\n")
        for k in killed or ["(none)"]:
            f.write(f"* {k}\n")
        f.write("\n## First things to check\n\n"
                "1. `curve[0].train_acc` — if epoch 1 already differs, the divergence is at "
                "initialisation or in the first update, not in the diagnostics.\n"
                "2. `hp` diffs above — a configuration change would show here and would mean "
                "this is not an RNG problem at all.\n"
                "3. Re-run the CUDA RNG-neutrality test of AUDIT A24 on *this* arm size: train "
                "the same arm with `--no-diagnostics` and with them, compare the `ta_state` "
                "sha1. That isolates the diagnostics from the other four files.\n")
    return path


def main() -> int:
    _log(f"watching {RESULTS} for {[p[0] for p in PAIRS]} (deadline {DEADLINE_S}s)")
    seen: set = set()
    failures: List[Tuple[str, int, List[str]]] = []
    t0 = time.time()
    todo = [(new, old, s) for new, old in PAIRS for s in SEEDS
            if os.path.exists(_p(old, s))]
    _log(f"{len(todo)} (label, seed) pairs to verify")
    while time.time() - t0 < DEADLINE_S:
        for new, old, s in list(todo):
            if (new, s) in seen or not os.path.exists(_p(new, s)):
                continue
            time.sleep(1.0)                      # the writer uses os.replace; be polite anyway
            try:
                a, b = json.load(open(_p(new, s))), json.load(open(_p(old, s)))
            except (OSError, json.JSONDecodeError) as exc:
                _log(f"{new} seed {s}: not readable yet ({exc}); retrying")
                continue
            seen.add((new, s))
            bad = compare(a, b)
            if bad:
                _log(f"*** MISMATCH {new} seed {s} vs {old} seed {s}: {len(bad)} differences")
                for m in bad[:8]:
                    _log(f"      {m}")
                failures.append((new, s, bad))
            else:
                _log(f"OK  {new} seed {s} == {old} seed {s}  "
                     f"(test {a['test_acc']*100:.2f}, sha {a['test_predictions_sha'][:12]})")
        if failures:
            killed = stop_queue()
            path = write_failure(failures, killed)
            _log(f"QUEUE STOPPED. {killed}")
            _log(f"wrote {path}")
            return 2
        if len(seen) == len(todo) and todo:
            _log(f"PASS: all {len(seen)} reproducibility cells identical to the digit.")
            with open(os.path.join(LOGS, "repro_check_PASS.txt"), "w") as f:
                f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}  "
                        f"{len(seen)} cells verified bit-identical across the "
                        f"2026-09-21 harness change (AUDIT A25).\n")
                for new, old, s in todo:
                    f.write(f"  {new}_seed{s} == {old}_seed{s}\n")
            return 0
        # If the queue died without producing the cells, say so rather than waiting six hours.
        if lock_owner() is None and not subprocess.run(
                ["pgrep", "-f", "launch_p3_block2.sh"], capture_output=True).stdout:
            if len(seen) < len(todo):
                _log(f"queue and launcher are both gone with {len(seen)}/{len(todo)} verified")
                return 4
        time.sleep(POLL_S)
    _log(f"deadline reached with {len(seen)}/{len(todo)} verified")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
