"""Arms, grid, paths and records of the compgen programme (GitHub issue #15).

The compositional generalization benchmark of cair/PyHierarchicalTsetlinMachineCUDA
(``examples/CompositionalGeneralizationBenchmark.py``, was ``SequenceDemo.py``) on torchtsetlin:
does the port reach 100 % test accuracy, how does batching change the learning dynamics, and what
does the standard Tsetlin machine (both literal polarities) do on it?

Plan: ``torchtsetlin_plans/experiments/compgen/PLAN.md``. Rules (../README.md): ``src/torchtsetlin/**``
is read-only; every reported number comes from a ``results/<phase>/<tag>.json`` record.

This module must import without torch: ``run_ref.py`` runs in the reference library's environment.
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/compgen
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("CG_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("CG_CACHE", PROG / ".cache"))      # git-ignored: data, checkpoints, claims
MODELS = Path(os.environ.get("CG_MODELS", PROG / "models"))    # git-ignored
FIGURES = PROG / "figures"
LOGS = PROG / "logs"
REF_DIR = Path(os.environ.get("CG_REF_DIR", CACHE / "ref" / "PyHierarchicalTsetlinMachineCUDA"))

# ---------------------------------------------------------------- the benchmark (issue #15)
# python3 ./examples/SequenceDemo.py --sequence-length 5 --s 25.0 --T 10 --noise 0.01
#   --number-of-examples 40000 --number-of-state-bits 8 --q 1.0 --clause-weight-update-ratio 1.0
#   --literal-budget 100 --number-of-or-alternatives 1 --number-of-clauses 200 --vanilla
#   --number-of-synonyms 5 --and-group-normalization --negate-input --epochs 10000 --generate-data
REF_REPO = "https://github.com/cair/PyHierarchicalTsetlinMachineCUDA.git"
REF_COMMIT = "226d8b64"                              # 2026-11-20, examples/CompositionalGeneralizationBenchmark.py
REF_SCRIPT = "examples/CompositionalGeneralizationBenchmark.py"
DATA = dict(sequence_length=5, number_of_synonyms=5, number_of_examples=40000, noise=0.01)
N_FEATURES = DATA["sequence_length"] ** 2 * DATA["number_of_synonyms"]            # 125
MODEL = dict(clauses=200, T=10, s=25.0, literal_budget=100, state_bits=8)
# What the issue reports for this setting (data statistics of its run, and the outcome).
ISSUE = dict(unique_total=375000, unique_train=29424, unique_test=29369, unique_both=9044, unseen_test=20325,
             test_acc_epoch_493=98.22)

# ---------------------------------------------------------------- design (plan §4-§6)
SEEDS = [0, 1, 2]
BATCHES = [8, 32, 128, 512]
FEEDBACKS = ["seq"] + [f"b{b}" for b in BATCHES]      # seq = one example at a time (the reference's mode)
BOTH_LARGE = [1000, 10000]                            # clauses per class of the larger standard TMs
STOP = dict(acc=99.9, hold=20, max_epochs=3000, time_cap_h=24.0)
MARKS = (90.0, 99.0, 99.9)                            # test accuracy reached and held for MARK_HOLD epochs
MARK_HOLD = 10
METRICS_DENSE_UNTIL, METRICS_EVERY = 200, 5           # all metrics every epoch, then every 5th
CKPT_EVERY = 25
SEQ_CHUNK = 1000                                      # examples per update() call in sequential mode
PRICE_EPOCHS = 3
PHASES = ["smoke", "check", "price", "p1", "p2"]
FINAL = ("done",)
PERCENTILES = (1, 5, 10, 50, 90, 95, 99)


def env_seeds():
    v = os.environ.get("CG_SEEDS", "")
    return [int(s) for s in v.split(",") if s.strip()] or SEEDS


def both_max_epochs():
    """Cap of the standard-TM runs (expected to hit it): CG_BOTH_MAX_EPOCHS=500 shortens phase 2."""
    return int(os.environ.get("CG_BOTH_MAX_EPOCHS") or STOP["max_epochs"])


def with_ref():
    """The reference-library runs (arm ``ref``) are off unless CG_REF=1 (user, 2026-11-23: only
    the torchtsetlin arms are run; the arm needs pycuda and a CUDA toolkit with a working nvcc)."""
    return os.environ.get("CG_REF", "0") == "1"


def tag_of(arm, feedback, seed, clauses=None):
    c = f"_c{clauses}" if clauses and clauses != MODEL["clauses"] // 2 else ""
    return f"{arm}{c}_{feedback}_s{seed}"


def job(phase, arm, feedback, seed, clauses=None, **extra):
    """arm: neg (torchtsetlin, negated literals only) | both (torchtsetlin, standard) | ref."""
    clauses = clauses or MODEL["clauses"] // 2          # per class (torchtsetlin: 2 classes)
    return dict(phase=phase, arm=arm, feedback=feedback, seed=seed, clauses=clauses,
                T=MODEL["T"] * clauses // (MODEL["clauses"] // 2),        # T scales with the clauses
                tag=tag_of(arm, feedback, seed, clauses), **extra)


def jobs(phase):
    seeds = env_seeds()
    if phase == "smoke":
        return [job(phase, "neg", f, 0) for f in ("seq", "b32")] + [job(phase, "both", "b32", 0)]
    if phase == "price":                              # seed 0, a few epochs each: s/epoch per cell
        out = [job(phase, "neg", f, 0) for f in FEEDBACKS] + [job(phase, "both", f, 0) for f in ("seq", "b32")]
        out += [job(phase, "both", "b32", 0, c) for c in BOTH_LARGE]
        return out + ([job(phase, "ref", "seq", 0)] if with_ref() else [])
    if phase == "p1":                                 # port and batching
        out = [job(phase, "neg", f, s) for f in FEEDBACKS for s in seeds]
        return out + ([job(phase, "ref", "seq", s) for s in seeds] if with_ref() else [])
    if phase == "p2":                                 # the standard TM
        return ([job(phase, "both", f, s) for f in ("seq", "b32") for s in seeds]
                + [job(phase, "both", "b32", s, c) for c in BOTH_LARGE for s in seeds])
    raise KeyError(phase)


def est_cost(j):
    """Seconds per epoch from the pricing record of the same cell (seed 0), for longest-first."""
    r = read_json(RESULTS / "price" / f"{tag_of(j['arm'], j['feedback'], 0, j['clauses'])}.json")
    if r and r.get("s_per_epoch"):
        return r["s_per_epoch"]
    return {"seq": 3.0}.get(j["feedback"], 1.0) * (5.0 if j["arm"] == "ref" else 1.0) * j["clauses"] / 100


# ---------------------------------------------------------------- records
def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def env(device=None):
    e = {"python": platform.python_version(), "host": platform.node(), "gpu_slot": os.environ.get("CG_GPU"),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    for mod in ("numpy", "scipy", "torch", "torchtsetlin", "triton", "pycuda"):
        try:
            e[mod] = __import__(mod).__version__ if mod != "pycuda" else __import__(mod).VERSION_TEXT
        except Exception:  # noqa: BLE001 - not every module exists in both environments
            pass
    if "torch" in e:
        import torch
        if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
            e["gpu"] = torch.cuda.get_device_name(torch.device(device))
            e["cuda"] = torch.version.cuda
    elif e.get("pycuda"):
        try:
            import pycuda.driver as drv
            e["gpu"] = drv.Device(0).name()
        except Exception:  # noqa: BLE001
            pass
    return e


def new_record(kind, device=None, **fields):
    return {"kind": kind, "argv": sys.argv, "git_sha": git_sha(), "env": env(device),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}


def write_json(rec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    tmp.replace(path)                                 # never leave a half-written record


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status_of(phase, tag):
    r = read_json(RESULTS / phase / f"{tag}.json")
    return None if r is None else r.get("status")


# ---------------------------------------------------------------- the curve -> statistics
def metrics_due(epoch):
    return epoch <= METRICS_DENSE_UNTIL or epoch % METRICS_EVERY == 0


def first_held(curve, threshold, hold=MARK_HOLD):
    """The first row of a stretch of ``hold`` epochs with test accuracy >= threshold, or None."""
    run = 0
    for i, c in enumerate(curve):
        run = run + 1 if c["test_acc"] >= threshold else 0
        if run == hold:
            return curve[i - hold + 1]
    return None


def should_stop(curve, max_epochs, elapsed_s, time_cap_h=STOP["time_cap_h"]):
    tail = curve[-STOP["hold"]:]
    if len(tail) == STOP["hold"] and all(c["test_acc"] >= STOP["acc"] for c in tail):
        return f"test accuracy >= {STOP['acc']} % for {STOP['hold']} epochs"
    if len(curve) >= max_epochs:
        return f"max epochs ({max_epochs})"
    if elapsed_s >= time_cap_h * 3600:
        return f"time cap ({time_cap_h:g} h)"
    return None


def finish(rec, reason):
    """Final statistics of a run: time to each accuracy mark (None = not reached)."""
    c = rec["curve"]
    marks = {}
    for thr in MARKS:
        row = first_held(c, thr)
        marks[f"{thr:g}"] = None if row is None else {"epoch": row["epoch"], "updates": row["updates"],
                                                       "train_s": row["train_s_total"]}
    with_m = [r for r in c if "coverage" in r]
    rec.update(status="done", stop_reason=reason, converged=reason.startswith("test accuracy"), marks=marks,
               epochs_run=len(c), test_acc_last=c[-1]["test_acc"], train_acc_last=c[-1]["train_acc"],
               test_acc_max=max(r["test_acc"] for r in c),
               test_acc_tail_mean=sum(r["test_acc"] for r in c[-20:]) / len(c[-20:]),
               train_s_per_epoch_mean=sum(r["train_s"] for r in c) / len(c),
               eval_s_per_epoch_mean=sum(r["eval_s"] for r in c) / len(c),
               metrics_last=with_m[-1] if with_m else None, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return rec


REF_COLUMNS = ("coverage", "test_acc", "train_acc", "precision_p1", "precision_p5", "precision_p50", "precision_p95",
               "precision_p99", "recall_p1", "recall_p5", "recall_p50", "recall_p95", "recall_p99", "precision_mean",
               "recall_mean", "max_precision_p1", "max_precision_p5", "max_precision_p50", "max_precision_p95",
               "max_precision_p99", "votes_mean")


def ref_line(run, row):
    """One line of the reference script's statistics file (23 columns), for its plotting tools."""
    return f"{run} {row['epoch'] - 1} " + " ".join(f"{row[k]:.2f}" for k in REF_COLUMNS)
