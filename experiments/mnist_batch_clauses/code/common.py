"""Shared paths, data, grid and result records for the mnist_batch_clauses programme.

Rules (../README.md): ``src/torchtsetlin/**`` is read-only here; every reported number comes from a
``results/<arm>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.
Plan: ``plans/mnist-batch-clauses.md``.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/mnist_batch_clauses
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("MBC_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("MBC_CACHE", PROG / ".cache"))     # git-ignored: checkpoints, W&B dir
LOGS = PROG / "logs"                                             # git-ignored
PAPER = Path("/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper")

# The paper's own MNIST Booleanizations (experiments/paper_experiments/code/prepare_data.py b4 / b7),
# reused byte for byte; the sha256 is checked against that programme's data manifest.
PE = REPO / "experiments" / "paper_experiments"
DATA = {"flat": "B4", "conv": "B7"}

# ---------------------------------------------------------------- grid (plan §2-§3)
CLAUSES = [1000, 2000, 4000, 8000, 16000, 32000, 64000]       # per class (D0)
BATCHES = [1, 16, 32, 64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]
MODELS = ["flat", "conv"]
C_REF = 2000
# D1: B4 (flat) and B7 (conv) of the paper, T given at C_REF clauses/class.
BASE = {
    "flat": dict(T_ref=50, s=10.0, weighted=False, boost_true_positive=True),
    "conv": dict(T_ref=5000, s=5.0, weighted=True, boost_true_positive=True, patch_size=10,
                 stride=1, position_encoding=True),
}
# D3: early stopping on validation accuracy (D3a).
EARLY = dict(patience=25, min_delta=0.05, min_epochs=30, max_epochs=500, time_cap_h=24.0)
N_VAL = 10000
SPLIT_SEED = 0


def T_of(model, clauses, t_mode):
    """D2: T scaled in proportion to clauses per class (main grid) or fixed at the reference."""
    t_ref = BASE[model]["T_ref"]
    return t_ref * clauses // C_REF if t_mode == "scaled" else t_ref


def tag_of(model, clauses, batch, t_mode, seed, sequential=False):
    return f"{model}_c{clauses}_b{batch}_T{t_mode}{'_seq' if sequential else ''}_s{seed}"


def jobs(arm, seeds=SEEDS):
    """(model, clauses, batch, t_mode, seed, sequential) for each arm of the plan."""
    out = []
    if arm in ("grid", "smoke"):
        for seed in seeds:
            for m in MODELS:
                for c in CLAUSES:
                    for b in BATCHES:
                        out.append((m, c, b, "scaled", seed, False))
    elif arm == "control_T":            # D2: T fixed; C = 2k equals the grid cell, so 16k and 64k
        for m in MODELS:
            for c in (16000, 64000):
                for b in (1, 32, 256):
                    out.append((m, c, b, "fixed", 0, False))
    elif arm == "control_seq":          # D6: exact sequential algorithm vs the b1 cells
        for m in MODELS:
            for c in (2000, 16000):
                out.append((m, c, 32, "scaled", 0, True))
    else:
        raise KeyError(arm)
    return out


# ---------------------------------------------------------------- data
def load_data(model):
    """Booleanized MNIST, split 50 000 train / 10 000 validation (fixed, stratified, D3a) + test."""
    bench = DATA[model]
    path = PE / ".cache" / "data" / f"{bench}.npz"
    manifest = json.loads((PE / "results" / "data_manifest.json").read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != manifest[bench]["sha256"]:
        raise RuntimeError(f"{path}: sha256 {sha} does not match the paper_experiments manifest")
    with np.load(path) as d:
        xtr, ytr, xte, yte = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    tr_idx, va_idx = split(ytr)
    return dict(x_train=xtr[tr_idx], y_train=ytr[tr_idx], x_val=xtr[va_idx], y_val=ytr[va_idx],
                x_test=xte, y_test=yte, sha256=sha, bench=bench)


def split(y):
    """Stratified validation split, the same for every run; saved to results/split.json and
    re-checked on every load (index hash)."""
    rng = np.random.default_rng(SPLIT_SEED)
    per_class = N_VAL // 10
    va = np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == k))[:per_class]
                                 for k in range(10)]))
    tr = np.setdiff1d(np.arange(len(y)), va)
    h = hashlib.sha256(va.astype(np.int64).tobytes()).hexdigest()
    f = PROG / "results" / "split.json"
    if f.exists():
        saved = json.loads(f.read_text())
        if saved["val_sha256"] != h:
            raise RuntimeError("validation split differs from results/split.json")
    else:
        write_json({"seed": SPLIT_SEED, "n_val": int(len(va)), "n_train": int(len(tr)),
                    "stratified": True, "val_sha256": h, "val_indices": va.tolist()}, f)
    return tr, va


# ---------------------------------------------------------------- records
def git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "--", "src"]).returncode != 0
        return sha + ("+src-dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def env(device=None):
    import torchtsetlin
    e = {"python": sys.version.split()[0], "torch": torch.__version__,
         "torchtsetlin": torchtsetlin.__version__, "numpy": np.__version__,
         "host": platform.node(), "cuda": torch.version.cuda}
    if device is not None and torch.device(device).type == "cuda":
        e["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return e


def new_record(kind, device=None, **fields):
    return {"kind": kind, "argv": sys.argv, "git_sha": git_sha(), "env": env(device),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}


def write_json(rec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    tmp.replace(path)                                 # never leave a half-written record


def read_json(path):
    return json.loads(Path(path).read_text())
