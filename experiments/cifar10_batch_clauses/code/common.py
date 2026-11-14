"""Shared paths, data, grid and result records for the cifar10_batch_clauses programme.

Rules (../README.md): ``src/torchtsetlin/**`` is read-only here; every reported number comes from a
``results/<arm>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.
Plan: ``plans/cifar10-batch-clauses.md`` (decisions D0-D12 are referenced below).
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/cifar10_batch_clauses
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("CBC_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("CBC_CACHE", PROG / ".cache"))     # git-ignored: data, checkpoints, claims, W&B
LOGS = PROG / "logs"                                             # git-ignored
DATA = CACHE / "data"

# ---------------------------------------------------------------- data (D5)
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR_SHA256 = {                                    # the official pickles (checked on every load)
    "data_batch_1": "54636561a3ce25bd3e19253c6b0d8538147b0ae398331ac4a2d86c6d987368cd",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}
N_VAL = 5000                                         # D4: 45 000 train / 5 000 validation
SPLIT_SEED = 0

# ---------------------------------------------------------------- grid (plan §2)
MODELS = ["conv", "conv3", "flat"]
CLAUSES = [1000, 2000, 4000, 8000, 16000, 32000]                  # per class (D0)
CLAUSES_OF = {"conv": CLAUSES, "conv3": CLAUSES, "flat": [1000, 2000, 4000, 8000]}   # D2: flat memory
BATCHES = [1, 16, 32, 64, 128, 256, 512, 1024]
BATCHES_PRICED = [1, 32, 1024]
SEED = 0
C_REF = 2000
# D1: conv = the Optimized Toolbox's 3x3 colour thermometer specialist (arXiv:2406.00704 Table III;
# experiments/toolbox_cifar10); conv3 = the same model on 8 bit planes per colour, window depth 8
# (mnist_batch_clauses addendum / fmnist_bitplanes conv3 x bit8); flat = a flat TM on a 3-level
# colour thermometer, (T_ref, s) from the pilot (D3).
MAX_LIT = 32                                         # every Toolbox script; budget_mode="soft"
BASE = {
    "conv": dict(enc="therm8", T_ref=3000, s=5.0, weighted=True, boost_true_positive=True,
                 patch_size=3, stride=1, position_encoding=True, max_included_literals=MAX_LIT),
    "conv3": dict(enc="bit8", T_ref=3000, s=5.0, weighted=True, boost_true_positive=True,
                  patch_size=3, kd=8, stride=1, position_encoding=True, max_included_literals=MAX_LIT),
    "flat": dict(enc="therm3", T_ref=None, s=None, weighted=False, boost_true_positive=True,
                 max_included_literals=MAX_LIT),
}
# D3: flat pilot at C = 2 000, B = 32, <= 60 epochs; first entry wins ties (B4's MNIST values).
FLAT_PILOT = [(50, 10.0), (200, 10.0), (800, 10.0), (50, 5.0), (200, 5.0), (800, 5.0)]
PILOT_MAX_EPOCHS = 60
# D6: early stopping on validation accuracy, as mnist_batch_clauses v4.
EARLY = dict(patience=25, min_delta=0.0, min_epochs=30, max_epochs=500, time_cap_h=24.0)
# D11: sanity gate. conv C=2 000 (T = T_ref, so the Toolbox's own setting except batch/early
# stopping): Toolbox Table IV 3x3 thermometers at 2 000 = 63.6 % (250 epochs, tmu).
SANITY = dict(tag="conv_c2000_b32_Tscaled_s0", min_test=60.0)

FINAL = ("done", "oom")                              # record statuses that end a job


def T_of(model, clauses, t_mode, t_ref=None):
    """D7: T scaled in proportion to clauses per class (grid) or fixed at the reference."""
    t_ref = t_ref if t_ref is not None else BASE[model]["T_ref"]
    return int(round(t_ref * clauses / C_REF)) if t_mode == "scaled" else int(t_ref)


def tag_of(model, clauses, batch, t_mode, seed, sequential=False, pilot=None):
    t = f"{model}_c{clauses}_b{batch}_T{t_mode}{'_seq' if sequential else ''}"
    if pilot is not None:
        t += f"_Tref{pilot[0]}_spec{pilot[1]:g}"
    return f"{t}_s{seed}"


def flat_choice():
    """(T_ref, s) chosen by the flat pilot (results/pilot/choice.json), or None before that."""
    f = RESULTS / "pilot" / "choice.json"
    if f.exists():
        c = json.loads(f.read_text())
        return int(c["T_ref"]), float(c["s"])
    return None


def hyper(model, pilot=None):
    """(T_ref, s) of a model: the base values, the flat pilot's choice, or an explicit pilot point."""
    if pilot is not None:
        return pilot
    if model == "flat":
        ch = flat_choice()
        if ch is None:
            raise RuntimeError("flat needs the pilot choice first (results/pilot/choice.json)")
        return ch
    return BASE[model]["T_ref"], BASE[model]["s"]


# ---------------------------------------------------------------- jobs
def jobs(arm):
    """Job dicts for one arm: model, clauses, batch, t_mode, seed, sequential, pilot."""
    def j(m, c, b, t="scaled", seq=False, pilot=None):
        return dict(arm=arm, model=m, clauses=c, batch=b, t_mode=t, seed=SEED, sequential=seq,
                    pilot=pilot, tag=tag_of(m, c, b, t, SEED, seq, pilot))
    out = []
    if arm == "smoke":
        for m in MODELS:
            for c in (CLAUSES_OF[m][0], CLAUSES_OF[m][-1]):
                for b in (1, 32, 1024):
                    out.append(j(m, c, b, pilot=FLAT_PILOT[0] if m == "flat" else None))
    elif arm == "pricing":
        for m in MODELS:
            for c in (1000, 8000, CLAUSES_OF[m][-1]) if m != "flat" else (1000, 8000):
                for b in BATCHES_PRICED:
                    out.append(j(m, c, b, pilot=FLAT_PILOT[0] if m == "flat" else None))
    elif arm == "pilot":
        for p in FLAT_PILOT:
            out.append(j("flat", C_REF, 32, pilot=p))
    elif arm == "grid":
        for m in MODELS:
            for c in CLAUSES_OF[m]:
                for b in BATCHES:
                    out.append(j(m, c, b))
    elif arm == "control_T":            # D7: T fixed (at C_REF fixed == scaled, so larger C only)
        for m, cs in (("conv", (8000, 32000)), ("flat", (8000,))):
            for c in cs:
                for b in (1, 32, 256):
                    out.append(j(m, c, b, t="fixed"))
    elif arm == "control_seq":          # D8: exact sequential algorithm vs the B = 1 cells
        for m in MODELS:
            out.append(j(m, C_REF, 32, seq=True))
    else:
        raise KeyError(arm)
    return out


def est_cost(job):
    """Relative cost for longest-first ordering (pricing replaces the guess when present)."""
    m, c, b = job["model"], job["clauses"], job["batch"]
    priced = [p for p in jobs("pricing") if p["model"] == m and p["batch"] == min(BATCHES_PRICED, key=lambda x: abs(math.log2(x / b)))]
    priced.sort(key=lambda p: abs(math.log(p["clauses"] / c)))
    for p in priced[:1]:
        r = read_json(RESULTS / "pricing" / f"{p['tag']}.json")
        if r and r.get("train_s_per_epoch"):
            return (r["train_s_per_epoch"] + r["eval_s_per_epoch"]) * c / p["clauses"]
    per = {"conv": 1.0, "conv3": 1.0, "flat": 0.6}[m]
    return per * c * (3.0 if b == 1 else 1.0 + 0.3 * (b >= 512))


# ---------------------------------------------------------------- data
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def raw_path():
    return DATA / "cifar10_uint8.npz"


def load_raw():
    """CIFAR-10 as uint8 (N, 32, 32, 3) RGB + int64 labels (official 50 000 / 10 000), prepared by
    ``code/data.py``."""
    p = raw_path()
    if not p.exists():
        raise SystemExit(f"{p} missing: run ./setup.sh (or python code/data.py) first")
    with np.load(p) as d:
        return {k: d[k] for k in ("x_train", "y_train", "x_test", "y_test")}


def encode(enc, x):
    """uint8 (N, 32, 32, 3) -> bool. therm8 / therm3: colour thermometer, ``x >= k*255/(r+1)``
    for k = 1..r per channel (the Toolbox's ``color_thermometer``), (N, 3*r, 32, 32), channel-major
    (colour, level). bit8: 8 bit planes per colour, MSB first, (N, 3, 8, 32, 32) for the 3D conv."""
    if enc.startswith("therm"):
        r = int(enc[5:])
        levels = np.arange(1, r + 1) * 255 / (r + 1)
        bits = x[..., None] >= levels                                   # (N, 32, 32, 3, r)
        n = x.shape[0]
        return np.ascontiguousarray(bits.reshape(n, 32, 32, 3 * r).transpose(0, 3, 1, 2))
    if enc == "bit8":
        shifts = np.arange(7, -1, -1, dtype=np.uint8)
        bits = (x[..., None] >> shifts) & 1                             # (N, 32, 32, 3, 8)
        return np.ascontiguousarray(bits.astype(bool).transpose(0, 3, 4, 1, 2))
    raise KeyError(enc)


def load_data(enc):
    """Encoded CIFAR-10, split 45 000 train / 5 000 validation (fixed, stratified, D4) + test."""
    raw = load_raw()
    tr_idx, va_idx = split(raw["y_train"])
    xtr = encode(enc, raw["x_train"])
    return dict(x_train=xtr[tr_idx], y_train=raw["y_train"][tr_idx], x_val=xtr[va_idx],
                y_val=raw["y_train"][va_idx], x_test=encode(enc, raw["x_test"]), y_test=raw["y_test"],
                sha256=CIFAR_SHA256, enc=enc)


def split(y):
    """Stratified validation split, the same for every run; saved to results/split.json and
    re-checked on every load (index hash)."""
    rng = np.random.default_rng(SPLIT_SEED)
    per_class = N_VAL // 10
    va = np.sort(np.concatenate([rng.permutation(np.flatnonzero(y == k))[:per_class]
                                 for k in range(10)]))
    tr = np.setdiff1d(np.arange(len(y)), va)
    h = hashlib.sha256(va.astype(np.int64).tobytes()).hexdigest()
    f = RESULTS / "split.json"
    if f.exists():
        if json.loads(f.read_text())["val_sha256"] != h:
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
         "host": platform.node(), "cuda": torch.version.cuda,
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "gpu_slot": os.environ.get("CBC_GPU")}
    try:
        import triton
        e["triton"] = triton.__version__
    except ImportError:  # pragma: no cover
        e["triton"] = None
    if device is not None and torch.device(device).type == "cuda":
        e["gpu"] = torch.cuda.get_device_name(torch.device(device))
        e["gpu_mem_gib"] = round(torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30, 1)
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


def status_of(arm, tag):
    r = read_json(RESULTS / arm / f"{tag}.json")
    return None if r is None else r.get("status")
