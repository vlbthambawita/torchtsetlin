"""Shared paths, data, encodings, cells and records for the fmnist_bitplanes programme.

Plan: ``torchtsetlin_plans/plans/fmnist-bitplanes.md``. Rules (../README.md): ``src/torchtsetlin/**`` is read-only;
every reported number comes from a ``results/<arm>/<tag>.json`` record carrying ``argv``, ``seed``,
``git_sha`` and ``env``.
"""

import gzip
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
PROG = HERE.parent                                   # experiments/fmnist_bitplanes
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("FMB_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("FMB_CACHE", PROG / ".cache"))     # git-ignored: checkpoints, W&B dir
LOGS = PROG / "logs"                                             # git-ignored
# Best-validation-epoch models (plan fmnist-bitplanes-augmentation §3): git-ignored (models/.gitignore).
MODELS = Path(os.environ.get("FMB_MODELS", PROG / "models"))

# Fashion-MNIST as cached by paper_experiments (prepare_data.py keras_fashion_mnist / b8).
PE = REPO / "experiments" / "paper_experiments"
RAW = PE / ".cache" / "data" / "raw"
B8 = PE / ".cache" / "data" / "B8.npz"

# ---------------------------------------------------------------- design (plan §2)
ENCODINGS = {          # planes per pixel
    "agt": 1,          # the paper's B8: OpenCV adaptive Gaussian threshold (block 11, C=2)
    "therm8": 8,       # thermometer, 8 global uniform thresholds 255*k/9
    "bit8": 8,         # plain bit planes, MSB first
    "gray8": 8,        # Gray-coded bit planes, MSB first
    "bit4": 4,         # ablation: the top 4 bits
    "therm4": 4,       # ablation: 4 uniform thresholds 255*k/5
}
ARCHS = ["flat", "conv2", "conv3"]
# D1 (user, 2026-11-13): no 3D cell with the usual Booleanization -> conv3 only with bit codes.
CELLS = [("flat", e) for e in ("agt", "therm8", "bit8", "gray8")] + \
        [("conv2", e) for e in ("agt", "therm8", "bit8", "gray8")] + \
        [("conv3", e) for e in ("bit8", "gray8")]
CORE = {("flat", "agt"), ("conv2", "agt"), ("conv3", "bit8")}
# Staging (user, 2026-11-13): stage A runs the 3D conv cells only; stage B (the comparison cells)
# starts only after the user has checked the stage-A results.
STAGES = {"A": [c for c in CELLS if c[0] == "conv3"], "B": [c for c in CELLS if c[0] != "conv3"]}
CLAUSES = [2000, 8000]                                 # per class (D3)
C_REF = 2000
BATCH = 32                                             # D5
SEEDS = [0, 1, 2]
# conv3 window depth (user, 2026-11-13, plan v4): fixed kd = 8 = all bit planes in one window. With
# D = 8 there is a single depth position, so this conv3 equals a 2D conv over 8 bit-plane channels.
KDS = [8]                                              # pilot / main
KDS_PRICING = [4, 8]                                   # 4 = the optional real-3D check (plan §4)
KDS_SMOKE = [1, 2, 4, 8]
KD_DEFAULT = 8
# D2: flat = B4 (unweighted), conv2/conv3 = B8 (weighted, patch 10, position encoding).
BASE = {
    "flat": dict(T_ref=50, s=10.0, weighted=False, boost_true_positive=True),
    "conv2": dict(T_ref=5000, s=5.0, weighted=True, boost_true_positive=True, patch=10,
                  position_encoding=True),
    "conv3": dict(T_ref=5000, s=5.0, weighted=True, boost_true_positive=True, patch=10,
                  position_encoding=True, max_chunk_elements=2**30),
}
# conv3 max_chunk_elements (LIBRARY_GAPS FB-1): the default budget (2**27) splits the patch axis of
# every conv3 model here into blocks (at 8k clauses/class and kd=4 into blocks of ONE patch), which
# takes the update off the fused CUDA path (~10x slower). 2**30 keeps every conv3 model in one block
# (peak < 2 GB on the 3080). A memory/speed knob of the public constructor, not a learning setting.
S_GRID = {a: [b["s"] / 2, b["s"], b["s"] * 2] for a, b in BASE.items()}   # pilot (Phase 3)
# D6: early stopping as mnist_batch_clauses v4, on the validation split.
EARLY = dict(patience=25, min_delta=0.0, min_epochs=30, max_epochs=500, time_cap_h=24.0)
PILOT_MAX_EPOCHS = 60
N_VAL = 10000
SPLIT_SEED = 0


def T_of(arch, clauses):
    """D4: T scaled in proportion to clauses per class."""
    return BASE[arch]["T_ref"] * clauses // C_REF


def tag_of(arch, enc, clauses, s, kd, seed, augment="none"):
    k = f"_kd{kd}" if arch == "conv3" else ""
    a = "" if augment in (None, "none") else f"_aug-{augment}"
    return f"{arch}_{enc}_c{clauses}_spec{s:g}{k}_seed{seed}{a}"


def chosen(arch, enc):
    """(s, kd) picked by the pilot (results/pilot/choice.json); the base values before that."""
    f = RESULTS / "pilot" / "choice.json"
    if f.exists():
        c = json.loads(f.read_text()).get(f"{arch}_{enc}")
        if c:
            return float(c["s"]), c.get("kd")
    return None


# Augmentation (torchtsetlin_plans/plans/fmnist-bitplanes-augmentation.md; user 2026-11-14: "apply both" = shift and
# flip together, no policy pilot; replay without augmentation only for the best stage-A cell).
AUGMENTS = {"none": None, "shift2": dict(shift=2, flip_p=0.0), "shift2flip": dict(shift=2, flip_p=0.5)}
AUG_POLICY = "shift2flip"
AUG_CELLS = [("conv3", "bit8"), ("conv3", "gray8")]
REPLAY = [("conv3", "gray8", 8000)]
SAVE_ARMS = {"aug", "replay", "aug_check"}


def jobs(arm, seeds=(0,), cells=None):
    """(arch, enc, clauses, s, kd, seed) for each arm of the plan, restricted to ``cells``."""
    out = []
    cells = CELLS if cells is None else cells
    if arm == "smoke":                     # every model size the plan can build, base s
        for a, e in CELLS:
            for c in CLAUSES:
                for kd in (KDS_SMOKE if a == "conv3" else [None]):
                    out.append((a, e, c, BASE[a]["s"], kd, 0))
        for e in ("bit4", "therm4"):
            out.append(("conv2", e, 2000, BASE["conv2"]["s"], None, 0))
    elif arm == "pricing":
        for a, e in cells:
            for c in CLAUSES:
                for kd in (KDS_PRICING if a == "conv3" else [None]):
                    out.append((a, e, c, BASE[a]["s"], kd, 0))
    elif arm == "pilot":                   # C = 2000, seed 0, s grid (x kd for conv3)
        for a, e in cells:
            for s in S_GRID[a]:
                for kd in (KDS if a == "conv3" else [None]):
                    out.append((a, e, C_REF, s, kd, 0))
    elif arm in ("main", "seeds"):
        for seed in seeds:
            for a, e in cells:
                if arm == "seeds" and not ((a, e) in CORE or e in ("therm8", "bit8")):
                    continue
                ch = chosen(a, e)
                if ch is None:
                    raise RuntimeError("no pilot choice yet: results/pilot/choice.json")
                kd = KD_DEFAULT if a == "conv3" else None          # fixed kd = 8 (plan v4)
                for c in CLAUSES:
                    out.append((a, e, c, ch[0], kd, seed))
    elif arm == "aug":
        for a, e in AUG_CELLS:
            s, _ = chosen(a, e)
            for c in CLAUSES:
                out.append((a, e, c, s, KD_DEFAULT if a == "conv3" else None, 0))
    elif arm == "replay":
        for a, e, c in REPLAY:
            out.append((a, e, c, chosen(a, e)[0], KD_DEFAULT if a == "conv3" else None, 0))
    else:
        raise KeyError(arm)
    return out


# ---------------------------------------------------------------- data
def _idx(path):
    with gzip.open(path, "rb") as f:
        b = f.read()
    ndim = b[3]
    shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(b, np.uint8, offset=4 + 4 * ndim).reshape(shape)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw():
    """Raw Fashion-MNIST, uint8 (N, 28, 28), + labels; sha256 of every file."""
    files = {k: RAW / f"fashion-{n}" for k, n in (
        ("x_train", "train-images-idx3-ubyte.gz"), ("y_train", "train-labels-idx1-ubyte.gz"),
        ("x_test", "t10k-images-idx3-ubyte.gz"), ("y_test", "t10k-labels-idx1-ubyte.gz"))}
    d = {k: _idx(p) for k, p in files.items()}
    d["y_train"], d["y_test"] = d["y_train"].astype(np.int64), d["y_test"].astype(np.int64)
    d["sha256"] = {k: sha256(p) for k, p in files.items()}
    return d


def encode(enc, x, agt=None):
    """uint8 images (N, 28, 28) -> Boolean planes (N, P, 28, 28), P = ENCODINGS[enc]."""
    import torchtsetlin as tt
    xt = torch.from_numpy(np.ascontiguousarray(x))
    if enc == "agt":
        return (np.asarray(agt) != 0)[:, None]
    if enc.startswith("therm"):
        n = ENCODINGS[enc]
        e = tt.data.ThermometerEncoder(n, "uniform", value_range=(0, 255), flatten=False)
        return e(xt).permute(0, 3, 1, 2).contiguous().numpy()        # (N,28,28,n) -> (N,n,28,28)
    if enc == "bit8":
        return tt.data.BitPlaneEncoder(8)(xt[:, None]).numpy()
    if enc == "gray8":
        g = xt ^ (xt >> 1)
        return tt.data.BitPlaneEncoder(8)(g[:, None]).numpy()
    if enc == "bit4":
        return tt.data.BitPlaneEncoder(4)((xt >> 4)[:, None]).numpy()
    raise KeyError(enc)


def load_data(enc):
    """Encoded Fashion-MNIST split 50 000 train / 10 000 validation (stratified, fixed) + test."""
    raw = load_raw()
    agt = {}
    if enc == "agt":
        with np.load(B8) as f:
            agt = {k: f[k] for k in ("x_train", "x_test", "y_train", "y_test")}
        man = json.loads((PE / "results" / "data_manifest.json").read_text())["B8"]["sha256"]
        if sha256(B8) != man:
            raise RuntimeError("B8.npz sha256 does not match the paper_experiments manifest")
        for k in ("y_train", "y_test"):
            if not np.array_equal(agt[k], raw[k]):
                raise RuntimeError(f"B8 {k} differs from the raw Fashion-MNIST labels")
    tr_idx, va_idx = split(raw["y_train"])
    xtr = encode(enc, raw["x_train"], agt.get("x_train"))
    xte = encode(enc, raw["x_test"], agt.get("x_test"))
    return dict(raw_train=raw["x_train"][tr_idx], raw_val=raw["x_train"][va_idx], raw_test=raw["x_test"],
                x_train=xtr[tr_idx], y_train=raw["y_train"][tr_idx], x_val=xtr[va_idx],
                y_val=raw["y_train"][va_idx], x_test=xte, y_test=raw["y_test"],
                sha256=dict(raw["sha256"], **({"B8": sha256(B8)} if enc == "agt" else {})))


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
