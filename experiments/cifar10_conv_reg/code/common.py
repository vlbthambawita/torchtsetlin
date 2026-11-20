"""Factors, job lists, paths, data and records of the cifar10_conv_reg programme.

Question: do drop clause (``drop_clause_p``) and training-time augmentation reduce the
overfitting of the single 3x3 colour-thermometer ConvTM on CIFAR-10, and does 64 000 clauses per
class continue the trend? Plan: ``torchtsetlin_plans/experiments/cifar10_conv_reg/PLAN.md``.
Rules (../README.md): ``src/torchtsetlin/**`` is read-only; every reported number comes from a
``results/<phase>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.

The CIFAR-10 files, the 45 000 / 5 000 split, the model settings and the record helpers are those
of ``experiments/cifar10_batch_clauses`` (imported from its ``code/common.py``).
"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/cifar10_conv_reg
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("CCR_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("CCR_CACHE", PROG / ".cache"))     # git-ignored: checkpoints, claims, W&B
MODELS = Path(os.environ.get("CCR_MODELS", PROG / "models"))   # git-ignored: best-validation models
LOGS = PROG / "logs"                                             # git-ignored
FIGURES = Path(os.environ.get("CCR_FIGURES", PROG / "figures"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cbc = _load("cbc_common", REPO / "experiments" / "cifar10_batch_clauses" / "code" / "common.py")
write_json, read_json = cbc.write_json, cbc.read_json

# ---------------------------------------------------------------- design (plan §3, §4, §6)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MODEL = dict(cbc.BASE["conv"])                         # therm8, 3x3, T_ref 3000 at 2000, s 5, weighted, budget 32
BATCH = 32
SEED = 0
SCREEN_CLAUSES = 8000
FACTORIAL_CLAUSES = [16000, 32000, 64000]
DROP_PS = [0.0, 0.1, 0.25, 0.5]
AUGS = ["none", "geo+col"]                             # geo+col: the policy of cifar10_joint_teams
ARMS = {"base": ("none", False), "aug": ("geo+col", False), "drop": ("none", True),
        "aug_drop": ("geo+col", True)}                 # arm -> (augmentation, uses p*)
TIE_PP = 0.5                                           # differences below this are ties (one seed)
FALLBACK_P = 0.1                                       # p* if drop clause does not help in the screen
EARLY = dict(patience=50, min_delta=0.0, min_epochs=30, max_epochs=500, time_cap_h=72.0)
VIZ_EVERY = 20
N_TRAIN_EVAL = 5000                                    # clean training images of ``train_eval_acc``
PANEL_PER_CLASS = 1
HEALTH_IMAGES = 1000                                   # validation images of the clause-firing statistic
FINAL = ("done", "oom")                                # record statuses that end a job
PHASES = ["pricing", "screen", "factorial"]

# Reference (not produced here): Optimized Toolbox, arXiv:2406.00704, Table IV, 3x3 colour
# thermometers, tmu, 250 epochs, mean of the last 25 epochs.
TOOLBOX = {"none": {8000: 68.4, 16000: 70.4, 32000: 72.3, 64000: 74.3},
           "augmented": {8000: 67.5, 16000: 69.1, 32000: 70.5, 64000: 72.7}}


def T_of(clauses):
    """T in proportion to the clauses per class (3 000 at 2 000 = 1.5 x C)."""
    return cbc.T_of("conv", clauses, "scaled")


def aug_tag(aug):
    return aug.replace("+", "")


def tag_of(clauses, aug, p, seed=SEED, batch=BATCH):
    return f"conv_c{clauses}_b{batch}_{aug_tag(aug)}_dc{p:g}_s{seed}"


# ---------------------------------------------------------------- screen choice (plan §5)
def choose(val):
    """``val``: {(aug, p): best validation accuracy} of the screen. Returns the choice dict.

    p* = the smallest p > 0 whose validation accuracy with augmentation is within TIE_PP of the
    best p > 0 there. If no p > 0 comes within TIE_PP of p = 0 in either row, drop clause is
    reported as not helping and p* = FALLBACK_P (the question at large C stays open)."""
    pos = [p for p in DROP_PS if p > 0]
    top = max(val[("geo+col", p)] for p in pos)
    p_star = min(p for p in pos if val[("geo+col", p)] >= top - TIE_PP)
    helps = any(val[(a, p)] >= val[(a, 0.0)] - TIE_PP for a in AUGS for p in pos)
    return {"p_star": p_star if helps else FALLBACK_P, "helps": helps,
            "best_val_acc": {f"{aug_tag(a)}/dc{p:g}": v for (a, p), v in sorted(val.items())},
            "rule": f"smallest p > 0 within {TIE_PP} pp of the best p > 0 with augmentation (validation, "
                    f"C = {SCREEN_CLAUSES}); p* = {FALLBACK_P} if no p > 0 is within {TIE_PP} pp of p = 0"}


def drop_choice():
    """p* of the factorial: CCR_DROP_P overrides, else results/screen/choice.json, else None."""
    v = os.environ.get("CCR_DROP_P", "").strip()
    if v:
        return float(v)
    c = read_json(RESULTS / "screen" / "choice.json")
    return float(c["p_star"]) if c else None


# ---------------------------------------------------------------- jobs
def jobs(phase):
    """Job dicts of one phase: phase, clauses, aug, p, seed, tag (factorial: + arm)."""
    def j(c, aug, p, **kw):
        return dict(phase=phase, clauses=c, aug=aug, p=float(p), seed=SEED, tag=tag_of(c, aug, p), **kw)
    if phase == "smoke":
        return [j(1000, "none", 0.0), j(1000, "geo+col", 0.25)]
    if phase == "pricing":                             # GPU time does not depend on the augmentation
        return [j(c, "none", p) for c in [SCREEN_CLAUSES] + FACTORIAL_CLAUSES for p in (0.0, 0.25)]
    if phase == "screen":
        return [j(SCREEN_CLAUSES, a, p) for a in AUGS for p in DROP_PS]
    if phase == "factorial":
        p_star = drop_choice()
        out = []
        for c in FACTORIAL_CLAUSES:
            for arm, (aug, drop) in ARMS.items():
                if drop and p_star is None:            # listed (and blocked) until the screen has chosen
                    out.append(dict(phase=phase, clauses=c, aug=aug, p=None, seed=SEED, arm=arm,
                                    tag=f"conv_c{c}_b{BATCH}_{aug_tag(aug)}_dc?_s{SEED}"))
                else:
                    out.append(j(c, aug, p_star if drop else 0.0, arm=arm))
        return out
    raise KeyError(phase)


def est_cost(job):
    """Seconds per epoch for longest-first ordering: the pricing record of this size, else the
    V100 timings of cifar10_batch_clauses (39 s at 8 000, linear in the clauses)."""
    for p in (job.get("p") or 0.0, 0.0):
        r = read_json(RESULTS / "pricing" / f"{tag_of(job['clauses'], 'none', 0.25 if p else 0.0)}.json")
        if r and r.get("train_s_per_epoch"):
            return r["train_s_per_epoch"] + r["eval_s_per_epoch"]
    return 39.0 * job["clauses"] / 8000


# ---------------------------------------------------------------- data
def load_cifar(subset=0):
    """uint8 RGB images (N, 32, 32, 3) and labels: train 45 000 / validation 5 000 (the
    cifar10_batch_clauses split, hash-checked) / test 10 000. ``subset``: smoke sizes."""
    raw = cbc.load_raw()
    tr, va = cbc.split(raw["y_train"])
    d = {"train": (raw["x_train"][tr], raw["y_train"][tr]), "val": (raw["x_train"][va], raw["y_train"][va]),
         "test": (raw["x_test"], raw["y_test"])}
    out = {}
    for k, (x, y) in d.items():
        n = len(x) if not subset else (subset if k == "train" else max(subset // 4, 200))
        out[k] = (np.ascontiguousarray(x[:n]), np.ascontiguousarray(y[:n]).astype(np.int64))
    return out


def _fixed_indices(name, idx, n_full, n):
    """Store a fixed index set in results/<name>.json on the first full-size call; check it later."""
    f = RESULTS / f"{name}.json"
    h = hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()
    old = read_json(f)
    if n == n_full:
        if old is None:
            write_json({"sha256": h, "indices": [int(i) for i in idx]}, f)
        elif old["sha256"] != h:
            raise RuntimeError(f"{name} indices differ from {f}")
    return idx


def train_eval_indices(n_train):
    """The fixed clean training images of ``train_eval_acc`` (the same in every run)."""
    n = min(N_TRAIN_EVAL, n_train)
    idx = np.sort(np.random.default_rng(12345).permutation(n_train)[:n])
    return _fixed_indices("train_eval", idx, cbc.N_VAL * 9, n_train)


def panel_indices(y_val):
    """The fixed validation images of the evidence panel: the first image of each class."""
    idx = np.array(sorted(int(i) for k in range(10) for i in np.flatnonzero(y_val == k)[:PANEL_PER_CLASS]))
    return _fixed_indices("panel", idx, cbc.N_VAL, len(y_val))


# ---------------------------------------------------------------- records
def new_record(kind, device=None, **fields):
    rec = cbc.new_record(kind, device, **fields)
    rec["env"]["gpu_slot"] = os.environ.get("CCR_GPU")
    try:
        from importlib.metadata import PackageNotFoundError, version
        for pkg in ("albumentationsx", "albumentations"):
            try:
                rec["env"]["augmentation_lib"] = f"{pkg} {version(pkg)}"
                break
            except PackageNotFoundError:
                continue
    except Exception:  # pragma: no cover
        pass
    return rec


def status_of(phase, tag):
    r = read_json(RESULTS / phase / f"{tag}.json")
    return None if r is None else r.get("status")
