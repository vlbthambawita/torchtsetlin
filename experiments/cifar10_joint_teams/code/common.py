"""Members, teams, grid, paths, data and records of the cifar10_joint_teams programme.

Question: how does the batch size change accuracy and training time of a *jointly trained* team
of convolutional Tsetlin machines on CIFAR-10, with training-time augmentation in every run?
Plan: ``torchtsetlin_plans/experiments/cifar10_joint_teams/PLAN.md``. Rules (../README.md):
``src/torchtsetlin/**`` is read-only; every reported number comes from a
``results/<phase>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.

CIFAR-10 files, the 45 000 / 5 000 split and the record helpers are those of
``experiments/cifar10_batch_clauses`` (imported from its ``code/common.py``); the joint update is
``experiments/joint_wide_tm/code/joint.py``.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/cifar10_joint_teams
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("CJT_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("CJT_CACHE", PROG / ".cache"))     # git-ignored: checkpoints, claims, W&B
MODELS = Path(os.environ.get("CJT_MODELS", PROG / "models"))   # git-ignored: best-validation teams
LOGS = PROG / "logs"                                             # git-ignored
FIGURES = PROG / "figures"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cbc = _load("cbc_common", REPO / "experiments" / "cifar10_batch_clauses" / "code" / "common.py")
write_json, read_json = cbc.write_json, cbc.read_json

# ---------------------------------------------------------------- design (plan §3, §4, §6, §7)
N_CLASSES = 10
CLAUSES = 2000                                         # per class and member
SEED = 0
# Every member: weighted clauses, soft literal budget 32, position bits (Toolbox scripts).
MODEL_KW = dict(weighted=True, max_included_literals=32, boost_true_positive=True,
                position_encoding=True, feedback_mode="batch")
# T and s: Optimized Toolbox (arXiv:2406.00704) Table III, as toolbox_cifar10/code/specialists.py;
# B3 = cifar10_batch_clauses' conv3 (8 bit planes per colour, window over all planes).
MEMBERS = {
    "H3": dict(enc="therm8", conv="2d", patch=3, T=3000, s=5.0),
    "H4": dict(enc="therm8", conv="2d", patch=4, T=3000, s=5.0),
    "H5": dict(enc="therm8", conv="2d", patch=5, T=3000, s=5.0),
    "G5": dict(enc="agauss", conv="2d", patch=5, T=2500, s=3.5),
    "M10": dict(enc="amean", conv="2d", patch=10, T=500, s=10.0),
    "B3": dict(enc="bit8", conv="3d", patch=3, kd=8, T=3000, s=5.0),
}
ENC_SHAPE = {"therm8": (24, 32, 32), "agauss": (3, 32, 32), "amean": (3, 32, 32), "bit8": (3, 8, 32, 32)}
CPU_ENCS = ("agauss", "amean")                        # OpenCV, computed with the augmentation
TEAMS = {"M": ["H3", "M10", "B3"],                    # mixed: binarization, filter size, 2D + 3D
         "S": ["H3", "H4", "H5"],                     # filter shape only
         "L": ["H3", "H5", "G5", "M10", "B3"]}        # everything, five members (stage 2)
RULE = "sum"                                           # joint_wide_tm pilot: sum >= one on both teams
BATCHES = [16, 32, 64, 128, 256, 512, 1024]            # core grid, teams M and S
EXT_BATCHES = [1, 4]                                   # team M only, after pricing
POLICIES = ["geo", "geo+col", "full"]                  # weakest -> strongest (augment.py)
PILOT = dict(team="M", batch=32, epochs=40, tie_pp=0.5)
EARLY = dict(patience=30, min_epochs=60, max_epochs=200, time_cap_h=48.0)
VIZ_EVERY = 10
PANEL_PER_CLASS = 2                                    # fixed validation images for the maps
# Clause sweep (2026-11-18, after the batch grid): best team at one batch size, larger members.
CLAUSE_SWEEP = dict(team="M", batch=64, clauses=[16000, 32000, 64000, 128000], scaled=[32000],
                    time_cap_h=120.0, price_updates=30)
# Longer training and drop clause (2026-11-23, plan PLAN_LONG_DROPOUT.md): the best run of the
# clause sweep (team M, B = 64, geo+col, T scaled with the clauses) with early stopping at
# patience 75 and no epoch limit. Stage 1: 32 000 clauses at four drop-clause probabilities
# (in this order). Stage 2: 64 000 clauses at p = 0 and at the p chosen from stage 1.
LONG = dict(team="M", batch=64, policy="geo+col", t_mode="scaled", clauses=32000, drops=[0.0, 0.25, 0.1, 0.5],
            big_clauses=64000, fallback_drop=0.25, tie_pp=0.5, patience=75, min_epochs=60, time_cap_h=240.0,
            price_updates=30, train_eval=5000)
PHASES = ["smoke", "check", "pricing", "pilot", "grid", "ext", "large", "cprice", "clauses", "lprice", "long"]
FINAL = ("done",)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def policy_tag(policy):
    return policy.replace("+", "")


def tag_of(team, batch, policy, seed=SEED, suffix=""):
    return f"{team}_b{batch}_{policy_tag(policy)}_s{seed}{suffix}"


def policy_choice():
    """The augmentation policy chosen by the pilot (results/pilot/choice.json), or None."""
    c = read_json(RESULTS / "pilot" / "choice.json")
    return c["policy"] if c else None


def large_batches():
    """Team L batch sizes: CJT_LARGE_BATCHES=32,128,1024 (chosen from the core result)."""
    v = os.environ.get("CJT_LARGE_BATCHES", "")
    return [int(b) for b in v.split(",") if b.strip()]


def sweep_clauses():
    """Clauses per class and member of the clause sweep: CJT_CLAUSES=16000,32000 overrides."""
    v = os.environ.get("CJT_CLAUSES", "")
    return [int(c) for c in v.split(",") if c.strip()] or CLAUSE_SWEEP["clauses"]


def sweep_scaled():
    """Sizes that also run with T scaled in proportion to the clauses (control of the fixed T):
    CJT_SCALED_CLAUSES=32000,64000 overrides, CJT_SCALED_CLAUSES=none switches it off."""
    v = os.environ.get("CJT_SCALED_CLAUSES", "")
    if v.strip().lower() == "none" or t_mode() == "scaled":
        return []
    return [int(c) for c in v.split(",") if c.strip()] or CLAUSE_SWEEP["scaled"]


def sweep_batch():
    return int(os.environ.get("CJT_CLAUSE_BATCH") or CLAUSE_SWEEP["batch"])


def t_mode():
    """``fixed``: every member keeps its Toolbox T at any size (the paper's setting up to 64 000
    clauses). ``scaled`` (CJT_T_MODE=scaled): T grows in proportion to the clauses."""
    m = os.environ.get("CJT_T_MODE") or "fixed"
    if m not in ("fixed", "scaled"):
        raise SystemExit(f"CJT_T_MODE must be fixed or scaled, not {m}")
    return m


def clause_suffix(clauses, mode):
    return f"_c{clauses}" + ("_Tscaled" if mode == "scaled" else "")


def drop_suffix(p):
    return f"_dc{p:g}"


def long_patience():
    """Patience of the ``long`` phase: CJT_LONG_PATIENCE=150 overrides (a larger value also lets
    finished runs continue from their last checkpoint, see queue_runner)."""
    return int(os.environ.get("CJT_LONG_PATIENCE") or LONG["patience"])


def long_big():
    """Stage-2 size (clauses per class and member): CJT_LONG_BIG=none switches stage 2 off."""
    v = (os.environ.get("CJT_LONG_BIG") or "").strip().lower()
    return None if v == "none" else int(v or LONG["big_clauses"])


def long_choice():
    """The drop-clause probability of the second stage-2 run: CJT_LONG_PSTAR=0.25 fixes it at
    once; otherwise results/long/choice.json, written when the four stage-1 runs are done."""
    v = os.environ.get("CJT_LONG_PSTAR")
    if v:
        return float(v)
    c = read_json(RESULTS / "long" / "choice.json")
    return None if c is None else float(c["drop_clause_p"])


def long_jobs(phase):
    """``long``: stage 1 in the order of LONG["drops"], then stage 2. ``lprice``: the same models,
    30 timed updates each (stage 2 at p = 0 and the fallback p)."""
    def j(clauses, p):
        return dict(phase=phase, team=LONG["team"], batch=LONG["batch"], policy=LONG["policy"], seed=SEED,
                    clauses=clauses, t_mode=LONG["t_mode"], drop=float(p),
                    tag=tag_of(LONG["team"], LONG["batch"], LONG["policy"],
                               suffix=clause_suffix(clauses, LONG["t_mode"]) + drop_suffix(p)))
    out = [j(LONG["clauses"], p) for p in LONG["drops"]]
    big = long_big()
    if big:
        out.append(j(big, 0.0))
        p = LONG["fallback_drop"] if phase == "lprice" else long_choice()
        if p:
            out.append(j(big, p))
    return out


def member_T(name, clauses, mode="fixed"):
    T = MEMBERS[name]["T"]
    return int(round(T * clauses / CLAUSES)) if mode == "scaled" else T


def jobs(phase):
    """Job dicts of one phase: team, batch, policy, seed, tag (clause sweep: + clauses, t_mode)."""
    def j(team, batch, policy, clauses=None, mode=None):
        d = dict(phase=phase, team=team, batch=batch, policy=policy, seed=SEED,
                 tag=tag_of(team, batch, policy))
        if clauses:
            mode = mode or t_mode()
            d.update(clauses=clauses, t_mode=mode,
                     tag=tag_of(team, batch, policy, suffix=clause_suffix(clauses, mode)))
        return d
    if phase in ("lprice", "long"):                    # policy fixed in LONG, no pilot needed
        return long_jobs(phase)
    pol = policy_choice()
    if phase == "smoke":
        return [j(t, b, "full") for t in TEAMS for b in (16, 1024)]
    if phase == "pricing":
        return [j(t, b, "full") for t in TEAMS for b in (1, 32, 1024)]
    if phase == "pilot":
        return [j(PILOT["team"], PILOT["batch"], p) for p in POLICIES]
    if pol is None:
        return []
    if phase == "grid":
        return [j(t, b, pol) for t in ("M", "S") for b in BATCHES]
    if phase == "ext":
        return [j("M", b, pol) for b in EXT_BATCHES]
    if phase == "large":
        return [j("L", b, pol) for b in large_batches()]
    if phase in ("cprice", "clauses"):
        return ([j(CLAUSE_SWEEP["team"], sweep_batch(), pol, c) for c in sweep_clauses()]
                + [j(CLAUSE_SWEEP["team"], sweep_batch(), pol, c, "scaled") for c in sweep_scaled()])
    raise KeyError(phase)


def est_cost(job):
    """Seconds per epoch from the nearest pricing record (for longest-first ordering)."""
    if job.get("clauses"):                             # clause sweep: its own pricing, else linear
        r = (read_json(RESULTS / "lprice" / f"{job['tag']}.json")
             or read_json(RESULTS / "cprice" / f"{job['tag']}.json")
             or read_json(RESULTS / "cprice" / f"{job['tag'].replace('_Tscaled', '')}.json"))
        if r and r.get("train_s_per_epoch"):
            return r["train_s_per_epoch"] + r["eval_s_per_epoch"]
        return 60.0 * job["clauses"] / CLAUSES
    best = None
    for p in jobs("pricing"):
        if p["team"] != job["team"]:
            continue
        r = read_json(RESULTS / "pricing" / f"{p['tag']}.json")
        if r and r.get("train_s_per_epoch"):
            d = abs(np.log2(p["batch"] / job["batch"]))
            if best is None or d < best[0]:
                best = (d, r["train_s_per_epoch"] + r["eval_s_per_epoch"])
    return best[1] if best else len(TEAMS[job["team"]]) * (30.0 / job["batch"] + 1.0)


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


def panel_indices(y_val):
    """The fixed validation images of the activation maps (PANEL_PER_CLASS per class, the first
    of each class); stored in results/panel.json and checked on every later call."""
    idx = sorted(int(i) for k in range(N_CLASSES) for i in np.flatnonzero(y_val == k)[:PANEL_PER_CLASS])
    f = RESULTS / "panel.json"
    old = read_json(f)
    if old is None and len(y_val) == cbc.N_VAL:
        write_json({"split": "val", "indices": idx, "labels": [int(y_val[i]) for i in idx]}, f)
    elif old is not None and len(y_val) == cbc.N_VAL and old["indices"] != idx:
        raise RuntimeError("panel images differ from results/panel.json")
    return idx


# ---------------------------------------------------------------- records
def new_record(kind, device=None, **fields):
    rec = cbc.new_record(kind, device, **fields)
    rec["env"]["gpu_slot"] = os.environ.get("CJT_GPU")
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
