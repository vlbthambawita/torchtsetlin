"""Cells, job lists, the pilot choice, the cost gate, paths, data and records of cifar10_graph_tm.

Questions (plan: ``torchtsetlin_plans/experiments/cifar10_graph_tm/PLAN.md``, issue #20):
does message passing between neighbouring image patches (graph TM depth 2, 3) raise CIFAR-10
accuracy over depth 1 (which is the convolutional coalesced TM), is a gain due to the spatial
edges, and does a best-effort rebuild of the graph TM paper's CIFAR-10 setting reach its 70.28 %?

Rules (../README.md): ``src/torchtsetlin/**`` is read-only; every reported number comes from a
``results/<phase>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.

The CIFAR-10 files, the 45 000 / 5 000 split and the record helpers are those of
``experiments/cifar10_batch_clauses`` (imported from its ``code/common.py``).
"""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/cifar10_graph_tm
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("CGT_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("CGT_CACHE", PROG / ".cache"))     # git-ignored: checkpoints, claims, W&B
MODELS = Path(os.environ.get("CGT_MODELS", PROG / "models"))   # git-ignored: kept models
LOGS = PROG / "logs"                                             # git-ignored
FIGURES = Path(os.environ.get("CGT_FIGURES", PROG / "figures"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cbc = _load("cbc_common", REPO / "experiments" / "cifar10_batch_clauses" / "code" / "common.py")
write_json, read_json = cbc.write_json, cbc.read_json

# ---------------------------------------------------------------- design (plan §3, §4)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SEED = 0
BATCH = 32                                             # arm D
BUDGET = 32                                            # soft literal budget, both arms
# Arm D: nodes = 4 x 4 windows at stride 2 of the 8-level colour thermometer (15 x 15 = 225 nodes,
# 384 patch bits + 28 position bits), four typed edges to the grid neighbours.
PATCH, STRIDE, GRID = 4, 2, 15
EDGE_TYPES = ["right", "left", "down", "up"]           # the direction a message travels
DEPTHS = [1, 2, 3]
DEPTH_CLAUSES = [20000, 80000]                         # shared clauses (total)
INBOX = 2048                                           # hashed inbox positions per message layer
MESSAGE_BITS = 2
EXTRA_AT = 20000                                       # the extra cells: depth 2 at this size
EXTRA_INBOX = [512, 8192]
FALLBACK_INBOX = 1024                                  # gate, first smaller option
FALLBACK_CLAUSES = 40000                               # gate, second smaller option (from 80 000)
# Pilot at depth 1, 20 000 clauses: T/C in {0.19, 0.5, 1.5} (0.19 = the paper's CIFAR-10 ratio) x s.
PILOT_CLAUSES = 20000
PILOT = [(3750, 5.0), (10000, 5.0), (30000, 5.0), (3750, 20.0), (10000, 20.0), (30000, 20.0)]
PILOT_MAX_EPOCHS = 60
PROVISIONAL = (0.5, 5.0)                               # (T / C, s) of the pricing runs before the pilot
TIE_PP = 0.5                                           # differences below this are ties (one seed)
EARLY = dict(patience=50, min_delta=0.0, min_epochs=30, max_epochs=300, time_cap_h=48.0)
GATE_S = float(os.environ.get("CGT_GATE_S", 1500))     # seconds per epoch (training + evaluation)
VIZ_EVERY = 20
N_TRAIN_EVAL = 5000                                    # clean training images of ``train_eval_acc``
HEALTH_IMAGES = 500                                    # held-out images of the clause-firing statistic
FINAL = ("done", "oom")                                # record statuses that end a job
PHASES = ["pricing", "paper", "pilot", "depth"]

# Arm P: the graph TM paper's CIFAR-10 setting (arXiv:2507.14874, Table 3 and 16), best effort.
PAPER = dict(clauses=80000, T=15000, s=20.0, patch=8, stride=1, grid=25, batch=16, epochs=30,
             last=5, time_cap_h=48.0)
PAPER_REPORTED = {"paper_graph": (70.28, 0.17), "paper_cotm": (66.42, 0.19)}
PAPER_CELLS = [dict(kind=k, clauses=PAPER["clauses"]) for k in ("paper_graph", "paper_cotm")]


# ---------------------------------------------------------------- cells and tags
def cell(depth, clauses, inbox=INBOX, edges="grid"):
    return dict(kind="grid", depth=depth, clauses=clauses, inbox=inbox if depth > 1 else 0, edges=edges)


MAIN = [cell(d, c) for c in DEPTH_CLAUSES for d in DEPTHS]
EXTRA = [cell(2, EXTRA_AT, m) for m in EXTRA_INBOX] + [cell(2, EXTRA_AT, edges="rewired")]


def tag_of(c, seed=SEED, pilot=None):
    if c["kind"] != "grid":
        return f"{c['kind']}{c.get('suffix', '')}_s{seed}"
    t = f"gtm_d{c['depth']}_c{c['clauses']}" + (f"_m{c['inbox']}" if c["depth"] > 1 else "") + f"_{c['edges']}"
    if pilot is not None:
        t += f"_T{pilot[0]}_spec{pilot[1]:g}"
    return f"{t}_s{seed}"


def fallbacks(c):
    """The smaller options the gate prices for a main cell with message passing (plan §5)."""
    if c not in MAIN or c["depth"] == 1:
        return []
    out = [cell(c["depth"], c["clauses"], FALLBACK_INBOX)]
    if c["clauses"] > FALLBACK_CLAUSES:
        out.append(cell(c["depth"], FALLBACK_CLAUSES))
    return out


# ---------------------------------------------------------------- pilot choice (plan §3)
def choose(val):
    """``val``: {(T, s): best validation accuracy}. The best wins; exact ties go to the first of PILOT."""
    T, s = max(PILOT, key=lambda k: val[k])
    return {"T": T, "s": s, "ratio": T / PILOT_CLAUSES,
            "best_val_acc": {f"T{t}/s{sp:g}": val[(t, sp)] for t, sp in PILOT},
            "rule": f"best validation accuracy at depth 1, {PILOT_CLAUSES} clauses, at most "
                    f"{PILOT_MAX_EPOCHS} epochs; T is then scaled with the clauses (T / C fixed)"}


def ts_choice():
    """(T / C, s) of the depth study: ``CGT_TS=<ratio>,<s>`` overrides, else
    results/pilot/choice.json, else None (the depth study waits)."""
    v = os.environ.get("CGT_TS", "").strip()
    if v:
        r, s = v.split(",")
        return float(r), float(s)
    c = read_json(RESULTS / "pilot" / "choice.json")
    return (float(c["ratio"]), float(c["s"])) if c else None


# ---------------------------------------------------------------- cost gate (plan §5)
def price(c):
    """The pricing record of a cell: (final?, passes the gate?, record)."""
    r = read_json(RESULTS / "pricing" / f"{tag_of(c)}.json")
    if r is None or r.get("status") not in FINAL:
        return False, False, r
    ok = r["status"] == "done" and r["train_s_per_epoch"] + r["eval_s_per_epoch"] <= GATE_S
    return True, ok, r


def resolve(c):
    """What the gate makes of a planned cell: {"state": "run" | "pending" | "dropped", "cell": the
    cell to run (or to price next), "tried": cells that failed the gate}. ``CGT_GATE=off`` runs
    every cell as planned; ``CGT_PAPER_FORCE=1`` does so for the paper arm only (it has no
    smaller option: a paper run above the gate is reported and left to the user)."""
    if os.environ.get("CGT_GATE", "on") == "off" or (
            c["kind"] != "grid" and os.environ.get("CGT_PAPER_FORCE", "0") == "1"):
        return {"state": "run", "cell": c, "tried": []}
    tried = []
    for cand in [c] + fallbacks(c):
        final, ok, _ = price(cand)
        if not final:
            return {"state": "pending", "cell": cand, "tried": tried}
        if ok:
            return {"state": "run", "cell": cand, "tried": tried}
        tried.append(cand)
    return {"state": "dropped", "cell": None, "tried": tried}


def epoch_s(c):
    """Priced seconds per epoch of a cell (None before pricing)."""
    final, _, r = price(c)
    return r["train_s_per_epoch"] + r["eval_s_per_epoch"] if final and r["status"] == "done" else None


# ---------------------------------------------------------------- jobs
def _job(phase, c, T, s, **kw):
    j = dict(c, phase=phase, T=T, s=s, seed=SEED, tag=kw.pop("tag", None) or tag_of(c), wait=None, **kw)
    return j


def jobs(phase):
    """Job dicts of one phase. ``wait`` (a reason) marks a job that cannot start yet."""
    if phase == "smoke":
        g = dict(kind="grid", depth=2, clauses=500, inbox=64, edges="grid")
        return [_job(phase, dict(g, depth=1, inbox=0), 250, 5.0), _job(phase, g, 250, 5.0),
                _job(phase, dict(g, edges="rewired", depth=3), 250, 5.0),
                _job(phase, dict(kind="paper_graph", clauses=500), 100, 20.0),
                _job(phase, dict(kind="paper_cotm", clauses=500), 100, 20.0)]
    if phase == "paper":
        out = []
        for c in PAPER_CELLS:
            r = resolve(c)
            if r["state"] != "dropped":                # above the gate: reported, left to the user
                j = _job(phase, c, PAPER["T"], PAPER["s"])
                j["wait"] = "waits for its pricing" if r["state"] == "pending" else None
                out.append(j)
        return out
    if phase == "pilot":
        c = cell(1, PILOT_CLAUSES)
        return [_job(phase, c, T, s, tag=tag_of(c, pilot=(T, s))) for T, s in PILOT]
    ratio, s = ts_choice() or PROVISIONAL
    if phase == "pricing":                             # timing depends little on T and s
        cells = []
        for c in MAIN + EXTRA:
            for cand in [c] + resolve(c)["tried"] + ([resolve(c)["cell"]] if resolve(c)["cell"] else []):
                if cand not in cells:
                    cells.append(cand)
        return [_job(phase, c, PAPER["T"], PAPER["s"]) for c in PAPER_CELLS] + [
            _job(phase, c, int(round(ratio * c["clauses"])), s) for c in cells]
    if phase == "depth":
        chosen = ts_choice()
        out = []
        for c in MAIN + EXTRA:
            r = resolve(c)
            if r["state"] == "dropped":                # reported by report.py, never run
                continue
            run = r["cell"]
            wait = ("waits for its pricing" if r["state"] == "pending"
                    else "waits for the pilot's choice of T and s" if chosen is None else None)
            j = _job(phase, run, None if chosen is None else int(round(ratio * run["clauses"])),
                     None if chosen is None else s, planned=tag_of(c),
                     role="main" if c in MAIN else "extra")
            j["wait"] = wait
            out.append(j)
        return out
    raise KeyError(phase)


def est_cost(job):
    """Seconds per epoch for longest-first ordering: the pricing record, else a rough guess."""
    if job["kind"] == "grid":
        return epoch_s({k: job[k] for k in ("kind", "depth", "clauses", "inbox", "edges")}) or (
            job["clauses"] / 1000 * (1 + 5 * (job["depth"] - 1)))
    return 1e9                                         # the paper runs go first


# ---------------------------------------------------------------- data
def load_cifar(subset=0):
    """uint8 RGB images (N, 32, 32, 3) and labels: train 45 000 / validation 5 000 (the
    cifar10_batch_clauses split, hash-checked) / test 10 000, plus ``train_all`` (the official
    50 000, used by the paper arm). ``subset``: smoke sizes."""
    raw = cbc.load_raw()
    tr, va = cbc.split(raw["y_train"])
    d = {"train": (raw["x_train"][tr], raw["y_train"][tr]), "val": (raw["x_train"][va], raw["y_train"][va]),
         "test": (raw["x_test"], raw["y_test"]), "train_all": (raw["x_train"], raw["y_train"])}
    out = {}
    for k, (x, y) in d.items():
        n = len(x) if not subset else (subset if k.startswith("train") else max(subset // 4, 200))
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
    """The fixed clean training images of ``train_eval_acc`` (the same in every run, and the same
    images as in cifar10_conv_reg)."""
    n = min(N_TRAIN_EVAL, n_train)
    idx = np.sort(np.random.default_rng(12345).permutation(n_train)[:n])
    return _fixed_indices("train_eval", idx, cbc.N_VAL * 9, n_train)


def panel_indices(y, name, n_full):
    """The fixed held-out images of the evidence panel: the first image of each class."""
    idx = np.array(sorted(int(np.flatnonzero(y == k)[0]) for k in range(10) if (y == k).any()))
    return _fixed_indices(f"panel_{name}", idx, n_full, len(y))


# ---------------------------------------------------------------- records
def new_record(kind, device=None, **fields):
    rec = cbc.new_record(kind, device, **fields)
    rec["env"]["gpu_slot"] = os.environ.get("CGT_GPU")
    return rec


def status_of(phase, tag):
    r = read_json(RESULTS / phase / f"{tag}.json")
    return None if r is None else r.get("status")
