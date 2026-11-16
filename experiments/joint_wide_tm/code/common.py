"""Members, teams, arms, paths and data for the joint_wide_tm programme (Fashion-MNIST).

Plan: ``torchtsetlin_plans/experiments/joint_wide_tm/PLAN_FMNIST.md``. Rules (../README.md):
``src/torchtsetlin/**`` is read-only; every reported number comes from a
``results/<phase>/<arm>/<tag>.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.

Data, encodings and the validation split are those of ``experiments/fmnist_bitplanes`` (imported
from its ``code/common.py``), so numbers are comparable with that programme.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/joint_wide_tm
REPO = PROG.parents[1]
RESULTS = Path(os.environ.get("JW_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("JW_CACHE", PROG / ".cache"))      # git-ignored: sums, checkpoints, W&B
MODELS = Path(os.environ.get("JW_MODELS", PROG / "models"))    # git-ignored: best-validation models
LOGS = PROG / "logs"                                             # git-ignored
FIGURES = PROG / "figures"

_spec = importlib.util.spec_from_file_location(
    "fmb_common", REPO / "experiments" / "fmnist_bitplanes" / "code" / "common.py")
fmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fmb)
new_record, write_json, read_json = fmb.new_record, fmb.write_json, fmb.read_json

# ---------------------------------------------------------------- design (plan part B)
N_CLASSES = 10
CLAUSES = 2000                                         # per class and member (B3)
BATCH = 32                                             # B2
SEEDS = [0, 1, 2]
EPOCHS = {"smoke": 2, "pilot": 40, "main": 100, "confirm": 100}
# Every member: ConvTsetlinMachine, weighted, soft literal budget 32, position bits (B3).
MODEL_KW = dict(weighted=True, max_included_literals=32, boost_true_positive=True,
                position_encoding=True, feedback_mode="batch")
MEMBERS = {   # T and s: Granmo 2023 Table 1 scaled from 8 000 to 2 000 clauses/class (A10, H3, H4)
    "A10": dict(enc="agt", patch=10, T=500, s=10.0),
    "H3": dict(enc="therm8", patch=3, T=1500, s=2.5),
    "H4": dict(enc="therm8", patch=4, T=1500, s=2.5),
    "H5": dict(enc="therm8", patch=5, T=1500, s=2.5),      # as H3 (not in the paper)
    "H10": dict(enc="therm8", patch=10, T=1500, s=2.5),    # as H3 (not in the paper)
}
TEAMS = {"S": ["H3", "H5", "H10"],                      # filter sizes only (user: 3 / 5 / 10)
         "P": ["A10", "H3", "H4"]}                      # the paper's team
RULES = ["sum", "one"]                                  # T_joint = sum of T  /  own T (A2, R1)
ARMS = ["I", "J", "Jown", "W"]
PHASES = ["smoke", "pricing", "pilot", "main", "confirm"]
W_FACTOR = 3                                            # W: one member with the team's clauses


def member_tag(name, seed, clauses=CLAUSES):
    return f"{name}_seed{seed}" if clauses == CLAUSES else f"{name}_c{clauses}_seed{seed}"


def team_tag(team, rule, seed, clauses=CLAUSES):
    c = "" if clauses == CLAUSES else f"_c{clauses}"
    return f"{team}_{rule}{c}_seed{seed}"


def wide_tag(team, rule, member, seed):
    return f"{team}_{rule}_{member}_seed{seed}"


def pilot_choice():
    """results/pilot/choice.json (report.py --choose): per team the joint rule(s) and W member."""
    f = RESULTS / "pilot" / "choice.json"
    return read_json(f) if f.exists() else None


# ---------------------------------------------------------------- data
def load_all(encs, device, subset=0):
    """Encoded Fashion-MNIST for every encoding in ``encs`` on ``device``.

    Returns ``x[enc][split]`` (bool ``(N, planes, 28, 28)``), ``y[split]`` (numpy int64), the raw
    uint8 images per split and the sha256 of the source files. Split = fmnist_bitplanes'
    (50 000 train / 10 000 validation, stratified, seed 0; hash-checked on load)."""
    x, y, raw, sha = {}, {}, {}, {}
    for enc in sorted(set(encs)):
        d = fmb.load_data(enc)
        sha.update(d["sha256"])
        x[enc] = {}
        for k in ("train", "val", "test"):
            a = d[f"x_{k}"]
            n = len(a) if not subset else (subset if k == "train" else max(subset // 4, 100))
            x[enc][k] = torch.from_numpy(np.ascontiguousarray(a[:n])).to(torch.bool).to(device)
            y[k] = d[f"y_{k}"][:n]
            raw[k] = d[f"raw_{k}"][:n]
    return x, y, raw, sha
