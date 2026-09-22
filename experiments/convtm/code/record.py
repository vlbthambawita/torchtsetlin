"""The result record: schema, construction, validation, and the one place selection happens.

PLAN.md Section 6.2 defines the schema; this module is its executable form. ``validate()``
is not a formality -- ``write()`` refuses to write a record that fails it, so an inadmissible
result never reaches ``results/`` and therefore never reaches the report (constraint C4).

The leakage guard is *structural*, not procedural:

* ``select_epoch()`` is the only selection function in the harness, and it reads ``val_acc``.
* ``validate()`` recomputes ``argmax(val_acc)`` from the curve and rejects the record if
  ``selected_epoch`` disagrees -- so a record selected any other way cannot be written.
* ``validate()`` rejects a ``test_acc`` key **inside a curve entry**. Per-epoch test accuracy
  may be interesting for a figure, but a per-epoch test column in the same table the
  selector reads is exactly how ``experiments/mctm`` ended up reporting ``best_test_acc``.
  If a phase needs test curves, they go in ``diagnostics['test_curve']`` where nothing in the
  selection path can see them, and the reviewer can see that they were added deliberately.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")
PREDS = os.path.join(RESULTS, "preds")

FAMILIES = ("baseline", "existing", "candidate", "control")
REQUIRED = ("arm", "family", "paper", "seed", "argv", "git_sha", "env", "started", "wall_s",
            "data", "hp", "capacity", "curve", "selected_epoch", "val_acc", "test_acc",
            "test_acc_per_class", "test_predictions_sha", "diagnostics", "notes")
DATA_REQUIRED = ("dataset", "booleanization", "n_train", "n_val", "n_test", "augment",
                 "split_hash")
CAPACITY_REQUIRED = ("n_automata", "n_clauses_total", "literals_evaluated_per_image",
                     "state_bytes")
EPOCH_REQUIRED = ("epoch", "train_acc", "val_acc", "epoch_s")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_info() -> Dict[str, object]:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(torch.cuda.current_device())
    return {"torch": torch.__version__, "gpu": gpu, "host": platform.node(),
            "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torchtsetlin": __import__("torchtsetlin").__version__}


# --------------------------------------------------------------------------- construction
def new_record(arm: str, family: str, seed: int, argv: Sequence[str], hp: dict, data: dict,
               paper: Optional[str] = None, notes: str = "") -> dict:
    """An empty record with everything that is known before training starts."""
    return {"arm": arm, "family": family, "paper": paper, "seed": int(seed),
            "argv": list(argv), "git_sha": git_sha(), "env": env_info(),
            "started": _dt.datetime.now().isoformat(timespec="seconds"), "wall_s": None,
            "data": dict(data), "hp": dict(hp), "capacity": None, "curve": [],
            "selected_epoch": None, "val_acc": None, "test_acc": None,
            "test_acc_per_class": None, "test_predictions_sha": None,
            "diagnostics": {}, "notes": notes}


def add_epoch(rec: dict, epoch: int, train_acc: float, val_acc: float, epoch_s: float,
              **diagnostics) -> None:
    """Append one epoch. ``diagnostics`` are the per-epoch cheap probes of
    ``diagnostics.py`` -- collected during the run because collecting them afterwards means
    re-running everything (PLAN Section 7.6)."""
    if "test_acc" in diagnostics:
        raise ValueError("per-epoch test accuracy does not belong in the curve (see module doc)")
    rec["curve"].append({"epoch": int(epoch), "train_acc": float(train_acc),
                         "val_acc": float(val_acc), "epoch_s": float(epoch_s),
                         **{k: _jsonable(v) for k, v in diagnostics.items()}})


def select_epoch(curve: Sequence[dict]) -> int:
    """``argmax(val_acc)``, earliest epoch on a tie. The **only** selection rule in the
    harness. Returns the record's 1-based ``epoch`` value, not the list position."""
    if not curve:
        raise ValueError("empty curve: nothing to select")
    best = max(range(len(curve)), key=lambda i: (curve[i]["val_acc"], -i))
    return int(curve[best]["epoch"])


def finalize(rec: dict, test_acc: float, test_preds, selected_epoch: int, capacity: dict,
             diagnostics: dict, wall_s: Optional[float] = None,
             preds_dir: Optional[str] = None, test_labels=None,
             test_acc_per_class: Optional[Sequence[float]] = None) -> None:
    """Close the record: the single test evaluation, the capacity table, the diagnostics.

    ``test_preds`` is the length-10000 predicted-class vector at ``selected_epoch``. It is
    written to ``results/preds/<arm>_seed<k>.npy`` (int8) and hashed, which is what makes the
    paired McNemar / bootstrap tests of PLAN Section 7.2 possible without re-running anything.
    """
    preds = np.asarray(test_preds.cpu() if torch.is_tensor(test_preds) else test_preds,
                       dtype=np.int8).reshape(-1)
    os.makedirs(preds_dir or PREDS, exist_ok=True)
    path = os.path.join(preds_dir or PREDS, f"{rec['arm']}_seed{rec['seed']}.npy")
    np.save(path, preds)
    rec["test_predictions_sha"] = hashlib.sha1(preds.tobytes()).hexdigest()
    rec["test_predictions_path"] = os.path.relpath(path, ROOT)
    rec["selected_epoch"] = int(selected_epoch)
    rec["val_acc"] = float(next(e["val_acc"] for e in rec["curve"]
                                if e["epoch"] == int(selected_epoch)))
    rec["test_acc"] = float(test_acc)
    if test_acc_per_class is None and test_labels is not None:
        # The class count comes from the RECORD, not from the labels: inferring it as
        # max(label)+1 silently produces a short list whenever the top class is absent, and
        # `validate` would then reject a perfectly good record for the wrong reason.
        test_acc_per_class = per_class_accuracy(preds, test_labels,
                                                int(rec["data"].get("n_classes", 10)))
        got = float((torch.as_tensor(preds).reshape(-1).cpu()
                     == torch.as_tensor(test_labels).reshape(-1).cpu()).float().mean())
        if abs(got - float(test_acc)) > 1e-6:
            raise ValueError(f"test_acc={test_acc} disagrees with the saved predictions ({got})")
    if test_acc_per_class is None:
        raise ValueError("finalize needs test_labels or an explicit test_acc_per_class")
    rec["test_acc_per_class"] = [float(v) for v in test_acc_per_class]
    rec["capacity"] = {k: _jsonable(v) for k, v in capacity.items()}
    rec["diagnostics"] = {k: _jsonable(v) for k, v in diagnostics.items()}
    if wall_s is not None:
        rec["wall_s"] = float(wall_s)


def per_class_accuracy(preds, labels, n_classes: Optional[int] = None) -> List[float]:
    """Per-class recall over ``n_classes`` classes (default: ``max(label) + 1``).

    ``finalize`` always passes ``rec['data']['n_classes']`` rather than relying on the
    default, so a two-class task (``data.task == 'cifar2'``) produces a length-2 list instead
    of ten entries of which eight are NaN, and a task whose top class happens to be absent
    from the probe still produces a full-length list."""
    p = torch.as_tensor(preds).reshape(-1).cpu()
    y = torch.as_tensor(labels).reshape(-1).cpu()
    K = int(n_classes) if n_classes is not None else int(y.max().item()) + 1
    return [float((p[y == c] == c).float().mean()) if int((y == c).sum()) else float("nan")
            for c in range(K)]


# --------------------------------------------------------------------------- validation
def admissible_as_reproduction(rec: dict) -> List[str]:
    """Extra conditions for a record that claims to reproduce a published result (PLAN §7.5).

    Separate from :func:`validate` on purpose: an exploratory run of a literature arm under a
    provisional configuration is a perfectly valid *record*, it just may not be the evidence
    behind a "reproduced" or "GAP" verdict. `T` and `s` span two orders of magnitude across
    the bibliography (LIBRARY_GAPS LG-010), so an arm running on the harness default is
    measuring our guess, not the paper.
    """
    e = validate(rec)
    if rec.get("paper") is None:
        e.append("no paper: nothing to reproduce")
    if rec.get("hp", {}).get("config_status") != "paper":
        e.append(f"config_status={rec.get('hp', {}).get('config_status')!r}: the arm is not "
                 "running the paper's stated configuration")
    if rec.get("hp", {}).get("T_source") != "explicit":
        e.append("T came from the harness ratio default, not from the paper (LG-010 A3)")
    return e


def validate(rec: dict) -> List[str]:
    """``[]`` means admissible. Every other return value is a reason to refuse the record."""
    e: List[str] = []
    for k in REQUIRED:
        if k not in rec:
            e.append(f"missing key {k!r}")
        elif rec[k] is None and k not in ("paper", "notes"):
            e.append(f"key {k!r} is null")
    if e:
        return e  # nothing below is meaningful with holes in the record
    if rec["family"] not in FAMILIES:
        e.append(f"family {rec['family']!r} not in {FAMILIES}")
    if not isinstance(rec["argv"], list) or not rec["argv"]:
        e.append("argv must be a non-empty list (C6: reproducible from a command line)")
    if rec["git_sha"] == "unknown":
        e.append("git_sha unknown (C6)")
    for k in DATA_REQUIRED:
        if k not in rec["data"]:
            e.append(f"data.{k} missing")
    if "split_hash" in rec["data"] and not isinstance(rec["data"]["split_hash"], str):
        e.append("data.split_hash must be a string")
    if rec["data"].get("n_val") not in (None, 5000):
        e.append(f"data.n_val={rec['data'].get('n_val')}: the protocol fixes it at 5000")
    if rec["data"].get("n_test") not in (None, 10000):
        e.append(f"data.n_test={rec['data'].get('n_test')}: the test set is never subsetted")
    for k in CAPACITY_REQUIRED:
        if k not in rec["capacity"]:
            e.append(f"capacity.{k} missing (Section 7.3 budget matching needs it)")
    if "batch_size" not in rec["hp"]:
        e.append("hp.batch_size missing (arms are only comparable at equal batch size)")
    # --- the curve and the selection rule ------------------------------------------------
    if not rec["curve"]:
        e.append("curve is empty")
    for i, ep in enumerate(rec["curve"]):
        for k in EPOCH_REQUIRED:
            if k not in ep:
                e.append(f"curve[{i}].{k} missing")
        if "test_acc" in ep:
            e.append(f"curve[{i}] carries test_acc: test data may not appear in the curve "
                     "the selector reads (leakage guard)")
    if not e:
        want = select_epoch(rec["curve"])
        if int(rec["selected_epoch"]) != want:
            e.append(f"selected_epoch={rec['selected_epoch']} but argmax(val_acc)={want}: "
                     "selection must be on validation only")
        sel = [x for x in rec["curve"] if x["epoch"] == int(rec["selected_epoch"])]
        if not sel:
            e.append(f"selected_epoch={rec['selected_epoch']} is not in the curve")
        elif abs(sel[0]["val_acc"] - float(rec["val_acc"])) > 1e-9:
            e.append("val_acc does not match the curve at selected_epoch")
    for k in ("val_acc", "test_acc"):
        if not (0.0 <= float(rec[k]) <= 1.0):
            e.append(f"{k}={rec[k]} outside [0,1]")
    # Ten unless the record declares another task. `data.n_classes` is absent from every
    # record written before 2026-09-21, and those are all CIFAR-10, so the default preserves
    # the guard exactly as it was for them.
    want_k = int(rec["data"].get("n_classes", 10))
    if (not isinstance(rec["test_acc_per_class"], list)
            or len(rec["test_acc_per_class"]) != want_k):
        e.append(f"test_acc_per_class must be a list of {want_k} floats "
                 f"(data.n_classes={rec['data'].get('n_classes', 'absent -> 10')})")
    sha = rec["test_predictions_sha"]
    if not (isinstance(sha, str) and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)):
        e.append("test_predictions_sha must be a sha1 hex digest")
    if not isinstance(rec["diagnostics"], dict) or not rec["diagnostics"]:
        e.append("diagnostics is empty (Section 7.6 requires per-run diagnostics)")
    elif int(rec["capacity"].get("n_clauses_total", 0)) > 0:
        for k in ("clause_len", "negation_fraction", "firing_rate", "clauses_used_frac"):
            if k not in rec["diagnostics"]:
                e.append(f"diagnostics.{k} missing (required for any clause-based arm)")
        # M2. Required for convolutional arms, where the patch aggregation is an OR whose
        # arity nobody has measured; null is correct for a flat or non-clause arm, where the
        # count is 1 by construction.
        if int(rec["capacity"].get("n_patches", 1)) > 1:
            if "match_count" not in rec["diagnostics"]:
                e.append("diagnostics.match_count missing (M2: required for conv arms)")
            elif rec["diagnostics"]["match_count"] is None:
                e.append("diagnostics.match_count is null on a convolutional arm")
    if rec["wall_s"] is not None and float(rec["wall_s"]) <= 0:
        e.append("wall_s must be positive")
    return e


def write(rec: dict, path: str, force: bool = False) -> str:
    """Validate, then write. An invalid record raises; nothing partial is left behind."""
    problems = validate(rec)
    if problems:
        raise ValueError("refusing to write an invalid record:\n  - " + "\n  - ".join(problems))
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} exists (re-run with --force to overwrite)")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=False)
    os.replace(tmp, path)
    return path


def read(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _jsonable(v):
    if torch.is_tensor(v):
        return v.tolist() if v.numel() > 1 else v.item()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


if __name__ == "__main__":  # self-test: validate() must actually refuse things
    import copy
    base = new_record("t", "existing", 0, ["run_arm.py"], {"batch_size": 50},
                      {"dataset": "cifar10", "booleanization": "therm4", "n_train": 45000,
                       "n_val": 5000, "n_test": 10000, "augment": None, "split_hash": "x"})
    add_epoch(base, 1, 0.2, 0.21, 1.0)
    add_epoch(base, 2, 0.3, 0.35, 1.0)
    add_epoch(base, 3, 0.4, 0.30, 1.0)
    yte = torch.zeros(10000, dtype=torch.long); yte[:6700] = 1
    finalize(base, 0.33, torch.zeros(10000, dtype=torch.long), select_epoch(base["curve"]),
             {"n_automata": 1, "n_clauses_total": 10, "literals_evaluated_per_image": 1,
              "state_bytes": 1},
             {"clause_len": {"median": 3}, "negation_fraction": 0.5,
              "firing_rate": {"median": 0.1}, "clauses_used_frac": 1.0}, wall_s=3.0,
             preds_dir="/tmp", test_labels=yte)
    assert validate(base) == [], validate(base)
    bad = copy.deepcopy(base); bad["selected_epoch"] = 3
    assert any("argmax" in m for m in validate(bad)), "test-selected epoch was not caught"
    bad = copy.deepcopy(base); bad["curve"][0]["test_acc"] = 0.4
    assert any("leakage" in m for m in validate(bad)), "test_acc in curve was not caught"
    bad = copy.deepcopy(base); del bad["capacity"]["n_automata"]
    assert validate(bad), "missing capacity key was not caught"
    bad = copy.deepcopy(base); bad["data"]["n_val"] = 10000
    assert validate(bad), "wrong val size was not caught"
    bad = copy.deepcopy(base); bad["diagnostics"] = {}
    assert validate(bad), "empty diagnostics was not caught"
    bad = copy.deepcopy(base); bad["test_acc_per_class"] = [0.1] * 9
    assert validate(bad), "wrong per-class length was not caught"
    bad = copy.deepcopy(base); bad["test_predictions_sha"] = "nope"
    assert validate(bad), "bad sha was not caught"
    # A two-class task (data.task='cifar2') carries data.n_classes=2 and therefore a
    # length-2 per-class list; a length-10 one on the same record must be refused.
    two = new_record("t2", "control", 0, ["run_arm.py"], {"batch_size": 50},
                     {"dataset": "cifar2", "task": "cifar2", "n_classes": 2,
                      "booleanization": "therm4", "n_train": 45000, "n_val": 5000,
                      "n_test": 10000, "augment": None, "split_hash": "x"})
    add_epoch(two, 1, 0.6, 0.62, 1.0)
    y2 = torch.zeros(10000, dtype=torch.long)
    y2[:6000] = 1
    finalize(two, 0.6, torch.ones(10000, dtype=torch.long), 1,
             {"n_automata": 1, "n_clauses_total": 10, "literals_evaluated_per_image": 1,
              "state_bytes": 1},
             {"clause_len": {"median": 3}, "negation_fraction": 0.5,
              "firing_rate": {"median": 0.1}, "clauses_used_frac": 1.0}, wall_s=3.0,
             preds_dir="/tmp", test_labels=y2)
    assert len(two["test_acc_per_class"]) == 2, two["test_acc_per_class"]
    assert validate(two) == [], validate(two)
    bad = copy.deepcopy(two)
    bad["test_acc_per_class"] = [0.1] * 10
    assert validate(bad), "a 10-entry per-class list on a 2-class record was not caught"
    print(f"record.py self-test OK (selected_epoch={base['selected_epoch']}, "
          f"val={base['val_acc']}, sha={base['test_predictions_sha'][:12]}...)")
