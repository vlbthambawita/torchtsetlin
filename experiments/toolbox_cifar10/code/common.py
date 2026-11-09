"""Shared paths, data loading and result records for the toolbox_cifar10 programme.

Rules (see ../README.md): ``src/torchtsetlin/**`` is read-only here; every reported number comes
from a ``results/*.json`` record carrying ``argv``, ``seed``, ``git_sha`` and ``env``.
"""

import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROG = HERE.parent                                   # experiments/toolbox_cifar10
REPO = PROG.parents[1]
CIFAR10 = REPO / "data" / "cifar-10-batches-py"
# TB_RESULTS / TB_CACHE redirect a run (smoke tests never touch the real records).
RESULTS = Path(os.environ.get("TB_RESULTS", PROG / "results"))
CACHE = Path(os.environ.get("TB_CACHE", PROG / ".cache"))   # git-ignored: encodings, sums, ckpts
ENC_CACHE = PROG / ".cache" / "enc"                         # encodings are shared by smoke and real runs
PAPER = Path("/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper")


def load_cifar10():
    """CIFAR-10 exactly as ``keras.datasets.cifar10.load_data()`` returns it (the reference
    scripts' loader): uint8 ``(N, 32, 32, 3)`` RGB, int64 labels, official 50 000 / 10 000."""
    def batch(name):
        with open(CIFAR10 / name, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return x, np.asarray(d[b"labels"], dtype=np.int64)

    parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.ascontiguousarray(np.concatenate([p[0] for p in parts]))
    ytr = np.concatenate([p[1] for p in parts])
    xte, yte = batch("test_batch")
    return xtr, ytr, np.ascontiguousarray(xte), yte


def test_labels():
    with open(CIFAR10 / "test_batch", "rb") as fh:
        return np.asarray(pickle.load(fh, encoding="bytes")[b"labels"], dtype=np.int64)


def git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "--", "src"]).returncode != 0
        return sha + ("+src-dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def env():
    import torchtsetlin
    e = {"python": sys.version.split()[0], "torch": torch.__version__,
         "torchtsetlin": torchtsetlin.__version__, "numpy": np.__version__,
         "host": platform.node(), "cuda": torch.version.cuda}
    try:
        import cv2
        e["opencv"] = cv2.__version__
    except ImportError:  # pragma: no cover
        pass
    if torch.cuda.is_available():
        e["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return e


def new_record(kind, **fields):
    return {"kind": kind, "argv": sys.argv, "git_sha": git_sha(), "env": env(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}


def write_json(rec, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=float) + "\n")
    tmp.replace(path)                                 # never leave a half-written record


def read_json(path):
    return json.loads(Path(path).read_text())
