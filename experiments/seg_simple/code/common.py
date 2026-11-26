"""Paths, environment knobs, records and small helpers shared by every script (seg_simple PLAN).

Environment variables (all optional):
  SS_DATA        Kvasir-SEG folder (holds images/ and masks/)
  SS_DEVICE      torch device of a runner (default cuda:0; run_gpu.sh restricts the visible GPU)
  SS_PHASES      phases a runner works on (queue_runner.py)
  SS_WANDB       1 (default) online, offline, 0 off;  SS_WANDB_PROJECT / SS_WANDB_ENTITY
  SS_PATIENCE    early-stopping patience in epochs (default 30);  SS_MAX_EPOCHS cap (300)
  SS_PANEL_EVERY epochs between prediction panels (default 15)
  SS_GPUS        number of GPUs, for the wall-time estimate of the cost table (default 3)
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

PROG = Path(__file__).resolve().parents[1]
REPO = PROG.parents[1]
CODE = PROG / "code"
RESULTS = PROG / "results"
CACHE = PROG / ".cache"
LOGS = PROG / "logs"
MODELS = PROG / "models"
FIGURES = PROG / "figures"
MANIFESTS = PROG / "manifests"

S = 64                                              # working resolution of the plan
IGNORE = 255                                        # label of pixels that give no feedback
WINDOW = 9                                          # window of the window models (section 12.2)
SAMPLED = 256                                       # pixels per image and epoch when pixels are sampled
DEVICE = os.environ.get("SS_DEVICE", "cuda:0")
PATIENCE = int(os.environ.get("SS_PATIENCE", "30"))
MAX_EPOCHS = int(os.environ.get("SS_MAX_EPOCHS", "300"))
PANEL_EVERY = int(os.environ.get("SS_PANEL_EVERY", "15"))
FINAL = ("done",)                                   # record statuses that need no more work


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(obj, path):
    """Atomic write: a reader never sees half a file, a killed writer leaves the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    try:
        import numpy as np
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def record_path(tag):
    return RESULTS / "runs" / f"{tag}.json"


def status_of(tag):
    r = read_json(record_path(tag))
    return r.get("status") if r else None


def is_done(tag):
    return status_of(tag) in FINAL


def model_dir(tag):
    return MODELS / tag


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_SRC_HASH = None


def src_tree_sha256():
    """Hash of ``src/torchtsetlin/**/*.py`` (rule C1: the library is read-only here)."""
    global _SRC_HASH
    if _SRC_HASH is None:
        h = hashlib.sha256()
        root = REPO / "src" / "torchtsetlin"
        for p in sorted(root.rglob("*.py")):
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
        _SRC_HASH = h.hexdigest()
    return _SRC_HASH


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                # noqa: BLE001
        return None


def env_info(device=None):
    import torch

    import torchtsetlin
    info = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "torchtsetlin": torchtsetlin.__version__, "commit": git_commit(),
            "src_tree_sha256": src_tree_sha256(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def set_seed(seed):
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stop_requested():
    return (PROG / "STOP").exists()
