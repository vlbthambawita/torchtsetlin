"""Run records: provenance capture, source-hash guard, immutable JSON writes (PLAN section 9)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Dict

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(HERE, "results")
ARTIFACTS = os.path.join(HERE, "artifacts")
SCHEMA = 1


def src_tree_hash() -> str:
    h = hashlib.sha256()
    root = os.path.join(REPO, "src", "torchtsetlin")
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                h.update(os.path.relpath(p, REPO).encode())
                with open(p, "rb") as fh:
                    h.update(fh.read())
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for d in ("code",):
        for f in sorted(os.listdir(os.path.join(HERE, d))):
            if f.endswith(".py"):
                with open(os.path.join(HERE, d, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", "-C", REPO, *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover
        return ""


def environment(device: str) -> Dict:
    dev = {}
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = {"gpu": torch.cuda.get_device_name(torch.device(device)), "cuda": torch.version.cuda}
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": device,
        "threads": torch.get_num_threads(),
        **dev,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_src_diff_sha256": hashlib.sha256(_git("diff", "HEAD", "--", "src").encode()).hexdigest(),
        "src_tree_sha256": src_tree_hash(),
        "exp_code_sha256": code_hash(),
    }


def manifest_hash(dataset: str) -> str:
    name = {"kvasir": "kvasir_split.json", "acdc": "acdc_split.json"}.get(dataset)
    if name is None:
        return ""
    with open(os.path.join(HERE, "manifests", name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, torch.Tensor):
        return _clean(o.detach().cpu().numpy())
    return o


def write_record(rec: Dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    rec = dict(rec)
    rec["schema"] = SCHEMA
    rec["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = os.path.join(RESULTS, f"{rec['run_id']}.json")
    if os.path.exists(path):
        raise FileExistsError(f"run record {path} exists (records are immutable)")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(rec), f)
    os.replace(tmp, path)
    return path


def load_records(prefix: str = "") -> Dict[str, Dict]:
    out = {}
    for f in sorted(os.listdir(RESULTS)):
        if f.endswith(".json") and f.startswith(prefix):
            with open(os.path.join(RESULTS, f)) as fh:
                out[f[:-5]] = json.load(fh)
    return out
