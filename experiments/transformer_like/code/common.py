"""Paths, provenance, deterministic RNG streams and record writing (PLAN §0, D13)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
RESULTS = EXP / "results"
ART = EXP / "artifacts"
MANIFESTS = EXP / "manifests"
LOGS = EXP / "logs"
FROZEN = RESULTS / "frozen_experiment.json"
TEST_LOG = RESULTS / "test_access_log.jsonl"


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


def src_diff() -> str:
    """``git diff --stat HEAD -- src/`` (must be empty, rule C1)."""
    return subprocess.run(["git", "diff", "--stat", "HEAD", "--", "src/"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def env_info(device: str = "cpu") -> Dict[str, Any]:
    import torchtsetlin as tt
    info = {
        "python": sys.version.split()[0], "torch": torch.__version__, "torchtsetlin": tt.__version__,
        "cuda": torch.version.cuda, "platform": platform.platform(), "threads": torch.get_num_threads(),
        "device": device, "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return info


def provenance(device: str = "cpu") -> Dict[str, Any]:
    return {"git_sha": git_sha(), "src_diff": src_diff(), "argv": list(sys.argv),
            "env": env_info(device), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_array(a) -> str:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(name: str) -> int:
    """Process-independent integer for a stream name (never Python's ``hash()``)."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def stream(seed: int, *keys) -> np.random.Generator:
    """Deterministic, independent RNG stream keyed by ``(seed, *keys)`` (D13)."""
    ent = [int(seed)] + [stable_int(str(k)) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(ent))


def torch_seed(seed: int, *keys) -> int:
    return int(stream(seed, "torch", *keys).integers(0, 2**31 - 1))


def seed_torch(seed: int, *keys) -> int:
    s = torch_seed(seed, *keys)
    torch.manual_seed(s)
    return s


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_default))
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def resolve_T(M: int, ratio: float) -> int:
    """Spec §11: ``T = max(1, floor(M * ratio + 0.5))``."""
    return max(1, int(np.floor(M * ratio + 0.5)))


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
