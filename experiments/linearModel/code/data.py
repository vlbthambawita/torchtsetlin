"""MNIST splits, manifests and the two input views (PLAN §4, spec §3, §14.1).

- ``X`` (float64, ``pixel / 255``, row-major, ``d = 784``) feeds the linear outputs.
- ``B = encoder(X)`` (bool) feeds the frozen clauses.
- ``Y`` is the one-hot coefficient-training target; ``y`` the class index.

The official 10k test split is sealed (rule C3): :func:`load_split` refuses ``"test"`` unless
``unseal=True`` is passed, and every unsealed access is appended to
``results/test_access_log.jsonl`` by the caller (:func:`log_test_access`).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
MANIFESTS = EXP / "manifests"
RESULTS = EXP / "results"

N_VALID = 10_000
SPLIT_SEED = 0
N_INT_TRAIN, N_INT_VALID = 2_000, 1_000
K = 10
D = 784

_MEMO: Dict = {}


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _raw():
    if "raw" not in _MEMO:
        from torchvision.datasets import MNIST
        root = str(REPO / "data")
        tr, te = MNIST(root, train=True, download=False), MNIST(root, train=False, download=False)
        _MEMO["raw"] = (tr.data.numpy().reshape(-1, D), tr.targets.numpy().astype(np.int64),
                        te.data.numpy().reshape(-1, D), te.targets.numpy().astype(np.int64))
    return _MEMO["raw"]


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` indices of ``y`` with class proportions preserved (largest-remainder rounding).
    Returned sorted so the order does not depend on the draw."""
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * n / len(y)
    take = np.floor(exact).astype(int)
    rem = n - take.sum()
    take[np.argsort(-(exact - take), kind="stable")[:rem]] += 1
    out = [rng.choice(np.flatnonzero(y == c), size=t, replace=False) for c, t in zip(classes, take)]
    return np.sort(np.concatenate(out))


def make_manifests(seed: int = SPLIT_SEED) -> Dict[str, dict]:
    """Main 50k/10k split of the official training pool + the 2k/1k integration subset."""
    _, ytr, _, _ = _raw()
    rng = np.random.default_rng(seed)
    va = stratified_take(ytr, N_VALID, rng)
    tr = np.setdiff1d(np.arange(len(ytr)), va)
    rng_i = np.random.default_rng([seed, 1])
    itr = tr[stratified_take(ytr[tr], N_INT_TRAIN, rng_i)]
    iva = va[stratified_take(ytr[va], N_INT_VALID, rng_i)]
    main = {"name": f"split_seed{seed}", "seed": seed, "pool": "mnist_official_train",
            "train": tr.tolist(), "valid": va.tolist()}
    integ = {"name": f"integration_seed{seed}", "seed": seed, "pool": "mnist_official_train",
             "parent": main["name"], "train": itr.tolist(), "valid": iva.tolist()}
    for m in (main, integ):
        m["sha256"] = sha256_array(np.concatenate([np.asarray(m["train"]), [-1], np.asarray(m["valid"])]))
    return {"main": main, "integration": integ}


def write_manifests(seed: int = SPLIT_SEED) -> Dict[str, str]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    ms = make_manifests(seed)
    (MANIFESTS / f"split_seed{seed}.json").write_text(json.dumps(ms["main"]))
    (MANIFESTS / f"integration_seed{seed}.json").write_text(json.dumps(ms["integration"]))
    return {k: v["sha256"] for k, v in ms.items()}


def read_manifest(name: str) -> dict:
    m = json.loads((MANIFESTS / f"{name}.json").read_text())
    sha = sha256_array(np.concatenate([np.asarray(m["train"]), [-1], np.asarray(m["valid"])]))
    if sha != m["sha256"]:
        raise RuntimeError(f"manifest {name} corrupted: {sha} != {m['sha256']}")
    return m


@dataclass
class Part:
    """One partition: float view ``X`` (float64), labels ``y``, one-hot ``Y``."""
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    name: str

    @property
    def Y(self) -> np.ndarray:
        Y = np.zeros((len(self.y), K), dtype=np.float64)
        Y[np.arange(len(self.y)), self.y] = 1.0
        return Y

    def __len__(self) -> int:
        return len(self.y)


def load_split(manifest: str, part: str, unseal: bool = False) -> Part:
    """``part`` in ``{"train", "valid", "test"}``. ``"test"`` is the official 10k test set
    and requires ``unseal=True`` (rule C3)."""
    Xtr, ytr, Xte, yte = _raw()
    if part == "test":
        if not unseal:
            raise PermissionError("test split is sealed (PLAN C3); pass unseal=True in Phase 7 only")
        idx = np.arange(len(yte))
        X, y = Xte, yte
    else:
        idx = np.asarray(read_manifest(manifest)[part])
        X, y = Xtr[idx], ytr[idx]
    Xf = X.astype(np.float64) / 255.0
    if not np.isfinite(Xf).all():
        raise ValueError("non-finite input")
    return Part(X=Xf, y=y.copy(), ids=idx, name=f"{manifest}/{part}")


def log_test_access(record: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    with open(RESULTS / "test_access_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def boolean_view(X: np.ndarray, threshold: float = 0.5, device: Optional[str] = None) -> torch.Tensor:
    """``tt.data.Binarizer(threshold)`` applied to the float view (spec §3.1: ``x > t``)."""
    import torchtsetlin as tt
    enc = tt.data.Binarizer(threshold)
    B = enc(torch.as_tensor(X, dtype=torch.float32))
    return B.to(device) if device is not None else B


if __name__ == "__main__":
    print(write_manifests())
