"""MNIST, the fixed split, manifests and the sealed test set (PLAN §4, spec §5).

Images stay ``uint8`` ``(N, 28, 28)``; every arm binarizes with ``u >= 128`` (D9).
``load_split(..., "test")`` refuses unless ``unseal=True`` *and* the frozen experiment
manifest exists (rule C3, spec test 32); every allowed access is appended to
``results/test_access_log.jsonl``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .common import FROZEN, MANIFESTS, REPO, TEST_LOG, sha256_array, sha256_file, write_json

SPLIT_SEED = 314159
N_VALID = 10_000
N_SMOKE_TRAIN, N_SMOKE_VALID = 1_000, 200
N_S2_TRAIN = 5_000
MAIN = f"split_seed{SPLIT_SEED}"
SMOKE = f"smoke_seed{SPLIT_SEED}"
S2 = f"s2_seed{SPLIT_SEED}"
_MEMO: Dict = {}


def raw_root() -> Path:
    return REPO / "data"


def _raw():
    if "raw" not in _MEMO:
        from torchvision.datasets import MNIST
        tr = MNIST(str(raw_root()), train=True, download=False)
        te = MNIST(str(raw_root()), train=False, download=False)
        _MEMO["raw"] = (tr.data.numpy().astype(np.uint8), tr.targets.numpy().astype(np.int64),
                        te.data.numpy().astype(np.uint8), te.targets.numpy().astype(np.int64))
    return _MEMO["raw"]


def stratified_take(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` indices with class proportions preserved (largest remainder), sorted."""
    classes, counts = np.unique(y, return_counts=True)
    exact = counts * n / len(y)
    take = np.floor(exact).astype(int)
    rem = n - take.sum()
    take[np.argsort(-(exact - take), kind="stable")[:rem]] += 1
    out = [rng.choice(np.flatnonzero(y == c), size=t, replace=False) for c, t in zip(classes, take)]
    return np.sort(np.concatenate(out))


def _sha(m: dict) -> str:
    return sha256_array(np.concatenate([np.asarray(m["train"]), [-1], np.asarray(m["valid"])]).astype(np.int64))


def make_manifests(seed: int = SPLIT_SEED) -> Dict[str, dict]:
    _, ytr, _, _ = _raw()
    rng = np.random.default_rng(seed)
    va = stratified_take(ytr, N_VALID, rng)
    tr = np.setdiff1d(np.arange(len(ytr)), va)
    rs = np.random.default_rng([seed, 1])
    s_tr = tr[stratified_take(ytr[tr], N_SMOKE_TRAIN, rs)]
    s_va = va[stratified_take(ytr[va], N_SMOKE_VALID, rs)]
    r2 = np.random.default_rng([seed, 2])
    s2_tr = tr[stratified_take(ytr[tr], N_S2_TRAIN, r2)]
    out = {}
    for name, a, b in ((MAIN, tr, va), (SMOKE, s_tr, s_va), (S2, s2_tr, va)):
        m = {"name": name, "seed": seed, "pool": "mnist_official_train",
             "train": a.tolist(), "valid": b.tolist(),
             "class_counts": {"train": np.bincount(ytr[a], minlength=10).tolist(),
                              "valid": np.bincount(ytr[b], minlength=10).tolist()}}
        m["sha256"] = _sha(m)
        out[name] = m
    return out


def write_manifests(seed: int = SPLIT_SEED) -> Dict[str, str]:
    ms = make_manifests(seed)
    for name, m in ms.items():
        write_json(MANIFESTS / f"{name}.json", m)
    write_json(MANIFESTS / "dataset_manifest.json", dataset_manifest(ms[MAIN]))
    return {k: v["sha256"] for k, v in ms.items()}


def read_manifest(name: str) -> dict:
    m = json.loads((MANIFESTS / f"{name}.json").read_text())
    if _sha(m) != m["sha256"]:
        raise RuntimeError(f"manifest {name} corrupted")
    if set(m["train"]) & set(m["valid"]):
        raise RuntimeError(f"manifest {name}: train/valid overlap")
    return m


def dataset_manifest(main: dict) -> dict:
    Xtr, ytr, Xte, yte = _raw()
    raw = raw_root() / "MNIST" / "raw"
    files = {p.name: sha256_file(p) for p in sorted(raw.iterdir()) if p.is_file()}
    # identical-content duplicates across partitions (reported, protocol unchanged; spec §5)
    def hashes(X):
        return [sha256_array(x) for x in X]
    h_tr = hashes(Xtr[main["train"]])
    h_va = hashes(Xtr[main["valid"]])
    h_te = hashes(Xte)
    s_tr, s_va, s_te = set(h_tr), set(h_va), set(h_te)
    return {
        "source": "torchvision.datasets.MNIST(root='data', download=False); files under data/MNIST/raw",
        "raw_files_sha256": files,
        "n_train_pool": int(len(ytr)), "n_test": int(len(yte)),
        "shape": [28, 28], "dtype": "uint8", "labels": list(range(10)),
        "content_duplicates": {
            "train_valid": len(s_tr & s_va), "train_test": len(s_tr & s_te), "valid_test": len(s_va & s_te),
            "within_train": len(h_tr) - len(s_tr),
        },
    }


@dataclass
class Part:
    images: np.ndarray  # uint8 (N, 28, 28)
    y: np.ndarray       # int64 (N,)
    ids: np.ndarray     # int64 (N,), index into the official pool
    name: str
    pool: str

    def __len__(self) -> int:
        return len(self.y)


def load_split(manifest: str, part: str, unseal: bool = False, purpose: str = "",
               frozen_path: Optional[Path] = None) -> Part:
    """``part`` in ``{"train", "valid", "test"}``. Test needs ``unseal=True`` and a frozen manifest."""
    Xtr, ytr, Xte, yte = _raw()
    if part == "test":
        fp = FROZEN if frozen_path is None else frozen_path
        if not unseal:
            raise PermissionError("test split is sealed (rule C3): pass unseal=True")
        if not Path(fp).exists():
            raise PermissionError("test split is sealed until results/frozen_experiment.json exists")
        log_test_access(purpose)
        ids = np.arange(len(yte))
        return Part(Xte, yte, ids, "test", "mnist_official_test")
    if part not in ("train", "valid"):
        raise ValueError(part)
    m = read_manifest(manifest)
    ids = np.asarray(m[part], dtype=np.int64)
    return Part(Xtr[ids], ytr[ids], ids, f"{manifest}:{part}", "mnist_official_train")


def log_test_access(purpose: str) -> None:
    import sys
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_LOG, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "purpose": purpose,
                            "argv": sys.argv}) + "\n")


if __name__ == "__main__":
    print(write_manifests())
