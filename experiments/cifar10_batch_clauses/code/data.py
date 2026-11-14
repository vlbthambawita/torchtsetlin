"""Download CIFAR-10 (official python version), verify it, and write .cache/data/cifar10_uint8.npz
plus the fixed validation split (results/split.json). Idempotent; run once by ../setup.sh.

Usage: python code/data.py [--from-dir DIR]   # DIR = an extracted cifar-10-batches-py (no download)
"""

import argparse
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from common import CIFAR_MD5, CIFAR_SHA256, CIFAR_URL, DATA, _sha256, raw_path, split


def fetch():
    tgz = DATA / "cifar-10-python.tar.gz"
    if not tgz.exists():
        print(f"downloading {CIFAR_URL} ...", flush=True)
        tmp = tgz.with_suffix(".part")
        urllib.request.urlretrieve(CIFAR_URL, tmp)
        tmp.replace(tgz)
    md5 = hashlib.md5(tgz.read_bytes()).hexdigest()
    if md5 != CIFAR_MD5:
        raise SystemExit(f"{tgz}: md5 {md5} != {CIFAR_MD5}; delete it and retry")
    out = DATA / "cifar-10-batches-py"
    if not out.exists():
        with tarfile.open(tgz) as t:
            t.extractall(DATA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, default=None)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    src = a.from_dir or fetch()
    for name, sha in CIFAR_SHA256.items():
        got = _sha256(src / name)
        if got != sha:
            raise SystemExit(f"{src / name}: sha256 {got} != {sha}")

    def batch(name):
        with open(src / name, "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return np.ascontiguousarray(x), np.asarray(d[b"labels"], dtype=np.int64)

    parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    xte, yte = batch("test_batch")
    assert xtr.shape == (50000, 32, 32, 3) and xte.shape == (10000, 32, 32, 3)
    tmp = raw_path().with_suffix(".tmp.npz")
    np.savez(tmp, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
    tmp.replace(raw_path())
    tr, va = split(ytr)
    print(f"wrote {raw_path()} (train {len(tr)}, val {len(va)}, test {len(yte)}); split in results/split.json")


if __name__ == "__main__":
    main()
