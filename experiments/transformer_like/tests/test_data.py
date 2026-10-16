"""Spec tests 1, 6, 7 and test-set sealing."""
import json

import numpy as np
import pytest
import torch

from experiments.transformer_like.code import data as D
from experiments.transformer_like.code.common import MANIFESTS, sha256_file, stream
from experiments.transformer_like.code.tokenization import binarize


def test_pixel_threshold():  # spec 1
    u = torch.tensor([[0, 127, 128, 255]], dtype=torch.uint8)
    assert binarize(u).tolist() == [[False, False, True, True]]


def test_split_disjoint_and_complete():  # spec 6
    m = D.read_manifest(D.MAIN)
    tr, va = set(m["train"]), set(m["valid"])
    assert len(tr) == 50_000 and len(va) == 10_000 and not (tr & va) and tr | va == set(range(60_000))
    s = D.read_manifest(D.SMOKE)
    assert set(s["train"]) <= tr and set(s["valid"]) <= va
    s2 = D.read_manifest(D.S2)
    assert set(s2["train"]) <= tr and len(s2["train"]) == 5000
    for cc in m["class_counts"].values():
        assert len(cc) == 10 and min(cc) > 0


def test_manifests_reproduce():  # spec 7
    ms = D.make_manifests()
    for name, m in ms.items():
        assert m["sha256"] == json.loads((MANIFESTS / f"{name}.json").read_text())["sha256"]
    a = stream(3, "q-patches", 0).random(5)
    b = stream(3, "q-patches", 0).random(5)
    c = stream(3, "k-patches", 0).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_raw_file_hashes():
    dm = json.loads((MANIFESTS / "dataset_manifest.json").read_text())
    raw = D.raw_root() / "MNIST" / "raw"
    for name, h in dm["raw_files_sha256"].items():
        assert sha256_file(raw / name) == h


def test_test_split_sealed(tmp_path):  # spec 32
    with pytest.raises(PermissionError):
        D.load_split(D.MAIN, "test")
    with pytest.raises(PermissionError):
        D.load_split(D.MAIN, "test", unseal=True, frozen_path=tmp_path / "missing.json")


def test_labels_not_in_transform():
    import inspect
    from experiments.transformer_like.code import features
    sig = inspect.signature(features.transform)
    assert not any("label" in p or p == "y" for p in sig.parameters)
