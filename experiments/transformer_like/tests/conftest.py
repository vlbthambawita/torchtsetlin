from __future__ import annotations

import numpy as np
import pytest
import torch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


@pytest.fixture(scope="session")
def smoke():
    from experiments.transformer_like.code import data as D
    return D.load_split(D.SMOKE, "train"), D.load_split(D.SMOKE, "valid")


@pytest.fixture(scope="session")
def tiny_encoders(smoke):
    """Small Q/K pair on the smoke subset (CPU), 1 epoch."""
    from experiments.transformer_like.code.patch_training import fit_encoders
    tr, _ = smoke
    q, k, info = fit_encoders(tr.images, tr.y, seed=5, device="cpu",
                              cfg={"M": 10, "T": 5, "epochs": 1}, log=lambda s: None)
    return q, k, info


def boolean_fixture(n=2000, seed=0):
    """10-class Boolean task: feature c is on for class c, plus 20 noise bits."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 10, n)
    X = rng.random((n, 30)) < 0.3
    X[:, :10] = False
    X[np.arange(n), y] = True
    return torch.as_tensor(X), torch.as_tensor(y)
