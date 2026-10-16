from __future__ import annotations

import numpy as np
import pytest
import torch

import torchtsetlin as tt

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="session")
def integ():
    from experiments.linearModel.code import data as D
    return D.load_split("integration_seed0", "train"), D.load_split("integration_seed0", "valid")


@pytest.fixture(scope="session")
def small_tm(integ):
    """A small trained flat TM (M = 4 per class) on the integration subset."""
    tr, _ = integ
    torch.manual_seed(0)
    m = tt.TsetlinMachine(784, 10, n_clauses=4, T=4, s=5.0)
    B = torch.as_tensor(tr.X, dtype=torch.float32) > 0.5
    y = torch.as_tensor(tr.y)
    for _ in range(2):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 32):
            idx = perm[i:i + 32]
            m.update(B[idx], y[idx])
    return m.eval()


@pytest.fixture(scope="session")
def small_conv(integ):
    tr, _ = integ
    torch.manual_seed(0)
    m = tt.ConvTsetlinMachine(10, 4, T=4, s=5.0, patch_size=10, input_shape=(1, 28, 28))
    B = (torch.as_tensor(tr.X[:500], dtype=torch.float32) > 0.5).reshape(-1, 1, 28, 28)
    y = torch.as_tensor(tr.y[:500])
    for i in range(0, 500, 50):
        m.update(B[i:i + 50], y[i:i + 50])
    return m.eval()


def random_problem(rng, N=40, d=3, K=2, M=2, clause_bias=True, slopes=True, p_gate=0.5):
    from experiments.linearModel.code.hybrid import Layout
    X = rng.random((N, d))
    X[rng.random((N, d)) < 0.3] = 0.0
    G = rng.random((N, K * M)) < p_gate
    y = rng.integers(0, K, N)
    Y = np.eye(K)[y]
    return X, G, Y, y, Layout(K, M, d, clause_bias, slopes)
