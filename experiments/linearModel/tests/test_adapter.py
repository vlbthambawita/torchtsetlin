"""Phase 1 / gate G1: encoder, complement mapping, adapter fidelity, empty/contradictory clauses."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from experiments.linearModel.code import adapter as A

import torchtsetlin as tt

from .conftest import DEVICES


def test_encoder_boundaries():
    t = np.float32(0.5)
    x = np.array([[np.nextafter(t, np.float32(0)), t, np.nextafter(t, np.float32(1))]], dtype=np.float64)
    B = tt.data.Binarizer(0.5)(torch.as_tensor(x, dtype=torch.float32))
    assert B.tolist() == [[False, False, True]]
    rules = A.FrozenRules(K=1, M=1, n_features=3, threshold=0.5, include=np.zeros((1, 6), bool),
                          empty_clause_output=[0], polarity=np.ones(1, np.int8), weight=None, source_identity="x")
    assert rules.encode(x).tolist() == [[False, False, True]]
    # the float64 view of every MNIST level maps to the same bit as its float32 cast
    lv = np.arange(256) / 255.0
    assert np.array_equal(lv > 0.5, lv.astype(np.float32) > np.float32(0.5))


def test_complement_mapping(integ):
    tr, _ = integ
    B = torch.as_tensor(tr.X[:200], dtype=torch.float32) > 0.5
    L = tt.functional.to_literals(B).bool()
    F = B.shape[1]
    assert torch.all(L[:, :F] != L[:, F:])
    assert torch.equal(L[:, :F], B)


@pytest.mark.parametrize("device", DEVICES)
def test_adapter_fidelity(small_tm, integ, device):
    tr, va = integ
    m = copy.deepcopy(small_tm).to(device)
    ident = A.snapshot_identity(m)
    probes = np.concatenate([tr.X, va.X, A.threshold_probes(tr.X, 0.5)])
    rules = A.freeze_rules(m, 0.5, probes, device=device)      # asserts element-wise equality
    assert A.snapshot_identity(m) == ident
    assert rules.include.sum() > 0
    # spec-literal per-sample evaluation agrees with the vectorised evaluator
    B = rules.encode(probes[:50]).numpy()
    G = rules.evaluate_bool(torch.as_tensor(B), device=device)
    for i in range(50):
        for c, rule in enumerate(rules.rules()):
            assert A.evaluate_frozen_clause(rule, B[i]) == G[i, c]


def test_empty_and_contradictory_clauses(small_tm, integ):
    tr, _ = integ
    m = copy.deepcopy(small_tm)
    n = m.n_states
    st = m.ta_state.clone()
    st[0] = n - 1                       # clause 0: empty
    st[1] = n - 1
    st[1, 5] = n                        # clause 1: x_5 AND NOT x_5
    st[1, 784 + 5] = n
    m.ta_state.copy_(st)
    m._refresh_include()
    rules = A.freeze_rules(m, 0.5, tr.X)
    G = rules(tr.X)
    assert not G[:, 0].any()            # empty clause -> False in prediction mode
    assert not G[:, 1].any()            # contradiction never fires
    assert rules.rules()[0].included_literals == ()
    assert rules.rules()[0].empty_clause_output == 0
    # the library agrees (also pinned by tests/test_models.py)
    raw = A.raw_clause_outputs(m, rules.encode(tr.X))
    assert not raw[:, 0].any() and not raw[:, 1].any()


def test_frozen_rules_immutable(small_tm, integ):
    rules = A.freeze_rules(small_tm, 0.5, integ[0].X[:100])
    with pytest.raises(ValueError):
        rules.include[0, 0] = True


def test_frozen_rules_state_roundtrip(small_tm, integ):
    rules = A.freeze_rules(small_tm, 0.5, integ[0].X[:100])
    r2 = A.FrozenRules.from_state(rules.state())
    assert r2.fingerprint == rules.fingerprint
    assert np.array_equal(r2(integ[1].X), rules(integ[1].X))


@pytest.mark.parametrize("device", DEVICES)
def test_library_gates_conv(small_conv, integ, device):
    _, va = integ
    m = copy.deepcopy(small_conv).to(device)
    gates = A.LibraryGates(m, 0.5)
    G = gates(va.X[:300])
    B = (torch.as_tensor(va.X[:300], dtype=torch.float32) > 0.5).reshape(-1, 1, 28, 28).to(device)
    m.eval()
    assert np.array_equal(G, m.evaluate_clauses(B).cpu().numpy())
    g2 = A.LibraryGates.from_state(gates.state(), device=device)
    assert g2.fingerprint == gates.fingerprint
    assert np.array_equal(g2(va.X[:300]), G)
    # prediction path draws no random numbers: repeated evaluation is identical
    assert np.array_equal(gates(va.X[:300]), G)
