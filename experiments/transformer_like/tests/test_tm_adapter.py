"""Spec tests 8-13, 34, 35 (backend contract)."""
import torch

from experiments.transformer_like.code.tm_adapter import TMAdapter
from .conftest import boolean_fixture


def _trained(device, epochs=5, M=20):
    X, y = boolean_fixture()
    torch.manual_seed(0)
    a = TMAdapter(30, M, M // 2, 3.0, device=device)
    for _ in range(epochs):
        a.fit_epoch(X, y)
    return a, X, y


def test_raw_scores_are_manual_signed_sums(device):  # spec 8, 9
    a, X, _ = _trained(device)
    co = a.clause_outputs(X[:200]).cpu()
    M = a.M
    manual = torch.zeros(200, 10)
    for c in range(10):
        for j in range(M):
            sign = 1.0 if j < (M + 1) // 2 else -1.0
            manual[:, c] += sign * co[:, c * M + j].float()
    raw = a.class_scores(X[:200]).cpu()
    assert torch.equal(raw, manual)
    assert torch.equal(raw.clamp(-a.T, a.T), a.votes(X[:200]).cpu())


def test_learns_boolean_fixture_and_class_order(device):  # spec 34, 9
    a, X, y = _trained(device)
    assert (a.predict(X).cpu() == y).double().mean() > 0.95


def test_tie_goes_to_lowest_index(device):  # spec 10
    a = TMAdapter(30, 4, 2, 3.0, device=device)
    X, _ = boolean_fixture(10)
    assert a.votes(X).abs().sum() == 0  # untrained: empty clauses are False when predicting
    assert a.predict(X).tolist() == [0] * 10


def test_fit_continues_and_changes_state(device):  # spec 11, 35
    X, y = boolean_fixture()
    torch.manual_seed(1)
    a = TMAdapter(30, 10, 5, 3.0, device=device)
    h0 = a.state_hash()
    a.fit_epoch(X, y)
    h1 = a.state_hash()
    s1 = a.model.ta_state.clone()
    a.fit_epoch(X, y)
    assert h0 != h1 != a.state_hash()
    # not reset: distance from the first-epoch state is small compared with distance from init
    init = torch.full_like(s1, 127)
    assert (a.model.ta_state - s1).abs().float().mean() < (a.model.ta_state - init).abs().float().mean() + 1
    assert int(a.model.ta_state.max()) <= 255 and int(a.model.ta_state.min()) >= 0


def test_save_load_exact(device, tmp_path):  # spec 12
    a, X, _ = _trained(device)
    a.save(tmp_path / "a.pt")
    b = TMAdapter.load(tmp_path / "a.pt", device)
    assert torch.equal(a.votes(X), b.votes(X)) and b.state_hash() == a.state_hash()


def test_no_stale_results(device):  # spec 13
    a, X, _ = _trained(device)
    v1 = a.votes(X[:100])
    v2 = a.votes(X[100:200])
    assert torch.equal(a.votes(X[:100]), v1) and torch.equal(a.votes(X[100:200]), v2)
    assert not torch.equal(v1, v2)


def test_scoring_uses_eval_semantics_and_restores_mode(device):
    a = TMAdapter(30, 4, 2, 3.0, device=device)
    X, _ = boolean_fixture(5)
    a.model.train()
    v = a.votes(X)
    assert a.model.training and v.abs().sum() == 0  # train-mode would count empty clauses as True


def test_conv_adapter_contract(device, smoke):
    tr, _ = smoke
    X = torch.as_tensor(tr.images[:300]) >= 128
    y = torch.as_tensor(tr.y[:300])
    torch.manual_seed(0)
    a = TMAdapter(None, 4, 4, 5.0, kind="conv", patch_size=7, stride=7, device=device)
    a.fit_epoch(X, y)
    assert torch.equal(a.class_scores(X[:50]).clamp(-a.T, a.T), a.votes(X[:50]))


def test_rejects_odd_clauses():
    import pytest
    with pytest.raises(ValueError):
        TMAdapter(30, 5, 2, 3.0)


def test_same_seed_same_state(device):  # D13: determinism is tested, not assumed
    X, y = boolean_fixture()
    hashes = []
    for _ in range(2):
        torch.manual_seed(123)
        a = TMAdapter(30, 20, 10, 3.0, device=device)
        a.fit_epoch(X, y)
        a.fit_epoch(X, y)
        hashes.append(a.state_hash())
    assert hashes[0] == hashes[1]
