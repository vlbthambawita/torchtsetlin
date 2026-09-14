import torch

from torchtsetlin import functional as F


def test_to_literals():
    x = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    lit = F.to_literals(x)
    assert lit.tolist() == [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]


def test_clause_outputs_and_empty_semantics():
    # features x0, x1 ; literals [x0, x1, ~x0, ~x1]
    include = torch.tensor(
        [
            [1, 0, 0, 0],  # x0
            [0, 0, 0, 1],  # NOT x1
            [1, 0, 1, 0],  # x0 AND NOT x0 -> never
            [0, 0, 0, 0],  # empty
        ],
        dtype=torch.float32,
    )
    lit = F.to_literals(torch.tensor([[1, 1], [0, 0]], dtype=torch.bool))
    out_train = F.clause_outputs(lit, include, empty_value=True)
    out_pred = F.clause_outputs(lit, include, include.sum(1), empty_value=False)
    assert out_train.tolist() == [[True, False, False, True], [False, True, False, True]]
    assert out_pred.tolist() == [[True, False, False, False], [False, True, False, False]]


def test_vote_sums_clamp():
    c = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    w = torch.tensor([[1, 0], [1, 0], [1, -1], [0, 5]], dtype=torch.int32)
    v = F.vote_sums(c, w, T=2)
    assert v.tolist() == [[2.0, -1.0]]


def test_feedback_probabilities():
    v = torch.tensor([-10.0, 0.0, 10.0])
    assert torch.allclose(F.feedback_probabilities(v, 10, True), torch.tensor([1.0, 0.5, 0.0]))
    assert torch.allclose(F.feedback_probabilities(v, 10, False), torch.tensor([0.0, 0.5, 1.0]))
    # clamps beyond T
    assert torch.allclose(F.feedback_probabilities(torch.tensor([50.0]), 10, True), torch.tensor([0.0]))


def test_sample_negative_classes_never_target():
    y = torch.randint(0, 5, (1000,))
    q = F.sample_negative_classes(y, 5)
    assert bool((q != y).all())
    votes = torch.randn(1000, 5)
    q2 = F.sample_negative_classes(y, 5, votes=votes, focused=True)
    assert bool((q2 != y).all())


def test_type_i_and_ii_counts():
    lit = F.to_literals(torch.tensor([[1, 0], [1, 1]], dtype=torch.bool))  # (2, 4)
    sel_fire = torch.tensor([[1.0, 0.0], [1.0, 1.0]])  # clause 0 fires on both, clause 1 on 2nd
    sel_nofire = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    n_true, n_false, n_ib = F.type_i_counts(lit, sel_fire, sel_nofire)
    assert n_true[0].tolist() == [2.0, 1.0, 0.0, 1.0]
    assert n_false[0].tolist() == [0.0, 1.0, 2.0, 1.0]
    assert n_ib.tolist() == [0.0, 1.0]
    n2 = F.type_ii_counts(lit, sel_fire)
    assert n2[1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_apply_feedback_deterministic_parts():
    N = 4
    state = torch.full((1, 4), N - 1, dtype=torch.int32)  # all excluded at the boundary
    z = lambda: torch.zeros(1, 4)  # noqa: E731  (counts are consumed in place -> fresh tensors)
    # Type II on literals 2,3 -> included (N), never beyond N
    n2 = torch.tensor([[0.0, 0.0, 5.0, 1.0]])
    F.apply_feedback(state, z(), z(), torch.zeros(1), n2, n_states=N, s=1e9, boost_true_positive=True)
    assert state.tolist() == [[N - 1, N - 1, N, N]]
    # Type Ia with boost: deterministic +1 per event, clamped at 2N-1
    n_true = torch.tensor([[100.0, 1.0, 0.0, 0.0]])
    F.apply_feedback(state, n_true, z(), torch.zeros(1), z(), n_states=N, s=1e9)
    assert state.tolist() == [[2 * N - 1, N, N, N]]
    # forgetting with s=1 (probability 1): Type Ib decrements everything
    F.apply_feedback(state, z(), z(), torch.tensor([2.0]), z(), n_states=N, s=1.0)
    assert state.tolist() == [[2 * N - 3, N - 2, N - 2, N - 2]]
    # never below 0
    F.apply_feedback(state, z(), z(), torch.tensor([100.0]), z(), n_states=N, s=1.0)
    assert state.tolist() == [[0, 0, 0, 0]]


def test_apply_feedback_literal_active_and_clause_size():
    N = 4
    state = torch.full((2, 4), N - 1, dtype=torch.int32)
    state[1] = 2 * N - 1  # clause 1 includes all 4 literals
    n_true = torch.ones(2, 4)
    active = torch.tensor([1.0, 0.0, 1.0, 1.0])
    F.apply_feedback(
        state, n_true, torch.zeros(2, 4), torch.zeros(2), torch.zeros(2, 4), n_states=N, s=1e9,
        max_included_literals=3, include_count=(state >= N).sum(1), literal_active=active,
    )
    assert state[0].tolist() == [N, N - 1, N, N]  # literal 1 frozen
    assert state[1].tolist() == [2 * N - 1] * 4  # oversized clause: no memorization (saturated anyway)
    # an oversized clause with s=1 forgets on every literal (its Type Ia events act as Ib)
    state = torch.full((1, 4), 2 * N - 1, dtype=torch.int32)
    F.apply_feedback(
        state, torch.ones(1, 4), torch.zeros(1, 4), torch.zeros(1), torch.zeros(1, 4), n_states=N,
        s=1.0, max_included_literals=3, include_count=torch.tensor([4]),
    )
    assert state.tolist() == [[2 * N - 2] * 4]


def test_confidence_and_proba():
    votes = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
    conf = F.confidence_from_votes(votes, 10)
    assert torch.allclose(conf, torch.tensor([1.0, 0.5]))
    p = F.predict_proba_from_votes(votes, 10)
    assert torch.allclose(p.sum(1), torch.ones(2))
