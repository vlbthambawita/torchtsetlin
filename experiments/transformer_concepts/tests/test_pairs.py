import torch
from experiments.transformer_concepts.code import tasks
from experiments.transformer_concepts.code.pairs import pair_feature_names, pair_tokens


def naive(x, V, window, abs_receiver, last_bit):
    N, L = x.shape
    out = []
    for n in range(N):
        cols = []
        for i in range(L):
            for j in range(L):
                f = [int(x[n, i] == a) for a in range(V)]
                for k in range(window):
                    t = int(x[n, j + k]) if j + k < L else V
                    f += [int(t == a) for a in range(V + 1)]
                f += [int(j - i > t) for t in range(-(L - 1), L - 1)]
                if abs_receiver:
                    f += [int(i > t) for t in range(L - 1)]
                if last_bit:
                    f += [int(i == L - 1)]
                cols.append(f)
        out.append(torch.tensor(cols, dtype=torch.bool).T)
    return torch.stack(out)


def test_pair_tokens_match_naive_reference():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 4, (3, 5), generator=g)
    for window in (1, 2):
        for ab in (False, True):
            for lb in (False, True):
                got = pair_tokens(x, 4, window, ab, lb)
                assert torch.equal(got, naive(x, 4, window, ab, lb))
                assert got.shape[1] == len(pair_feature_names(5, 4, window, ab, lb))


def test_hand_clause_solves_ordered_pair():
    # THEORY.md D: one clause q=A AND k0=B AND rel>0 decides "A before B" exactly
    g = torch.Generator().manual_seed(1)
    x, y = tasks.ordered_pair(500, 9, 6, g)
    cols = pair_tokens(x, 6)  # (N, F, P)
    names = pair_feature_names(9, 6)
    lit = [names.index("q=0"), names.index("k0=1"), names.index("rel>0")]
    pred = cols[:, lit, :].all(dim=1).any(dim=1).long()
    assert torch.equal(pred, y)


def test_hand_clauses_solve_associative_recall():
    # THEORY.md D: for value b, OR over keys a of (i=last AND q=a AND k0=a AND k1=b AND j<i)
    g = torch.Generator().manual_seed(2)
    nk, nv, L = 6, 3, 9
    x, y = tasks.associative_recall(400, L, nk, nv, g)
    V = nk + nv
    cols = pair_tokens(x, V, window=2, last_bit=True)
    names = pair_feature_names(L, V, window=2, last_bit=True)
    score = torch.zeros(400, nv, dtype=torch.long)
    for b in range(nv):
        for a in range(nk):
            lit = [names.index("i=last"), names.index(f"q={a}"), names.index(f"k0={a}"),
                   names.index(f"k1={nk + b}")]
            before = ~cols[:, names.index("rel>-1"), :]  # negated literal: j - i <= -1
            score[:, b] |= (cols[:, lit, :].all(dim=1) & before).any(dim=1).long()
    assert torch.equal(score.argmax(dim=1), y) and torch.all(score.sum(dim=1) == 1)
