"""Load hand-built clauses into real torchtsetlin models (the library is the object under test).

A clause is a list of literal indices: feature ``k`` is literal ``k``, its negation ``F + k``.
``load_clauses`` writes ``ta_state`` (included = ``n_states``, excluded = ``n_states - 1``) and
refreshes the include caches through the library's own ``_refresh_include``. Clause slots that
are not given stay empty, so in ``eval()`` mode they evaluate to False and vote nothing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def init_model(model, x_example) -> None:
    """Lazily initialise the model's state from one input batch (library ``_prepare``)."""
    model._prepare(x_example)


def literal(names: Sequence[str], name: str, negated: bool = False) -> int:
    k = list(names).index(name)
    return k + len(names) if negated else k


def load_clauses(model, clauses: Dict[Tuple[int, int], List[int]],
                 weights: Optional[Dict[Tuple[int, int], int]] = None) -> None:
    """``clauses[(class, slot)] = [literal indices]``; slot ``< ceil(n_clauses/2)`` is positive.

    Every other automaton is set to exclude. ``weights`` (weighted models) default to 1.
    """
    N = model.n_states
    st = torch.full_like(model.ta_state, N - 1)
    n = model.n_clauses
    for (c, slot), lits in clauses.items():
        row = c * n + slot
        for k in lits:
            st[row, k] = N
    with torch.no_grad():
        model.ta_state.copy_(st)
        if hasattr(model, "weights"):
            w = torch.ones_like(model.weights)
            for (c, slot), v in (weights or {}).items():
                w[c * n + slot] = v
            model.weights.copy_(w)
    model._refresh_include()
    model.eval()


def load_random_clauses(model, gen: torch.Generator, n_lits=(1, 4),
                        allowed: Optional[Sequence[int]] = None) -> List[List[int]]:
    """Fill every clause slot with a random clause of ``n_lits`` literals; returns them by row."""
    F2 = model.ta_state.shape[1]
    pool = torch.tensor(list(allowed) if allowed is not None else range(F2))
    rows = []
    N = model.n_states
    st = torch.full_like(model.ta_state, N - 1)
    for r in range(model.ta_state.shape[0]):
        k = int(torch.randint(n_lits[0], n_lits[1] + 1, (1,), generator=gen))
        lits = pool[torch.randperm(len(pool), generator=gen)[:k]].tolist()
        for q in lits:
            st[r, q] = N
        rows.append(lits)
    with torch.no_grad():
        model.ta_state.copy_(st)
    model._refresh_include()
    model.eval()
    return rows
