"""Automaton accounting and memory matching (spec §12, D4/D5)."""

from __future__ import annotations


def automata(n_classes: int, M: int, n_features: int) -> int:
    return n_classes * M * 2 * n_features


def matched_M(target_automata: int, n_features: int, n_classes: int = 10, extra: int = 0) -> int:
    """Closest valid even clauses-per-class whose automaton count (+ ``extra``) matches the target."""
    per = n_classes * 2 * n_features
    m = (target_automata - extra) / per
    lo = max(2, int(m // 2) * 2)
    cands = [lo, lo + 2]
    return min(cands, key=lambda c: (abs(c * per + extra - target_automata), c))
