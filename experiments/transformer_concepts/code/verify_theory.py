"""Brute-force checks of the combinatorial propositions in THEORY.md (sections A1-A3).

For each position code c: {0..n-1} -> {0,1}^m a clause (conjunction of literals over the code
bits) selects the positions whose code satisfies it. A set S of positions is a *single clause*
iff S equals the truth set of some conjunction, and the minimum number of clauses whose union
is exactly S is the DNF cover number of S. Truth sets of conjunctions restricted to the code
image are the closures cl(U) = {p : c(p) agrees with c(u) on every bit where all u in U agree},
so we enumerate closures of subsets of S. Writes results/theory_checks.json.

Run: python -m experiments.transformer_concepts.code.verify_theory
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path


def onehot(p, n):
    return tuple(int(p == k) for k in range(n))


def therm(p, n):
    return tuple(int(p > k) for k in range(n - 1))


def binary(p, n):
    m = max(1, (n - 1).bit_length())
    return tuple((p >> k) & 1 for k in range(m))


def gray(p, n):
    m = max(1, (n - 1).bit_length())
    g = p ^ (p >> 1)
    return tuple((g >> k) & 1 for k in range(m))


CODES = {"onehot": onehot, "thermometer": therm, "binary": binary, "gray": gray}


def closure(points, codes):
    """Truth set (over all positions) of the tightest conjunction true on every point."""
    ref = codes[points[0]]
    agree = [k for k in range(len(ref)) if all(codes[q][k] == ref[k] for q in points)]
    return frozenset(p for p, c in enumerate(codes) if all(c[k] == ref[k] for k in agree)), len(agree)


def implicants(S, codes):
    """All truth sets T with T subset of S of conjunctions, with the min literal count found."""
    S = sorted(S)
    out = {}
    for r in range(1, len(S) + 1):
        for U in itertools.combinations(S, r):
            T, _ = closure(list(U), codes)
            if T <= frozenset(S):
                out.setdefault(T, None)
    return list(out)


def min_cover(S, sets):
    """Exact minimum number of sets (each a subset of S) whose union is S (iterative deepening)."""
    S = frozenset(S)
    sets = sorted(set(sets), key=len, reverse=True)
    for k in range(1, len(S) + 1):
        for combo in itertools.combinations(sets, k):
            if frozenset().union(*combo) == S:
                return k
    return None


def min_literals(S, codes):
    """Fewest literals of one conjunction whose truth set is exactly S (None if impossible).

    A literal (bit k = v) is admissible iff every point of S has bit k = v; the conjunction of
    a subset of admissible literals has truth set exactly S iff every position outside S
    violates at least one of them: a minimum hitting set."""
    S = frozenset(S)
    T, _ = closure(sorted(S), codes)
    if T != S:
        return None
    ref = codes[min(S)]
    adm = [k for k in range(len(ref)) if all(codes[q][k] == ref[k] for q in S)]
    outside = [p for p in range(len(codes)) if p not in S]
    if not outside:
        return 0
    for r in range(1, len(adm) + 1):
        for lits in itertools.combinations(adm, r):
            if all(any(codes[p][k] != ref[k] for k in lits) for p in outside):
                return r
    return None


def intervals(n):
    return [(a, b) for a in range(n) for b in range(a, n)]


def check_intervals(n):
    res = {}
    for name, code in CODES.items():
        codes = [code(p, n) for p in range(n)]
        worst_lit, worst_cov, single = 0, 0, 0
        for a, b in intervals(n):
            S = set(range(a, b + 1))
            ml = min_literals(S, codes)
            if ml is not None:
                single += 1
                worst_lit = max(worst_lit, ml)
            cov = min_cover(S, implicants(S, codes))
            worst_cov = max(worst_cov, cov)
        res[name] = {"bits": len(codes[0]), "intervals": len(intervals(n)),
                     "single_clause_intervals": single,
                     "max_literals_when_single": worst_lit, "max_clause_cover": worst_cov}
    return res


def pair_cover(n, relation, code):
    """Min number of clauses over [c(i), c(j)] whose union is exactly {(i, j): relation}."""
    codes = [code(i, n) + code(j, n) for i in range(n) for j in range(n)]
    idx = {(i, j): i * n + j for i in range(n) for j in range(n)}
    S = {idx[(i, j)] for i in range(n) for j in range(n) if relation(i, j)}
    # implicant truth sets are products A x B; enumerate closures of pairs of points (enough
    # for the lower bound to be tight is not guaranteed in general, so use all subsets of
    # size <= 2 plus singletons, then verify the cover is exact)
    sets = set()
    pts = sorted(S)
    for r in (1, 2):
        for U in itertools.combinations(pts, r):
            T, _ = closure(list(U), codes)
            if T <= frozenset(S):
                sets.add(T)
    return greedy_exact_lower(S, sets)


def greedy_exact_lower(S, sets):
    # exact minimum by iterative deepening over maximal sets only
    maximal = [s for s in sets if not any(s < t for t in sets)]
    return min_cover(S, maximal)


def kgrams(x, k):
    return {tuple(x[a : a + k]) for a in range(len(x) - k + 1)}


def needle_kgram_check(max_d=6):
    """THEORY B6.3: for kernel k <= d, two needle examples with different labels and equal
    k-gram sets. ``fixed`` is the proof as written (c at d+k, L = d+2k); ``original`` is the
    first draft (c at 2d+k+1, L = 2d+k+2), kept to show why it was wrong."""
    out = {"fixed_all_equal": True, "original_failures": []}
    for d in range(1, max_d + 1):
        for k in range(1, d + 1):
            for name, q, L in (("fixed", d + k, d + 2 * k), ("original", 2 * d + k + 1, 2 * d + k + 2)):
                x1 = ["f"] * L
                x1[0], x1[d], x1[q] = "M", "b", "c"
                x2 = list(x1)
                x2[d], x2[q] = "c", "b"
                equal = kgrams(x1, k) == kgrams(x2, k)
                if name == "fixed":
                    out["fixed_all_equal"] &= equal
                elif not equal:
                    out["original_failures"].append([d, k])
    return out


def main():
    out = {"intervals": {}, "pairs": {}, "needle_kgram": needle_kgram_check()}
    for n in (4, 6, 8, 10):
        out["intervals"][n] = check_intervals(n)
    for n in (4, 5, 6):
        out["pairs"][n] = {}
        for name, code in CODES.items():
            out["pairs"][n][name] = {
                "j_minus_i_eq_1": pair_cover(n, lambda i, j: j - i == 1, code),
                "i_lt_j": pair_cover(n, lambda i, j: i < j, code),
            }
    path = Path(__file__).resolve().parents[1] / "results" / "theory_checks.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
