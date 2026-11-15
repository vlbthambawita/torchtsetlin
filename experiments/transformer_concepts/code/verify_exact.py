"""Part I of PROOF_PLAN.md: exact verification of the formal claims of THEORY.md.

Each check X1..X11 writes ``results/exact/<id>.json`` with ``pass`` and the evidence. Checks
that involve a model load hand-built clauses into real torchtsetlin models (``code/hand.py``)
and run them on CPU and, when available, CUDA (where prediction goes through the Triton
``conv_eval`` kernel).

Run: python -m experiments.transformer_concepts.code.verify_exact [--only X3,X8]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from experiments.transformer_concepts.code import tasks
from experiments.transformer_concepts.code.hand import (
    init_model,
    literal,
    load_clauses,
    load_random_clauses,
)
from experiments.transformer_concepts.code.pairs import (
    offset_bucket,
    pair_feature_names,
    pair_tokens,
)
from experiments.transformer_concepts.code.verify_theory import CODES, closure, min_literals

import torchtsetlin as tt

OUT = Path(__file__).resolve().parents[1] / "results" / "exact"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# ----------------------------------------------------------------------------- helpers
def exact_min_cover(universe, sets):
    """Exact minimum number of sets covering ``universe`` (branch and bound). Sets are frozensets."""
    universe = frozenset(universe)
    sets = [s & universe for s in sets]
    sets = [s for s in set(sets) if s]
    if not universe:
        return 0, []
    if frozenset().union(*sets) != universe:
        return None, None
    maxsize = max(len(s) for s in sets)
    by_elem = {e: [s for s in sets if e in s] for e in universe}
    best = [len(universe) + 1, None]

    def rec(uncovered, chosen):
        if not uncovered:
            if len(chosen) < best[0]:
                best[0], best[1] = len(chosen), list(chosen)
            return
        if len(chosen) + math.ceil(len(uncovered) / maxsize) >= best[0]:
            return
        e = min(uncovered, key=lambda u: len(by_elem[u]))
        for s in sorted(by_elem[e], key=lambda t: -len(t & uncovered)):
            chosen.append(s)
            rec(uncovered - s, chosen)
            chosen.pop()

    rec(universe, [])
    return best[0], best[1]


def prime_sets(cands):
    """Drop sets that are subsets of another set (keeps covers exact and minimal)."""
    uniq = sorted(set(cands), key=len, reverse=True)
    keep = []
    for s in uniq:
        if not any(s <= t for t in keep):
            keep.append(s)
    return keep


def save(cid: str, rec: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"id": cid, **rec}
    (OUT / f"{cid}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"[{cid}] pass={rec['pass']}  {rec.get('summary', '')}")
    return rec


def onehot(x: torch.Tensor, V: int) -> torch.Tensor:
    """``(N, L)`` ints -> ``(N, V, L)`` bool channels."""
    return torch.nn.functional.one_hot(x, V).bool().transpose(1, 2).contiguous()


def pair_model(n_classes, n_clauses, cols, device):
    m = tt.Conv1dTsetlinMachine(n_classes, n_clauses, T=max(1, n_clauses), s=10.0, kernel_size=1,
                                position_encoding=False).to(device)
    init_model(m, cols[:1].to(device))
    return m


# ----------------------------------------------------------------------------- X1, X1b
def x1_intervals():
    rows = {}
    ok = True
    for n in (4, 6, 8, 10, 12):
        rows[n] = {}
        for name, code in CODES.items():
            codes = [code(p, n) for p in range(n)]
            single, max_lit, max_cov, onehot_exact = 0, 0, 0, True
            for a in range(n):
                for b in range(a, n):
                    S = frozenset(range(a, b + 1))
                    ml = min_literals(S, codes)
                    if ml is not None:
                        single += 1
                        max_lit = max(max_lit, ml)
                        if name == "onehot" and len(S) >= 2 and ml != n - len(S):
                            onehot_exact = False
                    imps = set()
                    pts = sorted(S)
                    for r in range(1, len(pts) + 1):
                        for U in itertools.combinations(pts, r):
                            T, _ = closure(list(U), codes)
                            if T <= S:
                                imps.add(T)
                    k, _ = exact_min_cover(S, prime_sets(imps))
                    max_cov = max(max_cov, k)
            rows[n][name] = dict(bits=len(codes[0]), single_clause_intervals=single,
                                 intervals=n * (n + 1) // 2, max_literals_when_single=max_lit,
                                 max_clause_cover=max_cov)
            if name == "thermometer":
                ok &= max_lit <= 2 and single == n * (n + 1) // 2
            if name == "onehot":
                ok &= onehot_exact and single == n * (n + 1) // 2
    return save("X1", {"claims": ["A1", "A2", "A3"], "pass": bool(ok), "table": rows,
                       "summary": "thermometer <=2 literals; one-hot n-|S|; binary/Gray table"})


def x1b_binary_lower():
    res, ok = {}, True
    for m in (2, 3, 4):
        n = 2 ** m
        codes = [CODES["binary"](p, n) for p in range(n)]
        S = frozenset(range(1, n - 1))
        imps = set()
        pts = sorted(S)
        for r in range(1, len(pts) + 1):
            for U in itertools.combinations(pts, r):
                T, _ = closure(list(U), codes)
                if T <= S:
                    imps.add(T)
        k, _ = exact_min_cover(S, prime_sets(imps))
        res[n] = dict(m=m, min_cover=k, dyadic_upper=2 * m - 2)
        ok &= k == m
    return save("X1b", {"claims": ["A3b"], "pass": bool(ok), "table": res,
                        "summary": "binary [1, n-2] needs exactly log2 n clauses"})


# ----------------------------------------------------------------------------- X2
def t1_inputs(L, fillers=(2, 3)):
    xs, ys = [], []
    for pa in range(L):
        for pb in range(L):
            if pa == pb:
                continue
            rest = [q for q in range(L) if q not in (pa, pb)]
            for fill in itertools.product(fillers, repeat=len(rest)):
                x = [0] * L
                x[pa], x[pb] = 0, 1
                for q, f in zip(rest, fill):
                    x[q] = f
                xs.append(x)
                ys.append(int(pa < pb))
    return torch.tensor(xs), torch.tensor(ys)


def bayes_acc(feats, ys):
    groups: Dict = {}
    for f, y in zip(feats, ys):
        g = groups.setdefault(f, [0, 0])
        g[y] += 1
    return sum(max(g) for g in groups.values()) / len(ys)


def x2_set_invariance():
    res, ok = {}, True
    for L in (4, 5, 6):
        x, y = t1_inputs(L)
        xs, ys = x.tolist(), y.tolist()
        sym = bayes_acc([frozenset(r) for r in xs], ys)
        big = bayes_acc([frozenset(zip(r, r[1:])) for r in xs], ys)
        res[L] = dict(inputs=len(xs), symbol_set_bayes_acc=sym, bigram_set_bayes_acc=big)
        ok &= abs(sym - 0.5) < 1e-12 and big > 0.5
    return save("X2", {"claims": ["A5"], "pass": bool(ok), "table": res,
                       "summary": "any function of the symbol set: exactly 0.5; bigram set: above 0.5"})


# ----------------------------------------------------------------------------- X3
def x3_conv_vote():
    res, ok = {}, True
    V = 4
    for L in (4, 5, 6, 7, 9, 16):
        if L <= 7:
            x, y = t1_inputs(L)
        else:
            g = torch.Generator().manual_seed(L)
            x, y = tasks.ordered_pair(5000, L, V, g)
        X = onehot(x, V)
        for dev in DEVICES:
            m = tt.Conv1dTsetlinMachine(2, 4 * (L - 1), T=2 * (L - 1), s=10.0, kernel_size=1,
                                        position_encoding=True, input_shape=(V, L)).to(dev)
            init_model(m, X[:1].to(dev))
            Fn = m.n_features
            assert Fn == V + L - 1
            cl = {}
            for k in range(L - 1):
                pk = V + k  # [x > k]
                cl[(1, k)] = [0, pk + Fn]            # f_k: A and not x>k
                cl[(1, L - 1 + k)] = [1, pk]         # g_k: B and x>k
                cl[(0, k)] = [1, pk + Fn]            # h_k: B and not x>k
                cl[(0, L - 1 + k)] = [0, pk]         # e_k: A and x>k
            load_clauses(m, cl)
            votes = m(X.to(dev)).cpu()
            pred = votes.argmax(1)
            acc = (pred == y).float().mean().item()
            diff_ok = bool(((votes[:, 1] - votes[:, 0]) != 0).all())
            res[f"L{L}_{dev}"] = dict(inputs=len(y), acc=acc, no_ties=diff_ok)
            ok &= acc == 1.0 and diff_ok
    return save("X3", {"claims": ["A6"], "pass": bool(ok), "table": res,
                       "summary": "library conv TM with its own thermometer PE + 2(L-1) clauses/class solves T1"})


# ----------------------------------------------------------------------------- X4
def x4_relative():
    res, ok = {}, True
    for n in (4, 8, 16, 32):
        for name, code in CODES.items():
            pc = {(i, j): code(i, n) + code(j, n) for i in range(n) for j in range(n)}

            def tight_contains(p1, p2, q, pc=pc):
                a, b, c = pc[p1], pc[p2], pc[q]
                return all(c[k] == a[k] for k in range(len(a)) if a[k] == b[k])

            sep_d, sep_lt = True, True
            for d in (0, 1, max(1, n // 2)):
                pts = [(i, i + d) for i in range(n - d)]
                for (i, j), (i2, j2) in itertools.combinations(pts, 2):
                    if not tight_contains((i, j), (i2, j2), (i, j2)):
                        sep_d = False
            pts = [(k, k + 1) for k in range(n - 1)]
            for (i, j), (i2, j2) in itertools.combinations(pts, 2):
                if not tight_contains((i, j), (i2, j2), (i2, j)):
                    sep_lt = False
            injective = len({code(p, n) for p in range(n)}) == n
            res[f"{name}_n{n}"] = dict(fooling_separated_eq_d=sep_d, fooling_separated_lt=sep_lt,
                                       injective=injective)
            ok &= sep_d and sep_lt
    # explicit covers with the thermometer code
    cov_ok = True
    for n in (4, 8, 16):
        lt = {(i, j) for i in range(n) for j in range(n) if i < j}
        union = set()
        for k in range(n - 1):
            union |= {(i, j) for i in range(n) for j in range(n) if i <= k < j}
        cov_ok &= union == lt
    res["thermometer_cover_lt_exact"] = cov_ok
    ok &= cov_ok
    return save("X4", {"claims": ["A7"], "pass": bool(ok), "table": res,
                       "summary": "no conjunction over [c(i), c(j)] covers two fooling points, any code, n<=32"})


# ----------------------------------------------------------------------------- X5
def eval_literal(name: str, x: np.ndarray, i: np.ndarray, j: np.ndarray, L: int, V: int) -> np.ndarray:
    """Symbolic value of a pair literal for all (example, pair): x (N, L); i, j (P,)."""
    if name.startswith("q="):
        return x[:, i] == int(name[2:])
    if name.startswith("k") and "=" in name and name[1].isdigit():
        t, a = name[1:].split("=")
        jt = j + int(t)
        tok = np.where(jt[None, :] < L, x[:, np.minimum(jt, L - 1)], V)
        return tok == (V if a == "END" else int(a))
    if name.startswith("rel>"):
        return np.broadcast_to((j - i) > int(name[4:]), (x.shape[0], len(i)))
    if name.startswith("relb>"):
        b = offset_bucket(torch.tensor(j - i)).numpy()
        return np.broadcast_to(b > int(name[5:]), (x.shape[0], len(i)))
    if name.startswith("i>"):
        return np.broadcast_to(i > int(name[2:]), (x.shape[0], len(i)))
    if name.startswith("j>"):
        return np.broadcast_to(j > int(name[2:]), (x.shape[0], len(i)))
    if name == "i=last":
        return np.broadcast_to(i == L - 1, (x.shape[0], len(i)))
    raise ValueError(name)


def symbolic_clauses(rows: List[List[int]], names: List[str], x: np.ndarray, L: int, V: int,
                     self_only: bool = False) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    if self_only:
        keep = ii == jj
        ii, jj = ii[keep], jj[keep]
    Fn = len(names)
    cache = {}
    out = np.zeros((x.shape[0], len(rows)), dtype=bool)
    for r, lits in enumerate(rows):
        acc = np.ones((x.shape[0], len(ii)), dtype=bool)
        for k in lits:
            base = names[k % Fn]
            if base not in cache:
                cache[base] = eval_literal(base, x, ii, jj, L, V)
            v = cache[base]
            acc &= ~v if k >= Fn else v
        out[:, r] = acc.any(axis=1)
    return out


def x5_pair_factorisation():
    res, ok = {}, True
    V, w = 4, 2
    for rel in ("therm", "bucket", "abs", "none"):
        L = 12 if rel == "bucket" else 6  # at L <= 6 the buckets are exact offsets
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, V, (150, L), generator=g)
        names = pair_feature_names(L, V, w, False, True, rel)
        cols = pair_tokens(x, V, w, False, True, rel=rel)
        for dev in DEVICES:
            m = pair_model(2, 400, cols, dev)
            rows = load_random_clauses(m, torch.Generator().manual_seed(7), n_lits=(1, 4))
            got = m.evaluate_clauses(cols.to(dev)).cpu().numpy()
            want = symbolic_clauses(rows, names, x.numpy(), L, V)
            agree = bool((got == want).all())
            res[f"{rel}_{dev}"] = dict(clauses=len(rows), examples=x.shape[0], agree=agree,
                                       fraction_true=float(want.mean()))
            ok &= agree
    return save("X5", {"claims": ["B1"], "pass": bool(ok), "table": res,
                       "summary": "library pair TM = exists(i,j) Q and K and R and A, symbolic reference"})


# ----------------------------------------------------------------------------- X6
def x6_popcount():
    # Q1=[x_i=a], K1=[x_j=b], Q2=[x_i=c], K2=[x_j=d]; sequence a b c d (a,b,c,d = 0..3)
    x = [0, 1, 2, 3]
    L = 4
    pairs = [(i, j) for i in range(L) for j in range(L)]
    r1 = [(x[i] == 0) and (x[j] == 1) for i, j in pairs]
    r2 = [(x[i] == 2) and (x[j] == 3) for i, j in pairs]
    vote = int(any(r1)) + int(any(r2))
    popcount_best = max(int(a) + int(b) for a, b in zip(r1, r2))
    ok = vote == 2 and popcount_best == 1
    return save("X6", {"claims": ["B3-scope"], "pass": bool(ok),
                       "vote_over_pair_clauses": vote, "best_pair_popcount": popcount_best,
                       "summary": "TM vote (2) != popcount attention score (1) on 'a b c d'"})


# ----------------------------------------------------------------------------- X7
def x7_self_only_equals_conv():
    res, ok = {}, True
    L, V = 7, 4
    g = torch.Generator().manual_seed(11)
    x = torch.randint(0, V, (200, L), generator=g)
    for w in (1, 2, 3):
        names = pair_feature_names(L, V, w, False, False, "therm")
        Fp = len(names)
        cols = pair_tokens(x, V, w, self_only=True)
        xpad = torch.cat([x, torch.full((x.shape[0], w - 1), V, dtype=torch.long)], dim=1)
        conv_in = onehot(xpad, V + 1)  # (N, V+1, L+w-1)
        for dev in DEVICES:
            # direction 1: random conv clauses -> pair clauses
            cm = tt.Conv1dTsetlinMachine(2, 200, T=200, s=10.0, kernel_size=w,
                                         position_encoding=False).to(dev)
            init_model(cm, conv_in[:1].to(dev))
            Fc = cm.n_features
            rows = load_random_clauses(cm, torch.Generator().manual_seed(w), n_lits=(1, 3))
            pm = pair_model(2, 200, cols, dev)
            mapped = {}
            for r, lits in enumerate(rows):
                new = []
                for k in lits:
                    neg, f = k >= Fc, k % Fc
                    z, t = divmod(f, w)  # unfold order: channel-major, then kernel offset
                    sym = "END" if z == V else str(z)
                    new.append(literal(names, f"k{t}={sym}", neg))
                mapped[(r // 200, r % 200)] = new
            load_clauses(pm, mapped)
            a = cm.evaluate_clauses(conv_in.to(dev)).cpu()
            b = pm.evaluate_clauses(cols.to(dev)).cpu()
            d1 = bool(torch.equal(a, b))
            # direction 2: random pair clauses (q, k, rel literals) -> conv clauses
            prow = load_random_clauses(pm, torch.Generator().manual_seed(100 + w), n_lits=(1, 3))
            b2 = pm.evaluate_clauses(cols.to(dev)).cpu()
            mapped2, const_false = {}, []
            for r, lits in enumerate(prow):
                new, dead = [], False
                for k in lits:
                    neg, f = k >= Fp, k % Fp
                    nm = names[f]
                    if nm.startswith("rel>"):
                        val = 0 > int(nm[4:])  # j - i = 0 on self pairs
                        if val == neg:
                            dead = True
                        continue
                    if nm.startswith("q="):
                        z, t = int(nm[2:]), 0
                    else:
                        t_s, a_s = nm[1:].split("=")
                        z, t = (V if a_s == "END" else int(a_s)), int(t_s)
                    new.append(z * w + t + (Fc if neg else 0))
                const_false.append(dead)
                mapped2[(r // 200, r % 200)] = new if not dead else []
            load_clauses(cm, mapped2)
            a2 = cm.evaluate_clauses(conv_in.to(dev)).cpu()
            dead_t = torch.tensor(const_false)
            # empty mapped clauses (all literals constant-true) are True on every input
            empty_t = torch.tensor([len(mapped2[(r // 200, r % 200)]) == 0 and not const_false[r]
                                    for r in range(len(prow))])
            expect = a2.clone()
            expect[:, dead_t] = False
            expect[:, empty_t] = True
            d2 = bool(torch.equal(expect, b2))
            res[f"w{w}_{dev}"] = dict(conv_to_pair=d1, pair_to_conv=d2,
                                      constant_false_clauses=int(dead_t.sum()))
            ok &= d1 and d2
    return save("X7", {"claims": ["B4.1"], "pass": bool(ok), "table": res,
                       "summary": "self-only pair TM == conv TM (kernel w) on END-padded sequence, both directions"})


# ----------------------------------------------------------------------------- X8
def all_needle(L, d, V, n_classes):
    xs, ys = [], []
    others = list(range(1, V))
    for p in range(L - d):
        rest = [q for q in range(L) if q not in (p, p + d)]
        for c in range(n_classes):
            for fill in itertools.product(others, repeat=len(rest)):
                x = [0] * L
                x[p], x[p + d] = 0, c + 1
                for q, f in zip(rest, fill):
                    x[q] = f
                xs.append(x)
                ys.append(c)
    return torch.tensor(xs), torch.tensor(ys)


def all_recall(n, nk, nv):
    xs, ys = [], []
    for keys in itertools.permutations(range(nk), n):
        for vals in itertools.product(range(nv), repeat=n):
            for s in range(n):
                x = []
                for k, v in zip(keys, vals):
                    x += [k, nk + v]
                x.append(keys[s])
                xs.append(x)
                ys.append(vals[s])
    return torch.tensor(xs), torch.tensor(ys)


def all_repeat(L, V):
    xs = torch.tensor(list(itertools.product(range(V), repeat=L)))
    ys = torch.tensor([int(len(set(r)) < L) for r in xs.tolist()])
    return xs, ys


def run_pair_hand(name, x, y, V, n_classes, build, kw):
    out = {}
    L = x.shape[1]
    names = pair_feature_names(L, V, kw.get("window", 1), False, kw.get("last_bit", False))
    cols = pair_tokens(x, V, **kw)
    clauses, n_clauses = build(names)
    ok = True
    for dev in DEVICES:
        m = pair_model(n_classes, n_clauses, cols, dev)
        load_clauses(m, clauses)
        votes = m(cols.to(dev)).cpu()
        pred = votes.argmax(1)
        acc = (pred == y).float().mean().item()
        out[dev] = dict(acc=acc, inputs=len(y))
        ok &= acc == 1.0
    return out, ok


def x8_pair_constructions():
    res, ok = {}, True
    # B5: T1
    for L, exhaustive in ((5, True), (6, True), (16, False), (32, False)):
        V = 4
        if exhaustive:
            x, y = t1_inputs(L)
        else:
            x, y = tasks.ordered_pair(10000, L, V, torch.Generator().manual_seed(L))

        def b5(names):
            q, k, r = literal(names, "q=0"), literal(names, "k0=1"), literal(names, "rel>0")
            rn = literal(names, "rel>0", True)
            return {(1, 0): [q, k, r], (0, 0): [q, k, rn]}, 2
        o, p = run_pair_hand("B5", x, y, V, 2, b5, dict(window=1))
        res[f"B5_T1_L{L}"] = o
        ok &= p
    # B6.1: T4
    for L, d, exhaustive in ((6, 2, True), (7, 3, True), (16, 4, False), (32, 8, False)):
        V, nc = (4, 2) if exhaustive else (16, 4)
        if exhaustive:
            x, y = all_needle(L, d, V, nc)
        else:
            x, y = tasks.needle(10000, L, V, d, nc, torch.Generator().manual_seed(L))

        def b6(names, d=d, nc=nc):
            cl = {}
            for c in range(nc):
                cl[(c, 0)] = [literal(names, "q=0"), literal(names, f"k0={c + 1}"),
                              literal(names, f"rel>{d - 1}"), literal(names, f"rel>{d}", True)]
            return cl, 2
        o, p = run_pair_hand("B6", x, y, V, nc, b6, dict(window=1))
        res[f"B6.1_T4_L{L}_d{d}"] = o
        ok &= p
    # B7.1: T2
    for n, nk, nv, exhaustive in ((3, 4, 2, True), (8, 16, 4, False), (16, 16, 4, False)):
        if exhaustive:
            x, y = all_recall(n, nk, nv)
        else:
            x, y = tasks.associative_recall(10000, 2 * n + 1, nk, nv, torch.Generator().manual_seed(n))
        V = nk + nv

        def b7(names, nk=nk, nv=nv):
            cl = {}
            for b in range(nv):
                for a in range(nk):
                    cl[(b, a)] = [literal(names, "i=last"), literal(names, f"q={a}"),
                                  literal(names, f"k0={a}"), literal(names, f"k1={nk + b}"),
                                  literal(names, "rel>-1", True)]
            return cl, 2 * nk
        o, p = run_pair_hand("B7", x, y, V, nv, b7, dict(window=2, last_bit=True))
        res[f"B7.1_T2_n{n}"] = o
        ok &= p
    # B8: T3
    for L, V, exhaustive in ((4, 5, True), (5, 5, True), (16, 32, False), (32, 48, False)):
        if exhaustive:
            x, y = all_repeat(L, V)
        else:
            x, y = tasks.repeat_detect(10000, L, V, torch.Generator().manual_seed(L))

        def b8(names, V=V):
            return {(1, a): [literal(names, f"q={a}"), literal(names, f"k0={a}"), literal(names, "rel>0")]
                    for a in range(V)}, 2 * V
        o, p = run_pair_hand("B8", x, y, V, 2, b8, dict(window=1))
        res[f"B8_T3_L{L}"] = o
        ok &= p
    return save("X8", {"claims": ["B5", "B6.1", "B7.1", "B8"], "pass": bool(ok), "table": res,
                       "summary": "hand pair clauses in the library pair TM: accuracy 1.0 everywhere"})


# ----------------------------------------------------------------------------- X9
def x9_needle_conv():
    res, ok = {}, True
    V = 4  # M=0, b=1, c=2, filler=3
    for d in range(2, 7):
        for k in range(1, d + 1):
            L = d + 2 * k
            x1 = [3] * L
            x1[0], x1[d], x1[d + k] = 0, 1, 2
            x2 = list(x1)
            x2[d], x2[d + k] = 2, 1
            X = onehot(torch.tensor([x1, x2]), V)
            for dev in DEVICES:
                m = tt.Conv1dTsetlinMachine(2, 500, T=500, s=10.0, kernel_size=k,
                                            position_encoding=False).to(dev)
                init_model(m, X[:1].to(dev))
                load_random_clauses(m, torch.Generator().manual_seed(d * 10 + k), n_lits=(1, 4))
                out = m.evaluate_clauses(X.to(dev)).cpu()
                same = bool(torch.equal(out[0], out[1]))
                res[f"d{d}_k{k}_{dev}"] = dict(L=L, identical_outputs=same,
                                               clauses_true=int(out[0].sum()))
                ok &= same
    return save("X9", {"claims": ["B6.3"], "pass": bool(ok), "table": res,
                       "summary": "library conv TM (k<=d) gives identical clause outputs on the two differently labelled needles"})


# ----------------------------------------------------------------------------- X10
def min_positive_dnf(x: np.ndarray, pos: np.ndarray, chunk: int = 20000):
    """Exact minimum positive DNF for ``pos`` over the one-hot domain ``x`` (N, L).

    Implicants are products of per-position symbol sets (restricted to the symbols that occur at
    that position in the domain). Returns (min clauses, number of prime implicants)."""
    N, L = x.shape
    alph = [sorted(set(x[:, q].tolist())) for q in range(L)]
    sub = []
    for q in range(L):
        a = alph[q]
        masks = []
        for r in range(1, len(a) + 1):
            for S in itertools.combinations(a, r):
                masks.append(sum(1 << s for s in S))
        sub.append(masks)
    sizes = [len(s) for s in sub]
    total = int(np.prod(sizes))
    xs = x.astype(np.int64)
    pos_idx = np.nonzero(pos)[0]
    neg = ~pos
    covers = set()
    sub_arr = [np.array(s, dtype=np.int64) for s in sub]
    for start in range(0, total, chunk):
        idx = np.arange(start, min(total, start + chunk))
        digits = np.unravel_index(idx, sizes)
        contain = np.ones((len(idx), N), dtype=bool)
        for q in range(L):
            m = sub_arr[q][digits[q]]
            contain &= ((m[:, None] >> xs[None, :, q]) & 1).astype(bool)
        imp = ~(contain & neg[None, :]).any(axis=1)
        for row in contain[imp][:, pos_idx]:
            if row.any():
                covers.add(frozenset(np.nonzero(row)[0].tolist()))
    primes = prime_sets(covers)
    k, _ = exact_min_cover(range(len(pos_idx)), primes)
    return k, len(primes), total


def x10_flat_lower_bounds():
    res, ok = {}, True
    t0 = time.time()
    # B6.2 needle, V=3 (M=0, b=1, phi=2), class b
    for L, d in ((4, 1), (5, 1), (5, 2), (6, 2), (7, 3)):
        xs = []
        for p in range(L - d):
            rest = [q for q in range(L) if q != p]
            for fill in itertools.product((1, 2), repeat=len(rest)):
                x = [0] * L
                for q, f in zip(rest, fill):
                    x[q] = f
                xs.append(x)
        x = np.array(xs)
        p_of = (x == 0).argmax(1)
        pos = x[np.arange(len(x)), p_of + d] == 1
        k, primes, total = min_positive_dnf(x, pos)
        res[f"B6.2_needle_L{L}_d{d}"] = dict(min_clauses=k, bound=L - d, primes=primes, products=total)
        ok &= k is not None and k >= L - d
    # B7.2 recall, class value 0
    for n, nk, nv in ((2, 3, 2), (2, 4, 2), (2, 5, 2), (2, 6, 2), (3, 3, 2), (3, 4, 2)):
        xt, yt = all_recall(n, nk, nv)
        x, pos = xt.numpy(), (yt == 0).numpy()
        k, primes, total = min_positive_dnf(x, pos)
        res[f"B7.2_recall_n{n}_keys{nk}"] = dict(
            min_clauses=k, bound=n, n_times_keys=n * nk, n_times_min_keys_2pow_n=n * min(nk, 2 ** n),
            pair_tm_clauses=nk, primes=primes, products=total)
        ok &= k is not None and k >= n
    # D.1 flat T1 (A=0, B=1, phi=2), class "A before B"
    for L in (4, 5, 6, 7):
        xt, yt = t1_inputs(L, fillers=(2,))
        x, pos = xt.numpy(), (yt == 1).numpy()
        k, primes, total = min_positive_dnf(x, pos)
        res[f"D1_T1_L{L}"] = dict(min_clauses=k, lower_bound=math.ceil((L - 1) / 2),
                                  construction=L - 1, primes=primes, products=total)
        ok &= k is not None and math.ceil((L - 1) / 2) <= k <= L - 1
    return save("X10", {"claims": ["B6.2", "B7.2", "D.1"], "pass": bool(ok), "table": res,
                        "seconds": round(time.time() - t0, 1),
                        "summary": "exact minimum positive DNFs >= the proved lower bounds"})


# ----------------------------------------------------------------------------- X11
def x11_polarity():
    res, ok = {}, True
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.bool)
    for dev in DEVICES:
        for wp, wn in ((1, 1), (2, 3), (3, 3), (5, 9)):
            m = tt.TsetlinMachine(2, 2, 2, T=10, s=5.0, weighted=True).to(dev)
            init_model(m, X[:1].to(dev))
            load_clauses(m, {(1, 0): [0], (1, 1): [1]}, weights={(1, 0): wp, (1, 1): wn})
            v = m(X.to(dev)).cpu()[:, 1]
            want = torch.tensor([False, False, True, False])
            got = v > 0
            same = bool(torch.equal(got, want))
            res[f"w+{wp}_w-{wn}_{dev}"] = dict(votes=v.tolist(), match=same)
            ok &= same
    return save("X11", {"claims": ["B9"], "pass": bool(ok), "table": res,
                        "summary": "v>0 iff C+ and not C-, whenever w- >= w+"})


# ----------------------------------------------------------------------------- X12
def x12_flat_t1_construction():
    """THEORY D.1: ceil((L-1)/2) positive flat clauses solve T1 exactly (matching the fooling-set
    lower bound). Clause t fixes the free position f = 2t-1: positions < f hold no B, positions
    > f hold no A. Loaded into the library flat TM; A=0, B=1, fillers 2..V-1."""
    res, ok = {}, True
    V = 4
    for L in (4, 5, 6, 7, 9, 16, 33):
        if L <= 7:
            x, y = t1_inputs(L)
        else:
            x, y = tasks.ordered_pair(10000, L, V, torch.Generator().manual_seed(L))
        X = torch.nn.functional.one_hot(x, V).bool().reshape(len(x), -1)
        k = math.ceil((L - 1) / 2)
        for dev in DEVICES:
            m = tt.TsetlinMachine(L * V, 2, 2 * k, T=k, s=10.0).to(dev)
            init_model(m, X[:1].to(dev))
            Fn = L * V
            cl = {}
            for t in range(1, k + 1):
                f = 2 * t - 1
                lits = [q * V + 1 + Fn for q in range(f)] + [q * V + 0 + Fn for q in range(f + 1, L)]
                cl[(1, t - 1)] = lits  # class 1: A before B
            load_clauses(m, cl)
            fired = m.evaluate_clauses(X.to(dev)).cpu()[:, m.n_clauses: m.n_clauses + k].any(1)
            acc = (fired.long() == y).float().mean().item()
            res[f"L{L}_{dev}"] = dict(clauses=k, inputs=len(y), acc=acc)
            ok &= acc == 1.0
    return save("X12", {"claims": ["D.1"], "pass": bool(ok), "table": res,
                        "summary": "flat TM: ceil((L-1)/2) clauses decide T1 exactly (tight with the lower bound)"})


CHECKS = {
    "X1": x1_intervals, "X1b": x1b_binary_lower, "X2": x2_set_invariance, "X3": x3_conv_vote,
    "X4": x4_relative, "X5": x5_pair_factorisation, "X6": x6_popcount,
    "X7": x7_self_only_equals_conv, "X8": x8_pair_constructions, "X9": x9_needle_conv,
    "X10": x10_flat_lower_bounds, "X11": x11_polarity,
    "X12": x12_flat_t1_construction,
}


def main(only: Sequence[str] = ()) -> None:
    for cid, fn in CHECKS.items():
        if only and cid not in only:
            continue
        t0 = time.time()
        fn()
        print(f"    ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    main([s for s in a.only.split(",") if s])
