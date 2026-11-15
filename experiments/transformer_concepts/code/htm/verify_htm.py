"""Exact checks X13-X16 for THEORY_HTM.md (PLAN_HTM.md Part 0). CPU only.

    python -m experiments.transformer_concepts.code.htm.verify_htm [X13 X14 X15 X16]

Writes results/htm/exact/X*.json; each record has ``pass`` and the numbers behind it.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "htm" / "exact"


def save(name, rec):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(check=name, time=time.strftime("%Y-%m-%dT%H:%M:%S"), **rec)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(name, "PASS" if rec["pass"] else "FAIL", {k: v for k, v in rec.items() if k not in ("instances", "check", "time")})
    return rec


# ----------------------------------------------------------------------------- X13
def t3_domain(L, V):
    """All T3 sequences: negatives all distinct, positives exactly one symbol twice."""
    neg, pos, rep = [], [], []
    for x in itertools.product(range(V), repeat=L):
        counts = np.bincount(x, minlength=V)
        if counts.max() == 1:
            neg.append(x)
        elif counts.max() == 2 and (counts == 2).sum() == 1 and (counts > 2).sum() == 0:
            pos.append(x)
            rep.append(int(np.argmax(counts)))
    return np.array(neg), np.array(pos), np.array(rep)


def onehot_sets(n_sym):
    """Truth sets of conjunctions of one-hot literals over n_sym symbols: {a} or (all minus N)."""
    sets = {1 << a for a in range(n_sym)}
    full = (1 << n_sym) - 1
    for r in range(n_sym):  # complements of N, |N| < n_sym (N = all gives the empty set)
        for N in itertools.combinations(range(n_sym), r):
            sets.add(full & ~sum(1 << a for a in N))
    return np.array(sorted(sets), dtype=np.int64)


def column_tokens(X, L, V, w):
    """(n, L*L, 1+w) tokens read by pair column (i, j): x_i, x_j .. x_{j+w-1} (END = V)."""
    cols = []
    for i in range(L):
        for j in range(L):
            t = [X[:, i]] + [X[:, j + k] if j + k < L else np.full(len(X), V) for k in range(w)]
            cols.append(np.stack(t, 1))
    return np.stack(cols, 1)


def x13_instance(L, V, w):
    neg, pos, rep = t3_domain(L, V)
    tn, tp = column_tokens(neg, L, V, w), column_tokens(pos, L, V, w)
    qs = onehot_sets(V)
    ks = onehot_sets(V + 1)  # sender tokens may be END
    blocks = [qs] + [ks] * w
    P = L * L
    n_valid, n_cover, worst, cov, example = 0, 0, 0, [], None
    # The best position set for a token product is every column on which no negative matches
    # (THEORY_HTM X13: maximal clauses suffice for a minimum cover; any position code is a subset).
    for combo in itertools.product(*[range(len(b)) for b in blocks]):
        masks = [blocks[b][c] for b, c in enumerate(combo)]

        def match(T):
            m = np.ones(T.shape[:2], dtype=bool)
            for b, mk in enumerate(masks):
                m &= ((mk >> T[:, :, b]) & 1).astype(bool)
            return m  # (n, P)

        neg_hit = match(tn).any(0)  # (P,) columns where some negative matches
        good = ~neg_hit
        if not good.any():
            continue
        covered = (match(tp) & good[None]).any(1)
        if not covered.any():
            continue
        n_valid += 1
        syms = np.unique(rep[covered])
        if len(syms) > 1 and example is None:
            example = dict(sets_bitmask=[int(m) for m in masks], symbols_covered=syms.tolist(),
                           positives=[list(map(int, p)) for p in pos[covered][:4]])
        worst = max(worst, len(syms))
        cov.append(covered)
        n_cover += 1
    # construction: C_a = [q=a] & [k0=a] & [j-i>0]
    cover = np.zeros(len(pos), bool)
    for a in range(V):
        ii, jj = np.meshgrid(range(L), range(L), indexing="ij")
        ok = (jj > ii).reshape(-1)
        mp = (tp[:, :, 0] == a) & (tp[:, :, 1] == a) & ok[None]
        mn = (tn[:, :, 0] == a) & (tn[:, :, 1] == a) & ok[None]
        assert not mn.any()
        cover |= mp.any(1)
    # exact minimum cover over all maximal valid clauses (integer program)
    from scipy.optimize import Bounds, LinearConstraint, milp
    A = np.array(cov, dtype=float).T
    r = milp(np.ones(A.shape[1]), constraints=LinearConstraint(A, lb=1), integrality=np.ones(A.shape[1]),
             bounds=Bounds(0, 1))
    minimum = int(round(r.fun))
    return dict(L=L, V=V, w=w, n_neg=len(neg), n_pos=len(pos), n_valid_clauses=n_valid,
                max_repeat_symbols_per_clause=int(worst), one_symbol_lemma=bool(worst == 1),
                example_two_symbol_clause=example, construction_covers_all=bool(cover.all()),
                minimum=minimum, pass_=bool(minimum == V and cover.all() and (w > 1 or worst == 1)))


def equality_threshold_lp(V):
    """Is [q == k] (one-hot q, k) a linear threshold of literals? LP with margin 1."""
    from scipy.optimize import linprog
    # variables: w (V), u (V), theta. Literals of one-hot = affine in the one-hot, so this is general.
    A, b = [], []
    for a in range(V):
        for c in range(V):
            row = np.zeros(2 * V + 1)
            if a == c:  # w_a + u_a - theta >= 1  ->  -w_a - u_a + theta <= -1
                row[a] -= 1
                row[V + c] -= 1
                row[-1] += 1
                A.append(row)
                b.append(-1)
            else:  # w_a + u_c - theta <= -1
                row[a] += 1
                row[V + c] += 1
                row[-1] -= 1
                A.append(row)
                b.append(-1)
    r = linprog(np.zeros(2 * V + 1), A_ub=np.array(A), b_ub=np.array(b), bounds=[(None, None)] * (2 * V + 1))
    return dict(V=V, lp_status=int(r.status), feasible=bool(r.status == 0))


def X13():
    inst = []
    for (L, V, w) in [(3, 3, 1), (3, 4, 1), (4, 4, 1), (4, 5, 1), (3, 3, 2), (3, 4, 2), (4, 4, 2), (4, 5, 2)]:
        t0 = time.time()
        r = x13_instance(L, V, w)
        r["seconds"] = round(time.time() - t0, 1)
        print(" ", r, flush=True)
        inst.append(r)
    lp = [equality_threshold_lp(V) for V in range(2, 7)]
    ok = all(r["pass_"] for r in inst) and not any(r["feasible"] for r in lp)
    return save("X13", dict(claim="THEORY_HTM H3.1: T3 needs exactly V pair clauses (w=1: proof lemma, every valid clause "
                                  "covers one repeated symbol; w=2: exact minimum only, Conjecture H3.1b); no threshold "
                                  "of literals computes token equality", instances=inst, equality_lp=lp, **{"pass": ok}))


# ----------------------------------------------------------------------------- X14
def X14():
    inst = []
    ok = True
    for k in range(1, 11):
        pts = np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.int8)  # (2^k, k)
        wt = pts.sum(1)
        # every term: per variable 0 = absent, 1 = positive, 2 = negated
        for delta in range(k):
            f = wt >= k - delta
            minimal = np.nonzero(wt == k - delta)[0]
            worst = 0
            n_impl = 0
            for term in itertools.product([0, 1, 2], repeat=k):
                term = np.array(term)
                t = np.ones(len(pts), bool)
                t &= np.all((term != 1) | (pts == 1), 1)
                t &= np.all((term != 2) | (pts == 0), 1)
                if not t.any() or (t & ~f).any():
                    continue  # empty or not an implicant
                n_impl += 1
                worst = max(worst, int(t[minimal].sum()))
            good = worst == 1  # every implicant covers at most one minimal positive point
            ok &= good
            inst.append(dict(k=k, delta=delta, lower_bound=math.comb(k, delta) if good else None,
                             construction=math.comb(k, delta), n_implicants=n_impl, pass_=good))
        if k >= 8:  # 3^k terms x 2^k points gets slow; k <= 8 for every delta, then stop
            break
    # complements of intervals on the thermometer offset code
    therm = []
    for L in range(2, 6):
        d = np.arange(-(L - 1), L)
        ts = np.arange(-(L - 1), L - 1)
        bits = (d[:, None] > ts[None]).astype(np.int8)  # (2L-1, 2L-2)
        strict_sets = set()
        for term in itertools.product([0, 1, 2], repeat=bits.shape[1]):
            term = np.array(term)
            t = np.all((term != 1) | (bits == 1), 1) & np.all((term != 2) | (bits == 0), 1)
            strict_sets.add(tuple(t))
        n_comp, n_strict_hits = 0, 0
        for lo in range(len(d)):
            for hi in range(lo, len(d)):
                if lo == 0 and hi == len(d) - 1:
                    continue
                comp = np.ones(len(d), bool)
                comp[lo:hi + 1] = False
                if comp.all() or not comp.any() or comp[0] is False or not (comp[0] and comp[-1]):
                    continue  # only two-ray complements
                n_comp += 1
                n_strict_hits += tuple(comp) in strict_sets
                # the (k-1)-tolerant clause [d > hi_val] OR NOT [d > lo_val - 1] realises it
                tol = (d > d[hi]) | ~(d > d[lo] - 1)
                ok &= bool((tol == comp).all())
        therm.append(dict(L=L, two_ray_sets=n_comp, expressible_by_strict=n_strict_hits))
        ok &= n_strict_hits == 0
    return save("X14", dict(claim="THEORY_HTM H1.1/H1.2: Th_{k-delta} needs exactly C(k, delta) strict terms; "
                                  "two-ray offset sets are one tolerant clause and no strict clause",
                            instances=inst, thermometer=therm, **{"pass": bool(ok)}))


# ----------------------------------------------------------------------------- X15
def X15():
    rng = np.random.default_rng(0)
    rows, ok = [], True
    for n in (8, 63):
        pstar = 1 - 2 ** (-1 / n)
        grid = np.unique(np.concatenate([np.geomspace(1e-3, 0.3, 25), [pstar]]))
        ents = []
        for p in grid:
            T = 200_000
            bits = (rng.random((T, n)) < p).any(1)
            q = bits.mean()
            se = math.sqrt(q * (1 - q) / T)
            pred = 1 - (1 - p) ** n
            h = 0.0 if q in (0, 1) else -(q * math.log2(q) + (1 - q) * math.log2(1 - q))
            ents.append(h)
            good = abs(q - pred) <= 3 * max(se, 1 / T)  # se = 0 when every trial hits
            ok &= good
            rows.append(dict(n=n, p=float(p), density=float(q), predicted=float(pred), entropy=h, within_3se=bool(good)))
        k = int(np.argmax(ents))
        peak_ok = abs(math.log(grid[k] / pstar)) <= math.log(grid[1] / grid[0]) + 1e-9
        ok &= peak_ok
        rows.append(dict(n=n, p_star=pstar, entropy_peak_p=float(grid[k]), peak_within_one_step=bool(peak_ok)))
    # hashed graph inbox, with the library's binding (row t*C + j: clause j's bits shifted by t)
    import torch
    import torchtsetlin as tt
    C, M, b, nt, n = 2000, 1024, 2, 16, 63
    m = tt.GraphTsetlinMachine(206, 10, C, 80.0, 2.0, depth=2, message_size=M, message_bits=b, n_edge_types=nt)
    bits = m.msg_bits.cpu()
    hashed = []
    for f in (0.001, 0.005, 0.01, 0.02, 0.05):
        dens = []
        for _ in range(20):
            inbox = torch.zeros(M, dtype=torch.bool)
            et = torch.randint(0, nt, (n,))
            for s in range(n):
                fire = torch.nonzero(torch.rand(C) < f).flatten()
                inbox[(bits[fire] + int(et[s])) % M] = True
            dens.append(inbox.float().mean().item())
        pred = 1 - math.exp(-b * C * f * n / M)
        hashed.append(dict(f=f, density=float(np.mean(dens)), sd=float(np.std(dens)), predicted=pred))
        ok &= abs(np.mean(dens) - pred) <= 0.03
    return save("X15", dict(claim="THEORY_HTM H2.1/H2.2: OR-inbox density 1-(1-p)^n, entropy peak at p*; hashed "
                                  "graph inbox density ~ 1-exp(-bCfn/M)", or_inbox=rows, hashed_inbox=hashed,
                            **{"pass": bool(ok)}))


# ----------------------------------------------------------------------------- X16
def X16():
    rng = np.random.default_rng(1)
    Fv, K, n = 8, 3, 200
    centres = rng.integers(0, 2, (K, Fv))
    X = centres[rng.integers(0, K, n)] ^ (rng.random((n, Fv)) < 0.15)
    starts, ok = [], True
    for st in range(20):
        perm = rng.normal(0, 0.01, (K, Fv, 2))  # permanence of literal z_v=1 (0) and z_v=0 (1)
        perm[:, :, 0] += rng.random((K, Fv))
        perm[:, :, 1] += rng.random((K, Fv))
        converged = False
        for it in range(50_000):
            proto = (perm[:, :, 0] > perm[:, :, 1]).astype(int)  # connected literal per variable
            overlap = (X[:, None, :] == proto[None]).sum(2)  # = F - Hamming
            win = np.argmax(overlap, 1)
            drift = np.zeros((K, Fv))
            for c in range(K):
                Xc = X[win == c]
                if len(Xc) == 0:
                    continue
                ones = Xc.mean(0)
                drift[c] = 2 * ones - 1  # mean increment of literal z_v=1 minus that of z_v=0, per won input
                perm[c, :, 0] += 0.05 * (ones - (1 - ones))
                perm[c, :, 1] += 0.05 * ((1 - ones) - ones)
            perm = np.clip(perm, 0, 1)
            # converged: no connected literal can flip (the drift never favours the other literal)
            if np.all((drift == 0) | ((drift > 0) == (proto == 1))):
                converged = True
                break
        proto = (perm[:, :, 0] > perm[:, :, 1]).astype(int)
        overlap = (X[:, None, :] == proto[None]).sum(2)
        win = np.argmax(overlap, 1)
        nearest = bool((overlap[np.arange(n), win] == overlap.max(1)).all())
        median = True
        for c in range(K):
            Xc = X[win == c]
            if len(Xc) == 0:
                continue
            ones = Xc.mean(0)
            maj = ones > 0.5
            tie = ones == 0.5
            median &= bool(((proto[c] == maj) | tie).all())
        ok &= nearest and median and converged
        starts.append(dict(start=st, iterations=it + 1, converged=converged, assigned_to_nearest=nearest, prototypes_are_medians=median))
    return save("X16", dict(claim="THEORY_HTM H2.3: one-winner literal SP limits are Lloyd fixed points of Hamming "
                                  "k-medians", starts=starts, **{"pass": bool(ok)}))


CHECKS = {"X13": X13, "X14": X14, "X15": X15, "X16": X16}

if __name__ == "__main__":
    for name in (sys.argv[1:] or CHECKS):
        CHECKS[name]()
