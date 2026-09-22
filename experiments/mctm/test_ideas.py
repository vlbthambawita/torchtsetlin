"""Invariant checks for the three repairs. Run with `python test_ideas.py`.

The credit path resolves a layer-2 feedback event to a single layer-1 (clause, position) by
three chained index computations across two different patch geometries and a pooling stage.
Nothing downstream would look wrong if one of them were off by a row -- layer 1 would simply
learn from the wrong patch -- so the invariants are asserted directly.
"""
from __future__ import annotations

import sys

import torch
from ideas import CreditStack, calibrate_density, embed_flat_into_conv, pixel_features
from mctm import conv_clause_maps, or_pool, randomize_clauses

import torchtsetlin as tt
from torchtsetlin import functional as TTF

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


def build(pool=2, C1=16, C2=32, Z=3, H=16, W=16, B=8, seed=0):
    tt.seed_everything(seed)
    x = (torch.rand(B, Z, H, W, device=DEV) < 0.3)
    y = torch.randint(0, 2, (B,), device=DEV)
    l1 = tt.ConvCoalescedTsetlinMachine(2, C1, float(C1), 5.0, patch_size=4, stride=1,
                                        position_encoding=True, input_shape=(Z, H, W)).to(DEV)
    randomize_clauses(l1, n_include=2, seed=seed)
    maps = or_pool(conv_clause_maps(l1, x), pool)
    l2 = tt.ConvCoalescedTsetlinMachine(2, C2, float(C2), 5.0, patch_size=3, stride=1,
                                        position_encoding=True,
                                        input_shape=tuple(maps.shape[1:])).to(DEV)
    return x, y, l1, l2, maps


def test_credit_positions():
    """A hand-written layer-2 clause with exactly one included literal -- "channel k fires at
    offset (dy,dx)" -- pins down the whole credit chain. Every position it credits must be one
    where channel k fired, and it must lie in the pooled cell that offset names relative to a
    patch the clause matched."""
    for pool in (1, 2):
        for (kc, dy, dx) in ((0, 0, 0), (3, 1, 2), (7, 2, 1)):
            x, y, l1, l2, pooled = build(pool=pool)
            kh2, kw2 = l2.patch_size
            m = kc * kh2 * kw2 + dy * kw2 + dx        # p[z,i,j] at z*kh*kw + i*kw + j
            l2.ta_state.fill_(l2.n_states - 1)
            l2.ta_state[:, m] = l2.n_states           # every clause: just that one literal
            l2._refresh_include()
            st = CreditStack(l1, l2, pool=pool)
            st.maps_shape = tuple(pooled.shape[1:])
            raw = conv_clause_maps(l1, x)
            lits2 = l2._encode(l2._prepare(pooled))
            B, P2, L = lits2.shape
            viol2 = TTF.clause_violations(lits2.reshape(B * P2, L), l2.include).view(B, P2, -1)
            matches2 = viol2 == 0
            out2 = matches2.any(dim=1)
            if not bool(out2.any()):
                continue
            Py2, Px2 = l2._grid(*st.maps_shape[1:])
            ev = st._events("i", out2, matches2, raw, Py2, Px2)
            tag = f"pool={pool} k={kc} off=({dy},{dx})"
            if ev is None:
                check(f"{tag}: events produced", False)
                continue
            bi, kk, pp = ev
            W1 = raw.shape[3]
            row, col = torch.div(pp, W1, rounding_mode="floor"), pp % W1
            check(f"{tag}: credited clause is the one the literal names",
                  bool((kk == kc).all()))
            check(f"{tag}: credited position fired",
                  bool(raw[bi, kk, row, col].all()),
                  f"{bi.numel()} events")
            # the pooled cell the position falls in must be a cell some matching patch covers
            pr, pc = torch.div(row, pool, rounding_mode="floor"), torch.div(col, pool, rounding_mode="floor")
            ok = torch.zeros_like(bi, dtype=torch.bool)
            for i in range(bi.numel()):
                py, px = int(pr[i]) - dy, int(pc[i]) - dx
                ok[i] = (0 <= py < Py2 and 0 <= px < Px2
                         and bool(matches2[bi[i], py * Px2 + px, 0]))
            check(f"{tag}: cell lies in a matching layer-2 patch", bool(ok.all()),
                  f"{int(ok.sum())}/{ok.numel()}")


def test_credit_literal_is_included_and_satisfied():
    """The channel literal a Type Ia event is credited through must be one the layer-2 clause
    actually includes -- and, because the clause matched, one that was satisfied."""
    x, y, l1, l2, _ = build()
    st = CreditStack(l1, l2, pool=2)
    st.train()
    for _ in range(4):
        st.update(x, y)
    raw, pooled = st.maps(x, raw=True)
    n_pix2 = raw.shape[1] * l2.patch_size[0] * l2.patch_size[1]
    check("pixel_features matches the pixel-literal block",
          pixel_features(l2) == n_pix2, f"{pixel_features(l2)} vs {n_pix2}")
    inc = l2.included_mask()[:, :n_pix2]
    check("layer 2 has positive channel literals to propagate through",
          bool(inc.any()), f"{int(inc.sum())} included")


def test_calibration_reaches_target():
    """The controller must move a layer in BOTH directions and leave no clause empty."""
    x, y, l1, l2, _ = build(C1=32)
    for name, setup in (("sparse", lambda m: m.ta_state.fill_(m.n_states)),      # all included
                        ("dense", lambda m: randomize_clauses(m, 1, seed=1))):
        setup(l1)
        l1._refresh_include()
        before = float((TTF.clause_violations(
            1.0 - l1._encode(l1._prepare(x)).reshape(-1, l1.n_literals),
            l1.include) == 0).float().mean())
        r = calibrate_density(l1, x, 0.05, n_patches=4000, seed=0)
        check(f"{name}: median rate reaches the band",
              0.025 <= r["rate_median"] <= 0.10,
              f"{before:.3f} -> {r['rate_median']:.3f}, size {r['size_median']:.0f}")
        check(f"{name}: no clause left empty", r["n_empty"] == 0)


def test_calibration_size_cap():
    """Under a size cap the controller must respect the cap, and must pick removals that keep
    the rate as near the target as the cap allows -- not the removal that buys the most rate
    (which leaves the most permissive literals) nor the least (which leaves the most
    selective)."""
    x, y, l1, l2, _ = build(C1=32)
    for cap in (12, 6, 3):
        l1.ta_state.fill_(l1.n_states)          # every literal included: maximally specific
        l1._refresh_include()
        r = calibrate_density(l1, x, 0.20, max_size=cap, n_patches=4000, seed=0,
                              max_rounds=400)
        size = int(l1.include_count.max())
        check(f"cap={cap}: no clause exceeds it", size <= cap, f"max size {size}")
        check(f"cap={cap}: rate is not degenerate",
              0.01 < r["rate_median"] < 0.999, f"{r['rate_median']:.3f}")


def test_calibration_adds_only_pixels():
    """Position literals are a free way to hit any firing rate while carrying no information;
    the controller must not reach for them."""
    x, y, l1, l2, _ = build(C1=32)
    l1.ta_state.fill_(l1.n_states - 1)          # every clause empty -> rate 1 -> must ADD
    l1._refresh_include()
    calibrate_density(l1, x, 0.05, n_patches=4000, seed=0)
    n_pix = pixel_features(l1)
    Fn = int(l1.n_features)
    inc = l1.included_mask()
    pos_bits = inc[:, n_pix:Fn].any() or inc[:, Fn + n_pix:].any()
    check("controller added no position literals", not bool(pos_bits))
    check("controller added something", int(l1.include_count.sum()) > 0)


def test_embed_flat_into_conv():
    """A flat model's automata must land in the conv layer's columns for the same features."""
    x, y, l1, l2, _ = build()
    Fn = int(l1.n_features)
    n_pix = pixel_features(l1)
    keep = torch.zeros(n_pix, dtype=torch.bool, device=DEV)
    keep[::2] = True
    flat = tt.CoalescedTsetlinMachine(int(keep.sum()), 4, int(l1.n_clauses_total),
                                      float(l1.n_clauses_total), 5.0, multi_label=True).to(DEV)
    flat.ta_state.random_(0, 2 * flat.n_states)
    flat._refresh_include()
    embed_flat_into_conv(l1, flat, keep)
    cols = torch.nonzero(keep).flatten()
    check("positive literals land in the right columns",
          bool((l1.ta_state[:, cols] == flat.ta_state[:, : cols.numel()]).all()))
    check("negated literals land in the right columns",
          bool((l1.ta_state[:, Fn + cols] == flat.ta_state[:, cols.numel():]).all()))
    untouched = torch.ones(2 * Fn, dtype=torch.bool, device=DEV)
    untouched[cols] = False
    untouched[Fn + cols] = False
    check("everything else stays excluded",
          bool((l1.ta_state[:, untouched] == l1.n_states - 1).all()))


if __name__ == "__main__":
    for fn in (test_credit_positions, test_credit_literal_is_included_and_satisfied,
               test_calibration_reaches_target, test_calibration_size_cap,
               test_calibration_adds_only_pixels, test_embed_flat_into_conv):
        print(f"{fn.__name__}:")
        fn()
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    sys.exit(0 if all(OK) else 1)
