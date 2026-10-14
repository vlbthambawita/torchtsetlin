"""Generate the fixed config lists for each wave (written to configs/<wave>.json).

    python code/configs.py e0
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PYR_OR = {"scales": [1, 2, 4], "mode": "or", "patch": 3}
BITS = {"kvasir": 4, "acdc": 8, "scenes": 4, "shapes": 1, "longrange": 4}
SIZE = {"kvasir": 128, "acdc": 128, "scenes": 32, "shapes": 32, "longrange": 64}


def b1(ds, **kw):
    return {"family": "tm", "dataset": ds, "size": SIZE[ds], "spec": {"bits": BITS[ds]}, "patch": 3, **kw}


def b2(ds, **kw):
    return {"family": "tm", "dataset": ds, "size": SIZE[ds], "spec": {"bits": BITS[ds], "pyr": PYR_OR}, "patch": 1, **kw}


def unet(ds, **kw):
    return {"family": "unet", "dataset": ds, "size": SIZE[ds], "epochs": 60, "batch": 16, **kw}


def seeds(cfgs, ss):
    return [{**c, "seed": s} for c in cfgs for s in ss]


def e0():
    out = []
    for ds in ("scenes", "shapes"):
        ep = {"epochs": 10}
        cl = 300 if ds == "scenes" else 100
        base = [
            {"name": "B0_majority", "family": "majority", "dataset": ds, "size": SIZE[ds]},
            {"name": "B0_rowprior", "family": "rowprior", "dataset": ds, "size": SIZE[ds]},
            {"name": "B0_linear", "family": "linear", "dataset": ds, "size": SIZE[ds], "epochs": 30, "batch": 32},
            b1(ds, name="B1", clauses=cl, **ep),
            b2(ds, name="B2_pyr_or", clauses=cl, **ep),
            unet(ds, name="B4_unet16", epochs=30, batch=32),
        ]
        out += seeds(base, [0, 1, 2])
    for ds in ("kvasir", "acdc"):
        ep = 10 if ds == "kvasir" else 4
        cl = 300 if ds == "kvasir" else 200
        out += [
            {"name": "B0_majority", "family": "majority", "dataset": ds, "native_eval": True},
            {"name": "B0_rowprior", "family": "rowprior", "dataset": ds, "native_eval": True},
            {"name": "B0_linear", "family": "linear", "dataset": ds, "epochs": 30, "batch": 16, "native_eval": True},
            b1(ds, name="B1", clauses=cl, epochs=ep, native_eval=True),
            b2(ds, name="B2_pyr_or", clauses=cl, epochs=ep, native_eval=True),
            unet(ds, name="B4_unet16", native_eval=True),
        ]
    for c in out:
        c["phase"] = "E0"
    # slowest (real-data TM) first so they start early
    return sorted(out, key=lambda c: (c["dataset"] not in ("acdc", "kvasir"), c["family"] != "tm"))


def _f(ds):
    """Kvasir: 3 colour channels; ACDC/longrange: 1; scenes: 3."""
    return {"kvasir": 3, "acdc": 1, "scenes": 3, "shapes": 1, "longrange": 1}[ds]


def e1_arms(ds, cl, n_sample):
    """H1 order x commit size, H1 sampling mixtures, H2 pooling/contrast (+ capacity controls).
    All arms start from frozen B2 settings (OR pyramid (1,2,4), patch 1 head)."""
    bits = BITS[ds]
    C = _f(ds)
    b2s = {"bits": bits, "pyr": PYR_OR}
    F_b2 = C * bits * 9 * 3
    arms = []
    for order in ("raster", "perm", "disperse"):
        for q in (32, 128, 512):
            arms.append(b2(ds, name=f"H1_{order}_q{q}", order=order, q=q, clauses=cl))
    for a, b in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.25)):
        arms.append(b2(ds, name=f"H1_sample_a{a}_b{b}", sample=[n_sample, a, b], clauses=cl))
    arms.append(b2(ds, name="H1_classfb_balanced", class_fb="balanced", clauses=cl))
    occ3 = {"scales": [1, 2, 4], "mode": "occ", "t": [0.25, 0.5, 0.75], "patch": 3}
    occm = {"scales": [1, 2, 4], "mode": "occ", "t": [0.5], "patch": 3}
    andp = {"scales": [1, 2, 4], "mode": "and", "patch": 3}
    con = {"windows": [9, 25], "t": [-0.1, 0.0, 0.1]}
    F_occ3 = C * bits * 9 * 7
    F_con = (C * bits + C * 6) * 27
    arms += [
        b2(ds, name="H2_and", spec={"bits": bits, "pyr": andp}, clauses=cl),
        b2(ds, name="H2_occ_maj", spec={"bits": bits, "pyr": occm}, clauses=cl),
        b2(ds, name="H2_occ3", spec={"bits": bits, "pyr": occ3}, clauses=cl),
        b2(ds, name="H2_occ3_bytes", spec={"bits": bits, "pyr": occ3}, clauses=max(10, round(cl * F_b2 / F_occ3))),
        b2(ds, name="H2_contrast", spec={**b2s, "contrast": con}, clauses=cl),
        b2(ds, name="H2_contrast_bytes", spec={**b2s, "contrast": con}, clauses=max(10, round(cl * F_b2 / F_con))),
        b1(ds, name="H2_patch5", patch=5, clauses=cl),
        b1(ds, name="H2_extrabits_p3", spec={"bits": bits, "extra_bits": bits}, clauses=cl),
    ]
    for nm, sp in (("pyr_or", b2s), ("occ3", {"bits": bits, "pyr": occ3}), ("contrast", {**b2s, "contrast": con})):
        arms.append({"name": f"H2_linear_{nm}", "family": "linear", "dataset": ds, "size": SIZE[ds], "input": "bool",
                     "spec": sp, "epochs": 30, "batch": 16})
    return arms


def e1():
    out = seeds(e1_arms("scenes", 300, 256), [0, 1, 2])
    out += [{**c, "epochs": EPOCHS_KV} for c in e1_arms("kvasir", 300, 2048)]
    for c in out:
        c["phase"] = "E1"
    # slow (q=32 Kvasir) first
    return sorted(out, key=lambda c: (c["dataset"] != "kvasir", c.get("q", 128) != 32))


EPOCHS_KV = 10

def _sample_grid(grid, n, seed):
    import itertools
    import random

    keys = list(grid)
    combos = list(itertools.product(*[grid[k] for k in keys]))
    random.Random(seed).shuffle(combos)
    return [dict(zip(keys, c)) for c in combos[:n]]


def e2_cnn(ds="kvasir"):
    """B4 U-Net search (12 seeded configs) + B5 (U-Net on the TM's Boolean planes; local CNN
    with B1's 3x3 receptive field on B1 planes)."""
    out = []
    grid = {"lr": [1e-4, 3e-4, 1e-3], "wd": [0.0, 1e-4], "width": [16, 32], "dice_w": [0.0, 0.5]}
    for i, hp in enumerate(_sample_grid(grid, 12, 2026)):
        out.append(unet(ds, name=f"B4_t{i:02d}", native_eval=True, **hp))
    bits = BITS[ds]
    for i, hp in enumerate(_sample_grid({"lr": [3e-4, 1e-3], "width": [16, 32]}, 4, 7)):
        out.append(unet(ds, name=f"B5_unet_bool_t{i}", input="bool", spec={"bits": bits, "pyr": PYR_OR}, native_eval=True, **hp))
        out.append({"name": f"B5_local_bool_t{i}", "family": "localcnn", "dataset": ds, "size": SIZE[ds], "input": "bool",
                    "spec": {"bits": bits}, "epochs": 40, "batch": 16, "lr": hp["lr"], "native_eval": True})
    for c in out:
        c["phase"] = "E2"
    return out


def e2_tm(ds="kvasir"):
    """B2 search (12 seeded configs) and B3 coalesced search (6), raster order (library)."""
    out = []
    grid = {"clauses": [100, 300, 600], "T_ratio": [0.1, 0.2, 0.5], "s": [3.0, 10.0, 20.0], "max_lits": [16, 32, None]}
    for i, hp in enumerate(_sample_grid(grid, 12, 2027)):
        out.append(b2(ds, name=f"B2_t{i:02d}", order="raster", epochs=EPOCHS_KV, native_eval=True, **hp))
    for i, hp in enumerate(_sample_grid(grid, 6, 2028)):
        hp = {**hp, "clauses": 2 * hp["clauses"]}  # same total clauses / state bytes as class-owned
        out.append(b2(ds, name=f"B3_t{i:02d}", family="coalesced", order="raster", epochs=EPOCHS_KV, native_eval=True, **hp))
    for c in out:
        c["phase"] = "E2"
    return out


TM_KV = {"clauses": 600, "T_ratio": 0.2, "s": 20.0, "max_lits": None}  # best of the B2 search so far (E2_kvasir_B2_t07)


def ctx_arms(ds, tmhp, scales, coarse_scales, epochs):
    bits = BITS[ds]
    sc = list(scales)
    orp = {"scales": sc, "mode": "or", "patch": 3}
    andp = {"scales": sc, "mode": "and", "patch": 3}
    occm = {"scales": sc, "mode": "occ", "t": [0.5], "patch": 3}
    occ3 = {"scales": sc, "mode": "occ", "t": [0.25, 0.5, 0.75], "patch": 3}
    base = dict(order="raster", epochs=epochs, **tmhp)
    arms = [
        b2(ds, name="CTX_or_wide", spec={"bits": bits, "pyr": orp}, **base),
        b2(ds, name="CTX_and_wide", spec={"bits": bits, "pyr": andp}, **base),
        b2(ds, name="CTX_occmaj_wide", spec={"bits": bits, "pyr": occm}, **base),
        b2(ds, name="CTX_occ3_wide", spec={"bits": bits, "pyr": occ3}, **base),
    ]
    coarse = {"spec": {"bits": bits, "pyr": {"scales": list(coarse_scales), "mode": "or", "patch": 3}}, "patch": 1,
              "clauses": tmhp["clauses"], "T_ratio": tmhp["T_ratio"], "s": tmhp["s"], "epochs": epochs, "q": 128}
    st = {"factor": 4, "coarse": coarse, "taus": [-0.5, 0.0, 0.5], "folds": 3}
    for nm, extra in (("oof", {}), ("shuffled", {"mode": "shuffled"}), ("insample", {"mode": "insample"}),
                      ("bilinear", {"upsample": "bilinear"}), ("ORACLE", {"mode": "oracle"})):
        arms.append(b1(ds, name=f"H3_stack_{nm}", stack={**st, **extra}, **base))
    for nm, sp in (("or_wide", {"bits": bits, "pyr": orp}), ("occmaj_wide", {"bits": bits, "pyr": occm})):
        arms.append({"name": f"CTX_linear_{nm}", "family": "linear", "dataset": ds, "size": SIZE[ds], "input": "bool",
                     "spec": sp, "epochs": 30, "batch": 16})
        arms.append(unet(ds, name=f"CTX_unet_bool_{nm}", input="bool", spec=sp, lr=1e-3, width=16))
    return arms


def e2_ctx():
    out = []
    kv = ctx_arms("kvasir", TM_KV, (1, 2, 4, 8, 16, 32), (1, 2, 4, 8), EPOCHS_KV)
    out += [{**c, "native_eval": c["family"] != "linear", "save_votes": c["family"] in ("tm", "unet")} for c in kv]
    lr_hp = {"clauses": 100, "T_ratio": 0.2, "s": 10.0, "max_lits": None}
    lr = ctx_arms("longrange", lr_hp, (1, 2, 4, 8, 16, 32), (1, 2, 4, 8), 10)
    lr += [b1("longrange", name="B1", order="raster", epochs=10, **lr_hp),
           b2("longrange", name="B2_pyr_or", order="raster", epochs=10, **lr_hp),
           unet("longrange", name="B4_unet16", lr=1e-3, epochs=40, batch=32)]
    core = {"B2_pyr_or", "CTX_or_wide", "CTX_occmaj_wide", "H3_stack_oof", "H3_stack_shuffled", "B4_unet16"}
    out += seeds([c for c in lr if c["name"] in core], [0, 1, 2]) + [c for c in lr if c["name"] not in core]
    for c in out:
        c["phase"] = "E2"
    return sorted(out, key=lambda c: (c["dataset"] != "kvasir", c["family"] != "tm"))


def e1x():
    """Corrected H1 on Kvasir (wide OR features, B2-search settings): cross-image pools and
    sampled pixels; commit size, mixture and image-pool size."""
    bits = BITS["kvasir"]
    sp = {"bits": bits, "pyr": {"scales": [1, 2, 4, 8, 16, 32], "mode": "or", "patch": 3}}
    base = dict(spec=sp, epochs=EPOCHS_KV, native_eval=True, **TM_KV)
    arms = [
        b2("kvasir", name="H1x_xperm8_q128", order="perm", cross_image=True, batch=8, **base),
        b2("kvasir", name="H1x_disperse8_q128", order="disperse", cross_image=True, batch=8, **base),
    ]
    for q in (32, 128):
        for a, b in ((0.0, 0.0), (0.25, 0.25)):
            arms.append(b2("kvasir", name=f"H1x_samp512_a{a}_b{b}_q{q}", sample=[512, a, b], q=q, cross_image=True,
                           batch=16, **{**base, "epochs": 20}))
    arms += [
        b2("kvasir", name="H1x_samp512_a0.5_b0.0_q32", sample=[512, 0.5, 0.0], q=32, cross_image=True, batch=16, **{**base, "epochs": 20}),
        b2("kvasir", name="H1x_samp512_a0.0_b0.5_q32", sample=[512, 0.0, 0.5], q=32, cross_image=True, batch=16, **{**base, "epochs": 20}),
        b2("kvasir", name="H1x_samp2048_a0.0_b0.0_q128", sample=[2048, 0.0, 0.0], q=128, cross_image=True, batch=8, **base),
        b2("kvasir", name="H1x_samp512_a0.0_b0.0_q32_1img", sample=[512, 0.0, 0.0], q=32, batch=16, **{**base, "epochs": 20}),
    ]
    for c in arms:
        c["phase"] = "E1"
        c["save_votes"] = True
    return arms


SAMPLED = {"sample": [512, 0.25, 0.25], "q": 128, "cross_image": True, "gather": True, "batch": 16, "epochs": 20}


def e1y(ds="kvasir", recipe=None, tmhp=None, seeds_=(0,)):
    """H2 representations and H3 stack under the sampled recipe (H1 winner)."""
    rcp = recipe or SAMPLED
    hp = tmhp or TM_KV
    bits = BITS[ds]
    W = [1, 2, 4, 8, 16, 32]
    P = {
        "pyr124": {"scales": [1, 2, 4], "mode": "or", "patch": 3},
        "or_wide": {"scales": W, "mode": "or", "patch": 3},
        "and_wide": {"scales": W, "mode": "and", "patch": 3},
        "occmaj_wide": {"scales": W, "mode": "occ", "t": [0.5], "patch": 3},
        "occ3_wide": {"scales": W, "mode": "occ", "t": [0.25, 0.5, 0.75], "patch": 3},
    }
    con = {"windows": [9, 25], "t": [-0.1, 0.0, 0.1]}
    arms = [b2(ds, name=f"S_{k}", spec={"bits": bits, "pyr": v}, **rcp, **hp) for k, v in P.items()]
    arms.append(b2(ds, name="S_or_wide_contrast", spec={"bits": bits, "contrast": con, "pyr": P["or_wide"]}, **rcp, **hp))
    arms.append(b2(ds, name="S_occmaj_wide_contrast", spec={"bits": bits, "contrast": con, "pyr": P["occmaj_wide"]}, **rcp, **hp))
    coarse = {"spec": {"bits": bits, "pyr": {"scales": [1, 2, 4, 8], "mode": "or", "patch": 3}}, "patch": 1,
              "clauses": hp["clauses"], "T_ratio": hp["T_ratio"], "s": hp["s"], "epochs": rcp["epochs"], "q": rcp["q"],
              "sample": [128, rcp["sample"][1], rcp["sample"][2]], "batch": 16}
    st = {"factor": 4, "coarse": coarse, "taus": [-0.5, 0.0, 0.5], "folds": 3}
    fine = {"bits": bits, "pyr": P["pyr124"]}
    for nm, extra in (("oof", {}), ("shuffled", {"mode": "shuffled"}), ("ORACLE", {"mode": "oracle"})):
        arms.append(b2(ds, name=f"S_H3_{nm}", spec=fine, stack={**st, **extra}, **rcp, **hp))
    out = seeds(arms, list(seeds_))
    for c in out:
        c["phase"] = "E1"
        c["native_eval"] = True
        c["save_votes"] = True
    return out


def e4_acdc_screen():
    """ACDC screening (1 seed): sampled recipe mixtures, all-pixel controls, CNN search, B5."""
    ds = "acdc"
    bits = BITS[ds]
    W = [1, 2, 4, 8, 16, 32]
    orw = {"bits": bits, "pyr": {"scales": W, "mode": "or", "patch": 3}}
    p124 = {"bits": bits, "pyr": PYR_OR}
    rc = {k: v for k, v in SAMPLED.items() if k != "sample"}
    out = []
    for a, b in ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.25, 0.25)):
        out.append(b2(ds, name=f"S_or_wide_a{a}_b{b}", spec=orw, sample=[512, a, b], **rc, **TM_KV))
    out.append(b2(ds, name="S_pyr124_a0.25_b0.25", spec=p124, sample=[512, 0.25, 0.25], **rc, **TM_KV))
    out.append(b2(ds, name="B2_allpix_raster", spec=p124, order="raster", epochs=4, **TM_KV))
    out.append(b2(ds, name="CTX_or_wide_allpix", spec=orw, order="raster", epochs=4, **TM_KV))
    grid = {"lr": [1e-4, 3e-4, 1e-3], "wd": [0.0, 1e-4], "width": [16, 32], "dice_w": [0.0, 0.5]}
    for i, hp in enumerate(_sample_grid(grid, 12, 2026)[:6]):
        out.append(unet(ds, name=f"B4_t{i:02d}", **hp))
    out.append(unet(ds, name="B5_unet_bool_orw", input="bool", spec=orw, lr=1e-3, width=16))
    out.append({"name": "B5_local_bool", "family": "localcnn", "dataset": ds, "size": 128, "input": "bool",
                "spec": {"bits": bits}, "epochs": 40, "batch": 16, "lr": 1e-3})
    for c in out:
        c["phase"] = "E4"
        c["native_eval"] = True
        c["save_votes"] = c["family"] == "tm"
    return sorted(out, key=lambda c: c.get("sample") is not None)


def e4(ds="kvasir", mix=(0.25, 0.25), allpix_epochs=None, unet_hp=None, sd=(0, 1, 2, 3, 4)):
    """Frozen E4 factorial (DECISIONS D-011): base = tuned B2 (all pixels, raster, B2 pyramid),
    A = +sampled pixels, B = +wide OR context, A+B = finalist; plus tuned U-Net."""
    bits = BITS[ds]
    orw = {"bits": bits, "pyr": {"scales": [1, 2, 4, 8, 16, 32], "mode": "or", "patch": 3}}
    p124 = {"bits": bits, "pyr": PYR_OR}
    rc = {k: v for k, v in SAMPLED.items() if k != "sample"}
    ape = allpix_epochs or EPOCHS_KV
    arms = [
        b2(ds, name="F_base_B2", spec=p124, order="raster", epochs=ape, **TM_KV),
        b2(ds, name="F_A_sampled", spec=p124, sample=[512, *mix], **rc, **TM_KV),
        b2(ds, name="F_B_wide", spec=orw, order="raster", epochs=ape, **TM_KV),
        b2(ds, name="F_AB_sampled_wide", spec=orw, sample=[512, *mix], **rc, **TM_KV),
        unet(ds, name="F_unet", **(unet_hp or {"lr": 1e-3, "wd": 0.0, "width": 32, "dice_w": 0.0})),
    ]
    out = seeds(arms, list(sd))
    for c in out:
        c["phase"] = "E4"
        c["native_eval"] = True
        c["save_votes"] = True
    return sorted(out, key=lambda c: (c.get("sample") is not None or c["family"] != "tm", c["seed"]))


def e4kv():
    return e4("kvasir")


def e4acdc_final():
    return e4("acdc", allpix_epochs=4, unet_hp={"lr": 1e-3, "wd": 1e-4, "width": 32, "dice_w": 0.0})


WAVES = {"e4ac": e4acdc_final, "e4kv": e4kv, "e0": e0, "e1": e1, "e2cnn": e2_cnn, "e2tm": e2_tm, "e2ctx": e2_ctx, "e1x": e1x, "e1y": e1y,
         "e4acdc": e4_acdc_screen}

if __name__ == "__main__":
    w = sys.argv[1]
    cfgs = WAVES[w]()
    os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
    with open(os.path.join(HERE, "configs", f"{w}.json"), "w") as f:
        json.dump(cfgs, f, indent=1)
    print(len(cfgs), "configs ->", f"configs/{w}.json")
