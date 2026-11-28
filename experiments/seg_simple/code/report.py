"""Status, pricing and the report, all generated from ``results/runs/*.json`` (rule C2).

    python code/report.py --status     progress per phase, what is running, the cost table once priced
    python code/report.py --price      seconds per epoch from the pricing phases and the projected cost
    python code/report.py              writes report/REPORT.md and report/figures/*.png
    python code/report.py --wandb      the same, and uploads the figures to the W&B run ``summary``
"""

import argparse
import os
import statistics
import time

import jobs as JOBS
from common import CACHE, PATIENCE, PROG, RESULTS, read_json, record_path

CAP = {"dense": 160.0, "conv3d": 50.0}               # GPU-hours (plan section 9 and 12.2)
NGPU = int(os.environ.get("SS_GPUS", "3"))             # only for the wall-time estimate
SE2 = 2.0                                            # "more than twice the standard error"


def rec(tag):
    return read_json(record_path(tag))


def done(tag):
    r = rec(tag)
    return r if r and r.get("status") == "done" else None


# --------------------------------------------------------------------------- status and pricing
def status():
    claims = {d.name for d in (CACHE / "claims").glob("*") if d.is_dir()}
    spent = 0.0
    for ph in JOBS.PHASES:
        js, _ = JOBS.jobs(ph)
        n_done = sum(1 for j in js if done(j["tag"]))
        run = [j for j in js if j["tag"] in claims]
        wall = sum((rec(j["tag"]) or {}).get("wall_s") or 0.0 for j in js)
        spent += wall
        print(f"{ph:8s} {n_done:4d} / {len(js):4d} done, {len(run)} running, {wall / 3600:6.1f} GPU-h")
        for j in run:
            r = rec(j["tag"]) or {}
            print(f"         running {j['tag']}  epoch {r.get('epochs', 0)}  best {r.get('best')}"
                  f" at {r.get('best_epoch')}  (patience {PATIENCE})")
    print(f"total    {spent / 3600:.1f} GPU-hours recorded (caps of the plan: {CAP['dense']:.0f} without the 3D phase, "
          f"{CAP['conv3d']:.0f} for it)")
    for f in sorted((CACHE / "claims").glob("*.failed-*")):
        print(f"FAILED {f.name}")
    g = read_json(RESULTS / "gate_G0.json")
    if g:
        print(f"\ngate G0 (the tutorial's setting on its generated scenes): {'PASSED' if g['pass'] else 'FAILED'}, "
              f"test mIoU {g['test_mIoU']:.3f} (needs {g['needs']}, tutorial {g['tutorial']})")
    for ph in ("price", "price3d"):
        if all(done(j["tag"]) for j in JOBS.jobs(ph)[0]):
            print(f"\n{ph} complete:")
            price(ph)


def priced_seconds(ph):
    """price key -> seconds per epoch (median of the epochs after the first, which holds the compilation)."""
    out = {}
    for j in JOBS.jobs(ph)[0]:
        r = done(j["tag"])
        if r:
            h = [x["train_s"] + x["eval_s"] for x in r["history"]]
            out[JOBS.price_key(j)] = statistics.median(h[1:] or h)
    return out


def price(ph="price"):
    rows = priced_seconds(ph)
    if not rows:
        print(f"no records of phase {ph} yet: ./run_gpu0.sh {ph}")
        return
    phases = ["main", "side", "wide"] if ph == "price" else ["conv3d"]
    total, cells, missing = 0.0, {}, 0
    for p in phases:
        h = 0.0
        js = [j for j in JOBS.jobs(p)[0] if j["kind"] == "train"]
        for j in js:
            s = rows.get(JOBS.price_key(j))
            if s is None:
                missing += 1
                continue
            c = s * JOBS.ASSUMED_EPOCHS[j["aug"]] / 3600
            h += c
            cells[j["cell"]] = (s, c, cells.get(j["cell"], (0, 0, 0))[2] + 1)
        total += h
        print(f"  {p:7s} {len(js):4d} training runs  {h:7.1f} GPU-h")
    cap = CAP["dense" if ph == "price" else "conv3d"]
    print(f"  projected {total:.1f} GPU-h ({total / NGPU:.1f} h of wall time on {NGPU} GPUs with one runner each), cap {cap:.0f}: "
          f"{'within the cap' if total <= cap else 'OVER THE CAP: apply the trimming rule of the plan'}"
          + (f"; {missing} run(s) without a price" if missing else ""))
    print(f"  assumed epochs per run: {JOBS.ASSUMED_EPOCHS} (early stopping decides the real length)")
    print(f"  {'most expensive cells':40s} {'s/epoch':>8s} {'GPU-h per run':>14s} {'runs':>5s}")
    for cell, (s, c, n) in sorted(cells.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"  {cell:40s} {s:8.1f} {c:14.2f} {n:5d}")


# --------------------------------------------------------------------------- values of cells
def vals(cell, split="val", key="mIoU", seeds=None):
    """seed -> value of every finished run of a cell."""
    out = {}
    for s in (seeds if seeds is not None else range(9)):
        r = done(f"{cell}__s{s}") or done(f"{cell}__{s}")
        if r:
            out[s] = r[key] if split is None else r[split][key]
    return out


def cell(n, k, **kw):
    return JOBS.J("x", "x", n=n, k=k, **kw)["cell"]


def ms(v, digits=3):
    v = list(v.values()) if isinstance(v, dict) else list(v)
    if not v:
        return "—"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{statistics.mean(v):.{digits}f} ± {statistics.stdev(v):.{digits}f}"


def mean(v):
    return statistics.mean(v.values()) if v else None


def paired(a, b):
    """Mean and standard error of ``a - b`` over the seeds both have. ``None`` below two seeds."""
    d = [a[s] - b[s] for s in sorted(set(a) & set(b))]
    if len(d) < 2:
        return None
    return statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5, d


def verdict(p, name):
    if p is None:
        return f"{name}: not enough runs yet"
    m, se, d = p
    return (f"{name}: {m:+.4f} ± {se:.4f} (standard error, n = {len(d)}): "
            f"{'exceeds' if m > SE2 * se and m > 0 else 'does not exceed'} twice the standard error")


def best_of(cells, split="val"):
    c = [(mean(vals(x, split)), x) for x in cells if vals(x, split)]
    return max(c)[1] if c else None


# --------------------------------------------------------------------------- the report
def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def line_fig(path, xs, series, xlabel, logx=False, gap=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 if gap else 1, figsize=(10 if gap else 5.5, 3.8), squeeze=False)
    for ax, group, ylabel in ((axes[0, 0], series, "validation mIoU"),) + (((axes[0, 1], gap, "train − validation mIoU"),) if gap else ()):
        for name, cells_ in group.items():
            pts = [(x, vals(c) if ylabel.startswith("val") else vals(c, None, "gap_train_minus_val")) for x, c in zip(xs, cells_)]
            pts = [(x, v) for x, v in pts if v]
            if pts:
                ax.errorbar([x for x, _ in pts], [mean(v) for _, v in pts],
                            yerr=[statistics.stdev(v.values()) if len(v) > 1 else 0 for _, v in pts], marker="o", capsize=3, label=name)
        if logx:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def report(upload=False):
    out = PROG / "report"
    figs = []
    L = ["# Simple Tsetlin segmentation on Kvasir-SEG: clauses and patch size against quality", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `results/runs/*.json`. Every model is a Tsetlin machine. "
         "Values are mean ± standard deviation over seeds; validation selects the epoch, test is scored once.", ""]
    fl = read_json(RESULTS / "floors.json")
    if fl:
        L += ["## Floors (no image content)", ""] + table(
            ["Floor", "val mIoU", "val polyp IoU", "test mIoU"],
            [[k, f"{v['val']['mIoU']:.3f}", f"{v['val']['IoU_polyp']:.3f}", f"{v['test']['mIoU']:.3f}"]
             for k, v in fl.items() if k != "polyp_share"])

    def rows_for(cells_, labels):
        return [[lab, ms(vals(c)), ms(vals(c, "val", "IoU_polyp")), ms(vals(c, "test")), ms(vals(c, "test", "Dice")),
                 ms(vals(c, None, "gap_train_minus_val")), ms(vals(c, None, "epochs"), 0), ms(vals(c, None, "s_per_epoch"), 1)]
                for lab, c in zip(labels, cells_)]
    head = ["val mIoU", "val polyp IoU", "test mIoU", "test Dice", "train − val", "epochs", "s / epoch"]

    # clause line (H1, H4, H8)
    ns = [n for n, _ in JOBS.CLAUSE_LINE]
    cl, cls = [cell(n, 3) for n in ns], [cell(n, 3, aug="sym") for n in ns]
    figs.append(line_fig(out / "figures" / "clause_line.png", ns, {"no augmentation": cl, "sym": cls},
                         "clauses per class (k = 3)", True, {"no augmentation": cl, "sym": cls}))
    L += ["## Clause line (patch 3)", "", "![clause line](figures/clause_line.png)", ""]
    L += table(["clauses per class"] + head, rows_for(cl, [str(n) for n in ns]))
    L += ["With augmentation `sym`:", ""] + table(["clauses per class"] + head, rows_for(cls, [str(n) for n in ns]))

    def n_star(cells_):
        m = [(n, mean(vals(c))) for n, c in zip(ns, cells_) if vals(c)]
        return min(n for n, v in m if v >= max(v for _, v in m) - 0.01) if m else None
    ks = [k for _, k in JOBS.PATCH_LINE]
    pl, pls = [cell(400, k) for k in ks], [cell(400, k, aug="sym") for k in ks]
    figs.append(line_fig(out / "figures" / "patch_line.png", ks, {"no augmentation": pl, "sym": pls},
                         "patch size k (400 clauses per class)", False, {"no augmentation": pl, "sym": pls}))
    L += ["## Patch line (400 clauses per class)", "", "![patch line](figures/patch_line.png)", ""]
    L += table(["patch"] + head, rows_for(pl, [str(k) for k in ks]))
    L += ["With augmentation `sym`:", ""] + table(["patch"] + head, rows_for(pls, [str(k) for k in ks]))

    L += ["## Grid: clauses × patch (validation mIoU)", ""]
    L += table(["clauses \\ patch"] + [str(k) for k in JOBS.GRID_K],
               [[str(n)] + [ms(vals(cell(n, k))) for k in JOBS.GRID_K] for n in JOBS.GRID_N])
    figs.append(line_fig(out / "figures" / "grid.png", list(JOBS.GRID_N),
                         {f"k = {k}": [cell(n, k) for n in JOBS.GRID_N] for k in JOBS.GRID_K}, "clauses per class", True))
    L += ["![grid](figures/grid.png)", ""]

    L += ["## Side arms (patch 3, validation mIoU)", ""]
    arms = {"main": {}, "T = 0.1 n": {"t_ratio": 0.1}, "T = 0.4 n": {"t_ratio": 0.4}, "no position bits": {"pos": False},
            "balanced feedback": {"bal": True}, "shuffled pixels": {"order": "shuffle"}, "sym": {"aug": "sym"},
            "bit planes B8": {"enc": "B8"}, "128 px (seed 0)": {"px": 128}}
    L += table(["arm"] + [f"n = {n}" for n in JOBS.ARM_N], [[a] + [ms(vals(cell(n, 3, **kw))) for n in JOBS.ARM_N] for a, kw in arms.items()])

    L += ["## Encodings in the dense model (400 clauses per class, validation mIoU)", ""]
    ek = (1, 3, 5, 7, 9)
    L += table(["encoding"] + [f"k = {k}" for k in ek],
               [[e] + [ms(vals(cell(400, k, enc=e))) for k in ek] for e in ("T4", "T8", "B8")])

    L += ["## Wide teams", ""]
    names = ["W4x100", "W2x400", "W4x400", "W6x400", "W4x1600", "W4x400.sym", "Wgreedy", "Wmix", "Wsame-k3", "Wsame-k5",
             "Wsame-k9", "Wgreedy3d"]
    trows = []
    for nm in names:
        c = f"team-{nm}"
        rs = [r for r in (done(f"{c}__{s}") for s in range(3)) if r]
        if rs:
            trows.append([nm, str(rs[0]["clauses_per_class"]), ms(vals(c)), ms(vals(c, "test")),
                          ms([r["best_member_val_mIoU"] for r in rs]),
                          ms([r["mean_pairwise_disagreement"] for r in rs if r["mean_pairwise_disagreement"] is not None])])
    L += table(["team", "clauses per class", "val mIoU", "test mIoU", "best member (val)", "disagreement"], trows)
    L += ["`Wgreedy` and `Wgreedy3d` choose their members on validation: only their test numbers are fair estimates.", ""]

    L += ["## 3D convolution (400 clauses per class, 9×9 window, validation mIoU)", ""]
    for e in JOBS.WIN_ENC:
        w2 = {k: JOBS.J("x", "x", model="win2d", n=400, k=k, enc=e)["cell"] for k in JOBS.WIN_K}
        w3 = {(k, d): JOBS.J("x", "x", model="win3d", n=400, k=k, kd=d, enc=e)["cell"] for k in JOBS.WIN_K for d in JOBS.WIN_KD}
        L += [f"Encoding {e}. Dense references without position bits: all pixels {ms(vals(cell(400, 9, enc=e, pos=False)))}, "
              f"sampled pixels {ms(vals(cell(400, 9, enc=e, pos=False, sampled=True)))}.", ""]
        L += table(["patch", "2D window"] + [f"3D, depth {d}" for d in JOBS.WIN_KD],
                   [[str(k), ms(vals(w2[k]))] + [ms(vals(w3[k, d])) for d in JOBS.WIN_KD] for k in JOBS.WIN_K])
        figs.append(line_fig(out / "figures" / f"conv3d_{e}.png", list(JOBS.WIN_K),
                             {"2D window": [w2[k] for k in JOBS.WIN_K],
                              **{f"3D, depth {d}": [w3[k, d] for k in JOBS.WIN_K] for d in JOBS.WIN_KD}}, f"patch size k ({e})"))
        L += [f"![3D {e}](figures/conv3d_{e}.png)", ""]
    chk = paired(vals("win3d.T8.k3.d8.n400"), vals("win2d.T8.k3.n400"))
    if chk:
        L += [f"Check: 3D with the full depth minus the 2D window model, {chk[0]:+.5f} (must be 0 on the same GPU model).", ""]

    L += ["## Improvement arms (400 clauses per class, phase `improve`)", "",
          "Models with patch 1 on planes that hold the 3 × 3 neighbourhood at each scale and the position bits. "
          f"Training pixels: {JOBS.POOL['pool_n']} per image and epoch, drawn from all images and mixed before they are "
          f"committed 128 at a time (class share {JOBS.POOL['pool_a']}, boundary share {JOBS.POOL['pool_b']}). "
          "The first row is the dense model of the main phase (all pixels, raster order).", ""]
    ref = cell(JOBS.IMPROVE_N, 3)
    irows = [["dense, T4, 3 × 3 (main)", str(JOBS.features(JOBS.J("x", "x", n=JOBS.IMPROVE_N, k=3))), ms(vals(ref)),
              ms(vals(ref, "val", "IoU_polyp")), ms(vals(ref, "test")), ms(vals(ref, "test", "IoU_polyp")), "",
              ms(vals(ref, None, "gap_train_minus_val")), ms(vals(ref, None, "epochs"), 0), ms(vals(ref, None, "s_per_epoch"), 1)]]
    for e, sc in JOBS.IMPROVE_ARMS:
        j = JOBS.J("x", "x", model="pool", n=JOBS.IMPROVE_N, enc=e, scales=sc)
        c, d = j["cell"], paired(vals(j["cell"]), vals(ref))
        irows.append([f"sampled, {e}, " + ("3 × 3" if len(sc) == 1 else f"OR pyramid 1 to {max(sc)}"), str(JOBS.features(j)),
                      ms(vals(c)), ms(vals(c, "val", "IoU_polyp")), ms(vals(c, "test")), ms(vals(c, "test", "IoU_polyp")),
                      f"{d[0]:+.3f} ± {d[1]:.3f}" if d else "—", ms(vals(c, None, "gap_train_minus_val")),
                      ms(vals(c, None, "epochs"), 0), ms(vals(c, None, "s_per_epoch"), 1)])
    L += table(["model", "features", "val mIoU", "val polyp IoU", "test mIoU", "test polyp IoU", "val mIoU − main (± s.e.)",
                "train − val", "epochs", "s / epoch"], irows)

    L += ["## Coarse cell classifier (phase `coarse`)", "",
          "Images of 256 px. A convolutional machine decides for every cell whether at least half of its pixels are polyp, from a window "
          "around the cell. Its cell map is enlarged to the image and scored on the pixels, so the numbers compare with "
          "every table above. \"True cell map\" is the same score for the true cell labels: the most this cell size allows.", ""]
    crows = []
    ref_c = JOBS.coarse_job()
    fl2 = next((r["floors"] for r in (done(f"{ref_c['cell']}__s{s}") for s in range(3)) if r and r.get("floors")), None)
    if fl2:                                              # floors at the resolution of the reference, from its record
        crows += [[f"floor at {ref_c['px']} px: {k}", "", f"{fl2[k]['val']['mIoU']:.3f}", f"{fl2[k]['val']['IoU_polyp']:.3f}",
                   f"{fl2[k]['test']['mIoU']:.3f}", f"{fl2[k]['test']['IoU_polyp']:.3f}", "", "", "", ""]
                  for k in ("all_background", "position_prior_best_threshold")]
    names = {"": "reference: 256 px, cell 16, window 64, patch 5, T4, 400 clauses", "px": "image", "win": "window", "k": "patch", "enc": "encoding",
             "n": "clauses per class", "pos": "with the cell's position in the image", "csize": "cell"}
    for arm in JOBS.COARSE_ARMS:
        j = JOBS.coarse_job(**arm)
        c = j["cell"]
        orc = [r["cell_oracle"]["test"]["mIoU"] for r in (done(f"{c}__s{s}") for s in range(3)) if r and r.get("cell_oracle")]
        lab = names[""] if not arm else "; ".join(names[k] if k == "pos" else f"{names[k]} {v}" for k, v in arm.items())
        crows.append([lab, ms(orc[:1]), ms(vals(c)), ms(vals(c, "val", "IoU_polyp")), ms(vals(c, "test")),
                      ms(vals(c, "test", "IoU_polyp")), ms(vals(c, "test", "pixel_acc")), ms(vals(c, None, "gap_train_minus_val")),
                      ms(vals(c, None, "epochs"), 0), ms(vals(c, None, "s_per_epoch"), 1)])
    L += table(["model", "true cell map (test mIoU)", "val mIoU", "val polyp IoU", "test mIoU", "test polyp IoU",
                "test pixel accuracy", "train − val", "epochs", "s / epoch"], crows)

    L += ["## Clause groups of several neighbourhood sizes (phase `multi`)", "",
          "Images of 256 px. Every pixel is judged by groups of clauses that read its 3 × 3, 5 × 5, 9 × 9, 17 × 17 and "
          "25 × 25 neighbourhood (and the position bits). The groups learn from one joint vote sum, except in the row "
          "\"trained apart\". Training pixels are sampled across images as in phase `improve`. "
          "The last columns give the included literals per clause of the smallest and the largest group.", ""]
    mrows = []
    for arm in JOBS.MULTI_ARMS:
        j = JOBS.multi_job(**arm)
        c = j["cell"]
        if len(j["sizes"]) == 1:
            lab = f"one size: {j['sizes'][0]} × {j['sizes'][0]}, {j['counts'][0]} clauses"
        else:
            lab = "sizes " + ", ".join(map(str, j["sizes"])) + "; clauses " + (f"{j['counts'][0]} each" if len(set(j["counts"])) == 1
                                                                              else ", ".join(map(str, j["counts"])))
            lab += ("" if j["enc"] == "T4" else f"; {j['enc']}") + ("" if j["joint"] else "; trained apart")
        rs = [r for r in (done(f"{c}__s{s}") for s in range(3)) if r]
        lits = [ms([r["model"].get(f"literals_per_clause_k{k}") for r in rs if r["model"].get(f"literals_per_clause_k{k}") is not None], 0)
                for k in (j["sizes"][0], j["sizes"][-1])]
        mrows.append([lab, str(j["n"]), ms(vals(c)), ms(vals(c, "val", "IoU_polyp")), ms(vals(c, "test")), ms(vals(c, "test", "IoU_polyp")),
                      ms(vals(c, None, "gap_train_minus_val")), ms(vals(c, None, "epochs"), 0), ms(vals(c, None, "s_per_epoch"), 1)] + lits)
    L += table(["model", "clauses per class", "val mIoU", "val polyp IoU", "test mIoU", "test polyp IoU", "train − val", "epochs",
                "s / epoch", "literals, smallest", "literals, largest"], mrows)

    # ---------------------------------------------------------------- verdicts
    L += ["## Hypotheses", ""]
    V = []
    ns_main, ns_sym = n_star(cl), n_star(cls)
    m = [(n, mean(vals(c)), statistics.stdev(vals(c).values()) if len(vals(c)) > 1 else 0.0) for n, c in zip(ns, cl) if vals(c)]
    if ns_main:
        up = [b[1] >= a[1] - max(a[2], b[2]) for a, b in zip(m, m[1:]) if b[0] <= ns_main]
        V.append(f"H1: saturation point n* = {ns_main} clauses per class; the curve is "
                 f"{'monotone' if all(up) else 'not monotone'} within one standard deviation up to n*.")
    bp, bc = best_of(pl), best_of(cl)
    if bp and bc:
        base = vals(cell(400, 3))
        gp = {s: vals(bp)[s] - base[s] for s in set(vals(bp)) & set(base)}
        gc = {s: vals(bc)[s] - base[s] for s in set(vals(bc)) & set(base)}
        V.append(verdict(paired(gp, gc), f"H2: gain of the best patch (`{bp}`) minus gain of the best clause count (`{bc}`)"))
    g = {k: paired(vals(cell(1600, k)), vals(cell(100, k))) for k in (1, 9)}
    if g[1] and g[9]:
        seeds = sorted(set(vals(cell(1600, 9))) & set(vals(cell(100, 9))) & set(vals(cell(1600, 1))) & set(vals(cell(100, 1))))
        V.append(verdict(paired({s: vals(cell(1600, 9))[s] - vals(cell(100, 9))[s] for s in seeds},
                                {s: vals(cell(1600, 1))[s] - vals(cell(100, 1))[s] for s in seeds}),
                         "H3: gain from 100 to 1 600 clauses at patch 9 minus the same gain at patch 1"))
    for name, a, b in (("clauses 3 200 against 50", cl[-1], cl[0]), ("patch 13 against 1", pl[-1], pl[0])):
        V.append(verdict(paired(vals(a, None, "gap_train_minus_val"), vals(b, None, "gap_train_minus_val")),
                         f"H4: train − validation gap, {name}"))
    for tname, singles in (("W4x400", [cell(1600, k) for k in JOBS.GRID_K]), ("W4x100", [cell(400, k) for k in ks])):
        b = best_of(singles)
        if b:
            V.append(verdict(paired(vals(f"team-{tname}"), vals(b)), f"H5: `{tname}` minus the best single machine of equal clauses (`{b}`)"))
    b = best_of([f"team-Wsame-k{k}" for k in (3, 5, 9)])
    if b:
        V.append(verdict(paired(vals("team-Wmix"), vals(b)), f"H6: `Wmix` minus the best same-patch team (`{b[5:]}`)"))
    gain = {}
    for e in JOBS.WIN_ENC:
        gain[e] = {}
        for k in JOBS.WIN_K:
            b3 = best_of([f"win3d.{e}.k{k}.d{d}.n400" for d in JOBS.WIN_KD])
            if b3:
                p = paired(vals(b3), vals(f"win2d.{e}.k{k}.n400"))
                V.append(verdict(p, f"H7 ({e}, patch {k}): `{b3}` minus the 2D window model"))
                if p:
                    for s in set(vals(b3)) & set(vals(f"win2d.{e}.k{k}.n400")):
                        gain[e].setdefault(s, []).append(vals(b3)[s] - vals(f"win2d.{e}.k{k}.n400")[s])
    if gain.get("T8") and gain.get("B8"):
        V.append(verdict(paired({s: statistics.mean(v) for s, v in gain["T8"].items()},
                                {s: statistics.mean(v) for s, v in gain["B8"].items()}),
                         "H9: gain of the third dimension on T8 minus the gain on B8 (mean over patch sizes)"))
    for name, hi, lo, his, los in (("3 200 against 50 clauses", cl[-1], cl[0], cls[-1], cls[0]), ("patch 13 against 1", pl[-1], pl[0], pls[-1], pls[0])):
        a, b = paired(vals(his), vals(hi)), paired(vals(los), vals(lo))
        if a and b:
            seeds = sorted(set(vals(his)) & set(vals(hi)) & set(vals(los)) & set(vals(lo)))
            V.append(verdict(paired({s: vals(his)[s] - vals(hi)[s] for s in seeds}, {s: vals(los)[s] - vals(lo)[s] for s in seeds}),
                             f"H8: gain from augmentation, {name}"))
    if ns_main and ns_sym:
        V.append(f"H8: saturation point with augmentation n* = {ns_sym}, without n* = {ns_main}.")
    h10 = [(k, paired(vals(cell(400, k)), vals(cell(400, k, enc="B8")))) for k in ek]
    if all(p for _, p in h10):
        pos = sum(p[0] > 0 for _, p in h10)
        strong = sum(p[0] > SE2 * p[1] and p[0] > 0 for _, p in h10)
        V.append(f"H10: T4 minus B8 is positive at {pos} of 5 patch sizes and exceeds twice the standard error at {strong} "
                 f"({'supported' if pos == 5 and strong >= 3 else 'not supported'}): "
                 + ", ".join(f"k={k} {p[0]:+.3f}" for k, p in h10))
    L += [f"- {v}" for v in V] or ["No hypothesis can be checked yet."]
    L += ["", "## Limits", "", "Kvasir-SEG at 64 px; augmentation by flips and rotations only; three seeds (nine for the team "
          "controls); the margin T is tied to the clause count; window models train on 256 sampled pixels per image.", ""]
    (out / "REPORT.md").parent.mkdir(exist_ok=True)
    (out / "REPORT.md").write_text("\n".join(L))
    print(f"wrote {out / 'REPORT.md'} and {len(figs)} figures")
    if upload:
        from tracking import Tracker
        t = Tracker({"tag": "summary", "group": "summary", "phase": "report"}, {})
        for f in figs:
            t.image(0, f"summary/{f.stem}", f, f.stem)
        t.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--price", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    if a.status:
        status()
    elif a.price:
        price("price")
        price("price3d")
    else:
        report(a.wandb)
