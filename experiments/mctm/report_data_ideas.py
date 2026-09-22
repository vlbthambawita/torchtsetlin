"""Turn results/*.json into the CSVs and \\newcommand macros report2/ reads.

Same contract as report_data.py: nothing in the PDF is hand-typed. Every number in the text
comes from report2/macros.tex and every figure and table from report2/data/*.csv, so
re-running the experiments and re-running this script updates the report wholesale.

The E1 arms of the first report are carried over unchanged -- they are the baselines the
three repairs have to beat, and they were produced by the same driver at the same budget.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as stats
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "report2", "data")

# --- carried over from E1 (report 1) --------------------------------------------------
BASE = {
    "single-matched": "Single layer, matched budget",
    "single-rf":      "Single layer, 9x9 patch (matched RF)",
    "single-l1":      "Layer 1 alone",
    "flat-stack":     "Flat stack (global OR)",
    "mctm":           "MCTM, greedy layer 1",
    "mctm-random":    "MCTM, random layer 1",
}
# --- new in this report ---------------------------------------------------------------
NEW = {
    "mctm-auto":         "A: MCTM, autoencoder layer 1",
    "flat-auto":         "A: flat stack, autoencoder layer 1",
    "flat-random":       "flat stack, random layer 1",
    "mctm-calib-r005":   "B: density target 0.5",
    "mctm-calib-r01":    "B: density target 1",
    "mctm-calib-r02":    "B: density target 2",
    "mctm-calib-r05":    "B: density target 5",
    "mctm-calib-r10":    "B: density target 10",
    "mctm-calib-r20":    "B: density target 20",
    "mctm-calib-r20-tight": "B: density target 20, tight band",
    "mctm-size-k24":     "B: + size target 24",
    "mctm-size-k12":     "B: + size target 12",
    "mctm-size-k6":      "B: + size target 6",
    "mctm-size-k3":      "B: + size target 3",
    "mctm-auto-calib":   "A+B: autoencoder + calibrated",
    "mctm-rc-r05":       "B: random L1, density target 5",
    "mctm-rc-r10":       "B: random L1, density target 10",
    "mctm-random-calib": "B: random L1, density target 20",
    "mctm-rc-r40":       "B: random L1, density target 40",
    "mctm-rc-r60":       "B: random L1, density target 60",
    "mctm-credit-ia":    "C: credit, Type Ia+II only",
    "mctm-credit-ib":    "C: credit, + blaming Type Ib",
    "mctm-credit-bal":   "C: credit, + Ib balanced",
    "mctm-credit-warm":     "C: warm start (no controller)",
    "mctm-credit-cal-p1":   "B+C: controller / step 1.0",
    "mctm-credit-cal":      "B+C: controller / step 0.1",
    "mctm-credit-cal-p001": "B+C: controller / step 0.01",
    "mctm-auto-credit":  "A+B+C: all three",
}
LABEL = {**BASE, **NEW}
CALIB_SWEEP = ["mctm-calib-r005", "mctm-calib-r01", "mctm-calib-r02",
               "mctm-calib-r05", "mctm-calib-r10", "mctm-calib-r20"]
CALIB_TARGET = {"mctm-calib-r005": 0.5, "mctm-calib-r01": 1.0, "mctm-calib-r02": 2.0,
                "mctm-calib-r05": 5.0, "mctm-calib-r10": 10.0,
                "mctm-calib-r20": 20.0}
RAND_SWEEP = ["mctm-rc-r05", "mctm-rc-r10", "mctm-random-calib", "mctm-rc-r40",
              "mctm-rc-r60"]
RAND_TARGET = {"mctm-rc-r05": 5, "mctm-rc-r10": 10, "mctm-random-calib": 20,
               "mctm-rc-r40": 40, "mctm-rc-r60": 60}
SIZE_SWEEP = ["mctm-size-k24", "mctm-size-k12", "mctm-size-k6", "mctm-size-k3"]
SIZE_TARGET = {"mctm-size-k24": 24, "mctm-size-k12": 12, "mctm-size-k6": 6, "mctm-size-k3": 3}
CREDIT_RUNGS = ["mctm-credit-ia", "mctm-credit-ib", "mctm-credit-bal",
                "mctm-credit-warm", "mctm-credit-cal-p1", "mctm-credit-cal",
                "mctm-credit-cal-p001", "mctm-auto-credit"]
ORDER = (["single-matched", "single-rf", "single-l1", "mctm", "mctm-random", "flat-stack"]
         + ["mctm-auto", "flat-auto", "flat-random"] + CALIB_SWEEP
         + ["mctm-calib-r20-tight"] + SIZE_SWEEP
         + ["mctm-auto-calib"] + RAND_SWEEP + CREDIT_RUNGS)

_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
           "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_name(arm: str, field: str) -> str:
    """LaTeX macro names may only contain letters, so digits become words."""
    core = "".join(w.capitalize() for w in arm.replace("_", "-").split("-"))
    return f"\\{field}" + "".join(_DIGITS.get(c, c) for c in core)


def thou(n: int) -> str:
    """Thin-space thousands separators. pgfplotstable typesets a `string type` cell as TeX,
    so the separator can be a real one rather than a comma (which is the column separator)."""
    return f"{int(n):,}".replace(",", r"\,")


def tex(s: str) -> str:
    """Escape a label for a macro body. An unescaped % comments out the rest of macros.tex,
    which pdflatex reports as dozens of undefined control sequences far from the cause."""
    return s.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("_", r"\_") \
            .replace("&", r"\&").replace("#", r"\#")


def agg(vals: List[float]):
    if not vals:
        return 0.0, 0.0
    return stats.mean(vals), (stats.stdev(vals) if len(vals) > 1 else 0.0)


def load() -> Dict[str, List[dict]]:
    arms: Dict[str, List[dict]] = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*_seed*.json"))):
        with open(p) as f:
            r = json.load(f)
        name = os.path.basename(p).rsplit("_seed", 1)[0]
        arms.setdefault(name, []).append(r)
    return arms


def train_min(r: dict) -> float:
    return (sum(e["epoch_s"] for e in r.get("log_l1", []))
            + sum(e["epoch_s"] for e in r.get("log", []))) / 60.0


def csv(path: str, header: str, rows: List[str]) -> None:
    # pgfplotstable reads these as TeX: an unescaped '%' in a cell comments out the rest of
    # the line and every later column silently shifts up a row.
    bad = [r for r in rows if "%" in r]
    if bad:
        raise SystemExit(f"{path}: '%' in a CSV cell would break the table: {bad[:2]}")
    with open(os.path.join(OUT, path), "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(r + "\n")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    arms = load()
    macros: List[str] = []
    have = [a for a in ORDER if a in arms]

    # ---- master results table ---------------------------------------------------
    rows = []
    for a in have:
        recs = arms[a]
        m, sd = agg([r["best_test_acc"] * 100 for r in recs])
        wm, _ = agg([train_min(r) for r in recs])
        rows.append(f"{a},{LABEL.get(a, a).replace(',', ' /')},{len(recs)},{m:.2f},{sd:.2f},"
                    f"{wm:.2f},{recs[0].get('n_clauses_total', 0)},"
                    f"{recs[0].get('n_automata', 0)/1e6:.3f}")
        macros += [f"{macro_name(a, 'acc')}{{{m:.2f}}}", f"{macro_name(a, 'std')}{{{sd:.2f}}}",
                   f"{macro_name(a, 'wall')}{{{wm:.0f}}}"]
    csv("arms.csv", "arm,label,n_seeds,acc_mean,acc_std,wall_min,clauses,automata", rows)
    macros.append(f"\\nArms{{{len(rows)}}}")
    # The best arm introduced by THIS report -- what the three repairs actually reached.
    new_have = [a for a in have if a in NEW and not a.startswith("flat-")]
    if new_have:
        best_new = max(new_have, key=lambda a: agg([r["best_test_acc"] for r in arms[a]])[0])
        bm, bs = agg([r["best_test_acc"] * 100 for r in arms[best_new]])
        macros += [f"\\accBestNew{{{bm:.2f}}}", f"\\stdBestNew{{{bs:.2f}}}",
                   f"\\labelBestNew{{{tex(LABEL[best_new])}}}"]

    # ---- learning curves (one file per arm; pgfplots string filters are fragile) ----
    for a in have:
        log = arms[a][0]["log"]
        csv(f"curve_{a}.csv", "epoch,test_acc",
            [f"{e['epoch']},{e['test_acc']*100:.3f}" for e in log])

    # ---- layer-1 statistics by objective: the quantity E0 said was binding ----------
    #      "before" is the layer as its objective produced it; "after" is post-controller.
    l1_rows, seen = [], set()
    for a in have:
        r = arms[a][0]
        pre = r.get("l1_before")
        if not pre:
            continue
        key = (r.get("l1_kind"), r.get("calib"))
        if key in seen:
            continue
        seen.add(key)
        post = r.get("l1_after")
        c = pre["composition"]
        l1_rows.append(
            f"{a},{LABEL.get(a, a).replace(',', ' /')},{r.get('l1_kind')},"
            f"{pre['firing']['median']*100:.2f},{pre['firing']['frac_dead']*100:.1f},"
            f"{c['size_median']:.0f},{c['neg_fraction_mean']:.3f},"
            f"{(post['firing']['median']*100 if post else pre['firing']['median']*100):.2f}")
    csv("l1_stats.csv", "arm,label,kind,fire,dead,size,neg_frac,fire_post", l1_rows)

    # Per-objective layer-1 facts, for the inline text. "Pre" is the layer as its own
    # objective left it, before any controller touched it.
    KIND_ARM = {"greedy": "mctm-calib-r02", "auto": "mctm-auto", "random": "mctm-random-calib"}
    for kind, a in KIND_ARM.items():
        pre = arms.get(a, [{}])[0].get("l1_before")
        if not pre:
            continue
        cap = kind.capitalize()
        macros += [f"\\fire{cap}Pre{{{pre['firing']['median']*100:.2f}}}",
                   f"\\dead{cap}Pre{{{pre['firing']['frac_dead']*100:.1f}}}",
                   f"\\size{cap}Pre{{{pre['composition']['size_median']:.0f}}}",
                   f"\\orDensity{cap}{{{pre.get('or_density', 0)*100:.1f}}}"]
    greedy = arms.get("mctm-calib-r02", [{}])[0].get("l1_before")
    if greedy:
        macros.append(f"\\densityGreedySize{{{greedy['composition']['size_median']:.0f}}}")

    # ---- E3: the autoencoder ------------------------------------------------------
    for a in ("mctm-auto", "flat-auto"):
        ae = None
        for r in arms.get(a, [])[:1]:
            ae = r.get("ae")
        if ae:
            csv("ae_curve.csv", "epoch,recon_acc,clause_size",
                [f"{e['epoch']},{e['recon_acc']*100:.3f},{e['clause_size']:.1f}"
                 for e in ae["log"]])
            macros += [f"\\aeRecon{{{ae['recon_acc']*100:.1f}}}",
                       f"\\aeBaseline{{{ae['recon_baseline']*100:.1f}}}",
                       f"\\aePatches{{{ae['n_patches']//1000}}}",
                       f"\\aeContext{{{ae['n_context']}}}", f"\\aeTarget{{{ae['n_target']}}}",
                       f"\\aeEpochs{{{ae['epochs']}}}"]
            break

    # ---- E4: the calibration sweep ------------------------------------------------
    sw = []
    for a in CALIB_SWEEP:
        if a not in arms:
            continue
        recs = arms[a]
        m, sd = agg([r["best_test_acc"] * 100 for r in recs])
        r0 = recs[0]
        cal, post = r0.get("calibration", {}), r0.get("l1_after", {})
        sw.append(f"{CALIB_TARGET[a]},{m:.2f},{sd:.2f},"
                  f"{post.get('firing', {}).get('median', 0)*100:.2f},"
                  f"{cal.get('rate_median', 0)*100:.2f},{cal.get('size_median', 0):.0f},"
                  f"{cal.get('frac_in_band', 0):.2f},{cal.get('rounds', 0)}")
        macros += [f"{macro_name(a, 'calibFire')}{{{post.get('firing', {}).get('median', 0)*100:.2f}}}",
                   f"{macro_name(a, 'calibSize')}{{{cal.get('size_median', 0):.0f}}}"]
    csv("calib_sweep.csv", "target,acc,sd,fire_pooled,fire_raw,size,in_band,rounds", sw)
    run_sweep = [a for a in CALIB_SWEEP if a in arms]
    if run_sweep:
        best = max(run_sweep, key=lambda a: agg([r["best_test_acc"] for r in arms[a]])[0])
        bm, _ = agg([r["best_test_acc"] * 100 for r in arms[best]])
        cal = arms[best][0].get("calibration", {})
        macros += [f"\\densityBestTarget{{{CALIB_TARGET[best]:g}}}",
                   f"\\densityBestAcc{{{bm:.2f}}}",
                   f"\\densityCalibSize{{{cal.get('size_median', 0):.0f}}}",
                   f"\\densityCalibPooled{{{arms[best][0].get('l1_after', {}).get('firing', {}).get('median', 0)*100:.2f}}}"]

    sz = []
    for a in ["mctm-calib-r20-tight"] + SIZE_SWEEP + ["mctm-auto-calib"]:
        if a not in arms:
            continue
        m, sd = agg([r["best_test_acc"] * 100 for r in arms[a]])
        r0 = arms[a][0]
        cal, post = r0.get("calibration", {}), r0.get("l1_after", {})
        if a in SIZE_SWEEP or a == "mctm-calib-r20-tight":
            sz.append(f"{SIZE_TARGET.get(a, 'none')},{cal.get('size_median', 0):.0f},"
                       f"{cal.get('rate_median', 0)*100:.2f},"
                       f"{post.get('firing', {}).get('median', 0)*100:.2f},"
                       f"{post.get('firing', {}).get('frac_dead', 0)*100:.1f},"
                       f"{cal.get('frac_in_band', 0):.2f},{m:.2f},{sd:.2f}")
        macros += [f"{macro_name(a, 'calibFire')}{{{post.get('firing', {}).get('median', 0)*100:.2f}}}",
                   f"{macro_name(a, 'calibSize')}{{{cal.get('size_median', 0):.0f}}}",
                   f"{macro_name(a, 'calibRaw')}{{{cal.get('rate_median', 0)*100:.2f}}}",
                   f"{macro_name(a, 'inband')}{{{cal.get('frac_in_band', 0):.2f}}}"]
    csv("size_sweep.csv", "target,size,fire_raw,fire_pooled,dead,in_band,acc,sd", sz)
    for a in ("mctm-credit-cal",):
        if a in arms:
            macros.append(f"\\creditWarmEpochs{{{arms[a][0].get('warmup', 0)}}}")
    run_size = [a for a in SIZE_SWEEP if a in arms]
    if run_size:
        b = max(run_size, key=lambda a: agg([r["best_test_acc"] for r in arms[a]])[0])
        bm, bsd = agg([r["best_test_acc"] * 100 for r in arms[b]])
        macros += [f"\\sizeBestTarget{{{SIZE_TARGET[b]}}}", f"\\sizeBestAcc{{{bm:.2f}}}",
                   f"\\sizeBestSd{{{bsd:.2f}}}"]

    rc = []
    for a in RAND_SWEEP:
        if a not in arms:
            continue
        m, sd = agg([r["best_test_acc"] * 100 for r in arms[a]])
        r0 = arms[a][0]
        cal, post = r0.get("calibration", {}), r0.get("l1_after", {})
        rc.append(f"{RAND_TARGET[a]},{cal.get('size_median', 0):.1f},"
                  f"{cal.get('rate_median', 0)*100:.2f},"
                  f"{post.get('firing', {}).get('median', 0)*100:.2f},"
                  f"{cal.get('frac_in_band', 0):.2f},{m:.2f},{sd:.2f}")
        macros += [f"{macro_name(a, 'calibFire')}{{{post.get('firing', {}).get('median', 0)*100:.2f}}}",
                   f"{macro_name(a, 'calibSize')}{{{cal.get('size_median', 0):.1f}}}"]
    csv("rand_sweep.csv", "target,size,fire_raw,fire_pooled,in_band,acc,sd", rc)
    run_rc = [a for a in RAND_SWEEP if a in arms]
    if run_rc:
        b = max(run_rc, key=lambda a: agg([r["best_test_acc"] for r in arms[a]])[0])
        bm, bsd = agg([r["best_test_acc"] * 100 for r in arms[b]])
        macros += [f"\\randBestTarget{{{RAND_TARGET[b]}}}", f"\\randBestAcc{{{bm:.2f}}}",
                   f"\\randBestSd{{{bsd:.2f}}}"]

    # ---- E5: the credit ladder ----------------------------------------------------
    cr = []
    for a in CREDIT_RUNGS:
        if a not in arms:
            continue
        r0 = arms[a][0]
        m, sd = agg([r["best_test_acc"] * 100 for r in arms[a]])
        st = r0.get("credit_stats", {})
        last = r0["log"][-1]
        fm, fsd = agg([r["final_test_acc"] * 100 for r in arms[a]])
        macros += [f"{macro_name(a, 'fin')}{{{fm:.2f}}}"]
        cr.append(f"{a},{LABEL.get(a, a).replace(',', ' /')},{m:.2f},{sd:.2f},{fm:.2f},"
                  f"{r0.get('credit_rate', 1.0)},{r0.get('warmup', 0)},"
                  f"{thou(st.get('kept_i', 0))},{thou(st.get('kept_ib', 0))},"
                  f"{thou(st.get('kept_ii', 0))},"
                  f"{last.get('l1_size_median', 0):.0f},{last.get('l1_fire_median', 0)*100:.3f},"
                  f"{r0.get('warmup_acc', 0)*100:.2f}")
        macros += [f"{macro_name(a, 'events')}{{{st.get('kept_i', 0):,}}}".replace(",", r"\,"),
                   f"{macro_name(a, 'eventsIb')}{{{st.get('kept_ib', 0):,}}}".replace(",", r"\,"),
                   f"{macro_name(a, 'lOneSize')}{{{last.get('l1_size_median', 0):.0f}}}",
                   f"{macro_name(a, 'lOneFire')}{{{last.get('l1_fire_median', 0)*100:.2f}}}"]
        post = r0.get("l1_after", {}).get("firing")
        if post:
            macros.append(f"{macro_name(a, 'lOneDead')}{{{post['frac_dead']*100:.0f}}}")
        # epoch at which the credit path stopped delivering new Type Ia events
        ev = [e.get("credit", {}).get("kept_i", 0) for e in r0["log"]]
        stop = next((i + 1 for i in range(len(ev) - 1) if ev[i] == ev[-1]), len(ev))
        macros.append(f"{macro_name(a, 'lastEvent')}{{{stop}}}")
        if r0.get("warmup_acc"):
            macros.append(f"{macro_name(a, 'warm')}{{{r0['warmup_acc']*100:.2f}}}")
        # per-epoch trace of the layer-1 statistics under credit
        csv(f"credit_trace_{a}.csv", "epoch,test_acc,l1_size,l1_fire",
            [f"{e['epoch']},{e['test_acc']*100:.3f},{e.get('l1_size_median', 0):.1f},"
             f"{e.get('l1_fire_median', 0)*100:.4f}" for e in r0["log"]])
    csv("credit_rungs.csv",
        "arm,label,acc,sd,final,rate,warmup,ev_i,ev_ib,ev_ii,l1_size,l1_fire,warm_acc", cr)

    # ---- clause composition of the final layer -------------------------------------
    comp = []
    for a in have:
        c = arms[a][0].get("composition")
        if not c:
            continue
        comp.append(f"{a},{LABEL.get(a, a).replace(',', ' /')},{c['size_median']:.0f},"
                    f"{c['pos_median']:.0f},{c['neg_median']:.0f},"
                    f"{c['neg_fraction_mean']:.3f},{c['frac_empty']:.3f}")
        macros += [f"{macro_name(a, 'neg')}{{{c['neg_fraction_mean']*100:.1f}}}",
                   f"{macro_name(a, 'size')}{{{c['size_median']:.0f}}}",
                   f"{macro_name(a, 'pos')}{{{c['pos_median']:.0f}}}"]
    csv("composition.csv", "arm,label,size,pos,neg,neg_frac,empty", comp)

    # ---- E6: the compositional tasks ------------------------------------------------
    syn: List[dict] = []
    for p in ([os.path.join(RESULTS, fn) for fn in ("synthetic.json", "synthetic_oracle64.json")]
              + sorted(glob.glob(os.path.join(RESULTS, "synthetic_ideas*.json")))):
        if os.path.exists(p):
            with open(p) as f:
                syn.extend(json.load(f))
    SYN = {("single", "task"): ("single layer", 0),
           ("flat-stack", "task"): ("flat stack / greedy L1", 1),
           ("mctm", "task"): ("MCTM / greedy L1", 2),
           ("mctm", "oracle64"): ("MCTM / oracle L1 (64 ch)", 3),
           ("auto", "auto"): ("A: MCTM / autoencoder L1", 4),
           ("auto-calib", "auto"): ("A+B: autoencoder + calibrated", 5),
           ("calib", "task"): ("B: greedy L1 + calibrated", 6),
           ("random", "random"): ("MCTM / random L1", 7),
           ("credit-ia", "random"): ("C: credit / Ia+II only", 8),
           ("credit", "random"): ("C: credit / balanced", 9),
           ("credit-warm", "task"): ("C: warm start / step 1.0", 10),
           ("credit-warm-p1", "task"): ("C: warm start / step 0.1", 11),
           ("credit-cal", "task"): ("B+C: warm start + controller", 12)}
    groups: Dict[tuple, List[dict]] = {}
    for r in syn:
        groups.setdefault((r["task"], r["arm"], r.get("l1_mode", "task")), []).append(r)
    for task in ("xor", "near"):
        rows = []
        for (arm, mode), (label, o) in sorted(SYN.items(), key=lambda kv: kv[1][1]):
            recs = groups.get((task, arm, mode))
            if not recs:
                continue
            m, sd = agg([r["best_test_acc"] * 100 for r in recs])
            # Layer-1 firing rate is what E4 says should decide these tasks, so it travels
            # with the accuracy rather than being argued about separately.
            f0 = recs[0].get("l1_firing") or recs[0].get("l1_firing_pre") or {}
            fire = f"{f0['median']*100:.2f}" if f0 else "--"
            dead = f"{f0['frac_dead']*100:.0f}" if f0 else "--"
            # What layer 2 ended up being. The proposal predicted a small conjunction over
            # channels; the first report measured a thousand-literal negation soup instead.
            c2 = recs[0].get("composition") or {}
            l2s = f"{c2['size_median']:.0f}" if c2 else "--"
            l2p = f"{c2['pos_median']:.0f}" if c2 else "--"
            rows.append((o, f"{label.replace(',', ' /')},{m:.2f},{sd:.2f},"
                            f"{fire},{dead},{l2s},{l2p},{len(recs)}"))
            mac = "".join(w.capitalize() for w in f"{task}-{arm}-{mode}".split("-"))
            mac = "".join(_DIGITS.get(c, c) for c in mac)
            macros += [f"\\syn{mac}{{{m:.1f}}}", f"\\synsd{mac}{{{sd:.1f}}}"]
            if f0:
                macros += [f"\\synfire{mac}{{{f0['median']*100:.2f}}}",
                           f"\\syndead{mac}{{{f0['frac_dead']*100:.0f}}}"]
            if c2:
                macros += [f"\\synltwo{mac}{{{c2['size_median']:.0f}}}",
                           f"\\synltwopos{mac}{{{c2['pos_median']:.0f}}}",
                           f"\\synltwoneg{mac}{{{c2['neg_fraction_mean']*100:.0f}}}"]
        csv(f"syn_{task}.csv", "label,acc,sd,fire,dead,l2size,l2pos,seeds",
            [r for _, r in sorted(rows)])

    # ---- facts -------------------------------------------------------------------
    any_rec = arms[have[0]][0] if have else {}
    hp = any_rec.get("hp", {})
    macros += [f"\\nTrain{{{any_rec.get('n_train', 0):,}}}".replace(",", r"\,"),
               f"\\nTest{{{any_rec.get('n_test', 0):,}}}".replace(",", r"\,"),
               f"\\cOne{{{hp.get('C1', 0)}}}", f"\\cTwo{{{hp.get('C2', 0)}}}",
               f"\\pOne{{{hp.get('P1', 0)}}}", f"\\pTwo{{{hp.get('P2', 0)}}}",
               f"\\poolK{{{hp.get('POOL', 0)}}}",
               f"\\epochsRun{{{any_rec.get('epochs', 0)}}}",
               f"\\batchSize{{{any_rec.get('batch_size', 0)}}}",
               f"\\nBits{{{any_rec.get('n_bits', 0)}}}"]
    import torchtsetlin as tt
    macros.append(f"\\ttversionreport{{{tt.__version__}}}")

    # An arm that has not been run yet still needs its macros defined, or pdflatex drops the
    # value silently and the sentence reads as if a number were never there.
    defined = {m.split("{")[0] for m in macros}
    for cap in ("Greedy", "Auto", "Random"):
        for field in ("fire", "dead", "size", "orDensity"):
            n = f"\\{field}{cap}" + ("" if field == "orDensity" else "Pre")
            if n not in defined:
                macros.append(f"{n}{{n/a}}")
    for n in ("\\nArms", "\\randBestTarget", "\\randBestAcc", "\\randBestSd", "\\creditWarmEpochs", "\\sizeBestTarget", "\\sizeBestAcc", "\\sizeBestSd", "\\densityBestTarget", "\\densityBestAcc", "\\densityCalibSize",
              "\\densityCalibPooled", "\\densityGreedySize", "\\aeRecon",
              "\\aeBaseline", "\\aePatches", "\\aeContext", "\\aeTarget",
              "\\aeEpochs"):
        if n not in defined:
            macros.append(f"{n}{{n/a}}")
    for a in ORDER:
        for field in ("acc", "std", "wall", "neg", "size", "pos", "fire", "events",
                      "eventsIb", "lOneSize", "lOneFire", "lOneDead", "lastEvent",
                      "warm", "calibFire", "calibSize", "calibRaw", "inband", "fin"):
            n = macro_name(a, field)
            if n not in defined:
                macros.append(f"{n}{{n/a}}")
    names = [m.split("{")[0] for m in macros]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f"duplicate macro definitions (the first would silently win): {dup}")
    with open(os.path.join(HERE, "report2", "macros.tex"), "w") as f:
        f.write("% generated by report_data_ideas.py -- do not edit\n")
        for m in macros:
            f.write(f"\\newcommand{m}\n")
    print(f"wrote {len(have)} arms, {len(macros)} macros -> report2/")
    for a in have:
        m, sd = agg([r["best_test_acc"] * 100 for r in arms[a]])
        print(f"  {LABEL.get(a, a):<38} {m:6.2f} +- {sd:4.2f}  ({len(arms[a])} seeds)")


if __name__ == "__main__":
    main()
