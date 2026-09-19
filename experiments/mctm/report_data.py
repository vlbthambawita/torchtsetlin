"""Turn results/*.json into the CSVs and \newcommand macros the LaTeX report reads.

Nothing in the report is hand-typed: every number in the text comes from macros.tex and
every figure/table from data/*.csv, so re-running the experiments and re-running this
script updates the PDF wholesale.
"""
from __future__ import annotations

import glob, json, os, statistics as stats
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "report", "data")

ARM_LABEL = {
    "single-l1": "Layer 1 alone",
    "single-matched": "Single layer / matched budget",
    "single-rf": "Single layer / 9x9 patch (matched RF)",
    "flat-stack": "Flat stack (global OR + flat TM)",
    "mctm": "MCTM (2 layers)",
    "mctm-random": "MCTM / random layer 1",
    "mctm-nopool": "MCTM / no OR-pool",
    "mctm-nopos": "MCTM / no position encoding",
    "mctm-pool4": "MCTM / 4x4 OR-pool",
    "mctm-dense": "MCTM / dense layer 1 (s=2)",
}

ARM_ORDER = ["single-l1", "single-matched", "single-rf", "flat-stack",
             "mctm", "mctm-dense", "mctm-random", "mctm-nopool", "mctm-nopos",
             "mctm-pool4"]


def load_arms() -> Dict[str, List[dict]]:
    arms: Dict[str, List[dict]] = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*_seed*.json"))):
        with open(p) as f:
            r = json.load(f)
        arms.setdefault(r["arm"] + r.get("tag", ""), []).append(r)
    return arms


def agg(vals: List[float]):
    if not vals:
        return 0.0, 0.0
    return (stats.mean(vals), stats.stdev(vals) if len(vals) > 1 else 0.0)


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
           "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_name(arm: str, field: str) -> str:
    """LaTeX macro names may only contain letters, so digits become words."""
    core = "".join(w.capitalize() for w in arm.replace("_", "-").split("-"))
    core = "".join(_DIGITS.get(ch, ch) for ch in core)
    return f"\\{field}{core}"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    arms = load_arms()
    macros: List[str] = []

    # ---- main results table ------------------------------------------------------
    rows = []
    for arm in ARM_ORDER:
        if arm not in arms:
            continue
        recs = arms[arm]
        acc_m, acc_s = agg([r["best_test_acc"] * 100 for r in recs])
        # Layer 1 is cached across arms, so wall_s understates cost for cached runs.
        # Sum the per-epoch times actually recorded in both stages instead.
        def train_s(r):
            return (sum(e["epoch_s"] for e in r.get("log_l1", []))
                    + sum(e["epoch_s"] for e in r.get("log", [])))
        wall_m, _ = agg([train_s(r) / 60 for r in recs])
        rows.append({
            "arm": arm, "label": ARM_LABEL.get(arm, arm), "n_seeds": len(recs),
            "acc_mean": acc_m, "acc_std": acc_s, "wall_min": wall_m,
            "clauses": recs[0].get("n_clauses_total", 0),
            "automata": recs[0].get("n_automata", 0) / 1e6,
        })
        macros += [f"{macro_name(arm, 'acc')}{{{acc_m:.2f}}}",
                   f"{macro_name(arm, 'std')}{{{acc_s:.2f}}}",
                   f"{macro_name(arm, 'wall')}{{{wall_m:.0f}}}"]
    with open(os.path.join(OUT, "arms.csv"), "w") as f:
        f.write("arm,label,n_seeds,acc_mean,acc_std,wall_min,clauses,automata\n")
        for r in rows:
            r["label"] = r["label"].replace(",", " /")  # col sep=comma
            f.write(f"{r['arm']},{r['label']},{r['n_seeds']},{r['acc_mean']:.3f},"
                    f"{r['acc_std']:.3f},{r['wall_min']:.2f},{r['clauses']},{r['automata']:.3f}\n")

    # ---- learning curves (one file per arm: pgfplots string filters are fragile) ----
    for arm in arms:
        recs = arms[arm]
        with open(os.path.join(OUT, f"curve_{arm}.csv"), "w") as f:
            f.write("epoch,test_acc\n")
            for e in recs[0]["log"]:
                f.write(f"{e['epoch']},{e['test_acc']*100:.3f}\n")

    # ---- density sweep -----------------------------------------------------------
    dpath = os.path.join(RESULTS, "density_sweep.json")
    if os.path.exists(dpath):
        with open(dpath) as f:
            d = json.load(f)
        with open(os.path.join(OUT, "density.csv"), "w") as f:
            f.write("idx,label,s,budget,batch,mode,comparable,l1_acc,includes,"
                    "fire1,fire2,fire4,dead2\n")
            for i, r in enumerate(d["rows"]):
                fi = r["firing"]
                label = r["label"].replace(",", ";")  # col sep=comma
                f.write(f"{i},{label},{r['s']},"
                        f"{r['max_included_literals'] if r['max_included_literals'] else 'none'},"
                        f"{r['batch_size']},{r['feedback_mode']},{int(r['comparable_acc'])},"
                        f"{r['l1_test_acc']*100:.3f},{r['include_count']['median']},"
                        f"{fi['pool1']['median']:.6f},{fi['pool2']['median']:.6f},"
                        f"{fi['pool4']['median']:.6f},{fi['pool2']['frac_dead']:.4f}\n")
        # per-channel firing curve for the densest comparable setting
        # The headline density/accuracy trade-off is the pure s-sweep; the budget rows are
        # diagnostics of a different mechanism and would otherwise win "sparsest" trivially.
        comp = [r for r in d["rows"]
                if r["comparable_acc"] and r["max_included_literals"] is None]
        densest = max(comp, key=lambda r: r["firing"]["pool2"]["median"])
        with open(os.path.join(OUT, "firing_curve.csv"), "w") as f:
            f.write("channel,rate\n")
            for i, v in enumerate(sorted(densest["per_channel_pool2"], reverse=True)):
                f.write(f"{i},{v:.6f}\n")
        best_acc = max(comp, key=lambda r: r["l1_test_acc"])
        sparsest = min(comp, key=lambda r: r["firing"]["pool2"]["median"])
        budget = {r["label"]: r for r in d["rows"]}
        macros += [
            f"\\densityBestLabel{{{best_acc['label']}}}",
            f"\\densityBestAcc{{{best_acc['l1_test_acc']*100:.2f}}}",
            f"\\densityDensestLabel{{{densest['label']}}}",
            f"\\densityDensestFire{{{densest['firing']['pool2']['median']*100:.3f}}}",
            f"\\densityDensestFireRaw{{{densest['firing']['pool1']['median']*100:.3f}}}",
            f"\\densityDensestFirePoolFour{{{densest['firing']['pool4']['median']*100:.3f}}}",
            f"\\densityDensestAcc{{{densest['l1_test_acc']*100:.2f}}}",
            f"\\densityDensestIncludes{{{densest['include_count']['median']}}}",
            f"\\densitySparsestLabel{{{sparsest['label']}}}",
            f"\\densitySparsestFire{{{sparsest['firing']['pool2']['median']*100:.4f}}}",
            f"\\densitySparsestDead{{{sparsest['firing']['pool2']['frac_dead']*100:.0f}}}",
            f"\\densityEpochs{{{d['epochs']}}}",
            f"\\densitySubset{{{d['subset']}}}",
        ]
        for key, mac in (("s=10, b=8, bs=50", "BudgetBatchFifty"),
                         ("s=10, b=8, bs=5", "BudgetBatchFive"),
                         ("s=10, b=8, seq", "BudgetSeq")):
            if key in budget:
                macros.append(f"\\inc{mac}{{{budget[key]['include_count']['median']}}}")

    # ---- clause composition: how much of a clause is negations -----------------------
    comp_rows = []
    for arm in ARM_ORDER:
        for r in arms.get(arm, [])[:1]:
            c = r.get("composition")
            if not c:
                continue
            comp_rows.append((arm, ARM_LABEL.get(arm, arm), c, r.get("composition_l1")))
    if comp_rows:
        with open(os.path.join(OUT, "composition.csv"), "w") as f:
            f.write("arm,label,size,pos,neg,neg_frac,empty\n")
            for arm, label, c, _ in comp_rows:
                label = label.replace(",", " /")
                f.write(f"{arm},{label},{c['size_median']:.0f},{c['pos_median']:.0f},"
                        f"{c['neg_median']:.0f},{c['neg_fraction_mean']:.3f},"
                        f"{c['frac_empty']:.3f}\n")
        for arm, _, c, c1 in comp_rows:
            macros.append(f"{macro_name(arm, 'neg')}{{{c['neg_fraction_mean']*100:.1f}}}")
            macros.append(f"{macro_name(arm, 'size')}{{{c['size_median']:.0f}}}")
            if c1:
                macros.append(f"{macro_name(arm, 'negLOne')}{{{c1['neg_fraction_mean']*100:.1f}}}")

    # ---- per-channel firing histogram of the chosen layer 1 -----------------------
    for arm in ("mctm", "mctm-random"):
        for r in arms.get(arm, [])[:1]:
            fs = r.get("l1_firing")
            if not fs:
                continue
            with open(os.path.join(OUT, f"firing_{arm}.csv"), "w") as f:
                f.write("channel,rate\n")
                for i, v in enumerate(sorted(fs["per_channel"], reverse=True)):
                    f.write(f"{i},{v:.6f}\n")
            macros.append(f"{macro_name(arm, 'fire')}{{{fs['median']*100:.2f}}}")

    # ---- E2 synthetic compositional tasks --------------------------------------------
    spath = os.path.join(RESULTS, "synthetic.json")
    if os.path.exists(spath):
        syn = []
        for fn in ("synthetic.json", "synthetic_oracle64.json"):
            fp = os.path.join(RESULTS, fn)
            if os.path.exists(fp):
                with open(fp) as f:
                    syn.extend(json.load(f))
        groups: Dict[tuple, List[dict]] = {}
        for r in syn:
            groups.setdefault((r["task"], r["arm"], r["l1_mode"]), []).append(r)
        SYN_LABEL = {("single", "task"): "single layer",
                     ("flat-stack", "task"): "flat stack / greedy L1",
                     ("flat-stack", "oracle"): "flat stack / oracle L1 (2 ch)",
                     ("flat-stack", "oracle64"): "flat stack / oracle L1 (64 ch)",
                     ("mctm", "task"): "MCTM / greedy L1",
                     ("mctm", "oracle"): "MCTM / oracle L1 (2 ch)",
                     ("mctm", "oracle64"): "MCTM / oracle L1 (64 ch)"}
        order = [("single", "task"), ("flat-stack", "task"), ("mctm", "task"),
                 ("flat-stack", "oracle"), ("mctm", "oracle"),
                 ("flat-stack", "oracle64"), ("mctm", "oracle64")]
        for task in ("xor", "near"):
            syn_rows = []
            for arm, mode in order:
                recs = groups.get((task, arm, mode))
                if not recs:
                    continue
                m, sd = agg([r["best_test_acc"] * 100 for r in recs])
                syn_rows.append((SYN_LABEL[(arm, mode)], m, sd, len(recs)))
                mac = "".join(w.capitalize() for w in f"{task}-{arm}-{mode}".split("-"))
                mac = "".join(_DIGITS.get(c, c) for c in mac)
                macros += [f"\\syn{mac}{{{m:.1f}}}", f"\\synsd{mac}{{{sd:.1f}}}"]
            if syn_rows:
                with open(os.path.join(OUT, f"syn_{task}.csv"), "w") as f:
                    f.write("label,acc,sd,seeds\n")
                    for lab, m, sd, k in syn_rows:
                        f.write(f"{lab},{m:.2f},{sd:.2f},{k}\n")

    # ---- misc facts --------------------------------------------------------------
    any_rec = next(iter(next(iter(arms.values()), [{}])), {}) if arms else {}
    if any_rec:
        hp = any_rec.get("hp", {})
        macros += [f"\\nTrain{{{any_rec.get('n_train', 0):,}}}".replace(",", r"\,"),
                   f"\\nTest{{{any_rec.get('n_test', 0):,}}}".replace(",", r"\,"),
                   f"\\nBits{{{any_rec.get('n_bits', 0)}}}",
                   f"\\inShape{{{'x'.join(str(v) for v in any_rec.get('input_shape', []))}}}",
                   f"\\cOne{{{hp.get('C1', 0)}}}", f"\\cTwo{{{hp.get('C2', 0)}}}",
                   f"\\pOne{{{hp.get('P1', 0)}}}", f"\\pTwo{{{hp.get('P2', 0)}}}",
                   f"\\poolK{{{hp.get('POOL', 0)}}}",
                   f"\\epochsRun{{{any_rec.get('epochs', 0)}}}",
                   f"\\batchSize{{{any_rec.get('batch_size', 0)}}}"]

    import torchtsetlin as tt
    macros.append(f"\\ttversionreport{{{tt.__version__}}}")

    # Any arm that has not been run yet still needs its macros defined, or pdflatex drops the
    # value silently and the sentence reads as if a number were never there. Emit an explicit
    # placeholder instead so a partial run is visible in the PDF rather than invisible.
    defined = {m.split("{")[0] for m in macros}
    for arm in ARM_ORDER:
        for field in ("acc", "std", "wall", "neg", "size", "fire"):
            name = macro_name(arm, field)
            if name not in defined:
                macros.append(f"{name}{{n/a}}")

    with open(os.path.join(HERE, "report", "macros.tex"), "w") as f:
        f.write("% generated by report_data.py -- do not edit\n")
        for m in macros:
            f.write(f"\\newcommand{m}\n")
    print(f"wrote {len(rows)} arms, {len(macros)} macros -> report/")
    for r in rows:
        print(f"  {r['label']:<42} {r['acc_mean']:6.2f} +- {r['acc_std']:.2f}  "
              f"({r['n_seeds']} seeds, {r['wall_min']:.0f} min)")


if __name__ == "__main__":
    main()
