"""Generate REPORT.md from the records only (rule C2). Figures go to report_figures/.

    python -m experiments.transformer_concepts.code.report
"""
from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from experiments.transformer_concepts.code.exp_main import (
    FINAL_SEEDS,
    TASKS,
    arms_for,
    length,
)
from experiments.transformer_concepts.code.exp_threshold import LS, SS, s_pred
from experiments.transformer_concepts.code.runner import RESULTS, ROOT, make_data

FIG = ROOT / "report_figures"
BASELINES = ("B0_flat", "B1_conv_k1", "B2_conv_k3_pe", "B4_graph_chain")
LABEL = {
    "B0_flat": "B0 flat TM", "B1_conv_k1": "B1 conv k=1", "B2_conv_k3_pe": "B2 conv k=3 + PE",
    "B3_conv_kd1": "B3 conv k=d+1", "B4_graph_chain": "B4 graph chain (depth 2)",
    "P_onehot": "P one-hot PE", "P_therm": "P thermometer PE", "P_binary": "P binary PE", "P_gray": "P Gray PE",
    "A1_pair_w1": "A1 pair w=1", "A2_pair_w2": "**A2 pair w=2**", "A3_pair_bucket": "A3 pair, bucketed",
    "A4_graph_complete": "A4 graph complete (depth 2)", "A2_self": "A2-self (i=j only)",
    "A2_norel": "A2-norel", "A2_abs": "A2-abs (absolute codes)", "A2_shuf": "A2-shuf (shuffled rel.)",
}
TASK_NAME = {"T1": "T1 ordered pair", "T2": "T2 associative recall", "T3": "T3 repeat detection",
             "T4": "T4 needle (d = L/4)", "T5": "T5 co-occurrence (0 and 1 both present)",
             "T6": "T6 absolute position (control)", "T7": "T7 presence of symbol 0 (control)"}


def load(stage):
    out = {}
    for p in sorted((RESULTS / stage).glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- statistics
def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * cdf)


def compare(final, task, L, a, b, y):
    """Paired comparison of arms a and b over the seeds both have. Returns a dict or None."""
    diffs, ps, pa_all, pb_all = [], [], [], []
    for seed in FINAL_SEEDS:
        ka, kb = f"{task}_L{L}_{a}_seed{seed}", f"{task}_L{L}_{b}_seed{seed}"
        fa, fb = RESULTS / "final" / f"{ka}.pred.npy", RESULTS / "final" / f"{kb}.pred.npy"
        if not (fa.exists() and fb.exists()):
            continue
        pa, pb = np.load(fa) == y, np.load(fb) == y
        diffs.append(100 * (pa.mean() - pb.mean()))
        ps.append(mcnemar_p(int((pa & ~pb).sum()), int((~pa & pb).sum())))
        pa_all.append(pa)
        pb_all.append(pb)
    n = len(diffs)
    if n < 2:
        return None
    da = np.array(diffs)
    sa = np.std([100 * p.mean() for p in pa_all], ddof=1)
    sb = np.std([100 * p.mean() for p in pb_all], ddof=1)
    band = 2 * max(sa, sb)
    rng = np.random.default_rng(0)
    A, B = np.stack(pa_all).astype(float), np.stack(pb_all).astype(float)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, A.shape[1], A.shape[1])
        boots.append(100 * (A[:, idx].mean() - B[:, idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    need = math.ceil(0.8 * n)
    sig = sum(p < 0.05 for p in ps)

    def rule(sign):
        return (all(sign * d > 0 for d in da) and sign * da.mean() >= 0.5 and abs(da.mean()) > band
                and sig >= need)
    verdict = "supported" if rule(1) else "reverse supported" if rule(-1) else "inconclusive"
    return dict(n=n, mean=float(da.mean()), diffs=[float(d) for d in da], ci=(float(lo), float(hi)),
                sig=sig, verdict=verdict)


# ----------------------------------------------------------------------------- figures
def fig_d1(d1):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for boost, mk in ((True, "o"), (False, "s")):
        c = [x for x in d1["cells"] if x["boost"] == boost]
        ax.errorbar([x["pred_f1"] for x in c], [x["ratio_f1"] for x in c], yerr=[3 * x["se_f1"] for x in c],
                    fmt=mk, ms=4, label=f"boost_true_positive={boost}", alpha=0.8)
    ax.plot([-0.6, 1], [-0.6, 1], "k--", lw=0.8)
    ax.set_xlabel("predicted  ρ − (1−ρ)/(s·p_mem)")
    ax.set_ylabel("measured  E[Δ f1] / E[Δ f0]")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "d1_drift.png", dpi=130)
    plt.close(fig)
    return "report_figures/d1_drift.png"


def fig_d2(curves, sstar):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (task, arm, title) in zip(axes, (("T4", "A2_pair_w2", "pair TM (V = 16, distractors)"),
                                           ("T4", "B0_flat", "flat TM (control)"),
                                           ("T4", "B3_conv_kd1", "conv k=d+1 (control)"))):
        for ci, L in enumerate(LS):
            acc = [curves.get((task, arm, L, s)) for s in SS]
            if all(a is None for a in acc):
                continue
            ax.plot(SS, [a if a is not None else np.nan for a in acc], marker="o", ms=3, label=f"L={L}",
                    color=f"C{ci}")
        ax.set_xscale("log")
        ax.axhline(0.9, color="k", lw=0.6, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("s")
    axes[0].set_ylabel("validation accuracy (mean of 2 seeds)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "d2_threshold.png", dpi=130)
    plt.close(fig)
    return "report_figures/d2_threshold.png"


# ----------------------------------------------------------------------------- sections
def part1():
    lines = ["## Part I — exact verification of the formal claims", "",
             "Each check loads hand-built clauses into real torchtsetlin 0.7.0 models (CPU and CUDA, where "
             "CUDA prediction runs the Triton `conv_eval` kernel) or computes the claim exactly on every "
             "instance small enough to enumerate. Records: `results/exact/X*.json`.", "",
             "| check | claims | result | evidence |", "|---|---|---|---|"]
    recs = load("exact")
    for cid in ("X1", "X1b", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12"):
        r = recs.get(cid)
        if r is None:
            lines.append(f"| {cid} | – | **missing** | – |")
            continue
        lines.append(f"| {cid} | {', '.join(r['claims'])} | {'pass' if r['pass'] else '**FAIL**'} | {r['summary']} |")
    x10 = recs.get("X10")
    if x10:
        lines += ["", "**Exact minimum positive DNFs (X10).** Positive-clause flat TM, exact over the whole "
                  "one-hot domain; compared with the proved lower bound and with the pair TM.", "",
                  "| instance | exact minimum | proved lower bound | other reference |", "|---|---|---|---|"]
        for k, v in x10["table"].items():
            if k.startswith("B6.2"):
                lines.append(f"| needle {k.split('_', 2)[2]} | {v['min_clauses']} | L − d = {v['bound']} | pair TM: 1 |")
            elif k.startswith("B7.2"):
                lines.append(f"| recall {k.split('_', 2)[2]} | {v['min_clauses']} | n = {v['bound']} | "
                             f"n·min(keys, 2ⁿ) = {v['n_times_min_keys_2pow_n']}; pair TM: {v['pair_tm_clauses']} |")
            else:
                lines.append(f"| T1 {k.split('_', 2)[2]} | {v['min_clauses']} | ⌈(L−1)/2⌉ = {v['lower_bound']} | "
                             f"negation construction: {v['construction']}; pair TM: 1 |")
    x1 = recs.get("X1")
    if x1:
        lines += ["", "**Interval costs (X1).** Intervals that are a single clause, and the worst-case number of "
                  "clauses over all intervals.", "", "| n | thermometer | one-hot | binary | Gray |", "|---|---|---|---|---|"]
        for n, t in x1["table"].items():
            cell = lambda c, t=t: f"{t[c]['single_clause_intervals']}/{t[c]['intervals']}, max {t[c]['max_clause_cover']}"  # noqa: E731
            lines.append(f"| {n} | {cell('thermometer')} | {cell('onehot')} | {cell('binary')} | {cell('gray')} |")
    return lines


def part2():
    lines = ["## Part II — learning dynamics", ""]
    d1p = RESULTS / "dynamics" / "D1.json"
    if d1p.exists():
        d1 = json.loads(d1p.read_text())
        first = json.loads((RESULTS / "dynamics" / "D1_first_run.json").read_text()) if (RESULTS / "dynamics" / "D1_first_run.json").exists() else None
        lines += ["### D1 — the drawn pair is uniform; the Type Ia drift has the predicted form", "",
                  "One clause including only an always-true feature f0, on examples with m matching columns of which "
                  "r carry f1 (ρ = r/m). The clause row is reset after every library update, and the state drift of f1 "
                  "is compared with that of f0. Prediction: E[Δf1]/E[Δf0] = ρ − (1−ρ)/(s·p_mem), where p_mem = 1 with "
                  "`boost_true_positive=True` (the library default) and (s−1)/s without.", ""]
        if first:
            lines += [f"- **First run (prediction p_mem = (s−1)/s for the default model): {first['n_within_3se']}/"
                      f"{first['n_cells']} cells within 3 SE — failed.** The miss was systematic (at ρ = 0 the "
                      "ratio was −1/s, not −1/(s−1)): the library memorises true literals with probability 1 by "
                      "default. The theory (THEORY C2) assumed the textbook (s−1)/s and was corrected.", ""]
        groups = defaultdict(lambda: [0, 0])
        for c in d1["cells"]:
            g = groups[(c["boost"], c["device"], "sequential" if c["sequential"] else "batch")]
            g[0] += c["within_3se"]
            g[1] += 1
        lines += [f"- **Corrected prediction: {d1['n_within_3se']}/{d1['n_cells']} cells within 3 SE.**", "",
                  "| boost_true_positive | device | mode | cells within 3 SE |", "|---|---|---|---|"]
        for (b, dev, mode), (k, n) in sorted(groups.items()):
            lines.append(f"| {b} | {dev} | {mode} | {k}/{n} |")
        f = fig_d1(d1)
        if f:
            lines += ["", f"![D1]({f})"]
        lines += ["", "Consequence: with the default model a literal's mean drift is towards inclusion iff its rate on "
                  "the drawn pair exceeds 1/(s+1) (1/s without boosting).", ""]
    thr = load("threshold")
    if thr:
        acc = defaultdict(list)
        for r in thr.values():
            acc[(r["task"], r["V"], r["arm"], r["L"], int(r["s"]))].append(r["best_val"])
        cur = {k: float(np.mean(v)) for k, v in acc.items()}
        lines += ["### D2 — the specificity a pair clause needs grows with the number of pairs", "",
                  "Needle task, d = L/4, validation accuracy after ≤ 15 epochs on 8,000 examples, mean of 2 seeds. "
                  "Prediction (written before the runs, `code/exp_threshold.py`): an under-specified pair clause needs "
                  "s > s_pred = 1/ρ − 1, ρ = 1/L + (1 − 1/L)·π, π the distractor rate. Last column: the grid values of s "
                  "at which the mean accuracy is ≥ 0.9.", "",
                  "| task | V | arm | L | s_pred | " + " | ".join(f"s={x}" for x in SS) + " | s with acc ≥ 0.9 |",
                  "|---|---|---|---|---|" + "---|" * len(SS) + "---|"]
        rows = sorted({(k[0], k[1], k[2], k[3]) for k in cur})
        order = {"A2_pair_w2": 0, "B0_flat": 1, "B3_conv_kd1": 2}
        rows.sort(key=lambda r: (r[0] != "T4", r[1], order.get(r[2], 9), r[3]))
        for task, V, arm, L in rows:
            accs = [cur.get((task, V, arm, L, x)) for x in SS]
            good = [str(x) for x, a in zip(SS, accs) if a is not None and a >= 0.9]
            sp = fmt(s_pred(L, V, task == "T4")) if arm == "A2_pair_w2" else "–"
            lines.append(f"| {task} | {V} | {LABEL[arm]} | {L} | {sp} | " + " | ".join(fmt(a, 2) if a is not None else "–" for a in accs)
                         + f" | {', '.join(good) or 'none'} |")
        lines += ["", "T4n = no distractors (degenerate, see DECISIONS D13): one literal [k0 = b] solves it.", ""]
        f = fig_d2({(k[0], k[2], k[3], k[4]): v for k, v in cur.items() if k[1] == 16}, None)
        if f:
            lines += [f"![D2]({f})", ""]
    return lines


def part3():
    lines = ["## Part III — the empirical hypotheses (torchtsetlin_plans/experiments/transformer_concepts/PLAN.md H1–H5)", ""]
    fz_p = RESULTS / "frozen_experiment.json"
    if not fz_p.exists():
        return lines + ["Not run (no frozen experiment)."]
    fz = json.loads(fz_p.read_text())
    final = load("final")
    lines += [f"Search: s ∈ {list(fz['s_grid'])} at L = 32 (T2: 33), seed {fz['search_seed']}, validation only; "
              f"frozen {fz['frozen_at']} (`results/frozen_experiment.json`). Finals: seeds {list(fz['final_seeds'])}, "
              "20,000 train / 5,000 validation / 10,000 test, M = 100 clauses per class, T = 25, weighted, batch 32, "
              "≤ 40 epochs with patience 8. Accuracy is on the test set, mean ± SD over seeds (n = seeds available).", ""]
    # main table
    for task in TASKS:
        lines += [f"### {TASK_NAME[task]}", "", "| arm | s | L = " + str(length(task, 16)) + " | L = " + str(length(task, 32)) + " |", "|---|---|---|---|"]
        for arm in arms_for(task):
            c = fz["chosen"].get(f"{task}|{arm}")
            cells = []
            for L in (length(task, 16), length(task, 32)):
                v = [final[k]["test_acc"] * 100 for k in final
                     if final[k]["task"] == task and final[k]["L"] == L and final[k]["arm"] == arm and "test_acc" in final[k]]
                cells.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f} (n={len(v)})" if v else "–")
            lines.append(f"| {LABEL[arm]} | {int(c['s']) if c else '–'} | {cells[0]} | {cells[1]} |")
        lines.append("")
    # decision rules
    lines += ["### Decision rules (torchtsetlin_plans/experiments/transformer_concepts/PLAN.md §5)", "",
              "Supported = all paired seed differences positive, mean gap ≥ 0.5 pp and above twice the larger seed SD, "
              "McNemar p < 0.05 on ≥ 80 % of seeds. Gap = first arm − second arm, test accuracy, pp.", "",
              "| task | L | comparison | mean gap (pp) | 95 % CI | McNemar p<0.05 | verdict |", "|---|---|---|---|---|---|---|"]
    verdicts = {}
    for task in TASKS:
        for L0 in (16, 32):
            L = length(task, L0)
            y = make_data(task, L, "test")[1].numpy()
            comps = [("A2_pair_w2", b) for b in BASELINES] + [("A2_pair_w2", "A2_self"), ("A2_pair_w2", "A2_norel"),
                                                             ("A2_pair_w2", "A2_abs"), ("A2_pair_w2", "A2_shuf")]
            if task == "T4":
                comps.append(("A2_pair_w2", "B3_conv_kd1"))
            if task in ("T1", "T4", "T6"):
                comps += [("P_therm", "P_onehot"), ("P_therm", "P_binary"), ("P_therm", "P_gray")]
            for a, b in comps:
                r = compare(final, task, L, a, b, y)
                if r is None:
                    continue
                verdicts[(task, L0, a, b)] = r
                lines.append(f"| {task} | {L} | {LABEL[a]} vs {LABEL[b]} | {r['mean']:+.2f} | "
                             f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['sig']}/{r['n']} | {r['verdict']} |")
    lines.append("")
    # hypothesis summary
    lines += ["### Verdicts", ""]
    for task in TASKS:
        for L0 in (16, 32):
            bs = BASELINES + (("B3_conv_kd1",) if task == "T4" else ())
            vs = [verdicts.get((task, L0, "A2_pair_w2", b)) for b in bs]
            if any(v is None for v in vs):
                continue
            h2 = all(v["verdict"] == "supported" for v in vs)
            beats = [b for b, v in zip(bs, vs) if v["verdict"] == "supported"]
            loses = [b for b, v in zip(bs, vs) if v["verdict"] == "reverse supported"]
            a_self = verdicts.get((task, L0, "A2_pair_w2", "A2_self"))
            a_norel = verdicts.get((task, L0, "A2_pair_w2", "A2_norel"))
            v_self = a_self["verdict"] if a_self else "–"
            v_norel = a_norel["verdict"] if a_norel else "–"
            lines.append(f"- **{task}, L = {length(task, L0)}**: A2 beats every baseline: **{'yes' if h2 else 'no'}** "
                         f"(beats {', '.join(beats) or 'none'}; loses to {', '.join(loses) or 'none'}). "
                         f"Cross-token pairs needed (A2 vs A2-self): **{v_self}**; relative code needed "
                         f"(A2 vs A2-norel): **{v_norel}**.")
    lines.append("")
    # K1
    search = load("search")
    lines += ["### Kill criterion K1 (validation, search records)", ""]
    for task in ("T1", "T2", "T4"):
        L = length(task, 32)
        a2 = max((r["best_val"] for r in search.values() if r["task"] == task and r["arm"] == "A2_pair_w2"), default=None)
        sf = max((r["best_val"] for r in search.values() if r["task"] == task and r["arm"] == "A2_self"), default=None)
        if a2 is not None and sf is not None:
            lines.append(f"- {task} (L = {L}): A2 {a2 * 100:.2f} %, A2-self {sf * 100:.2f} %, gap {100 * (a2 - sf):+.2f} pp.")
    lines.append("")
    # H5 curves
    curves = load("curves")
    if curves:
        lines += ["### H5 — sample efficiency on T2 (L = 17)", "", "| arm | n_train = 1,000 | 4,000 | 16,000 |", "|---|---|---|---|"]
        for arm in ("A2_pair_w2", "B0_flat", "A2_self"):
            row = [LABEL[arm]]
            for n in (1000, 4000, 16000):
                v = [r["test_acc"] * 100 for r in curves.values() if r["arm"] == arm and r["n_train"] == n and "test_acc" in r]
                row.append(f"{np.mean(v):.2f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.2f} (n={len(v)})" if v else "–")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    # exploratory (validation only, after the freeze)
    expl = load("explore")
    if expl:
        lines += ["### Exploratory: are T2 and T3 a settings problem? (validation only, after the freeze)", "",
                  "Not part of the preregistered comparison: these runs read no test data and change settings the "
                  "freeze fixed (s, M, up to 60 epochs with patience 12). Mean best validation accuracy over 3 seeds.", "",
                  "| task | L | arm | s=15, M=100 | s=40, M=100 | s=80, M=100 | s=40, M=300 (T=75) |", "|---|---|---|---|---|---|---|"]
        agg = defaultdict(list)
        for r in expl.values():
            agg[(r["task"], r["L"], r["arm"], int(r["s"]), r["M"])].append(100 * r["best_val"])
        for task, L in (("T2", 17), ("T3", 16)):
            for arm in ("A2_pair_w2", "A2_norel", "B0_flat"):
                cells = []
                for s_, M_ in ((15, 100), (40, 100), (80, 100), (40, 300)):
                    v = agg.get((task, L, arm, s_, M_))
                    cells.append(f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if v and len(v) > 1 else "–")
                lines.append(f"| {task} | {L} | {LABEL[arm]} | " + " | ".join(cells) + " |")
        lines.append("")
    # interventions
    interp = load("interpret")
    if interp:
        lines += ["### H4 and interventions on the trained A2 models (test set)", "",
                  "Accuracy when the same trained model may only use some pairs at inference; clause composition "
                  "(share of clause weight); share of the matching pairs of voting clauses that sit at the planted "
                  "position (T4: offset d; T2: the query's key slot).", "",
                  "| task | L | all pairs | self only | |j−i| ≤ 1 | shuffled rel. | cross-token clauses | with rel. literals | planted share |",
                  "|---|---|---|---|---|---|---|---|---|"]
        agg = defaultdict(list)
        for r in interp.values():
            agg[(r["task"], r["L"])].append(r)
        for (task, L), rs in sorted(agg.items()):
            m = lambda f, rs=rs: np.mean([f(r) for r in rs])  # noqa: E731
            ps = [r["planted_share"] for r in rs if r["planted_share"] is not None]
            lines.append(f"| {task} | {L} | {100 * m(lambda r: r['acc']['all']):.2f} | {100 * m(lambda r: r['acc']['self']):.2f} | "
                         f"{100 * m(lambda r: r['acc']['window1']):.2f} | {100 * m(lambda r: r['acc']['shuffled_rel']):.2f} | "
                         f"{m(lambda r: r['composition']['cross']):.2f} | {m(lambda r: r['composition']['rel']):.2f} | "
                         f"{fmt(np.mean(ps), 3) if ps else '–'} |")
        lines.append("")
    return lines


def values():
    """Numbers used in the prose, computed from the records."""
    ex = load("exact")
    v = {"exact_pass": f"{sum(r['pass'] for r in ex.values())}/{len(ex)}"}
    d1 = json.loads((RESULTS / "dynamics" / "D1.json").read_text())
    v["d1_pass"] = f"{d1['n_within_3se']}/{d1['n_cells']}"
    first = json.loads((RESULTS / "dynamics" / "D1_first_run.json").read_text())
    v["d1_first"] = f"{first['n_within_3se']}/{first['n_cells']}"
    v["x5_clauses"] = str(ex["X5"]["table"]["therm_cpu"]["clauses"])
    rec = [t for k, t in ex["X10"]["table"].items() if "recall" in k]
    v["b73_fit"] = f"{sum(t['min_clauses'] == t['n_times_min_keys_2pow_n'] for t in rec)}/{len(rec)}"
    thr = load("threshold")
    acc = defaultdict(list)
    for r in thr.values():
        if r["task"] == "T4" and r["V"] == 16 and r["arm"] == "A2_pair_w2" and r["L"] == 32:
            acc[int(r["s"])].append(r["best_val"])
    good = [x for x in SS if acc.get(x) and np.mean(acc[x]) >= 0.9]
    v["d2_ratio"] = f"{good[0] / s_pred(32, 16, True):.1f}" if good else "n/a"
    ex_ = defaultdict(list)
    for r in load("explore").values():
        ex_[(r["task"], r["arm"], int(r["s"]), r["M"])].append(100 * r["best_val"])
    best = lambda task, arm: max(np.mean(x) for k, x in ex_.items() if k[0] == task and k[1] == arm)  # noqa: E731
    v["t2_a2_best"] = f"{best('T2', 'A2_pair_w2'):.1f}"
    v["t2_norel_best"] = f"{best('T2', 'A2_norel'):.1f}"
    v["t2_flat_best"] = f"{best('T2', 'B0_flat'):.1f}"
    v["t3_best"] = f"{max(best('T3', a) for a in ('A2_pair_w2', 'A2_norel', 'B0_flat')):.1f}"
    return v


def summary(v):
    final = load("final")

    def acc(task, L, arm):
        xs = [r["test_acc"] * 100 for r in final.values() if r["task"] == task and r["L"] == L and r["arm"] == arm]
        return np.mean(xs) if xs else float("nan")
    best_base = lambda task, L: max(acc(task, L, b) for b in BASELINES)  # noqa: E731

    def best_arm(task, L):
        return max(arms_for(task), key=lambda a: acc(task, L, a))
    lines = ["## Summary", "",
             f"- **Formal claims:** {v['exact_pass']} exact checks pass, with the library as the object under test "
             "(hand-built clauses loaded into real models, CPU and CUDA) or by exhaustive computation. New exact "
             "results: Proposition A3b, Proposition D1, tightness of B6.2, and Conjecture B7.3 "
             f"({v['b73_fit']} instances).",
             f"- **Learning dynamics:** the Type Ia drift formula fits {v['d1_pass']} measurement cells after one "
             f"correction (the first run, {v['d1_first']}, used the textbook memorisation probability; the library "
             "boosts true positives by default). The dilution prediction (C2) holds in direction, not in size.",
             f"- **Attention, where a relation between two tokens decides the label:** co-occurrence T5 at L = 32: pair TM "
             f"{acc('T5', 32, 'A2_pair_w2'):.1f} % vs best baseline {best_base('T5', 32):.1f} %; order T1 at L = 32: "
             f"{acc('T1', 32, 'A2_pair_w2'):.2f} % vs {best_base('T1', 32):.2f} %. Both gains vanish without the "
             "cross-token pairs (A2-self).",
             f"- **Ties, as predicted:** needle T4 at L = 32: pair TM {acc('T4', 32, 'A2_pair_w2'):.2f} %, flat TM "
             f"{acc('T4', 32, 'B0_flat'):.2f} %, conv kernel d+1 {acc('T4', 32, 'B3_conv_kd1'):.2f} %. Controls T6/T7: "
             "no cost.",
             f"- **Not solved:** recall T2 at L = 17 (chance 25 %): pair TM {acc('T2', 17, 'A2_pair_w2'):.1f} %, best arm "
             f"{LABEL[best_arm('T2', 17)].strip('*')} {acc('T2', 17, best_arm('T2', 17)):.1f} %; repeat detection T3 at L = 16 "
             f"(chance 50 %): best arm {acc('T3', 16, best_arm('T3', 16)):.1f} %. Exact pair-clause solutions for both "
             "run in the library (X8). Exploratory runs (validation only) show T2 is learned with more clauses: pair TM "
             f"{v['t2_a2_best']} %, flat TM {v['t2_flat_best']} %; T3 stays at chance.",
             f"- **Positional codes:** binary beats thermometer on T1 at L = 32 ({acc('T1', 32, 'P_binary'):.1f} % vs "
             f"{acc('T1', 32, 'P_therm'):.1f} %), the opposite of the clause-cost ranking.", ""]
    return lines


def main():
    head = (ROOT / "report_head.md").read_text() if (ROOT / "report_head.md").exists() else ""
    tail = (ROOT / "report_tail.md").read_text() if (ROOT / "report_tail.md").exists() else ""
    v = values()
    for k, x in v.items():
        tail = tail.replace("{{" + k + "}}", x)
    head = "\n".join(summary(v)) + "\n" + head
    body = ["# Positional encoding and attention for Tsetlin machines: report", "",
            "Generated by `python -m experiments.transformer_concepts.code.report` from `results/**/*.json` "
            "(rule C2: no hand-typed numbers in the tables). Plan: `torchtsetlin_plans/experiments/transformer_concepts/PROOF_PLAN.md`; theory: `THEORY.md`.", ""]
    body += [head, ""] + part1() + [""] + part2() + [""] + part3() + ["", tail]
    (ROOT / "REPORT.md").write_text("\n".join(body))
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
