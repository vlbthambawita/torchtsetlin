"""Generate REPORT_HTM.md from results/htm/ (rule C2). Prose that is not computed lives in
report_htm_tail.md; every number here is read from a record.

    python -m experiments.transformer_concepts.code.htm.report
"""
from __future__ import annotations

import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "htm"


def J(p):
    return json.loads(Path(p).read_text())


def recs(stage):
    return [J(p) for p in sorted(glob.glob(str(R / stage / "*.json")))]


def ms(v, scale=100):
    v = [scale * x for x in v]
    if not v:
        return "–"
    return f"{mean(v):.2f} ± {stdev(v):.2f} (n={len(v)})" if len(v) > 1 else f"{v[0]:.2f} (n=1)"


def boot_diff(a, b, n=4000, seed=0):
    """95 % bootstrap CI of mean(a) - mean(b), unpaired, in percentage points."""
    rnd = random.Random(seed)
    d = []
    for _ in range(n):
        aa = [rnd.choice(a) for _ in a]
        bb = [rnd.choice(b) for _ in b]
        d.append(100 * (mean(aa) - mean(bb)))
    d.sort()
    return 100 * (mean(a) - mean(b)), d[int(0.025 * n)], d[int(0.975 * n)]


def part0():
    out = ["## Part 0. Exact checks", "", "| check | claim | result |", "|---|---|---|"]
    for name in ("X13", "X14", "X15", "X16"):
        p = R / "exact" / f"{name}.json"
        if not p.exists():
            out.append(f"| {name} | – | not run |")
            continue
        r = J(p)
        extra = ""
        if name == "X13":
            extra = "; minima " + ", ".join(f"(L={i['L']},V={i['V']},w={i['w']}): {i['minimum']}"
                                            f"{'' if i['one_symbol_lemma'] else ' [one-symbol lemma fails]'}"
                                            for i in r["instances"])
        if name == "X15":
            extra = "; hashed inbox density at f = " + ", ".join(f"{h['f']}: {h['density']:.3f} (pred. {h['predicted']:.3f})"
                                                                for h in r["hashed_inbox"])
        if name == "X16":
            extra = f"; {sum(s['converged'] for s in r['starts'])}/{len(r['starts'])} starts converged to Lloyd fixed points"
        first = " (first run failed, see D18)" if (R / "exact" / f"{name}_first_run.json").exists() else ""
        out.append(f"| {name} | {r['claim']} | **{'pass' if r['pass'] else 'FAIL'}**{first}{extra} |")
    return out + [""]


def part_h1():
    out = ["## Part H1. Tolerant clauses", ""]
    a = recs("h1a")
    if a:
        out += ["### E-H1a: from-scratch TM at δ = 0 vs the library (gate G1; validation accuracy, 10 seeds)", "",
                "| task | library | from scratch | difference (pp) [95 % CI] | included literals / clause (lib vs scratch, last epoch) | gate |",
                "|---|---|---|---|---|---|"]
        by = defaultdict(lambda: defaultdict(list))
        for r in a:
            by[r["task"]][r["impl"]].append(r)
        for task, d in by.items():
            la = [r["best_val"] for r in d["lib"]]
            ma = [r["best_val"] for r in d["mine"]]
            if not la or not ma:
                continue
            diff, lo, hi = boot_diff(ma, la)
            li = mean(r["included_per_clause"][-1] for r in d["lib"])
            mi = mean(r["included_per_clause"][-1] for r in d["mine"])
            gate = "pass" if lo <= 0 <= hi or abs(diff) < 0.5 else "FAIL"
            out.append(f"| {task} | {ms(la)} | {ms(ma)} | {diff:+.2f} [{lo:+.2f}, {hi:+.2f}] | {li:.1f} vs {mi:.1f} | {gate} |")
        out.append("")
    b = recs("h1b_final")
    if b:
        out += ["### E-H1b: tolerance under bit noise (test accuracy, 5 seeds, s frozen per cell on validation)", ""]
        by = defaultdict(list)
        bayes = {}
        for r in b:
            by[(r["task"], r["eta"], r["delta"])].append(r["test_acc"])
            if "bayes_test_acc" in r:
                bayes[r["eta"]] = r["bayes_test_acc"]
        tasks = sorted({k[0] for k in by})
        for task in tasks:
            etas = sorted({k[1] for k in by if k[0] == task})
            deltas = sorted({k[2] for k in by if k[0] == task})
            out += [f"**{task}**" + (" (Bayes accuracy in the last column)" if task == "proto" else ""), "",
                    "| η | " + " | ".join(f"δ = {d}" for d in deltas) + " | δ=2 − δ=0 (pp) [95 % CI] |"
                    + (" Bayes |" if task == "proto" else ""),
                    "|---|" + "---|" * len(deltas) + "---|" + ("---|" if task == "proto" else "")]
            for eta in etas:
                cells = [ms(by[(task, eta, d)]) for d in deltas]
                if by.get((task, eta, 2)) and by.get((task, eta, 0)):
                    dd, lo, hi = boot_diff(by[(task, eta, 2)], by[(task, eta, 0)])
                    cmp = f"{dd:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                else:
                    cmp = "–"
                row = f"| {eta} | " + " | ".join(cells) + f" | {cmp} |"
                if task == "proto":
                    row += f" {100 * bayes.get(eta, float('nan')):.2f} |"
                out.append(row)
            out.append("")
    c = recs("h1c")
    if c:
        out += ["### E-H1c: CIFAR-10 flat TM with tolerance (test)", "", "| δ | test % |", "|---|---|"]
        by = defaultdict(list)
        for r in c:
            by[r["delta"]].append(r["test_acc"])
        for d in sorted(by):
            out.append(f"| {d} | {ms(by[d])} |")
        out.append("")
    return out


def part_h2():
    out = ["## Part H2. Sparse value codes", ""]
    p = R / "dh2a.json"
    if p.exists():
        h = J(p)
        out += ["### D-H2a: the trained Q/K/V model's inbox", "",
                f"- layer-1 parts matching per patch: f = {100 * h['f_mean']:.2f} % (saturation threshold for 99 % density: "
                f"{100 * h['f_threshold_for_q_099']:.2f} %)",
                f"- inbox density: mean {100 * h['inbox_density_mean']:.1f} %, minimum {100 * h['inbox_density_min']:.1f} % "
                f"(H2.2 predicts {100 * h['inbox_density_predicted_H2_2']:.2f} %)",
                f"- inbox literals among all included literals: {100 * h['inbox_literal_share']:.1f} %",
                f"- prediction P-H2a (f > 2 % and density > 95 %): **{'holds' if h['prediction_holds'] else 'fails'}**", ""]
    s = recs("h2b_search")
    if s:
        sp = {J(q)["k"]: J(q) for q in glob.glob(str(R / "h2b_sp" / "*.json"))}
        out += ["### E-H2b: SP-value attention on CIFAR-10", "",
                f"Effective $p^\\star$ for the 16 edge types (mean inbox entropy): {100 * s[0]['effective_p_star']:.1f} % "
                f"(senders per slot {s[0]['senders']['senders_min']:.0f}–{s[0]['senders']['senders_max']:.0f}, "
                f"mean {s[0]['senders']['senders_mean']:.1f}).", "",
                "| SP density k/m | dead columns | inbox density | val s=2 (10 ep.) | val s=10 (10 ep.) |", "|---|---|---|---|---|"]
        by = defaultdict(dict)
        dens = {}
        for r in s:
            by[r["k"]][r["s"]] = r["best_val"]
            dens[r["k"]] = r["inbox_density"]
        for k in sorted(by):
            dead = sp.get(k, {}).get("dead_columns", "–")
            out.append(f"| {100 * k / 256:.1f} % (k={k}) | {dead} | {100 * dens[k]:.1f} % | "
                       + " | ".join(f"{100 * by[k].get(sv, float('nan')):.2f}" for sv in (2.0, 10.0)) + " |")
        out.append("")
    f = recs("h2b_final")
    if f:
        out += [f"Final (frozen k = {f[0]['k']}, s = {f[0]['s']:g}, 30 epochs, test): {ms([r['test_acc'] for r in f])}. "
                "CIFAR-10 comparators (REPORT_CIFAR10.md, same budget): see the table there.", ""]
        cif = {}
        for p in glob.glob(str(ROOT / "results" / "cifar" / "*_e30.json")):
            r = J(p)
            if "test_acc" in r:
                cif.setdefault(r["arm"], []).append(r["test_acc"])
        out += ["| arm | test % |", "|---|---|", f"| SP-value attention (this) | {ms([r['test_acc'] for r in f])} |"]
        for a in ("ctm", "pair_local", "pair_self", "pair", "flat", "qkv_d1", "qkv"):
            if a in cif:
                out.append(f"| {a} | {ms(cif[a])} |")
        out.append("")
    return out


def part_h3():
    out = ["## Part H3. Repeat detection, recurrent state, local targets", ""]
    a = recs("eh3a")
    if a:
        out += ["### E-H3a: pair TM on T3 at very high s (L = 16, validation, 3 seeds)", "",
                "| s | M = 100 | M = 300 |", "|---|---|---|"]
        by = defaultdict(list)
        for r in a:
            by[(r["s"], r["M"])].append(r["best_val"])
        for s in sorted({k[0] for k in by}):
            out.append(f"| {s:g} | {ms(by[(s, 100)])} | {ms(by[(s, 300)])} |")
        out += ["", "Prediction P-H3a (chance, ≤ 52 %, in every cell): "
                + ("**fails**" if max(max(v) for v in by.values()) > 0.52 else "holds") + " (D19).", ""]
    f = recs("h3_final")
    if f:
        out += ["### E-H3b: fixed HTM registers (test, 5 seeds, s frozen on validation)", "",
                "| task | arm | L = 16 / 17 | L = 32 / 33 |", "|---|---|---|---|"]
        by = defaultdict(list)
        for r in f:
            by[(r["task"], r["arm"], r["L"])].append(r["test_acc"])
        for task, arm in sorted({(k[0], k[1]) for k in by}):
            Ls = sorted({k[2] for k in by if k[:2] == (task, arm)})
            out.append(f"| {task} | {arm} | " + " | ".join(ms(by[(task, arm, L)]) for L in Ls) + " |")
        out.append("")
    c = recs("h3c_final")
    if c:
        fz = J(R / "frozen_h3c.json")["chosen"] if (R / "frozen_h3c.json").exists() else {}
        out += ["### E-H3c: learned state with local targets (test, 5 seeds)", "",
                "Frozen state-TM settings (selected by self-supervised state-bit accuracy): "
                + "; ".join(f"{a}: s = {v['s']:g}, T = {v['T']:g}, clauses = {v['clause_mult']:g}×" for a, v in fz.items()), "",
                "| arm | L | free-running state-bit accuracy | all bits right | readout test % |", "|---|---|---|---|---|"]
        by = defaultdict(list)
        for r in c:
            by[(r["arm"], r["L"])].append(r)
        for (arm, L), rs in sorted(by.items()):
            out.append(f"| {arm} | {L} | {ms([r['state_bit_acc'] for r in rs])} | {ms([r['state_all_bits_acc'] for r in rs])} | "
                       f"{ms([r['test_acc'] for r in rs])} |")
        out.append("")
        seen = [r for r in c if r["arm"] == "seen_tf"]
        if seen:
            out += ["Error growth along the sequence (H3.9), seen_tf, mean bit error at step t:", ""]
            for L in sorted({r["L"] for r in seen}):
                rs = [r for r in seen if r["L"] == L]
                errs = [1 - mean(r["state_bit_acc_by_t"][t] for r in rs) for t in range(L)]
                pts = ", ".join(f"t={t}: {100 * errs[t]:.2f} %" for t in sorted({0, L // 4, L // 2, 3 * L // 4, L - 1}))
                out.append(f"- L = {L}: {pts}")
            out.append("")
    return out


def main():
    lines = ["# HTM ideas in Tsetlin machines: results (H1–H3)", "",
             "Generated by `code/htm/report.py` from `results/htm/`; do not edit by hand. Theory: `THEORY_HTM.md`;",
             "plan: `PLAN_HTM.md`; deviations: `DECISIONS.md` D18–. Library `src/torchtsetlin` unchanged.", ""]
    lines += part0() + part_h3() + part_h1() + part_h2()
    tail = ROOT / "report_htm_tail.md"
    if tail.exists():
        lines.append(tail.read_text())
    (ROOT / "REPORT_HTM.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
