"""Reports of the joint_wide_tm programme, generated from results/ only (no hand-typed numbers).

    python code/report.py --phase checks            -> REPORT_checks.md   (T0-T8)
    python code/report.py --phase pricing           -> REPORT_pricing.md  (T7: s/epoch, memory, GPU-hours)
    python code/report.py --phase smoke|pilot|main  -> REPORT_<phase>.md
    python code/report.py --phase pilot --choose    -> also results/pilot/choice.json

Decision rule (plan B7, fixed before any run): Team S, J vs the composite with the validation
scale, test accuracy at the best-validation epoch, seeds paired; band +-0.3 pp.
"""

import argparse

import numpy as np
from common import (
    EPOCHS,
    MEMBERS,
    PROG,
    RESULTS,
    RULES,
    TEAMS,
    member_tag,
    read_json,
    team_tag,
    wide_tag,
    write_json,
)

BAND = 0.3            # pp (B7)
RULE_TIE = 0.2        # pp on validation: both joint rules go to the main phase (B10)


def rec(phase, arm, tag):
    f = RESULTS / phase / arm / f"{tag}.json"
    return read_json(f) if f.exists() else None


def done(r):
    return r is not None and r.get("status", "done") == "done"


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def checks():
    r = read_json(RESULTS / "checks" / "check_joint.json")
    what = {"T0": "one member, zero offset, own negative: bit-identical to the library model (CPU, CUDA eager, events, graphs)",
            "T1": "the score a member learns from equals the external joint score on every path",
            "T2": "shared mode: one negative class per example for all members, never the target",
            "T3": "CUDA graphs: a replayed update reads the new offset",
            "T4": "pass A equals the votes recomputed inside update (with T1)",
            "T5": "toy: joint team of a 2x2 and a 3x3 member is not worse than the better member (paired, 2 SE)",
            "T6": "`sum` team of identical members = one model with M*n clauses and M*T",
            "T8": "state_dict round trip; member files load into the library model"}
    L = ["# joint_wide_tm — phase 0 correctness checks", "",
         f"Record: `results/checks/check_joint.json` ({r['env'].get('gpu', 'cpu')}, torchtsetlin "
         f"{r['env']['torchtsetlin']}, git `{r['git_sha'][:10]}`). **{'All passed' if r['passed'] else 'FAILED'}.**", "",
         "| Check | What | Result |", "|---|---|---|"]
    for k in sorted(r["checks"]):
        L.append(f"| {k} | {what.get(k, '')} | {'pass' if r['checks'][k]['passed'] else '**FAIL**'} |")
    c = r["checks"]
    if "T5" in c:
        d = c["T5"]["detail"]
        L += ["", f"T5 (toy, {d['config']['seeds']} seeds, accuracy mean ± SE): "
              + ", ".join(f"{k} {d[k]['acc_mean']:.3f} ± {d[k]['acc_se']:.3f}" for k in ("m2x2", "m3x3", "joint-sum", "joint-one"))
              + ". Paired difference to the better member: "
              + ", ".join(f"{k} {d[k]['paired_diff_mean']:+.3f} ± {d[k]['paired_diff_se']:.3f}" for k in ("joint-sum", "joint-one")) + "."]
    if "T6" in c:
        d = c["T6"]["detail"]
        q = d["b_selections_per_update"]
        L += ["", f"T6: (a) largest difference between the score a split member learns from and the big model's votes: "
              f"{d['a_max_abs_vote_diff']:g}. (b) feedback selections per update, paired by batch over "
              f"{d['config']['batches']} batches: `sum` team − big model = {q['sum'][1]:+.2f} ± {q['sum'][2]:.2f} "
              f"({q['sum'][3]:.1f} SE); control (`one` rule, threshold T) = {q['one'][1]:+.2f} ± {q['one'][2]:.2f} "
              f"({q['one'][3]:.1f} SE). (c) trained from scratch, {d['config']['seeds']} seeds: accuracy team "
              f"{d['team']['acc_mean']:.3f} ± {d['team']['acc_se']:.3f}, big model {d['single']['acc_mean']:.3f} ± "
              f"{d['single']['acc_se']:.3f} ({d['c_team_vs_single_in_se']['acc']:.1f} SE apart)."]
    (PROG / "REPORT_checks.md").write_text("\n".join(L) + "\n")


def pricing():
    L = ["# joint_wide_tm — pricing (T7)", "",
         "Timed updates after a warm-up, extrapolated to one epoch of 50 000 images (batch 32); evaluation = "
         "validation + test. GPU-hours = (train + eval) × epochs.", "",
         "| Arm | Run | Train s/epoch | Eval s/epoch | Peak MiB | GPU-h, 40 epochs | GPU-h, 100 epochs |", "|---|---|---|---|---|---|---|"]
    rows = {}
    for arm in ("I", "J", "W"):
        for f in sorted((RESULTS / "pricing" / arm).glob("*.json")):
            r = read_json(f)
            s = r["train_s_per_epoch"] + r["eval_s_per_epoch"]
            rows[(arm, r["tag"])] = r
            L.append(f"| {arm} | {r['tag']} | {r['train_s_per_epoch']:.1f} | {r['eval_s_per_epoch']:.1f} | "
                     f"{r['peak_gpu_mib']:.0f} | {s * 40 / 3600:.2f} | {s * 100 / 3600:.2f} |")
    if rows:
        gpu = next(iter(rows.values()))["env"].get("gpu", "")
        h = lambda arm, tag, ep: (rows[(arm, tag)]["train_s_per_epoch"] + rows[(arm, tag)]["eval_s_per_epoch"]) * ep / 3600 \
            if (arm, tag) in rows else float("nan")                                                   # noqa: E731
        I = lambda ep: sum(h("I", member_tag(m, 0), ep) for m in MEMBERS)                              # noqa: E731
        J = lambda ep, rules: sum(h("J", team_tag(t, r, 0), ep) for t in TEAMS for r in rules)         # noqa: E731
        Wmax = lambda ep: sum(max(h("W", wide_tag(t, "sum", m, 0), ep) for m in TEAMS[t]) for t in TEAMS)  # noqa: E731
        pilot = I(EPOCHS["pilot"]) + J(EPOCHS["pilot"], RULES)
        main = 3 * (I(EPOCHS["main"]) + J(EPOCHS["main"], ["sum"]) + Wmax(EPOCHS["main"]))
        L += ["", f"Device: {gpu}. Largest peak: {max(r['peak_gpu_mib'] for r in rows.values()):.0f} MiB.", "",
              "| Phase | Runs | GPU-hours |", "|---|---|---|",
              f"| 1 pilot | 5 members + 4 joint, {EPOCHS['pilot']} epochs | {pilot:.1f} |",
              f"| 2 main | 3 seeds × (5 members + 2 joint + 2 W), {EPOCHS['main']} epochs; joint priced with rule `sum`, "
              f"W with each team's most expensive member | {main:.1f} |",
              f"| **Total** | | **{pilot + main:.1f}** |"]
    (PROG / "REPORT_pricing.md").write_text("\n".join(L) + "\n")


def table(phase, seeds):
    L = [f"# joint_wide_tm — {phase}", "",
         "Test accuracy (%) at the best-validation epoch; `val` = that validation accuracy. Composite = members "
         "trained independently, scale from the validation set (`valalpha`); `fixed` = v/T; `paper` = test-set scale "
         "(transductive, reference).", ""]
    summary = {}
    for team, names in TEAMS.items():
        L += [f"## Team {team}: {' + '.join(names)}", "", "| Seed | " + " | ".join(names)
              + " | Composite (val) | fixed | paper | J-sum (val) | J-one (val) | Jown | W-sum | W-one |",
              "|---|" + "---|" * (len(names) + 8)]
        cols = {k: [] for k in ("comp", "J-sum", "J-one", "W-sum", "W-one")}
        for seed in seeds:
            ms = [rec(phase, "I", member_tag(n, seed)) for n in names]
            comp = rec(phase, "composite", f"{team}_seed{seed}")
            cell = [fmt(m["test_acc_at_best"]) if done(m) else "—" for m in ms]
            cell += ([f"{comp['test_acc_at_best']:.2f} ({comp['best_val_acc']:.2f})", fmt(comp["best"]["fixed"]["test_acc"]),
                      fmt(comp["best"]["paper"]["test_acc"])] if comp else ["—"] * 3)
            cols["comp"].append(comp["test_acc_at_best"] if comp else None)
            for r in RULES:
                j = rec(phase, "J", team_tag(team, r, seed))
                cell.append(f"{j['test_acc_at_best']:.2f} ({j['best_val_acc']:.2f})" if done(j) else "—")
                cols[f"J-{r}"].append(j["test_acc_at_best"] if done(j) else None)
            jo = [rec(phase, "Jown", team_tag(team, r, seed)) for r in RULES]
            cell.append(" / ".join(fmt(j["test_acc_at_best"]) for j in jo if done(j)) or "—")
            for r in RULES:
                w = [rec(phase, "W", wide_tag(team, r, m, seed)) for m in names]
                w = [q for q in w if done(q)]
                cell.append(f"{w[0]['test_acc_at_best']:.2f} ({next(iter(w[0]['hp']['members']))})" if w else "—")
                cols[f"W-{r}"].append(w[0]["test_acc_at_best"] if w else None)
            L.append(f"| {seed} | " + " | ".join(cell) + " |")
        L.append("")
        summary[team] = cols
        for r in RULES:
            pairs = [(j, c) for j, c in zip(cols[f"J-{r}"], cols["comp"]) if j is not None and c is not None]
            if not pairs:
                continue
            d = np.array([j - c for j, c in pairs])
            w = [v for v in cols[f"W-{r}"] if v is not None]
            line = (f"J-{r} − composite: Δ = {d.mean():+.2f} pp over {len(d)} seed(s) "
                    f"(per seed {', '.join(f'{v:+.2f}' for v in d)})")
            if w:
                line += f"; mean J-{r} {np.mean([j for j, _ in pairs]):.2f} vs W-{r} {np.mean(w):.2f}"
            if phase == "main" and len(d) >= 3:
                jm, ahead = np.mean([j for j, _ in pairs]), bool((d > 0).all())
                if d.mean() >= BAND and ahead and w and jm > np.mean(w):
                    v = "Works"
                elif d.mean() >= BAND and w and jm <= np.mean(w):
                    v = "Only T"
                elif d.mean() >= BAND:
                    v = "Works (pending W or a seed behind)" if not w else "Equal (J not ahead in every seed)"
                elif d.mean() <= -BAND:
                    v = "Worse"
                else:
                    v = "Equal"
                line += f". **Verdict (B7): {v}**"
            L += [line + ".", ""]
    ens = sorted((RESULTS / phase / "ensemble").glob("*.json")) if (RESULTS / phase / "ensemble").exists() else []
    if ens:
        L += ["## Seed ensembles (arm E)", "", "| Member | Seeds | Test at best val |", "|---|---|---|"]
        for f in ens:
            r = read_json(f)
            L.append(f"| {r['member']} | {r['seeds']} | {r['test_acc_at_best']:.2f} |")
        L.append("")
    (PROG / f"REPORT_{phase}.md").write_text("\n".join(L) + "\n")
    return summary


def choose():
    ch = {}
    for team, names in TEAMS.items():
        js = {r: rec("pilot", "J", team_tag(team, r, 0)) for r in RULES}
        if not all(done(j) for j in js.values()):
            raise SystemExit(f"pilot J runs of team {team} are not finished")
        best = max(RULES, key=lambda r: js[r]["best_val_acc"])
        rules = [r for r in RULES if js[best]["best_val_acc"] - js[r]["best_val_acc"] <= RULE_TIE]
        ms = {n: rec("pilot", "I", member_tag(n, 0)) for n in names}
        comp = rec("pilot", "composite", f"{team}_seed0")
        ch[team] = {"rules": rules, "best_rule": best, "val": {r: js[r]["best_val_acc"] for r in RULES},
                    "w_member": max(names, key=lambda n: ms[n]["best_val_acc"]),
                    "member_val": {n: ms[n]["best_val_acc"] for n in names},
                    "composite_val": comp["best_val_acc"] if comp else None}
    kill = all(c["composite_val"] is not None and c["composite_val"] - max(c["val"].values()) >= 1.0 for c in ch.values())
    ch["kill"] = bool(kill)                           # B6: both rules >= 1 pp below the composite in both teams
    write_json(ch, RESULTS / "pilot" / "choice.json")
    print(ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["checks", "pricing", "smoke", "pilot", "main"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--choose", action="store_true")
    args = ap.parse_args()
    if args.phase == "checks":
        return checks()
    if args.phase == "pricing":
        return pricing()
    table(args.phase, args.seeds or ([0, 1, 2] if args.phase == "main" else [0]))
    if args.choose:
        choose()


if __name__ == "__main__":
    main()
