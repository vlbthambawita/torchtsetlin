"""REPORT.md, results/summary.json and figures/accuracy_vs_depth.png from the run records.

Every number comes from ``results/<phase>/<tag>.json``; nothing is typed by hand (the paper's
published numbers are the constants ``common.PAPER_REPORTED``).
Usage:  python code/report.py            # write the report
        python code/report.py --status   # progress only (used by status.sh)
"""

import argparse
import time

from common import (
    DEPTH_CLAUSES,
    DEPTHS,
    EXTRA,
    FIGURES,
    GATE_S,
    MAIN,
    PAPER,
    PAPER_CELLS,
    PAPER_REPORTED,
    PILOT,
    PILOT_CLAUSES,
    PROG,
    REPO,
    RESULTS,
    TIE_PP,
    cell,
    jobs,
    read_json,
    resolve,
    tag_of,
    ts_choice,
    write_json,
)

CONV_REF = REPO / "experiments" / "cifar10_batch_clauses" / "results" / "grid" / "conv_c32000_b32_Tscaled_s0.json"


def rec_of(phase, tag):
    return read_json(RESULTS / phase / f"{tag}.json")


def done(r):
    return bool(r) and r.get("status") == "done"


def val(r, key, fmt="{:.2f}"):
    if r is None:
        return "–"
    if r.get("status") == "oom":
        return "oom"
    if not done(r):
        return f"(epoch {len(r.get('curve') or [])})"
    v = r.get(key)
    return "–" if v is None else fmt.format(v)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def run_of(c):
    """(the record of the run that answers planned cell ``c``, the cell actually run, gate state)."""
    r = resolve(c)
    return (rec_of("depth", tag_of(r["cell"])) if r["state"] == "run" else None), r["cell"], r


def describe(c):
    return f"depth {c['depth']}, {c['clauses']} clauses" + (
        f", inbox {c['inbox']}" if c["depth"] > 1 else "") + (", rewired edges" if c["edges"] == "rewired" else "")


def gate_note(c, r):
    if r["state"] == "pending":
        return "waits for pricing"
    if r["state"] == "dropped":
        return ("not run: above the gate (CGT_PAPER_FORCE=1 runs it anyway)" if c["kind"] != "grid"
                else "not run: every option is above the gate")
    return "as planned" if r["cell"] == c else f"shrunk by the gate to {describe(r['cell'])}"


def status():
    lines = []
    for phase in ("paper", "pricing", "pilot", "depth"):
        js = jobs(phase)
        st = [(rec_of(phase, j["tag"]) or {}).get("status") for j in js]
        lines.append(f"{phase:8s} {sum(s in ('done', 'oom') for s in st):2d} / {len(js)} finished, "
                     f"{sum(s == 'running' for s in st)} with a record in progress, "
                     f"{sum(bool(j['wait']) for j in js)} waiting")
        for j, s in zip(js, st):
            r = rec_of(phase, j["tag"])
            if r and s == "oom":
                lines.append(f"    {j['tag']:44s} out of memory")
            elif r and s == "running" and r.get("curve"):
                c = r["curve"][-1]
                more = (f"val {c['val_acc']:.2f}  best {c['best_val_acc']:.2f}  wait {c['epochs_since_improvement']}"
                        if "val_acc" in c else f"test {c['test_acc']:.2f}")
                lines.append(f"    {j['tag']:44s} epoch {c['epoch']:3d}  {more}  ({c['train_s']:.0f} s/epoch)")
            elif r and s == "done" and phase == "pricing":
                t = r["train_s_per_epoch"] + r["eval_s_per_epoch"]
                lines.append(f"    {j['tag']:44s} {r['train_s_per_epoch']:.0f} + {r['eval_s_per_epoch']:.0f} s/epoch, "
                             f"peak {r['peak_gpu_mib'] / 1024:.1f} GiB{'' if t <= GATE_S else '  ABOVE THE GATE'}")
            elif r and s == "done" and phase == "paper":
                lines.append(f"    {j['tag']:44s} done   mean of the last {r['last_n']} test epochs "
                             f"{r['test_acc_last_mean']:.2f} ({r['stop_reason']}, {r['elapsed_s'] / 3600:.1f} GPU-h)")
            elif r and s == "done":
                lines.append(f"    {j['tag']:44s} done   test {r['test_acc_at_best']:.2f} @ epoch {r['best_epoch']} "
                             f"({r['stop_reason']}, {r['elapsed_s'] / 3600:.1f} GPU-h)")
            elif j["wait"]:
                lines.append(f"    {j['tag']:44s} {j['wait']}")
    for c in PAPER_CELLS + MAIN + EXTRA:
        r = resolve(c)
        if r["state"] == "dropped" or (r["state"] == "run" and r["cell"] != c):
            lines.append(f"gate: {tag_of(c)}: {gate_note(c, r)}")
    ch = ts_choice()
    lines.append("T / C, s = " + (f"{ch[0]:g}, {ch[1]:g}" if ch else "not chosen yet (pilot not finished)"))
    k = read_json(RESULTS / "checks.json")
    lines.append("phase-0 checks: " + ("not run (./setup.sh)" if not k else "ok" if k["ok"] else "FAILED"))
    return "\n".join(lines)


def figure(main):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(DEPTH_CLAUSES, ("#1f6feb", "#b3541e")))
    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="white")
    drawn = False
    for C in DEPTH_CLAUSES:
        pts = [(d, main[(d, C)][0]["test_acc_at_best"]) for d in DEPTHS
               if done(main[(d, C)][0]) and main[(d, C)][1] == cell(d, C)]
        if pts:
            drawn = True
            ax.plot(*zip(*pts), "o-", color=colors[C], linewidth=1.8, markersize=5, label=f"{C} clauses")
    ax.set_xticks(DEPTHS)
    ax.set_xlabel("depth (1 = no message passing)")
    ax.set_ylabel("test accuracy at the best-validation epoch (%)")
    ax.grid(color="#d0d7de", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if drawn:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "accuracy_vs_depth.png", dpi=130)
    plt.close(fig)
    return drawn


COLS = ["Test @ best val", "Best val", "Train (5 000) @ best", "Gap (pp)", "Message literals (%)", "Best epoch",
        "Epochs", "Stop", "GPU-h"]


def row_of(r):
    return [val(r, "test_acc_at_best"), val(r, "best_val_acc"), val(r, "train_eval_acc_at_best"),
            val(r, "gap_at_best"), "–" if not done(r) else f"{100 * r['message_literal_share_at_best']:.1f}",
            val(r, "best_epoch", "{:d}"), val(r, "epochs_run", "{:d}"), val(r, "stop_reason", "{}"),
            "–" if not done(r) else f"{r['elapsed_s'] / 3600:.1f}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(status())
        return

    choice = read_json(RESULTS / "pilot" / "choice.json")
    ch = ts_choice()
    checks = read_json(RESULTS / "checks.json")
    main_runs = {(c["depth"], c["clauses"]): run_of(c) for c in MAIN}
    t = lambda r: r["test_acc_at_best"]                 # noqa: E731

    L = ["# cifar10_graph_tm: the graph Tsetlin machine on CIFAR-10", "",
         f"Generated by `code/report.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/`. Do not edit by hand.",
         "Depth study: `GraphTsetlinMachine` on a 15 × 15 grid of 4 × 4 colour-thermometer windows (stride 2), four",
         "typed edges to the grid neighbours, shared clauses with signed weights, literal budget 32, batch 32, seed 0;",
         "early stopping on the 5 000-image validation split (patience 50); statistic = test accuracy at the",
         f"best-validation epoch. One seed: differences below {TIE_PP} pp are ties.", ""]

    if checks:
        d, f = checks["depth1_equals_conv"], checks["fast_path"]
        L += ["## Phase-0 checks", "",
              f"- Depth 1 equals `ConvCoalescedTsetlinMachine` bit for bit over {d.get('updates', 0)} updates: "
              f"**{'yes' if d['ok'] else 'NO'}**.",
              f"- Updates of depth 1, 2 and 3 run on the fast CUDA path and a depth-2 update is free of host "
              f"synchronisation: **{'yes' if f['ok'] else 'NO'}**.",
              f"- The rewired control keeps the in-edges per node and type: **{'yes' if checks['rewired']['ok'] else 'NO'}**.",
              ""]

    price_rows = []
    for j in jobs("pricing"):
        r = rec_of("pricing", j["tag"])
        ok = done(r) and r["train_s_per_epoch"] + r["eval_s_per_epoch"] <= GATE_S
        price_rows.append([j["tag"], val(r, "ms_per_update"), val(r, "train_s_per_epoch", "{:.0f}"),
                           val(r, "eval_s_per_epoch", "{:.0f}"),
                           "–" if not done(r) else f"{r['peak_gpu_mib'] / 1024:.1f}",
                           "–" if not done(r) else r.get("cuda_graphs_captured", "–"),
                           "–" if r is None else "pass" if ok else "above the gate",
                           (r or {}).get("env", {}).get("gpu", "–")])
    if any(rec_of("pricing", j["tag"]) for j in jobs("pricing")):
        L += [f"## Cost (pricing, 100 updates) and the gate ({GATE_S:.0f} s per epoch)", "",
              table(["Cell", "ms/update", "Train s/epoch", "Eval s/epoch", "Peak GiB", "CUDA graphs", "Gate", "GPU"],
                    price_rows), ""]
        notes = [f"- `{tag_of(c)}`: {gate_note(c, resolve(c))}." for c in PAPER_CELLS + MAIN + EXTRA
                 if resolve(c)["state"] == "dropped" or (resolve(c)["state"] == "run" and resolve(c)["cell"] != c)]
        L += notes + ([""] if notes else [])

    L += [f"## Pilot (depth 1, {PILOT_CLAUSES} clauses, at most 60 epochs)", "",
          table(["T", "T / C", "s", "Best val", "Test @ best val", "Best epoch", "Epochs", "GPU-h"],
                [[T, f"{T / PILOT_CLAUSES:g}", f"{s:g}"] + [val(r, k, f) for k, f in
                                                         (("best_val_acc", "{:.2f}"), ("test_acc_at_best", "{:.2f}"),
                                                          ("best_epoch", "{:d}"), ("epochs_run", "{:d}"))]
                 + ["–" if not done(r) else f"{r['elapsed_s'] / 3600:.1f}"]
                 for (T, s) in PILOT for r in [rec_of("pilot", tag_of(cell(1, PILOT_CLAUSES), pilot=(T, s)))]]), ""]
    if choice:
        L += [f"**Chosen: T / C = {choice['ratio']:g}, s = {choice['s']:g}.** Rule: {choice['rule']}.", ""]
    elif ch:
        L += [f"**T / C = {ch[0]:g}, s = {ch[1]:g}** (set by `CGT_TS`, not by the pilot).", ""]
    else:
        L += ["The pilot is not finished: T and s are not chosen yet.", ""]

    L += ["## Depth study", "",
          table(["Planned cell", "Run"] + COLS,
                [[describe(c), gate_note(c, g)] + row_of(r) for c in MAIN for r, _, g in [main_runs[(c["depth"], c["clauses"])]]]),
          ""]
    if figure(main_runs):
        L += ["![Test accuracy against depth](figures/accuracy_vs_depth.png)", ""]
    L += ["## Extra cells (depth 2)", "",
          table(["Planned cell", "Run"] + COLS, [[describe(c), gate_note(c, g)] + row_of(r)
                                                 for c in EXTRA for r, _, g in [run_of(c)]]), ""]

    L += ["## Hypotheses (plan §2)", ""]
    for C in DEPTH_CLAUSES:
        got = {d: main_runs[(d, C)] for d in DEPTHS}
        if not all(done(got[d][0]) and got[d][1] == cell(d, C) for d in DEPTHS):
            L.append(f"- {C} clauses: not all three depths ran as planned (see the table).")
            continue
        a, b, c3 = (t(got[d][0]) for d in DEPTHS)
        L.append(f"- **{C} clauses.** Test: depth 1 {a:.2f}, depth 2 {b:.2f} ({b - a:+.2f}), depth 3 {c3:.2f} "
                 f"({c3 - b:+.2f} over depth 2). H1 (depth 2 > depth 1 by more than {TIE_PP} pp): "
                 f"**{'holds' if b - a > TIE_PP else 'does not hold'}**. H3 (depth 3 adds less than depth 2 did): "
                 f"**{'holds' if c3 - b < b - a else 'does not hold'}**.")
    rw, base = run_of(EXTRA[-1])[0], main_runs[(2, EXTRA[-1]["clauses"])]
    d1 = main_runs[(1, EXTRA[-1]["clauses"])][0]
    if done(rw) and done(base[0]) and done(d1):
        L.append(f"- **H2 (rewired edges).** Depth 2 with grid edges {t(base[0]):.2f}, with rewired edges {t(rw):.2f}, "
                 f"depth 1 {t(d1):.2f}: the rewired model is "
                 f"{'within' if abs(t(rw) - t(d1)) < TIE_PP else 'not within'} {TIE_PP} pp of depth 1.")
    L.append("")

    L += ["## Paper setting (best effort; the paper's protocol, never mixed into the depth table)", "",
          f"{PAPER['clauses']} clauses, T = {PAPER['T']}, s = {PAPER['s']:g}, 8 × 8 windows, literal budget 32, "
          f"batch {PAPER['batch']}, 50 000 training images plus horizontal flips, {PAPER['epochs']} epochs, "
          f"statistic = mean test accuracy of the last {PAPER['last']} epochs. The GraphTM node features are our "
          "reading of the paper (1 778 symbols, one feature per symbol; Table 16 of the paper says 2 500 symbols in "
          "hypervectors), so a difference from the published number cannot be attributed to the implementation alone.", ""]
    rows = []
    paper = {}
    for kind, name in (("paper_graph", "GraphTM"), ("paper_cotm", "CoTM baseline")):
        r = paper[kind] = rec_of("paper", tag_of(dict(kind=kind)))
        if r is None and resolve(dict(kind=kind, clauses=PAPER["clauses"]))["state"] == "dropped":
            name += " (not run: above the cost gate)"
        rep = PAPER_REPORTED[kind]
        rows.append([name, val(r, "test_acc_last_mean"), val(r, "test_acc_last_sd"), val(r, "test_acc_last"),
                     val(r, "epochs_run", "{:d}") + ("" if not done(r) or r["complete"] else " (stopped early)"),
                     f"{rep[0]:.2f} ± {rep[1]:.2f}",
                     "–" if not done(r) else f"{r['test_acc_last_mean'] - rep[0]:+.2f}",
                     "–" if not done(r) else f"{r['elapsed_s'] / 3600:.1f}"])
    L += [table(["Model", "Mean of the last 5 epochs", "SD", "Final epoch", "Epochs", "Paper (arXiv:2507.14874)",
                 "Difference (pp)", "GPU-h"], rows), ""]
    g, c = paper["paper_graph"], paper["paper_cotm"]
    if done(g) and done(c):
        pub, gm, cm = PAPER_REPORTED["paper_graph"][0], g["test_acc_last_mean"], c["test_acc_last_mean"]
        L += [f"H4 (within 2 pp of {pub:.2f} % and above the CoTM baseline): GraphTM {gm:.2f}, CoTM {cm:.2f}: "
              f"**{'holds' if abs(gm - pub) <= 2 and gm > cm else 'does not hold'}**.", ""]

    ref = read_json(CONV_REF)
    best = max((r for r, _, _ in main_runs.values() if done(r)), key=t, default=None)
    L += ["## Reference: per-class ConvTM", ""]
    if ref and done(ref):
        L += [f"`cifar10_batch_clauses`, 3 × 3 colour-thermometer ConvTM, 32 000 clauses per class: "
              f"{ref['test_acc_at_best']:.2f} % test at the best-validation epoch (patience 25)."
              + (f" Best graph model of the depth study: {t(best):.2f} % (`{best['tag']}`). H5 (no graph model "
                 f"beats it): **{'holds' if t(best) <= ref['test_acc_at_best'] else 'does not hold'}**." if best else ""),
              "The comparison mixes clause sharing and window size; it is context, not a controlled comparison.", ""]
    else:
        L += ["The record `cifar10_batch_clauses/results/grid/conv_c32000_b32_Tscaled_s0.json` is not on this "
              "machine.", ""]

    (PROG / "REPORT.md").write_text("\n".join(L))
    write_json({"ts_choice": ch, "pilot_choice": choice,
                "depth": {tag_of(c): {"run": None if g["cell"] is None else tag_of(g["cell"]), "gate": gate_note(c, g),
                                      "test_acc_at_best": (r or {}).get("test_acc_at_best")}
                          for c in MAIN + EXTRA for r, _, g in [run_of(c)]},
                "paper": {k: (r or {}).get("test_acc_last_mean") for k, r in paper.items()}},
               RESULTS / "summary.json")
    print(f"wrote {PROG / 'REPORT.md'}")


if __name__ == "__main__":
    main()
