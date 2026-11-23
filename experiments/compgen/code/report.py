"""REPORT.md, figures and results/summary.json from the records (plan §7); ``--status`` prints progress.

Every number comes from ``results/<phase>/<tag>.json``. Per cell (arm, clauses, feedback): epochs,
updates and training time to the first 10-epoch stretch at >= 90 %, 99 % and 99.9 % test accuracy
(median over the seeds that reached it, and how many did), the final test accuracy, and the
seconds per epoch.

Usage: python code/report.py [--status]
"""

import argparse
import statistics
import time

from common import CACHE, FIGURES, ISSUE, MARKS, MODEL, PROG, RESULTS, jobs, read_json, write_json

PHASE_LIST = ("price", "p1", "p2")
ARM_NAME = {"neg": "torchtsetlin, negated literals only", "both": "torchtsetlin, both literal kinds (standard TM)",
            "ref": "reference library (PyHierarchicalTsetlinMachineCUDA)"}


def records(phase):
    out = []
    for j in jobs(phase):
        r = read_json(RESULTS / phase / f"{j['tag']}.json")
        if r:
            out.append((j, r))
    return out


def status():
    for ph in PHASE_LIST:
        js = jobs(ph)
        done = run = 0
        lines = []
        for j in js:
            r = read_json(RESULTS / ph / f"{j['tag']}.json")
            claimed = (CACHE / "claims" / ph / j["tag"]).exists()
            if r and r.get("status") == "done":
                done += 1
                if ph == "price":
                    lines.append(f"    {j['tag']:22s} {r['s_per_epoch']:7.2f} s/epoch (training {r['train_s_per_epoch']:.2f} s)")
                else:
                    m = r["marks"]
                    lines.append(f"    {j['tag']:22s} done    test {r['test_acc_last']:6.2f} after {r['epochs_run']:4d} epochs; "
                                 "epochs to " + ", ".join(f"{k} %: {v['epoch'] if v else '—'}" for k, v in m.items())
                                 + f"  ({r['elapsed_s'] / 3600:.1f} h)")
            elif r and r.get("curve"):
                run += claimed
                c = r["curve"][-1]
                lines.append(f"    {j['tag']:22s} {'running' if claimed else 'paused '} epoch {c['epoch']:4d}  test "
                             f"{c['test_acc']:6.2f}  train {c['train_acc']:6.2f}  ({r['elapsed_s'] / 3600:.1f} h)")
            elif claimed:
                run += 1
                lines.append(f"    {j['tag']:22s} starting")
        print(f"{ph:6s} {done}/{len(js)} done, {run} running")
        print("\n".join(lines)) if lines else None


def med(v):
    return statistics.median(v) if v else None


def cell_rows(recs):
    """One row per (arm, clauses, feedback) from the finished runs."""
    cells = {}
    for j, r in recs:
        if r.get("status") == "done":
            cells.setdefault((j["arm"], j["clauses"], j["feedback"]), []).append((j, r))
    rows = []
    for (arm, clauses, fb), rs in cells.items():
        row = dict(arm=arm, clauses_per_class=clauses, feedback=fb, seeds=sorted(j["seed"] for j, _ in rs), n=len(rs),
                   test_last_mean=statistics.mean(r["test_acc_last"] for _, r in rs),
                   test_last_min=min(r["test_acc_last"] for _, r in rs),
                   train_last_mean=statistics.mean(r["train_acc_last"] for _, r in rs),
                   epochs_run=[r["epochs_run"] for _, r in rs], converged=sum(bool(r["converged"]) for _, r in rs),
                   train_s_per_epoch=statistics.mean(r["train_s_per_epoch_mean"] for _, r in rs),
                   gpu=sorted({r["env"].get("gpu") or "?" for _, r in rs}), marks={})
        for thr in MARKS:
            k = f"{thr:g}"
            hit = [r["marks"][k] for _, r in rs if r["marks"].get(k)]
            row["marks"][k] = dict(reached=len(hit), epochs=med([h["epoch"] for h in hit]),
                                   epochs_range=[min(h["epoch"] for h in hit), max(h["epoch"] for h in hit)] if hit else None,
                                   updates=med([h["updates"] for h in hit]), train_s=med([h["train_s"] for h in hit]))
        rows.append(row)
    order = {"ref": 0, "neg": 1, "both": 2}
    fbo = lambda f: 0 if f == "seq" else int(f[1:])  # noqa: E731
    return sorted(rows, key=lambda r: (order[r["arm"]], r["clauses_per_class"], fbo(r["feedback"])))


def fmt_mark(m, n, what):
    if not m["reached"]:
        return f"not reached (0/{n})"
    v = m[what]
    txt = f"{v:,.0f}" if what != "train_s" else (f"{v:.0f} s" if v < 600 else f"{v / 3600:.2f} h")
    if what == "epochs" and m["epochs_range"][0] != m["epochs_range"][1]:
        txt += f" ({m['epochs_range'][0]}–{m['epochs_range'][1]})"
    return txt + ("" if m["reached"] == n else f" [{m['reached']}/{n}]")


def table(rows):
    L = ["| Model | Clauses | Feedback | Seeds | Final test % (mean, min) | Final train % | "
         + " | ".join(f"Epochs to {t:g} %" for t in MARKS) + " | Updates to 99 % | Training time to 99 % | Training s/epoch |",
         "|" + "---|" * (9 + len(MARKS))]
    for r in rows:
        total = r["clauses_per_class"] * 2 if r["arm"] != "ref" else MODEL["clauses"]
        L.append(f"| {r['arm']} | {total} | {'sequential' if r['feedback'] == 'seq' else 'batch ' + r['feedback'][1:]} | {r['n']} | "
                 f"{r['test_last_mean']:.2f}, {r['test_last_min']:.2f} | {r['train_last_mean']:.2f} | "
                 + " | ".join(fmt_mark(r["marks"][f"{t:g}"], r["n"], "epochs") for t in MARKS)
                 + f" | {fmt_mark(r['marks']['99'], r['n'], 'updates')} | {fmt_mark(r['marks']['99'], r['n'], 'train_s')} | "
                 f"{r['train_s_per_epoch']:.2f} |")
    return L


def figures(recs):
    """Accuracy against epochs and updates, and the clause precision percentiles over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    done = [(j, r) for j, r in recs if r.get("curve")]
    if not done:
        return []
    FIGURES.mkdir(parents=True, exist_ok=True)
    made = []
    cells = sorted({(j["arm"], j["clauses"], j["feedback"]) for j, _ in done},
                   key=lambda c: (c[0] != "ref", c[0], c[1], 0 if c[2] == "seq" else int(c[2][1:])))
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(cells)}
    for xkey, name, xlabel in (("epoch", "accuracy_vs_epochs", "epoch"), ("updates", "accuracy_vs_updates", "updates")):
        for group, arms in (("p1", ("ref", "neg")), ("p2", ("both",))):
            sel = [(j, r) for j, r in done if j["arm"] in arms]
            if not sel:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            seen = set()
            for j, r in sel:
                c = (j["arm"], j["clauses"], j["feedback"])
                label = None if c in seen else f"{j['arm']} {2 * j['clauses'] if j['arm'] != 'ref' else MODEL['clauses']} clauses, {j['feedback']}"
                seen.add(c)
                ax.plot([row[xkey] for row in r["curve"]], [row["test_acc"] for row in r["curve"]], color=colors[c],
                        lw=1, alpha=0.8, label=label)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("test accuracy (%)")
            ax.set_title(f"Test accuracy against {xlabel} (one line per seed)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            f = FIGURES / f"{name}_{group}.png"
            fig.tight_layout()
            fig.savefig(f, dpi=130)
            plt.close(fig)
            made.append(f)
    show = [(j, r) for j, r in done if j["seed"] == min(x["seed"] for x, _ in done)
            and (j["arm"], j["feedback"]) in (("ref", "seq"), ("neg", "seq"), ("neg", "b32"), ("neg", "b512"), ("both", "b32"))
            and j["clauses"] == MODEL["clauses"] // 2]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8), sharey=True, squeeze=False)
        for ax, (j, r) in zip(axes[0], show):
            rows = [c for c in r["curve"] if "precision_p50" in c]
            ep = [c["epoch"] for c in rows]
            for p, ls in ((1, ":"), (5, "--"), (50, "-"), (95, "--"), (99, ":")):
                ax.plot(ep, [c[f"precision_p{p}"] for c in rows], ls, color="tab:blue", lw=1, label=f"p{p}")
            ax.plot(ep, [c["coverage"] for c in rows], color="tab:gray", lw=1, label="coverage")
            ax.plot([c["epoch"] for c in r["curve"]], [c["test_acc"] for c in r["curve"]], color="tab:red", lw=1.2,
                    label="test accuracy")
            ax.set_title(f"{j['tag']}", fontsize=9)
            ax.set_xlabel("epoch")
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("%")
        axes[0][0].legend(fontsize=7)
        fig.suptitle("Clause precision on the test set (percentiles over the clauses), coverage and accuracy", fontsize=10)
        f = FIGURES / "clause_precision.png"
        fig.tight_layout()
        fig.savefig(f, dpi=130)
        plt.close(fig)
        made.append(f)
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return
    L = ["# compgen — the compositional generalization benchmark (issue #15) on torchtsetlin", "",
         f"Generated by `code/report.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/`; do not edit by hand.", "",
         "Task: a permutation of 5 concepts, each shown as one of 5 synonyms (125 one-hot features); the label is the "
         "parity of the permutation. 40 000 training examples (1 % label noise) and 40 000 test examples out of 375 000 "
         "possible inputs. Setting of the issue: 200 clauses, T = 10, s = 25, literal budget 100. A mark is reached at "
         "the first epoch of 10 epochs in a row at or above that test accuracy; values are medians over the seeds that "
         "reached it.", ""]
    data = [read_json(f) for f in sorted((RESULTS / "data").glob("seed*.json"))]
    if data:
        L += ["## Data", "", "| Data seed | Distinct training inputs | Distinct test inputs | In both | Test inputs unseen in training |",
              "|---|---|---|---|---|"]
        for d in data:
            s = d["stats"]
            L.append(f"| {d['seed']} | {s['unique_train']:,} | {s['unique_test']:,} | {s['unique_both']:,} | {s['unseen_test']:,} |")
        L += [f"| issue #15 | {ISSUE['unique_train']:,} | {ISSUE['unique_test']:,} | {ISSUE['unique_both']:,} | {ISSUE['unseen_test']:,} |", "",
              "The data come from a line-by-line port of the reference script's generator (check M0: identical arrays "
              "for the same NumPy seed). The issue's statistics are those of its author's run.", ""]
    pr = [(j, r) for j, r in records("price") if r.get("status") == "done"]
    if pr:
        L += ["## Cost (3 epochs, seed 0; the first epoch is left out)", "", "| Model | Clauses | Feedback | Training s/epoch | With evaluation s/epoch | GPU |",
              "|---|---|---|---|---|---|"]
        for j, r in pr:
            L.append(f"| {j['arm']} | {2 * j['clauses'] if j['arm'] != 'ref' else MODEL['clauses']} | {j['feedback']} | "
                     f"{r['train_s_per_epoch']:.2f} | {r['s_per_epoch']:.2f} | {r['env'].get('gpu', '?')} |")
        L.append("")
    summary, figs_from = {}, []
    for ph, title in (("p1", "Port and batching"), ("p2", "The standard Tsetlin machine (both literal kinds)")):
        recs = records(ph)
        figs_from += recs
        rows = cell_rows(recs)
        if not rows:
            continue
        summary[ph] = rows
        L += [f"## {title}", ""] + table(rows) + [""]
        left = [j["tag"] for j, r in recs if r.get("status") != "done"] + \
               [j["tag"] for j in jobs(ph) if read_json(RESULTS / ph / f"{j['tag']}.json") is None]
        if left:
            L += [f"Not finished: {', '.join(left)}.", ""]
    if not summary:
        L += ["No finished runs yet.", ""]
    for name in ARM_NAME:
        if any(r["arm"] == name for rows in summary.values() for r in rows):
            L.append(f"- `{name}`: {ARM_NAME[name]}")
    made = figures(figs_from)
    if made:
        L += ["", "## Figures", ""] + [f"![{f.stem}](figures/{f.name})" for f in made]
    (PROG / "REPORT.md").write_text("\n".join(L) + "\n")
    write_json({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "cells": summary}, RESULTS / "summary.json")
    print(f"wrote {PROG / 'REPORT.md'}" + (f" and {len(made)} figure(s)" if made else ""))


if __name__ == "__main__":
    main()
