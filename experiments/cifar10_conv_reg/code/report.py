"""REPORT.md, results/summary.json and figures/accuracy_vs_clauses.png from the run records.

Every number comes from ``results/<phase>/<tag>.json`` (plan §9); nothing is typed by hand.
Usage:  python code/report.py            # write the report
        python code/report.py --status   # progress only (used by status.sh)
"""

import argparse
import time

from common import (
    ARMS,
    AUGS,
    DROP_PS,
    FACTORIAL_CLAUSES,
    FIGURES,
    PROG,
    REPO,
    RESULTS,
    SCREEN_CLAUSES,
    TIE_PP,
    TOOLBOX,
    aug_tag,
    drop_choice,
    jobs,
    read_json,
    tag_of,
    write_json,
)

OLD_GRID = REPO / "experiments" / "cifar10_batch_clauses" / "results" / "grid"


def rec_of(phase, tag):
    return read_json(RESULTS / phase / f"{tag}.json")


def cell(r, key, fmt="{:.2f}"):
    if r is None:
        return "–"
    if r.get("status") == "oom":
        return "oom"
    if r.get("status") != "done":
        c = r.get("curve") or []
        return f"(epoch {len(c)})" if key == "test_acc_at_best" else "…"
    v = r.get(key)
    return "–" if v is None else fmt.format(v)


def row_of(r):
    """The report columns of one finished run."""
    return [cell(r, "test_acc_at_best"), cell(r, "best_val_acc"), cell(r, "train_eval_acc_at_best"),
            cell(r, "gap_at_best"), cell(r, "best_epoch", "{:d}"), cell(r, "epochs_run", "{:d}"),
            cell(r, "stop_reason", "{}"),
            "–" if not r or r.get("status") != "done" else f"{r['elapsed_s'] / 3600:.1f}"]


COLS = ["Test @ best val", "Best val", "Train (clean) @ best", "Gap (pp)", "Best epoch", "Epochs", "Stop", "GPU-h"]


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def status():
    lines = []
    for phase in ("pricing", "screen", "factorial"):
        js = jobs(phase)
        st = [None if j["p"] is None else (rec_of(phase, j["tag"]) or {}).get("status") for j in js]
        lines.append(f"{phase:10s} {sum(s in ('done', 'oom') for s in st):2d} / {len(js)} finished, "
                     f"{sum(s == 'running' for s in st)} with a record in progress, "
                     f"{sum(j['p'] is None for j in js)} waiting for p*")
        for j, s in zip(js, st):
            r = rec_of(phase, j["tag"]) if j["p"] is not None else None
            if r and s == "running" and r.get("curve"):
                c = r["curve"][-1]
                lines.append(f"    {j['tag']:38s} epoch {c['epoch']:3d}  val {c['val_acc']:.2f}  best "
                             f"{c['best_val_acc']:.2f}  gap {c['gap']:.1f}  wait {c['epochs_since_improvement']}")
            elif r and s == "done" and phase == "pricing":
                lines.append(f"    {j['tag']:38s} {r['train_s_per_epoch']:.0f} + {r['eval_s_per_epoch']:.0f} s/epoch, "
                             f"peak {r['peak_gpu_mib'] / 1024:.1f} GiB")
            elif r and s == "done":
                lines.append(f"    {j['tag']:38s} done   test {r['test_acc_at_best']:.2f} @ epoch {r['best_epoch']} "
                             f"({r['stop_reason']}, {r['elapsed_s'] / 3600:.1f} GPU-h)")
    c = read_json(RESULTS / "screen" / "choice.json")
    lines.append(f"p* = {drop_choice()}" + (f"  (drop clause {'helps' if c['helps'] else 'does not help'} at "
                                           f"{SCREEN_CLAUSES})" if c else "  (screen not finished)"))
    return "\n".join(lines)


def figure(fact):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#59636e", "aug": "#1f6feb", "drop": "#b3541e", "aug_drop": "#1a7f37"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor="white")
    drawn = False
    for arm in ARMS:
        pts = [(c, fact[(c, arm)]["test_acc_at_best"]) for c in FACTORIAL_CLAUSES
               if fact.get((c, arm)) and fact[(c, arm)].get("status") == "done"]
        if pts:
            drawn = True
            ax.plot(*zip(*pts), "o-", color=colors[arm], linewidth=1.8, markersize=5, label=arm)
    xs = sorted(TOOLBOX["none"])
    ax.plot(xs, [TOOLBOX["none"][c] for c in xs], "--", color="#8c959f", linewidth=1.2,
            label="Toolbox Table IV (tmu, no augmentation)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c // 1000}k" for c in xs])
    ax.set_xlabel("clauses per class")
    ax.set_ylabel("test accuracy at the best-validation epoch (%)")
    ax.grid(color="#d0d7de", linewidth=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "accuracy_vs_clauses.png", dpi=130)
    plt.close(fig)
    return drawn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(status())
        return

    p_star = drop_choice()
    choice = read_json(RESULTS / "screen" / "choice.json")
    screen = {(a, p): rec_of("screen", tag_of(SCREEN_CLAUSES, a, p)) for a in AUGS for p in DROP_PS}
    fact = {}
    for j in jobs("factorial"):
        fact[(j["clauses"], j["arm"])] = rec_of("factorial", j["tag"]) if j["p"] is not None else None
    done = lambda r: bool(r) and r.get("status") == "done"   # noqa: E731

    L = ["# cifar10_conv_reg: drop clause, augmentation and 64 000 clauses for the single ConvTM on CIFAR-10", "",
         f"Generated by `code/report.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/`. Do not edit by hand.",
         "Model: 3 × 3 colour-thermometer ConvTM, weighted, literal budget 32, s = 5, T = 1.5 × clauses per class,",
         "batch 32, seed 0; early stopping on the 5 000-image validation split (patience 50). “Train (clean)” is the",
         "accuracy on 5 000 fixed un-augmented training images in prediction mode; gap = train (clean) − validation.",
         f"One seed: differences below {TIE_PP} pp are ties.", ""]

    price = [(j, rec_of("pricing", j["tag"])) for j in jobs("pricing")]
    if any(r for _, r in price):
        L += ["## Cost (pricing, 200 updates)", "",
              table(["Clauses/class", "drop_clause_p", "ms/update", "Train s/epoch", "Eval s/epoch", "Peak GiB", "GPU"],
                    [[j["clauses"], f"{j['p']:g}", cell(r, "ms_per_update"), cell(r, "train_s_per_epoch", "{:.0f}"),
                      cell(r, "eval_s_per_epoch", "{:.0f}"),
                      "–" if not done(r) else f"{r['peak_gpu_mib'] / 1024:.1f}",
                      (r or {}).get("env", {}).get("gpu", "–")] for j, r in price]), ""]

    L += [f"## Screen at {SCREEN_CLAUSES} clauses per class", "",
          table(["Augmentation", "drop_clause_p"] + COLS,
                [[a, f"{p:g}"] + row_of(screen[(a, p)]) for a in AUGS for p in DROP_PS]), ""]
    if choice:
        L += [f"**Chosen p\\* = {choice['p_star']:g}.** Rule: {choice['rule']}. Drop clause "
              f"{'is within the tie band of, or better than, no drop clause' if choice['helps'] else 'does not help'} "
              f"at this size.", ""]
    elif p_star is not None:
        L += [f"**p\\* = {p_star:g}** (set by `CCR_DROP_P`, not by the screen).", ""]
    else:
        L += ["The screen is not finished: p\\* is not chosen yet.", ""]

    L += ["## Factorial", "",
          table(["Clauses/class", "Arm", "Augmentation", "drop_clause_p"] + COLS,
                [[c, arm, ARMS[arm][0], "0" if not ARMS[arm][1] else ("p*" if p_star is None else f"{p_star:g}")]
                 + row_of(fact[(c, arm)]) for c in FACTORIAL_CLAUSES for arm in ARMS]), ""]
    if figure(fact):
        L += ["![Test accuracy against clauses per class](figures/accuracy_vs_clauses.png)", ""]

    ref = []
    for c in [SCREEN_CLAUSES] + FACTORIAL_CLAUSES:
        old = read_json(OLD_GRID / f"conv_c{c}_b32_Tscaled_s0.json")
        new = screen[("none", 0.0)] if c == SCREEN_CLAUSES else fact.get((c, "base"))
        ref.append([c, cell(new, "test_acc_at_best"),
                    cell(old, "test_acc_at_best") if old else "– (record not on this machine)",
                    f"{TOOLBOX['none'][c]:.1f}", f"{TOOLBOX['augmented'][c]:.1f}"])
    L += ["## Baseline against earlier numbers", "",
          table(["Clauses/class", "Baseline here (patience 50)", "cifar10_batch_clauses (patience 25, V100)",
                 "Toolbox Table IV", "Toolbox Table IV, augmented"], ref), "",
          "Toolbox: arXiv:2406.00704, 3 × 3 colour thermometers, tmu, 250 epochs, mean of the last 25 epochs.", ""]

    L += ["## Hypotheses (plan §2)", ""]
    for c in FACTORIAL_CLAUSES:
        b, a, d, ad = (fact.get((c, k)) for k in ("base", "aug", "drop", "aug_drop"))
        if not all(done(r) for r in (b, a, d, ad)):
            L.append(f"- {c} clauses: not all four arms are finished.")
            continue
        t = lambda r: r["test_acc_at_best"]                 # noqa: E731
        best = max(("base", b), ("aug", a), ("drop", d), ("aug_drop", ad), key=lambda kv: t(kv[1]))
        L.append(f"- **{c} clauses.** Test: base {t(b):.2f}, aug {t(a):.2f} ({t(a) - t(b):+.2f}), drop {t(d):.2f} "
                 f"({t(d) - t(b):+.2f}), aug + drop {t(ad):.2f} ({t(ad) - t(b):+.2f}); best arm: {best[0]}"
                 f"{'' if t(best[1]) - t(b) >= TIE_PP or best[0] == 'base' else ' (a tie with base)'}. "
                 f"Gap: base {b['gap_at_best']:.1f}, aug {a['gap_at_best']:.1f}, drop {d['gap_at_best']:.1f}, "
                 f"aug + drop {ad['gap_at_best']:.1f} pp. Best epoch: {b['best_epoch']} / {a['best_epoch']} / "
                 f"{d['best_epoch']} / {ad['best_epoch']}.")
    b64 = fact.get((64000, "base"))
    if done(b64):
        L.append(f"- **H3.** Baseline at 64 000 clauses: {b64['test_acc_at_best']:.2f} % (hypothesis 74–75.5 %, "
                 f"Toolbox 74.3 %).")
    L.append("")

    (PROG / "REPORT.md").write_text("\n".join(L))
    write_json({"p_star": p_star, "choice": choice,
                "screen": {f"{aug_tag(a)}/dc{p:g}": (r or {}).get("test_acc_at_best") for (a, p), r in screen.items()},
                "factorial": {f"c{c}/{arm}": (r or {}).get("test_acc_at_best") for (c, arm), r in fact.items()}},
               RESULTS / "summary.json")
    print(f"wrote {PROG / 'REPORT.md'}")


if __name__ == "__main__":
    main()
