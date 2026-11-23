"""REPORT.md and results/summary.json from the records (plan §11); ``--status`` prints progress.

Every number comes from ``results/<phase>/<tag>.json``. Per team and batch size:
  - test accuracy at the best-validation epoch (the statistic),
  - test accuracy at fixed epochs and at fixed numbers of updates (at the best-validation epoch
    up to that point), to separate "larger batches learn worse" from "larger batches make fewer
    updates",
  - time per epoch, time to the best epoch,
  - B*: the largest batch size within 1.0 pp of the team's best cell.

Usage: python code/report.py [--status]
"""

import argparse
import time

from common import (
    BATCHES,
    CACHE,
    EXT_BATCHES,
    LONG,
    PROG,
    RESULTS,
    TEAMS,
    jobs,
    policy_choice,
    read_json,
    write_json,
)

EPOCH_MARKS = (60, 100)
UPDATE_MARKS = (20_000, 50_000, 100_000)
TOL_PP = 1.0
PHASE_LIST = ("pricing", "pilot", "grid", "ext", "large", "cprice", "clauses", "lprice", "long")
# The run the `long` phase starts from (W&B cjt-clauses-M_b64_geocol_s0_c32000_Tscaled; its record
# stayed on the old server, so it is quoted here and marked as such in the report).
LONG_REFERENCE = dict(tag="M_b64_geocol_s0_c32000_Tscaled", val=77.70, test=77.47, best_epoch=97, epochs=127,
                      patience=30)


def records(phase):
    out = {}
    for j in jobs(phase):
        r = read_json(RESULTS / phase / f"{j['tag']}.json")
        if r:
            out[j["tag"]] = (j, r)
    return out


def best_until(curve, key, limit):
    """Test accuracy at the best-validation epoch among rows with ``row[key] <= limit``."""
    rows = [c for c in curve if c[key] <= limit]
    if not rows or max(c[key] for c in curve) < limit:
        return None
    b = max(rows, key=lambda c: (c["val_acc"], -c["epoch"]))
    return b["test_acc"]


def fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def status():
    for ph in PHASE_LIST:
        js = jobs(ph)
        if not js:
            if ph in ("grid", "ext") and policy_choice() is None:
                print(f"{ph:8s} waits for the pilot (augmentation policy)")
            continue
        done = run = 0
        lines = []
        for j in js:
            r = read_json(RESULTS / ph / f"{j['tag']}.json")
            claimed = (CACHE / "claims" / ph / j["tag"]).exists()
            if r and r.get("status") == "done":
                done += 1
                if ph in ("pricing", "cprice", "lprice"):
                    lines.append(f"    {j['tag']:30s} {r['train_s_per_epoch'] + r['eval_s_per_epoch']:8.0f} s/epoch  "
                                 f"peak {r['peak_gpu_mib'] / 1024:5.1f} GiB  aug {r['aug_s_per_epoch']:.0f} s")
                else:
                    lines.append(f"    {j['tag']:30s} done    test {r['test_acc_at_best']:.2f} @ epoch "
                                 f"{r['best_epoch']}/{r['epochs_run']}  ({r['elapsed_s'] / 3600:.1f} h)")
            elif r and r.get("curve"):
                run += claimed
                c = r["curve"][-1]
                lines.append(f"    {j['tag']:30s} {'running' if claimed else 'paused '} epoch {c['epoch']:3d}  val "
                             f"{c['val_acc']:.2f}  best {c['best_val_acc']:.2f}  ({r['elapsed_s'] / 3600:.1f} h)")
            elif claimed:
                run += 1
                lines.append(f"    {j['tag']:30s} starting")
        print(f"{ph:8s} {done}/{len(js)} done, {run} running")
        print("\n".join(lines)) if lines else None
    c = read_json(RESULTS / "pilot" / "choice.json")
    if c:
        print(f"augmentation policy: {c['policy']} (pilot validation accuracy {c['best_val_acc']})")
    c = read_json(RESULTS / "long" / "choice.json")
    if c:
        print(f"long, stage 2: drop_clause_p = {c['drop_clause_p']:g} (stage-1 validation accuracy {c['best_val_acc']})")


def long_section():
    """Longer training and drop clause (plan PLAN_LONG_DROPOUT.md): one row per run."""
    recs = records("long")
    if not recs:
        return [], []
    ref = LONG_REFERENCE
    L = ["## Longer training and drop clause (team M, B = 64, `geo+col`, T scaled with the clauses)", "",
         "Early stopping on validation accuracy without an epoch limit. *Tail* = mean ± standard deviation of the "
         "test accuracy over the last epochs of the run (the patience window): the plateau, which does not depend "
         "on picking one epoch. *Train (clean)* = accuracy in prediction mode on "
         f"{LONG['train_eval']} fixed un-augmented training images at the best epoch; *gap* = that minus the "
         "validation accuracy.", "",
         "| Clauses | drop_clause_p | Test | Val | Best epoch / run | Tail (epochs) | Train (clean) | Gap | "
         "Train (running) | s/epoch | h total | Stop |", "|" + "---|" * 12,
         f"| 32000 | 0 (patience {ref['patience']}, earlier run, from W&B) | {ref['test']:.2f} | {ref['val']:.2f} | "
         f"{ref['best_epoch']} / {ref['epochs']} | — | — | — | — | — | — | patience |"]
    rows = []
    for j, r in recs.values():
        if r.get("status") != "done":
            L.append(f"| {j['clauses']} | {j['drop']:g} | *running, epoch {len(r.get('curve', []))}* |" + " |" * 9)
            continue
        b = r["curve"][r["best_epoch"] - 1]
        row = dict(tag=j["tag"], clauses=j["clauses"], drop=j["drop"], test=r["test_acc_at_best"], val=r["best_val_acc"],
                   best_epoch=r["best_epoch"], epochs=r["epochs_run"], tail_mean=r["test_acc_tail_mean"],
                   tail_std=r["test_acc_tail_std"], tail_epochs=r["tail_epochs"],
                   train_eval=r.get("train_eval_acc_at_best"), gap=r.get("gap_at_best"),
                   train_running=b["train_acc_running"],
                   s_per_epoch=r["train_s_per_epoch_mean"] + r["eval_s_per_epoch_mean"], h_total=r["elapsed_s"] / 3600,
                   stop=r["stop_reason"], collapses=sum(1 for x in r["curve"] if x["val_acc"] < r["best_val_acc"] - 5
                                                        and x["epoch"] > r["best_epoch"] // 2),
                   wandb=r.get("wandb_url"))
        rows.append(row)
        L.append(f"| {row['clauses']} | {row['drop']:g} | {row['test']:.2f} | {row['val']:.2f} | {row['best_epoch']} / "
                 f"{row['epochs']} | {row['tail_mean']:.2f} ± {row['tail_std']:.2f} ({row['tail_epochs']}) | "
                 f"{fmt(row['train_eval'])} | {fmt(row['gap'])} | {row['train_running']:.2f} | {row['s_per_epoch']:.0f} | "
                 f"{row['h_total']:.1f} | {row['stop']} |")
    L.append("")
    base = {r["clauses"]: r for r in rows if r["drop"] == 0}
    if LONG["clauses"] in base:
        d = base[LONG["clauses"]]["test"] - ref["test"]
        L += [f"More epochs (p = 0, {LONG['clauses']} clauses): {base[LONG['clauses']]['test']:.2f} % against "
              f"{ref['test']:.2f} % of the earlier run, {d:+.2f} pp"
              + (" (below the 0.5 pp tie band)." if abs(d) < LONG["tie_pp"] else "."), ""]
    for r in rows:
        if r["drop"] > 0 and r["clauses"] in base:
            d = r["test"] - base[r["clauses"]]["test"]
            L.append(f"- drop_clause_p = {r['drop']:g} at {r['clauses']} clauses: {d:+.2f} pp test against p = 0"
                     + (" (tie)" if abs(d) < LONG["tie_pp"] else "") + ".")
    ch = read_json(RESULTS / "long" / "choice.json")
    if ch:
        L += ["", f"Stage 2 drop_clause_p: **{ch['drop_clause_p']:g}** ({ch['rule']})."]
    if any(r["collapses"] for r in rows):
        L += ["", "Epochs later than half the best epoch whose validation accuracy is more than 5 pp below the "
              "run's best: " + ", ".join(f"p = {r['drop']:g} / {r['clauses']}: {r['collapses']}"
                                                            for r in rows) + "."]
    L.append("")
    return L, rows


def team_table(team, recs):
    rows = []
    for (j, r) in sorted(recs, key=lambda t: t[0]["batch"]):
        if j["team"] != team or r.get("status") != "done":
            continue
        c = r["curve"]
        rows.append(dict(batch=j["batch"], tag=j["tag"], test=r["test_acc_at_best"], val=r["best_val_acc"],
                         best_epoch=r["best_epoch"], epochs=r["epochs_run"], stop=r["stop_reason"],
                         s_per_epoch=r["train_s_per_epoch_mean"] + r["eval_s_per_epoch_mean"],
                         h_to_best=r["gpu_s_to_best"] / 3600, h_total=r["elapsed_s"] / 3600,
                         updates_to_best=r["updates_to_best"],
                         at_epoch={e: best_until(c, "epoch", e) for e in EPOCH_MARKS},
                         at_updates={u: best_until(c, "updates", u) for u in UPDATE_MARKS},
                         members={n: m["test_acc"] for n, m in r["members_at_best"].items()},
                         vote_share={n: m["vote_share"] for n, m in r["members_at_best"].items()},
                         map_correlation=next((x.get("map_correlation") for x in reversed(c)
                                               if x.get("map_correlation") is not None), None),
                         wandb=r.get("wandb_url")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
        return
    L = ["# cifar10_joint_teams — results", "",
         f"Generated by `code/report.py` on {time.strftime('%Y-%m-%d %H:%M')} from `results/`; do not edit by hand.",
         "", "Jointly trained teams (rule `sum`) on CIFAR-10 with training-time augmentation; seed 0. "
         "Accuracy = test accuracy (%) at the best-validation epoch.", ""]
    summary = {}
    pr = records("pricing")
    if pr:
        L += ["## Cost (pricing, 300 timed updates)", "", "| Team | B | s/epoch | Augmentation s/epoch | Peak GPU GiB |",
              "|---|---|---|---|---|"]
        for j, r in pr.values():
            L.append(f"| {j['team']} | {j['batch']} | {r['train_s_per_epoch'] + r['eval_s_per_epoch']:.0f} | "
                     f"{r['aug_s_per_epoch']:.0f} | {r['peak_gpu_mib'] / 1024:.1f} |")
        L.append("")
    ch = read_json(RESULTS / "pilot" / "choice.json")
    pl = records("pilot")
    if pl:
        L += ["## Augmentation pilot (team M, B = 32, 40 epochs)", "", "| Policy | Best val | Test at best | Epoch |",
              "|---|---|---|---|"]
        for j, r in pl.values():
            if r.get("status") == "done":
                L.append(f"| `{j['policy']}` | {r['best_val_acc']:.2f} | {r['test_acc_at_best']:.2f} | {r['best_epoch']} |")
        L += ["", f"Chosen policy: **`{ch['policy']}`** ({ch['rule']})." if ch else "Policy not chosen yet.", ""]
    recs = [v for ph in ("grid", "ext", "large") for v in records(ph).values()]
    for team in TEAMS:
        rows = team_table(team, recs)
        if not rows:
            continue
        summary[team] = rows
        top = max(r["test"] for r in rows)
        bstar = max(r["batch"] for r in rows if r["test"] >= top - TOL_PP)
        summary[f"{team}_Bstar"] = bstar
        L += [f"## Team {team}: {' + '.join(TEAMS[team])}", "",
              "| B | Test | Val | Best epoch / run | " + " | ".join(f"Test @ {e} ep" for e in EPOCH_MARKS) + " | "
              + " | ".join(f"Test @ {u // 1000}k upd" for u in UPDATE_MARKS) + " | s/epoch | h to best | h total |",
              "|" + "---|" * (7 + len(EPOCH_MARKS) + len(UPDATE_MARKS))]
        for r in rows:
            L.append(f"| {r['batch']} | {r['test']:.2f} | {r['val']:.2f} | {r['best_epoch']} / {r['epochs']} | "
                     + " | ".join(fmt(r["at_epoch"][e]) for e in EPOCH_MARKS) + " | "
                     + " | ".join(fmt(r["at_updates"][u]) for u in UPDATE_MARKS)
                     + f" | {r['s_per_epoch']:.0f} | {r['h_to_best']:.1f} | {r['h_total']:.1f} |")
        missing = [b for b in BATCHES + (EXT_BATCHES if team == "M" else []) if b not in {r["batch"] for r in rows}]
        L += ["", f"Best cell {top:.2f} %; B* (largest batch within {TOL_PP} pp of it) = **{bstar}**."
              + (f" Not finished: B = {missing}." if missing and team != "L" else ""), "",
              "Members at the best epoch (each member's own test accuracy / share of the team's votes) and the "
              "correlation between member activation maps:", "",
              "| B | " + " | ".join(TEAMS[team]) + " | Map correlation |", "|" + "---|" * (2 + len(TEAMS[team]))]
        for r in rows:
            L.append(f"| {r['batch']} | " + " | ".join(f"{r['members'][n]:.2f} / {100 * r['vote_share'][n]:.0f} %"
                                                      for n in TEAMS[team]) + f" | {fmt(r['map_correlation'], 3)} |")
        L.append("")
    long_lines, long_rows = long_section()
    L += long_lines
    if long_rows:
        summary["long"] = long_rows
    if not summary:
        L += ["No finished grid runs yet.", ""]
    (PROG / "REPORT.md").write_text("\n".join(L) + "\n")
    write_json({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "policy": ch, "teams": summary},
               RESULTS / "summary.json")
    print(f"wrote {PROG / 'REPORT.md'}")


if __name__ == "__main__":
    main()
