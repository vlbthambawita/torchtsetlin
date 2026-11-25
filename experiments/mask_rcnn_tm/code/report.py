"""Status, pricing and the report, all generated from ``results/runs/*.json`` (rule C2).

    python code/report.py --status     progress per phase, what is running, what waits
    python code/report.py --price      seconds per epoch from the pricing phase and the projected cost
    python code/report.py              writes report/REPORT.md
"""

import argparse
import statistics
import time

import jobs as JOBS
from common import CACHE, PATIENCE, PROG, read_json, record_path

ASSUMED_EPOCHS = 300                                 # projection only: early stopping decides the real length


def rec(tag):
    return read_json(record_path(tag))


def status():
    claims = {d.name for d in (CACHE / "claims").glob("*") if d.is_dir()}
    spent = 0.0
    for ph in JOBS.PHASES:
        js, blocked = JOBS.jobs(ph)
        done = sum(1 for j in js if (rec(j["tag"]) or {}).get("status") == "done")
        run = [j for j in js if j["tag"] in claims]
        wall = sum((rec(j["tag"]) or {}).get("wall_s") or 0.0 for j in js)
        spent += wall
        print(f"{ph:6s} {done:4d} / {len(js):4d} done, {len(run)} running, {wall / 3600:6.1f} GPU-h"
              + (f"   (more jobs appear after: {'; '.join(sorted(set(blocked))[:2])})" if blocked else ""))
        for j in run:
            r = rec(j["tag"]) or {}
            print(f"         running {j['tag']}  epoch {r.get('epochs', 0)}  best {r.get('best')}"
                  f" at {r.get('best_epoch')}  (patience {PATIENCE})")
    print(f"total  {spent / 3600:.1f} GPU-hours recorded (cap of the plan: 230)")
    for f in sorted((CACHE / "claims").glob("*.failed-*")):
        print(f"FAILED {f.name}")


def price():
    js, _ = JOBS.jobs("price")
    rows = {}
    for j in js:
        r = rec(j["tag"])
        if r and r.get("status") == "done" and "s_per_epoch" in r:
            key = (j.get("head") or j.get("model"), j.get("aug", "sym"), "grad" if "grad" in j["tag"] else "")
            rows[key] = r["s_per_epoch"]
    if not rows:
        print("no pricing records yet: KV_PHASES=price ./run_gpu0.sh")
        return {}
    print(f"{'model':12s} {'aug':5s} {'s/epoch':>8s} {'h per 100 epochs':>18s}")
    for (m, aug, g), s in sorted(rows.items()):
        print(f"{m + ('.' + g if g else ''):12s} {aug:5s} {s:8.1f} {s * 100 / 3600:18.2f}")
    print(f"\nProjected cost if every run lasts {ASSUMED_EPOCHS} epochs (early stopping decides the real length;"
          f" a run lasts at least {PATIENCE + 1}):")
    total = 0.0
    for ph in JOBS.PHASES[1:]:
        pj, blocked = JOBS.jobs(ph)
        h = 0.0
        for j in pj:
            if j["kind"] not in ("train", "baseline"):
                continue
            s = rows.get((j.get("head") or j.get("model"), "alb" if str(j.get("aug", "")).startswith("alb") else "sym",
                          "grad" if "grad" in j["tag"] else ""))
            s = s if s is not None else rows.get((j.get("head") or j.get("model"), "sym", ""), 0.0)
            h += s * len(j.get("members", [1])) * ASSUMED_EPOCHS / 3600
        total += h
        print(f"  {ph:6s} {len(pj):4d} jobs listed now  {h:7.1f} GPU-h" + ("  (+ jobs that appear after selections)" if blocked else ""))
    print(f"  total of the jobs listed now: {total:.1f} GPU-h on one GPU, {total / 4:.1f} h of wall time on four")
    return rows


def fmt(vals, digits=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{statistics.mean(vals):.{digits}f} ± {statistics.stdev(vals):.{digits}f} (n={len(vals)})"


def report():
    L = ["# Detect-then-segment with Tsetlin machines on Kvasir-SEG", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `results/runs/*.json`. Plan: `PLAN_KVASIR.md`.", ""]
    o = rec("oracle-K0b")
    if o:
        L += ["## Gate K0b: ceilings of the design", "", "| Ceiling | train | val |", "|---|---|---|"]
        t, v = o["ceilings"]["train"], o["ceilings"]["val"]
        for k in t:
            if isinstance(t[k], dict):
                for kk in t[k]:
                    L.append(f"| {k} @ {kk} | {t[k][kk]:.3f} | {v[k][kk]:.3f} |")
            else:
                L.append(f"| {k} | {t[k]:.3f} | {v[k]:.3f} |")
        L += ["", f"K0b {'passed' if o['K0b_pass'] else 'FAILED'}: {o['gates']}", ""]
    L += ["## Selections", ""]
    for head in JOBS.SEARCH_HEADS:
        for st in ("A", "B", "C", "D", "aug"):
            s = rec(f"select-{head}-{st}")
            if not s:
                continue
            top = s["ranking"][0]
            extra = "".join(f", {k} {s[k]}" for k in ("picked", "adopted", "level", "team_minus_W", "alb_minus_sym") if k in s)
            L.append(f"- `{head}` stage {st}: best `{top['name']}` ({top['mean']:.4f}){extra}")
    k3 = rec("k3")
    if k3:
        L += ["", "## Gate K3 (hypothesis H1): mask arms on the same validation regions", "",
              "| Arm | mean mask IoU |", "|---|---|"]
        L += [f"| {k} | {v:.3f} |" for k, v in k3["mean_iou"].items()]
        L += ["", f"A − B = {k3['A-B']['mean']:+.3f}, 95 % interval {k3['A-B']['ci95']}: H1 "
              f"{'supported' if k3['H1_supported'] else 'not supported'}.", ""]
    L += ["", "## End to end on validation", "",
          "| Arm | mask AP50 | mask AP | box AP50 | Dice | mIoU | recall@30 |", "|---|---|---|---|---|---|---|"]
    arms = {}
    for ph in ("base", "alb", "p3", "final"):
        for j in JOBS.jobs(ph)[0]:
            r = rec(j["tag"])
            if r and r.get("status") == "done" and r.get("metrics") and j["kind"] in ("e2e", "baseline"):
                arms.setdefault(j["tag"].rsplit("__s", 1)[0], []).append(r["metrics"])
    for arm, ms in sorted(arms.items()):
        L.append(f"| `{arm}` | " + " | ".join(fmt([m.get(k) for m in ms]) for k in
                                              ("mask_AP50", "mask_AP", "box_AP50", "Dice", "mIoU", "recall@30")) + " |")
    sealed = [j for j in JOBS.jobs("final")[0] if j["kind"] == "sealed" and rec(j["tag"])]
    if sealed:
        L += ["", "## Sealed split (150 images; in brackets the 134 never shown in a panel)", "",
              "| Arm | mask AP50 | mask AP | Dice |", "|---|---|---|---|"]
        by = {}
        for j in sealed:
            by.setdefault(j["arm"], []).append(rec(j["tag"]))
        for arm, rs in by.items():
            L.append(f"| {arm} | " + " | ".join(
                f"{fmt([r['metrics'][k] for r in rs])} [{fmt([r['metrics_never_viewed'][k] for r in rs])}]"
                for k in ("mask_AP50", "mask_AP", "Dice")) + " |")
    L += ["", "Limits of the study: 96 % of the images hold one polyp; instances are the connected components of",
          "the mask, so semantic segmentation plus connected components is the process that made the labels;",
          "16 images of the sealed split were viewed during training (never scored).", ""]
    out = PROG / "report" / "REPORT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--price", action="store_true")
    a = ap.parse_args()
    status() if a.status else price() if a.price else report()
