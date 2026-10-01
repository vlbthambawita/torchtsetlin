"""Turn results/g0/*.json into the tables that decide gate G0."""
from __future__ import annotations
import json, glob, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results" / "g0"


def load(sweep):
    out = []
    for f in sorted(RES.glob(f"{sweep}__*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


def agg(rows, keyf, fields):
    """Group by keyf, report mean +- sd over seeds for each field."""
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    out = []
    for k, rs in g.items():
        row = {"key": k, "n_seeds": len(rs)}
        for f in fields:
            vals = [r["result"].get(f) for r in rs if isinstance(r["result"].get(f), (int, float))]
            if vals:
                row[f] = st.mean(vals)
                row[f + "_sd"] = st.pstdev(vals) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def p01_table():
    rows = load("p01")
    if not rows:
        return
    fields = ["mae_analytic", "mae_isotonic", "mae_platt", "slope", "r2",
              "vote_abs_max", "mae_train_semantics", "empty_clauses"]
    a = agg(rows, lambda r: (r["config"]["clauses"], r["config"]["T"]), fields)
    a.sort(key=lambda d: (d["key"][0], d["key"][1]))
    print("\n=== p01: calibration vs clause budget and T "
          "(mae against KNOWN p_true; lower is better) ===")
    print(f"{'clauses':>8} {'T':>6} {'|v|max':>7} {'|v|max/T':>9} "
          f"{'analytic':>10} {'isotonic':>10} {'platt':>8} {'slope':>7} {'r2':>6} {'empty':>6} {'n':>3}")
    for d in a:
        c, T = d["key"]
        vm = d.get("vote_abs_max", float("nan"))
        print(f"{c:>8} {T:>6.0f} {vm:>7.1f} {vm/T:>9.2f} "
              f"{d.get('mae_analytic', float('nan')):>10.4f} "
              f"{d.get('mae_isotonic', float('nan')):>10.4f} "
              f"{d.get('mae_platt', float('nan')):>8.4f} "
              f"{d.get('slope', float('nan')):>7.3f} {d.get('r2', float('nan')):>6.3f} "
              f"{d.get('empty_clauses', float('nan')):>6.0f} {d['n_seeds']:>3}")
    best = min(a, key=lambda d: d.get("mae_isotonic", 9e9))
    print(f"  best (isotonic): clauses={best['key'][0]} T={best['key'][1]:.0f} "
          f"-> {best.get('mae_isotonic'):.4f}  (analytic {best.get('mae_analytic'):.4f})")


def simple_table(sweep, keyf, header, fields=("mae_analytic", "mae_isotonic", "mae", "slope")):
    rows = load(sweep)
    if not rows:
        return
    a = agg(rows, keyf, list(fields))
    a.sort(key=lambda d: str(d["key"]))
    print(f"\n=== {sweep}: {header} ===")
    cols = "".join(f"{f:>14}" for f in fields)
    print(f"{'key':>24}{cols}{'n':>4}")
    for d in a:
        vals = "".join(f"{d.get(f, float('nan')):>14.4f}" for f in fields)
        print(f"{str(d['key']):>24}{vals}{d['n_seeds']:>4}")


def p04_table():
    rows = load("p04")
    if not rows:
        return
    fields = ["mae_all_outputs", "total_variation", "kl_true_to_hat", "top1_match",
              "support_mass", "offsupport_max"]
    a = agg(rows, lambda r: (r["config"]["K"], r["config"]["variant"], r["config"]["T"]), fields)
    a.sort(key=lambda d: (d["key"][0], d["key"][1], d["key"][2]))
    print("\n=== p04: are ALL K outputs calibrated, or only the argmax? ===")
    print(f"{'K':>5} {'variant':>24} {'T':>6} {'mae_all':>9} {'TV':>7} {'KL':>7} "
          f"{'top1':>6} {'supp_mass':>10} {'n':>3}")
    for d in a:
        K, v, T = d["key"]
        print(f"{K:>5} {v:>24} {T:>6.0f} {d.get('mae_all_outputs', float('nan')):>9.5f} "
              f"{d.get('total_variation', float('nan')):>7.3f} "
              f"{d.get('kl_true_to_hat', float('nan')):>7.3f} "
              f"{d.get('top1_match', float('nan')):>6.2f} "
              f"{d.get('support_mass', float('nan')):>10.3f} {d['n_seeds']:>3}")


if __name__ == "__main__":
    p01_table()
    simple_table("p02", lambda r: f"{r['config']['feedback_mode']}/b{r['config']['batch']}",
                 "batch size and feedback mode")
    simple_table("p03a", lambda r: f"contexts={r['config']['n_contexts']}",
                 "context repetition (total examples fixed)")
    simple_table("p03b", lambda r: f"noise_bits={r['config']['n_noise']}",
                 "irrelevant context bits (canvas vs window)",
                 fields=("mae", "within_group_spread", "slope", "r2"))
    p04_table()
    simple_table("p07", lambda r: f"s={r['config']['s']},C={r['config']['clauses']},T={r['config']['T']:.0f}",
                 "specificity x clause budget (T tuned per budget)",
                 fields=("mae_analytic", "mae_isotonic", "slope", "empty_clauses"))
    for sw in ("p05", "p06"):
        for r in load(sw):
            print(f"\n=== {sw} ({r['arm']}) ===")
            print(json.dumps({k: v for k, v in r["result"].items()
                              if not isinstance(v, (list, dict))}, indent=2))
