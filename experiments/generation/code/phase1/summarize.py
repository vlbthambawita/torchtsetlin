"""Phase 1 tables from results/phase1/*.json."""
from __future__ import annotations
import glob, json, statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parents[2] / "results" / "phase1"


def load():
    out = []
    for f in sorted(RES.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


def main():
    rows = load()
    base = [r for r in rows if r["arm"] in ("mnist-marginal",) or "logreg" in r["arm"]]
    tms = [r for r in rows if "-tm-" in r["arm"] or r["arm"].startswith("screen")]
    gens = [r for r in rows if r["arm"].endswith("-gen")]

    print("=== baselines (held-out NLL, nats/image; lower is better) ===")
    for r in base:
        print(f"  {r['arm']:<26} test {r['result'].get('nll_test', float('nan')):>8.2f}   "
              f"val {r['result'].get('nll_val', float('nan')):>8.2f}")

    if tms:
        print("\n=== Tsetlin arms ===")
        print(f"  {'arm':<26}{'T':>6}{'C':>7}{'batch':>7}{'wt':>4}{'exM':>6}"
              f"{'analytic':>10}{'isotonic':>10}{'empty':>7}")
        g = defaultdict(list)
        for r in tms:
            c = r["config"]
            g[(r["arm"], c.get("T"), c.get("clauses"), c.get("batch"),
               int(bool(c.get("weighted"))), c.get("examples"))].append(r)
        for k in sorted(g, key=lambda k: g[k][0]["result"].get("nll_test_isotonic", 9e9)):
            rs = g[k]
            an = [x["result"].get("nll_test_analytic") for x in rs if x["result"].get("nll_test_analytic")]
            iso = [x["result"].get("nll_test_isotonic") for x in rs if x["result"].get("nll_test_isotonic")]
            em = [x["result"].get("empty_clauses", 0) for x in rs]
            arm, T, Cc, b, wt, ex = k
            sd = f" +-{st.pstdev(iso):.2f}" if len(iso) > 1 else ""
            print(f"  {arm:<26}{T:>6.0f}{Cc:>7}{b:>7}{wt:>4}{(ex or 0)/1e6:>6.0f}"
                  f"{(st.mean(an) if an else float('nan')):>10.2f}"
                  f"{(st.mean(iso) if iso else float('nan')):>10.2f}{sd}"
                  f"{st.mean(em):>7.0f}  n={len(rs)}")

    if gens:
        print("\n=== generated-sample quality ===")
        print(f"  {'arm':<26}{'judge':>8}{'distinct':>10}{'pairHam':>9}{'ink':>7}"
              f"{'inkReal':>9}{'NNtrain':>9}{'NNreal':>8}")
        for r in gens:
            q = r["result"]
            print(f"  {r['arm']:<26}{q['judge_accuracy']:>8.3f}"
                  f"{q['distinct_per_digit_mean']:>10.0f}{q['pairwise_hamming_mean']:>9.1f}"
                  f"{q['ink_fraction']:>7.3f}{q['ink_fraction_real']:>9.3f}"
                  f"{q['nn_train_hamming_mean']:>9.1f}"
                  f"{q['nn_train_hamming_real_reference']:>8.1f}")

    print("\n=== gate G1 ===")
    marg = next((r["result"]["nll_test"] for r in base if r["arm"] == "mnist-marginal"), None)
    lg = next((r["result"]["nll_test"] for r in base if "logreg" in r["arm"]), None)
    best = min((r["result"].get("nll_test_isotonic", 9e9) for r in tms), default=9e9)
    bestg = max((r["result"]["judge_accuracy"] for r in gens), default=0.0)
    mind = min((r["result"]["distinct_per_digit_min"] for r in gens), default=0)
    print(f"  NLL: best TM {best:.2f} vs marginal {marg} vs logreg {lg}")
    print(f"  beats marginal: {best < (marg or 9e9)} | beats logreg: {best < (lg or 9e9)}")
    print(f"  judge accuracy {bestg:.3f} (gate >= 0.40) | min distinct/digit {mind} (gate >= 20)")


if __name__ == "__main__":
    main()
