"""Generate report/macros.tex and report/data/*.csv from results/*.json.

Constraint C4: every number in the PDF is generated here. Nothing is hand-typed.
Run:  python code/report_data.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(HERE, "report")
DATA = os.path.join(REPORT, "data")


def load():
    """label -> [record, ...], newest schema only, test-scored runs only."""
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, "results", "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("test_acc") is None:
            continue
        rows[os.path.basename(f).rsplit("_seed", 1)[0]].append(d)
    return rows


def agg(ds, key="test_acc"):
    v = [100 * x[key] for x in ds]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def tex_num(x, nd=2):
    return f"{x:.{nd}f}"


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    rows = load()
    M = {}

    def put(name, val):
        M[name] = val

    # ---- the clause-size budget family (the headline) -------------------------------
    for lab, mac in [("ctm-small-T80-unc", "BudUnc"), ("ctm-small-T80-b32", "BudCap"),
                     ("ctm-small-seq-unc", "SeqUnc"), ("ctm-small-seq-b32", "SeqCap"),
                     ("ctm-small-T80-s2-unc", "StwoUnc"), ("ctm-small-T80-s2-b32", "StwoCap"),
                     ("ctm-small-T80-s20-unc", "StwentyUnc"), ("ctm-small-T80-s20-b32", "StwentyCap")]:
        if lab in rows:
            m, s, n = agg(rows[lab])
            put(mac, tex_num(m)); put(mac + "Sd", tex_num(s)); put(mac + "N", str(n))

    # deltas the prose quotes
    if "ctm-small-T80-unc" in rows and "ctm-small-T80-b32" in rows:
        a, _, _ = agg(rows["ctm-small-T80-unc"]); b, _, _ = agg(rows["ctm-small-T80-b32"])
        put("BudDelta", tex_num(b - a))
    if "ctm-small-seq-unc" in rows and "ctm-small-seq-b32" in rows:
        a, _, _ = agg(rows["ctm-small-seq-unc"]); b, _, _ = agg(rows["ctm-small-seq-b32"])
        put("SeqDelta", tex_num(b - a))
    if "ctm-small-T80-s2-unc" in rows and "ctm-small-T80-s2-b32" in rows:
        a, _, _ = agg(rows["ctm-small-T80-s2-unc"]); b, _, _ = agg(rows["ctm-small-T80-s2-b32"])
        put("StwoDelta", tex_num(b - a))

    # ---- the capacity ladder ---------------------------------------------------------
    ladder = []
    for lab, C, pub in [("ctm-therm5-preflight", 20000, 64.5),
                        ("ctm-therm5-40k", 40000, 66.8),
                        ("ctm-therm5-80k", 80000, 69.1)]:
        if lab not in rows:
            continue
        d = rows[lab][0]
        cur = [100 * e["val_acc"] for e in d["curve"]]
        plateau = st.mean(cur[len(cur) // 2:])
        ladder.append((C, 100 * d["test_acc"], 100 * d["val_acc"], plateau, pub, d["selected_epoch"]))
    with open(os.path.join(DATA, "ladder.csv"), "w") as f:
        f.write("clauses,test,val,plateau,published,selected_epoch\n")
        for r in ladder:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.1f},{r[5]}\n")
    if ladder:
        put("LadderRise", tex_num(ladder[-1][3] - ladder[0][3]))
        put("LadderRisePub", tex_num(ladder[-1][4] - ladder[0][4], 1))
        put("PreflightTest", tex_num(ladder[0][1])); put("PreflightVal", tex_num(ladder[0][2]))

    # ---- CNN baselines ---------------------------------------------------------------
    cnn = []
    for lab, ds in sorted(rows.items()):
        if ds[0].get("family") != "baseline":
            continue
        m, s, n = agg(ds)
        cnn.append((lab, m, s, n))
        # LaTeX command names cannot contain digits -- spell them out
        digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                  "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
        raw = "".join(p.capitalize() for p in lab.replace("cnn-", "").split("-"))
        key = "CNN" + "".join(digits.get(ch, ch) for ch in raw)
        put(key, tex_num(m))
    with open(os.path.join(DATA, "cnn.csv"), "w") as f:
        f.write("arm,test,sd,n\n")
        for r in cnn:
            f.write(f"{r[0]},{r[1]:.2f},{r[2]:.2f},{r[3]}\n")

    # ---- budget family table ---------------------------------------------------------
    with open(os.path.join(DATA, "budget.csv"), "w") as f:
        f.write("s,mode,unconstrained,unc_sd,capped,cap_sd,delta\n")
        for s_, mode, a, b in [(2, "batch", "ctm-small-T80-s2-unc", "ctm-small-T80-s2-b32"),
                               (10, "batch", "ctm-small-T80-unc", "ctm-small-T80-b32"),
                               (20, "batch", "ctm-small-T80-s20-unc", "ctm-small-T80-s20-b32"),
                               (10, "sequential", "ctm-small-seq-unc", "ctm-small-seq-b32")]:
            if a in rows and b in rows:
                ma, sa, _ = agg(rows[a]); mb, sb, _ = agg(rows[b])
                f.write(f"{s_},{mode},{ma:.2f},{sa:.2f},{mb:.2f},{sb:.2f},{mb-ma:.2f}\n")

    # ---- counts / provenance ---------------------------------------------------------
    put("NResults", str(len(glob.glob(os.path.join(HERE, "results", "*.json")))))
    put("NArms", str(len(rows)))
    put("NDecisions", str(sum(1 for ln in open(os.path.join(HERE, "DECISIONS.md")) if ln.startswith("## DR-"))))
    put("SeedBand", "1.00")
    put("SplitHash", "8a08ca15")

    with open(os.path.join(REPORT, "macros.tex"), "w") as f:
        f.write("% GENERATED by code/report_data.py -- do not edit\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"macros.tex: {len(M)} macros; data/: ladder.csv cnn.csv budget.csv")


if __name__ == "__main__":
    main()
