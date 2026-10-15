"""Phase-1 choice of the clause arms' L1 regulariser on validation AUROC (DECISIONS.md D17).

    python -m experiments.IMN.code.select_reg            # -> results/reg/SELECTION.md + selection.json
"""

from __future__ import annotations

import json

from .report import RESULTS

DEFAULT = "all_0.1"


def main() -> None:
    import pandas as pd
    rows = []
    for p in (RESULTS / "reg").glob("**/*.json"):
        r = json.loads(p.read_text())
        rows.append({"arm": r["arm"], "tag": r["tag"], "dataset": r["dataset"], "seed": r["seed"],
                     "valid": r["metrics"]["valid"]["auroc"], "train": r["metrics"]["train"]["auroc"],
                     "test": r["metrics"]["test"]["auroc"]})
    df = pd.DataFrame(rows)
    per_ds = df.groupby(["arm", "tag", "dataset"])[["valid", "train"]].mean().reset_index()
    n_ds = per_ds.groupby(["arm", "tag"]).dataset.nunique()
    agg = per_ds.groupby(["arm", "tag"])[["valid", "train"]].mean()
    agg["n_datasets"] = n_ds
    agg["n_records"] = df.groupby(["arm", "tag"]).size()
    out, choice = ["# Phase-1 regulariser selection (validation AUROC; test not used)\n"], {}
    for arm, g in agg.groupby(level=0):
        g = g.droplevel(0).sort_values("valid", ascending=False)
        full = g[g.n_datasets == g.n_datasets.max()]
        best = full.index[0]
        if DEFAULT in full.index and full.loc[best, "valid"] - full.loc[DEFAULT, "valid"] <= 0.001:
            best = DEFAULT
        mode, lam = best.split("_")
        choice[arm] = {"l1_mode": mode, "weight_norm": float(lam), "tag": best,
                       "valid_auroc": float(full.loc[best, "valid"])}
        out.append(f"## {arm} -> `{best}`\n")
        out.append("| setting | mean valid AUROC | mean train AUROC | datasets | records |\n|---|---|---|---|---|")
        for t, r in g.iterrows():
            out.append(f"| {t} | {r.valid:.4f} | {r.train:.4f} | {int(r.n_datasets)} | {int(r.n_records)} |")
        out.append("")
    (RESULTS / "reg" / "SELECTION.md").write_text("\n".join(out) + "\n")
    (RESULTS / "reg" / "selection.json").write_text(json.dumps(choice, indent=1))
    print("\n".join(out))
    print(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()
