"""Tables and statistics from ``results/<stage>/**/*.json`` only (rule C2).

    python -m experiments.IMN.code.report --stage main [--tag ""] [--out results/main/TABLES.md]

Outputs (Markdown): per-dataset test AUROC (mean +- std over seeds) with the paper's Table 8
value for IMN alongside, average ranks, Friedman test with the Nemenyi critical difference,
preregistered pairwise Wilcoxon signed-rank tests (Holm-corrected), cost, and the
interpretability metrics of the arms that have local models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS = EXP_DIR / "results"

ARM_ORDER = ["lr", "dt", "rf", "gbdt", "tabresnet", "imn", "tm", "clause_lr", "imn_tm", "imn_tm_rand",
             "imn_hybrid", "tm_gated"]
# Preregistered comparisons (PLAN.md section 4): (arm A, arm B, question id)
PAIRS: List[Tuple[str, str, str]] = [
    ("imn_tm", "tm", "Q1"), ("imn_tm", "imn", "Q2"), ("imn_hybrid", "imn", "Q3"),
    ("imn_hybrid", "imn_tm", "Q3"), ("tm_gated", "lr", "Q4"), ("tm_gated", "tm", "Q4"),
    ("tm_gated", "imn", "Q4"), ("imn_tm", "clause_lr", "Q5"), ("imn_tm", "imn_tm_rand", "Q6"),
    ("imn", "tabresnet", "R"),
]


def load(stage: str, tag: Optional[str] = "") -> pd.DataFrame:
    """Records of ``stage`` with ``tag`` (``None`` = every tag); skipped runs excluded."""
    rows = []
    for p in sorted((RESULTS / stage).glob("*/*__s*.json")):  # records only (not selection/winners files)
        r = json.loads(p.read_text())
        if (tag is not None and r.get("tag", "") != tag) or r.get("status", "ok") != "ok":
            continue
        row = {"dataset_id": r["dataset_id"], "dataset": r["dataset"], "arm": r["arm"], "seed": r["seed"],
               "tag": r.get("tag", ""),
               "train_time": r["info"].get("train_time"), "inference_time": r["info"].get("inference_time"),
               "wall_time": r["wall_time"], "n_clauses_basis": r["info"].get("n_clauses_basis"),
               "n_params": r["info"].get("n_params")}
        for split, m in r["metrics"].items():
            for k, v in m.items():
                row[f"{split}_{k}"] = v
        for k, v in r.get("explain", {}).items():
            if isinstance(v, (int, float)):
                row[f"x_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def _arms(df: pd.DataFrame) -> List[str]:
    present = set(df.arm)
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


def auroc_table(df: pd.DataFrame, paper: dict) -> str:
    arms = _arms(df)
    g = df.groupby(["dataset_id", "dataset", "arm"])["test_auroc"]
    mean, std, cnt = g.mean().unstack("arm"), g.std().unstack("arm"), g.count().unstack("arm")
    head = "| dataset | " + " | ".join(arms) + " | paper IMN (T8) |"
    lines = [head, "|" + "---|" * (len(arms) + 2)]
    for (did, name), r in mean.iterrows():
        best = np.nanmax(r.values)
        cells = []
        for a in arms:
            if a not in r or pd.isna(r[a]):
                cells.append("–")
                continue
            s = std.loc[(did, name), a]
            txt = f"{r[a]:.3f}" + (f" ± {s:.3f}" if cnt.loc[(did, name), a] > 1 and not pd.isna(s) else "")
            cells.append(f"**{txt}**" if np.isclose(r[a], best) else txt)
        ref = paper.get(str(did), {}).get("imn")
        lines.append(f"| {name} ({did}) | " + " | ".join(cells) + f" | {ref if ref is not None else '–'} |")
    return "\n".join(lines)


def ranks_and_friedman(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    mean = df.groupby(["dataset_id", "arm"])["test_auroc"].mean().unstack("arm")
    mean = mean[[a for a in _arms(df) if a in mean]].dropna(axis=0, how="any")
    if mean.shape[0] < 2 or mean.shape[1] < 3:
        return "(not enough complete datasets for ranks)", mean
    rk = mean.rank(axis=1, ascending=False).mean().sort_values()
    k, n = mean.shape[1], mean.shape[0]
    fr = stats.friedmanchisquare(*[mean[c].values for c in mean.columns])
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    out = ["| arm | average rank |", "|---|---|"] + [f"| {a} | {v:.2f} |" for a, v in rk.items()]
    out.append(f"\nFriedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3g} over {n} datasets x {k} arms; "
               f"Nemenyi CD (alpha 0.05) = {cd:.2f} ranks.")
    return "\n".join(out), mean


def pairwise(mean: pd.DataFrame) -> str:
    rows = []
    for a, b, q in PAIRS:
        if a not in mean or b not in mean:
            continue
        d = (mean[a] - mean[b]).dropna()
        if len(d) < 2:
            continue
        try:
            p = stats.wilcoxon(d.values, zero_method="wilcox").pvalue if np.any(d.values != 0) else 1.0
        except ValueError:
            p = float("nan")
        g = np.random.default_rng(0)
        boot = d.values[g.integers(len(d), size=(10000, len(d)))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append([q, a, b, len(d), d.mean(), lo, hi, d.median(), int((d > 0).sum()), int((d < 0).sum()), p])
    if not rows:
        return "(no complete pairs)"
    ps = np.array([r[-1] for r in rows])
    order = np.argsort(ps)
    holm = np.empty_like(ps)
    m = len(ps)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (m - i) * ps[j]))
        holm[j] = running
    lines = ["| Q | A | B | datasets | mean A−B | bootstrap 95% CI | median A−B | A wins | B wins | "
             "p (Wilcoxon) | p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, ph in zip(rows, holm):
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:+.4f} | [{r[5]:+.4f}, {r[6]:+.4f}] | "
                     f"{r[7]:+.4f} | {r[8]} | {r[9]} | {r[10]:.3g} | {ph:.3g} |")
    return "\n".join(lines)


def per_arm_table(df: pd.DataFrame, cols: List[str], fmt: str = "{:.3f}") -> str:
    cols = [c for c in cols if c in df]
    if not cols:
        return "(none)"
    agg = df.groupby("arm")[cols].mean()
    agg = agg.loc[[a for a in _arms(df) if a in agg.index]].dropna(how="all")
    lines = ["| arm | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for a, r in agg.iterrows():
        lines.append(f"| {a} | " + " | ".join("–" if pd.isna(v) else fmt.format(v) for v in r.values) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = load(a.stage, a.tag)
    if df.empty:
        raise SystemExit(f"no records under results/{a.stage}")
    paper = json.loads((EXP_DIR / "reference" / "paper_tables.json").read_text())["table8_default_hp"]
    rk, mean = ranks_and_friedman(df)
    parts = [
        f"# Results: stage `{a.stage}`" + (f" tag `{a.tag}`" if a.tag else ""),
        "Generated by `code/report.py` from the JSON records; do not hand-edit.\n",
        f"{len(df)} records, {df.dataset_id.nunique()} datasets, arms: {', '.join(_arms(df))}, "
        f"seeds: {sorted(df.seed.unique().tolist())}.\n",
        "## Test AUROC", auroc_table(df, paper),
        "\nThe last column is the paper's Table 8 (default hyper-parameters, its own 80/20 split, 10 seeds) "
        "and is a sanity reference only.",
        "## Average rank (test AUROC)", rk,
        "## Preregistered pairwise comparisons (dataset-level means, two-sided)", pairwise(mean),
        "## Accuracy and log-loss", per_arm_table(df, ["test_accuracy", "test_log_loss", "valid_auroc", "train_auroc"]),
        "## Cost", per_arm_table(df, ["train_time", "inference_time", "n_params", "n_clauses_basis"], "{:.4g}"),
        "## Interpretability (test sub-sample)",
        per_arm_table(df, ["x_linear_terms_90", "x_clause_terms_90", "x_active_clauses_mean",
                           "x_active_clause_len_mean", "x_local_vs_full_removal_corr",
                           "x_local_acc_k10", "x_local_acc_k25", "x_local_acc_k50"]),
    ]
    if "x_gt_corr" in df:
        parts += ["## Synthetic attribution quality", per_arm_table(df, ["x_gt_corr", "x_faithfulness", "x_infidelity"])]
    text = "\n\n".join(parts) + "\n"
    out = Path(a.out) if a.out else RESULTS / a.stage / ("TABLES" + (f"_{a.tag}" if a.tag else "") + ".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
