"""Render ``mnist/REPORT.md`` from ``mnist/report_template.md`` and the MNIST records (rule C2).

    python -m experiments.IMN.code.mnist_report

Every number in the report is a ``{{key}}`` placeholder filled here from ``results/mnist*``;
keys whose records do not exist yet render as *(pending)*. Figures go to ``mnist/figures/``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .mnist import BARCODE, FRACS, PILOT_STEPS, PILOT_TM, pilot_tag

EXP_DIR = Path(__file__).resolve().parents[1]
RES = EXP_DIR / "results"
OUT = EXP_DIR / "mnist"
FIG = OUT / "figures"
PENDING = "*(pending)*"

ARMS = ["lr", "tm", "clause_lr", "tabresnet", "imn", "imn_tm"]
LABEL = {"lr": "Logistic regression", "tm": "TM (source)", "clause_lr": "Clause-LR",
         "tabresnet": "Hypernetwork-only (TabResNet)", "imn": "IMN", "imn_tm": "IMN-TM"}
SHORT = {"lr": "LR", "tm": "TM", "clause_lr": "Clause-LR", "tabresnet": "HN-only", "imn": "IMN", "imn_tm": "IMN-TM"}
# (arm, method) -> display name; the first method of each arm is its primary explanation
METHODS = {
    "lr": [("removal", "occlusion")],
    "tm": [("credit", "clause credit"), ("removal", "occlusion")],
    "clause_lr": [("credit", "clause credit"), ("removal", "occlusion")],
    "tabresnet": [("ig", "Integrated Gradients"), ("removal", "occlusion")],
    "imn": [("removal", "impacts w(x)·x"), ("weights", "weight map w(x)")],
    "imn_tm": [("removal", "removal (local rules)"), ("credit", "clause credit")],
}
COLORS = {"lr": "#7f7f7f", "tm": "#b07aa1", "clause_lr": "#9c755f", "tabresnet": "#4e79a7",
          "imn": "#f28e2b", "imn_tm": "#59a14f"}


# ---------------------------------------------------------------------------------- loading
def load(stage: str, dataset: str) -> Dict[str, List[Dict]]:
    d = RES / stage / f"554_{dataset}"
    out: Dict[str, List[Dict]] = defaultdict(list)
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.loads(p.read_text())
        if r.get("status") == "ok":
            r["_npz"] = p.with_suffix(".npz")
            out[r["arm"] if not r.get("tag") else f"{r['arm']}__{r['tag']}"].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["seed"])
    return out


def ms(vals, scale=1.0, nd=2) -> str:
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float) * scale
    if len(v) == 0:
        return "–"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}" if len(v) > 1 else f"{v.mean():.{nd}f}"


def mean(vals, scale=1.0, nd=2) -> str:
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], float) * scale
    return f"{v.mean():.{nd}f}" if len(v) else "–"


def table(head: List[str], rows: List[List[str]], align: Optional[str] = None) -> str:
    al = align or ("l" + "r" * (len(head) - 1))
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in al]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------- values
def pilot_values(V: Dict[str, str]) -> None:
    P = load("mnist_pilot", "mnist")
    sel_p = RES / "mnist_pilot" / "selection.json"
    sel = json.loads(sel_p.read_text()) if sel_p.exists() else {}
    rows = []
    for t in PILOT_TM:
        rs = P.get(f"tm__{pilot_tag(t)}", [])
        if not rs:
            continue
        r = rs[0]
        chosen = "**✓**" if sel.get("tm") == r["tm_overrides"] else ""
        rows.append([" / ".join(f"{x:g}" for x in t["thresholds"]), str(t["n_clauses"]), f"{t['s']:g}",
                     f"{r['info']['n_bool']}", f"{r['info']['n_clauses_basis']}",
                     f"{100 * r['metrics']['valid']['accuracy']:.2f}", f"{r['info']['train_time'] / 60:.1f}", chosen])
    if rows:
        V["table_pilot_tm"] = table(["thresholds", "clauses/class", "s", "bits", "clauses kept (K)",
                                     "valid acc %", "train min", "chosen"], rows)
    rows = []
    for arm in ("tabresnet", "imn", "imn_tm"):
        for k in PILOT_STEPS:
            rs = P.get(f"{arm}__k{k // 1000}", [])
            if not rs:
                continue
            r = rs[0]
            chosen = "**✓**" if sel.get(arm, {}).get("max_steps") == k else ""
            rows.append([LABEL[arm], f"{k // 1000}k", str(r["info"]["epochs"]),
                         f"{100 * r['metrics']['valid']['accuracy']:.2f}", f"{r['info']['train_time'] / 60:.1f}", chosen])
    if rows:
        V["table_pilot_nn"] = table(["arm", "max steps", "epochs", "valid acc %", "train min", "chosen"], rows)
    if "tm" in sel:
        t = sel["tm"]
        V["sel_tm"] = (f"thresholds {t['thresholds']}, {t['n_clauses']} clauses per class, s = {t['s']:g}, "
                       f"T = {0.5 * t['n_clauses']:g}")
    for arm in ("tabresnet", "imn", "imn_tm", "clause_lr"):
        if arm in sel:
            V[f"sel_steps_{arm}"] = f"{sel[arm]['max_steps'] // 1000}k"


def main_values(V: Dict[str, str]) -> Dict[str, List[Dict]]:
    R = load("mnist", "mnist")
    if not R:
        return R
    V["n_seeds"] = str(max(len(R[a]) for a in R))
    rows = []
    for a in ARMS:
        rs = R.get(a, [])
        if not rs:
            continue
        acc = [r["metrics"]["test"]["accuracy"] for r in rs]
        V[f"acc_{a}"] = ms(acc, 100)
        info = rs[0]["info"]
        params = info.get("n_params")
        K = info.get("n_clauses_basis")
        rows.append([LABEL[a], ms(acc, 100), ms([r["metrics"]["test"]["log_loss"] for r in rs], 1, 3),
                     ms([r["metrics"]["valid"]["accuracy"] for r in rs], 100),
                     f"{params / 1e6:.2f} M" if params else "–", str(K) if K else "–",
                     mean([r["info"].get("train_time") for r in rs], 1 / 60, 1), str(len(rs))])
    V["table_accuracy"] = table(["arm", "test acc %", "test log-loss", "valid acc %", "parameters",
                                 "clauses K", "train min", "seeds"], rows)
    # paired seed-wise differences
    def diff(a, b):
        if not R.get(a) or not R.get(b):
            return None
        sa = {r["seed"]: r["metrics"]["test"]["accuracy"] for r in R[a]}
        sb = {r["seed"]: r["metrics"]["test"]["accuracy"] for r in R[b]}
        s = sorted(set(sa) & set(sb))
        d = np.array([sa[i] - sb[i] for i in s]) * 100
        return d
    for a, b in [("imn_tm", "imn"), ("imn_tm", "tabresnet"), ("imn", "tabresnet"), ("imn_tm", "tm"),
                 ("imn_tm", "clause_lr")]:
        d = diff(a, b)
        if d is not None and len(d):
            V[f"dacc_{a}_{b}"] = f"{d.mean():+.2f} pp ({int((d > 0).sum())}/{len(d)} seeds)"
    # deletion
    rows = []
    for a in ARMS:
        for m, name in METHODS[a]:
            rs = [r for r in R.get(a, []) if f"deletion_{m}" in r["explain"]]
            if not rs:
                continue
            D = [r["explain"][f"deletion_{m}"] for r in rs]
            key = f"{a}_{m}"
            V[f"retgain_{key}"] = ms([d["ret_gain"] for d in D], 1, 3)
            V[f"retauc_{key}"] = ms([d["ret_auc"] for d in D], 1, 3)
            rows.append([LABEL[a], name, ms([d["ret_auc"] for d in D], 1, 3), ms([d["ret_auc_random"] for d in D], 1, 3),
                         "**" + ms([d["ret_gain"] for d in D], 1, 3) + "**", ms([d["auc"] for d in D], 1, 3),
                         ms([d["gain"] for d in D], 1, 3)])
    if rows:
        V["table_deletion"] = table(["arm", "explanation", "retention AUC ↓", "random-order AUC",
                                     "gain ↑", "prob. AUC ↓", "prob. gain ↑"], rows)
    # per-image paired comparisons of retention gain
    pairs = [(("imn_tm", "removal"), ("imn", "removal")), (("imn_tm", "removal"), ("tabresnet", "ig")),
             (("imn", "removal"), ("tabresnet", "ig")), (("imn_tm", "credit"), ("imn_tm", "removal")),
             (("imn_tm", "removal"), ("tm", "credit"))]
    rows = []
    for (a, m), (b, n) in pairs:
        res = paired_gain(R, a, m, b, n)
        if res is None:
            continue
        dm, wins, pmax = res
        V[f"pg_{a}_{m}__{b}_{n}"] = f"{dm:+.3f} ({wins} seeds, max p = {pmax:.1g})"
        rows.append([f"{SHORT[a]} {dict(METHODS[a])[m]}", f"{SHORT[b]} {dict(METHODS[b])[n]}", f"{dm:+.3f}",
                     wins, f"{pmax:.1g}"])
    if rows:
        V["table_deletion_pairs"] = table(["A", "B", "mean gain A − B", "seeds A > B", "largest per-seed Wilcoxon p"],
                                          rows, "llrrr")
    # sanity
    rows = []
    for a in ("tabresnet", "imn", "imn_tm"):
        for m, name in METHODS[a]:
            vals = [r["explain"].get(f"sanity_spearman_{m}") for r in R.get(a, [])]
            if any(v is not None for v in vals):
                V[f"sanity_{a}_{m}"] = ms(vals, 1, 2)
                rows.append([LABEL[a], name, ms(vals, 1, 2)])
    if rows:
        V["table_sanity"] = table(["arm", "explanation", "Spearman ρ, trained vs random model ↓"], rows)
    # readability
    rows = []
    for a in ARMS:
        rs = R.get(a, [])
        if not rs:
            continue
        E = [r["explain"] for r in rs]
        m0 = METHODS[a][0][0]
        g = lambda k, sc=1.0, nd=1: ms([e.get(k) for e in E], sc, nd)  # noqa: E731
        rows.append([LABEL[a], g(f"terms90_{m0}"), g("clause_terms_90"), g("active_clauses_mean"),
                     g("active_clause_len_mean"), g("local_acc_k10", 100), g("local_acc_k50", 100),
                     g("local_vs_full_removal_corr", 1, 2), g("credit_background_share", 100, 0)])
        for k in ("clause_terms_90", "active_clauses_mean", "active_clause_len_mean", "local_acc_k50",
                  "local_vs_full_removal_corr", "credit_background_share", "ink_pixels_mean", f"terms90_{m0}"):
            sc = 100 if k in ("local_acc_k50", "credit_background_share") else 1
            V[f"read_{a}_{k}"] = mean([e.get(k) for e in E], sc, 2 if k == "local_vs_full_removal_corr" else 1)
    V["table_readability"] = table(["arm", "pixels for 90% (primary)", "clauses for 90%", "active clauses",
                                    "active clause length", "local acc k=10 %", "local acc k=50 %",
                                    "local vs full corr", "credit on background %"], rows)
    return R


def paired_gain(R, a, m, b, n):
    from scipy.stats import wilcoxon
    diffs, wins, ps = [], 0, []
    sa = {r["seed"]: r for r in R.get(a, [])}
    sb = {r["seed"]: r for r in R.get(b, [])}
    for s in sorted(set(sa) & set(sb)):
        za, zb = np.load(sa[s]["_npz"]), np.load(sb[s]["_npz"])
        if f"ret_auc_{m}" not in za.files or f"ret_auc_{n}" not in zb.files:
            continue
        assert np.array_equal(za["rows"], zb["rows"])
        ga = za["ret_auc_random"] - za[f"ret_auc_{m}"]
        gb = zb["ret_auc_random"] - zb[f"ret_auc_{n}"]
        d = ga - gb
        diffs.append(d.mean())
        wins += int(d.mean() > 0)
        ps.append(wilcoxon(ga, gb).pvalue if np.any(d != 0) else 1.0)
    if not diffs:
        return None
    return float(np.mean(diffs)), f"{wins}/{len(diffs)}", float(max(ps))


def decoy_values(V: Dict[str, str]) -> Dict[str, List[Dict]]:
    R = load("mnist_decoy", "mnist_decoy")
    if not R:
        return R
    rows, rows2 = [], []
    for a in ARMS:
        rs = R.get(a, [])
        if not rs:
            continue
        E = [r["explain"] for r in rs]
        g = lambda k, sc=1.0, nd=1: ms([e.get(k) for e in E], sc, nd)  # noqa: E731
        rows.append([LABEL[a], g("acc_consistent", 100), g("acc_clean", 100), g("acc_swapped", 100),
                     "**" + g("reliance", 100) + "**", g("n_follow", 1, 0)])
        V[f"rel_{a}"] = mean([e.get("reliance") for e in E], 100, 1)
        V[f"accclean_{a}"] = mean([e.get("acc_clean") for e in E], 100, 1)
        for m, name in METHODS[a]:
            rows2.append([LABEL[a], name, g(f"barcode_hit10_{m}", 100, 0), g(f"barcode_hit10_notfollow_{m}", 100, 0),
                          g(f"barcode_share_{m}", 100, 1)])
            V[f"hit_{a}_{m}"] = mean([e.get(f"barcode_hit10_{m}") for e in E], 100, 0)
            V[f"share_{a}_{m}"] = mean([e.get(f"barcode_share_{m}") for e in E], 100, 1)
    V["table_decoy_acc"] = table(["arm", "consistent acc %", "clean acc %", "swapped acc %",
                                  "reliance % (follows barcode)", "explained swapped images that follow"], rows)
    V["table_decoy_expl"] = table(["arm", "explanation", "barcode in top-10, follows ↑ (%)",
                                   "barcode in top-10, does not follow (%)", "|attr.| on barcode, consistent (%)"],
                                  rows2, "llrrr")
    return R


# ---------------------------------------------------------------------------------- figures
def _imshow(ax, m, img=None, lim=None):
    lim = lim or (np.abs(m).max() or 1.0)
    if img is not None:
        ax.imshow(img.reshape(28, 28), cmap="gray_r", alpha=0.25)
    ax.imshow(m.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.9 if img is not None else 1)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_gallery(R) -> Optional[str]:
    import matplotlib.pyplot as plt
    from .mnist import load_mnist
    cols = [("input", None, None), ("imn", "weights", "IMN\nweight map w(x)"), ("imn", "removal", "IMN\nw(x)·x"),
            ("imn_tm", "removal", "IMN-TM\nremoval"), ("imn_tm", "credit", "IMN-TM\nclause credit"),
            ("tabresnet", "removal", "HN-only\nocclusion"), ("tabresnet", "ig", "HN-only\nInt. Gradients"),
            ("tm", "credit", "TM\nclause credit"), ("lr", "removal", "LR\nocclusion")]
    Z = {}
    for a in {c[0] for c in cols if c[1]}:
        rs = [r for r in R.get(a, []) if r["seed"] == 0]
        if not rs:
            return None
        Z[a] = np.load(rs[0]["_npz"])
    S = load_mnist("mnist", 0)
    show = Z["imn"]["showcase"]
    fig, axes = plt.subplots(len(show), len(cols), figsize=(1.25 * len(cols), 1.3 * len(show)))
    for i, row in enumerate(show):
        x = S.X_test[row]
        for j, (a, m, title) in enumerate(cols):
            ax = axes[i, j]
            if a == "input":
                ax.imshow(x.reshape(28, 28), cmap="gray_r")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(f"y = {S.y_test[row]}", fontsize=8)
            else:
                _imshow(ax, Z[a][f"show_{m}"][i])
                pred = Z[a]["showcase_pred"][i]
                if pred != S.y_test[row]:
                    ax.text(1, 3, f"pred {pred}", color="k", fontsize=6, weight="bold")
            if i == 0:
                ax.set_title(title, fontsize=7)
    fig.suptitle("Explanations of the predicted class (seed 0). Red: supports it; blue: opposes it.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    p = FIG / "gallery.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p.name


def fig_clauses(R) -> Optional[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from .mnist import load_mnist
    rs = [r for r in R.get("imn_tm", []) if r["seed"] == 0]
    rt = [r for r in R.get("tm", []) if r["seed"] == 0]
    if not rs or not rt:
        return None
    Z, ZT = np.load(rs[0]["_npz"]), np.load(rt[0]["_npz"])
    S = load_mnist("mnist", 0)
    picks = [0, 3, 7]
    n_cl = 6
    cmap = ListedColormap(["#ffffff00", "#d62728", "#1f77b4"])
    fig, axes = plt.subplots(2 * len(picks), n_cl + 1, figsize=(1.35 * (n_cl + 1), 1.55 * 2 * len(picks)))
    for b, i in enumerate(picks):
        x = S.X_test[Z["showcase"][i]].reshape(28, 28)
        for r_off, (ZZ, who) in enumerate([(Z, "IMN-TM"), (ZT, "TM")]):
            rr = 2 * b + r_off
            ax = axes[rr, 0]
            ax.imshow(x, cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{who}\npred {ZZ['showcase_pred'][i]}", fontsize=7)
            w = ZZ["show_clause_w"][i]
            for k in range(n_cl):
                ax = axes[rr, k + 1]
                ax.imshow(x, cmap="gray_r", alpha=0.18)
                lab = np.zeros(784)
                lab[ZZ["show_clause_pos"][i, k]] = 1
                lab[ZZ["show_clause_neg"][i, k]] = 2
                ax.imshow(lab.reshape(28, 28), cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"w = {w[k]:+.2f}, L = {ZZ['show_clause_len'][i, k]}", fontsize=6.5)
    fig.suptitle("Largest active rules for the predicted class. Red pixel: must be ink (x ≥ t); "
                 "blue pixel: must be blank (x < t).\nIMN-TM: weights generated for this image; TM: the "
                 "same global votes for every image (seed 0).", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIG / "clauses.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p.name


def fig_deletion(R) -> Optional[str]:
    import matplotlib.pyplot as plt
    if not R:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    styles = ["-", "--"]
    for a in ARMS:
        for si, (m, name) in enumerate(METHODS[a]):
            D = [r["explain"][f"deletion_{m}"] for r in R.get(a, []) if f"deletion_{m}" in r["explain"]]
            if not D:
                continue
            for ax, key in zip(axes, ("ret_curve", "curve")):
                ax.plot(FRACS, np.mean([d[key] for d in D], 0), styles[si], color=COLORS[a], lw=1.8,
                        label=f"{SHORT[a]}: {name}")
        D = [r["explain"][f"deletion_{METHODS[a][0][0]}"] for r in R.get(a, [])]
        if D:
            for ax, key in zip(axes, ("ret_curve_random", "curve_random")):
                ax.plot(FRACS, np.mean([d[key] for d in D], 0), ":", color=COLORS[a], lw=1.0)
    axes[0].set_ylabel("share still predicted as before")
    axes[1].set_ylabel("probability of the original class")
    for ax in axes:
        ax.set_xlabel("fraction of ink pixels removed (highest attribution first)")
        ax.grid(alpha=0.3)
    axes[0].set_title("(a) retention (primary)", fontsize=9)
    axes[1].set_title("(b) probability", fontsize=9)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.text(0.5, 0.005, "Dotted: random pixel order for the same model. Lower curves = more faithful explanation. "
             "Mean over seeds and 500 test images.", ha="center", fontsize=7)
    p = FIG / "deletion.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p.name


def fig_sanity(R) -> Optional[str]:
    import matplotlib.pyplot as plt
    from .mnist import load_mnist
    cols = [("imn", "removal", "IMN w(x)·x"), ("imn", "weights", "IMN w(x)"), ("imn_tm", "removal", "IMN-TM removal"),
            ("imn_tm", "credit", "IMN-TM credit"), ("tabresnet", "ig", "HN-only IG"), ("tabresnet", "removal", "HN-only occl.")]
    Z = {}
    for a in {c[0] for c in cols}:
        rs = [r for r in R.get(a, []) if r["seed"] == 0]
        if not rs:
            return None
        Z[a] = np.load(rs[0]["_npz"])
    S = load_mnist("mnist", 0)
    rows = Z["imn"]["rows"][:3]
    fig, axes = plt.subplots(2 * len(rows), len(cols) + 1, figsize=(1.3 * (len(cols) + 1), 1.35 * 2 * len(rows)))
    for i, row in enumerate(rows):
        x = S.X_test[row]
        for k, lab in enumerate(("trained", "random")):
            ax = axes[2 * i + k, 0]
            ax.imshow(x.reshape(28, 28), cmap="gray_r")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lab, fontsize=7)
            for j, (a, m, title) in enumerate(cols):
                key = f"first_{m}" if k == 0 else f"show_random_{m}"
                if key not in Z[a].files:
                    axes[2 * i + k, j + 1].axis("off")
                    continue
                _imshow(axes[2 * i + k, j + 1], Z[a][key][i])
                if i == 0 and k == 0:
                    axes[0, j + 1].set_title(title, fontsize=7)
    fig.suptitle("Model-randomisation check: explanation of the trained model vs the same architecture "
                 "with random parameters (seed 0)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = FIG / "sanity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p.name


def fig_decoy(R) -> Optional[str]:
    import matplotlib.pyplot as plt
    from .mnist import load_mnist
    arms = [("imn", "removal", "IMN w(x)·x"), ("imn_tm", "removal", "IMN-TM removal"),
            ("tabresnet", "ig", "HN-only IG"), ("tm", "credit", "TM credit")]
    Z = {}
    for a, _, _ in arms:
        rs = [r for r in R.get(a, []) if r["seed"] == 0]
        if not rs:
            return None
        Z[a] = np.load(rs[0]["_npz"])
    S = load_mnist("mnist_decoy", 0)
    n = 5
    if all("swapall_rows" in Z[a] for a, _, _ in arms):
        # show swapped images whose prediction follows the barcode in as many arms as possible
        rows_all = Z["imn"]["swapall_rows"]
        n_follow = sum(Z[a]["swapall_pred"] == S.swap_class[rows_all] for a, _, _ in arms)
        pick = np.argsort(-n_follow, kind="stable")[:n]
        maps = {a: Z[a][f"swapall_{m}"][pick].astype(np.float32) for a, m, _ in arms}
        preds = {a: Z[a]["swapall_pred"][pick] for a, _, _ in arms}
        rows = rows_all[pick]
    else:
        maps = {a: Z[a][f"swap_{m}"][:n] for a, m, _ in arms}
        preds = {a: Z[a]["swap_pred"][:n] for a, _, _ in arms}
        rows = Z["imn"]["swap_rows"][:n]
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(11, 1.75 * n + 1.0))
    gs = fig.add_gridspec(n, len(arms) + 3, width_ratios=[1] * (len(arms) + 1) + [0.35, 3])
    for i in range(n):
        row = rows[i]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(S.X_test_swapped[row].reshape(28, 28), cmap="gray_r")
        ax.add_patch(Rectangle((8.5, -0.5), 10, 1, fill=False, ec="#d62728", lw=0.8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(f"digit {S.y_test[row]}\nbarcode {S.swap_class[row]}", fontsize=7)
        ax.set_xlabel("input", fontsize=6.5, labelpad=9)
        for j, (a, m, title) in enumerate(arms):
            ax = fig.add_subplot(gs[i, j + 1])
            _imshow(ax, maps[a][i])
            ax.add_patch(Rectangle((8.5, -0.5), 10, 1, fill=False, ec="0.4", lw=0.6))
            follows = preds[a][i] == S.swap_class[row]
            ax.set_xlabel(f"pred {preds[a][i]}" + (" (barcode)" if follows else ""), fontsize=6.5,
                          weight="bold" if follows else "normal", labelpad=9)
            # the ten barcode pixels, enlarged, on the same colour scale; lit one outlined
            lim = np.abs(maps[a][i]).max() or 1.0
            ins = ax.inset_axes((0.0, -0.13, 1.0, 0.1))
            ins.imshow(maps[a][i].reshape(28, 28)[0:1, 9:19], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            ins.add_patch(Rectangle((S.swap_class[row] - 0.5, -0.5), 1, 1, fill=False, ec="k", lw=0.8))
            ins.set_xticks([])
            ins.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
    ax = fig.add_subplot(gs[:, -1])
    arms_all = [a for a in ARMS if R.get(a)]
    rel = [np.mean([r["explain"]["reliance"] for r in R[a]]) * 100 for a in arms_all]
    y = np.arange(len(arms_all))
    ax.barh(y, rel, color=[COLORS[a] for a in arms_all])
    ax.set_yticks(y, [SHORT[a] for a in arms_all], fontsize=8)
    ax.set_xlabel("swapped test images that\nfollow the barcode (%)", fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Reliance on the spurious barcode", fontsize=8)
    fig.suptitle("Decoy-MNIST: swapped test images whose barcode (boxed, top row) shows a wrong class, seed 0.\n"
                 "Strip under each map: the ten barcode pixels enlarged; the lit one outlined.", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.01, 0.98, 0.95))
    p = FIG / "decoy.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p.name


# ----------------------------------------------------------------------------------- render
def render() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    V: Dict[str, str] = {}
    pilot_values(V)
    R = main_values(V)
    D = decoy_values(V)
    for fn, data in ((fig_gallery, R), (fig_clauses, R), (fig_deletion, R), (fig_sanity, R), (fig_decoy, D)):
        name = fn(data) if data else None
        V[f"fig_{fn.__name__[4:]}"] = (f"![{fn.__name__[4:]}](figures/{name})" if name else PENDING)
    tpl = (OUT / "report_template.md").read_text()
    missing = set()

    def sub(m):
        k = m.group(1)
        if k not in V:
            missing.add(k)
            return PENDING
        return V[k]
    text = re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", sub, tpl)
    header = ("<!-- Generated by `python -m experiments.IMN.code.mnist_report` from report_template.md and "
              "results/mnist*/ (rule C2). Edit the template, not this file. -->\n\n")
    (OUT / "REPORT.md").write_text(header + text)
    print(f"wrote {OUT / 'REPORT.md'}; {len(V)} values, {len(missing)} pending: {sorted(missing)[:12]}")


if __name__ == "__main__":
    render()
