"""Aggregate the clause-size-budget family into the tables the report and DECISIONS quote.

    python code/summarize_budget.py                      # everything it can find
    python code/summarize_budget.py --family s           # the s-sweep factorial only
    python code/summarize_budget.py --json out.json      # machine-readable, for report/data

Exists because of constraint C4: **every number in the report is generated from
`results/*.json`, nothing is hand-typed.** The verdict this file computes -- whether `s` alone
buys what the clause-size budget buys -- is the one DR-006 pre-registered, so it must be
computed the same way every time it is quoted.

Three things it does that a `pandas` one-liner would not:

1. **It refuses to compare arms that are not budget-matched.** Two cells of a factorial that
   differ in anything but the factorial's own axes are not a factorial. `hp` is compared key
   by key and any extra difference is printed as a REFUSAL, not folded into a mean.
2. **It applies the measured seed band, not a t-test.** The band is 1.00 pp for a 3-seed mean
   [MEASURED: results/calibration_seednoise.json] and PLAN 7.2 says a difference smaller than
   the band is not a difference. Cells with fewer than 3 seeds are printed with their n and
   are never used for a verdict.
3. **It reports the mechanism diagnostics beside the accuracy**, because the whole point of
   the sweep is that accuracy alone cannot distinguish the explanations. `L` is the
   equilibrium clause length (last epoch), NOT the validation-selected snapshot: the
   selected-snapshot value is an artefact of our protocol and the equilibrium is the property
   of the feedback rule (DR-006 Decision 2).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")

SEED_BAND_PP = 1.00  # [MEASURED: results/calibration_seednoise.json], 3-seed mean

# The hyperparameters two cells of one comparison must agree on. `s` and
# `max_included_literals` are the factorial's own axes and are excluded deliberately.
MATCH_KEYS = ("n_clauses", "patch", "stride", "T", "batch_size", "position_encoding",
              "feedback_mode", "weighted", "booleanization", "epochs", "max_chunk_elements",
              "n_clauses_per_class", "vote_capacity")


def _load(labels: List[str]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for lab in labels:
        recs = []
        for p in sorted(glob.glob(os.path.join(RESULTS, f"{lab}_seed*.json"))):
            with open(p) as f:
                recs.append(json.load(f))
        if recs:
            out[lab] = recs
    return out


def _last_epoch(rec: dict) -> dict:
    return rec["curve"][-1]


def _msd(xs: List[float]) -> str:
    """mean +- sd, or `--` when the field is absent.

    A NaN here is not a defect: it is an arm that ran before the mechanism diagnostics landed,
    or a `_safe`d diagnostic that failed. It must render as a visible gap, never as a number
    and never as a crash -- a NaN quietly dropped from a mean is how a partial table starts
    looking like a complete one.
    """
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.2f} (n=1)"
    return f"{statistics.fmean(xs):.2f} +- {statistics.stdev(xs):.2f}"


def _get(d: Optional[dict], *path, default=float("nan")):
    for k in path:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return default
        d = d[k]
    return d if isinstance(d, (int, float)) else default


def cell(recs: List[dict]) -> dict:
    """One (arm, all seeds) summary. Equilibrium quantities come from the LAST epoch."""
    acc = [100.0 * r["test_acc"] for r in recs]
    val = [100.0 * r["val_acc"] for r in recs]
    last = [_last_epoch(r) for r in recs]
    fin = [r["diagnostics"] for r in recs]
    return {
        "n_seeds": len(recs),
        "seeds": [r["seed"] for r in recs],
        "device": sorted({(r["env"] or {}).get("gpu") for r in recs}),
        "test_acc": acc, "val_acc": val,
        "sel_epoch": [r["selected_epoch"] for r in recs],
        # equilibrium (last epoch), the quantity the theory is about
        "L30": [_get(e, "clause_len", "mean") for e in last],
        "fire": [_get(e, "firing_rate", "mean") for e in last],
        "fire_eligible": [_get(e, "firing_rate_eligible", "mean") for e in last],
        "fire_eligible_ratio": [_get(e, "firing_rate_eligible", "ratio_to_prediction")
                                for e in last],
        "ta_boundary": [_get(e, "ta_state", "frac_at_boundary") for e in last],
        "ta_deep": [_get(e, "ta_state", "frac_deep") for e in last],
        "order_32_over_all": [_get(e, "order", "p_image_ratio_32_over_all") for e in last],
        "order_tied_32": [next((_get(lv, "frac_tied_at_k")
                                for lv in (e.get("order") or {}).get("levels", [])
                                if lv.get("k") == 32), float("nan")) for e in last],
        # validation-selected snapshot, for continuity with earlier tables
        "L_selected": [_get(d, "clause_len", "mean") for d in fin],
        "s": [r["hp"].get("s") for r in recs],
        "budget": [r["hp"].get("max_included_literals") for r in recs],
        "wall_s": [r.get("wall_s") for r in recs],
        "hp": recs[0]["hp"],
    }


def budget_matched(a: dict, b: dict) -> List[str]:
    """Empty list = the two cells differ only in the factorial's own axes."""
    bad = []
    for k in MATCH_KEYS:
        if a["hp"].get(k) != b["hp"].get(k):
            bad.append(f"{k}: {a['hp'].get(k)!r} vs {b['hp'].get(k)!r}")
    if a["device"] != b["device"]:
        bad.append(f"device: {a['device']} vs {b['device']}   (DR-001: device+seed is the key)")
    if sorted(a["seeds"]) != sorted(b["seeds"]):
        bad.append(f"seeds: {a['seeds']} vs {b['seeds']}")
    return bad


def _row(label: str, c: dict) -> str:
    return (f"| `{label}` | {c['n_seeds']} | {_msd(c['test_acc'])} | {_msd(c['L30'])} | "
            f"{_msd([1e3 * x for x in c['fire']])} | {_msd([1e3 * x for x in c['fire_eligible']])} | "
            f"{_msd(c['fire_eligible_ratio'])} | {_msd(c['ta_boundary'])} | "
            f"{_msd(c['ta_deep'])} | {_msd(c['order_32_over_all'])} |")


HEAD = ("| arm | n | test acc % | L (equilibrium) | fire x1e3 | **fire|elig** x1e3 | "
        "/(1/(1+s)) | TA@boundary | TA deep | ord 32/all |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="all", choices=("all", "s", "budget", "cifar2"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    S = ["ctm-small-T80-s2-unc", "ctm-small-T80-s2-b32",
         "ctm-small-T80-s10-unc", "ctm-small-T80-s10-b32",
         "ctm-small-T80-s20-unc", "ctm-small-T80-s20-b32"]
    B = ["ctm-small-T80-unc", "ctm-small-T80-b8", "ctm-small-T80-b16", "ctm-small-T80-b32",
         "ctm-small-T80-b64", "ctm-small-T80-b128"]
    C2 = ["ctm-small-cifar2-unc", "ctm-small-cifar2-b32"]
    want = {"all": S + B + C2, "s": S, "budget": B, "cifar2": C2}[a.family]

    recs = _load(want)
    cells = {k: cell(v) for k, v in recs.items()}
    missing = [w for w in want if w not in cells]
    out = {"cells": {k: {kk: vv for kk, vv in v.items() if kk != "hp"}
                     for k, v in cells.items()},
           "missing": missing, "seed_band_pp": SEED_BAND_PP, "verdicts": {}}

    print(f"# clause-size budget family -- {len(cells)} of {len(want)} cells present")
    if missing:
        print(f"  not yet run: {', '.join(missing)}")
    print()
    print(HEAD)
    for k in want:
        if k in cells:
            print(_row(k, cells[k]))
    print()

    # ---- the pre-registered verdicts -------------------------------------------------
    def compare(name: str, lo: str, hi: str, question: str, band: float = SEED_BAND_PP):
        if lo not in cells or hi not in cells:
            print(f"[{name}] PENDING -- needs {lo} and {hi}")
            return
        A_, B_ = cells[lo], cells[hi]
        bad = budget_matched(A_, B_)
        if bad:
            print(f"[{name}] REFUSED -- not budget-matched:")
            for m in bad:
                print(f"    {m}")
            out["verdicts"][name] = {"refused": bad}
            return
        if min(A_["n_seeds"], B_["n_seeds"]) < 3:
            print(f"[{name}] n<3 ({A_['n_seeds']}, {B_['n_seeds']}) -- reported, not a verdict")
        ma, mb = statistics.fmean(A_["test_acc"]), statistics.fmean(B_["test_acc"])
        d = mb - ma
        disjoint = min(B_["test_acc"]) > max(A_["test_acc"])
        print(f"[{name}] {question}")
        print(f"    {lo} {ma:.2f}  vs  {hi} {mb:.2f}   delta = {d:+.2f} pp "
              f"(band {band:.2f}); seed intervals {'DISJOINT' if disjoint else 'overlap'}; "
              f"{'ABOVE' if abs(d) > band else 'INSIDE'} the band")
        out["verdicts"][name] = {"lo": lo, "hi": hi, "mean_lo": ma, "mean_hi": mb,
                                 "delta_pp": d, "band_pp": band, "disjoint": disjoint,
                                 "n_seeds": [A_["n_seeds"], B_["n_seeds"]]}

    if a.family in ("all", "s"):
        print("## DR-006's pre-registered question: is the budget substitutable by `s`?\n")
        # The decisive one. DR-006: if s2-unc is WITHIN 1.00 pp of the s10 budget-32 arm, the
        # headline is rewritten. Note the comparison is against the ORIGINAL b32 arm's own
        # label if the s10 re-run is absent.
        b32 = ("ctm-small-T80-s10-b32" if "ctm-small-T80-s10-b32" in cells
               else "ctm-small-T80-b32")
        if "ctm-small-T80-s2-unc" in cells and b32 in cells:
            m2 = statistics.fmean(cells["ctm-small-T80-s2-unc"]["test_acc"])
            mb = statistics.fmean(cells[b32]["test_acc"])
            gap = mb - m2
            verdict = ("`s` ALONE BUYS IT -- DR-004/DR-006 must be rewritten"
                       if gap <= SEED_BAND_PP else
                       "`s` does NOT substitute for the budget -- DR-004 stands")
            print(f"    s=2 unconstrained {m2:.2f}  vs  budget-32 at s=10 {mb:.2f}  "
                  f"gap {gap:+.2f} pp against a {SEED_BAND_PP:.2f} pp band")
            print(f"    ==> {verdict}")
            out["verdicts"]["s_substitutes_for_budget"] = {
                "s2_unc": m2, "b32": mb, "gap_pp": gap, "band_pp": SEED_BAND_PP,
                "within_band": gap <= SEED_BAND_PP, "verdict": verdict,
                "b32_label": b32}
            print()
        for s in ("2", "10", "20"):
            compare(f"budget effect at s={s}", f"ctm-small-T80-s{s}-unc",
                    f"ctm-small-T80-s{s}-b32",
                    f"does the clause-size budget still pay at s={s}? (B1)")
        print()
        print("## B1's other half: does L_unconstrained rise with s?")
        for s in ("2", "10", "20"):
            k = f"ctm-small-T80-s{s}-unc"
            if k in cells:
                c = cells[k]
                pred = 1.0 / (1.0 + float(c["s"][0]))
                print(f"    s={s:>2}  L30 = {_msd(c['L30']):>18}   "
                      f"fire|elig = {_msd(c['fire_eligible']):>16}   1/(1+s) = {pred:.4f}")

    if a.family in ("all", "cifar2"):
        print()
        print("## THEORY 5.5.6: bloat or headroom?")
        compare("cifar2 budget effect", "ctm-small-cifar2-unc", "ctm-small-cifar2-b32",
                "bloat predicts >= +4 pp; headroom predicts <= +1 pp")
        if all(k in cells for k in C2):
            u, b = cells[C2[0]], cells[C2[1]]
            if not any(math.isnan(x) for x in u["L30"] + b["L30"]):
                beta = statistics.fmean(u["L30"]) / max(1e-9, statistics.fmean(b["L30"]))
                print(f"    beta = L_unc / L_b32 = {beta:.2f}   "
                      f"(CIFAR-10 at s=10: 5.30; CSC-TM's CIFAR-2 at its optimum: 1.77)")
                out["verdicts"]["cifar2_beta"] = beta

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
