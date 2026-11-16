"""Composites of independently trained members (arm I) and seed ensembles (arm E), from the saved
class sums (plan B5). No training.

Rules for the scale alpha_m of member m (score = sum_m sums_m / alpha_m):
    valalpha  max - min of the member's sums over the VALIDATION set   (primary, not transductive)
    fixed     T_m
    paper     max - min over the TEST set (Granmo 2023 Eqs. 7-8; transductive, reference only)
Epoch selection: best validation accuracy of the composite itself (earliest epoch on a tie).

Writes results/<phase>/composite/<team>_seed<k>.json and results/<phase>/ensemble/<member>.json.
If the members' per-epoch states were kept (train.py --keep-epochs), the composite-best epoch is
also written as a team file models/<phase>/Icomp/<team>_seed<k>.pt (the model behind the number).

Usage: python code/composite.py --phase pilot [--seeds 0 1 2]
"""

import argparse

import numpy as np
import torch
from common import (
    CACHE,
    MEMBERS,
    MODELS,
    RESULTS,
    SEEDS,
    TEAMS,
    fmb,
    member_tag,
    new_record,
    read_json,
    write_json,
)

RULE_NAMES = ("valalpha", "fixed", "paper")


def labels(n_val, n_test):
    raw = fmb.load_raw()
    _, va = fmb.split(raw["y_train"])
    return raw["y_train"][va][:n_val], raw["y_test"][:n_test]


def load_sums(phase, tag, names=None):
    """{epoch: {name_split: array}} for a run's saved class sums."""
    d = CACHE / "sums" / phase / "I" / tag
    out = {}
    for f in sorted(d.glob("e*.npz")):
        with np.load(f) as z:
            out[int(f.stem[1:])] = {k: z[k] for k in z.files}
    return out


def alphas(rule, sums, T):
    """Per member: the scale of its sums. ``sums`` = list of (val, test) arrays."""
    if rule == "fixed":
        return [float(t) for t in T]
    i = 0 if rule == "valalpha" else 1
    return [float(s[i].max() - s[i].min()) or 1.0 for s in sums]


def composite(parts, T, yv, yt, epochs):
    """parts: list (per member) of {epoch: (val, test)}. Curve of the three rules + best epochs."""
    curve = []
    for e in epochs:
        sums = [p[e] for p in parts]
        row = {"epoch": e}
        for rule in RULE_NAMES:
            a = alphas(rule, sums, T)
            sv = sum(s[0] / q for s, q in zip(sums, a))
            st = sum(s[1] / q for s, q in zip(sums, a))
            row[rule] = {"val_correct": int((sv.argmax(1) == yv).sum()), "val_acc": 100 * float((sv.argmax(1) == yv).mean()),
                         "test_acc": 100 * float((st.argmax(1) == yt).mean()), "alpha": a}
        curve.append(row)
    best = {}
    for rule in RULE_NAMES:
        b = max(curve, key=lambda r: (r[rule]["val_correct"], -r["epoch"]))
        best[rule] = {"epoch": b["epoch"], "val_acc": b[rule]["val_acc"], "test_acc": b[rule]["test_acc"],
                      "alpha": b[rule]["alpha"]}
    return curve, best


def parts_of(phase, tags, names):
    parts = []
    for tag, n in zip(tags, names):
        s = load_sums(phase, tag)
        parts.append({e: (z[f"{n}_val"], z[f"{n}_test"]) for e, z in s.items()})
    epochs = sorted(set.intersection(*[set(p) for p in parts])) if parts else []
    return parts, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    done = lambda tag: (RESULTS / args.phase / "I" / f"{tag}.json").exists() and \
        read_json(RESULTS / args.phase / "I" / f"{tag}.json").get("status") == "done"   # noqa: E731
    yv = yt = None
    for team, names in TEAMS.items():
        for seed in args.seeds:
            tags = [member_tag(n, seed) for n in names]
            if not all(done(t) for t in tags):
                continue
            parts, epochs = parts_of(args.phase, tags, names)
            if yv is None:
                yv, yt = labels(*[len(a) for a in parts[0][epochs[0]]])
            T = [MEMBERS[n]["T"] for n in names]
            curve, best = composite(parts, T, yv, yt, epochs)
            recs = {n: read_json(RESULTS / args.phase / "I" / f"{t}.json") for n, t in zip(names, tags)}
            rec = new_record("composite", phase=args.phase, team=team, seed=seed, members=names,
                             member_records=tags, curve=curve, best=best, epochs=len(epochs),
                             best_epoch=best["valalpha"]["epoch"], best_val_acc=best["valalpha"]["val_acc"],
                             test_acc_at_best=best["valalpha"]["test_acc"],
                             member_test_acc_at_own_best={n: r["test_acc_at_best"] for n, r in recs.items()})
            e = best["valalpha"]["epoch"]
            files = [CACHE / "epochs" / args.phase / "I" / t / f"e{e:03d}.pt" for t in tags]
            if all(f.exists() for f in files):     # the model behind the composite number
                sts = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
                hp = {"arm": "Icomp", "team": team, "rule": "composite-valalpha", "seed": seed,
                      "members": {n: s["hp"]["members"][n] for n, s in zip(names, sts)},
                      "combine_weights": {n: 1.0 / a for n, a in zip(names, best["valalpha"]["alpha"])},
                      "model_kw": sts[0]["hp"]["model_kw"], "data": sts[0]["hp"]["data"]}
                path = MODELS / args.phase / "Icomp" / f"{team}_seed{seed}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"members": {n: s["members"][n] for n, s in zip(names, sts)}, "hp": hp, "epoch": e,
                            "val_acc": best["valalpha"]["val_acc"], "test_acc": best["valalpha"]["test_acc"],
                            "tag": f"{team}_seed{seed}", "phase": args.phase, "arm": "Icomp",
                            "git_sha": rec["git_sha"], "env": rec["env"]}, path)
                rec["best_model"] = str(path)
            write_json(rec, RESULTS / args.phase / "composite" / f"{team}_seed{seed}.json")
            print(f"composite {team} seed {seed}: valalpha {best['valalpha']['test_acc']:.2f} @ {e}, fixed "
                  f"{best['fixed']['test_acc']:.2f}, paper {best['paper']['test_acc']:.2f} | members "
                  + " ".join(f"{n}:{a:.2f}" for n, a in rec["member_test_acc_at_own_best"].items()), flush=True)
    for n in MEMBERS:                                 # E: one member type over the seeds
        tags = [member_tag(n, s) for s in args.seeds]
        if len(tags) < 2 or not all(done(t) for t in tags):
            continue
        parts, epochs = parts_of(args.phase, tags, [n] * len(tags))
        if yv is None:
            yv, yt = labels(*[len(a) for a in parts[0][epochs[0]]])
        curve, best = composite(parts, [MEMBERS[n]["T"]] * len(tags), yv, yt, epochs)
        write_json(new_record("ensemble", phase=args.phase, member=n, seeds=args.seeds, curve=curve, best=best,
                              best_epoch=best["valalpha"]["epoch"], best_val_acc=best["valalpha"]["val_acc"],
                              test_acc_at_best=best["valalpha"]["test_acc"]),
                   RESULTS / args.phase / "ensemble" / f"{n}.json")
        print(f"ensemble {n} seeds {args.seeds}: {best['valalpha']['test_acc']:.2f}", flush=True)


if __name__ == "__main__":
    main()
