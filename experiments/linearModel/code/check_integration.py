"""Phase 6: G4 on MNIST (ref vs Gram move sequences) + cost summary -> results/integration_summary.json."""

from __future__ import annotations

import json

import numpy as np

from .data import RESULTS


def moves(path):
    z = np.load(path)
    n = len({k.split("_")[0] for k in z.files})
    out = []
    for i in range(n):
        p, qb, qa = z[f"s{i}_perm"], z[f"s{i}_qb"], z[f"s{i}_qa"]
        out.append([(k, int(r), int(qa[k, r]) - int(qb[k, r])) for k in range(p.shape[0]) for r in p[k] if qa[k, r] != qb[k, r]])
    return out


def main():
    d = RESULTS / "integration"
    recs = {p.stem: json.loads(p.read_text()) for p in d.glob("*.json")}
    eq, cost = [], []
    for rid, r in sorted(recs.items()):
        if r.get("config", {}).get("engine") != "ref":
            continue
        gid = rid.replace("__ref__", "__gram__")
        g = recs[gid]
        mr, mg = moves(d / f"{rid}__moves.npz"), moves(d / f"{gid}__moves.npz")
        first_div = next((i for i, (a, b) in enumerate(zip(mr, mg)) if a != b), None)
        Jr, Jg = r["trace"][-1]["J"], g["trace"][-1]["J"]
        eq.append(dict(run=rid, sweeps=len(mr), identical_moves=mr == mg, first_divergent_sweep=first_div,
                       n_moves=sum(map(len, mr)), J_ref=Jr, J_gram=Jg, rel_J_diff=abs(Jr - Jg) / abs(Jr),
                       max_drift_gram=max(t["drift"] for t in g["trace"]),
                       max_drift_ref=max(t["drift"] for t in r["trace"])))
        for x in (r, g):
            tr = x["trace"][1:]
            cost.append(dict(run=x["run_id"], arm=x["arm"], engine=x["config"]["engine"], D_total=x["layout"]["D_total"],
                             gram_build_s=x["timings"].get("gram_build", 0.0),
                             s_per_sweep=float(np.mean([t["seconds"] for t in tr])),
                             accepted_per_sweep=[t["accepted"] for t in tr], J=[t["J"] for t in x["trace"]],
                             frac_at_bound=x["frac_at_bound"], frac_nonzero=x["frac_nonzero"],
                             terminal_improving=x["terminal_audit"]["improving_coordinate_count"],
                             stop=x["stop_reason"], val_acc=x["val"]["acc"], peak_rss_gb=x["peak_rss_gb"]))
    acc = {r["run_id"]: dict(arm=r["arm"], val_acc=r["val"]["acc"], train_acc=r.get("train", {}).get("acc"))
           for r in recs.values()}
    out = dict(equivalence=eq, gate_G4_mnist=all(e["identical_moves"] and e["rel_J_diff"] < 1e-9 for e in eq),
               cost=cost, accuracy=acc)
    (RESULTS / "integration_summary.json").write_text(json.dumps(out, indent=1))
    for e in eq:
        print(f"{e['run'][:60]:60s} identical={e['identical_moves']} moves={e['n_moves']} relJ={e['rel_J_diff']:.1e} drift_g={e['max_drift_gram']:.1e}")
    print("G4 MNIST:", out["gate_G4_mnist"])
    for c in cost:
        print(f"{c['run'][:55]:55s} {c['s_per_sweep']:.3f}s/sweep acc/sw={c['accepted_per_sweep'][::3]} bound={c['frac_at_bound']:.3f} nz={c['frac_nonzero']:.3f}")
    for k, v in sorted(acc.items()):
        print(f"{k[:70]:70s} val={v['val_acc']:.4f} train={v['train_acc']}")


if __name__ == "__main__":
    main()
