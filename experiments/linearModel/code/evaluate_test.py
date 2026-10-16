"""Phase 7 test pass (rule C3): one evaluation of the official 10k test set per frozen
(family, M, arm, seed). Requires ``configs/final.json``; every access is logged in
``results/test_access_log.jsonl`` together with the SHA-256 of ``final.json``.

    python -m experiments.linearModel.code.evaluate_test --dry-run   # list what would be tested
    python -m experiments.linearModel.code.evaluate_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from . import adapter as AD
from . import source_tm as ST
from .data import EXP, load_split, log_test_access
from .hybrid import HybridModel, scores_direct
from .main_grid import CONFIGS, RESULTS, _cfg_key
from .ridge import ridge_solve
from .run import layout_for, make_gates, metrics

TEST = RESULTS / "test"


def runs_to_test():
    final = json.loads((CONFIGS / "final.json").read_text())
    recs = {p.stem: json.loads(p.read_text()) for p in (RESULTS / "main").glob("*.json")}
    out = []
    for o in final:
        fam, M, arm = o["family"], o["M"], o["arm"]
        for r in recs.values():
            g = r["group"]
            gfam = g.split("_")[0]
            if r["arm"] != arm:
                continue
            same_fam = gfam == fam or (fam.startswith("rand") and gfam == fam) or \
                (fam == "flat" and arm in ("A0", "A1", "A3") and gfam == "thr03")
            if not same_fam:
                continue
            gM = int(g.split("_")[1][1:]) if g.split("_")[1].startswith("M") else 0
            if gM != M or (g.endswith("_ref") != (o["group"].endswith("_ref"))):
                continue
            if arm != "A0" and _cfg_key(r["config"]) != _cfg_key(o["config"]):
                continue
            out.append(r)
    seen = {}
    for r in out:
        seen[r["run_id"]] = r
    return list(seen.values())


def test_scores(r, te, device="cuda"):
    arm = r["arm"]
    if arm == "A0":
        model, srec = ST.load(r["source"], device=device)
        B = ST.bool_view(te.X, srec["threshold"], srec["kind"], device)
        with torch.no_grad():
            return torch.cat([model(B[i:i + 2048]) for i in range(0, len(B), 2048)]).float().cpu().numpy()
    if "sizes" in r and r["sizes"].get("artifact", "").endswith(".npz"):
        m = HybridModel.load(EXP / r["sizes"]["artifact"], device=device)
        if m.checksum() != r["sizes"]["checksum"]:
            raise RuntimeError("artifact checksum mismatch")
        return m.predict_scores(te.X)
    # ridge arms: saved theta, else deterministic refit checked against the recorded validation accuracy
    tr, va = load_split(r["manifest"], "train"), load_split(r["manifest"], "valid")
    spec = dict(gates=r["gates"], source=r["source"])
    gates = AD.ConstantGate(10) if r["arm"] in ("A2", "A2c") else make_gates(spec, tr, device)[0]
    if gates.fingerprint != r["gate_fingerprint"] and r["arm"] not in ("A2", "A2c"):
        raise RuntimeError("gate fingerprint differs from the fitted run")
    lay = layout_for(r["arm"], gates)
    p = EXP / "artifacts" / "ridge" / f"{r['run_id']}.npy"
    if p.exists():
        theta = np.load(p)
    else:
        theta = ridge_solve(tr.X, gates(tr.X), tr.Y, lay, r["config"]["lam"])["theta"]
        acc = float((scores_direct(va.X, gates(va.X), lay, theta, 1.0).argmax(1) == va.y).mean())
        if abs(acc - r["val"]["acc"]) > 1e-12:
            raise RuntimeError(f"ridge refit does not reproduce validation accuracy {acc} != {r['val']['acc']}")
        np.save(p, theta)
    return scores_direct(te.X, gates(te.X), lay, theta, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    fsha = hashlib.sha256((CONFIGS / "final.json").read_bytes()).hexdigest()
    runs = runs_to_test()
    print(f"{len(runs)} runs; final.json sha256 {fsha}")
    if a.dry_run:
        for r in sorted(runs, key=lambda r: r["run_id"]):
            print(" ", r["run_id"])
        return
    TEST.mkdir(parents=True, exist_ok=True)
    te = load_split("split_seed0", "test", unseal=True)
    for r in sorted(runs, key=lambda r: r["run_id"]):
        out = TEST / f"{r['run_id']}.json"
        if out.exists():
            continue                                              # one pass per run
        t0 = time.perf_counter()
        S = test_scores(r, te, a.device)
        rec = dict(run_id=r["run_id"], group=r["group"], arm=r["arm"], test=metrics(S, te.y),
                   seconds=time.perf_counter() - t0, final_sha256=fsha)
        out.write_text(json.dumps(rec))
        log_test_access(dict(run_id=r["run_id"], final_sha256=fsha, n_test=len(te)))
        print(f"{r['run_id']}: test acc {rec['test']['acc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
