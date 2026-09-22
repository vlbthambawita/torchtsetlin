"""Price a CNN job list in GPU-minutes, from MEASURED MACs and a MEASURED calibration point.

    python code/cnn/price_jobs.py jobs/p2_cnn.txt jobs/p2_cnn_ref.txt jobs/p2_cnn_curve.txt \
        --seeds 3 --json jobs/p2_cnn_cost.json

Nothing here is hand-typed (constraint C4). Every arm's parameter and MAC count is taken by
instantiating the model the registry would build; the seconds-per-MAC constant is taken from
the only CNN record this programme has measured on the 3080
(`screen/cnn-ctmshape-max_seed0.json`), and re-derived on every call so it tracks new records.

**These are estimates and are labelled as such.** The calibration arm is a single very wide
convolution at batch 64 -- memory-bound, and therefore a pessimistic rate for the dense arms.
Two corrections are applied on top of the MAC model:

* an **AMP factor** (Ampere tensor cores roughly double achieved MAC rate in fp16) and a
  larger factor for pure GEMM arms (the MLPs and linear probes);
* a **kernel-launch floor**, ``n_modules x steps_per_epoch x 3 x 40 us``, which is what makes
  a deep thin net like ResNet-20 cost far more than its 41 MMAC/image suggests.

A 1.3x contingency is then applied. The first arms in the queue replace all of this with
measurement, which is the point of ordering the queue cheapest-first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
for p in (CODE, ROOT, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from cnn.arms_cnn import CNN_ARMS, DEFAULTS, _build_model  # noqa: E402
from cnn import models as M                                # noqa: E402

LAUNCH_US = 40e-6          # per kernel launch, per module, forward; x3 for fwd+bwd
CONTINGENCY = 1.3
N_TRAIN, N_VAL = 45000, 5000


def calibrate() -> float:
    """Effective fp32 MAC/s on the 3080, from the one measured CNN record we have."""
    path = os.path.join(ROOT, "screen", "cnn-ctmshape-max_seed0.json")
    r = json.load(open(path))
    macs = r["capacity"]["macs_per_image"]
    s_ep = sum(e["epoch_s"] for e in r["curve"]) / len(r["curve"])
    n = r["data"]["n_train"]
    return 3.0 * macs * n / s_ep


def price_arm(name: str, args: str, rate: float) -> dict:
    cfg = {**DEFAULTS, **CNN_ARMS[name].defaults}
    cfg["arch_kw"] = dict(cfg.get("arch_kw") or {})
    toks = args.split()
    subset = None
    for i, t in enumerate(toks):
        if t == "--subset" and i + 1 < len(toks):
            subset = int(toks[i + 1])
    hw = tuple(cfg.get("input_hw") or (32, 32))
    model = _build_model(cfg)
    params = M.count_parameters(model, trainable_only=False)
    macs = M.count_macs(model, (int(cfg["in_ch"]), hw[0], hw[1]))
    n_modules = sum(1 for _ in model.modules())
    amp = bool(cfg["amp"]) and cfg["arch"] != "bnn_vgg"
    gemm = cfg["arch"] in ("mlp", "linear")
    eff = rate * ((3.0 if amp else 1.5) if gemm else (2.0 if amp else 1.0))
    n_tr = subset or N_TRAIN
    steps = max(1, -(-n_tr // int(cfg["batch_size"])))
    s_ep = (3.0 * macs * n_tr + macs * N_VAL) / eff + steps * n_modules * 3 * LAUNCH_US + 0.5
    total_s = s_ep * int(cfg["epochs"]) * CONTINGENCY
    return {"arm": name, "args": args.strip(), "subset": subset, "params": params,
            "macs_per_image": macs, "epochs": int(cfg["epochs"]), "batch_size": int(cfg["batch_size"]),
            "booleanization": cfg.get("booleanization"), "augment": bool(cfg["augment"]),
            "s_per_epoch_est": round(s_ep, 2), "minutes_per_seed_est": round(total_s / 60.0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    rate = calibrate()
    print(f"[calib] {rate/1e12:.2f} TMAC/s effective (fp32, batch 64) from "
          f"screen/cnn-ctmshape-max_seed0.json; contingency x{CONTINGENCY}\n")
    out, grand = [], 0.0
    for f in a.files:
        path = f if os.path.isabs(f) else os.path.join(ROOT, f)
        with open(path) as fh:
            jobs = [ln.strip() for ln in fh
                    if ln.strip() and not ln.lstrip().startswith("#")]
        sub = 0.0
        print(f"--- {os.path.basename(path)} "
              f"({len(jobs)} arms x {a.seeds} seeds) ---")
        print(f"{'arm':26s} {'sub':>6s} {'params':>11s} {'MMAC':>7s} {'ep':>4s} "
              f"{'s/ep':>6s} {'min/seed':>9s} {'min x seeds':>11s}")
        rows = []
        for j in jobs:
            name, _, args = j.partition("|")
            rows.append(price_arm(name.strip(), args, rate))
        for r in rows:
            tot = r["minutes_per_seed_est"] * a.seeds
            sub += tot
            print(f"{r['arm']:26s} {str(r['subset'] or '-'):>6s} {r['params']:11,d} "
                  f"{r['macs_per_image']/1e6:7.1f} {r['epochs']:4d} "
                  f"{r['s_per_epoch_est']:6.1f} {r['minutes_per_seed_est']:9.1f} {tot:11.1f}")
            r["seeds"] = a.seeds
            r["minutes_total_est"] = round(tot, 1)
            out.append(r)
        print(f"{'':26s} {'':>6s} {'':>11s} {'':>7s} {'':>4s} {'':>6s} "
              f"{'FILE TOTAL':>9s} {sub:11.1f} min = {sub/60:.1f} h\n")
        grand += sub
    print(f"=== P2 TOTAL (estimate, {a.seeds} seeds): {grand:.0f} GPU-min = {grand/60:.1f} GPU-h ===")
    if a.json:
        p = a.json if os.path.isabs(a.json) else os.path.join(ROOT, a.json)
        json.dump({"calibration_tmac_s": rate / 1e12, "contingency": CONTINGENCY,
                   "seeds": a.seeds, "total_minutes_est": round(grand, 1),
                   "source": "screen/cnn-ctmshape-max_seed0.json",
                   "note": "ESTIMATE from measured MACs + one measured 3080 datapoint; "
                           "replaced by measurement as the queue runs",
                   "jobs": out}, open(p, "w"), indent=1)
        print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
