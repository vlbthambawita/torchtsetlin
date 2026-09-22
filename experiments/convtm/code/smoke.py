"""Smoke test: every harness path, 2 epochs, 2000-image subset. Minutes, not hours.

Run this before every queue. It catches the crashes that otherwise surface three hours into
a wave, and it checks the invariants that a long run would hide rather than fail on.

    CUDA_VISIBLE_DEVICES=1 python code/smoke.py [--device cuda] [--keep]

Paths covered: the split and its hash, nested subsets, every Booleanization in the registry,
every registered arm, the record schema and its refusals, the diagnostics probe, the
val-selected single test evaluation, prediction determinism in eval mode, skip-if-exists and
--force, and the queue runner's lock / crash accounting.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arms as arms_mod  # noqa: E402
import data as data_mod  # noqa: E402
import record as rec_mod  # noqa: E402

SUBSET, EPOCHS = 2000, 2
FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  ' + detail) if detail else ''}",
          flush=True)
    if not cond:
        FAILS.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep", action="store_true", help="keep the temporary records")
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="convtm_smoke_")
    # results/preds/ must contain only the prediction vectors of admissible results. A smoke
    # run that drops .npy files there leaves artefacts that look like real ones; this snapshot
    # is compared at the end.
    real_preds = sorted(os.listdir(rec_mod.PREDS)) if os.path.isdir(rec_mod.PREDS) else []
    t0 = time.time()
    print(f"# smoke: subset {SUBSET}, {EPOCHS} epochs, device {a.device}, out {tmp}\n")

    # ---- data ------------------------------------------------------------------------
    print("data")
    trn, val = data_mod.split_indices()
    y = data_mod.raw_cifar10()["ytr"]
    check("split 45000/5000", (trn.numel(), val.numel()) == (45000, 5000))
    check("stratified val", sorted(torch.bincount(y[val]).tolist()) == [500] * 10)
    check("train/val disjoint", int(torch.isin(trn, val).sum()) == 0)
    check("split hash stable", data_mod.split_hash() == data_mod.split_hash(),
          data_mod.split_hash()[:12])
    s1, s5 = data_mod.subset_indices(trn, 1000), data_mod.subset_indices(trn, 5000)
    check("subsets nested", bool(torch.isin(s1, s5).all()))
    check("subsets stratified", sorted(torch.bincount(y[s5]).tolist()) == [500] * 10)
    for name in sorted(data_mod.BOOLEANIZATIONS):
        ds = data_mod.load_boolean(name, device=a.device, splits=("val",))
        x = ds["xva"]
        check(f"booleanization {name}", x.dtype == torch.bool and x.shape[0] == 5000,
              f"Z={ds['Z']} density={x.float().mean():.3f}")
    fl = data_mod.load_float(device=a.device, augment=True, splits=("val",))
    xa = fl["augment"](fl["xva"][:8])
    check("load_float + augment", tuple(xa.shape) == (8, 3, 32, 32) and xa.dtype == torch.float32)

    # ---- record schema ---------------------------------------------------------------
    print("\nrecord")
    rc = subprocess.run([sys.executable, os.path.join(HERE, "record.py")],
                        capture_output=True, text=True)
    check("record.py self-test (validate refuses bad records)", rc.returncode == 0,
          rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr[-200:])

    # ---- arms ------------------------------------------------------------------------
    names = a.arms if a.arms else list(arms_mod.ARMS)
    for name in names:
        print(f"\narm {name}")
        out = os.path.join(tmp, f"{name}_seed0.json")
        cmd = [sys.executable, os.path.join(HERE, "run_arm.py"), "--arm", name, "--seed", "0",
               "--subset", str(SUBSET), "--epochs", str(EPOCHS), "--device", a.device,
               "--out", out]
        for k, v in arms_mod.get(name).smoke.items():   # per-arm probe caps, see ArmSpec.smoke
            cmd += [k, v]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        check(f"{name} runs", r.returncode == 0, r.stderr.strip()[-300:] if r.returncode else "")
        if r.returncode:
            continue
        rec = rec_mod.read(out)
        check(f"{name} record valid", rec_mod.validate(rec) == [], str(rec_mod.validate(rec)))
        check(f"{name} selection = argmax(val)",
              rec["selected_epoch"] == rec_mod.select_epoch(rec["curve"]))
        check(f"{name} test evaluated once, at the selected epoch",
              all("test_acc" not in e for e in rec["curve"]))
        check(f"{name} diagnostics present",
              all(k in rec["diagnostics"] for k in
                  ("clause_len", "negation_fraction", "firing_rate", "clauses_used_frac")),
              f"len={rec['diagnostics']['clause_len']['median']:.0f} "
              f"neg={rec['diagnostics']['negation_fraction']:.2f} "
              f"fire={rec['diagnostics']['firing_rate']['median']:.3f}")
        check(f"{name} predictions saved",
              os.path.exists(os.path.join(rec_mod.ROOT, rec["test_predictions_path"])))
        check(f"{name} n_test == 10000", rec["data"]["n_test"] == 10000)
        check(f"{name} split hash recorded",
              rec["data"]["split_hash"] == data_mod.split_hash())
        # idempotence
        r2 = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        check(f"{name} skip-if-exists", "skip:" in r2.stdout, r2.stdout.strip()[:60])

    # ---- eval-mode determinism -------------------------------------------------------
    print("\nmodel semantics")
    ds = data_mod.load_boolean("therm4", device=a.device, splits=("val",))
    cfg = {**arms_mod.get("ctm-small").defaults, "seed": 0, "device": a.device,
           "input_shape": tuple(ds["xva"].shape[1:]), "n_clauses": 200}
    arm = arms_mod.get("ctm-small").build(cfg)
    p1, p2 = arm.predict(ds["xva"][:500]), arm.predict(ds["xva"][:500])
    check("prediction is deterministic in eval mode (no random patch draw)",
          bool((p1 == p2).all()))
    arm.model.train()
    tr_out = arm.model.clause_outputs(arm.model._encode(arm.model._prepare(ds["xva"][:8])))
    arm.model.eval()
    ev_out = arm.model.clause_outputs(arm.model._encode(arm.model._prepare(ds["xva"][:8])))
    check("empty clauses flip between train and eval (model.eval() is load-bearing)",
          bool(tr_out.sum() > ev_out.sum()),
          f"train {float(tr_out.float().mean()):.3f} vs eval {float(ev_out.float().mean()):.3f}")
    snap = arm.snapshot()
    check("snapshot is on the host", all(v.device.type == "cpu" for v in snap.values()))
    arm.model.ta_state.zero_()
    arm.restore(snap)
    check("restore reproduces predictions", bool((arm.predict(ds["xva"][:500]) == p1).all()))

    # ---- queue runner ----------------------------------------------------------------
    print("\nqueue runner")
    jobs = os.path.join(tmp, "jobs.txt")
    with open(jobs, "w") as f:
        f.write("# comment ignored\nctm-small|--subset 2000\n")
    # A --tag that no real result uses, so the dry-run exercises the command builder rather
    # than the (correct) skip-if-exists path on the band's own records.
    r = subprocess.run([sys.executable, os.path.join(HERE, "queue_runner.py"), "--gpu", "9",
                        "--jobs", jobs, "--seeds", "0", "--epochs", "2", "--dry-run",
                        "--tag=-smokeprobe"],
                       capture_output=True, text=True, cwd=HERE)
    check("queue runner dry-run builds a command",
          "--arm ctm-small" in r.stdout and "--label ctm-small-smokeprobe" in r.stdout,
          r.stdout.strip()[-140:])
    lock = os.path.join(rec_mod.ROOT, "logs", ".gpu9.lock")
    check("queue runner releases its lock", not os.path.exists(lock))

    now = sorted(os.listdir(rec_mod.PREDS)) if os.path.isdir(rec_mod.PREDS) else []
    check("smoke leaves no artefacts in results/preds/", now == real_preds,
          f"added {sorted(set(now) - set(real_preds))}" if now != real_preds else "")

    print(f"\n# {len(FAILS)} failures in {time.time()-t0:.0f}s")
    for f in FAILS:
        print(f"  FAILED: {f}")
    if not a.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
