"""Run one (benchmark, library, mode, seed) cell and write one results/*.json record.

Always started through code/launch.py, which picks the library's virtualenv, thread count, CPU
pinning and GPU. Direct use (inside the right venv):

    python code/run.py --bench B5 --lib pytm --mode default --seed 0 --threads 1 --out results/x.json

The loop is the one every upstream demo uses: per epoch, one timed `fit` epoch, then one timed
prediction of the whole test set. Accuracy is read at the final epoch (rule P4).
"""
import argparse
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402
from configs import cell_config  # noqa: E402


def run(args):
    cfg = cell_config(args.bench, args.lib, args.mode)
    epochs = args.epochs or cfg["epochs"]
    split = args.seed if cfg.get("splits") else 0
    xtr, ytr, xte, yte = harness.load(cfg["data"], split)
    # run k = a permutation of the training order (pyTsetlinMachine cannot be seeded, so this is
    # the source of run-to-run variance for every library alike; config_map a.11)
    perm = np.random.default_rng(args.seed).permutation(len(ytr))
    xtr, ytr = np.ascontiguousarray(xtr[perm]), np.ascontiguousarray(ytr[perm])
    task = cfg["task"]
    n_classes = int(max(ytr.max(), yte.max()) + 1) if task == "classification" else None

    rec = {"kind": "cell", "bench": args.bench, "lib": args.lib, "mode": args.mode,
           "seed": args.seed, "split": split if cfg.get("splits") else None,
           "threads": args.threads, "device": args.device, "timing_run": args.timing,
           "argv": sys.argv, "git_sha": harness.git_sha(), "config": cfg, "epochs": epochs,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "env": harness.environment(),
           "status": "running", "curve": []}
    harness.write_json(args.out, rec)

    np.random.seed(args.seed)
    mod = importlib.import_module(f"adapters.{args.lib.split('-')[0]}")
    ad = mod.Adapter(cfg, threads=args.threads, device=args.device, seed=args.seed)
    rec["lib_version"] = ad.version()

    last = {}

    def on_epoch(ep, train_s, test_s, pred, **flags):
        pred = np.asarray(pred)
        row = {"epoch": ep, "train_s": train_s, "test_s": test_s, **flags}
        if task == "classification":
            row["accuracy"] = float((pred.astype(np.int64) == yte.astype(np.int64)).mean())
        else:
            row.update(harness.regression_metrics(yte, pred))
        rec["curve"].append(row)
        last["pred"] = pred
        if args.verbose:
            print(json.dumps(row), flush=True)
        if ep % 5 == 0 or ep == epochs:
            harness.write_json(args.out, rec)

    with harness.MemoryProbe() as mem:
        t_protocol = time.perf_counter()
        ad.build(xtr, ytr, n_classes)
        ad.sync()
        rec["build_s"] = time.perf_counter() - t_protocol
        if hasattr(ad, "run_all"):          # green-tsetlin: its Trainer owns the epoch loop
            ad.run_all(xtr, ytr, xte, yte, epochs, on_epoch)
        else:
            for ep in range(1, epochs + 1):
                t0 = time.perf_counter()
                ad.fit_epoch(xtr, ytr, epoch=ep)
                ad.sync()
                t1 = time.perf_counter()
                pred = ad.predict(xte)
                ad.sync()
                on_epoch(ep, t1 - t0, time.perf_counter() - t1, pred)
        # B6 protocol time: construction + all epochs incl. test scoring (GT Table I)
        rec["protocol_total_s"] = time.perf_counter() - t_protocol
    pred = last["pred"]
    rec["memory"] = mem.result()
    rec["memory"].update(ad.extra_memory())

    rec["final"] = (harness.classification_metrics(yte, pred, n_classes) if task == "classification"
                    else harness.regression_metrics(yte, pred))
    tr = [r["train_s"] for r in rec["curve"]]
    te = [r["test_s"] for r in rec["curve"]]
    rec["timing"] = {
        "train_s_epoch1": tr[0],
        "test_s_median_after1": float(np.median(te[1:])) if len(te) > 1 else None,
        "train_s_median_after1": float(np.median(tr[1:])) if len(tr) > 1 else None,
        "test_s_median": float(np.median(te)),
        "train_s_total": float(np.sum(tr)),
        "test_s_total": float(np.sum(te)),
        "n_test": int(len(yte)), "n_train": int(len(ytr)),
        "inference_examples_per_s": float(len(yte) / np.median(te[1:] if len(te) > 1 else te)),
    }
    if task == "classification":
        acc = [r["accuracy"] for r in rec["curve"]]
        rec["best_epoch_logged_only"] = {"epoch": int(np.argmax(acc)) + 1, "accuracy": float(max(acc))}
    size_path = Path(args.out).with_suffix(".model")
    rec["model_size"] = ad.model_size(size_path)
    if size_path.exists() and not args.keep_model:
        size_path.unlink()
    rec["env"]["loadavg_end"] = os.getloadavg()
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["status"] = "done"
    harness.write_json(args.out, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=0, help="override (smoke tests only)")
    ap.add_argument("--timing", action="store_true", help="exclusive timing run (rule P6)")
    ap.add_argument("--keep-model", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception:
        p = Path(args.out)
        rec = json.loads(p.read_text()) if p.exists() else {"argv": sys.argv}
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
        harness.write_json(p, rec)
        raise


if __name__ == "__main__":
    main()
