"""Single-arm driver: trains one (arm, seed) and writes exactly one result record.

    python code/run_arm.py --arm ctm-small --seed 0 [--epochs N] [--subset N]
                           [--bool therm4] [--device cuda] [--out results/x.json] [--force]

**The protocol, made structural.** The training loop is given ``splits=('train','val')``:
the test tensor is not loaded, and there is therefore no test accuracy in existence while
any decision is being made. Selection (``record.select_epoch``) runs on the finished
validation curve; the best-validation automata state is restored from a CPU snapshot; only
then is the test set loaded and the model evaluated on it, once. ``grep -n test`` over this
file shows every occurrence of the word below the ``--- selection is over ---`` line.

Idempotent: refuses to overwrite an existing record unless ``--force``, so a queue can be
re-run after a partial crash without losing or duplicating work.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Optional

import arms as arms_mod
import data as data_mod
import diagnostics as diag_mod
import record as rec_mod
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))


def _sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--label", default=None,
                   help="name this variant is recorded and filed under (default: --arm). "
                        "Use it for a sweep of one arm so that results/ and results/preds/ "
                        "do not collide; the registry entry stays in hp/notes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="default: the arm's own")
    p.add_argument("--subset", type=int, default=None,
                   help="stratified nested training subset (sample-efficiency curve / smoke)")
    p.add_argument("--bool", dest="booleanization", default=None,
                   help=f"one of {sorted(data_mod.BOOLEANIZATIONS)}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--notes", default="")
    # evaluation / diagnostics cost knobs
    p.add_argument("--eval-n", type=int, default=5000,
                   help="training images used for the per-epoch train_acc estimate")
    p.add_argument("--diag-n", type=int, default=2000, help="images in the diagnostic probe")
    p.add_argument("--m2-n", type=int, default=2000,
                   help="validation images in the M2 match-count probe (exact histogram)")
    p.add_argument("--order-n", type=int, default=256,
                   help="training images in the D-ORDER prefix probe (THEORY 5.5.9); 0 = off")
    p.add_argument("--order-clauses", type=int, default=2000,
                   help="clauses subsampled for D-ORDER; 0 = all. Recorded in the field.")
    p.add_argument("--no-diagnostics", action="store_true")
    p.add_argument("--test-curve-last", type=int, default=0,
                   help="also report the PAPER's statistics (mean over the last N epochs, and "
                        "peak) for an `existing`-family arm. Implemented by keeping the last N "
                        "automata snapshots on the host and evaluating test on them AFTER the "
                        "training loop has ended, so no test accuracy exists while any "
                        "decision is being made. Costs N test evaluations and "
                        "N x capacity.state_bytes of host RAM.")
    # arm hyperparameters (override the arm's defaults; recorded in hp either way)
    p.add_argument("--n-clauses", type=int, default=None, help="TOTAL clauses (see arms.py)")
    p.add_argument("--patch", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--T", type=float, default=None)
    p.add_argument("--T-ratio", type=float, default=None)
    p.add_argument("--s", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--position-encoding", dest="position_encoding", default=None,
                   type=lambda v: str(v).lower() in ("1", "true", "yes", "on"))
    p.add_argument("--max-included-literals", type=int, default=None,
                   help="clause-size budget. Omitted = the arm's own default (DR-001 makes that "
                        "32 for most arms); **-1 means explicitly unconstrained**, which is how "
                        "the control arm for the budget is expressed.")
    p.add_argument("--drop-clause-p", type=float, default=None)
    p.add_argument("--feedback-mode", default=None, choices=("batch", "sequential"))
    p.add_argument("--max-chunk-elements", type=int, default=None,
                   help="throughput/memory dial (LG-004); arms are comparable only at equal "
                        "values. Default: the library's 2**27.")
    p.add_argument("--weighted", default=None,
                   type=lambda v: str(v).lower() in ("1", "true", "yes", "on"))
    p.add_argument("--task", default=None, choices=sorted(data_mod.TASKS),
                   help="label map. 'cifar2' is the vehicle/animal merge on the IDENTICAL "
                        "split, encoding and protocol (data.py, ARMS.md A9); it changes the "
                        "labels and n_classes and nothing else.")
    return p


OVERRIDABLE = ("n_clauses", "patch", "stride", "T", "T_ratio", "s", "batch_size",
               "position_encoding", "max_included_literals", "drop_clause_p",
               "feedback_mode", "weighted", "booleanization", "epochs",
               "max_chunk_elements", "task")


def run(a: argparse.Namespace, argv, preds_dir: Optional[str] = None) -> dict:
    spec = arms_mod.get(a.arm)
    cfg = dict(spec.defaults)
    for k in OVERRIDABLE:
        v = getattr(a, k, None)
        if v is not None:
            cfg[k] = v
    # -1 is the only way to say "explicitly unconstrained": None already means "use the arm's
    # default", which since DR-001 is a budget of 32 on most arms. Without this the control
    # arm for the clause-size budget could not be expressed from the command line at all.
    if cfg.get("max_included_literals") is not None and int(cfg["max_included_literals"]) < 0:
        cfg["max_included_literals"] = None
    cfg["seed"] = a.seed
    cfg["device"] = a.device
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])

    t0 = _sync()
    task = str(cfg.get("task") or "cifar10")
    ds = data_mod.load_boolean(cfg["booleanization"], device=a.device,
                              splits=("train", "val"), subset=a.subset, task=task)
    xtr, ytr, xva, yva = ds["xtr"], ds["ytr"], ds["xva"], ds["yva"]
    cfg["input_shape"] = tuple(xtr.shape[1:])
    # The number of output classes is a property of the TASK, not of the harness: a
    # class-owned model divides its clause budget, its vote capacity and therefore its T by
    # it (arms.N_CLASSES).
    cfg["n_classes"] = int(ds["n_classes"])
    cfg["n_order"] = int(a.order_n)
    cfg["order_clauses"] = int(a.order_clauses)
    eval_n = min(int(a.eval_n), xtr.shape[0])
    if not a.no_diagnostics:
        cfg["probe_x"] = xtr[: min(int(a.diag_n), xtr.shape[0])]
        # D-FIRE conditions on the clause's Type-I-eligible class, so it needs the probe
        # labels -- the same slice of the same split as probe_x.
        cfg["probe_y"] = ytr[: min(int(a.diag_n), xtr.shape[0])]
        # M2 is probed on VALIDATION so that it is directly comparable to the CNN-side
        # measurement, which is taken on validation. Validation already drives selection, so
        # a read-only probe of it reveals nothing selection does not; the test split is
        # still untouched until the loop has ended.
        cfg["probe_x_match"] = xva[: min(int(a.m2_n), xva.shape[0])]

    arm = spec.build(cfg)
    resolved = cfg.get("_resolved", {})
    hp = {k: cfg[k] for k in OVERRIDABLE if k in cfg}
    hp.update(resolved)
    hp["input_shape"] = list(cfg["input_shape"])
    hp["config_status"] = ("paper" if (spec.config_status == "paper"
                                       and resolved.get("T_source") == "explicit")
                           else spec.config_status)
    hp["config_gaps"] = list(spec.config_gaps)
    hp["chunk_size_at_batch"] = int(arm.model._chunk_size(xtr[:batch_size]))
    hp["eval_n"] = eval_n
    hp["diag_n"] = 0 if a.no_diagnostics else int(min(a.diag_n, xtr.shape[0]))
    hp["m2_n"] = 0 if a.no_diagnostics else int(min(a.m2_n, xva.shape[0]))
    hp["order_n"] = 0 if a.no_diagnostics else int(min(a.order_n, xtr.shape[0]))
    hp["order_clauses"] = int(a.order_clauses)
    hp["test_curve_last"] = int(a.test_curve_last)

    dat = {"dataset": ds["dataset"], "task": task, "n_classes": int(ds["n_classes"]),
           "booleanization": ds["name"], "n_train": ds["n_train"],
           "n_val": ds["n_val"], "n_test": ds["n_test"], "augment": None,
           "split_hash": ds["split_hash"], "Z": ds["Z"], "subset": a.subset}
    label = a.label or a.arm
    if label != a.arm:
        hp["registry_arm"] = a.arm
    rec = rec_mod.new_record(label, spec.family, a.seed, argv, hp, dat, paper=spec.paper,
                             notes=(spec.notes + (" | " + a.notes if a.notes else "")))
    print(f"=== {label} seed {a.seed} | {json.dumps(resolved)} | {ds['name']} "
          f"{ds['n_train']}/{ds['n_val']} | batch {batch_size} x {epochs} epochs "
          f"| chunk {hp['chunk_size_at_batch']}/{batch_size}", flush=True)
    if spec.paper and hp["config_status"] != "paper":
        src = resolved.get("T_source")
        why = ("T and s are the harness fallback, not the paper's"
               if src != "explicit" else
               "T and s ARE the paper's, but the configuration is incomplete")
        print(f"  !! config_status={hp['config_status']}: {why}. NOT admissible as a "
              "reproduction (record.admissible_as_reproduction). Open gaps:", flush=True)
        for g in (spec.config_gaps or ("(none recorded -- add them to ArmSpec.config_gaps)",)):
            print(f"       - {g}", flush=True)
    cap = resolved.get("vote_capacity")
    # Only a hard warning for an UNWEIGHTED model, where each clause contributes at most 1 and
    # the capacity is exact. With integer weights the sum is unbounded above (weights start at
    # 1 and grow), so `vote_capacity` is a lower bound and T > capacity is legitimate -- the
    # Toolbox's own T=3000 at 2 000 weighted clauses per class is exactly that case. Crying
    # wolf on the flagship arm is how a real warning gets ignored later.
    if cfg.get("weighted") and cap and float(resolved.get("T", 0)) > cap:
        print(f"  .. T={resolved['T']:.0f} exceeds the UNWEIGHTED capacity ({cap}), which is "
              "fine here: clause weights are on, so the achievable sum grows past it. "
              "vote_capacity is a lower bound for this arm.", flush=True)
    elif cap and float(resolved.get("T", 0)) > cap:
        print(f"  !! T={resolved['T']:.0f} EXCEEDS the achievable class sum ({cap}): the margin "
              f"is unreachable, (T-v)/2T stays near 0.5 for every example and T is inert. "
              f"This is ARMS.md convention 2. Set T <= {cap} or use a weighted model.",
              flush=True)
    if hp["chunk_size_at_batch"] < batch_size:
        print(f"  !! chunk {hp['chunk_size_at_batch']} < batch {batch_size}: the mini-batch is "
              f"executed as {-(-batch_size // hp['chunk_size_at_batch'])} sequential passes "
              "(LG-004). Raise --max-chunk-elements, but only in step with the arms you "
              "compare this one against.", flush=True)

    best_val, best_epoch, best_snap = -1.0, None, None
    # PLAN 7.1 forbids test data from reaching any decision; LITERATURE_TM 2.5 says no TM image
    # paper uses a validation split, so their reported number is a last-N-epoch mean or a peak.
    # Both are satisfied by keeping the last N *snapshots* and scoring them after the loop.
    tail = collections.deque(maxlen=max(0, int(a.test_curve_last)))
    if tail.maxlen:
        print(f"  protocol control: keeping the last {tail.maxlen} snapshots to report the "
              f"paper's own statistic; test is still not touched until the loop ends",
              flush=True)
    for ep in range(1, epochs + 1):
        te = _sync()
        arm.fit_epoch(xtr, ytr, batch_size)
        epoch_s = _sync() - te
        tr_acc = arm.accuracy(xtr[:eval_n], ytr[:eval_n])
        va_acc = arm.accuracy(xva, yva)
        d = {} if a.no_diagnostics else arm.diagnostics()
        rec_mod.add_epoch(rec, ep, tr_acc, va_acc, epoch_s, **d)
        if va_acc > best_val:                      # validation, and only validation
            best_val, best_epoch, best_snap = va_acc, ep, arm.snapshot()
        if tail.maxlen:
            tail.append((ep, best_snap if ep == best_epoch else arm.snapshot()))
        print(f"  ep {ep:3d}/{epochs}  train {tr_acc:.4f}  val {va_acc:.4f}  "
              f"({epoch_s:.1f}s)  {diag_mod.one_line(d) if d else ''}"
              f"{'  <- best' if ep == best_epoch else ''}", flush=True)

    # ---------------------------------------------------------------- selection is over ---
    selected = rec_mod.select_epoch(rec["curve"])
    assert selected == best_epoch, (selected, best_epoch)   # two independent paths must agree
    arm.restore(best_snap)
    ts = data_mod.load_boolean(cfg["booleanization"], device=a.device, splits=("test",),
                               task=task)
    preds = arm.predict(ts["xte"])
    test_acc = float((preds == ts["yte"]).float().mean())
    diags = arm.final_diagnostics() if not a.no_diagnostics else {"none": True}
    if tail.maxlen:
        # The paper's own statistic, computed the paper's way, on models chosen without test.
        series = []
        for ep, snap in tail:
            arm.restore(snap)
            series.append({"epoch": ep,
                           "test_acc": float((arm.predict(ts["xte"]) == ts["yte"])
                                             .float().mean())})
        arm.restore(best_snap)                     # leave the selected model in place
        accs = [e["test_acc"] for e in series]
        peak = max(series, key=lambda e: e["test_acc"])
        diags["test_curve"] = series
        diags["paper_statistics"] = {
            "n_last_epochs": len(series),
            "val_selected_test": test_acc,                       # ours (PLAN 7.1)
            "last_n_mean_test": sum(accs) / len(accs),           # theirs, variant 1
            "peak_test": peak["test_acc"], "peak_test_epoch": peak["epoch"],  # theirs, v2
            "penalty_vs_last_n_mean": sum(accs) / len(accs) - test_acc,
            "penalty_vs_peak": peak["test_acc"] - test_acc,
            "note": "val-selected is the headline (PLAN 7.1). The other two are the "
                    "statistics the TM image literature reports, computed over the last N "
                    "epochs only, and exist so a reproduction is not scored against a "
                    "different statistic than the paper used.",
        }
        ps = diags["paper_statistics"]
        print(f"  protocol control: val-selected {test_acc:.4f} | last-{len(series)} mean "
              f"{ps['last_n_mean_test']:.4f} (+{ps['penalty_vs_last_n_mean']*100:.2f} pp) | "
              f"peak {ps['peak_test']:.4f} @ep{ps['peak_test_epoch']} "
              f"(+{ps['penalty_vs_peak']*100:.2f} pp)", flush=True)
    rec_mod.finalize(rec, test_acc, preds, selected, arm.capacity(), diags,
                     wall_s=_sync() - t0, test_labels=ts["yte"], preds_dir=preds_dir)
    print(f"  selected epoch {selected} (val {rec['val_acc']:.4f}) -> test {test_acc:.4f}",
          flush=True)
    return rec


def main() -> int:
    a = build_parser().parse_args()
    out = a.out or os.path.join(ROOT, "results",
                                f"{a.label or a.arm}_seed{a.seed}.json")
    if os.path.exists(out) and not a.force:
        print(f"skip: {out} exists (use --force)", flush=True)
        return 0
    # Predictions live in a `preds/` directory beside the record, not in a fixed global one:
    # a smoke run writing its record to a temporary directory must not drop .npy files into
    # results/preds/, where they would look like the artefacts of an admissible result.
    preds_dir = os.path.join(os.path.dirname(os.path.abspath(out)), "preds")
    rec = run(a, ["run_arm.py"] + sys.argv[1:], preds_dir=preds_dir)
    rec_mod.write(rec, out, force=a.force)
    print(f"-> {out}  val={rec['val_acc']:.4f} test={rec['test_acc']:.4f} "
          f"wall={rec['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
