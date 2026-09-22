"""Train one CNN baseline and write one record conforming to `code/record.py`.

Usage
-----
    python code/cnn/train.py --arm cnn-small --seed 0
    python code/cnn/train.py --arm cnn-boolean --seed 0 --epochs 120
    python code/cnn/train.py --arm cnn-small --seed 0 --subset 1000     # sample efficiency
    python code/cnn/train.py --arm cnn-resnet18 --seed 0 --device cuda:1
    python code/cnn/train.py --list                                     # registry
    python code/cnn/train.py --arm cnn-small --seed 0 --smoke           # 2 epochs, 2000 imgs

The protocol this file enforces (PLAN.md §7, CHARTER.md):

* The 45 000 / 5 000 split comes from `data.split_indices()` and nothing else.
* **The test tensor is not loaded while training happens.** `load_*(splits=('train','val'))`
  is used for the training loop; the test split is fetched only after the validation-selected
  weights have been restored. This makes test leakage a structural impossibility, not a
  discipline problem.
* Selection is `argmax(val_acc)`; `record.validate()` recomputes it and refuses the record
  if it disagrees.
* `--subset n` replaces the training split with the nested stratified subset of size ``n``.
  Validation and test are never subsetted.

Sample-efficiency runs write to `results/<arm>-n<subset>_seed<k>.json` so that a curve point
can never be confused with the full-data arm.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.path.dirname(CODE)
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data as D            # noqa: E402  code/data.py
import record as R          # noqa: E402  code/record.py

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(HERE))
    from cnn.arms_cnn import PLANES, SPATIAL, CNN_ARMS, DEFAULTS, CnnArm  # noqa: E402
else:  # imported as a package
    from .arms_cnn import PLANES, SPATIAL, CNN_ARMS, DEFAULTS, CnnArm  # noqa: E402


# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed everything that can move. cuDNN stays benchmarking (deterministic kernels cost
    ~2x here and the programme quantifies seed noise rather than eliminating it -- see
    PLAN.md §7.2, which measures the seed band in P0)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_data(cfg: Dict[str, Any], device: str, subset: Optional[int],
              splits=("train", "val")) -> Dict[str, Any]:
    """Dispatch to the Boolean or the float loader according to the arm's `booleanization`."""
    b = cfg.get("booleanization")
    if b:
        return D.load_boolean(name=b, device=device, splits=splits, subset=subset)  # type: ignore[return-value]
    return D.load_float(device=device, augment=bool(cfg.get("augment")),
                        splits=splits, subset=subset)  # type: ignore[return-value]


def out_path(arm: str, seed: int, subset: Optional[int], out: Optional[str],
             screen: bool = False) -> str:
    """Where the record goes. A screen goes to `screen/`, never to `results/`.

    A screen keeps the *registered* arm name: the 10 000-image subset is part of the protocol
    and is already recorded in `data.subset`, so a `-n10000` suffix would only make the record
    harder to match against `ARMS.md`. Sample-efficiency points in `results/` DO carry the
    suffix, because there the subset size is the variable under study and a curve point must
    never be mistaken for the full-data arm.
    """
    if out:
        return out if os.path.isabs(out) else os.path.join(ROOT, out)
    if screen:
        return os.path.join(ROOT, "screen", f"{arm}_seed{seed}.json")
    tag = f"{arm}-n{subset}" if subset else arm
    return os.path.join(ROOT, "results", f"{tag}_seed{seed}.json")


# --------------------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", help="a key of code/cnn/arms_cnn.py::CNN_ARMS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--schedule", default=None)
    ap.add_argument("--width", type=int, default=None,
                    help="override the architecture width (cnn_small / resnet18 / bnn_vgg)")
    ap.add_argument("--filters", type=int, default=None, help="ctm_shaped: number of filters")
    ap.add_argument("--patch", type=int, default=None, help="ctm_shaped: patch size")
    ap.add_argument("--pool", default=None, help="ctm_shaped: max | mean | logsumexp | topk")
    ap.add_argument("--bool", dest="bool_name", default=None,
                    help="override the Booleanization (therm4 | therm8 | adaptive)")
    ap.add_argument("--augment", dest="augment", action="store_true", default=None)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--subset", type=int, default=None,
                    help="train on a nested stratified subset (1000 | 5000 | 10000)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--list", action="store_true", help="print the arm registry and exit")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 2000 images; writes nothing, just proves it runs")
    ap.add_argument("--screen", action="store_true",
                    help="the PLAN Section 8 P5 screening protocol: 10000 training images, "
                         "15 epochs, 1 seed, validation-selected, written to screen/ and never "
                         "to results/. Overrides --subset/--epochs/--out so that a run labelled "
                         "a screen cannot quietly be something else.")
    a = ap.parse_args(argv)

    if a.list:
        for name, spec in sorted(CNN_ARMS.items()):
            print(f"{name:26s} {spec.family:9s} {spec.doc}")
        return 0
    if not a.arm:
        ap.error("--arm is required (or --list)")
    if a.arm not in CNN_ARMS:
        ap.error(f"unknown arm {a.arm!r}; --list shows the registry")

    spec = CNN_ARMS[a.arm]
    # Merge the module defaults first so the CLI, the registry and CnnArm all see
    # the same fully populated cfg (the arm merges DEFAULTS again; this is what the
    # record's hp block is built from, so it must be complete here too).
    cfg: Dict[str, Any] = {**DEFAULTS, **spec.defaults}
    cfg["arch_kw"] = dict(cfg.get("arch_kw") or {})
    for k, v in (("epochs", a.epochs), ("batch_size", a.batch_size), ("lr", a.lr),
                 ("weight_decay", a.weight_decay), ("schedule", a.schedule),
                 ("augment", a.augment)):
        if v is not None:
            cfg[k] = v
    if a.bool_name is not None:
        cfg["booleanization"] = a.bool_name
        cfg["in_ch"] = PLANES.get(a.bool_name, cfg["in_ch"])
        cfg["input_hw"] = SPATIAL.get(a.bool_name)
    for k, v in (("width", a.width), ("n_filters", a.filters), ("patch", a.patch),
                 ("pool", a.pool)):
        if v is not None:
            cfg["arch_kw"][k] = v

    subset = a.subset
    if a.screen:
        # The screening protocol is fixed by PLAN Section 8 P5. It is applied here rather than
        # left to the caller's command line so that a record in screen/ is guaranteed to have
        # been produced under it -- a screen that was quietly run for 60 epochs on 45k images
        # is not a screen, and there would be no way to tell from the record afterwards.
        cfg["epochs"], subset = 15, 10000
        if a.subset not in (None, 10000) or a.epochs not in (None, 15):
            print("[screen] --subset/--epochs ignored: the screen protocol fixes 10000/15")
    if a.smoke:
        cfg["epochs"], subset = 2, 2000

    set_seed(a.seed)
    device = a.device
    t_start = time.time()

    # ---- data: train + val only. The test tensor is not even materialised yet. ----------
    d = load_data(cfg, device, subset, splits=("train", "val"))
    xtr, ytr = d["xtr"], d["ytr"]
    xva, yva = d["xva"], d["yva"]
    in_ch = int(xtr.shape[1])
    if in_ch != int(cfg["in_ch"]):
        print(f"[train] in_ch {cfg['in_ch']} -> {in_ch} (from the data)")
        cfg["in_ch"] = in_ch
    # The spatial extent comes from the tensor, never from an assumption: the HOG
    # Booleanization of the reference scripts is (N, 5832, 1, 1), not (N, C, 32, 32), and a
    # hard-coded (in_ch, 32, 32) would have built the wrong model AND reported MACs 1024x too
    # high for the arm that DR-003 Decision 2 made the primary HOG evidence.
    hw = (int(xtr.shape[2]), int(xtr.shape[3]))
    if tuple(cfg.get("input_hw") or (32, 32)) != hw:
        print(f"[train] input_hw {cfg.get('input_hw') or (32, 32)} -> {hw} (from the data)")
    cfg["input_hw"] = hw
    if hw != (32, 32) and cfg["augment"]:
        raise SystemExit(f"[train] refusing to augment a {hw} input: crop+flip is defined "
                         "over an image, and HOG bits have no spatial layout to flip.")

    arm = CnnArm(cfg, device=device, augment_fn=d.get("augment"))
    cap = arm.capacity((in_ch, hw[0], hw[1]))
    print(f"[train] arm={a.arm} seed={a.seed} params={cap['n_parameters']:,} "
          f"macs={cap['macs_per_image']:,} n_train={int(xtr.shape[0])} "
          f"bool={cfg.get('booleanization')} augment={cfg['augment']} "
          f"epochs={cfg['epochs']} device={device}", flush=True)

    hp = {k: v for k, v in cfg.items() if k not in ("arch_kw",)}
    hp["arch_kw"] = dict(cfg["arch_kw"])
    hp["n_clauses"] = None
    rec_arm = a.arm if (a.screen or not subset) else f"{a.arm}-n{subset}"
    rec = R.new_record(
        arm=rec_arm, family=spec.family, seed=a.seed,
        argv=["code/cnn/train.py"] + argv, hp=hp,
        data={"dataset": "cifar10", "booleanization": cfg.get("booleanization"),
              "n_train": int(xtr.shape[0]), "n_val": int(xva.shape[0]), "n_test": 10000,
              "augment": ("crop4+flip" if cfg["augment"] else None),
              "split_hash": d["split_hash"], "subset": subset},
        paper=spec.paper, notes=a.notes or spec.doc,
    )

    # ---- train ---------------------------------------------------------------------------
    for ep in range(1, int(cfg["epochs"]) + 1):
        t0 = time.time()
        tr = arm.fit_epoch(xtr, ytr, cfg["batch_size"])
        va = arm.evaluate(xva, yva)
        arm.maybe_select(va)
        dt = time.time() - t0
        R.add_epoch(rec, ep, tr["train_acc"], va, dt, lr=tr["lr"], train_loss=tr["train_loss"])
        print(f"  ep {ep:3d}/{cfg['epochs']}  train {tr['train_acc']:.4f}  val {va:.4f}  "
              f"lr {tr['lr']:.4f}  {dt:.1f}s", flush=True)

    # ---- selection, THEN the single test evaluation ---------------------------------------
    sel = R.select_epoch(rec["curve"])
    arm.restore_selected()
    if arm.selected_epoch != sel:
        raise RuntimeError(
            f"arm restored epoch {arm.selected_epoch} but argmax(val_acc)={sel}; "
            "the arm's own tracker and the record disagree -- refusing to report")
    dte = load_data(cfg, device, None, splits=("test",))
    xte, yte = dte["xte"], dte["yte"]
    preds = arm.predict(xte)
    test_acc = float((preds == yte.cpu()).float().mean())

    diag = arm.diagnostics()
    diag["throughput_img_s"] = round(arm.throughput(xte), 1)
    diag["throughput_batch"] = arm._eval_bs()
    diag["params"] = cap["n_parameters"]
    diag["macs_per_image"] = cap["macs_per_image"]
    diag["val_acc_final_epoch"] = rec["curve"][-1]["val_acc"]
    # For CTM-shaped arms, measure the per-(image, filter) match count on VALIDATION at the
    # selected epoch. Same diagnostic key as the TM side's forthcoming `diagnostics.match_count`
    # so the two substrates can be compared directly.
    mc = arm.match_count_stats(xva)
    if mc:
        diag["match_count"] = mc
        print(f"[train] match_count (val): median|firing={mc['median_given_firing']:.0f} "
              f"mean|firing={mc['mean_given_firing']:.1f} "
              f"P(>=5|firing)={mc['frac_ge5_given_firing']:.3f} "
              f"P(==1|firing)={mc['frac_eq1_given_firing']:.3f} "
              f"fire_frac={mc['fire_frac']:.3f}", flush=True)
    diag["selected_epoch_of"] = int(cfg["epochs"])

    # Prediction vectors belong next to the record they describe. A --smoke run writes no
    # record, and a --out outside results/ is by definition not an admissible result, so
    # neither may drop an .npy into results/preds/ -- that directory feeds the paired
    # McNemar / bootstrap tests and must contain only real results.
    rec_path = out_path(a.arm, a.seed, subset, a.out, screen=a.screen)
    if a.smoke:
        preds_dir = os.path.join(ROOT, "logs", "smoke", "preds")
    else:
        preds_dir = os.path.join(os.path.dirname(os.path.abspath(rec_path)), "preds")
    R.finalize(rec, test_acc=test_acc, test_preds=preds, selected_epoch=sel, capacity=cap,
               diagnostics=diag, wall_s=time.time() - t_start, test_labels=yte,
               preds_dir=preds_dir)

    print(f"[train] selected epoch {sel} (val {rec['val_acc']:.4f})  ->  test {test_acc:.4f}  "
          f"in {rec['wall_s']:.0f}s", flush=True)

    if a.smoke:
        problems = R.validate(rec)
        print("[smoke] record validate():", problems or "OK")
        return 0 if not problems else 1

    path = rec_path
    R.write(rec, path, force=a.force)
    print(f"[train] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
