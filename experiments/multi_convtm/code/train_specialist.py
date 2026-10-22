"""Train one TM Composites specialist on CIFAR-10 and save its test class sums every epoch.

Configurations are Table 1 of Granmo (2023), arXiv:2309.04801, CIFAR-10 column, cross-checked
against the authors' scripts (see README.md). Protocol as in the paper: full 50 000 training
images, 100 epochs, test accuracy after every epoch, no validation-based selection.

Usage:  python code/train_specialist.py --spec C3 --seed 0 [--device cuda:0]
        python code/train_specialist.py --spec C3 --calibrate 200   # time 200 batches, no record
"""

import argparse
import sys
import time

import numpy as np
import torch
from common import CACHE, RESULTS, load_cifar10, new_record, write_json
from encoders import ENCODERS

import torchtsetlin as tt
from torchtsetlin.utils import chunk_indices

# Table 1, CIFAR-10 column. "2K clauses" is per class: the scripts pass number_of_clauses=2000 to
# tmu's TMClassifier, which builds one clause bank of that size per class.
SPECIALISTS = {
    "T10": dict(name="10x10 Thresholding", encoding="thresh", model="conv", patch=10,
                n_clauses=2000, T=500, s=10.0, weighted=True),
    "C3": dict(name="3x3 Color Thermometers", encoding="therm8", model="conv", patch=3,
               n_clauses=2000, T=1500, s=2.5, weighted=True),
    "C4": dict(name="4x4 Color Thermometers", encoding="therm8", model="conv", patch=4,
               n_clauses=2000, T=1500, s=2.5, weighted=True),
    "HoG": dict(name="Histogram of Gradients", encoding="hog", model="flat", patch=None,
                n_clauses=2000, T=50, s=10.0, weighted=False),
}
PAPER_ACC_EPOCH100 = {"T10": 57.0, "C3": 62.1, "C4": 62.7, "HoG": 63.5}   # Table 2, CIFAR-10
MAX_INCLUDED_LITERALS = 32                                                 # Table 1 caption


def build_model(cfg, n_features, input_shape, max_chunk_elements, feedback_mode="batch",
                max_included_literals=MAX_INCLUDED_LITERALS, budget_mode="hard"):
    common = dict(weighted=cfg["weighted"], max_included_literals=max_included_literals,
                  feedback_mode=feedback_mode, max_chunk_elements=max_chunk_elements)
    # Always explicit: the library default changed from "hard" (0.3.x) to "soft", and the
    # original reproduction records are "hard".
    common["budget_mode"] = budget_mode
    if cfg["model"] == "conv":
        # TMU's convolution appends thermometer-encoded patch coordinates; so does this.
        return tt.ConvTsetlinMachine(10, cfg["n_clauses"], cfg["T"], cfg["s"],
                                     patch_size=cfg["patch"], stride=1, position_encoding=True,
                                     input_shape=input_shape, **common)
    return tt.TsetlinMachine(n_features, 10, cfg["n_clauses"], cfg["T"], cfg["s"], **common)


@torch.no_grad()
def class_sums(model, x, eval_batch):
    """Unclamped class sums, as tmu's ``predict(..., return_class_sums=True)`` returns them.

    ``model.forward()`` clamps vote sums to [-T, T] (classifier.py ``_votes(clamp=True)``), which
    the composite's normalisation cannot use: tmu does not clamp, and 72-78% of the authors'
    released convolutional class sums lie outside [-T, T]. This mirrors ``forward()`` line for line
    with ``clamp=False`` (LIBRARY_GAPS.md MC-1); ``check_against_forward`` proves the equivalence.
    """
    was = model.training
    model.eval()                     # empty clauses evaluate False when predicting
    try:
        out = []
        for i in range(0, x.shape[0], eval_batch):
            xb = model._prepare(x[i:i + eval_batch])
            for sl in chunk_indices(xb.shape[0], model._chunk_size(xb)):
                o, _ = model._evaluate(model._encode(xb[sl]), empty_value=False)
                out.append(model._votes(o, clamp=False))
        return torch.cat(out)
    finally:
        model.train(was)


@torch.no_grad()
def check_against_forward(model, x, sums):
    was = model.training
    model.eval()
    try:
        ref = model(x)
    finally:
        model.train(was)
    # Before the MC-1 fix forward() clamped to [-T, T]; after it, forward() is the raw sum.
    expect = sums.clamp(-model.T, model.T) if getattr(model, "clamp_prediction_votes", True) else sums
    if not torch.equal(ref, expect):
        raise RuntimeError("class sums disagree with model.forward()")


def encode(cfg):
    xtr, ytr, xte, yte = load_cifar10()
    t0 = time.time()
    enc = ENCODERS[cfg["encoding"]]
    btr, bte = enc(xtr), enc(xte)
    return btr, ytr, bte, yte, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECIALISTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=500)
    ap.add_argument("--max-chunk-elements", type=int, default=2**29)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--calibrate", type=int, default=0, help="time this many batches and exit")
    ap.add_argument("--feedback-mode", default="batch", choices=["batch", "sequential"])
    ap.add_argument("--max-included-literals", type=int, default=MAX_INCLUDED_LITERALS,
                    help="clause budget; -1 = none (diagnostics only)")
    ap.add_argument("--label", default="", help="suffix for diagnostic runs, e.g. bs10")
    ap.add_argument("--budget-mode", default="hard", choices=["hard", "soft"])
    ap.add_argument("--variant", default="", help="named full-protocol variant, e.g. softbudget "
                    "(recorded as kind=specialist with this variant; composites are per variant)")
    args = ap.parse_args()

    cfg = SPECIALISTS[args.spec]
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)

    btr, ytr, bte, yte, enc_s = encode(cfg)
    # Booleanize on the CPU, then move once; plain blocking copies (CLAUDE.md: never
    # non_blocking device-to-host or pageable host-to-device tricks).
    xtr = torch.from_numpy(btr).to(dev)
    xte = torch.from_numpy(bte).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    yte_np = yte
    n_features = int(np.prod(btr.shape[1:]))
    input_shape = tuple(btr.shape[1:]) if cfg["model"] == "conv" else None
    budget = None if args.max_included_literals < 0 else args.max_included_literals
    model = build_model(cfg, n_features, input_shape, args.max_chunk_elements,
                        feedback_mode=args.feedback_mode, max_included_literals=budget,
                        budget_mode=args.budget_mode).to(dev)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 7919)

    hp = {**cfg, "max_included_literals": budget, "batch_size": args.batch_size,
          "epochs": args.epochs, "feedback_mode": args.feedback_mode, "budget_mode": args.budget_mode,
          "position_encoding": cfg["model"] == "conv",
          "max_chunk_elements": args.max_chunk_elements, "n_states": int(model.n_states),
          "chunk_size_at_batch": int(model._chunk_size(model._prepare(xtr[:args.batch_size])))}
    print(f"[{args.spec}] {cfg['name']} | x {tuple(xtr.shape)} | clauses {model.n_clauses_total} total "
          f"| literals {model.n_literals} | chunk {hp['chunk_size_at_batch']}/{args.batch_size} "
          f"| encode {enc_s:.1f}s | {args.device}", flush=True)

    if args.calibrate:
        perm = torch.randperm(xtr.shape[0], generator=gen).to(dev)
        model.train()
        for i in range(5):                                   # warm-up
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            model.update(xtr[idx], ytr_t[idx])
        torch.cuda.synchronize(dev)
        t0 = time.time()
        for i in range(5, 5 + args.calibrate):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            model.update(xtr[idx], ytr_t[idx])
        torch.cuda.synchronize(dev)
        per_batch = (time.time() - t0) / args.calibrate
        t0 = time.time()
        sums = class_sums(model, xte[:2000], args.eval_batch)
        torch.cuda.synchronize(dev)
        ev = (time.time() - t0) * xte.shape[0] / 2000
        epoch_s = per_batch * xtr.shape[0] / args.batch_size
        print(f"[{args.spec}] calibrate: {per_batch*1000:.1f} ms/batch -> train {epoch_s:.0f} s/epoch, "
              f"test {ev:.0f} s/epoch -> {(epoch_s+ev)*args.epochs/3600:.1f} h for {args.epochs} epochs "
              f"| peak {torch.cuda.max_memory_reserved(dev)/2**30:.1f} GiB", flush=True)
        check_against_forward(model, xte[:args.eval_batch], sums[:args.eval_batch])
        return

    suffix = args.variant or args.label
    tag = f"{args.spec}_seed{args.seed}" + (f"_{suffix}" if suffix else "")
    out = RESULTS / f"{tag}.json"
    sums_dir = CACHE / "class_sums" / tag
    sums_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CACHE / "ckpt" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    rec = new_record("diagnostic" if args.label else "specialist", spec=args.spec, seed=args.seed,
                     label=args.label or None, variant=args.variant or None, hp=hp,
                     paper={"ref": "Granmo 2023, arXiv:2309.04801, Tables 1-2",
                            "test_acc_epoch100": PAPER_ACC_EPOCH100[args.spec]},
                     data={"dataset": "CIFAR-10", "n_train": int(xtr.shape[0]),
                           "n_test": int(xte.shape[0]), "encoding_s": enc_s,
                           "n_features": n_features, "shape": list(btr.shape[1:])},
                     curve=[], status="running")
    start_epoch = 0
    if ckpt.exists():                                       # resume an interrupted run
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        gen.set_state(state["gen"])
        torch.set_rng_state(state["cpu_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], dev)
        rec = state["rec"]
        rec["status"] = "running"
        rec["hp"]["epochs"] = args.epochs
        rec.setdefault("resumed_at", []).append({"epoch": state["epoch"], "argv": sys.argv})
        start_epoch = state["epoch"]
        print(f"[{args.spec}] resumed after epoch {start_epoch}", flush=True)

    t_run = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(xtr.shape[0], generator=gen).to(dev)
        correct = 0
        for i in range(0, xtr.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            votes = model.update(xtr[idx], ytr_t[idx])        # pre-update (clamped) vote sums
            correct += int((votes.argmax(1) == ytr_t[idx]).sum())
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        train_s = time.time() - t0

        t0 = time.time()
        sums = class_sums(model, xte, args.eval_batch)
        check_against_forward(model, xte[:args.eval_batch], sums[:args.eval_batch])
        test_s = time.time() - t0
        s_np = sums.cpu().numpy()
        np.save(sums_dir / f"e{epoch:03d}.npy", s_np.astype(np.float32))
        acc = float((s_np.argmax(1) == yte_np).mean() * 100)
        acc_clamped = float((np.clip(s_np, -model.T, model.T).argmax(1) == yte_np).mean() * 100)
        rec["curve"].append({"epoch": epoch, "test_acc": acc, "test_acc_clamped_votes": acc_clamped,
                             "train_acc_running": 100 * correct / xtr.shape[0],
                             "train_s": train_s, "test_s": test_s})
        print(f"[{args.spec}] epoch {epoch:3d} test {acc:5.2f}% (clamped votes {acc_clamped:5.2f}%) "
              f"train-running {100*correct/xtr.shape[0]:5.2f}% | {train_s:.0f}s + {test_s:.0f}s", flush=True)

        rec["wall_s"] = rec.get("wall_s", 0.0) + (time.time() - t_run)
        t_run = time.time()
        if epoch == args.epochs:
            rec.update(status="done", test_acc_final=rec["curve"][-1]["test_acc"],
                       test_acc_epoch100=next((c["test_acc"] for c in rec["curve"] if c["epoch"] == 100), None),
                       test_acc_epoch99=next((c["test_acc"] for c in rec["curve"] if c["epoch"] == 99), None),
                       class_sums_dir=str(sums_dir.relative_to(sums_dir.parents[2])))
        write_json(rec, out)
        torch.save({"model": model.state_dict(), "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                    "epoch": epoch, "rec": rec}, ckpt)


if __name__ == "__main__":
    main()
