"""Train one Optimized Toolbox specialist with torchtsetlin and save its test class sums every epoch.

Hyper-parameters: Table III of arXiv:2406.00704 (code/specialists.py), clauses per class
(default: the paper's best column, 64 000), max_included_literals=32, 250 epochs, full training
set (100 000 images for the augmented specialists), test accuracy after every epoch, no selection.

Usage:  python code/train.py --spec therm5 [--clauses 64000] [--device cuda:0]
        python code/train.py --spec therm5 --calibrate 20     # time 20 batches + a test pass, no record
"""

import argparse
import sys
import time

import numpy as np
import torch

import torchtsetlin as tt
from common import CACHE, RESULTS, new_record, write_json
from encoders import encoded
from specialists import EPOCHS, LAST_EPOCHS, MAX_INCLUDED_LITERALS, SPECIALISTS


def build_model(cfg, n_clauses, n_features, input_shape, max_chunk_elements, batch_size):
    common = dict(weighted=cfg["weighted"], max_included_literals=MAX_INCLUDED_LITERALS,
                  budget_mode="soft", feedback_mode="batch", max_chunk_elements=max_chunk_elements)
    if cfg["model"] == "conv":
        # tmu's convolution appends thermometer-coded patch coordinates; so does position_encoding.
        return tt.ConvTsetlinMachine(10, n_clauses, cfg["T"], cfg["s"], patch_size=cfg["window"],
                                     stride=1, position_encoding=True, input_shape=input_shape,
                                     **common)
    return tt.TsetlinMachine(n_features, 10, n_clauses, cfg["T"], cfg["s"], **common)


@torch.no_grad()
def class_sums(model, x, eval_batch):
    """Raw (unclamped) class sums, as tmu's ``predict(..., return_class_sums=True)``: since
    0.4 ``forward()`` does not clamp (``clamp_prediction_votes=False``)."""
    assert not model.clamp_prediction_votes
    model.eval()                                   # empty clauses evaluate False when predicting
    try:
        return torch.cat([model(x[i:i + eval_batch]) for i in range(0, x.shape[0], eval_batch)])
    finally:
        model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, choices=sorted(SPECIALISTS))
    ap.add_argument("--clauses", type=int, default=64000, help="clauses per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--max-chunk-elements", type=int, default=2**29)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--calibrate", type=int, default=0, help="time this many batches and exit")
    ap.add_argument("--subset", type=int, default=0, help="smoke tests: first N train/test images")
    ap.add_argument("--tag", default="", help="record name (default <spec>_c<clauses>_seed<seed>)")
    args = ap.parse_args()

    cfg = SPECIALISTS[args.spec]
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)

    t0 = time.time()
    btr, ytr, bte, yte = encoded(cfg["encoding"], cfg["augmented"])
    enc_s = time.time() - t0
    if args.subset:
        btr, ytr, bte, yte = btr[:args.subset], ytr[:args.subset], bte[:args.subset], yte[:args.subset]
    n_features = int(np.prod(btr.shape[1:]))
    input_shape = tuple(btr.shape[1:]) if cfg["model"] == "conv" else None
    # Model first: moving it refreshes the include cache, whose bool .sum() transiently needs
    # 8 bytes per automaton (6.2 GiB at 640 000 x 1 308; LIBRARY_GAPS.md TB-1).
    model = build_model(cfg, args.clauses, n_features, input_shape, args.max_chunk_elements,
                        args.batch_size).to(dev)
    torch.cuda.empty_cache()
    xtr = torch.from_numpy(btr).to(dev)            # bool, blocking copies (CLAUDE.md)
    xte = torch.from_numpy(bte).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 7919)
    hp = {k: cfg[k] for k in ("encoding", "augmented", "model", "window", "weighted", "T", "s")}
    hp.update(clauses_per_class=args.clauses, clauses_total=int(model.n_clauses_total),
              n_literals=int(model.n_literals), max_included_literals=MAX_INCLUDED_LITERALS,
              budget_mode=model.budget_mode, feedback_mode="batch", batch_size=args.batch_size,
              epochs=args.epochs, position_encoding=cfg["model"] == "conv",
              n_states=int(model.n_states), boost_true_positive=bool(model.boost_true_positive),
              max_chunk_elements=args.max_chunk_elements)
    print(f"[{args.spec}] {cfg['name']} | x {tuple(xtr.shape)} | {args.clauses}/class = "
          f"{model.n_clauses_total} clauses | {model.n_literals} literals | encode {enc_s:.1f}s | "
          f"{args.device}", flush=True)

    if args.calibrate:
        perm = torch.randperm(xtr.shape[0], generator=gen).to(dev)
        bs = args.batch_size
        t0 = time.time()
        for i in range(10):                        # warm-up, incl. CUDA-graph capture
            model.update(xtr[perm[i * bs:(i + 1) * bs]], ytr_t[perm[i * bs:(i + 1) * bs]])
        torch.cuda.synchronize(dev)
        warm = time.time() - t0
        t0 = time.time()
        for i in range(10, 10 + args.calibrate):
            model.update(xtr[perm[i * bs:(i + 1) * bs]], ytr_t[perm[i * bs:(i + 1) * bs]])
        torch.cuda.synchronize(dev)
        per_batch = (time.time() - t0) / args.calibrate
        t0 = time.time()
        n_ev = min(2000, xte.shape[0])
        class_sums(model, xte[:n_ev], args.eval_batch)
        torch.cuda.synchronize(dev)
        test_s = (time.time() - t0) * xte.shape[0] / n_ev
        train_s = per_batch * xtr.shape[0] / bs
        rec = new_record("calibration", spec=args.spec, hp=hp, warmup_s=warm,
                         ms_per_batch=per_batch * 1000, train_s_per_epoch=train_s,
                         test_s_per_epoch=test_s, hours_250=(train_s + test_s) * EPOCHS / 3600,
                         peak_gib_reserved=torch.cuda.max_memory_reserved(dev) / 2**30,
                         peak_gib_allocated=torch.cuda.max_memory_allocated(dev) / 2**30)
        print(f"[{args.spec}] calibrate: {per_batch*1000:.1f} ms/batch -> train {train_s:.0f} s/epoch, "
              f"test {test_s:.0f} s -> {rec['hours_250']:.1f} h for {EPOCHS} epochs | peak "
              f"{rec['peak_gib_reserved']:.1f} GiB", flush=True)
        write_json(rec, RESULTS / "calibration" / f"{args.spec}_c{args.clauses}_{dev.type}{dev.index}.json")
        return

    tag = args.tag or f"{args.spec}_c{args.clauses}_seed{args.seed}"
    out = RESULTS / f"{tag}.json"
    sums_dir = CACHE / "class_sums" / tag
    sums_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CACHE / "ckpt" / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    paper = cfg["paper"].get(args.clauses)
    rec = new_record("specialist", spec=args.spec, name=cfg["name"], clauses=args.clauses,
                     seed=args.seed, hp=hp,
                     paper={"ref": "Grønningsæter et al. 2024, arXiv:2406.00704, Tables III-IV",
                            "table_iv": paper},
                     data={"dataset": "CIFAR-10", "n_train": int(xtr.shape[0]),
                           "n_test": int(xte.shape[0]), "encoding_s": enc_s,
                           "n_features": n_features, "shape": list(btr.shape[1:])},
                     curve=[], status="running")
    start_epoch = 0
    if ckpt.exists():                              # resume an interrupted run
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        gen.set_state(state["gen"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], dev)
        rec = state["rec"]
        rec["status"] = "running"
        rec.setdefault("resumed_at", []).append({"epoch": state["epoch"], "argv": sys.argv,
                                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        rec["curve"] = [c for c in rec["curve"] if c["epoch"] <= state["epoch"]]
        start_epoch = state["epoch"]
        print(f"[{args.spec}] resumed after epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(xtr.shape[0], generator=gen).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, xtr.shape[0], args.batch_size):
            idx = perm[i:i + args.batch_size]
            votes = model.update(xtr[idx], ytr_t[idx])          # pre-update vote sums
            correct += (votes.argmax(1) == ytr_t[idx]).sum()
        torch.cuda.synchronize(dev)
        train_s = time.time() - t0

        t0 = time.time()
        sums = class_sums(model, xte, args.eval_batch)
        torch.cuda.synchronize(dev)
        test_s = time.time() - t0
        s_np = sums.float().cpu().numpy()
        np.save(sums_dir / f"e{epoch:03d}.npy", s_np)
        acc = float((s_np.argmax(1) == yte).mean() * 100)
        tr = 100 * int(correct) / xtr.shape[0]
        rec["curve"].append({"epoch": epoch, "test_acc": acc, "train_acc_running": tr,
                             "train_s": train_s, "test_s": test_s,
                             "peak_gib": torch.cuda.max_memory_reserved(dev) / 2**30})
        print(f"[{args.spec}] epoch {epoch:3d} test {acc:5.2f}% train-running {tr:5.2f}% | "
              f"{train_s:.0f}s + {test_s:.0f}s", flush=True)
        if epoch == args.epochs:
            last = np.array([c["test_acc"] for c in rec["curve"][-LAST_EPOCHS:]])
            rec.update(status="done", test_acc_final=acc,
                       last25_mean=float(last.mean()), last25_variance=float(last.var(ddof=1)),
                       last25_sd=float(last.std(ddof=1)),
                       train_s_per_epoch_mean=float(np.mean([c["train_s"] for c in rec["curve"]])),
                       test_s_per_epoch_mean=float(np.mean([c["test_s"] for c in rec["curve"]])),
                       finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_json(rec, out)
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "gen": gen.get_state(),
                        "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(dev),
                        "epoch": epoch, "rec": rec}, tmp)
            tmp.replace(ckpt)


if __name__ == "__main__":
    main()
