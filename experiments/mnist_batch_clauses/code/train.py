"""Train one cell of the batch-size x clause-count grid on MNIST until early stopping.

Plan: plans/mnist-batch-clauses.md. Model = the paper's B4 (flat) or B7 (conv) configuration with
``n_clauses`` per class and T scaled with it (D2); batched feedback with ``--batch-size`` (or the
exact sequential algorithm with ``--sequential``, D6). Training set = 50 000 MNIST images, the other
10 000 are the validation set that early stopping watches (D3a); the test set is evaluated every
epoch but never used for a decision. Stops after ``--patience`` epochs without a validation
improvement of at least ``--min-delta`` pp, not before ``--min-epochs``, at most ``--max-epochs`` or
``--time-cap-h`` hours (D3). The primary statistic is the test accuracy at the best-validation epoch,
read from the per-epoch curve (D4).

Usage:  python code/train.py --model conv --clauses 8000 --batch-size 32 --seed 0 --device cuda:0
        python code/train.py ... --calibrate 50      # Phase 0: time 50 batches + one eval, no curve
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch

import torchtsetlin as tt
from common import BASE, CACHE, EARLY, RESULTS, T_of, load_data, new_record, tag_of, write_json
from tracking import Tracker


def build_model(model, clauses, T, sequential):
    b = BASE[model]
    kw = dict(weighted=b["weighted"], boost_true_positive=b["boost_true_positive"],
              feedback_mode="sequential" if sequential else "batch")
    if model == "conv":
        return tt.ConvTsetlinMachine(10, clauses, T, b["s"], patch_size=b["patch_size"],
                                     stride=b["stride"], position_encoding=b["position_encoding"],
                                     input_shape=(1, 28, 28), **kw)
    return tt.TsetlinMachine(784, 10, clauses, T, b["s"], **kw)


@torch.no_grad()
def class_sums(model, x, eval_batch):
    """Raw (unclamped) class sums in prediction mode (empty clauses evaluate False)."""
    model.eval()
    try:
        return torch.cat([model(x[i:i + eval_batch]) for i in range(0, x.shape[0], eval_batch)])
    finally:
        model.train()


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def peak_mib(dev):
    return torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(BASE))
    ap.add_argument("--clauses", type=int, required=True, help="clauses per class")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--t-mode", default="scaled", choices=["scaled", "fixed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequential", action="store_true", help="feedback_mode='sequential' (D6)")
    ap.add_argument("--arm", default="grid",
                    choices=["grid", "control_T", "control_seq", "smoke", "phase0", "resume_check"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patience", type=int, default=EARLY["patience"])
    ap.add_argument("--min-delta", type=float, default=EARLY["min_delta"], help="pp of validation acc")
    ap.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    ap.add_argument("--max-epochs", type=int, default=EARLY["max_epochs"])
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"])
    ap.add_argument("--eval-batch", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="time this many batches and exit")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)
    T = T_of(args.model, args.clauses, args.t_mode)
    tag = tag_of(args.model, args.clauses, args.batch_size, args.t_mode, args.seed,
                 args.sequential) + args.tag_suffix

    d = load_data(args.model)
    if args.subset:
        n, m = args.subset, max(args.subset // 4, 100)
        for k, v in (("train", n), ("val", m), ("test", m)):
            d[f"x_{k}"], d[f"y_{k}"] = d[f"x_{k}"][:v], d[f"y_{k}"][:v]

    model = build_model(args.model, args.clauses, T, args.sequential).to(dev)

    def to_x(a):
        t = torch.from_numpy(np.ascontiguousarray(a)).to(torch.bool)
        return (t.reshape(len(a), 1, 28, 28) if args.model == "conv" else t.reshape(len(a), -1)).to(dev)

    xtr, xva, xte = to_x(d["x_train"]), to_x(d["x_val"]), to_x(d["x_test"])
    ytr = torch.from_numpy(d["y_train"]).to(dev)
    yva, yte = d["y_val"], d["y_test"]
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    # D7: the data order depends on the seed only, so it is identical across B and C.
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)
    b = BASE[args.model]
    hp = dict(model=args.model, clauses_per_class=args.clauses,
              clauses_total=int(model.n_clauses_total), n_literals=int(model.n_literals),
              T=T, T_mode=args.t_mode, T_ref=b["T_ref"], s=b["s"], weighted=b["weighted"],
              boost_true_positive=b["boost_true_positive"],
              feedback_mode="sequential" if args.sequential else "batch", batch_size=args.batch_size,
              commits_per_epoch=math.ceil(n_tr / args.batch_size), seed=args.seed,
              n_states=int(model.n_states), max_chunk_elements=int(model.max_chunk_elements),
              **({k: b[k] for k in ("patch_size", "stride", "position_encoding")}
                 if args.model == "conv" else {}),
              early_stopping=dict(monitor="val_acc", patience=args.patience, min_delta_pp=args.min_delta,
                                  min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                  time_cap_h=args.time_cap_h),
              data=dict(bench=d["bench"], sha256=d["sha256"], n_train=n_tr, n_val=n_va,
                        n_test=int(xte.shape[0]), subset=args.subset))
    print(f"[{tag}] {args.model} {args.clauses}/class = {model.n_clauses_total} clauses, "
          f"{model.n_literals} literals, T={T}, B={args.batch_size}"
          f"{' sequential' if args.sequential else ''}, {args.device}", flush=True)

    if args.calibrate:
        calibrate(args, model, hp, tag, xtr, ytr, xva, gen, dev)
        return

    out = RESULTS / args.arm / f"{tag}.json"
    ckpt = CACHE / "ckpt" / args.arm / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", device=dev, arm=args.arm, tag=tag, hp=hp, curve=[], status="running",
                     early={"best_val_correct": -1, "best_epoch": 0, "since": 0}, elapsed_s=0.0)
    start = 0
    if ckpt.exists():                              # resume an interrupted run
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(st["cuda_rng"], dev)
        rec = st["rec"]
        start = st["epoch"]
        rec["curve"] = [c for c in rec["curve"] if c["epoch"] <= start]
        rec.setdefault("resumed_at", []).append({"epoch": start, "argv": sys.argv,
                                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[{tag}] resumed after epoch {start}", flush=True)
    elif out.exists() and json_status(out) == "done":
        print(f"[{tag}] already done", flush=True)
        return
    if rec["status"] == "done":
        print(f"[{tag}] already done", flush=True)
        return
    rec["status"] = "running"

    tags = [args.model, f"c{args.clauses}", f"b{args.batch_size}", f"T-{args.t_mode}",
            f"s{args.seed}", rec["env"].get("gpu", args.device)] + (["sequential"] if args.sequential else [])
    tracker = Tracker(args.arm, tag, dict(hp, tag=tag, arm=args.arm, git_sha=rec["git_sha"],
                                          env=rec["env"]), tags, enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        train(args, model, rec, T, xtr, ytr, xva, yva, xte, yte, gen, dev, out, ckpt, start, tracker)
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]),
                     "error": f"{type(e).__name__}: {e}"})
        raise
    summary_keys = ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch",
                    "epochs_run", "commits_to_best", "gpu_s_to_best", "test_acc_epoch30",
                    "test_acc_last", "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")
    tracker.end({k: rec.get(k) for k in summary_keys})


def json_status(path):
    import json
    try:
        return json.loads(path.read_text()).get("status")
    except Exception:  # noqa: BLE001
        return None


def train(args, model, rec, T, xtr, ytr, xva, yva, xte, yte, gen, dev, out, ckpt, start, tracker):
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    bs = args.batch_size
    need = math.ceil(args.min_delta / 100 * n_va - 1e-9)          # validation examples to improve
    es = rec["early"]
    epoch = start
    while True:
        epoch += 1
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=gen).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            votes = model.update(xtr[idx], ytr[idx])               # pre-update vote sums
            correct += (votes.argmax(1) == ytr[idx]).sum()
        sync(dev)
        train_s = time.time() - t0

        t0 = time.time()
        va_pred = class_sums(model, xva, args.eval_batch).argmax(1).cpu().numpy()
        sync(dev)
        val_s = time.time() - t0
        t0 = time.time()
        te_sums = class_sums(model, xte, args.eval_batch)
        sync(dev)
        test_s = time.time() - t0
        te_pred = te_sums.argmax(1).cpu().numpy()
        saturation = float((te_sums.max(1).values >= T).float().mean())
        val_correct = int((va_pred == yva).sum())
        val_acc = 100 * val_correct / n_va
        test_acc = 100 * float((te_pred == yte).mean())
        inc = model.include_count.float()

        if val_correct >= es["best_val_correct"] + need or es["best_val_correct"] < 0:
            es.update(best_val_correct=val_correct, best_epoch=epoch, since=0)
        else:
            es["since"] += 1
        rec["elapsed_s"] += train_s + val_s + test_s
        commits = epoch * math.ceil(n_tr / bs)
        row = {"epoch": epoch, "val_acc": val_acc, "best_val_acc": 100 * es["best_val_correct"] / n_va,
               "epochs_since_improvement": es["since"], "test_acc": test_acc,
               "train_acc_running": 100 * int(correct) / n_tr, "train_s": train_s, "val_s": val_s,
               "test_s": test_s, "examples_per_s": n_tr / train_s, "commits": commits,
               "peak_gpu_mib": peak_mib(dev), "mean_clause_len": float(inc.mean()),
               "frac_empty_clauses": float((inc == 0).float().mean()), "vote_saturation": saturation,
               "elapsed_s": rec["elapsed_s"]}
        rec["curve"].append(row)
        tracker.epoch(row)
        print(f"[{rec['tag']}] epoch {epoch:3d} val {val_acc:6.2f} test {test_acc:6.2f} "
              f"train {row['train_acc_running']:6.2f} | best {row['best_val_acc']:6.2f}@{es['best_epoch']} "
              f"wait {es['since']:2d} | {train_s:.1f}s + {val_s + test_s:.1f}s | len "
              f"{row['mean_clause_len']:.1f} sat {saturation:.2f}", flush=True)

        stop = None
        if epoch >= args.min_epochs and es["since"] >= args.patience:
            stop = "patience"
        elif epoch >= args.max_epochs:
            stop = "max_epochs"
        elif rec["elapsed_s"] >= args.time_cap_h * 3600:
            stop = "time_cap"
        if stop:
            finish(rec, stop, es, n_tr, bs)
        write_json(rec, out)
        if stop or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "gen": gen.get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                        "epoch": epoch, "rec": rec}, tmp)
            tmp.replace(ckpt)
        if stop:
            ckpt.unlink(missing_ok=True)            # the record holds everything that is reported
            print(f"[{rec['tag']}] stop ({stop}) after {epoch} epochs: best val "
                  f"{rec['best_val_acc']:.2f} @ {rec['best_epoch']}, test there {rec['test_acc_at_best']:.2f}",
                  flush=True)
            return
        if epoch == args.stop_after:
            print(f"[{rec['tag']}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


def finish(rec, stop, es, n_tr, bs):
    c = rec["curve"]
    best = c[es["best_epoch"] - 1]
    assert best["epoch"] == es["best_epoch"]
    rec.update(status="done", stop_reason=stop, best_epoch=es["best_epoch"],
               best_val_acc=best["val_acc"], test_acc_at_best=best["test_acc"],
               epochs_run=len(c), commits_to_best=es["best_epoch"] * math.ceil(n_tr / bs),
               gpu_s_to_best=best["elapsed_s"],
               test_acc_epoch30=c[29]["test_acc"] if len(c) >= 30 else None,
               test_acc_last=c[-1]["test_acc"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               peak_gpu_mib=max(r["peak_gpu_mib"] for r in c),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))


def calibrate(args, model, hp, tag, xtr, ytr, xva, gen, dev):
    """Phase 0: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch estimate."""
    n_tr, bs = xtr.shape[0], args.batch_size
    perm = torch.randperm(n_tr, generator=gen).to(dev)
    n_batches = min(args.calibrate, n_tr // bs - 10)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for i in range(10):
        model.update(xtr[perm[i * bs:(i + 1) * bs]], ytr[perm[i * bs:(i + 1) * bs]])
    sync(dev)
    warm = time.time() - t0
    t0 = time.time()
    for i in range(10, 10 + n_batches):
        model.update(xtr[perm[i * bs:(i + 1) * bs]], ytr[perm[i * bs:(i + 1) * bs]])
    sync(dev)
    per = (time.time() - t0) / n_batches
    t0 = time.time()
    class_sums(model, xva, args.eval_batch)
    sync(dev)
    eval_s = time.time() - t0
    train_s = per * math.ceil(n_tr / bs)
    rec = new_record("calibration", device=dev, tag=tag, hp=hp, warmup_s=warm, n_batches=n_batches,
                     ms_per_update=per * 1000, train_s_per_epoch=train_s,
                     eval_s_per_epoch=2 * eval_s * 10000 / xva.shape[0],
                     peak_gpu_mib=peak_mib(dev))
    print(f"[{tag}] calibrate: {per*1000:.2f} ms/update -> train {train_s:.1f} s/epoch, eval "
          f"{rec['eval_s_per_epoch']:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / "phase0" / f"{tag}_{dev.type}{dev.index if dev.index is not None else ''}.json")


if __name__ == "__main__":
    main()
