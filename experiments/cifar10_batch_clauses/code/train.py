"""Train one cell of the batch-size x clause-count grid on CIFAR-10 until early stopping.

Plan: plans/cifar10-batch-clauses.md. Models (D1): ``conv`` (3x3 colour-thermometer ConvTM, the
Optimized Toolbox setting), ``conv3`` (3D ConvTM over 8 bit planes per colour, window (8, 3, 3)),
``flat`` (flat TM on a 3-level colour thermometer). ``n_clauses`` per class, T scaled with it (D7),
batched feedback with ``--batch-size`` (or the exact sequential algorithm, ``--sequential``, D8).
Training set = 45 000 images, the other 5 000 are the validation set early stopping watches (D4);
the test set is evaluated every epoch but never used for a decision. Stopping rule D6. Primary
statistic: test accuracy at the best-validation epoch.

Usage:  python code/train.py --model conv --clauses 8000 --batch-size 32 --device cuda:0
        python code/train.py ... --calibrate 200     # pricing: time 200 updates + one evaluation
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch
from common import (
    BASE,
    CACHE,
    EARLY,
    PILOT_MAX_EPOCHS,
    RESULTS,
    T_of,
    hyper,
    load_data,
    new_record,
    tag_of,
    write_json,
)
from tracking import Tracker

import torchtsetlin as tt

IN_SHAPE = {"conv": (24, 32, 32), "conv3": (3, 8, 32, 32), "flat": None}


def chunk_budget(model, clauses):
    """D9: max_chunk_elements for conv and conv3: large enough that conv3's patch axis is ONE block
    (fmnist_bitplanes LIBRARY_GAPS FB-1, GitHub #17), so its update stays on the fused CUDA path;
    conv gets the same value, so the two differ only in the input encoding (same literals, same
    chunking). Never below the library default (2**27). A memory/speed knob only, recorded in hp."""
    P, C = 30 * 30, 10 * clauses
    L = model.n_literals
    need = 3 * P * C + 2 * C * L + P * L
    return max(2 ** 27, 2 ** math.ceil(math.log2(1.05 * need)))


def build_model(model, clauses, T, s, sequential):
    b = BASE[model]
    kw = dict(weighted=b["weighted"], boost_true_positive=b["boost_true_positive"],
              max_included_literals=b["max_included_literals"], budget_mode="soft",
              feedback_mode="sequential" if sequential else "batch")
    if model == "flat":
        return tt.TsetlinMachine(9 * 32 * 32, 10, clauses, T, s, **kw)

    def make(budget=None):
        extra = {} if budget is None else {"max_chunk_elements": budget}
        if model == "conv":
            return tt.ConvTsetlinMachine(10, clauses, T, s, patch_size=b["patch_size"], stride=b["stride"],
                                         position_encoding=b["position_encoding"],
                                         input_shape=IN_SHAPE["conv"], **kw, **extra)
        return tt.Conv3dTsetlinMachine(10, clauses, T, s, patch_size=(b["kd"], b["patch_size"], b["patch_size"]),
                                       stride=b["stride"], position_encoding=b["position_encoding"],
                                       input_shape=IN_SHAPE["conv3"], **kw, **extra)
    m = make()
    budget = chunk_budget(m, clauses)
    return m if budget == int(m.max_chunk_elements) else make(budget)


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
    ap.add_argument("--sequential", action="store_true", help="feedback_mode='sequential' (D8)")
    ap.add_argument("--pilot-tref", type=int, default=None, help="flat pilot: T at 2 000/class")
    ap.add_argument("--pilot-s", type=float, default=None, help="flat pilot: specificity")
    ap.add_argument("--arm", default="grid",
                    choices=["grid", "control_T", "control_seq", "pilot", "smoke", "pricing", "resume_check"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patience", type=int, default=EARLY["patience"])
    ap.add_argument("--min-delta", type=float, default=EARLY["min_delta"])
    ap.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    ap.add_argument("--max-epochs", type=int, default=None, help="default 500 (pilot: 60)")
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"])
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="pricing: time this many updates and exit")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.max_epochs is None:
        args.max_epochs = PILOT_MAX_EPOCHS if args.arm == "pilot" else EARLY["max_epochs"]

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)
    pilot = (args.pilot_tref, args.pilot_s) if args.pilot_tref is not None else None
    t_ref, s = hyper(args.model, pilot)
    T = T_of(args.model, args.clauses, args.t_mode, t_ref)
    tag = tag_of(args.model, args.clauses, args.batch_size, args.t_mode, args.seed, args.sequential,
                 pilot) + args.tag_suffix
    out = RESULTS / args.arm / f"{tag}.json"

    b = BASE[args.model]
    d = load_data(b["enc"])
    if args.subset:
        n, m = args.subset, max(args.subset // 4, 100)
        for k, v in (("train", n), ("val", m), ("test", m)):
            d[f"x_{k}"], d[f"y_{k}"] = d[f"x_{k}"][:v], d[f"y_{k}"][:v]

    try:
        model = build_model(args.model, args.clauses, T, s, args.sequential).to(dev)

        def to_x(a):
            t = torch.from_numpy(np.ascontiguousarray(a))
            return (t.reshape(len(a), -1) if args.model == "flat" else t).to(dev)

        xtr, xva, xte = to_x(d["x_train"]), to_x(d["x_val"]), to_x(d["x_test"])
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not is_oom(e):
            raise
        return oom(args, tag, out, dev, e)
    ytr = torch.from_numpy(d["y_train"]).to(dev)
    yva, yte = d["y_val"], d["y_test"]
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    # D10: the data order depends on the seed only, so it is identical across B and C.
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)
    hp = dict(model=args.model, encoding=b["enc"], clauses_per_class=args.clauses,
              clauses_total=int(model.n_clauses_total), n_literals=int(model.n_literals),
              T=T, T_mode=args.t_mode, T_ref=t_ref, s=s, weighted=b["weighted"],
              boost_true_positive=b["boost_true_positive"],
              max_included_literals=b["max_included_literals"], budget_mode="soft",
              feedback_mode="sequential" if args.sequential else "batch", batch_size=args.batch_size,
              commits_per_epoch=math.ceil(n_tr / args.batch_size), seed=args.seed,
              n_states=int(model.n_states), max_chunk_elements=int(model.max_chunk_elements),
              input_shape=IN_SHAPE[args.model] or [9 * 32 * 32],
              **({k: b[k] for k in ("patch_size", "stride", "position_encoding")}
                 if args.model != "flat" else {}),
              **({"kd": b["kd"]} if args.model == "conv3" else {}),
              pilot=list(pilot) if pilot else None,
              early_stopping=dict(monitor="val_acc", patience=args.patience, min_delta_pp=args.min_delta,
                                  min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                  time_cap_h=args.time_cap_h),
              data=dict(dataset="CIFAR-10", n_train=n_tr, n_val=n_va, n_test=int(xte.shape[0]),
                        subset=args.subset))
    print(f"[{tag}] {args.model} {args.clauses}/class = {model.n_clauses_total} clauses, "
          f"{model.n_literals} literals, T={T}, s={s:g}, B={args.batch_size}"
          f"{' sequential' if args.sequential else ''}, {args.device}", flush=True)

    if args.calibrate:
        try:
            calibrate(args, model, hp, tag, xtr, ytr, xva, xte, gen, dev)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not is_oom(e):
                raise
            return oom(args, tag, RESULTS / "pricing" / f"{tag}.json", dev, e, kind="calibration")
        return

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
    elif out.exists() and (json_status(out) in ("done", "oom")):
        print(f"[{tag}] already finished", flush=True)
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
        if is_oom(e):
            tracker.end({"status": "oom", "epochs_run": len(rec["curve"])})
            return oom(args, tag, out, dev, e, rec=rec)
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]),
                     "error": f"{type(e).__name__}: {e}"})
        raise
    summary_keys = ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch",
                    "epochs_run", "commits_to_best", "gpu_s_to_best", "test_acc_epoch30",
                    "test_acc_last", "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")
    tracker.end({k: rec.get(k) for k in summary_keys})


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def oom(args, tag, out, dev, err, rec=None, kind="run"):
    """D2: a cell that does not fit the GPU is a result ("oom"), reported, never silently dropped."""
    r = rec or new_record(kind, device=dev, arm=args.arm, tag=tag)
    r.update(status="oom", error=f"{type(err).__name__}: {str(err)[:400]}",
             finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(r, out)
    print(f"[{tag}] out of memory: recorded as status=oom", flush=True)


def json_status(path):
    import json
    try:
        return json.loads(path.read_text()).get("status")
    except Exception:  # noqa: BLE001
        return None


def train(args, model, rec, T, xtr, ytr, xva, yva, xte, yte, gen, dev, out, ckpt, start, tracker):
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    bs = args.batch_size
    # Validation examples a new best must add: >= 1 (a tie is not an improvement), more with min_delta.
    need = max(1, math.ceil(args.min_delta / 100 * n_va - 1e-9))
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

        if es["best_val_correct"] < 0 or val_correct >= es["best_val_correct"] + need:
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


def calibrate(args, model, hp, tag, xtr, ytr, xva, xte, gen, dev):
    """Pricing: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch estimate."""
    n_tr, bs = xtr.shape[0], args.batch_size
    perm = torch.randperm(n_tr, generator=gen).to(dev)
    n_batches = max(1, min(args.calibrate, n_tr // bs - 10))
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
    eval_epoch = eval_s * (xva.shape[0] + xte.shape[0]) / xva.shape[0]
    rec = new_record("calibration", device=dev, arm="pricing", tag=tag, hp=hp, status="done",
                     warmup_s=warm, n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=train_s, eval_s_per_epoch=eval_epoch, peak_gpu_mib=peak_mib(dev))
    print(f"[{tag}] calibrate: {per*1000:.2f} ms/update -> train {train_s:.1f} s/epoch, eval "
          f"{eval_epoch:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / "pricing" / f"{tag}.json")


if __name__ == "__main__":
    main()
