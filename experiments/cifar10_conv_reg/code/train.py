"""Train one single ConvTM on CIFAR-10 with optional augmentation and drop clause.

Plan: torchtsetlin_plans/experiments/cifar10_conv_reg/PLAN.md. The model is the ``conv`` model of
cifar10_batch_clauses (3x3 colour-thermometer ConvTM, weighted, literal budget 32, s = 5,
T = 1.5 x clauses per class, batch 32). New here: ``--aug`` (training images are augmented once
per epoch before the thermometer), ``--drop-clause-p``, patience 50, a 72 h cap, and every
``--viz-every`` epochs three figures (code/viz.py). Each epoch also measures ``train_eval_acc``:
accuracy in eval mode on 5 000 fixed *clean* training images, comparable between arms (the
running training accuracy is measured under augmentation and dropout); ``gap`` = that minus the
validation accuracy. Early stopping watches validation only; the statistic is the test accuracy at
the best-validation epoch, and that model is kept in ``models/<phase>/<tag>.pt``.

Usage:  python code/train.py --clauses 8000 --aug geo+col --drop-clause-p 0.25 --phase screen
        python code/train.py --clauses 64000 --calibrate 200 --phase pricing   # time 200 updates
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch
from common import (
    BATCH,
    CACHE,
    EARLY,
    FIGURES,
    MODEL,
    MODELS,
    RESULTS,
    VIZ_EVERY,
    T_of,
    load_cifar,
    new_record,
    panel_indices,
    read_json,
    tag_of,
    train_eval_indices,
    write_json,
)
from tracking import Tracker

import torchtsetlin as tt

IN_SHAPE = (24, 32, 32)
_LEVELS = torch.tensor(np.arange(1, 9) * 255 / 9, dtype=torch.float64)
NO_MODELS = {"pricing", "resume_check"}            # throw-away runs: no model kept


def gpu_encode(x, dev, chunk=2000):
    """uint8 (N, 32, 32, 3) array -> bool (N, 24, 32, 32) on ``dev``: the colour thermometer of
    cifar10_batch_clauses (``x >= k*255/9``, k = 1..8, plane = channel * 8 + level)."""
    xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    lv = _LEVELS.to(dev)
    out = []
    for i in range(0, len(xt), chunk):
        b = xt[i:i + chunk]
        bits = b.unsqueeze(-1).to(torch.float64) >= lv                       # (n, 32, 32, 3, 8)
        out.append(bits.reshape(len(b), 32, 32, 24).permute(0, 3, 1, 2).contiguous())
    return torch.cat(out)


def chunk_budget(model, clauses):
    """max_chunk_elements as in cifar10_batch_clauses (D9), so the 16 000 and 32 000 base runs
    chunk exactly like the old grid. A memory/speed knob only, recorded in hp."""
    P, C = 30 * 30, 10 * clauses
    L = model.n_literals
    need = 3 * P * C + 2 * C * L + P * L
    return max(2 ** 27, 2 ** math.ceil(math.log2(1.05 * need)))


def build_model(clauses, T, s, drop_clause_p=0.0, budget=None):
    def make(budget=None):
        extra = {} if budget is None else {"max_chunk_elements": budget}
        return tt.ConvTsetlinMachine(
            10, clauses, T, s, patch_size=MODEL["patch_size"], stride=MODEL["stride"],
            position_encoding=MODEL["position_encoding"], input_shape=IN_SHAPE, weighted=MODEL["weighted"],
            boost_true_positive=MODEL["boost_true_positive"], max_included_literals=MODEL["max_included_literals"],
            budget_mode="soft", feedback_mode="batch", drop_clause_p=drop_clause_p, **extra)
    if budget is not None:
        return make(budget)
    m = make()
    budget = chunk_budget(m, clauses)
    return m if budget == int(m.max_chunk_elements) else make(budget)


def load_model(path, dev):
    """A saved best-validation model (``models/<phase>/<tag>.pt``) -> (model in eval mode, file dict)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    hp = st["hp"]
    model = build_model(hp["clauses_per_class"], hp["T"], hp["s"], 0.0, hp["max_chunk_elements"]).to(dev)
    model.load_state_dict(st["model"])
    return model.eval(), st


@torch.no_grad()
def class_sums(model, x, eval_batch):
    """Raw (unclamped) class sums in prediction mode (empty clauses evaluate False, no dropout)."""
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


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def oom(args, tag, out, dev, err, rec=None, kind="run"):
    """A run that does not fit the GPU is a result ("oom"), reported, never silently dropped."""
    r = rec or new_record(kind, device=dev, phase=args.phase, tag=tag)
    r.update(status="oom", error=f"{type(err).__name__}: {str(err)[:400]}",
             finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(r, out)
    print(f"[{tag}] out of memory: recorded as status=oom", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", type=int, required=True, help="clauses per class")
    ap.add_argument("--aug", default="none", choices=["none", "geo", "geo+col", "full"])
    ap.add_argument("--drop-clause-p", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="screen",
                    choices=["screen", "factorial", "smoke", "pricing", "resume_check", "extra"])
    ap.add_argument("--arm", default=None, help="factorial arm label (base, aug, drop, aug_drop)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patience", type=int, default=EARLY["patience"])
    ap.add_argument("--min-delta", type=float, default=EARLY["min_delta"])
    ap.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    ap.add_argument("--max-epochs", type=int, default=EARLY["max_epochs"])
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"])
    ap.add_argument("--viz-every", type=int, default=VIZ_EVERY, help="0 = no figures")
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0, help="augmentation processes (0 = automatic)")
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="pricing: time this many updates and exit")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)          # the fast CUDA path needs the model's GPU to be current
    tt.seed_everything(args.seed)
    s, p = MODEL["s"], args.drop_clause_p
    T = T_of(args.clauses)
    tag = tag_of(args.clauses, args.aug, p, args.seed, args.batch_size) + args.tag_suffix
    out = RESULTS / args.phase / f"{tag}.json"
    if (read_json(out) or {}).get("status") in ("done", "oom") and not args.calibrate:
        print(f"[{tag}] already finished", flush=True)
        return

    data = load_cifar(args.subset)
    (raw_tr, y_tr), (raw_va, yva), (raw_te, yte) = data["train"], data["val"], data["test"]
    n_tr, n_va = len(raw_tr), len(raw_va)
    try:
        model = build_model(args.clauses, T, s, p).to(dev)
        xva, xte = gpu_encode(raw_va, dev), gpu_encode(raw_te, dev)
        te_idx = train_eval_indices(n_tr)
        xtc, ytc = gpu_encode(raw_tr[te_idx], dev), y_tr[te_idx]
        xtr = gpu_encode(raw_tr, dev)               # replaced every epoch when augmenting
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not is_oom(e):
            raise
        return oom(args, tag, out, dev, e)
    ytr = torch.from_numpy(y_tr).to(dev)
    pidx = panel_indices(yva)
    pan = {"x": xva[torch.as_tensor(pidx, device=dev)], "raw": raw_va[pidx], "y": yva[pidx]}
    # The data order depends on the seed only (as cifar10_batch_clauses D10).
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)
    hp = dict(model="conv", arm=args.arm, encoding=MODEL["enc"], clauses_per_class=args.clauses,
              clauses_total=int(model.n_clauses_total), n_literals=int(model.n_literals), T=T, T_mode="scaled",
              T_ref=MODEL["T_ref"], s=s, weighted=MODEL["weighted"],
              boost_true_positive=MODEL["boost_true_positive"],
              max_included_literals=MODEL["max_included_literals"], budget_mode="soft", feedback_mode="batch",
              batch_size=args.batch_size, commits_per_epoch=math.ceil(n_tr / args.batch_size), seed=args.seed,
              drop_clause_p=p, drop_granularity=model.drop_granularity if hasattr(model, "drop_granularity") else "batch",
              augmentation=dict(policy=args.aug, per_image_seed="SeedSequence([seed, epoch, index])"),
              n_states=int(model.n_states), max_chunk_elements=int(model.max_chunk_elements),
              input_shape=list(IN_SHAPE), patch_size=MODEL["patch_size"], stride=MODEL["stride"],
              position_encoding=MODEL["position_encoding"], viz_every=args.viz_every,
              early_stopping=dict(monitor="val_acc", patience=args.patience, min_delta_pp=args.min_delta,
                                  min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                  time_cap_h=args.time_cap_h),
              data=dict(dataset="CIFAR-10", n_train=n_tr, n_val=n_va, n_test=len(raw_te),
                        n_train_eval=len(te_idx), subset=args.subset))
    print(f"[{tag}] conv {args.clauses}/class = {model.n_clauses_total} clauses, {model.n_literals} literals, "
          f"T={T}, s={s:g}, B={args.batch_size}, aug={args.aug}, drop_clause_p={p:g}, {args.device}", flush=True)

    if args.calibrate:
        try:
            calibrate(args, model, hp, tag, xtr, ytr, xva, xte, gen, dev)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not is_oom(e):
                raise
            return oom(args, tag, RESULTS / "pricing" / f"{tag}.json", dev, e, kind="calibration")
        return

    ckpt = CACHE / "ckpt" / args.phase / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", device=dev, phase=args.phase, tag=tag, hp=hp, curve=[], status="running",
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
    rec["status"] = "running"

    source = None
    if args.aug != "none":
        from augment import EpochSource
        source = EpochSource(raw_tr, args.aug, args.seed, workers=args.workers or None)
        rec["hp"]["augmentation"]["workers"] = source.workers
    tags = ["conv", f"c{args.clauses}", f"b{args.batch_size}", f"aug-{args.aug}", f"dc{p:g}", f"s{args.seed}",
            rec["env"].get("gpu", args.device)] + ([args.arm] if args.arm else [])
    tracker = Tracker(args.phase, tag, dict(rec["hp"], tag=tag, phase=args.phase, git_sha=rec["git_sha"],
                                            env=rec["env"]), tags, enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    S = dict(model=model, rec=rec, T=T, xtr=xtr, ytr=ytr, raw_tr=raw_tr, xva=xva, yva=yva, xte=xte, yte=yte,
             xtc=xtc, ytc=ytc, pan=pan, gen=gen, dev=dev, out=out, ckpt=ckpt, start=start, tracker=tracker,
             source=source)
    try:
        train(args, S)
    except BaseException as e:
        if is_oom(e):
            tracker.end({"status": "oom", "epochs_run": len(rec["curve"])})
            return oom(args, tag, out, dev, e, rec=rec)
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]),
                     "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        if source is not None:
            source.close()
    keys = ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch", "train_eval_acc_at_best",
            "gap_at_best", "epochs_run", "commits_to_best", "gpu_s_to_best", "test_acc_last",
            "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")
    tracker.end({k: rec.get(k) for k in keys})


def train(args, S):
    model, rec, T, dev, tracker = S["model"], S["rec"], S["T"], S["dev"], S["tracker"]
    xtr, ytr, xva, yva, xte, yte = S["xtr"], S["ytr"], S["xva"], S["yva"], S["xte"], S["yte"]
    n_tr, n_va, bs, tag = xtr.shape[0], xva.shape[0], args.batch_size, rec["tag"]
    # Validation examples a new best must add: >= 1 (a tie is not an improvement), more with min_delta.
    need = max(1, math.ceil(args.min_delta / 100 * n_va - 1e-9))
    es = rec["early"]
    epoch = S["start"]
    while True:
        epoch += 1
        t0 = time.time()
        if S["source"] is not None:                    # this epoch's augmented images -> thermometer
            xtr = None
            xtr = gpu_encode(S["source"].get(epoch), dev)
            sync(dev)
        aug_wait = time.time() - t0
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=S["gen"]).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            votes = model.update(xtr[idx], ytr[idx])               # pre-update vote sums
            correct += (votes.argmax(1) == ytr[idx]).sum()
        sync(dev)
        train_s = time.time() - t0

        t0 = time.time()
        va_sums = class_sums(model, xva, args.eval_batch).float().cpu().numpy()
        sync(dev)
        val_s = time.time() - t0
        t0 = time.time()
        te_sums = class_sums(model, xte, args.eval_batch)
        sync(dev)
        test_s = time.time() - t0
        t0 = time.time()
        tc_pred = class_sums(model, S["xtc"], args.eval_batch).argmax(1).cpu().numpy()
        sync(dev)
        train_eval_s = time.time() - t0
        te_pred = te_sums.argmax(1).cpu().numpy()
        saturation = float((te_sums.max(1).values >= T).float().mean())
        val_correct = int((va_sums.argmax(1) == yva).sum())
        val_acc = 100 * val_correct / n_va
        test_acc = 100 * float((te_pred == yte).mean())
        train_eval_acc = 100 * float((tc_pred == S["ytc"]).mean())
        inc = model.include_count.float()

        improved = es["best_val_correct"] < 0 or val_correct >= es["best_val_correct"] + need
        if improved:
            es.update(best_val_correct=val_correct, best_epoch=epoch, since=0)
            save_best(args, model, rec, epoch, val_acc, test_acc)
        else:
            es["since"] += 1
        rec["elapsed_s"] += train_s + val_s + test_s           # as cifar10_batch_clauses (comparable GPU time)
        row = {"epoch": epoch, "val_acc": val_acc, "best_val_acc": 100 * es["best_val_correct"] / n_va,
               "epochs_since_improvement": es["since"], "test_acc": test_acc,
               "train_acc_running": 100 * int(correct) / n_tr, "train_eval_acc": train_eval_acc,
               "gap": train_eval_acc - val_acc, "train_s": train_s, "val_s": val_s, "test_s": test_s,
               "train_eval_s": train_eval_s, "aug_wait_s": aug_wait, "examples_per_s": n_tr / train_s,
               "commits": epoch * math.ceil(n_tr / bs), "peak_gpu_mib": peak_mib(dev),
               "mean_clause_len": float(inc.mean()), "frac_empty_clauses": float((inc == 0).float().mean()),
               "vote_saturation": saturation, "elapsed_s": rec["elapsed_s"]}
        rec["curve"].append(row)

        stop = None
        if epoch >= args.min_epochs and es["since"] >= args.patience:
            stop = "patience"
        elif epoch >= args.max_epochs:
            stop = "max_epochs"
        elif rec["elapsed_s"] >= args.time_cap_h * 3600:
            stop = "time_cap"
        images = None
        if args.viz_every and (epoch % args.viz_every == 0 or stop):       # not timed: explanation only
            from viz import draw
            images, scalars = draw(model, S["pan"], xva, va_sums, yva, T, rec["curve"],
                                   f"{tag}  epoch {epoch}  val {val_acc:.2f} %  test {test_acc:.2f} %  "
                                   f"train (clean) {train_eval_acc:.2f} %",
                                   FIGURES / args.phase / tag, f"e{epoch:03d}")
            row.update(scalars)
        tracker.epoch(row, images)
        print(f"[{tag}] epoch {epoch:3d} val {val_acc:6.2f} test {test_acc:6.2f} train {row['train_acc_running']:6.2f} "
              f"clean {train_eval_acc:6.2f} gap {row['gap']:5.2f} | best {row['best_val_acc']:6.2f}@{es['best_epoch']} "
              f"wait {es['since']:2d} | {train_s:.1f}s + {val_s + test_s + train_eval_s:.1f}s (aug {aug_wait:.1f}s) | "
              f"len {row['mean_clause_len']:.1f} sat {saturation:.2f}", flush=True)
        if stop:
            finish(rec, stop, es, n_tr, bs)
        write_json(rec, S["out"])
        if stop or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = S["ckpt"].with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "gen": S["gen"].get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                        "epoch": epoch, "rec": rec}, tmp)
            tmp.replace(S["ckpt"])
        if stop:
            S["ckpt"].unlink(missing_ok=True)       # the record holds everything that is reported
            best_figures(args, S)
            write_json(rec, S["out"])
            print(f"[{tag}] stop ({stop}) after {epoch} epochs: best val {rec['best_val_acc']:.2f} @ "
                  f"{rec['best_epoch']}, test there {rec['test_acc_at_best']:.2f}, train (clean) there "
                  f"{rec['train_eval_acc_at_best']:.2f}", flush=True)
            return
        if epoch == args.stop_after:
            print(f"[{tag}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


def save_best(args, model, rec, epoch, val_acc, test_acc):
    """The model at the best-validation epoch: models/<phase>/<tag>.pt, overwritten at every new
    best, never committed (git-ignored). Written after the epoch's timing. A resumed run replays
    the epochs after its checkpoint bit for bit, so it rewrites the same best model. Load it
    with ``train.load_model(path, device)``."""
    if args.phase in NO_MODELS:
        return
    path = MODELS / args.phase / f"{rec['tag']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc, "test_acc": test_acc,
                "tag": rec["tag"], "hp": rec["hp"], "git_sha": rec["git_sha"], "env": rec["env"]}, tmp)
    tmp.replace(path)
    rec["best_model"] = {"path": str(path.relative_to(MODELS.parent)) if MODELS.parent in path.parents
                         else str(path), "epoch": epoch}


def best_figures(args, S):
    """Evidence panel and clause health of the best-validation model (plan §7), after the run."""
    rec = S["rec"]
    path = MODELS / args.phase / f"{rec['tag']}.pt"
    if not args.viz_every or args.phase in NO_MODELS or not path.exists():
        return
    from viz import health, panel
    model, st = load_model(path, S["dev"])
    sums = class_sums(model, S["xva"], args.eval_batch).float().cpu().numpy()
    model.eval()
    assert abs(100 * float((sums.argmax(1) == S["yva"]).mean()) - st["val_acc"]) < 1e-6, "saved model differs"
    title = (f"{rec['tag']}  BEST epoch {st['epoch']}  val {st['val_acc']:.2f} %  test {st['test_acc']:.2f} %")
    folder = FIGURES / args.phase / rec["tag"]
    panel(model, S["pan"]["x"], S["pan"]["raw"], S["pan"]["y"], title, folder / "best_panel.png")
    rec["best_health"] = health(model, S["xva"], sums, S["yva"], S["T"], title, folder / "best_health.png")
    S["tracker"].epoch({"epoch": rec["epochs_run"]}, {"viz/best_panel": (folder / "best_panel.png", title),
                                                      "viz/best_health": (folder / "best_health.png", title)})


def finish(rec, stop, es, n_tr, bs):
    c = rec["curve"]
    best = c[es["best_epoch"] - 1]
    assert best["epoch"] == es["best_epoch"]
    rec.update(status="done", stop_reason=stop, best_epoch=es["best_epoch"],
               best_val_acc=best["val_acc"], test_acc_at_best=best["test_acc"],
               train_eval_acc_at_best=best["train_eval_acc"], gap_at_best=best["gap"],
               train_acc_running_at_best=best["train_acc_running"],
               epochs_run=len(c), commits_to_best=es["best_epoch"] * math.ceil(n_tr / bs),
               gpu_s_to_best=best["elapsed_s"], test_acc_last=c[-1]["test_acc"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               aug_wait_s_mean=float(np.mean([r["aug_wait_s"] for r in c])),
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
    eval_epoch = eval_s * (xva.shape[0] + xte.shape[0] + min(5000, n_tr)) / xva.shape[0]
    rec = new_record("calibration", device=dev, phase="pricing", tag=tag, hp=hp, status="done",
                     warmup_s=warm, n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=train_s, eval_s_per_epoch=eval_epoch, peak_gpu_mib=peak_mib(dev))
    print(f"[{tag}] calibrate: {per*1000:.2f} ms/update -> train {train_s:.1f} s/epoch, eval "
          f"{eval_epoch:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / "pricing" / f"{tag}.json")


if __name__ == "__main__":
    main()
