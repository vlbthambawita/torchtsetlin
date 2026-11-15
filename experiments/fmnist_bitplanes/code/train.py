"""Train one cell (architecture x encoding x clauses x s [x kd]) on Fashion-MNIST until early stopping.

Plan: torchtsetlin_plans/plans/fmnist-bitplanes.md. Architectures: ``flat`` (B4 base, planes flattened), ``conv2``
(B8 base, planes as channels), ``conv3`` (B8 base, planes along depth, window ``(kd, 10, 10)``).
Encodings: ``agt`` (the paper's B8), ``therm8``/``therm4`` (thermometer), ``bit8``/``bit4`` (bit
planes), ``gray8`` (Gray-coded bit planes). T scales with the clauses per class (D4); batched
feedback with B = 32 (D5). Training set = 50 000 images, the other 10 000 are the validation set
that early stopping watches (D6); the test set is evaluated every epoch but never used for a
decision. Primary statistic: test accuracy at the best-validation epoch (D7).

Per epoch the record also holds ``plane_usage``: the share of all included literals that sit on
each plane (positive / negated), and on the position bits (conv), read from the automata (Q3).

Usage:  python code/train.py --arch conv3 --enc bit8 --clauses 2000 --kd 2 --device cuda:1
        python code/train.py ... --calibrate 200 --arm pricing   # time 200 updates + one eval
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch

import torchtsetlin as tt
from augment import Augmenter, encode_gpu
from common import (AUGMENTS, BASE, BATCH, CACHE, EARLY, ENCODINGS, MODELS, PILOT_MAX_EPOCHS,
                    RESULTS, SAVE_ARMS, T_of, load_data, new_record, tag_of, write_json)
from tracking import Tracker

ARMS = ["smoke", "pricing", "sanity", "pilot", "main", "ablation", "seeds", "resume_check",
        "aug", "replay", "aug_check"]


def build_model(arch, planes, clauses, T, s, kd):
    b = BASE[arch]
    kw = dict(weighted=b["weighted"], boost_true_positive=b["boost_true_positive"],
              feedback_mode="batch")
    if arch == "flat":
        return tt.TsetlinMachine(planes * 784, 10, clauses, T, s, **kw)
    if arch == "conv2":
        return tt.ConvTsetlinMachine(10, clauses, T, s, patch_size=b["patch"], stride=1,
                                     position_encoding=b["position_encoding"],
                                     input_shape=(planes, 28, 28), **kw)
    if not 1 <= kd <= planes:
        raise ValueError(f"kd={kd} must be in 1..{planes}")
    return tt.Conv3dTsetlinMachine(10, clauses, T, s, patch_size=(kd, b["patch"], b["patch"]),
                                   stride=1, position_encoding=b["position_encoding"],
                                   input_shape=(1, planes, 28, 28),
                                   max_chunk_elements=b["max_chunk_elements"], **kw)


def feature_groups(arch, planes, kd, n_features):
    """Group id per Boolean feature: the plane (flat, conv2) or the depth offset in the window
    (conv3); position bits get the last id. Layouts: flat features are plane-major (p*784 + pixel);
    conv patches are unfolded channel-major (Z*kh*kw, then position bits)."""
    g = np.empty(n_features, np.int64)
    if arch == "flat":
        g[:] = np.arange(n_features) // 784
        return g, [f"p{i}" for i in range(planes)], None
    n_pix = (planes if arch == "conv2" else kd) * 100
    g[:n_pix] = np.arange(n_pix) // 100
    g[n_pix:] = n_pix // 100
    k = n_pix // 100
    return g, [f"{'p' if arch == 'conv2' else 'd'}{i}" for i in range(k)], k


@torch.no_grad()
def plane_usage(model, groups, names, pos_id, rows=4096):
    """Share of all included literals per group, split into positive and negated literals."""
    st, N = model.ta_state, model.n_states
    cnt = torch.zeros(st.shape[1], dtype=torch.long, device=st.device)
    for i in range(0, st.shape[0], rows):
        cnt += (st[i:i + rows] >= N).sum(0)
    cnt = cnt.cpu().numpy()
    F = st.shape[1] // 2
    G = len(names) + (1 if pos_id is not None else 0)
    pos = np.bincount(groups, weights=cnt[:F], minlength=G)
    neg = np.bincount(groups, weights=cnt[F:], minlength=G)
    tot = max(1.0, float(pos.sum() + neg.sum()))
    out = {"names": names, "pos": (pos[:len(names)] / tot).tolist(),
           "neg": (neg[:len(names)] / tot).tolist(), "n_included": int(tot)}
    if pos_id is not None:
        out["position"] = float((pos[pos_id] + neg[pos_id]) / tot)
    return out


@torch.no_grad()
def class_sums(model, x, eval_batch):
    """Raw class sums in prediction mode (empty clauses evaluate False)."""
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
    ap.add_argument("--arch", required=True, choices=sorted(BASE))
    ap.add_argument("--enc", required=True, choices=sorted(ENCODINGS))
    ap.add_argument("--clauses", type=int, required=True, help="clauses per class")
    ap.add_argument("--s", type=float, default=None, help="specificity (default: the base config's)")
    ap.add_argument("--kd", type=int, default=None, help="conv3 window depth")
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="main", choices=ARMS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--patience", type=int, default=EARLY["patience"])
    ap.add_argument("--min-delta", type=float, default=EARLY["min_delta"])
    ap.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    ap.add_argument("--max-epochs", type=int, default=None,
                    help=f"default {EARLY['max_epochs']} ({PILOT_MAX_EPOCHS} in the pilot)")
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"])
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="time this many updates and exit")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--augment", default="none", choices=sorted(AUGMENTS),
                    help="training-time augmentation on the uint8 pixels (bit8/gray8 only)")
    ap.add_argument("--save-best", type=int, default=None,
                    help="1 = keep the best-validation model in models/<arm>/<tag>.pt "
                         f"(default: on for arms {sorted(SAVE_ARMS)})")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.max_epochs is None:
        args.max_epochs = PILOT_MAX_EPOCHS if args.arm == "pilot" else EARLY["max_epochs"]
    if args.s is None:
        args.s = BASE[args.arch]["s"]
    if args.arch == "conv3" and args.kd is None:
        raise SystemExit("--kd is required for conv3")
    if args.arch != "conv3":
        args.kd = None
    if args.save_best is None:
        args.save_best = int(args.arm in SAVE_ARMS)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)
    T = T_of(args.arch, args.clauses)
    planes = ENCODINGS[args.enc]
    tag = tag_of(args.arch, args.enc, args.clauses, args.s, args.kd, args.seed, args.augment) + args.tag_suffix

    d = load_data(args.enc)
    if args.subset:
        n, m = args.subset, max(args.subset // 4, 100)
        for k, v in (("train", n), ("val", m), ("test", m)):
            d[f"x_{k}"], d[f"y_{k}"] = d[f"x_{k}"][:v], d[f"y_{k}"][:v]

    model = build_model(args.arch, planes, args.clauses, T, args.s, args.kd).to(dev)

    def to_x(a):                                    # a: (N, P, 28, 28) bool (numpy or tensor)
        t = a if torch.is_tensor(a) else torch.from_numpy(np.ascontiguousarray(a))
        t = t.to(torch.bool)
        if args.arch == "flat":
            t = t.reshape(len(a), -1)               # plane-major features: p*784 + pixel
        elif args.arch == "conv3":
            t = t.unsqueeze(1)                      # (N, 1, D=P, 28, 28)
        return t.to(dev)

    xtr, xva, xte = to_x(d["x_train"]), to_x(d["x_val"]), to_x(d["x_test"])
    ytr = torch.from_numpy(d["y_train"]).to(dev)
    yva, yte = d["y_val"], d["y_test"]
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)   # D9: order = f(seed)
    aug = AUGMENTS[args.augment]
    args.augmenter = None
    if aug:
        aug_gen = torch.Generator(device="cpu").manual_seed(2000 + args.seed)
        args.augmenter = Augmenter(torch.from_numpy(d["raw_train"]).to(dev), aug["shift"],
                                   aug["flip_p"], aug_gen)
        args.batch_x = lambda idx: to_x(encode_gpu(args.enc, args.augmenter.batch(idx)))
    b = BASE[args.arch]
    hp = dict(arch=args.arch, enc=args.enc, planes=planes, clauses_per_class=args.clauses,
              clauses_total=int(model.n_clauses_total), n_features=int(model.n_features),
              n_literals=int(model.n_literals), T=T, T_ref=b["T_ref"], s=args.s, kd=args.kd,
              weighted=b["weighted"], boost_true_positive=b["boost_true_positive"],
              feedback_mode="batch", batch_size=args.batch_size,
              augment=dict(policy=args.augment, **(aug or {}),
                           **({"aug_seed": 2000 + args.seed, "applied_to": "uint8 pixels before encoding, train only"}
                              if aug else {})),
              input_shape=list(xtr.shape[1:]),
              commits_per_epoch=math.ceil(n_tr / args.batch_size), seed=args.seed,
              n_states=int(model.n_states), max_chunk_elements=int(model.max_chunk_elements),
              patch_blocks=(lambda P: -(-P // model._patch_block(P)))(model._patches_per_example(xtr[:1]))
              if args.arch == "conv3" else None,
              **({"patch": b["patch"], "position_encoding": b["position_encoding"]}
                 if args.arch != "flat" else {}),
              early_stopping=dict(monitor="val_acc", patience=args.patience, min_delta_pp=args.min_delta,
                                  min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                  time_cap_h=args.time_cap_h),
              data=dict(sha256=d["sha256"], n_train=n_tr, n_val=n_va, n_test=int(xte.shape[0]),
                        subset=args.subset))
    print(f"[{tag}] {args.arch}/{args.enc} {args.clauses}/class = {model.n_clauses_total} clauses, "
          f"{model.n_literals} literals, T={T}, s={args.s:g}, B={args.batch_size}, {args.device}",
          flush=True)

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
        if args.augmenter is not None:
            args.augmenter.gen.set_state(st["aug_gen"])
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
    rec["status"] = "running"

    tags = [args.arch, args.enc, f"c{args.clauses}", f"spec{args.s:g}", f"seed{args.seed}",
            rec["env"].get("gpu", args.device)] + ([f"kd{args.kd}"] if args.kd else []) + \
        [f"aug-{args.augment}"]
    tracker = Tracker(args.arm, tag, dict(hp, tag=tag, arm=args.arm, git_sha=rec["git_sha"],
                                          env=rec["env"]), tags, enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    groups = feature_groups(args.arch, planes, args.kd, int(model.n_features))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        train(args, model, rec, T, xtr, ytr, xva, yva, xte, yte, gen, dev, out, ckpt, start,
              tracker, groups)
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]),
                     "error": f"{type(e).__name__}: {e}"})
        raise
    summary_keys = ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch",
                    "epochs_run", "gpu_s_to_best", "test_acc_epoch30", "test_acc_last",
                    "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")
    tracker.end({k: rec.get(k) for k in summary_keys})


def json_status(path):
    import json
    try:
        return json.loads(path.read_text()).get("status")
    except Exception:  # noqa: BLE001
        return None


def train(args, model, rec, T, xtr, ytr, xva, yva, xte, yte, gen, dev, out, ckpt, start, tracker,
          groups):
    n_tr, n_va = xtr.shape[0], xva.shape[0]
    bs = args.batch_size
    need = max(1, math.ceil(args.min_delta / 100 * n_va - 1e-9))   # a tie is not an improvement
    es = rec["early"]
    epoch = start
    while True:
        epoch += 1
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=gen).to(dev)
        if args.augmenter is not None:
            args.augmenter.draw_epoch()
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            xb = xtr[idx] if args.augmenter is None else args.batch_x(idx)
            votes = model.update(xb, ytr[idx])                     # pre-update vote sums
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
        per_class = [100 * float((te_pred[yte == k] == k).mean()) for k in range(10)]
        inc = model.include_count.float()
        pu = plane_usage(model, *groups)

        improved = es["best_val_correct"] < 0 or val_correct >= es["best_val_correct"] + need
        if improved:
            es.update(best_val_correct=val_correct, best_epoch=epoch, since=0)
        else:
            es["since"] += 1
        rec["elapsed_s"] += train_s + val_s + test_s
        row = {"epoch": epoch, "val_acc": val_acc, "best_val_acc": 100 * es["best_val_correct"] / n_va,
               "epochs_since_improvement": es["since"], "test_acc": test_acc,
               "test_acc_per_class": per_class,
               "train_acc_running": 100 * int(correct) / n_tr, "train_s": train_s, "val_s": val_s,
               "test_s": test_s, "examples_per_s": n_tr / train_s,
               "peak_gpu_mib": peak_mib(dev), "mean_clause_len": float(inc.mean()),
               "frac_empty_clauses": float((inc == 0).float().mean()), "vote_saturation": saturation,
               "plane_usage": pu, "elapsed_s": rec["elapsed_s"]}
        rec["curve"].append(row)
        if improved and args.save_best:
            rec["best_model"] = str(save_best(args, model, rec, row))
        tracker.epoch(row)
        top = " ".join(f"{n}:{p + q:.2f}" for n, p, q in zip(pu["names"], pu["pos"], pu["neg"]))
        print(f"[{rec['tag']}] epoch {epoch:3d} val {val_acc:6.2f} test {test_acc:6.2f} "
              f"train {row['train_acc_running']:6.2f} | best {row['best_val_acc']:6.2f}@{es['best_epoch']} "
              f"wait {es['since']:2d} | {train_s:.1f}s + {val_s + test_s:.1f}s | len "
              f"{row['mean_clause_len']:.1f} | {top}", flush=True)

        stop = None
        if epoch >= args.min_epochs and es["since"] >= args.patience:
            stop = "patience"
        elif epoch >= args.max_epochs:
            stop = "max_epochs"
        elif rec["elapsed_s"] >= args.time_cap_h * 3600:
            stop = "time_cap"
        if stop:
            finish(rec, stop, es)
        write_json(rec, out)
        if stop or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "gen": gen.get_state(),
                        "cpu_rng": torch.get_rng_state(),
                        "aug_gen": args.augmenter.gen.get_state() if args.augmenter is not None else None,
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


def save_best(args, model, rec, row):
    """The model at the best validation epoch so far -> models/<arm>/<tag>.pt (git-ignored).
    Self-contained: hp holds everything build_model needs; written to .tmp then renamed."""
    path = MODELS / args.arm / f"{rec['tag']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": row["epoch"], "val_acc": row["val_acc"], "test_acc": row["test_acc"],
                "test_acc_per_class": row["test_acc_per_class"], "tag": rec["tag"], "arm": args.arm,
                "hp": rec["hp"], "git_sha": rec["git_sha"], "env": rec["env"],
                "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, tmp)
    tmp.replace(path)
    return path


def finish(rec, stop, es):
    c = rec["curve"]
    best = c[es["best_epoch"] - 1]
    assert best["epoch"] == es["best_epoch"]
    rec.update(status="done", stop_reason=stop, best_epoch=es["best_epoch"],
               best_val_acc=best["val_acc"], test_acc_at_best=best["test_acc"],
               test_acc_per_class_at_best=best["test_acc_per_class"],
               plane_usage_at_best=best["plane_usage"],
               epochs_run=len(c), gpu_s_to_best=best["elapsed_s"],
               test_acc_epoch30=c[29]["test_acc"] if len(c) >= 30 else None,
               test_acc_last=c[-1]["test_acc"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               peak_gpu_mib=max(r["peak_gpu_mib"] for r in c),
               finished=time.strftime("%Y-%m-%dT%H:%M:%S"))


def calibrate(args, model, hp, tag, xtr, ytr, xva, gen, dev):
    """Phase 1 pricing: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch."""
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
    rec = new_record("calibration", device=dev, arm=args.arm, tag=tag, hp=hp, warmup_s=warm,
                     n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=per * math.ceil(50000 / bs),
                     eval_s_per_epoch=eval_s * 20000 / xva.shape[0], peak_gpu_mib=peak_mib(dev))
    print(f"[{tag}] calibrate: {per*1000:.2f} ms/update -> train {rec['train_s_per_epoch']:.1f} s/epoch, "
          f"eval {rec['eval_s_per_epoch']:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / args.arm / f"{tag}.json")


if __name__ == "__main__":
    main()
