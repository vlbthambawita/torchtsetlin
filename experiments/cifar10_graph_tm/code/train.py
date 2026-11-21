"""Train one model of cifar10_graph_tm on CIFAR-10.

Plan: torchtsetlin_plans/experiments/cifar10_graph_tm/PLAN.md. Three kinds of run:

* ``--kind grid`` (arm D, pilot): ``GraphTsetlinMachine`` of ``--depth`` layers on the 15 x 15
  grid of 4 x 4 thermometer windows with four typed neighbour edges (``--edges rewired``: the
  control). Protocol of cifar10_conv_reg: 45 000 / 5 000 split, batch 32, early stopping on
  validation (patience 50), statistic = test accuracy at the best-validation epoch, that model
  kept in ``models/<phase>/<tag>.pt``. ``train_eval_acc`` = accuracy on 5 000 fixed training images.
* ``--kind paper_graph`` / ``paper_cotm`` (arm P): the paper's protocol, not ours: all 50 000
  training images plus their horizontal flips, 30 epochs, statistic = mean test accuracy of the
  last 5 epochs; the final model is kept.

Node features are built per mini-batch on the GPU (code/graphs.py). Every ``--viz-every`` epochs
the figures of code/viz.py are drawn.

Usage:  python code/train.py --kind grid --depth 2 --clauses 20000 --T 10000 --s 5 --phase depth
        python code/train.py --kind grid --depth 2 --clauses 20000 --T 10000 --s 5 --calibrate 100 --phase pricing
"""

import argparse
import math
import signal
import sys
import time
import warnings

import numpy as np
import torch
from common import (
    BATCH,
    BUDGET,
    CACHE,
    EARLY,
    EDGE_TYPES,
    FIGURES,
    GRID,
    INBOX,
    MESSAGE_BITS,
    MODELS,
    PAPER,
    PATCH,
    RESULTS,
    STRIDE,
    VIZ_EVERY,
    load_cifar,
    new_record,
    panel_indices,
    read_json,
    tag_of,
    train_eval_indices,
    write_json,
)
from graphs import adaptive_threshold, make_input, thermometer, two_views
from tracking import Tracker

import torchtsetlin as tt

NO_MODELS = {"pricing", "pilot", "resume_check"}       # throw-away or screening runs: no model kept
N_FEATURES = {"grid": 24 * PATCH * PATCH + 2 * (GRID - 1), "paper_graph": 27 * PAPER["patch"] ** 2 + 2 * PAPER["grid"],
              "paper_cotm": 3 * PAPER["patch"] ** 2 + 2 * (PAPER["grid"] - 1)}
N_NODES = {"grid": GRID * GRID, "paper_graph": PAPER["grid"] ** 2, "paper_cotm": PAPER["grid"] ** 2}


def chunk_budget(kind, depth, clauses, inbox, batch):
    """max_chunk_elements so that one training batch is one chunk (the Triton commit and the CUDA
    graphs need that). Depth 1 runs on the bit-packed kernel (no (P, C) tensor); with message
    passing the layers are evaluated densely. A memory/speed knob only, recorded in hp."""
    P, F0, C = N_NODES[kind], N_FEATURES[kind], clauses
    if depth == 1:
        per = 4 * P * 2 * F0 + 8 * C
    else:
        nt, Ft = len(EDGE_TYPES), F0 + (depth - 1) * inbox
        E = 4 * GRID * (GRID - 1)
        per = P * (C * (3 + nt) + 3 * Ft + inbox) + E * C
    return max(2 ** 27, 2 ** math.ceil(math.log2(1.05 * batch * per)))


def build_model(hp):
    kw = dict(max_included_literals=hp["max_included_literals"], budget_mode="soft", feedback_mode="batch",
              max_chunk_elements=hp["max_chunk_elements"])
    C, T, s = hp["clauses"], hp["T"], hp["s"]
    if hp["kind"] == "paper_cotm":
        return tt.ConvCoalescedTsetlinMachine(10, C, T, s, patch_size=PAPER["patch"], stride=PAPER["stride"],
                                              position_encoding=True, input_shape=(3, 32, 32), **kw)
    if hp["kind"] == "paper_graph":
        return tt.GraphTsetlinMachine(N_FEATURES["paper_graph"], 10, C, T, s, depth=1, **kw)
    return tt.GraphTsetlinMachine(N_FEATURES["grid"], 10, C, T, s, depth=hp["depth"],
                                  message_size=hp["inbox"] or INBOX, message_bits=MESSAGE_BITS,
                                  message_encoding="random", n_edge_types=len(EDGE_TYPES),
                                  edge_type_names=EDGE_TYPES, **kw)


def load_model(path, dev):
    """A kept model (``models/<phase>/<tag>.pt``) -> (model in eval mode, its input builder, file dict)."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    hp = st["hp"]
    model = build_model(hp).to(dev)
    model.load_state_dict(st["model"])
    return model.eval(), make_input(hp["kind"], dev, hp["edges"], hp["seed"]), st


@torch.no_grad()
def class_sums(model, inp, x, eval_batch):
    """Raw (unclamped) class sums in prediction mode (empty clauses evaluate False)."""
    model.eval()
    try:
        return torch.cat([model(inp(x[i:i + eval_batch])) for i in range(0, x.shape[0], eval_batch)])
    finally:
        model.train()


def message_share(model):
    """Share of all included literals that are message literals (0 at depth 1)."""
    if getattr(model, "depth", 1) == 1:
        return 0.0
    Fn, F0 = int(model.n_features), model.n_node_features
    inc = model.ta_state >= model.n_states
    msg = inc[:, F0:Fn].sum() + inc[:, Fn + F0:].sum()
    return float(msg) / max(1.0, float(model.include_count.sum()))


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def peak_mib(dev):
    return torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else 0.0


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def oom(args, tag, out, dev, err, rec=None, kind="run", hp=None):
    """A run that does not fit the GPU is a result ("oom"), reported, never silently dropped."""
    r = rec or new_record(kind, device=dev, phase=args.phase, tag=tag, hp=hp)
    r.update(status="oom", error=f"{type(err).__name__}: {str(err)[:400]}",
             finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    write_json(r, out)
    print(f"[{tag}] out of memory: recorded as status=oom", flush=True)


def load_data(args, dev):
    """Image planes on the GPU. Returns (xtr, ytr, {name: (x, y as numpy)}, raw held-out images, n_full)."""
    d = load_cifar(args.subset)
    if args.kind == "grid":
        (raw_tr, y_tr), (raw_va, yva), (raw_te, yte) = d["train"], d["val"], d["test"]
        idx = train_eval_indices(len(raw_tr))
        sets = {"val": (thermometer(raw_va, dev), yva), "test": (thermometer(raw_te, dev), yte),
                "train_eval": (thermometer(raw_tr[idx], dev), y_tr[idx])}
        return thermometer(raw_tr, dev), y_tr, sets, raw_va, 5000
    (raw_tr, y_tr), (raw_te, yte) = d["train_all"], d["test"]
    raw_tr = np.concatenate([raw_tr, raw_tr[:, :, ::-1]])              # the paper: plus horizontal flips
    y_tr = np.concatenate([y_tr, y_tr])
    enc = two_views if args.kind == "paper_graph" else (lambda x, dv: torch.from_numpy(adaptive_threshold(x)).to(dv))
    return enc(raw_tr, dev), y_tr, {"test": (enc(raw_te, dev), yte)}, raw_te, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="grid", choices=["grid", "paper_graph", "paper_cotm"])
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--clauses", type=int, required=True, help="shared clauses (total)")
    ap.add_argument("--inbox", type=int, default=INBOX, help="hashed inbox positions per message layer")
    ap.add_argument("--edges", default="grid", choices=["grid", "rewired"])
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--s", type=float, required=True)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = 32 (grid) or the paper arm's 16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="depth",
                    choices=["depth", "pilot", "paper", "smoke", "pricing", "resume_check", "extra"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--planned", default=None, help="tag of the planned cell (differs after the gate)")
    ap.add_argument("--role", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--patience", type=int, default=EARLY["patience"])
    ap.add_argument("--min-delta", type=float, default=EARLY["min_delta"])
    ap.add_argument("--min-epochs", type=int, default=EARLY["min_epochs"])
    ap.add_argument("--max-epochs", type=int, default=0, help="0 = 300 (grid) or the paper arm's 30")
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"], help="GPU-hours of one run")
    ap.add_argument("--viz-every", type=int, default=VIZ_EVERY, help="0 = no figures")
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--ckpt-every", type=int, default=0, help="0 = 5 (grid) or 1 (paper arm)")
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 held-out images")
    ap.add_argument("--calibrate", type=int, default=0, help="pricing: time this many updates and exit")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    paper = args.kind != "grid"
    if paper:
        args.depth, args.edges = 1, "grid"
    args.batch_size = args.batch_size or (PAPER["batch"] if paper else BATCH)
    args.max_epochs = args.max_epochs or (PAPER["epochs"] if paper else EARLY["max_epochs"])
    args.ckpt_every = args.ckpt_every or (1 if paper else 5)
    inbox = args.inbox if args.depth > 1 else 0
    cell = dict(kind=args.kind, depth=args.depth, clauses=args.clauses, inbox=inbox, edges=args.edges)
    tag = args.tag or tag_of(cell, args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)          # the fast CUDA path needs the model's GPU to be current
    tt.seed_everything(args.seed)
    out = RESULTS / args.phase / f"{tag}.json"
    if (read_json(out) or {}).get("status") in ("done", "oom"):
        print(f"[{tag}] already finished", flush=True)
        return

    hp = dict(cell, model="ConvCoalescedTsetlinMachine" if args.kind == "paper_cotm" else "GraphTsetlinMachine",
              T=args.T, s=args.s, seed=args.seed, batch_size=args.batch_size, role=args.role,
              planned=args.planned or tag, protocol="paper" if paper else "early_stopping",
              max_included_literals=BUDGET, budget_mode="soft", feedback_mode="batch",
              max_chunk_elements=chunk_budget(args.kind, args.depth, args.clauses, inbox, args.batch_size),
              n_nodes=N_NODES[args.kind], n_node_features=N_FEATURES[args.kind],
              message_bits=MESSAGE_BITS if args.depth > 1 else 0,
              message_encoding="random" if args.depth > 1 else None,
              edge_types=EDGE_TYPES if args.kind == "grid" else [], viz_every=args.viz_every)
    if args.kind == "grid":
        hp.update(encoding="therm8", patch_size=PATCH, stride=STRIDE,
                  early_stopping=dict(monitor="val_acc", patience=args.patience, min_delta_pp=args.min_delta,
                                      min_epochs=args.min_epochs, max_epochs=args.max_epochs,
                                      time_cap_h=args.time_cap_h))
    else:
        hp.update(encoding="agt+therm8, one feature per symbol" if args.kind == "paper_graph" else "agt",
                  patch_size=PAPER["patch"], stride=PAPER["stride"], epochs=args.max_epochs, last=PAPER["last"],
                  time_cap_h=PAPER["time_cap_h"], best_effort=True)
    try:
        model = build_model(hp).to(dev)
        inp = make_input(args.kind, dev, args.edges, args.seed)
        xtr, y_tr, sets, raw_held, n_held_full = load_data(args, dev)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if not is_oom(e):
            raise
        return oom(args, tag, out, dev, e, hp=hp, kind="calibration" if args.calibrate else "run")
    ytr = torch.from_numpy(y_tr).to(dev)
    held = "test" if paper else "val"
    pidx = panel_indices(sets[held][1], held, n_held_full)
    pan = {"x": sets[held][0][torch.as_tensor(pidx, device=dev)], "raw": raw_held[pidx], "y": sets[held][1][pidx]}
    n_tr = xtr.shape[0]
    hp.update(clauses_total=int(model.n_clauses_total), n_states=int(model.n_states),
              boost_true_positive=bool(model.boost_true_positive),
              commits_per_epoch=math.ceil(n_tr / args.batch_size),
              data=dict(dataset="CIFAR-10", n_train=n_tr, subset=args.subset, flips=paper,
                        **{f"n_{k}": len(v[1]) for k, v in sets.items()}))
    print(f"[{tag}] {hp['model']} {args.kind} depth {args.depth}, {args.clauses} clauses, T={args.T}, "
          f"s={args.s:g}, inbox {inbox}, edges {args.edges}, B={args.batch_size}, {n_tr} training images, "
          f"{args.device}", flush=True)
    # The data order depends on the seed only.
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)

    if args.calibrate:
        try:
            calibrate(args, model, inp, hp, tag, xtr, ytr, sets, gen, dev, out)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not is_oom(e):
                raise
            return oom(args, tag, out, dev, e, kind="calibration", hp=hp)
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

    tags = [args.kind, f"d{args.depth}", f"c{args.clauses}", f"m{inbox}", args.edges, f"s{args.seed}",
            rec["env"].get("gpu", args.device)] + ([args.role] if args.role else [])
    tracker = Tracker(args.phase, tag, dict(rec["hp"], tag=tag, phase=args.phase, git_sha=rec["git_sha"],
                                            env=rec["env"]), tags, enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    S = dict(model=model, inp=inp, rec=rec, xtr=xtr, ytr=ytr, sets=sets, held=held, pan=pan, gen=gen, dev=dev,
             out=out, ckpt=ckpt, start=start, tracker=tracker, paper=paper)
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
    keys = ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch", "train_eval_acc_at_best",
            "gap_at_best", "test_acc_last_mean", "test_acc_last_sd", "epochs_run", "test_acc_last",
            "message_literal_share", "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")
    tracker.end({k: rec.get(k) for k in keys})


def train(args, S):
    model, inp, rec, dev, tracker, paper = S["model"], S["inp"], S["rec"], S["dev"], S["tracker"], S["paper"]
    xtr, ytr, sets, held = S["xtr"], S["ytr"], S["sets"], S["held"]
    n_tr, bs, tag, T = xtr.shape[0], args.batch_size, rec["tag"], rec["hp"]["T"]
    n_va = len(sets[held][1])
    # Validation examples a new best must add: >= 1 (a tie is not an improvement), more with min_delta.
    need = max(1, math.ceil(args.min_delta / 100 * n_va - 1e-9))
    es = rec["early"]
    epoch = S["start"]
    while True:
        epoch += 1
        model.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=S["gen"]).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            votes = model.update(inp(xtr[idx]), ytr[idx])           # pre-update vote sums
            correct += (votes.argmax(1) == ytr[idx]).sum()
        sync(dev)
        train_s = time.time() - t0

        acc, secs, sums = {}, {}, {}
        for name, (x, y) in sets.items():
            t0 = time.time()
            sums[name] = class_sums(model, inp, x, args.eval_batch).float()
            sync(dev)
            secs[name] = time.time() - t0
            acc[name] = (int((sums[name].argmax(1).cpu().numpy() == y).sum()), len(y))
        pct = {k: 100 * c / n for k, (c, n) in acc.items()}
        inc = model.include_count.float()
        rec["elapsed_s"] += train_s + sum(v for k, v in secs.items() if k != "train_eval")
        row = {"epoch": epoch, "test_acc": pct["test"], "train_acc_running": 100 * int(correct) / n_tr,
               "train_s": train_s, "test_s": secs["test"], "examples_per_s": n_tr / train_s,
               "commits": epoch * math.ceil(n_tr / bs), "peak_gpu_mib": peak_mib(dev),
               "mean_clause_len": float(inc.mean()), "frac_empty_clauses": float((inc == 0).float().mean()),
               "vote_saturation": float((sums["test"].max(1).values >= T).float().mean()),
               "message_literal_share": message_share(model), "elapsed_s": rec["elapsed_s"]}
        if not paper:
            improved = es["best_val_correct"] < 0 or acc["val"][0] >= es["best_val_correct"] + need
            if improved:
                es.update(best_val_correct=acc["val"][0], best_epoch=epoch, since=0)
                save_model(args, model, rec, epoch, val_acc=pct["val"], test_acc=pct["test"])
            else:
                es["since"] += 1
            row.update(val_acc=pct["val"], best_val_acc=100 * es["best_val_correct"] / n_va,
                       epochs_since_improvement=es["since"], train_eval_acc=pct["train_eval"],
                       gap=pct["train_eval"] - pct["val"], val_s=secs["val"], train_eval_s=secs["train_eval"])
        rec["curve"].append(row)

        stop = None
        if paper:
            if epoch >= args.max_epochs:
                stop = "epochs"
        elif epoch >= args.min_epochs and es["since"] >= args.patience:
            stop = "patience"
        elif epoch >= args.max_epochs:
            stop = "max_epochs"
        if stop is None and rec["elapsed_s"] >= args.time_cap_h * 3600:
            stop = "time_cap"
        images = None
        head = (f"{tag}  epoch {epoch}  test {pct['test']:.2f} %" if paper else
                f"{tag}  epoch {epoch}  val {pct['val']:.2f} %  test {pct['test']:.2f} %  "
                f"train (5 000) {pct['train_eval']:.2f} %")
        if args.viz_every and (epoch % args.viz_every == 0 or stop):       # not timed: explanation only
            from viz import draw
            images, scalars = draw(model, inp, S["pan"], sets[held][0], sums[held].cpu().numpy(), sets[held][1],
                                   T, rec["curve"], head, FIGURES / args.phase / tag, f"e{epoch:03d}", held)
            row.update(scalars)
        tracker.epoch(row, images)
        mid = (f"test {pct['test']:6.2f} train {row['train_acc_running']:6.2f}" if paper else
               f"val {pct['val']:6.2f} test {pct['test']:6.2f} train {row['train_acc_running']:6.2f} "
               f"gap {row['gap']:5.2f} | best {row['best_val_acc']:6.2f}@{es['best_epoch']} wait {es['since']:2d}")
        print(f"[{tag}] epoch {epoch:3d} {mid} | {train_s:.1f}s + {sum(secs.values()):.1f}s | len "
              f"{row['mean_clause_len']:.1f} msg {100 * row['message_literal_share']:.1f}% sat "
              f"{row['vote_saturation']:.2f}", flush=True)
        if stop:
            finish(args, rec, stop, es, n_tr, bs, paper)
            if paper:
                save_model(args, model, rec, epoch, test_acc=pct["test"])
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
            if not paper:
                best_figures(args, S)
            write_json(rec, S["out"])
            what = (f"mean of the last {rec['last_n']} test epochs {rec['test_acc_last_mean']:.2f}" if paper else
                    f"best val {rec['best_val_acc']:.2f} @ {rec['best_epoch']}, test there "
                    f"{rec['test_acc_at_best']:.2f}")
            print(f"[{tag}] stop ({stop}) after {epoch} epochs: {what}", flush=True)
            return
        if epoch == args.stop_after:
            print(f"[{tag}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


def save_model(args, model, rec, epoch, **accs):
    """models/<phase>/<tag>.pt (git-ignored): the model at the best-validation epoch, overwritten
    at every new best (arm D), or the final model (arm P). Written after the epoch's timing. A
    resumed run replays the epochs after its checkpoint bit for bit, so it rewrites the same
    model. Load it with ``train.load_model(path, device)``."""
    if args.phase in NO_MODELS:
        return
    path = MODELS / args.phase / f"{rec['tag']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "epoch": epoch, "tag": rec["tag"], "hp": rec["hp"],
                "git_sha": rec["git_sha"], "env": rec["env"], **accs}, tmp)
    tmp.replace(path)
    rec["kept_model"] = {"path": str(path.relative_to(MODELS.parent)) if MODELS.parent in path.parents
                         else str(path), "epoch": epoch}


def best_figures(args, S):
    """Evidence panel, clause health and rules of the best-validation model, after the run."""
    rec = S["rec"]
    path = MODELS / args.phase / f"{rec['tag']}.pt"
    if not args.viz_every or args.phase in NO_MODELS or not path.exists():
        return
    from viz import draw
    model, inp, st = load_model(path, S["dev"])
    x, y = S["sets"]["val"]
    sums = class_sums(model, inp, x, args.eval_batch).float().cpu().numpy()
    assert abs(100 * float((sums.argmax(1) == y).mean()) - st["val_acc"]) < 1e-6, "saved model differs"
    title = f"{rec['tag']}  BEST epoch {st['epoch']}  val {st['val_acc']:.2f} %  test {st['test_acc']:.2f} %"
    images, scalars = draw(model, inp, S["pan"], x, sums, y, rec["hp"]["T"], None, title,
                           FIGURES / args.phase / rec["tag"], "best", "val")
    rec["best_health"] = scalars
    S["tracker"].epoch({"epoch": rec["epochs_run"]}, {k.replace("viz/", "viz/best_"): v for k, v in images.items()})


def finish(args, rec, stop, es, n_tr, bs, paper):
    c = rec["curve"]
    rec.update(status="done", stop_reason=stop, epochs_run=len(c), test_acc_last=c[-1]["test_acc"],
               message_literal_share=c[-1]["message_literal_share"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               peak_gpu_mib=max(r["peak_gpu_mib"] for r in c), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if paper:
        last = [r["test_acc"] for r in c[-PAPER["last"]:]]
        rec.update(last_n=len(last), test_acc_last_mean=float(np.mean(last)), test_acc_last_sd=float(np.std(last)),
                   test_acc_best_logged_only=max(r["test_acc"] for r in c),
                   complete=len(c) >= args.max_epochs)
        return
    best = c[es["best_epoch"] - 1]
    assert best["epoch"] == es["best_epoch"]
    rec.update(best_epoch=es["best_epoch"], best_val_acc=best["val_acc"], test_acc_at_best=best["test_acc"],
               train_eval_acc_at_best=best["train_eval_acc"], gap_at_best=best["gap"],
               message_literal_share_at_best=best["message_literal_share"],
               commits_to_best=es["best_epoch"] * math.ceil(n_tr / bs), gpu_s_to_best=best["elapsed_s"])


def calibrate(args, model, inp, hp, tag, xtr, ytr, sets, gen, dev, out):
    """Pricing: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch estimate, and
    the evaluation time of 512 held-out images scaled to all evaluated images of an epoch."""
    n_tr, bs = xtr.shape[0], args.batch_size
    perm = torch.randperm(n_tr, generator=gen).to(dev)
    n_batches = max(1, min(args.calibrate, n_tr // bs - 10))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def step(i):
        idx = perm[i * bs:(i + 1) * bs]
        model.update(inp(xtr[idx]), ytr[idx])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.time()
        for i in range(10):
            step(i)
        sync(dev)
        warm = time.time() - t0
        t0 = time.time()
        for i in range(10, 10 + n_batches):
            step(i)
        sync(dev)
        per = (time.time() - t0) / n_batches
        x = next(iter(sets.values()))[0][:512]
        t0 = time.time()
        class_sums(model, inp, x, args.eval_batch)
        sync(dev)
        eval_s = time.time() - t0
    n_eval = sum(len(v[1]) for v in sets.values()) if not args.subset else (
        20000 if args.kind == "grid" else 10000)
    full_tr = n_tr if not args.subset else (45000 if args.kind == "grid" else 100000)
    train_s = per * math.ceil(full_tr / bs)
    graphs = [v for v in model.__dict__.get("_graphs", {}).values() if not isinstance(v, str)]
    rec = new_record("calibration", device=dev, phase=args.phase, tag=tag, hp=hp, status="done",
                     warmup_s=warm, n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=train_s, eval_s_per_epoch=eval_s * n_eval / len(x),
                     peak_gpu_mib=peak_mib(dev), cuda_graphs_captured=len(graphs),
                     warnings=sorted({str(w.message)[:300] for w in caught})[:8])
    print(f"[{tag}] calibrate: {per*1000:.2f} ms/update -> train {train_s:.1f} s/epoch, eval "
          f"{rec['eval_s_per_epoch']:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB, "
          f"{len(graphs)} CUDA graph(s), {len(rec['warnings'])} warning(s)", flush=True)
    write_json(rec, out)


if __name__ == "__main__":
    main()
