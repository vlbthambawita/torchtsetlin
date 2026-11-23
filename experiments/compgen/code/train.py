"""One torchtsetlin run of the benchmark (arms ``neg`` and ``both``).

Model: ``TsetlinMachine`` with two classes x ``--clauses`` clauses, T, s = 25, 8 state bits,
unweighted, boost true positive, literal budget 100. Arm ``neg`` uses ``literals="negated"`` (the
reference setting: negated input, no appended negated literals), arm ``both`` the standard literals.
Feedback: ``seq`` (one example at a time, the reference's mode) or ``b<N>`` (batches of N).

Every epoch: one pass over the training set in a new random order, then test and training accuracy;
the benchmark's clause metrics every epoch up to 200, then every 5th. The run stops when test accuracy
is >= 99.9 % for 20 epochs in a row, at ``--max-epochs`` or at the time cap. Nothing is selected: the
statistic is the time to reach each accuracy mark (common.finish).

Usage: python code/train.py --phase p1 --arm neg --feedback b32 --seed 0
"""

import argparse
import signal
import sys
import time

import torch
from common import (
    CACHE,
    CKPT_EVERY,
    MODEL,
    MODELS,
    N_FEATURES,
    PHASES,
    PRICE_EPOCHS,
    RESULTS,
    SEQ_CHUNK,
    STOP,
    finish,
    metrics_due,
    new_record,
    read_json,
    ref_line,
    should_stop,
    tag_of,
    write_json,
)
from data import load
from metrics import evaluate
from tracking import Tracker

import torchtsetlin as tt


def build(arm, clauses, T, feedback, dev):
    return tt.TsetlinMachine(
        N_FEATURES, 2, clauses, T=T, s=MODEL["s"], n_states=2 ** (MODEL["state_bits"] - 1), weighted=False,
        boost_true_positive=True, max_included_literals=MODEL["literal_budget"],
        feedback_mode="sequential" if feedback == "seq" else "batch",
        literals="negated" if arm == "neg" else "both").to(dev)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=PHASES + ["resume_check"])
    ap.add_argument("--arm", required=True, choices=["neg", "both"])
    ap.add_argument("--feedback", required=True, help="seq or b<batch size>")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clauses", type=int, default=MODEL["clauses"] // 2, help="per class")
    ap.add_argument("--T", type=int, default=None, help="default: 10 per 100 clauses per class")
    ap.add_argument("--max-epochs", type=int, default=STOP["max_epochs"])
    ap.add_argument("--time-cap-h", type=float, default=STOP["time_cap_h"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N training and test examples")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    T = args.T or MODEL["T"] * args.clauses // (MODEL["clauses"] // 2)
    bs = SEQ_CHUNK if args.feedback == "seq" else int(args.feedback[1:])

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)                    # the fast CUDA path runs on the current device
    tt.seed_everything(args.seed)
    tag = tag_of(args.arm, args.feedback, args.seed, args.clauses) + args.tag_suffix
    d, digest = load(args.seed)
    n = args.subset or len(d["y_train"])
    x = {k: torch.from_numpy(d[f"x_{k}"][:n]).bool().to(dev) for k in ("train", "test")}
    y = {k: torch.from_numpy(d[f"y_{k}"][:n]).to(dev) for k in ("train", "test")}
    model = build(args.arm, args.clauses, T, args.feedback, dev)
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)       # data order = f(seed)
    hp = dict(arm=args.arm, library="torchtsetlin", model="TsetlinMachine", n_classes=2, clauses_per_class=args.clauses,
              clauses_total=int(model.n_clauses_total), T=T, s=MODEL["s"], n_states=model.n_states, weighted=False,
              boost_true_positive=True, max_included_literals=MODEL["literal_budget"], literals=model.literals,
              feedback=args.feedback, batch_size=1 if args.feedback == "seq" else bs,
              updates_per_epoch=n if args.feedback == "seq" else -(-n // bs), seed=args.seed, shuffle=True,
              data=dict(sha256=digest, n_train=n, n_test=n, subset=args.subset,
                        stats=read_json(RESULTS / "data" / f"seed{args.seed}.json")["stats"]),
              stop=dict(STOP, max_epochs=args.max_epochs, time_cap_h=args.time_cap_h))
    print(f"[{tag}] {args.arm}: 2 x {args.clauses} clauses, T={T}, s={MODEL['s']:g}, literals={model.literals}, "
          f"feedback={args.feedback}, seed {args.seed}, {args.device}", flush=True)

    out = RESULTS / args.phase / f"{tag}.json"
    ckpt = CACHE / "ckpt" / args.phase / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", device=dev, phase=args.phase, tag=tag, hp=hp, curve=[], status="running", train_s_total=0.0,
                     elapsed_s=0.0)
    start = 0
    if ckpt.exists():                                 # resume an interrupted run
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(st["cuda_rng"], dev)
        rec, start = st["rec"], st["epoch"]
        rec["curve"] = [c for c in rec["curve"] if c["epoch"] <= start]
        rec.setdefault("resumed_at", []).append({"epoch": start, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[{tag}] resumed after epoch {start}", flush=True)
    elif (read_json(out) or {}).get("status") == "done":
        print(f"[{tag}] already done", flush=True)
        return
    rec["status"] = "running"
    tracker = Tracker(args.phase, args.arm, tag, dict(hp, tag=tag, phase=args.phase, git_sha=rec["git_sha"], env=rec["env"]),
                      [args.phase, args.arm, args.feedback, f"c{args.clauses}", f"seed{args.seed}"],
                      enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    lines = RESULTS / args.phase / f"{tag}.stats.txt"   # the reference script's 23-column file
    if start == 0:
        lines.parent.mkdir(parents=True, exist_ok=True)
        lines.write_text("")
    price = args.phase == "price"
    try:
        for epoch in range(start + 1, args.max_epochs + 1):
            model.train()
            sync(dev)
            t0 = time.time()
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                model.update(x["train"][idx], y["train"][idx])
            sync(dev)
            train_s = time.time() - t0
            t0 = time.time()
            full = metrics_due(epoch)
            te = evaluate(model, x["test"], y["test"], with_clauses=full)
            tr = evaluate(model, x["train"], y["train"], with_clauses=False)
            sync(dev)
            eval_s = time.time() - t0
            rec["train_s_total"] += train_s
            rec["elapsed_s"] += train_s + eval_s
            inc = model.include_count.float()
            row = {"epoch": epoch, "test_acc": te["acc"], "train_acc": tr["acc"], "updates": epoch * hp["updates_per_epoch"],
                   "train_s": train_s, "eval_s": eval_s, "train_s_total": rec["train_s_total"],
                   "elapsed_s": rec["elapsed_s"], "mean_clause_len": float(inc.mean()),
                   "frac_empty_clauses": float((inc == 0).float().mean())}
            if full:
                row.update({k: v for k, v in te.items() if k not in ("acc", "_arrays")})
                with open(lines, "a") as fh:
                    fh.write(ref_line(args.seed, row) + "\n")
            rec["curve"].append(row)
            tracker.epoch(row)
            if epoch <= 5 or epoch % 10 == 0:
                print(f"[{tag}] epoch {epoch:4d} test {row['test_acc']:6.2f} train {row['train_acc']:6.2f} | "
                      + (f"coverage {row['coverage']:6.2f} precision p50 {row['precision_p50']:5.1f} "
                         f"mean {row['precision_mean']:5.1f} votes {row['votes_mean']:.1f} | " if full else "")
                      + f"len {row['mean_clause_len']:.1f} | {train_s:.2f}s + {eval_s:.2f}s", flush=True)
            reason = should_stop(rec["curve"], args.max_epochs, rec["elapsed_s"], args.time_cap_h)
            if price and epoch >= PRICE_EPOCHS:
                reason = f"pricing ({PRICE_EPOCHS} epochs)"
            if reason:
                finish(rec, reason)
                if price:                             # the first epoch compiles and captures CUDA graphs
                    steady = rec["curve"][1:] or rec["curve"]
                    rec["s_per_epoch"] = sum(r["train_s"] + r["eval_s"] for r in steady) / len(steady)
                    rec["train_s_per_epoch"] = sum(r["train_s"] for r in steady) / len(steady)
                    rec["peak_gpu_mib"] = torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else 0.0
                MODELS.joinpath(args.phase).mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "hp": hp, "tag": tag, "epoch": epoch},
                           MODELS / args.phase / f"{tag}.pt")
            write_json(rec, out)
            if reason or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
                tmp = ckpt.with_suffix(".tmp")
                torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "gen": gen.get_state(),
                            "cpu_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                            "epoch": epoch, "rec": rec}, tmp)
                tmp.replace(ckpt)
            if reason:
                ckpt.unlink(missing_ok=True)
                m = rec["marks"]
                print(f"[{tag}] done after {epoch} epochs ({reason}): test {rec['test_acc_last']:.2f} %, epochs to "
                      + ", ".join(f"{k} %: {v['epoch'] if v else 'not reached'}" for k, v in m.items()), flush=True)
                break
            if epoch == args.stop_after:
                print(f"[{tag}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
                raise SystemExit(3)
    except BaseException as e:
        if not (isinstance(e, SystemExit) and e.code == 0):
            tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"])})
        raise
    tracker.end({k: rec.get(k) for k in ("status", "stop_reason", "converged", "epochs_run", "test_acc_last",
                                         "test_acc_max", "train_s_per_epoch_mean", "elapsed_s")}
                | {f"epochs_to_{k}": (v or {}).get("epoch") for k, v in (rec.get("marks") or {}).items()})


if __name__ == "__main__":
    main()
