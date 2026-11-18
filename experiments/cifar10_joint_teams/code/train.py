"""One run: a jointly trained team on augmented CIFAR-10 at one batch size (plan §5-§7, §10).

Every epoch: an augmented copy of the 45 000 training images (the same for every batch size, see
augment.py), joint updates in mini-batches, then the members' class sums on the un-augmented
validation and test sets. Early stopping watches validation accuracy; the team of the best
validation epoch is kept in ``models/<phase>/<tag>.pt``. Every ``--viz-every`` epochs the
activation-map panel of the fixed validation images goes to ``figures/activation/`` and W&B.
Statistic: test accuracy at the best-validation epoch.

Usage:  python code/train.py --phase grid --team M --batch-size 32 --policy full
        python code/train.py --phase pricing --team M --batch-size 32 --calibrate 300
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch
from augment import CPU_ENCS, POLICIES, EpochSource, telemetry_off
from common import (
    CACHE,
    CLAUSES,
    EARLY,
    FIGURES,
    MODEL_KW,
    MODELS,
    PHASES,
    PILOT,
    RESULTS,
    RULE,
    SEED,
    TEAMS,
    VIZ_EVERY,
    cbc,
    load_cifar,
    new_record,
    panel_indices,
    read_json,
    tag_of,
    write_json,
)
from encoders import encode_fixed, gpu_encode
from team import build_team, member_cfgs, member_states, save_team
from tracking import Tracker
from viz import panel

import torchtsetlin as tt


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def peak_mib(dev):
    return torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else 0.0


def acc(score, y):
    return 100 * float((np.asarray(score).argmax(1) == y).mean())


class TrainData:
    """The augmented training set of one epoch on the device, encoded per mini-batch."""

    def __init__(self, x_train, encs, policy, seed, dev):
        self.encs, self.dev = encs, dev
        self.src = EpochSource(x_train, policy, seed, [e for e in set(encs) if e in CPU_ENCS])
        self.raw, self.cpu = None, {}

    def load(self, epoch, last=False):
        t0 = time.time()
        img, enc = self.src.get(epoch, last)
        self.raw = torch.from_numpy(img).to(self.dev)
        self.cpu = {e: torch.from_numpy(a).to(self.dev) for e, a in enc.items()}
        return time.time() - t0

    def batch(self, idx):
        raw, done, out = self.raw[idx], {}, []
        for e in self.encs:
            if e not in done:
                done[e] = self.cpu[e][idx] if e in CPU_ENCS else gpu_encode(e, raw)
            out.append(done[e])
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=PHASES + ["resume_check"])
    ap.add_argument("--team", required=True, choices=sorted(TEAMS))
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--policy", default="full", choices=POLICIES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--clauses", type=int, default=CLAUSES, help="per class and member")
    ap.add_argument("--t-mode", default="fixed", choices=["fixed", "scaled"],
                    help="scaled: every member's T in proportion to --clauses")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--min-epochs", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--time-cap-h", type=float, default=EARLY["time_cap_h"])
    ap.add_argument("--viz-every", type=int, default=VIZ_EVERY, help="0 = no activation maps")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="time this many updates and exit")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    fixed = PILOT["epochs"] if args.phase == "pilot" else None      # pilot: same budget for every policy
    args.min_epochs = args.min_epochs or fixed or EARLY["min_epochs"]
    args.max_epochs = args.max_epochs or fixed or EARLY["max_epochs"]
    args.patience = args.patience or (10**6 if fixed else EARLY["patience"])
    if not telemetry_off():
        raise SystemExit("augmentation telemetry is not switched off")

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)
    tag = tag_of(args.team, args.batch_size, args.policy, args.seed, args.tag_suffix)
    cfgs = member_cfgs(args.team, args.clauses, args.t_mode)
    names = list(cfgs)
    encs = [cfgs[n]["enc"] for n in names]
    data = load_cifar(args.subset)
    y = {k: v[1] for k, v in data.items()}
    n_tr = len(y["train"])
    team = build_team(cfgs, args.batch_size, dev)
    fixed_x = {k: {e: encode_fixed(e, data[k][0], dev) for e in set(encs)} for k in ("val", "test")}
    xs = {k: [fixed_x[k][e] for e in encs] for k in fixed_x}
    ytr = torch.from_numpy(y["train"]).to(dev)
    pidx = panel_indices(y["val"])
    pan = dict(raw=data["val"][0][pidx], y=y["val"][pidx], xs=[x[pidx] for x in xs["val"]])
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)      # data order = f(seed)
    train_data = TrainData(data["train"][0], encs, args.policy, args.seed, dev)
    hp = dict(team=args.team, rule=RULE, shared_negative=True, clauses=args.clauses, t_mode=args.t_mode,
              members={n: dict(c, clauses_total=int(m.n_clauses_total), n_features=int(m.n_features),
                               n_literals=int(m.n_literals), max_chunk_elements=int(m.max_chunk_elements))
                       for (n, c), m in zip(cfgs.items(), team.members)},
              T_joint=dict(zip(names, team.T_joint)), model_kw=MODEL_KW, batch_size=args.batch_size,
              augmentation=dict(policy=args.policy, per_image_seed="SeedSequence([seed, epoch, index])",
                                workers=train_data.src.workers),
              early_stopping=dict(monitor="val_acc", patience=args.patience, min_epochs=args.min_epochs,
                                  max_epochs=args.max_epochs, time_cap_h=args.time_cap_h),
              seed=args.seed, updates_per_epoch=math.ceil(n_tr / args.batch_size), viz_every=args.viz_every,
              data=dict(sha256=cbc.CIFAR_SHA256, n_train=n_tr, n_val=len(y["val"]), n_test=len(y["test"]),
                        subset=args.subset, split="cifar10_batch_clauses/results/split.json", panel=pidx))
    print(f"[{tag}] team {args.team} rule={RULE} members="
          + ", ".join(f"{n}({c['enc']},{c['conv']},{c['patch']}x{c['patch']},{c['clauses']}/class,T={c['T']},s={c['s']})"
                      for n, c in cfgs.items())
          + f" B={args.batch_size} aug={args.policy} ({train_data.src.workers} workers) {args.device}", flush=True)
    try:
        if args.calibrate:
            calibrate(args, team, hp, tag, train_data, xs, ytr, gen, dev)
            return
        run(args, team, hp, tag, train_data, xs, y, ytr, pan, gen, dev)
    finally:
        train_data.src.close()


def run(args, team, hp, tag, train_data, xs, y, ytr, pan, gen, dev):
    out = RESULTS / args.phase / f"{tag}.json"
    ckpt = CACHE / "ckpt" / args.phase / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", device=dev, phase=args.phase, tag=tag, hp=hp, curve=[], status="running",
                     best={"val_correct": -1, "epoch": 0}, elapsed_s=0.0)
    start = 0
    if ckpt.exists():                              # resume an interrupted run
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        for m, s in zip(team.members, st["members"]):
            m.load_state_dict(s)
        gen.set_state(st["gen"])
        torch.set_rng_state(st["cpu_rng"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state(st["cuda_rng"], dev)
        rec, start = st["rec"], st["epoch"]
        rec["curve"] = [c for c in rec["curve"] if c["epoch"] <= start]
        rec.setdefault("resumed_at", []).append({"epoch": start, "argv": sys.argv,
                                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        print(f"[{tag}] resumed after epoch {start}", flush=True)
    elif (read_json(out) or {}).get("status") == "done":
        print(f"[{tag}] already done", flush=True)
        return
    rec["status"] = "running"
    tracker = Tracker(args.phase, args.team, tag,
                      dict(hp, tag=tag, phase=args.phase, git_sha=rec["git_sha"], env=rec["env"]),
                      [args.phase, f"team{args.team}", f"b{args.batch_size}", f"c{args.clauses}", f"aug-{args.policy}",
                       f"seed{args.seed}", rec["env"].get("gpu", args.device)] + team.names,
                      enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        loop(args, team, rec, train_data, xs, y, ytr, pan, gen, dev, out, ckpt, start, tracker)
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]), "error": f"{type(e).__name__}: {e}"})
        raise
    tracker.end({k: rec.get(k) for k in ("status", "stop_reason", "test_acc_at_best", "best_val_acc", "best_epoch",
                                         "epochs_run", "test_acc_last", "train_s_per_epoch_mean", "elapsed_s",
                                         "peak_gpu_mib")})


def loop(args, team, rec, train_data, xs, y, ytr, pan, gen, dev, out, ckpt, start, tracker):
    n_tr, bs, names, tag = len(y["train"]), args.batch_size, team.names, rec["tag"]
    best = rec["best"]
    for epoch in range(start + 1, args.max_epochs + 1):
        aug_wait = train_data.load(epoch, last=epoch == args.max_epochs)
        team.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        sync(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=gen).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            score = team.update(train_data.batch(idx), ytr[idx])            # pre-update team score
            correct += (score.argmax(1) == ytr[idx]).sum()
        sync(dev)
        train_s = time.time() - t0
        fb = [m.read_feedback() for m in team.members]

        t0 = time.time()
        sums = {k: [s.float().cpu().numpy() for s in team.member_sums(xs[k], args.eval_batch)]
                for k in ("val", "test")}
        sync(dev)
        eval_s = time.time() - t0
        score = {k: team.combine(sums[k]) for k in sums}
        val_correct = int((score["val"].argmax(1) == y["val"]).sum())
        te_pred = score["test"].argmax(1)
        members = {}
        for i, (n, m) in enumerate(zip(names, team.members)):
            inc = m.include_count.float()
            members[n] = {"val_acc": acc(sums["val"][i], y["val"]), "test_acc": acc(sums["test"][i], y["test"]),
                          "mean_clause_len": float(inc.mean()), "frac_empty_clauses": float((inc == 0).float().mean()),
                          "mean_weight": float(m.weights.float().mean()),
                          "type_i_per_example": fb[i]["type_i"] / max(1.0, fb[i]["examples"]),
                          "type_ii_per_example": fb[i]["type_ii"] / max(1.0, fb[i]["examples"]),
                          # share of the team's absolute score that this member supplies (test set)
                          "vote_share": float(np.abs(sums["test"][i]).sum()
                                              / max(1e-12, sum(np.abs(s).sum() for s in sums["test"])))}
        improved = val_correct > best["val_correct"]               # a tie keeps the earlier epoch
        if improved:
            best.update(val_correct=val_correct, epoch=epoch)
        rec["elapsed_s"] += train_s + eval_s
        row = {"epoch": epoch, "val_acc": 100 * val_correct / len(y["val"]),
               "best_val_acc": 100 * best["val_correct"] / len(y["val"]),
               "epochs_since_improvement": epoch - best["epoch"],
               "test_acc": 100 * float((te_pred == y["test"]).mean()),
               "test_acc_per_class": [100 * float((te_pred[y["test"] == k] == k).mean()) for k in range(10)],
               "train_acc_running": 100 * int(correct) / n_tr, "members": members, "train_s": train_s,
               "eval_s": eval_s, "aug_wait_s": aug_wait, "examples_per_s": n_tr / train_s,
               "updates": epoch * math.ceil(n_tr / bs), "peak_gpu_mib": peak_mib(dev),
               "elapsed_s": rec["elapsed_s"], "gpu_hours": rec["elapsed_s"] / 3600}
        if improved:
            rec["best_model"] = str(save_team(MODELS / args.phase / f"{tag}.pt", team, rec, row))
        if epoch - best["epoch"] >= args.patience and epoch >= args.min_epochs:
            reason = f"no validation improvement for {args.patience} epochs"
        elif epoch >= args.max_epochs:
            reason = f"max epochs ({args.max_epochs})"
        elif rec["elapsed_s"] >= args.time_cap_h * 3600:
            reason = f"time cap ({args.time_cap_h:g} h)"
        else:
            reason = None
        images = None
        if args.viz_every and (epoch % args.viz_every == 0 or reason):   # not timed: explanation only
            png = FIGURES / "activation" / args.phase / tag / f"e{epoch:03d}.png"
            v = panel(team, pan["xs"], pan["raw"], pan["y"],
                      f"{tag}  epoch {epoch}  val {row['val_acc']:.2f} %  test {row['test_acc']:.2f} %", png)
            row.update(map_correlation=v["map_correlation"], panel_acc=v["panel_acc"])
            images = {"viz/panel": (png, f"{tag} epoch {epoch}")}
        rec["curve"].append(row)
        tracker.epoch(row, images)
        print(f"[{tag}] epoch {epoch:3d} val {row['val_acc']:6.2f} test {row['test_acc']:6.2f} train "
              f"{row['train_acc_running']:6.2f} | best {row['best_val_acc']:6.2f}@{best['epoch']} | "
              f"{train_s:.1f}s + {eval_s:.1f}s (aug wait {aug_wait:.1f}s) | "
              + " ".join(f"{n}:{m['test_acc']:.2f}/len{m['mean_clause_len']:.1f}" for n, m in members.items()),
              flush=True)
        if reason:
            finish(rec, best, reason)
        write_json(rec, out)
        if reason or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"members": member_states(team), "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                        "epoch": epoch, "rec": rec}, tmp)
            tmp.replace(ckpt)
        if reason:
            ckpt.unlink(missing_ok=True)
            print(f"[{tag}] done after {epoch} epochs ({reason}): best val {rec['best_val_acc']:.2f} @ "
                  f"{rec['best_epoch']}, test there {rec['test_acc_at_best']:.2f}", flush=True)
            return
        if epoch == args.stop_after:
            print(f"[{tag}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


def finish(rec, best, reason):
    c = rec["curve"]
    b = c[best["epoch"] - 1]
    assert b["epoch"] == best["epoch"]
    rec.update(status="done", stop_reason=reason, best_epoch=best["epoch"], best_val_acc=b["val_acc"],
               test_acc_at_best=b["test_acc"], test_acc_per_class_at_best=b["test_acc_per_class"],
               members_at_best=b["members"], epochs_run=len(c), gpu_s_to_best=b["elapsed_s"],
               updates_to_best=b["updates"], test_acc_last=c[-1]["test_acc"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               eval_s_per_epoch_mean=float(np.mean([r["eval_s"] for r in c])),
               aug_wait_s_per_epoch_mean=float(np.mean([r["aug_wait_s"] for r in c])),
               peak_gpu_mib=max(r["peak_gpu_mib"] for r in c), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))


def calibrate(args, team, hp, tag, train_data, xs, ytr, gen, dev):
    """Pricing: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch, peak memory,
    and the time the workers need for one augmented epoch."""
    n_tr, bs = ytr.shape[0], args.batch_size
    t0 = time.time()
    train_data.load(1, last=True)
    aug_s = time.time() - t0
    perm = torch.randperm(n_tr, generator=gen).to(dev)
    n_batches = max(1, min(args.calibrate, n_tr // bs - 10))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def step(i):
        idx = perm[i * bs:(i + 1) * bs]
        team.update(train_data.batch(idx), ytr[idx])

    team.train()
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
    t0 = time.time()
    team.member_sums(xs["val"], args.eval_batch)
    sync(dev)
    eval_s = time.time() - t0
    full = 45000 / n_tr
    rec = new_record("calibration", device=dev, phase=args.phase, tag=tag, hp=hp, warmup_s=warm,
                     n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=per * math.ceil(45000 / bs),
                     eval_s_per_epoch=eval_s * 15000 / xs["val"][0].shape[0],
                     aug_s_per_epoch=aug_s * full, peak_gpu_mib=peak_mib(dev), status="done")
    print(f"[{tag}] calibrate: {per * 1000:.2f} ms/update -> train {rec['train_s_per_epoch']:.1f} s/epoch, "
          f"eval {rec['eval_s_per_epoch']:.1f} s/epoch, augmentation {rec['aug_s_per_epoch']:.1f} s/epoch "
          f"({train_data.src.workers} workers), peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / args.phase / f"{tag}.json")


if __name__ == "__main__":
    main()
