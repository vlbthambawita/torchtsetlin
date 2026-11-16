"""One training run of the joint_wide_tm programme on Fashion-MNIST (plan B2-B5, B8).

Arms:  I     one member trained alone                        --member H3
       J     joint team, shared negative class               --team S --rule sum|one
       Jown  joint team, each member draws its own negative  --team S --rule sum|one
       W     width control: one member with the team's clauses (x3) and T x3 (rule sum) or
             T x1 (rule one)                                 --team S --rule sum --member H10

Fixed number of epochs (no early stopping: every arm gets the same budget). Every epoch the raw
class sums of every member on validation and test go to ``.cache/sums/<phase>/<arm>/<tag>/``, so
composites are recomputed, never stored by hand. The model of the best validation epoch is kept
in ``models/<phase>/<arm>/<tag>.pt``. Statistic: test accuracy at the best-validation epoch.

Usage:  python code/train.py --phase pilot --arm J --team S --rule sum --device cuda:1
        python code/train.py --phase pricing --arm I --member H10 --calibrate 300
"""

import argparse
import math
import signal
import sys
import time

import numpy as np
import torch
from common import (
    ARMS,
    BATCH,
    CACHE,
    CLAUSES,
    EPOCHS,
    MEMBERS,
    MODEL_KW,
    MODELS,
    N_CLASSES,
    PHASES,
    RESULTS,
    RULES,
    TEAMS,
    W_FACTOR,
    fmb,
    load_all,
    member_tag,
    new_record,
    team_tag,
    wide_tag,
    write_json,
)
from joint import JointConv, JointTeam
from tracking import Tracker

import torchtsetlin as tt


def spec_of(arm, member=None, team=None, rule=None, seed=0, clauses=CLAUSES):
    """(tag, member configs, rule, shared negative) of a run."""
    if arm == "I":
        return member_tag(member, seed, clauses), {member: dict(MEMBERS[member], clauses=clauses)}, "none", True
    if arm == "W":
        c = dict(MEMBERS[member], clauses=clauses * W_FACTOR)
        if rule == "sum":
            c["T"] = c["T"] * W_FACTOR
        return wide_tag(team, rule, member, seed), {member: c}, "none", True
    cfgs = {n: dict(MEMBERS[n], clauses=clauses) for n in TEAMS[team]}
    return team_tag(team, rule, seed, clauses), cfgs, rule, arm == "J"


def build_team(cfgs, rule, shared, max_batch, dev):
    ms = []
    for c in cfgs.values():
        planes = fmb.ENCODINGS[c["enc"]]
        m = JointConv(N_CLASSES, c["clauses"], c["T"], c["s"], patch_size=c["patch"], stride=1,
                      input_shape=(planes, 28, 28), **MODEL_KW)
        ms.append(m.joint_setup(max_batch).to(dev))
    return JointTeam(list(cfgs), ms, rule, shared_negative=shared)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def peak_mib(dev):
    return torch.cuda.max_memory_reserved(dev) / 2**20 if dev.type == "cuda" else 0.0


def acc(score, y):
    return 100 * float((np.asarray(score).argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--member", choices=sorted(MEMBERS))
    ap.add_argument("--team", choices=sorted(TEAMS))
    ap.add_argument("--rule", choices=RULES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clauses", type=int, default=CLAUSES, help="per class and member")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--subset", type=int, default=0, help="smoke: first N train / N//4 val+test images")
    ap.add_argument("--calibrate", type=int, default=0, help="time this many updates and exit")
    ap.add_argument("--keep-epochs", action="store_true",
                    help="keep every epoch's member states in .cache/epochs (arm I: composite-best model)")
    ap.add_argument("--stop-after", type=int, default=0, help="testing: exit after this epoch's checkpoint")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    need = {"I": ("member",), "J": ("team", "rule"), "Jown": ("team", "rule"), "W": ("team", "rule", "member")}
    for k in need[args.arm]:
        if getattr(args, k) is None:
            raise SystemExit(f"arm {args.arm} needs --{k}")
    if args.epochs is None:
        args.epochs = EPOCHS.get(args.phase, 1)

    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tt.seed_everything(args.seed)
    tag, cfgs, rule, shared = spec_of(args.arm, args.member, args.team, args.rule, args.seed, args.clauses)
    names = list(cfgs)
    encs = [cfgs[n]["enc"] for n in names]
    x, y, _, sha = load_all(encs, dev, args.subset)
    team = build_team(cfgs, rule, shared, args.batch_size, dev)
    xs = {k: [x[e][k] for e in encs] for k in ("train", "val", "test")}
    ytr = torch.from_numpy(y["train"]).to(dev)
    n_tr = len(y["train"])
    gen = torch.Generator(device="cpu").manual_seed(1000 + args.seed)      # data order = f(seed)
    hp = dict(arm=args.arm, team=args.team, rule=rule, rule_arg=args.rule, shared_negative=shared,
              members={n: dict(c, planes=fmb.ENCODINGS[c["enc"]], clauses_total=int(m.n_clauses_total),
                               n_features=int(m.n_features), n_literals=int(m.n_literals))
                       for (n, c), m in zip(cfgs.items(), team.members)},
              T_joint=dict(zip(names, team.T_joint)), combine_weights=dict(zip(names, team.weights)),
              model_kw=MODEL_KW, batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
              commits_per_epoch=math.ceil(n_tr / args.batch_size),
              data=dict(sha256=sha, n_train=n_tr, n_val=len(y["val"]), n_test=len(y["test"]),
                        subset=args.subset, split="fmnist_bitplanes/results/split.json"))
    print(f"[{args.arm}/{tag}] rule={rule} shared_neg={shared} members="
          + ", ".join(f"{n}({c['enc']},{c['patch']}x{c['patch']},{c['clauses']}/class,T={c['T']},s={c['s']})"
                      for n, c in cfgs.items()) + f" B={args.batch_size} {args.device}", flush=True)

    if args.calibrate:
        calibrate(args, team, hp, tag, xs, ytr, gen, dev)
        return

    out = RESULTS / args.phase / args.arm / f"{tag}.json"
    ckpt = CACHE / "ckpt" / args.phase / args.arm / f"{tag}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rec = new_record("run", device=dev, phase=args.phase, arm=args.arm, tag=tag, hp=hp, curve=[],
                     status="running", best={"val_correct": -1, "epoch": 0}, elapsed_s=0.0)
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
    elif out.exists() and fmb.read_json(out).get("status") == "done":
        print(f"[{tag}] already done", flush=True)
        return
    rec["status"] = "running"
    tracker = Tracker(args.phase, args.arm, tag, dict(hp, tag=tag, phase=args.phase, git_sha=rec["git_sha"],
                                                      env=rec["env"]),
                      [args.phase, args.arm, f"seed{args.seed}", rec["env"].get("gpu", args.device)]
                      + ([f"team{args.team}", f"rule-{args.rule}"] if args.team else []) + names,
                      enabled=not args.no_wandb)
    rec["wandb_url"] = tracker.url or rec.get("wandb_url")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        train(args, team, rec, xs, y, ytr, gen, dev, out, ckpt, start, tracker)
    except BaseException as e:
        if isinstance(e, SystemExit) and e.code == 0:
            raise
        tracker.end({"status": "interrupted", "epochs_run": len(rec["curve"]), "error": f"{type(e).__name__}: {e}"})
        raise
    tracker.end({k: rec.get(k) for k in ("status", "test_acc_at_best", "best_val_acc", "best_epoch", "epochs_run",
                                         "test_acc_last", "train_s_per_epoch_mean", "elapsed_s", "peak_gpu_mib")})


def member_states(team):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in team.members]


def train(args, team, rec, xs, y, ytr, gen, dev, out, ckpt, start, tracker):
    n_tr, bs, names = len(y["train"]), args.batch_size, team.names
    sums_dir = CACHE / "sums" / args.phase / args.arm / rec["tag"]
    sums_dir.mkdir(parents=True, exist_ok=True)
    best = rec["best"]
    for epoch in range(start + 1, args.epochs + 1):
        team.train()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time()
        perm = torch.randperm(n_tr, generator=gen).to(dev)
        correct = torch.zeros((), dtype=torch.long, device=dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            score = team.update([xm[idx] for xm in xs["train"]], ytr[idx])   # pre-update team score
            correct += (score.argmax(1) == ytr[idx]).sum()
        sync(dev)
        train_s = time.time() - t0
        fb = [m.read_feedback() for m in team.members]

        t0 = time.time()
        sums = {k: [s.float().cpu().numpy() for s in team.member_sums(xs[k], args.eval_batch)]
                for k in ("val", "test")}
        sync(dev)
        eval_s = time.time() - t0
        np.savez_compressed(sums_dir / f"e{epoch:03d}.npz",
                            **{f"{n}_{k}": sums[k][i] for k in sums for i, n in enumerate(names)})
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
                          "vote_share": float(np.abs(team.weights[i] * sums["test"][i]).sum()
                                              / max(1e-12, sum(np.abs(w * s).sum() for w, s in
                                                               zip(team.weights, sums["test"]))))}
        improved = val_correct > best["val_correct"]               # a tie keeps the earlier epoch
        if improved:
            best.update(val_correct=val_correct, epoch=epoch)
        rec["elapsed_s"] += train_s + eval_s
        row = {"epoch": epoch, "val_acc": 100 * val_correct / len(y["val"]),
               "best_val_acc": 100 * best["val_correct"] / len(y["val"]),
               "test_acc": 100 * float((te_pred == y["test"]).mean()),
               "test_acc_per_class": [100 * float((te_pred[y["test"] == k] == k).mean()) for k in range(10)],
               "train_acc_running": 100 * int(correct) / n_tr, "members": members, "train_s": train_s,
               "eval_s": eval_s, "examples_per_s": n_tr / train_s, "peak_gpu_mib": peak_mib(dev),
               "elapsed_s": rec["elapsed_s"]}
        rec["curve"].append(row)
        if improved:
            rec["best_model"] = str(save_model(MODELS / args.phase / args.arm / f"{rec['tag']}.pt", team, rec, row))
        if args.keep_epochs:
            save_model(CACHE / "epochs" / args.phase / args.arm / rec["tag"] / f"e{epoch:03d}.pt", team, rec, row)
        tracker.epoch(row)
        print(f"[{rec['tag']}] epoch {epoch:3d} val {row['val_acc']:6.2f} test {row['test_acc']:6.2f} train "
              f"{row['train_acc_running']:6.2f} | best {row['best_val_acc']:6.2f}@{best['epoch']} | "
              f"{train_s:.1f}s + {eval_s:.1f}s | "
              + " ".join(f"{n}:{m['test_acc']:.2f}/len{m['mean_clause_len']:.1f}/fb{m['type_i_per_example'] + m['type_ii_per_example']:.0f}"
                         for n, m in members.items()), flush=True)
        done = epoch >= args.epochs
        if done:
            finish(rec, best)
        write_json(rec, out)
        if done or epoch % args.ckpt_every == 0 or epoch == args.stop_after:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"members": member_states(team), "gen": gen.get_state(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None,
                        "epoch": epoch, "rec": rec}, tmp)
            tmp.replace(ckpt)
        if done:
            ckpt.unlink(missing_ok=True)
            print(f"[{rec['tag']}] done after {epoch} epochs: best val {rec['best_val_acc']:.2f} @ "
                  f"{rec['best_epoch']}, test there {rec['test_acc_at_best']:.2f}", flush=True)
            return
        if epoch == args.stop_after:
            print(f"[{rec['tag']}] --stop-after {epoch}: exiting with a checkpoint", flush=True)
            raise SystemExit(3)


def save_model(path, team, rec, row):
    """Self-contained team file: hp holds what build_team needs; .tmp then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"members": dict(zip(team.names, member_states(team))), "hp": rec["hp"], "epoch": row["epoch"],
                "val_acc": row["val_acc"], "test_acc": row["test_acc"],
                "member_test_acc": {n: m["test_acc"] for n, m in row["members"].items()},
                "tag": rec["tag"], "phase": rec["phase"], "arm": rec["arm"], "git_sha": rec["git_sha"],
                "env": rec["env"], "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}, tmp)
    tmp.replace(path)
    return path


def finish(rec, best):
    c = rec["curve"]
    b = c[best["epoch"] - 1]
    assert b["epoch"] == best["epoch"]
    rec.update(status="done", best_epoch=best["epoch"], best_val_acc=b["val_acc"], test_acc_at_best=b["test_acc"],
               test_acc_per_class_at_best=b["test_acc_per_class"], members_at_best=b["members"],
               epochs_run=len(c), gpu_s_to_best=b["elapsed_s"], test_acc_last=c[-1]["test_acc"],
               train_s_per_epoch_mean=float(np.mean([r["train_s"] for r in c])),
               eval_s_per_epoch_mean=float(np.mean([r["eval_s"] for r in c])),
               peak_gpu_mib=max(r["peak_gpu_mib"] for r in c), finished=time.strftime("%Y-%m-%dT%H:%M:%S"))


def calibrate(args, team, hp, tag, xs, ytr, gen, dev):
    """T7 pricing: ms per update after a warm-up (incl. CUDA-graph capture) -> s/epoch, peak memory."""
    n_tr, bs = ytr.shape[0], args.batch_size
    perm = torch.randperm(n_tr, generator=gen).to(dev)
    n_batches = max(1, min(args.calibrate, n_tr // bs - 10))
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def step(i):
        idx = perm[i * bs:(i + 1) * bs]
        team.update([xm[idx] for xm in xs["train"]], ytr[idx])

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
    rec = new_record("calibration", device=dev, phase=args.phase, arm=args.arm, tag=tag, hp=hp, warmup_s=warm,
                     n_batches=n_batches, ms_per_update=per * 1000,
                     train_s_per_epoch=per * math.ceil(50000 / bs),
                     eval_s_per_epoch=eval_s * 20000 / xs["val"][0].shape[0], peak_gpu_mib=peak_mib(dev))
    print(f"[{tag}] calibrate: {per * 1000:.2f} ms/update -> train {rec['train_s_per_epoch']:.1f} s/epoch, "
          f"eval {rec['eval_s_per_epoch']:.1f} s/epoch, peak {rec['peak_gpu_mib']:.0f} MiB", flush=True)
    write_json(rec, RESULTS / args.phase / args.arm / f"{tag}.json")


if __name__ == "__main__":
    main()
