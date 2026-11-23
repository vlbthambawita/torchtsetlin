"""Checks of drop clause in a joint team and of the ``long`` phase (plan PLAN_LONG_DROPOUT.md §5):

  D1  with drop_clause_p = 0 the team update is the update from before drop clause existed
      (a verbatim copy below), bit for bit, on every path
  D2  with p > 0 a member's pass-A votes are its votes without the dropped clauses, and the votes
      its update sees = those + the others' (exact); the update keeps the mask
  D3  JointTeam.update: a dropped clause keeps its automata and its weight; a new mask per update
  D4  a run with drop clause, interrupted and resumed, equals the uninterrupted run
  D5  a run stopped by patience 2 and continued with patience 4 from its kept last checkpoint
      equals an uninterrupted patience-4 run
  D6  --max-epochs 0: these runs have no epoch limit and stop by patience only

Writes results/checks/check_dropout.json. Usage: python code/check_dropout.py [--device cuda:0]
"""

import argparse
import os
import subprocess
import sys
import time

import torch
from check_team import BATCH, IMG, KW, TINY, K, data, n_graphs
from common import CACHE, HERE, RESULTS, new_record, read_json, write_json
from team import JointConv, JointConv3d, JointTeam, drop_kw, learning_votes

import torchtsetlin as tt
from torchtsetlin import functional as F

CL, P_DROP = 40, 0.3


def modes(dev, graphs=True):
    out = [("cpu", "cpu", None, False), ("cpu-chunks", "cpu", TINY, False)]
    if torch.device(dev).type == "cuda":
        out += [("cuda-eager-chunks", dev, TINY, False), ("cuda-events", dev, None, False)]
        out += [("cuda-graphs", dev, None, True)] if graphs else []
    return out


def make(three_d, patch, T, dev, seed, chunk, graphs, p=0.0):
    tt.seed_everything(seed)
    kw = dict(KW, **drop_kw({"drop_clause_p": p}), **({"max_chunk_elements": chunk} if chunk else {}))
    if three_d:
        m = JointConv3d(K, CL, T, 4.0, patch_size=(4, patch, patch), input_shape=(2, 4, IMG, IMG), **kw)
    else:
        m = JointConv(K, CL, T, 4.0, patch_size=patch, input_shape=(8, IMG, IMG), **kw)
    m = m.joint_setup(BATCH).to(dev)
    m.cuda_graphs = graphs
    return m


def team_of(d, chunk, graphs, p=0.0):
    ms = [make(False, 3, 20, d, 3, chunk, graphs, p), make(True, 3, 30, d, 4, chunk, graphs, p),
          make(False, 5, 40, d, 5, chunk, graphs, p)]
    return JointTeam(["a2d", "b3d", "c2d"], ms, "sum")


def xin(xb):
    return [xb.reshape(BATCH, 8, IMG, IMG), xb, xb.reshape(BATCH, 8, IMG, IMG)]


@torch.no_grad()
def old_update(self, xs, y):
    """JointTeam.update as it was before drop clause (joint_wide_tm, 2026-11-16), verbatim."""
    vs = [learning_votes(m, x) for m, x in zip(self.members, xs)]
    B = y.shape[0]
    neg = F.sample_negative_classes(y, self.members[0].n_classes) if self.shared else None
    for m, x, off in zip(self.members, xs, self.offsets(vs)):
        m._joint_offset[:B].copy_(off)
        if neg is not None:
            m._joint_neg[:B].copy_(neg)
        m._joint_on, m._joint_cur = True, 0
        try:
            m.update(x, y)
        finally:
            m._joint_on = False
    return self.combine(vs)


def d1(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        x, y = data(10, 2, d)
        teams, scores = [], []
        for upd in (lambda t, xs, yy: t.update(xs, yy), old_update):
            t = team_of(d, chunk, graphs)
            tt.seed_everything(5)
            scores.append([upd(t, xin(x[i]), y[i]).clone() for i in range(10)])
            teams.append(t)
        fresh = team_of(d, chunk, graphs)
        same = all(torch.equal(a.ta_state, b.ta_state) and torch.equal(a.weights, b.weights)
                   for a, b in zip(teams[0].members, teams[1].members))
        out[label] = {"identical": bool(same and all(torch.equal(a, b) for a, b in zip(*scores))),
                      "no_mask": all(m._clause_active is None for m in teams[0].members),
                      "automata_moved": int(sum((a.ta_state != b.ta_state).sum()
                                                for a, b in zip(teams[0].members, fresh.members))),
                      "graphs_replayed": [n_graphs(m) for m in teams[0].members]}
    ok = all(v["identical"] and v["no_mask"] and v["automata_moved"] > 0 for v in out.values())
    ok = ok and all(min(v["graphs_replayed"]) > 0 for k, v in out.items() if k == "cuda-graphs")
    return ok, out


def votes_without(m, x, mask):
    """Reference for the masked pass A: the member's votes when the dropped clauses weigh nothing."""
    w, act = m.weights.clone(), m._clause_active
    m._clause_active = None
    m.weights[~mask] = 0
    v = learning_votes(m, x)
    m.weights.copy_(w)
    m._clause_active = act
    return v


def d2(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev, graphs=False):       # the library never graphs dropout
        x, y = data(10, 2, d)
        team = team_of(d, chunk, graphs, P_DROP)
        tt.seed_everything(5)
        worst_a = worst_b = 0.0
        kept, differs, dropped = True, 0, []
        for i in range(10):
            xs = xin(x[i])
            team.resample_dropout()
            masks = [m._clause_active.clone() for m in team.members]
            dropped.append(float(sum((~k).sum() for k in masks)) / sum(k.numel() for k in masks))
            vs = [learning_votes(m, xm) for m, xm in zip(team.members, xs)]
            for m, xm, v, k in zip(team.members, xs, vs, masks):
                worst_a = max(worst_a, float((v - votes_without(m, xm, k)).abs().max()))
                m._clause_active = None
                differs += int((learning_votes(m, xm) != v).sum())
                m._clause_active = k.clone()
            offs = team.offsets(vs)
            expect = [(v + o).clamp(-tj, tj) for v, o, tj in zip(vs, offs, team.T_joint)]
            neg = (y[i] + torch.randint(1, K, (BATCH,), device=y.device)) % K
            for m, xm, o, e, k in zip(team.members, xs, offs, expect, masks):
                m._joint_offset[:BATCH].copy_(o)
                m._joint_neg[:BATCH].copy_(neg)
                m._joint_on, m._joint_cur = True, 0
                got = m.update(xm, y[i])
                m._joint_on = False
                worst_b = max(worst_b, float((got - e).abs().max()))
                kept = kept and torch.equal(m._clause_active, k)
        out[label] = {"pass_a_max_abs_diff": worst_a, "pass_b_max_abs_diff": worst_b, "mask_kept_by_update": bool(kept),
                      "votes_changed_by_mask": differs, "mean_dropped_fraction": sum(dropped) / len(dropped)}
    ok = all(v["pass_a_max_abs_diff"] == 0.0 and v["pass_b_max_abs_diff"] == 0.0 and v["mask_kept_by_update"]
             and v["votes_changed_by_mask"] > 0 and abs(v["mean_dropped_fraction"] - P_DROP) < 0.1
             for v in out.values())
    return ok, out


def d3(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev, graphs=False):
        x, y = data(12, 3, d)
        team = team_of(d, chunk, graphs, P_DROP)
        tt.seed_everything(6)
        touched = moved = 0
        new_masks, prev = 0, None
        for i in range(12):
            before = [(m.ta_state.clone(), m.weights.clone()) for m in team.members]
            team.update(xin(x[i]), y[i])
            masks = [m._clause_active.clone() for m in team.members]
            new_masks += int(prev is not None and any(not torch.equal(a, b) for a, b in zip(masks, prev)))
            prev = masks
            for m, (st, w), k in zip(team.members, before, masks):
                rows = (m.ta_state != st).any(dim=1) | (m.weights != w)
                touched += int((rows & ~k).sum())
                moved += int((rows & k).sum())
        out[label] = {"dropped_clauses_touched": touched, "active_clauses_moved": moved, "updates_with_new_mask": new_masks}
    ok = all(v["dropped_clauses_touched"] == 0 and v["active_clauses_moved"] > 0 and v["updates_with_new_mask"] == 11
             for v in out.values())
    return ok, out


# ------------------------------------------------------------------ D4-D6: whole runs (train.py)
BASE = [sys.executable, "train.py", "--phase", "resume_check", "--team", "M", "--batch-size", "32", "--policy", "geo+col",
        "--clauses", "200", "--t-mode", "scaled", "--drop-clause-p", "0.25", "--train-eval", "500", "--subset", "2000",
        "--ckpt-every", "1", "--viz-every", "0", "--no-wandb", "--time-cap-h", "0.5"]
TAG = "M_b32_geocol_s0"


def train(dev, suffix, *extra):
    return subprocess.call(BASE + ["--device", dev, "--tag-suffix", suffix, *extra], cwd=HERE,
                           stdout=subprocess.DEVNULL if os.environ.get("CJT_QUIET") else None)


def rec_of(suffix):
    return read_json(RESULTS / "resume_check" / f"{TAG}{suffix}.json")


def curve(r):
    return [[c["epoch"], c["val_acc"], c["test_acc"], c["train_acc_running"], c["train_eval_acc"]]
            + [c["members"][n]["mean_clause_len"] for n in c["members"]] for c in r["curve"]]


def clean(*suffixes):
    for s in suffixes:
        (RESULTS / "resume_check" / f"{TAG}{s}.json").unlink(missing_ok=True)
        for ext in (".pt", ".final.pt"):
            (CACHE / "ckpt" / "resume_check" / f"{TAG}{s}{ext}").unlink(missing_ok=True)


def d4(dev):
    clean("_dfull", "_dres")
    fixed = ["--min-epochs", "4", "--max-epochs", "4"]
    rc = [train(dev, "_dfull", *fixed), train(dev, "_dres", *fixed, "--stop-after", "2"), train(dev, "_dres", *fixed)]
    a, b = rec_of("_dfull"), rec_of("_dres")
    out = {"exit_codes": rc, "identical": bool(a and b and curve(a) == curve(b) and a["best_epoch"] == b["best_epoch"]),
           "drop_clause_p": a and a["hp"]["drop_clause_p"], "curve": a and curve(a),
           "train_eval_images": a and a["hp"]["data"]["n_train_eval"]}
    return rc == [0, 3, 0] and out["identical"] and out["drop_clause_p"] == 0.25, out


def d5(dev):
    clean("_p4", "_p2c")
    free = ["--min-epochs", "1", "--max-epochs", "0", "--keep-final"]
    rc = [train(dev, "_p4", *free, "--patience", "4"), train(dev, "_p2c", *free, "--patience", "2")]
    first = rec_of("_p2c")
    final = CACHE / "ckpt" / "resume_check" / f"{TAG}_p2c.final.pt"
    kept = final.exists()
    rc.append(train(dev, "_p2c", *free, "--patience", "4", "--continue-final"))
    a, b = rec_of("_p4"), rec_of("_p2c")
    out = {"exit_codes": rc, "final_checkpoint_kept": kept, "stopped_first_at": first and first["epochs_run"],
           "epochs": [a and a["epochs_run"], b and b["epochs_run"]],
           "stop_reasons": [first and first["stop_reason"], a and a["stop_reason"], b and b["stop_reason"]],
           "identical": bool(a and b and curve(a) == curve(b) and a["best_epoch"] == b["best_epoch"]
                             and a["test_acc_at_best"] == b["test_acc_at_best"]),
           "continued": bool(b and first and b["epochs_run"] > first["epochs_run"]),
           "history": b and b.get("early_stopping_history")}
    ok = rc == [0, 0, 0] and kept and out["identical"] and out["continued"]
    return ok, out


def d6(dev):
    """Read from D5's records: no epoch limit was set and patience ended every run."""
    recs = [rec_of("_p4"), rec_of("_p2c")]
    out = {"max_epochs": [r and r["hp"]["early_stopping"]["max_epochs"] for r in recs],
           "stop_reasons": [r and r["stop_reason"] for r in recs],
           "tail": [r and [r["test_acc_tail_mean"], r["test_acc_tail_std"], r["tail_epochs"]] for r in recs]}
    ok = all(r and r["hp"]["early_stopping"]["max_epochs"] is None
             and r["stop_reason"].startswith("no validation improvement") and r["tail_epochs"] == min(4, r["epochs_run"])
             for r in recs)
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="e.g. D1,D2,D3 (the unit checks)")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        torch.cuda.set_device(torch.device(dev))
    only = [s for s in args.only.split(",") if s]
    rec = new_record("checks", device=dev if dev != "cpu" else None, checks={})
    allok = True
    for name, fn in (("D1", d1), ("D2", d2), ("D3", d3), ("D4", d4), ("D5", d5), ("D6", d6)):
        if only and name not in only:
            continue
        t0 = time.time()
        ok, detail = fn(dev)
        rec["checks"][name] = {"ok": bool(ok), "detail": detail}
        allok = allok and ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)
        if not ok:
            print("   ", detail, flush=True)
    rec["ok"] = allok
    write_json(rec, RESULTS / "checks" / "check_dropout.json")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
