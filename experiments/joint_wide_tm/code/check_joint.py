"""Phase-0 correctness checks of the joint update (plan B6, T0-T6 and T8; T7 is the pricing arm).

    T0  one member, zero offset, own negative class: bit-identical to the library model
    T1  the score a member learns from equals the externally computed joint score (all paths)
    T2  shared mode: every member uses the same negative class, never the target
    T3  CUDA graphs: a replayed update reads the *new* offset
    T4  pass A equals the votes the member recomputes inside update (checked with T1)
    T5  toy: a joint team of a 2x2 and a 3x3 member is at least as good as the better member
    T6  toy: `sum` team of identical members == one model with M*n clauses and M*T (in distribution)
    T8  state_dict round trip of a joint team

Writes results/checks/check_joint.json; exit code 1 if any check fails.

Usage: python code/check_joint.py --device cuda:1 [--only T0 T3]
"""

import argparse
import sys
import time

import numpy as np
import torch
from common import RESULTS, new_record, write_json
from joint import JointConv, JointTeam, learning_votes

import torchtsetlin as tt

K, CL, IMG, PLANES, BATCH = 10, 40, 12, 2, 16
KW = dict(weighted=True, max_included_literals=8, boost_true_positive=True, position_encoding=True,
          feedback_mode="batch")
TINY = 1 << 15          # max_chunk_elements that splits a batch of 16 into several chunks


def data(n_batches, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_batches, BATCH, PLANES, IMG, IMG, generator=g) < 0.4
    y = torch.randint(0, K, (n_batches, BATCH), generator=g)
    return x.to(dev), y.to(dev)


def member(cls, patch, T, dev, seed, chunk=None, graphs=True, joint=True):
    tt.seed_everything(seed)
    kw = dict(KW, **({"max_chunk_elements": chunk} if chunk else {}))
    m = cls(K, CL, T, 4.0, patch_size=patch, input_shape=(PLANES, IMG, IMG), **kw)
    if joint:
        m.joint_setup(BATCH)
    m = m.to(dev)
    m.cuda_graphs = graphs
    return m


def modes(dev):
    """(label, device, max_chunk_elements, cuda_graphs)"""
    out = [("cpu", "cpu", None, False), ("cpu-chunks", "cpu", TINY, False)]
    if torch.device(dev).type == "cuda":
        out += [("cuda-eager-chunks", dev, TINY, False), ("cuda-events", dev, None, False),
                ("cuda-graphs", dev, None, True)]
    return out


def n_graphs(m):
    return sum(isinstance(v, tuple) for v in m.__dict__.get("_graphs", {}).values())


# ------------------------------------------------------------------------------------------ T0
def t0(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        x, y = data(20, dev=d)
        for variant in ("joint-own", "none"):
            plain = member(tt.ConvTsetlinMachine, 4, 60, d, 7, chunk, graphs, joint=False)
            jm = member(JointConv, 4, 60, d, 7, chunk, graphs)
            team = JointTeam(["m"], [jm], "sum" if variant == "joint-own" else "none", shared_negative=False)
            tt.seed_everything(11)
            for i in range(20):
                plain.update(x[i], y[i])
            tt.seed_everything(11)
            for i in range(20):
                team.update([x[i]], y[i])
            same = bool(torch.equal(plain.ta_state, jm.ta_state) and torch.equal(plain.weights, jm.weights))
            moved = int((plain.ta_state != member(tt.ConvTsetlinMachine, 4, 60, d, 7, chunk, graphs,
                                                  joint=False).ta_state).sum())
            out[f"{label}/{variant}"] = {"identical": same, "automata_moved": moved,
                                         "graphs_replayed": n_graphs(jm)}
    ok = all(v["identical"] and v["automata_moved"] > 0 for v in out.values())
    ok = ok and all(v["graphs_replayed"] > 0 for k, v in out.items() if k.startswith("cuda-graphs"))
    return ok, out


# ------------------------------------------------------------------------------------- T1 + T4
def team3(d, chunk, graphs, rule, shared=True, seed=3):
    ms = [member(JointConv, p, T, d, seed + i, chunk, graphs) for i, (p, T) in enumerate(((2, 4), (3, 6), (5, 12)))]
    return JointTeam(["a", "b", "c"], ms, rule, shared_negative=shared)


def t1(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        for rule in ("sum", "one"):
            x, y = data(10, seed=1, dev=d)
            team = team3(d, chunk, graphs, rule)
            tt.seed_everything(5)
            worst, n_clamped = 0.0, 0
            for i in range(10):
                vs = [learning_votes(m, x[i]) for m in team.members]
                offs = team.offsets(vs)
                expect = [(v + o).clamp(-tj, tj) for v, o, tj in zip(vs, offs, team.T_joint)]
                got = []
                neg = torch.randint(1, K, (BATCH,), device=y.device)
                neg = (y[i] + neg) % K
                for m, o in zip(team.members, offs):
                    m._joint_offset[:BATCH].copy_(o)
                    m._joint_neg[:BATCH].copy_(neg)
                    m._joint_on, m._joint_cur = True, 0
                    got.append(m.update(x[i], y[i]))
                    m._joint_on = False
                    assert m._joint_cur == BATCH or graphs, "cursor did not cover the batch"
                worst = max(worst, max(float((g - e).abs().max()) for g, e in zip(got, expect)))
                n_clamped += sum(int((e.abs() == tj).sum()) for e, tj in zip(expect, team.T_joint))
            out[f"{label}/{rule}"] = {"max_abs_diff": worst, "clamped_entries": n_clamped,
                                      "graphs_replayed": [n_graphs(m) for m in team.members]}
    ok = all(v["max_abs_diff"] == 0.0 for v in out.values())
    return ok, out


# ------------------------------------------------------------------------------------------ T2
def t2(dev):
    out = {}
    for label, d, chunk, graphs in modes(dev):
        if graphs:
            continue                                  # the trace is Python: eager paths only
        for shared in (True, False):
            x, y = data(6, seed=2, dev=d)
            team = team3(d, chunk, False, "sum", shared=shared)
            for m in team.members:
                m._joint_trace = []
            tt.seed_everything(9)
            for i in range(6):
                team.update([x[i]] * 3, y[i])
            negs = [torch.cat([t[1] for t in m._joint_trace]) for m in team.members]
            yy = y.reshape(-1)
            out[f"{label}/{'shared' if shared else 'own'}"] = {
                "all_equal": bool(all(torch.equal(negs[0], n) for n in negs[1:])),
                "never_target": bool(all((n != yy).all() for n in negs)),
                "n": int(negs[0].numel())}
    ok = all(v["never_target"] and v["all_equal"] == k.endswith("/shared") for k, v in out.items())
    return ok, out


# ------------------------------------------------------------------------------------------ T3
def t3(dev):
    if torch.device(dev).type != "cuda":
        return True, {"skipped": "no CUDA device"}
    x, y = data(8, seed=4, dev=dev)
    team = team3(dev, None, True, "sum")
    tt.seed_everything(13)
    for i in range(6):                                # third use captures, then replays
        team.update([x[i]] * 3, y[i])
    out = {}
    for name, m, tj in zip(team.names, team.members, team.T_joint):
        g0, keys0 = n_graphs(m), len(m.__dict__["_graphs"])
        tgt = torch.zeros(BATCH, K, dtype=torch.bool, device=dev).scatter_(1, y[7].view(-1, 1), True)
        neg = (y[7] + 1) % K
        res = {}
        for kind, sign in (("saturated", 1.0), ("opposed", -1.0)):
            v = learning_votes(m, x[7])
            off = torch.where(tgt, sign * 10 * tj, -sign * 10 * tj).float()
            state, w = m.ta_state.clone(), m.weights.clone()
            m.read_feedback()
            m._joint_offset[:BATCH].copy_(off)
            m._joint_neg[:BATCH].copy_(neg)
            m._joint_on, m._joint_cur = True, 0
            got = m.update(x[7], y[7])
            m._joint_on = False
            fb = m.read_feedback()
            res[kind] = {"votes_match": bool(torch.equal(got, (v + off).clamp(-tj, tj))),
                         "selections": fb["type_i"] + fb["type_ii"],
                         "state_changed": bool(not torch.equal(state, m.ta_state) or not torch.equal(w, m.weights))}
        res["replayed"] = bool(n_graphs(m) == g0 > 0 and len(m.__dict__["_graphs"]) == keys0)
        res["expected_opposed"] = BATCH * 2 * CL       # every clause of the target and the negative class
        out[name] = res
    ok = all(r["replayed"] and r["saturated"]["votes_match"] and r["opposed"]["votes_match"]
             and r["saturated"]["selections"] == 0 and not r["saturated"]["state_changed"]
             and r["opposed"]["selections"] == r["expected_opposed"] and r["opposed"]["state_changed"]
             for r in out.values())
    return ok, out


# ------------------------------------------------------------------------------------- T5, T6
def toy_member(patch, clauses, T, dev, max_batch, size=4):
    m = JointConv(2, clauses, T, 3.0, patch_size=patch, input_shape=(1, size, size), weighted=True,
                  boost_true_positive=True, position_encoding=False, feedback_mode="batch")
    return m.joint_setup(max_batch).to(dev)


def toy_run(spec, rule, seed, dev, n_train, noise, epochs, bs=32):
    """spec: list of (patch, clauses, T). Returns (test accuracy, mean included literals per
    clause, feedback selections per training example summed over the members, mean test score of
    the true class)."""
    xtr, ytr = tt.data.make_2d_noisy_xor(n_train, size=4, noise=noise, seed=100 + seed)
    xte, yte = tt.data.make_2d_noisy_xor(2000, size=4, noise=0.0, seed=999)
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)
    tt.seed_everything(seed)
    team = JointTeam([f"m{i}" for i in range(len(spec))],
                     [toy_member(p, c, T, dev, bs) for p, c, T in spec], rule)
    team.train()
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n_train, generator=g).to(dev)
        for i in range(0, n_train - bs + 1, bs):
            idx = perm[i:i + bs]
            team.update([xtr[idx]] * len(spec), ytr[idx])
    fbs = [m.read_feedback() for m in team.members]
    fb = sum(f["type_i"] + f["type_ii"] for f in fbs) / fbs[0]["examples"]
    score = team.combine(team.member_sums([xte] * len(spec)))
    acc = float((score.argmax(1) == yte).float().mean())
    length = float(np.mean([float(m.include_count.float().mean()) for m in team.members]))
    margin = float(score.gather(1, yte.view(-1, 1)).mean())      # mean score of the true class
    return acc, length, fb, margin


STATS = ("acc", "len", "fb", "margin")


def stats(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def t5(dev, seeds=8):
    """Paired over seeds (the toy itself is seed-sensitive: a seed's data fails every arm alike):
    a joint rule passes if its mean paired difference to the better member is >= -2 SE."""
    cfg = dict(n_train=4000, noise=0.1, epochs=5)
    cl, T = 40, 15
    arms = {"m2x2": ([(2, cl, T)], "none"), "m3x3": ([(3, cl, T)], "none"),
            "joint-sum": ([(2, cl, T), (3, cl, T)], "sum"), "joint-one": ([(2, cl, T), (3, cl, T)], "one")}
    out = {"config": dict(cfg, clauses=cl, T=T, seeds=seeds)}
    accs = {}
    for name, (spec, rule) in arms.items():
        accs[name] = np.array([toy_run(spec, rule, s, dev, **cfg)[0] for s in range(seeds)])
        out[name] = {"acc_mean": stats(accs[name])[0], "acc_se": stats(accs[name])[1],
                     "acc_per_seed": accs[name].tolist()}
    best = max(("m2x2", "m3x3"), key=lambda k: out[k]["acc_mean"])
    ok = True
    for rule in ("joint-sum", "joint-one"):
        d, se = stats(accs[rule] - accs[best])
        out[rule].update(vs=best, paired_diff_mean=d, paired_diff_se=se)
        ok = ok and d >= -2 * se
    return ok, out


def split_state(single, n, M, j):
    """Clauses of member j when a model with M*n clauses per class is cut into M members of n
    (each keeps half positive, half negative clauses per class)."""
    h, big = n // 2, M * n
    rows = []
    for k in range(single.n_classes):
        base = k * big
        rows += list(range(base + j * h, base + (j + 1) * h))
        rows += list(range(base + big // 2 + j * h, base + big // 2 + (j + 1) * h))
    idx = torch.tensor(rows, device=single.ta_state.device)
    return {"ta_state": single.ta_state[idx].clone(), "weights": single.weights[idx].clone()}


def t6(dev, seeds=32, batches=200):
    """R1: a `sum` team of M identical members is one model with M*n clauses and threshold M*T.

    (a) exact: cut a trained big model into M members; on the same batch every member learns
        from exactly the big model's (clamped) vote sums.
    (b) same feedback: selections per update, team vs big model, from the same state, agree
        within 3 SE over many batches; the `one` rule (threshold T) is the control that must not.
    (c) end to end: accuracy of teams and big models trained from scratch differ by < 2 SE.
    """
    n, T, M, bs = 20, 8, 3, 32
    out = {"config": dict(clauses_per_member=n, T=T, members=M, seeds=seeds, batches=batches)}
    x, y = tt.data.make_2d_noisy_xor(bs * (batches + 60), size=4, noise=0.1, seed=7)
    x, y = x.to(dev), y.to(dev)
    tt.seed_everything(1)
    big = toy_member(2, n * M, T * M, dev, bs)
    for i in range(60):                               # some training, so votes and weights are not trivial
        big.update(x[i * bs:(i + 1) * bs], y[i * bs:(i + 1) * bs])
    big_state = {k: v.clone() for k, v in big.state_dict().items()}
    teams = {}
    for rule in ("sum", "one"):
        ms = [toy_member(2, n, T, dev, bs) for _ in range(M)]
        teams[rule] = (JointTeam([f"m{j}" for j in range(M)], ms, rule),
                       [split_state(big, n, M, j) for j in range(M)])
    worst, sel = 0.0, {"big": [], "sum": [], "one": []}
    big.read_feedback()
    for b in range(batches):
        sl = slice((60 + b) * bs, (61 + b) * bs)
        big.load_state_dict(big_state)
        v_big = big.update(x[sl], y[sl])              # pre-update votes, clamped to M*T
        f = big.read_feedback()
        sel["big"].append(f["type_i"] + f["type_ii"])
        for rule, (team, states) in teams.items():
            for m, st in zip(team.members, states):
                m.load_state_dict(st)
                m.read_feedback()
            vs = [learning_votes(m, x[sl]) for m in team.members]
            team.update([x[sl]] * M, y[sl])
            fs = [m.read_feedback() for m in team.members]
            sel[rule].append(sum(q["type_i"] + q["type_ii"] for q in fs))
            if rule == "sum":
                seen = [(v + o).clamp(-tj, tj) for v, o, tj in zip(vs, team.offsets(vs), team.T_joint)]
                worst = max(worst, max(float((q - v_big).abs().max()) for q in seen))
    out["a_max_abs_vote_diff"] = worst
    big_m, big_se = stats(sel["big"])
    out["b_selections_per_update"] = {"big": [big_m, big_se]}
    for rule in ("sum", "one"):                       # paired by batch: [mean, mean diff, SE, |diff| in SE]
        d, se = stats(np.array(sel[rule]) - np.array(sel["big"]))
        out["b_selections_per_update"][rule] = [stats(sel[rule])[0], d, se, abs(d) / se]
    cfg = dict(n_train=3000, noise=0.1, epochs=2)
    for name, spec, rule in (("team", [(2, n, T)] * M, "sum"), ("single", [(2, n * M, T * M)], "none")):
        runs = np.array([toy_run(spec, rule, s, dev, **cfg) for s in range(seeds)])
        out[name] = {}
        for i, k in enumerate(STATS):
            out[name][f"{k}_mean"], out[name][f"{k}_se"] = stats(runs[:, i])
    out["c_team_vs_single_in_se"] = {
        k: abs(out["team"][f"{k}_mean"] - out["single"][f"{k}_mean"])
        / float(np.hypot(out["team"][f"{k}_se"], out["single"][f"{k}_se"])) for k in STATS}
    q = out["b_selections_per_update"]
    out["informative"] = bool(q["one"][3] > 5)        # the control is detected
    ok = worst == 0.0 and q["sum"][3] <= 3 and out["informative"] and out["c_team_vs_single_in_se"]["acc"] <= 2
    return ok, out


# ------------------------------------------------------------------------------------------ T8
def t8(dev):
    x, y = data(6, seed=6, dev=dev)
    team = team3(dev, None, True, "one")
    tt.seed_everything(17)
    for i in range(6):
        team.update([x[i]] * 3, y[i])
    before = team.combine(team.member_sums([x[0]] * 3))
    states = [{k: v.cpu().clone() for k, v in m.state_dict().items()} for m in team.members]
    fresh = team3(dev, None, True, "one", seed=50)
    for m, st in zip(fresh.members, states):
        m.load_state_dict(st)
    after = fresh.combine(fresh.member_sums([x[0]] * 3))
    plain = tt.ConvTsetlinMachine(K, CL, 4, 4.0, patch_size=2, input_shape=(PLANES, IMG, IMG), **KW)
    plain.load_state_dict(states[0])                  # a member file is an ordinary library checkpoint
    plain = plain.to(dev).eval()
    lib = bool(torch.equal(plain(x[0]), team.member_sums([x[0]] * 3)[0]))
    keys = sorted(states[0])
    return bool(torch.equal(before, after)) and lib and not any("joint" in k for k in keys), \
        {"team_scores_equal": bool(torch.equal(before, after)), "loads_into_library_model": lib, "keys": keys}


CHECKS = {"T0": t0, "T1": t1, "T2": t2, "T3": t3, "T5": t5, "T6": t6, "T8": t8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--allow-3090", action="store_true")
    args = ap.parse_args()
    if torch.device(args.device).type == "cuda":
        gpu = torch.cuda.get_device_name(torch.device(args.device))
        if "3090" in gpu and not args.allow_3090:
            raise SystemExit(f"{args.device} is the {gpu}: this programme runs on the 3080 only")
        torch.cuda.set_device(torch.device(args.device))
    rec = new_record("check_joint", device=args.device, checks={})
    f = RESULTS / "checks" / "check_joint.json"
    if args.only and f.exists():                      # a partial re-run keeps the other checks
        rec["checks"] = __import__("json").loads(f.read_text()).get("checks", {})
    failed = []
    for name, fn in CHECKS.items():
        if args.only and name not in args.only:
            continue
        t = time.time()
        ok, detail = fn(args.device)
        rec["checks"][name] = {"passed": bool(ok), "seconds": time.time() - t, "detail": detail}
        print(f"{name}: {'PASS' if ok else 'FAIL'} ({time.time() - t:.0f}s)", flush=True)
        if not ok:
            failed.append(name)
            print(__import__("json").dumps(detail, indent=1, default=float), flush=True)
    rec["checks"]["T4"] = {"passed": rec["checks"].get("T1", {}).get("passed"),
                           "detail": "pass A vs in-update votes: the max_abs_diff of T1 on every path"}
    rec["passed"] = not failed and all(c.get("passed") for c in rec["checks"].values())
    write_json(rec, f)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
