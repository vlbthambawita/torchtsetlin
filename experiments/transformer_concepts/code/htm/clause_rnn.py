"""torchtsetlin_plans/experiments/transformer_concepts/PLAN_HTM.md E-H3c: a clause RNN whose state is learned with local, self-supervised targets.

State h_t in {0,1}^m = outputs of a library ``CoalescedTsetlinMachine(multi_label=True)`` that
reads [e(x_t) | (e(x_{t-1})) | h_{t-1}] at every step; no gradients, no credit across time.
A fixed comparator and a library flat TM read the state out (THEORY_HTM H3.3-H3.8).

Arms (``arm``):
- ``seen_tf``     T3. Target s_t[a] = [a in x_0..x_t] (union pooling), teacher forcing: the
                  input state is the true s_{t-1}. Evaluated free running.
- ``next_set``    T3. Target y_t[a] = [x_{t+1} = a] (next token, HTM's own objective), trained
                  free running (the state is the model's own previous prediction); the state is
                  the predicted *set* (votes > 0, HTM's predictive cells). Initial state all ones.
- ``next_argmax`` as ``next_set`` but the state is the one-hot argmax (a single decision).
- ``mem_tf``      T2. Target b_t[k, v] = transitions seen up to t; input [e(x_t) | e(x_{t-1}) | b_{t-1}];
                  teacher forcing; evaluated free running.
Readout features: T3: [OR_{t>=1} <e(x_t), h_{t-1}>, OR_{t>=1} NOT <e(x_t), h_{t-1}>] (repeat bit,
surprise bit); T2: [e(x_{L-1}) | h_{L-2}].

    python -m experiments.transformer_concepts.code.htm.clause_rnn jobs | freeze | final
    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.htm.clause_rnn work jobs.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as Fnn

import torchtsetlin as tt
from experiments.transformer_concepts.code import runner

R = runner.RESULTS / "htm"
ROOT = runner.ROOT
ARMS = ("seen_tf", "next_set", "next_argmax", "mem_tf")
TASK = {"seen_tf": "T3", "next_set": "T3", "next_argmax": "T3", "mem_tf": "T2"}
S_GRID = (2.0, 5.0, 15.0)
FINAL_SEEDS = (23, 29, 31, 37, 41)
READOUT_S = {"T3": 5.0, "T2": 40.0}  # the frozen E-H3b readout settings (results/htm/frozen_h3.json)


def e1(x, V):
    return Fnn.one_hot(x, V).bool()


def true_states(arm, x, V):
    """(N, L, m) target state per step."""
    e = e1(x, V)
    if arm in ("seen_tf", "next_set", "next_argmax"):
        if arm == "seen_tf":
            return running_or(e)  # seen up to and including t
        nxt = torch.zeros_like(e)
        nxt[:, :-1] = e[:, 1:]
        return nxt  # next token
    N, L, _ = e.shape
    prev = torch.zeros_like(e)
    prev[:, 1:] = e[:, :-1]
    step = (prev.unsqueeze(3) & e.unsqueeze(2)).reshape(N, L, V * V)
    step[:, -1] = False  # the query slot adds no transition (as registers.transitions)
    return running_or(step)


def running_or(b):
    """OR over the time axis up to and including t, in Boolean (an int32 cumsum of the T2
    transition tensor needs ~2 GB)."""
    out = b.clone()
    for t in range(1, out.shape[1]):
        out[:, t] |= out[:, t - 1]
    return out


def inputs(arm, x, V, h_prev, t):
    e = e1(x[:, t], V)
    if arm == "mem_tf":
        p = e1(x[:, t - 1], V) if t > 0 else torch.zeros_like(e)
        return torch.cat([e, p, h_prev], 1)
    return torch.cat([e, h_prev], 1)


def h0(arm, N, m, dev):
    if arm in ("next_set", "next_argmax"):
        return torch.ones(N, m, dtype=torch.bool, device=dev)
    return torch.zeros(N, m, dtype=torch.bool, device=dev)


@torch.no_grad()
def free_run(model, arm, x, V, m, bs=512):
    """States h_0..h_{L-1} computed by the model from its own previous state: (N, L, m)."""
    model.eval()
    out = []
    for k in range(0, len(x), bs):
        xb = x[k:k + bs]
        h = h0(arm, len(xb), m, x.device)
        hs = []
        for t in range(x.shape[1]):
            h = state_from(model, arm, inputs(arm, xb, V, h, t))
            hs.append(h)
        out.append(torch.stack(hs, 1))
    model.train()
    return torch.cat(out)


def state_from(model, arm, inp):
    if arm == "next_argmax":
        v = model.forward(inp)
        return Fnn.one_hot(v.argmax(1), v.shape[1]).bool()
    return model.predict(inp)


def readout_features(arm, x, V, hs):
    e = e1(x, V)
    if TASK[arm] == "T3":
        hit = (e[:, 1:] & hs[:, :-1]).any(2)  # (N, L-1): x_t in h_{t-1}
        return torch.stack([hit.any(1), (~hit).any(1)], 1)
    return torch.cat([e[:, -1], hs[:, -2]], 1)


def run(stage, key, arm, L, s, seed, epochs=5, test=False, n_clauses=None, T=15.0, device="cuda", batch=32,
        readout_epochs=20):
    path = R / stage / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    task = TASK[arm]
    xtr, ytr, spec = runner.make_data(task, L, "train")
    xva, yva, _ = runner.make_data(task, L, "valid")
    V = spec.V
    xtr, xva = xtr.to(device), xva.to(device)
    m = V * V if arm == "mem_tf" else V
    n_clauses = n_clauses or (1000 if arm == "mem_tf" else 4 * V)
    torch.manual_seed(seed)
    model = tt.CoalescedTsetlinMachine(None, m, n_clauses, T, s, multi_label=True).to(device)
    g = torch.Generator().manual_seed(seed)
    S_tr = true_states(arm, xtr, V)
    t_train = 0.0
    for ep in range(epochs):
        t0 = time.perf_counter()
        if arm.endswith("_tf"):  # teacher forcing: rows (sequence, t) shuffled across sequences
            N = len(xtr)
            rows = torch.randperm(N * L, generator=g).to(device)
            for k in range(0, len(rows), batch):
                r = rows[k:k + batch]
                n, t = r // L, r % L
                hp = torch.where((t > 0).unsqueeze(1), S_tr[n, (t - 1).clamp(min=0)], h0(arm, len(r), m, device))
                e = e1(xtr[n, t], V)
                if arm == "mem_tf":
                    p = torch.where((t > 0).unsqueeze(1), e1(xtr[n, (t - 1).clamp(min=0)], V), torch.zeros_like(e))
                    inp = torch.cat([e, p, hp], 1)
                else:
                    inp = torch.cat([e, hp], 1)
                model.update(inp, S_tr[n, t])
        else:  # free running: the state is the model's own previous output
            perm = torch.randperm(len(xtr), generator=g).to(device)
            for k in range(0, len(xtr), batch):
                xb = xtr[perm[k:k + batch]]
                h = h0(arm, len(xb), m, device)
                for t in range(L - 1):
                    inp = inputs(arm, xb, V, h, t)
                    model.update(inp, S_tr[perm[k:k + batch], t])
                    model.eval()
                    with torch.no_grad():
                        h = state_from(model, arm, inp)
                    model.train()
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
    # state quality on validation, free running (and per step, for H3.9)
    S_va = true_states(arm, xva, V)
    H_va = free_run(model, arm, xva, V, m)
    ref = S_va if arm != "next_set" and arm != "next_argmax" else ~running_or(e1(xva, V))
    # next-token arms are compared with the Bayes set of THEORY_HTM H3.8: the symbols not yet seen
    steps = slice(0, L - 1) if arm in ("next_set", "next_argmax") else slice(0, L)
    agree = (H_va[:, steps] == ref[:, steps]).float()
    rec = dict(stage=stage, key=key, arm=arm, task=task, L=L, V=V, s=s, seed=seed, T=T, n_clauses=n_clauses,
               epochs=epochs, batch=batch, state_bits=m,
               state_bit_acc=float(agree.mean()), state_all_bits_acc=float(agree.all(2).float().mean()),
               state_bit_acc_by_t=[round(float(v), 5) for v in agree.mean((0, 2))],
               state_density=float(H_va.float().mean()), reference=("true register" if ref is S_va else
                                                                    "not-yet-seen set (Bayes, H3.8)"),
               train_seconds=round(t_train, 1), torchtsetlin=tt.__version__)
    # readout: a library flat TM on the comparator features of the free-running state
    ftr = readout_features(arm, xtr, V, free_run(model, arm, xtr, V, m))
    fva = readout_features(arm, xva, V, H_va)
    ro = tt.TsetlinMachine(None, spec.n_classes, 100, 25, READOUT_S[task], weighted=True).to(device)
    ytr_d, yva_d = ytr.to(device), yva.to(device)
    best, best_state = -1.0, None
    for ep in range(readout_epochs):
        p = torch.randperm(len(ftr), generator=g).to(device)
        for k in range(0, len(ftr), batch):
            ro.update(ftr[p[k:k + batch]], ytr_d[p[k:k + batch]])
        ro.eval()
        acc = float((ro.predict(fva) == yva_d).float().mean())
        ro.train()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in ro.state_dict().items()}
    rec["readout_best_val"] = best
    if test:
        xte, yte, _ = runner.make_data(task, L, "test")
        xte = xte.to(device)
        H_te = free_run(model, arm, xte, V, m)
        ro.load_state_dict(best_state)
        ro.eval()
        runner.log_test_access(f"htm/{stage}/{key}", len(xte))
        rec["test_acc"] = float((ro.predict(readout_features(arm, xte, V, H_te)).cpu() == yte).float().mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def big_L(arm):
    return 33 if TASK[arm] == "T2" else 32


CLAUSE_MULT = {"T3": (4, 16), "T2": (2.5, 10)}  # clauses = mult * (V for T3, V^2 for T2); D22
T_GRID = (15.0, 50.0)


def n_cl(a, mult):
    V = runner.task_spec(TASK[a], big_L(a)).V
    return int(mult * (V * V if TASK[a] == "T2" else V))


def jobs():
    out = [dict(stage="h3c_search", key=f"{a}_L{big_L(a)}_c{n_cl(a, cm)}_T{int(T)}_s{int(s)}", arm=a, L=big_L(a),
                s=s, seed=7, n_clauses=n_cl(a, cm), T=T)
           for a in ARMS for cm in CLAUSE_MULT[TASK[a]] for T in T_GRID for s in S_GRID]
    (ROOT / "jobs_htm_h3c_search.json").write_text(json.dumps(out, indent=1))
    print(len(out))


def freeze():
    p = R / "frozen_h3c.json"
    if p.exists():
        raise SystemExit("freeze exists")
    chosen = {}
    for a in ARMS:  # self-supervised criterion: free-running state-bit accuracy (no labels)
        recs = [json.loads((R / "h3c_search" / f"{a}_L{big_L(a)}_c{n_cl(a, cm)}_T{int(T)}_s{int(s)}.json").read_text())
                for cm in CLAUSE_MULT[TASK[a]] for T in T_GRID for s in S_GRID]
        b = max(recs, key=lambda r: (r["state_bit_acc"], -r["n_clauses"], -r["s"]))
        chosen[a] = dict(s=b["s"], T=b["T"], clause_mult=b["n_clauses"] / (b["V"] ** 2 if TASK[a] == "T2" else b["V"]),
                         state_bit_acc=b["state_bit_acc"], readout_best_val=b["readout_best_val"])
    p.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), chosen=chosen,
                                 rule="best free-running state-bit accuracy on validation, seed 7"), indent=1))
    print(json.dumps(chosen, indent=1))


def final():
    fz = json.loads((R / "frozen_h3c.json").read_text())["chosen"]
    out = []
    for a, c in fz.items():
        for L in ((17, 33) if TASK[a] == "T2" else (8, 16, 32)):
            for seed in FINAL_SEEDS:
                V = runner.task_spec(TASK[a], L).V
                ncl = int(c["clause_mult"] * (V * V if TASK[a] == "T2" else V))
                out.append(dict(stage="h3c_final", key=f"{a}_L{L}_seed{seed}", arm=a, L=L, s=c["s"], seed=seed,
                                T=c["T"], n_clauses=ncl, test=True))
    (ROOT / "jobs_htm_h3c_final.json").write_text(json.dumps(out, indent=1))
    print(len(out))


def work(paths):
    for path in paths:
        for job in json.loads(open(path).read()):
            d = R / job["stage"]
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{job['key']}.json").exists():
                continue
            try:
                os.close(os.open(d / f"{job['key']}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            t0 = time.time()
            try:
                r = run(**job)
                (d / f"{job['key']}.lock").unlink(missing_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] {job['stage']}/{job['key']} state={r['state_bit_acc']:.4f} "
                      f"readout={r['readout_best_val']:.4f} {time.time() - t0:.0f}s", flush=True)
            except Exception:
                (d / f"{job['key']}.err").write_text(traceback.format_exc())
                print("FAILED", job["key"], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "work":
        work(sys.argv[2:])
    else:
        {"jobs": jobs, "freeze": freeze, "final": final}[cmd]()
