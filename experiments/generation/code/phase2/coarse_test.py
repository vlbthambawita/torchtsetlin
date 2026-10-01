"""Is the TM's Stage B failure a CAPACITY limit rather than an encoding one?

G0 Finding 6 measured the capacity cliff directly: 800 clauses/class calibrated 112 arbitrary
contexts fine, collapsed at 1000, and 16x more data did not fix it. Stage B asks a TM with 2 000
shared clauses to learn a 1024-context x 1024-output conditional — far past that cliff, while a
count table does it trivially.

This script shrinks the alphabet by clustering the stage-0 codebook centroids into N coarse
symbols and runs BOTH the TM and a matched count-based Markov model on the same N-way problem.
If the TM closes the gap as N falls, capacity is the cause and the fix is clauses, not encoding.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from torchtsetlin import CoalescedTsetlinMachine
from torchtsetlin.utils import seed_everything

import tokenise as TK
from run_stage_a import write_record


def coarse_map(tok, n_coarse, seed=0):
    """1024 stage-0 symbols -> n_coarse clusters, by k-means on the centroids themselves."""
    from sklearn.cluster import KMeans
    cen = tok["codebook"].kms[0].cluster_centers_
    km = KMeans(n_clusters=n_coarse, random_state=seed, n_init=4).fit(cen)
    return km.labels_.astype(np.int64)


def markov_nll(tr, te, N):
    """Count table with add-one smoothing: P(c_t | c_{t-1}). The arm to beat."""
    cnt = np.ones((N, N))
    for rec in tr:
        np.add.at(cnt, (rec[:-1], rec[1:]), 1.0)
    p = cnt / cnt.sum(1, keepdims=True)
    tot, n = 0.0, 0
    for rec in te:
        tot += np.log(p[rec[:-1], rec[1:]]).sum()
        n += len(rec) - 1
    marg = np.bincount(np.concatenate(tr), minlength=N) + 1.0
    marg = marg / marg.sum()
    mt = sum(np.log(marg[rec[1:]]).sum() for rec in te)
    return -tot / n, -mt / n


def run(a):
    dev = torch.device(a.device)
    tok = TK.build_and_tokenise()
    cmap = coarse_map(tok, a.n_coarse)
    tr = [cmap[r[:, 0].astype(int)] for r in tok["train"]]
    va = [cmap[r[:, 0].astype(int)] for r in tok["val"][: a.n_val]]
    te = [cmap[r[:, 0].astype(int)] for r in tok["test"][: a.n_eval]]

    mk_nll, mg_nll = markov_nll(tr, te, a.n_coarse)

    def stream(seqs):
        x = torch.tensor(np.concatenate([s[:-1] for s in seqs]), device=dev)
        y = torch.tensor(np.concatenate([s[1:] for s in seqs]), device=dev)
        return x, y

    xtr, ytr = stream(tr)
    xva, yva = stream(va)
    xte, yte = stream(te)
    seed_everything(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    # negative_scale down-weights Type II on the N-1 negative outputs. With one-hot targets
    # the negative:positive ratio is N-1:1, so 1.0 may swamp the positive signal. G0 Finding 8
    # fixed it at 1.0 for CALIBRATION fidelity at K<=256; that may not transfer here.
    h = CoalescedTsetlinMachine(a.n_coarse, a.n_coarse, a.clauses, a.T, a.s,
                                multi_label=True,
                                negative_scale=a.negative_scale).to(dev)
    t0 = time.time()
    h.train()
    for _ in range(a.examples // a.batch):
        i = torch.randint(0, xtr.shape[0], (a.batch,), generator=g, device=dev)
        h.update(torch.nn.functional.one_hot(xtr[i], a.n_coarse).bool(),
                 torch.nn.functional.one_hot(ytr[i], a.n_coarse).bool())
    empty = int((h.include_count == 0).sum())
    h.eval()
    with torch.no_grad():
        vv = h(torch.nn.functional.one_hot(xva, a.n_coarse).bool()).float()
        vt = h(torch.nn.functional.one_hot(xte, a.n_coarse).bool()).float()
    best, bt = float("inf"), a.T
    arv = torch.arange(vv.shape[0], device=dev)
    for f in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        nll = float(-torch.log_softmax(vv / (a.T * f), dim=1)[arv, yva].mean())
        if nll < best:
            best, bt = nll, a.T * f
    art = torch.arange(vt.shape[0], device=dev)
    tm_nll = float(-torch.log_softmax(vt / bt, dim=1)[art, yte].mean())

    res = {"n_coarse": a.n_coarse, "negative_scale": a.negative_scale, "tm_nll": tm_nll, "markov_nll": mk_nll,
           "marginal_nll": mg_nll, "uniform_nll": float(np.log(a.n_coarse)),
           "gap_tm_minus_markov": tm_nll - mk_nll,
           "empty_clauses": empty, "clauses": a.clauses,
           "vote_over_T": float(vt.abs().max() / a.T), "temperature": float(bt)}
    arm = f"coarse-N{a.n_coarse}-C{a.clauses}-T{a.T:.0f}-ns{a.negative_scale}"
    r = write_record(arm, vars(a), res, time.time() - t0)
    print(f"[{arm}] TM {tm_nll:.3f} | markov {mk_nll:.3f} | marginal {mg_nll:.3f} | "
          f"uniform {res['uniform_nll']:.3f} | gap {res['gap_tm_minus_markov']:+.3f} | "
          f"empty {empty}/{h.n_clauses_total} ({r['wall_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-coarse", type=int, default=32)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=float, default=2000.0)
    ap.add_argument("--s", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=240)
    ap.add_argument("--examples", type=int, default=1_500_000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--negative-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
