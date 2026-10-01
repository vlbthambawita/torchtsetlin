"""Stage B baselines: the floor and the arm that can falsify H4.

`ecg-markov` conditions on the same history a TM would see and is closed-form. If the Tsetlin
machine does not beat it on held-out token NLL, the TM contributes only interpretability, and
VALIDITY.md P6 says to report that rather than bury it.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

import ecg_data as E
import tokenise as TK
from roundtrip import rhythm_metrics
from run_stage_a import write_record


def nll_report(total_nats, n_sub):
    return {"nll_per_subsymbol": float(total_nats / n_sub),
            "nll_bits_per_subsymbol": float(total_nats / n_sub / np.log(2)),
            "uniform_nll_per_subsymbol": float(np.log(1024))}


class Marginal:
    name = "ecg-marginal"

    def fit(self, tr, K, M):
        self.K, self.M = K, M
        self.p = np.zeros((M, K))
        for m in range(M):
            c = np.bincount(tr[:, :, m].ravel(), minlength=K) + 1.0
            self.p[m] = c / c.sum()
        return self

    def logp(self, seq, per_stage=None):    # seq (n_tok, M)
        tot = 0.0
        for m in range(self.M):
            v = float(np.log(self.p[m, seq[:, m]]).sum())
            tot += v
            if per_stage is not None:
                per_stage[m] += v
        return tot

    def sample(self, n_tok, rng):
        return np.stack([rng.choice(self.K, size=n_tok, p=self.p[m])
                         for m in range(self.M)], axis=1).astype(np.int16)


class MarkovCoarse:
    """P(s_m^t | s_1^{t-1}, ..., s_1^{t-L}) — conditions on the COARSE (stage-1) history,
    which is the informative part of an RVQ token and keeps the table tractable."""

    def __init__(self, L=1):
        self.L = L
        self.name = f"ecg-markov{L}"

    def fit(self, tr, K, M):
        self.K, self.M = K, M
        self.back = Marginal().fit(tr, K, M)
        self.tab = [defaultdict(lambda: np.zeros(K, dtype=np.float32)) for _ in range(M)]
        for rec in tr:
            for t in range(self.L, rec.shape[0]):
                ctx = tuple(int(rec[t - 1 - i, 0]) for i in range(self.L))
                for m in range(M):
                    self.tab[m][ctx][int(rec[t, m])] += 1.0
        self.tab = [dict(d) for d in self.tab]
        return self

    def _p(self, m, ctx):
        c = self.tab[m].get(ctx)
        if c is None:
            return self.back.p[m]
        tot = c.sum()
        # Simple interpolation with the marginal; weight grows with evidence.
        lam = tot / (tot + 20.0)
        return lam * (c / tot) + (1 - lam) * self.back.p[m]

    def logp(self, seq, per_stage=None):
        tot = 0.0
        for t in range(seq.shape[0]):
            for m in range(self.M):
                if t < self.L:
                    v = float(np.log(self.back.p[m, seq[t, m]]))
                else:
                    ctx = tuple(int(seq[t - 1 - i, 0]) for i in range(self.L))
                    v = float(np.log(self._p(m, ctx)[seq[t, m]] + 1e-12))
                tot += v
                if per_stage is not None:
                    per_stage[m] += v
        return tot

    def sample(self, n_tok, rng):
        out = np.zeros((n_tok, self.M), dtype=np.int16)
        for t in range(n_tok):
            if t < self.L:
                out[t] = self.back.sample(1, rng)[0]
                continue
            ctx = tuple(int(out[t - 1 - i, 0]) for i in range(self.L))
            for m in range(self.M):
                p = self._p(m, ctx)
                out[t, m] = rng.choice(self.K, p=p / p.sum())
        return out


def evaluate(model, tok, n_eval, n_gen, seed=0):
    t0 = time.time()
    K, M, n_tok = tok["K"], tok["M"], tok["n_tokens"]
    te = tok["test"][:n_eval].astype(np.int32)
    per_stage = np.zeros(M)
    total = sum(model.logp(te[i], per_stage) for i in range(te.shape[0]))
    res = nll_report(-total, te.shape[0] * n_tok * M)
    res["nll_per_stage"] = [round(float(-v / (te.shape[0] * n_tok)), 3) for v in per_stage]

    rng = np.random.default_rng(seed)
    gen = np.stack([model.sample(n_tok, rng) for _ in range(n_gen)])
    xg = TK.decode_tokens(tok, gen.astype(np.int32))
    xr = E.derive_full12(tok["test_raw"][:n_gen][:, E.INDEP_IDX])
    res.update(rhythm_metrics(xr, xg))
    res["n_eval"] = int(te.shape[0]); res["n_gen"] = int(n_gen)
    r = write_record(model.name, dict(arm=model.name, B=tok["B"], K=K, M=M,
                                      L=getattr(model, "L", 0), n_eval=int(te.shape[0])),
                     res, time.time() - t0)
    print(f"  per-stage NLL: {res['nll_per_stage']}", flush=True)
    print(f"[{model.name}] NLL {res['nll_per_subsymbol']:.3f} nats/sub-symbol "
          f"({res['nll_bits_per_subsymbol']:.2f} bits; uniform = 6.93 nats) | "
          f"R-peak recall {res['r_peak_recall']:.3f} | RR KS {res['rr_ks']:.3f} | "
          f"RR mean {res['rr_mean_recon']:.3f}s vs {res['rr_mean_real']:.3f}s "
          f"({r['wall_s']:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=100)
    ap.add_argument("--L", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    tok = TK.build_and_tokenise()
    tr = tok["train"].astype(np.int32)
    print(f"  tokens {tok['train'].shape}, alphabet {tok['K']}, stages {tok['M']}")
    evaluate(Marginal().fit(tr, tok["K"], tok["M"]), tok, a.n_eval, a.n_gen)
    for L in a.L:
        evaluate(MarkovCoarse(L).fit(tr, tok["K"], tok["M"]), tok, a.n_eval, a.n_gen)


if __name__ == "__main__":
    main()
