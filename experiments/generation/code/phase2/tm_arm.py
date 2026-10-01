"""Stage B: the Tsetlin machine token model.

Design follows the measured results of G0 and Phase 1 rather than the original document:

* One head per residual stage, coarse-to-fine — head m sees the symbols of stages < m at the
  same time step, so the factorisation P(s_1..s_M | ctx) = prod_m P(s_m | s_<m, ctx) is exact.
* Each head is a CoalescedTsetlinMachine in multi_label mode (G0 Finding 8: a class-owned
  machine puts ~2 % of its mass on the true support and is unusable for sampling).
* Context is the decoded WAVEFORM of the previous tokens, thermometer-encoded, not raw symbol
  ids — VALIDITY.md P2: one-hot token ids make clauses memorise exact n-grams, and an n-gram
  baseline then wins by construction.
* T is swept per variant (G0 Finding 1/8), weighted clauses, s=40, and the empty-clause gate is
  asserted every check (G0 Finding 4).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from torchtsetlin import CoalescedTsetlinMachine
from torchtsetlin.utils import seed_everything

import ecg_data as E
import tokenise as TK
from baselines import nll_report
from roundtrip import rhythm_metrics
from run_stage_a import write_record


# --------------------------------------------------------------------------- context
class WaveContext:
    """Previous L tokens as decoded waveform summaries + position + cardiac phase."""

    def __init__(self, tok, L=2, n_down=3, n_therm=8, device="cuda:0",
                 prev_id=0, use_wave=True):
        self.L, self.n_down, self.n_therm = L, n_down, n_therm
        self.prev_id, self.use_wave = prev_id, use_wave
        self.B, self.M, self.K = tok["B"], tok["M"], tok["K"]
        self.dev = torch.device(device)
        cen = tok["codebook"].kms                       # per-stage centroids
        cs = [torch.tensor(km.cluster_centers_, dtype=torch.float32, device=self.dev)
              for km in cen]
        # (M, K, C, B) -> summary (M, K, C*n_down) so a token's waveform is a lookup
        C = 8
        sums = []
        for c in cs:
            w = c.reshape(self.K, C, self.B)
            idx = torch.linspace(0, self.B - 1, n_down, device=self.dev).long()
            sums.append(w[:, :, idx].reshape(self.K, C * n_down))
        self.stage_sum = torch.stack(sums)              # (M, K, C*n_down)
        self.n_wave = C * n_down
        q = torch.linspace(0.02, 0.98, n_therm, device=self.dev)
        allv = self.stage_sum.reshape(-1)
        self.edges = torch.quantile(allv, q) * self.M   # summed over stages
        self.n_features = ((L * self.n_wave * n_therm) if use_wave else 0) \
            + prev_id * self.K + 16 + 16

    def token_wave(self, codes):
        """(B_, M) sub-symbols -> (B_, n_wave) summed decoded summary."""
        out = torch.zeros(codes.shape[0], self.n_wave, device=self.dev)
        for m in range(self.M):
            out += self.stage_sum[m][codes[:, m].long()]
        return out

    def encode(self, prev_codes, pos, phase):
        """prev_codes (B_, L, M) ints; pos (B_,) token index; phase (B_,) 0..15."""
        parts = []
        if self.use_wave:
            for j in range(self.L):
                w = self.token_wave(prev_codes[:, j])
                parts.append((w.unsqueeze(2) >= self.edges.view(1, 1, -1)).reshape(w.shape[0], -1))
        for j in range(self.prev_id):
            # exactly what ecg-markov1 conditions on: the previous token's stage-0 symbol id
            parts.append(torch.nn.functional.one_hot(prev_codes[:, j, 0].long(),
                                                     self.K).bool())
        ks = torch.arange(16, device=self.dev)
        parts.append((pos.unsqueeze(1) * 16 // 200) >= ks)
        parts.append(phase.unsqueeze(1) >= ks)
        return torch.cat(parts, dim=1)


# --------------------------------------------------------------------------- data stream
def make_stream(tokens, L, device):
    """(N, n_tok, M) -> flat arrays of (prev_codes, pos, target) for every position."""
    N, n_tok, M = tokens.shape
    t = torch.tensor(tokens.astype(np.int64), device=device)
    pad = torch.zeros(N, L, M, dtype=torch.long, device=device)
    tp = torch.cat([pad, t], dim=1)                       # (N, L+n_tok, M)
    prev = torch.stack([tp[:, L - 1 - j: L - 1 - j + n_tok] for j in range(L)], dim=2)
    pos = torch.arange(n_tok, device=device).unsqueeze(0).expand(N, -1)
    return prev.reshape(-1, L, M), pos.reshape(-1), t.reshape(-1, M)


def cardiac_phase(raw, B, n_tok, device):
    """Samples since the last R peak, quantised to 16 levels, per token position."""
    out = np.zeros((raw.shape[0], n_tok), dtype=np.int64)
    for i in range(raw.shape[0]):
        pk = E.detect_r_peaks(raw[i, 1])
        if len(pk) == 0:
            continue
        centres = np.arange(n_tok) * B + B // 2
        prev_r = np.searchsorted(pk, centres) - 1
        d = np.where(prev_r >= 0, centres - pk[np.clip(prev_r, 0, None)], 0)
        out[i] = np.clip(d * 16 // int(1.2 * E.FS), 0, 15)
    return torch.tensor(out.reshape(-1), device=device)


# --------------------------------------------------------------------------- arm
def run(a):
    dev = torch.device(a.device)
    tok = TK.build_and_tokenise()
    K, M, n_tok = tok["K"], tok["M"], tok["n_tokens"]
    ctx = WaveContext(tok, L=a.L, n_down=a.n_down, device=a.device,
                      prev_id=a.prev_id, use_wave=not a.no_wave)
    print(f"[{a.arm}] F={ctx.n_features} (wave={not a.no_wave}, prev_id={a.prev_id}) "
          f"K={K} clauses={a.clauses} T={a.T} s={a.s} batch={a.batch} "
          f"ns={a.negative_scale if a.negative_scale > 0 else 1.0/K:.5f}", flush=True)

    prev_tr, pos_tr, y_tr = make_stream(tok["train"], a.L, dev)
    prev_va, pos_va, y_va = make_stream(tok["val"], a.L, dev)
    ph_tr = cardiac_phase(tok["train_raw"], tok["B"], n_tok, dev)
    ph_va = cardiac_phase(tok["val_raw"], tok["B"], n_tok, dev)

    stages = list(range(M)) if a.stages is None else a.stages
    heads, t0 = {}, time.time()
    for m in stages:
        seed_everything(a.seed + m)
        # head m also sees the already-decided coarser stages of the same token
        extra = m * a.n_therm_prev
        # negative_scale ~ 1/K: with one-hot targets there are K-1 negatives per positive,
        # and ns=1.0 swamps the positive signal (measured: gap to the count baseline shrinks
        # 6x at N=32 when ns is set to ~1/N). Supersedes the ns=1.0 advice in G0 Finding 8,
        # which was derived for calibration fidelity at small K, not one-hot sequence targets.
        ns = a.negative_scale if a.negative_scale > 0 else 1.0 / K
        h = CoalescedTsetlinMachine(ctx.n_features + extra, K, a.clauses, a.T, a.s,
                                    multi_label=True, negative_scale=ns).to(dev)
        heads[m] = h

    def feats(prev, pos, ph, y, m):
        x = ctx.encode(prev, pos, ph)
        if m == 0:
            return x
        cw = [ctx.stage_sum[j][y[:, j]] for j in range(m)]
        cw = torch.cat(cw, dim=1)
        ed = ctx.edges[: a.n_therm_prev]
        bits = (cw.unsqueeze(2) >= ed.view(1, 1, -1)).reshape(cw.shape[0], -1)
        return torch.cat([x, bits[:, : m * a.n_therm_prev]], dim=1)

    n = prev_tr.shape[0]
    g = torch.Generator(device=dev).manual_seed(a.seed)
    steps = a.examples // a.batch
    for m, h in heads.items():
        h.train()
        for i in range(1, steps + 1):
            idx = torch.randint(0, n, (a.batch,), generator=g, device=dev)
            x = feats(prev_tr[idx], pos_tr[idx], ph_tr[idx], y_tr[idx], m)
            h.update(x, torch.nn.functional.one_hot(y_tr[idx, m], K).to(torch.bool))
        empty = int((h.include_count == 0).sum())
        frac = empty / h.n_clauses_total
        print(f"    head {m}: {steps} steps, empty {empty}/{h.n_clauses_total} ({frac:.1%})",
              flush=True)
        if frac > 0.01 and not a.allow_empty:
            raise SystemExit(f"[{a.arm}] ABORT head {m}: {frac:.1%} empty clauses (G0 Finding 4)")

    # held-out token NLL, teacher-forced
    res = {}
    tot, cnt = 0.0, 0
    per_stage = np.zeros(M); per_cnt = np.zeros(M)
    prev_te, pos_te, y_te = make_stream(tok["test"][: a.n_eval], a.L, dev)
    ph_te = cardiac_phase(tok["test_raw"][: a.n_eval], tok["B"], n_tok, dev)
    with torch.no_grad():
        for m, h in heads.items():
            h.eval()
            for lo in range(0, prev_te.shape[0], a.eval_chunk):
                sl = slice(lo, lo + a.eval_chunk)
                x = feats(prev_te[sl], pos_te[sl], ph_te[sl], y_te[sl], m)
                p = h.predict_proba(x, method="linear").clamp_min(1e-9)
                p = p / p.sum(1, keepdim=True)
                v = float(torch.log(p[torch.arange(p.shape[0], device=dev),
                                      y_te[sl, m]]).sum())
                tot += v; cnt += p.shape[0]
                per_stage[m] += v; per_cnt[m] += p.shape[0]
    res.update(nll_report(-tot, cnt))
    # Temperature-calibrated softmax, fitted on VALIDATION. The analytic map understates
    # badly when |v|max/T is small -- that mistake cost a whole sweep earlier.
    with torch.no_grad():
        for m, h in heads.items():
            vv = torch.cat([h(feats(prev_va[lo:lo + a.eval_chunk], pos_va[lo:lo + a.eval_chunk],
                                    ph_va[lo:lo + a.eval_chunk], y_va[lo:lo + a.eval_chunk], m))
                            for lo in range(0, min(200000, prev_va.shape[0]), a.eval_chunk)]).float()
            yv = y_va[: vv.shape[0], m]
            vt = torch.cat([h(feats(prev_te[lo:lo + a.eval_chunk], pos_te[lo:lo + a.eval_chunk],
                                    ph_te[lo:lo + a.eval_chunk], y_te[lo:lo + a.eval_chunk], m))
                            for lo in range(0, prev_te.shape[0], a.eval_chunk)]).float()
            best, bt = float("inf"), a.T
            arv = torch.arange(vv.shape[0], device=dev)
            for f in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
                nll = float(-torch.log_softmax(vv / (a.T * f), dim=1)[arv, yv].mean())
                if nll < best:
                    best, bt = nll, a.T * f
            art = torch.arange(vt.shape[0], device=dev)
            res[f"stage{m}_nll_calibrated"] = float(
                -torch.log_softmax(vt / bt, dim=1)[art, y_te[:, m]].mean())
            res[f"stage{m}_temperature"] = float(bt)
            res[f"stage{m}_vote_over_T"] = float(vt.abs().max() / a.T)
    res["nll_per_stage"] = {m: round(float(-per_stage[m] / max(1, per_cnt[m])), 3)
                            for m in heads}
    res["stages_trained"] = stages
    res["vote_amplitude"] = {}
    with torch.no_grad():
        for m, h in heads.items():
            x = feats(prev_te[:4096], pos_te[:4096], ph_te[:4096], y_te[:4096], m)
            v = h(x).float()
            res["vote_amplitude"][m] = [round(float(v.abs().max()), 1), float(h.T)]

    cfg = dict(arm=a.arm, B=tok["B"], K=K, M=M, L=a.L, clauses=a.clauses, T=a.T, s=a.s,
               batch=a.batch, examples=a.examples, n_features=ctx.n_features, seed=a.seed)
    r = write_record(a.arm, cfg, res, time.time() - t0)
    cal = {m: round(res[f"stage{m}_nll_calibrated"], 3) for m in heads}
    print(f"  per-stage NLL linear {res['nll_per_stage']} | calibrated {cal} "
          f"(uniform 6.931, markov1 stage0 4.429)", flush=True)
    print(f"[{a.arm}] held-out NLL {res['nll_per_subsymbol']:.3f} nats/sub-symbol "
          f"({res['nll_bits_per_subsymbol']:.2f} bits)  ({r['wall_s']:.0f}s)", flush=True)
    torch.save({"heads": {m: h.state_dict() for m, h in heads.items()}, "cfg": cfg},
               TK.CACHE.parent / f"{a.arm}_seed{a.seed}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ecg-tm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--n-down", type=int, default=3)
    ap.add_argument("--n-therm-prev", type=int, default=8)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=float, default=8000.0)
    ap.add_argument("--s", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=240)
    ap.add_argument("--examples", type=int, default=2_000_000)
    ap.add_argument("--eval-chunk", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-empty", action="store_true")
    ap.add_argument("--negative-scale", type=float, default=-1.0,
                    help="<=0 means 1/K (the measured optimum for one-hot targets)")
    ap.add_argument("--prev-id", type=int, default=0,
                    help="append one-hot stage-0 ids of the last N tokens (what markov1 uses)")
    ap.add_argument("--no-wave", action="store_true")
    ap.add_argument("--stages", type=int, nargs="+", default=None,
                    help="which residual stages to model (default all). Per-stage analysis "
                         "showed all predictability is in stage 0.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
