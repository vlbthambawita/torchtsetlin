"""Re-score trained stage-0 heads with a temperature-calibrated softmax.

Why: the first evaluation used `predict_proba(method="linear")`, i.e. p_k ∝ 0.5(1 + v_k/T).
At |v|max/T = 0.34 that maps 1024 outputs into [0.33, 0.50] and normalises to almost exactly
uniform, so a head that HAS learned structure (vote sd across inputs = 232) scores 6.91 nats,
indistinguishable from uniform. Phase 1 established that the fitted calibrator is worth
12-29 nats/image and must always be applied; it was not applied here. No retraining — this
re-scores existing checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import time

import numpy as np
import torch

from torchtsetlin import CoalescedTsetlinMachine

import tokenise as TK
from baselines import nll_report
from tm_arm import WaveContext, cardiac_phase, make_stream
from run_stage_a import write_record


@torch.no_grad()
def votes_for(h, ctx, prev, pos, ph, chunk=20000):
    out = []
    for lo in range(0, prev.shape[0], chunk):
        sl = slice(lo, lo + chunk)
        out.append(h(ctx.encode(prev[sl], pos[sl], ph[sl])).float())
    return torch.cat(out)


def fit_temperature(v, y, grid):
    """Pick the softmax temperature that minimises validation NLL."""
    best, bt = float("inf"), grid[0]
    ar = torch.arange(v.shape[0], device=v.device)
    for t in grid:
        lp = torch.log_softmax(v / t, dim=1)
        nll = float(-lp[ar, y].mean())
        if nll < best:
            best, bt = nll, t
    return bt, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=100)
    a = ap.parse_args()
    dev = torch.device(a.device)
    tok = TK.build_and_tokenise()

    for ckpt in sorted(glob.glob(str(TK.CACHE.parent / "tm-s0-T*_seed0.pt"))):
        t0 = time.time()
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        cfg = ck["cfg"]
        ctx = WaveContext(tok, L=cfg["L"], n_down=3, device=a.device)
        h = CoalescedTsetlinMachine(cfg["n_features"], cfg["K"], cfg["clauses"], cfg["T"],
                                    cfg["s"], multi_label=True, negative_scale=1.0).to(dev)
        h.load_state_dict(ck["heads"][0])
        h.eval()

        pv, sv, yv = make_stream(tok["val"][: a.n_val], cfg["L"], dev)
        phv = cardiac_phase(tok["val_raw"][: a.n_val], tok["B"], tok["n_tokens"], dev)
        pt, st, yt = make_stream(tok["test"][: a.n_eval], cfg["L"], dev)
        pht = cardiac_phase(tok["test_raw"][: a.n_eval], tok["B"], tok["n_tokens"], dev)

        vv = votes_for(h, ctx, pv, sv, phv)
        vt = votes_for(h, ctx, pt, st, pht)
        grid = [cfg["T"] * f for f in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)]
        T_opt, val_nll = fit_temperature(vv, yv[:, 0], grid)

        ar = torch.arange(vt.shape[0], device=dev)
        lp = torch.log_softmax(vt / T_opt, dim=1)
        test_nll = float(-lp[ar, yt[:, 0]].mean())
        lin = h.predict_proba(ctx.encode(pt[:20000], st[:20000], pht[:20000]),
                              method="linear").clamp_min(1e-9)
        lin = lin / lin.sum(1, keepdim=True)
        lin_nll = float(-torch.log(lin[torch.arange(lin.shape[0], device=dev),
                                       yt[:20000, 0]]).mean())

        res = {"stage0_nll_softmax_calibrated": test_nll,
               "stage0_nll_linear_uncalibrated": lin_nll,
               "temperature": float(T_opt), "temperature_over_T": float(T_opt / cfg["T"]),
               "val_nll": val_nll, "vote_abs_max": float(vt.abs().max()),
               "vote_over_T": float(vt.abs().max() / cfg["T"]),
               "uniform_nll": float(np.log(cfg["K"]))}
        arm = f"recal-{cfg['arm']}"
        r = write_record(arm, cfg, res, time.time() - t0)
        print(f"[{arm}] stage-0 NLL: calibrated **{test_nll:.3f}** vs linear {lin_nll:.3f} "
              f"(uniform {res['uniform_nll']:.3f}, markov1 4.429) | temp {T_opt:.0f} "
              f"= {T_opt/cfg['T']:.3f}xT | |v|max/T {res['vote_over_T']:.2f} "
              f"({r['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
