"""`ecg-roundtrip`: the Stage B ceiling.

Encode real held-out ECG through the chosen codec and decode it. Nothing the TM generates can
beat this, so it is the first thing to measure at the chosen operating point — and it answers
the risk attached to that choice: does QRS morphology and rhythm survive at ~8 dB?
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import ks_2samp

import codebook as CB
import ecg_data as E
from run_stage_a import write_record


def rhythm_metrics(x_real, x_hat, fs=E.FS):
    """Can you still find the beats, and are the RR statistics preserved?"""
    rr_r, rr_h, n_r, n_h, matched, tol = [], [], 0, 0, 0, int(0.05 * fs)
    for i in range(x_real.shape[0]):
        pr = E.detect_r_peaks(x_real[i, 1], fs)
        ph = E.detect_r_peaks(x_hat[i, 1], fs)
        n_r += len(pr); n_h += len(ph)
        if len(pr) > 1:
            rr_r.extend(E.rr_intervals(pr, fs))
        if len(ph) > 1:
            rr_h.extend(E.rr_intervals(ph, fs))
        for p in pr:                                   # recovered within +-50 ms?
            if len(ph) and np.min(np.abs(ph - p)) <= tol:
                matched += 1
    rr_r, rr_h = np.asarray(rr_r), np.asarray(rr_h)
    out = {
        "r_peaks_real": int(n_r), "r_peaks_recon": int(n_h),
        "r_peak_recall": float(matched / max(1, n_r)),
        "rr_mean_real": float(rr_r.mean()) if len(rr_r) else float("nan"),
        "rr_mean_recon": float(rr_h.mean()) if len(rr_h) else float("nan"),
        "rr_std_real": float(rr_r.std()) if len(rr_r) else float("nan"),
        "rr_std_recon": float(rr_h.std()) if len(rr_h) else float("nan"),
    }
    out["rr_ks"] = float(ks_2samp(rr_r, rr_h).statistic) if len(rr_r) and len(rr_h) else float("nan")
    return out


def build(data, B, K, M, n_eval, seed=0, n_train=None):
    xtr = data["train"][0] if n_train is None else data["train"][0][:n_train]
    xte = data["test"][0][:n_eval]
    xtr, xte = CB.highpass(xtr), CB.highpass(xte)
    sc = CB.GlobalScaler().fit(xtr[:, E.INDEP_IDX])
    ztr, zte = sc.transform(xtr[:, E.INDEP_IDX]), sc.transform(xte[:, E.INDEP_IDX])
    off = np.zeros(ztr.shape[0], dtype=int)
    btr, _ = CB.to_blocks(ztr, B, off)
    bte, nb = CB.to_blocks(zte, B, np.zeros(zte.shape[0], dtype=int))
    cb = CB.ResidualCodebook(K, n_stages=M, seed=seed).fit(btr)
    return cb, sc, xte, zte, bte, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=25)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=4000)
    a = ap.parse_args()
    t0 = time.time()

    data = E.load(n_train=a.n_train, n_val=500, n_test=500, verbose=False)
    cb, sc, xte, zte, bte, nb = build(data, a.B, a.K, a.M, a.n_eval)
    codes = cb.encode(bte)
    rec = cb.decode(codes)
    xhat8 = sc.inverse(CB.from_blocks(rec, xte.shape[0], nb))
    xhat12 = E.derive_full12(xhat8)
    T = xhat12.shape[-1]
    real12 = xte[:, :, :T]

    res = CB.evaluate(real12, xhat12, a.B)
    res.update(rhythm_metrics(real12, xhat12))
    res["token_usage"] = [int(len(np.unique(codes[:, m]))) for m in range(a.M)]
    res["tokens_per_10s"] = int(T // a.B)
    cfg = dict(arm="ecg-roundtrip", B=a.B, K=a.K, M=a.M, channels=8, highpass=True,
               n_eval=int(xte.shape[0]), n_train=a.n_train)
    r = write_record("ecg-roundtrip", cfg, res, time.time() - t0)
    print(f"[ecg-roundtrip B={a.B} M={a.M}] SNR {res['snr_db_mean']:.1f} dB "
          f"(min lead {res['snr_db_min_lead']:.1f}) | QRS err {res['qrs_amp_rel_err']:.3f}\n"
          f"  R-peak recall {res['r_peak_recall']:.3f}  "
          f"({res['r_peaks_recon']} found vs {res['r_peaks_real']} real)\n"
          f"  RR mean {res['rr_mean_recon']:.3f}s vs real {res['rr_mean_real']:.3f}s | "
          f"RR KS {res['rr_ks']:.3f}\n"
          f"  codebook usage per stage: {res['token_usage']}  ({r['wall_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
