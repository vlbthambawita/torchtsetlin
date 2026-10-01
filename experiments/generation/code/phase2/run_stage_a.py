"""Stage A / gate G2: can a block codebook represent 12-lead ECG well enough to generate from?

Gate: per-lead held-out SNR >= 20 dB and seam ratio <= 2 (PLAN.md sec.3.3).
No Tsetlin machine is trained here — reconstruction bounds everything downstream, so it is
resolved first and cheaply.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import codebook as CB
import ecg_data as E

OUT = Path(__file__).resolve().parents[2] / "results" / "phase2"


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(OUT.parents[2]), text=True).strip()
    except Exception:
        return "unknown"


def write_record(arm, config, result, wall):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"arm": arm, "config": config, "result": result, "wall_s": round(wall, 2),
           "git_sha": git_sha(), "host": platform.node(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (OUT / f"{arm}__{int(time.time()*1000)}.json").write_text(json.dumps(rec, indent=2))
    return rec


def run_one(data, channels, B, K, aligned, quant, seed, n_eval, highpass=False,
            overlap=False):
    t0 = time.time()
    xtr = data["train"][0]
    xte = data["test"][0][:n_eval]
    idx = E.INDEP_IDX if channels == 8 else list(range(12))

    if highpass:
        xtr = CB.highpass(xtr)
        xte = CB.highpass(xte)
    sc = CB.GlobalScaler().fit(xtr[:, idx])
    ztr, zte = sc.transform(xtr[:, idx]), sc.transform(xte[:, idx])

    off_tr = CB.grid_offsets(xtr, B, aligned)
    off_te = CB.grid_offsets(xte, B, aligned)
    if overlap:
        win = 2 * B
        btr, _ = CB.to_blocks_ov(ztr, B, win)
        bte, nb_te = CB.to_blocks_ov(zte, B, win)
    else:
        btr, _ = CB.to_blocks(ztr, B, off_tr)
        bte, nb_te = CB.to_blocks(zte, B, off_te)

    if quant == "pq":
        cb = CB.ProductCodebook(K, n_sub=4, seed=seed)
    elif quant.startswith("rvq"):
        cb = CB.ResidualCodebook(K, n_stages=int(quant[3:]), seed=seed)
    else:
        cb = CB.KMeansCodebook(K, seed=seed)
    cb = cb.fit(btr)
    rec = cb.decode(cb.encode(bte))

    zhat = (CB.from_blocks_ov(rec, xte.shape[0], nb_te, B, 2 * B) if overlap
            else CB.from_blocks(rec, xte.shape[0], nb_te))
    xhat = sc.inverse(zhat)
    if channels == 8:
        xhat12 = E.derive_full12(xhat)                 # 4 leads derived, not modelled
    else:
        xhat12 = xhat
    # compare on the same trimmed window the blocks cover
    T = xhat12.shape[-1]
    if overlap:
        real12 = xte[:, :, :T]
    else:
        real12 = np.stack([xte[i, :, int(off_te[i]):int(off_te[i]) + T]
                           for i in range(xte.shape[0])])

    res = CB.evaluate(real12, xhat12, B)
    n_sym = 4 if quant == "pq" else (int(quant[3:]) if quant.startswith("rvq") else 1)
    res["effective_bits_per_token"] = float(np.log2(K) * n_sym)
    res["tokens_per_10s"] = int(T // B)
    res["bits_per_sample"] = float(np.log2(K) * n_sym * (T // B) / (T * channels))
    cfg = dict(channels=channels, B=B, K=K, aligned=bool(aligned), quantiser=quant,
               seed=seed, n_train=int(xtr.shape[0]), n_eval=int(xte.shape[0]),
               highpass=bool(highpass), overlap=bool(overlap))
    arm = (f"cb-c{channels}-B{B}-K{K}-{'ral' if aligned else 'grid'}-{quant}"
           f"{'-hp' if highpass else ''}{'-ov' if overlap else ''}")
    r = write_record(arm, cfg, res, time.time() - t0)
    print(f"[{arm}] SNR mean {res['snr_db_mean']:.1f} dB / min-lead "
          f"{res['snr_db_min_lead']:.1f} dB | seam {res['seam_ratio']:.2f} | "
          f"QRS err {res['qrs_amp_rel_err']:.3f} | {res['tokens_per_10s']} tok/10s "
          f"({r['wall_s']:.0f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, nargs="+", default=[8, 12])
    ap.add_argument("--B", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--K", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--aligned", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--quant", nargs="+", default=["kmeans"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--highpass", type=int, nargs="+", default=[0])
    ap.add_argument("--overlap", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    data = E.load(n_train=a.n_train, n_val=500, n_test=500)
    for ch, B, K, al, q, hp, ov in itertools.product(a.channels, a.B, a.K, a.aligned,
                                                     a.quant, a.highpass, a.overlap):
        run_one(data, ch, B, K, al, q, a.seed, a.n_eval, highpass=bool(hp), overlap=bool(ov))


if __name__ == "__main__":
    main()
