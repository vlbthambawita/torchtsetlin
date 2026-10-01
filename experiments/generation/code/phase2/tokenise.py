"""Tokenise PTB-XL at the chosen Stage B operating point and cache it.

Operating point (user decision, 2026-09-24): 8 independent leads, high-passed, B=25,
K=1024, M=8 residual stages -> 200 tokens per 10 s, each an 8-tuple of 1024-way sub-symbols.
Roundtrip ceiling at this point: R-peak recall 0.971, RR KS 0.029, SNR 14.1 dB.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

import codebook as CB
import ecg_data as E

CACHE = Path(__file__).resolve().parents[2] / "results" / "phase2" / "cache"


def build_and_tokenise(B=25, K=1024, M=8, n_train=4000, seed=0):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"tokens_B{B}_K{K}_M{M}_n{n_train}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)

    data = E.load(n_train=n_train, n_val=500, n_test=500, verbose=False)
    xtr = CB.highpass(data["train"][0])
    sc = CB.GlobalScaler().fit(xtr[:, E.INDEP_IDX])
    btr, _ = CB.to_blocks(sc.transform(xtr[:, E.INDEP_IDX]), B,
                          np.zeros(xtr.shape[0], dtype=int))
    t0 = time.time()
    cb = CB.ResidualCodebook(K, n_stages=M, seed=seed).fit(btr)
    print(f"  codebook fitted in {time.time()-t0:.0f}s", flush=True)

    out = {"B": B, "K": K, "M": M, "scaler_scale": sc.scale}
    for split in ("train", "val", "test"):
        x = CB.highpass(data[split][0])
        z = sc.transform(x[:, E.INDEP_IDX])
        blk, nb = CB.to_blocks(z, B, np.zeros(z.shape[0], dtype=int))
        codes = cb.encode(blk).reshape(x.shape[0], nb, M)      # (N, 200, 8)
        out[split] = codes.astype(np.int16)
        out[f"{split}_raw"] = x[:, :, : nb * B]
        print(f"  {split}: {codes.shape}", flush=True)
    out["codebook"] = cb
    out["n_tokens"] = out["train"].shape[1]
    with open(f, "wb") as fh:
        pickle.dump(out, fh)
    print(f"  cached -> {f.name}")
    return out


def decode_tokens(tok, codes):
    """(N, n_tok, M) sub-symbols -> (N, 12, T) mV, with 4 leads derived."""
    cb, B, M = tok["codebook"], tok["B"], tok["M"]
    N, nb, _ = codes.shape
    blocks = cb.decode(codes.reshape(-1, M).astype(np.int32))
    z = CB.from_blocks(blocks, N, nb)
    return E.derive_full12(z * tok["scaler_scale"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=25)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=4000)
    a = ap.parse_args()
    t = build_and_tokenise(a.B, a.K, a.M, a.n_train)
    print(f"tokens/record = {t['n_tokens']}, sub-symbols = {t['M']}, alphabet = {t['K']}")
