"""PTB-XL loading with the official patient-disjoint folds, plus lead algebra.

PTB-XL v1.0.1 at /work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1
21 837 records, 18 885 patients, 10 s at 500 Hz, 12 leads in mV.
Folds 1-8 train, 9 validation, 10 test (folds 9-10 are the human-validated ones).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1")
CACHE = Path(__file__).resolve().parents[2] / "results" / "phase2" / "cache"
FS = 500
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# Only these 8 are independent; the other four are exact linear combinations (verified on
# PTB-XL to 0.3-0.9 % relative error, i.e. quantisation noise). VALIDITY.md P3.
INDEP = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
INDEP_IDX = [LEADS.index(l) for l in INDEP]


def derive_full12(x8):
    """(..., 8, T) independent leads -> (..., 12, T) with III/aVR/aVL/aVF derived."""
    I, II = x8[..., 0, :], x8[..., 1, :]
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    chest = x8[..., 2:, :]
    return np.concatenate([I[..., None, :], II[..., None, :], III[..., None, :],
                           aVR[..., None, :], aVL[..., None, :], aVF[..., None, :],
                           chest], axis=-2)


def lead_identity_residual(x12):
    """Relative mean-abs error of the four derived leads. A free validity metric on generated
    output: it must not exceed what the codebook's own reconstruction error explains."""
    I, II = x12[..., 0, :], x12[..., 1, :]
    out = {}
    for name, idx, pred in (("III", 2, II - I), ("aVR", 3, -(I + II) / 2),
                            ("aVL", 4, I - II / 2), ("aVF", 5, II - I / 2)):
        num = np.abs(x12[..., idx, :] - pred).mean()
        den = np.abs(pred).mean() + 1e-9
        out[name] = float(num / den)
    return out


def _meta():
    rows = list(csv.DictReader(open(ROOT / "ptbxl_database.csv")))
    for r in rows:
        r["strat_fold"] = int(r["strat_fold"])
    return rows


def load(n_train=4000, n_val=500, n_test=500, seed=0, verbose=True):
    """Patient-disjoint subsets as float32 arrays (N, 12, 5000) in mV."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"{n_train}_{n_val}_{n_test}_{seed}"
    f = CACHE / f"ptbxl_{tag}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        if verbose:
            print(f"  loaded cache {f.name}")
        return {k: (d[f"{k}_x"], d[f"{k}_fold"]) for k in ("train", "val", "test")}

    import wfdb
    rows = _meta()
    rng = np.random.default_rng(seed)
    splits = {"train": [r for r in rows if r["strat_fold"] <= 8],
              "val": [r for r in rows if r["strat_fold"] == 9],
              "test": [r for r in rows if r["strat_fold"] == 10]}
    want = {"train": n_train, "val": n_val, "test": n_test}
    out = {}
    for k, rs in splits.items():
        idx = rng.permutation(len(rs))[: want[k]]
        sel = [rs[i] for i in idx]
        sigs = np.zeros((len(sel), 12, 5000), dtype=np.float32)
        for i, r in enumerate(sel):
            rec = wfdb.rdrecord(str(ROOT / r["filename_hr"]))
            sigs[i] = rec.p_signal.T.astype(np.float32)
            if verbose and i % 1000 == 0:
                print(f"    {k} {i}/{len(sel)}", flush=True)
        folds = np.array([r["strat_fold"] for r in sel], dtype=np.int16)
        out[k] = (sigs, folds)
    np.savez(f, **{f"{k}_x": v[0] for k, v in out.items()},
             **{f"{k}_fold": v[1] for k, v in out.items()})
    if verbose:
        print(f"  cached -> {f.name}")
    return out


# --------------------------------------------------------------------------- R peaks
def detect_r_peaks(lead2, fs=FS):
    """Pan-Tompkins-flavoured detector on one lead. (T,) -> peak sample indices."""
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(3, [5 / (fs / 2), 25 / (fs / 2)], btype="band")
    y = filtfilt(b, a, lead2)
    d = np.diff(y, prepend=y[0]) ** 2
    w = int(0.12 * fs)
    e = np.convolve(d, np.ones(w) / w, mode="same")
    thr = np.percentile(e, 98) * 0.35
    pk, _ = find_peaks(e, height=thr, distance=int(0.28 * fs))
    return pk


def rr_intervals(peaks, fs=FS):
    return np.diff(peaks) / fs if len(peaks) > 1 else np.array([])
