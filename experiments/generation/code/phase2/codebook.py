"""Stage A: block codebooks for 12-lead ECG, and the metrics that gate G2.

The generator cannot recover anything the codebook discards, so reconstruction quality is
measured FIRST and bounds everything downstream (PLAN.md §3.3 Stage A).
"""
from __future__ import annotations

import numpy as np

import ecg_data as E


# --------------------------------------------------------------------------- preprocessing
def highpass(x, fs=E.FS, cutoff=0.5):
    """Remove baseline wander. Standard ECG preprocessing, and a candidate fix for the
    first Stage-A failure: raw blocks are dominated by their DC level, so a plain VQ spends
    its capacity on offsets rather than morphology."""
    from scipy.signal import butter, filtfilt
    b, a = butter(3, cutoff / (fs / 2), btype="high")
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- scaling
class GlobalScaler:
    """One scale for all leads, fitted on train only.

    Deliberately NOT per-lead: per-lead scaling destroys the inter-lead amplitude geometry
    that the Einthoven/Goldberger identities live in, and k-means distance would then weight
    leads arbitrarily.
    """

    def fit(self, x):
        self.scale = float(np.percentile(np.abs(x), 99.5)) or 1.0
        return self

    def transform(self, x):
        return x / self.scale

    def inverse(self, z):
        return z * self.scale


# --------------------------------------------------------------------------- blocking
def grid_offsets(x, B, aligned):
    """Per-record start offset for the block grid. `aligned` puts R peaks at a consistent
    phase within their block, which is the VALIDITY.md P1 fix for phase smearing."""
    n = x.shape[0]
    if not aligned:
        return np.zeros(n, dtype=int)
    off = np.zeros(n, dtype=int)
    target = B // 2
    for i in range(n):
        pk = E.detect_r_peaks(x[i, 1])
        if len(pk) == 0:
            continue
        phase = int(np.median(pk % B))
        off[i] = (phase - target) % B
    return off


def to_blocks(x, B, offsets):
    """(N, C, T) -> (M, C, B) plus an index so blocks can be put back."""
    N, C, T = x.shape
    nb = (T - int(offsets.max())) // B
    out = np.empty((N, nb, C, B), dtype=np.float32)
    for i in range(N):
        s = int(offsets[i])
        out[i] = x[i, :, s:s + nb * B].reshape(C, nb, B).transpose(1, 0, 2)
    return out.reshape(N * nb, C, B), nb


def from_blocks(blocks, N, nb):
    C, B = blocks.shape[1], blocks.shape[2]
    return blocks.reshape(N, nb, C, B).transpose(0, 2, 1, 3).reshape(N, C, nb * B)


# --------------------------------------------------------------------------- quantiser
class KMeansCodebook:
    def __init__(self, K, seed=0, n_fit=200_000):
        self.K, self.seed, self.n_fit = K, seed, n_fit

    def fit(self, blocks):
        from sklearn.cluster import MiniBatchKMeans
        f = blocks.reshape(blocks.shape[0], -1)
        if f.shape[0] > self.n_fit:
            rng = np.random.default_rng(self.seed)
            f = f[rng.permutation(f.shape[0])[: self.n_fit]]
        self.km = MiniBatchKMeans(n_clusters=self.K, random_state=self.seed,
                                  batch_size=4096, n_init=3, max_iter=200).fit(f)
        self.shape = blocks.shape[1:]
        return self

    def encode(self, blocks):
        return self.km.predict(blocks.reshape(blocks.shape[0], -1)).astype(np.int32)

    def decode(self, codes):
        return self.km.cluster_centers_[codes].reshape(-1, *self.shape).astype(np.float32)


class ProductCodebook:
    """Split each block into `n_sub` lead-groups and quantise each independently.

    Gives a *compositional* token code (one sub-symbol per group), which is the encoding
    VALIDITY.md P2 argues the TM needs in order to generalise across similar morphologies
    instead of memorising exact n-grams.
    """

    def __init__(self, K, n_sub=4, seed=0, n_fit=200_000):
        self.K, self.n_sub, self.seed, self.n_fit = K, n_sub, seed, n_fit

    def fit(self, blocks):
        from sklearn.cluster import MiniBatchKMeans
        C = blocks.shape[1]
        self.groups = np.array_split(np.arange(C), self.n_sub)
        self.kms = []
        rng = np.random.default_rng(self.seed)
        for g in self.groups:
            f = blocks[:, g].reshape(blocks.shape[0], -1)
            if f.shape[0] > self.n_fit:
                f = f[rng.permutation(f.shape[0])[: self.n_fit]]
            self.kms.append(MiniBatchKMeans(n_clusters=self.K, random_state=self.seed,
                                            batch_size=4096, n_init=3, max_iter=200).fit(f))
        self.shape = blocks.shape[1:]
        return self

    def encode(self, blocks):
        return np.stack([km.predict(blocks[:, g].reshape(blocks.shape[0], -1))
                         for km, g in zip(self.kms, self.groups)], axis=1).astype(np.int32)

    def decode(self, codes):
        B = self.shape[1]
        out = np.empty((codes.shape[0], self.shape[0], B), dtype=np.float32)
        for j, (km, g) in enumerate(zip(self.kms, self.groups)):
            out[:, g] = km.cluster_centers_[codes[:, j]].reshape(-1, len(g), B)
        return out


# --------------------------------------------------------------------------- metrics
def snr_db(x, xh, axis=-1):
    num = (x ** 2).sum(axis=axis)
    den = ((x - xh) ** 2).sum(axis=axis) + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def seam_ratio(xh, B):
    """|first difference| at block joins vs within blocks.

    NOTE: this divides a MEAN at joins by a MEDIAN within blocks, and |dx| on ECG is heavy
    tailed, so the statistic's floor is mean/median ~ 4.6 on real data, NOT 1.0. Always read it
    against `seam_ratio` of the real signal (see `evaluate`, which reports the normalised
    version). The original <=2 gate in PLAN.md was set without computing this null and is not
    achievable by real ECG either.
    """
    d = np.abs(np.diff(xh, axis=-1))
    T = d.shape[-1]
    at = np.zeros(T, dtype=bool)
    at[B - 1::B] = True
    if at.sum() == 0 or (~at).sum() == 0:
        return float("nan")
    return float(d[..., at].mean() / (np.median(d[..., ~at]) + 1e-12))


def qrs_amplitude_error(x, xh, fs=E.FS):
    """Relative error of R-peak amplitude in lead II, at peaks found in the REAL signal."""
    errs = []
    for i in range(x.shape[0]):
        pk = E.detect_r_peaks(x[i, 1], fs)
        if len(pk) == 0:
            continue
        a, b = x[i, 1, pk], xh[i, 1, pk]
        errs.append(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9))
    return float(np.mean(errs)) if errs else float("nan")


def evaluate(x_real12, x_hat12, B):
    """Everything gate G2 needs, in physical units (mV)."""
    per_lead = snr_db(x_real12, x_hat12).mean(axis=0)     # (12,)
    seam_hat = seam_ratio(x_hat12, B)
    seam_real = seam_ratio(x_real12, B)                   # the metric's own null
    return {
        "seam_ratio_real": seam_real,
        "seam_excess": float(seam_hat / (seam_real + 1e-9)),   # 1.0 = indistinguishable
        "snr_db_mean": float(per_lead.mean()),
        "snr_db_min_lead": float(per_lead.min()),
        "snr_db_per_lead": [round(float(v), 2) for v in per_lead],
        "rmse_mv": float(np.sqrt(((x_real12 - x_hat12) ** 2).mean())),
        "seam_ratio": seam_ratio(x_hat12, B),
        "qrs_amp_rel_err": qrs_amplitude_error(x_real12, x_hat12),
        "lead_identity_residual": {k: round(v, 4)
                                   for k, v in E.lead_identity_residual(x_hat12).items()},
    }


class ResidualCodebook:
    """Multi-stage (residual) VQ: each stage quantises what the previous stage left over.

    Motivation, measured in Stage A: a single VQ has log2(K) bits to locate a point on a
    manifold with 48-130 significant principal directions, so it cannot approach 20 dB at any
    feasible K. Residual stages spend M*log2(K) bits and reduce error geometrically. The token
    is then a tuple of M sub-symbols, which is also the *compositional* encoding VALIDITY.md P2
    argues the Tsetlin machine needs.
    """

    def __init__(self, K, n_stages=8, seed=0, n_fit=100_000):
        self.K, self.M, self.seed, self.n_fit = K, n_stages, seed, n_fit

    def fit(self, blocks):
        from sklearn.cluster import MiniBatchKMeans
        rng = np.random.default_rng(self.seed)
        f = blocks.reshape(blocks.shape[0], -1)
        if f.shape[0] > self.n_fit:
            f = f[rng.permutation(f.shape[0])[: self.n_fit]]
        self.kms, res = [], f.copy()
        for m in range(self.M):
            km = MiniBatchKMeans(n_clusters=self.K, random_state=self.seed + m,
                                 batch_size=4096, n_init=3, max_iter=100).fit(res)
            self.kms.append(km)
            res = res - km.cluster_centers_[km.predict(res)]
        self.shape = blocks.shape[1:]
        return self

    def encode(self, blocks):
        res = blocks.reshape(blocks.shape[0], -1).copy()
        codes = []
        for km in self.kms:
            c = km.predict(res)
            codes.append(c)
            res = res - km.cluster_centers_[c]
        return np.stack(codes, axis=1).astype(np.int32)

    def decode(self, codes):
        out = sum(km.cluster_centers_[codes[:, m]] for m, km in enumerate(self.kms))
        return out.reshape(-1, *self.shape).astype(np.float32)


# --------------------------------------------------------------------------- overlap-add
def to_blocks_ov(x, stride, win, offsets=None):
    """Overlapping blocks: length `win`, hop `stride`. (N,C,T) -> (M,C,win), n_blocks."""
    N, C, T = x.shape
    nb = (T - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(nb)[:, None]      # (nb, win)
    out = x[:, :, idx]                                                   # (N, C, nb, win)
    return out.transpose(0, 2, 1, 3).reshape(N * nb, C, win), nb


def from_blocks_ov(blocks, N, nb, stride, win):
    """Overlap-add with a Hann synthesis window, normalised by the window sum.

    Independently quantised adjacent blocks disagree at their join; crossfading the overlap
    is the standard fix and is what VALIDITY.md P1(c) proposed for the seam problem.
    """
    C = blocks.shape[1]
    T = stride * (nb - 1) + win
    b = blocks.reshape(N, nb, C, win)
    w = np.hanning(win + 2)[1:-1].astype(np.float32)
    acc = np.zeros((N, C, T), dtype=np.float32)
    nrm = np.zeros(T, dtype=np.float32)
    for j in range(nb):
        s = j * stride
        acc[:, :, s:s + win] += b[:, j] * w
        nrm[s:s + win] += w
    return acc / np.maximum(nrm, 1e-6)
