"""float64 loop reference of spec §7 (tests only)."""

from __future__ import annotations

import math

import numpy as np


def normalize(u: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(u, dtype=np.float64)
    for idx in np.ndindex(u.shape[:-1]):
        v = u[idx].astype(np.float64)
        c = v - v.mean()
        out[idx] = c / math.sqrt((c * c).mean() + eps)
    return out


def shuffle_keys(k: np.ndarray, active: np.ndarray) -> np.ndarray:
    out = k.copy()
    for b in range(k.shape[0]):
        slots = [j for j in range(k.shape[1]) if active[b, j]]
        m = len(slots)
        for r, j in enumerate(slots):
            out[b, j] = k[b, slots[(r + 1) % m]]
    return out


def attention(q, k, active, temperature=1.0, mode="qk"):
    B, n = active.shape
    alpha = np.zeros((B, n, n), dtype=np.float64)
    if mode == "shuffled_keys":
        k = shuffle_keys(k, active)
    for b in range(B):
        for i in range(n):
            if not active[b, i]:
                continue
            J = [j for j in range(n) if active[b, j]]
            if mode == "uniform":
                for j in J:
                    alpha[b, i, j] = 1.0 / len(J)
                continue
            if mode == "self_only":
                alpha[b, i, i] = 1.0
                continue
            d = q.shape[-1]
            s = {j: float(np.dot(q[b, i], k[b, j])) / (math.sqrt(d) * temperature) for j in J}
            mx = max(s.values())
            e = {j: math.exp(s[j] - mx) for j in J}
            z = sum(e.values())
            for j in J:
                alpha[b, i, j] = e[j] / z
    return alpha


def context(alpha, values):
    B, n, _ = alpha.shape
    out = np.zeros((B, n, values.shape[-1]), dtype=np.float64)
    for b in range(B):
        for i in range(n):
            for j in range(n):
                out[b, i] += alpha[b, i, j] * values[b, j]
    return out


def quantize(x, thresholds=(0.25, 0.5, 0.75)):
    return np.stack([x >= t for t in thresholds], axis=-1)
