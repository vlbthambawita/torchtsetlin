"""Raw images -> Boolean head inputs for every arm, cache and attention diagnostics (spec §7, §9).

Modes: ``qk`` (N0), ``uniform`` (A0), ``self_only`` (A1), ``shuffled_keys`` (A3),
``direct`` (A2), ``image_flat`` (B0), ``image_conv`` (B1). ``uniform``/``self_only`` never call
the encoders (their deployment cost must not include Q/K, spec §9).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor

from . import attention as A
from .common import ART, sha256_array
from .tokenization import encode_images

ATTENTION_MODES = ("qk", "uniform", "self_only", "shuffled_keys")
NEEDS_ENCODERS = ("qk", "shuffled_keys", "direct")
N_FEATURES = {"qk": 3424, "uniform": 3424, "self_only": 3424, "shuffled_keys": 3424, "direct": 1744,
              "image_flat": 784, "image_conv": 784}
CODE_FILES = ("attention.py", "features.py", "tokenization.py")


def code_version() -> str:
    here = Path(__file__).parent
    return hashlib.sha256(b"".join((here / f).read_bytes() for f in CODE_FILES)).hexdigest()[:16]


@torch.no_grad()
def transform_batch(u8: Tensor, mode: str, q=None, k=None, temperature: float = 1.0,
                    intermediates: bool = False):
    bits, tokens, active = encode_images(u8)
    B = u8.shape[0]
    if mode == "image_flat":
        return bits.reshape(B, -1)
    if mode == "image_conv":
        return bits.unsqueeze(1)
    out: Dict[str, Tensor] = {"active": active}
    if mode in NEEDS_ENCODERS:
        flat = tokens.reshape(B * 16, -1)
        uq = q.unit_scores(flat).view(B, 16, -1)
        uk = k.unit_scores(flat).view(B, 16, -1)
        out.update(uq=uq, uk=uk)
        if mode == "direct":
            f = A.direct_features(bits, uq, uk, active)
            return (f, out) if intermediates else f
        qn, kn = A.normalize_votes(uq), A.normalize_votes(uk)
        alpha = A.attention_weights(active, mode, qn, kn, temperature)
    elif mode in ("uniform", "self_only"):
        alpha = A.attention_weights(active, mode)
    else:
        raise ValueError(mode)
    if mode == "uniform":
        # exact: integer sender counts / number of active senders (D19). Summing (1/m)*bit
        # rounds differently across devices exactly at the thresholds, where k/m often lands.
        allowed = (active.unsqueeze(-1) & active.unsqueeze(-2)).float()
        ctx = (allowed @ tokens.float()) / allowed.sum(-1, keepdim=True).clamp_min(1.0)
    else:
        ctx = A.aggregate(alpha, tokens.float())
    f = A.context_features(bits, A.quantize(ctx))
    if intermediates:
        out.update(alpha=alpha, context=ctx)
        return f, out
    return f


@torch.no_grad()
def transform(images: np.ndarray, mode: str, q=None, k=None, temperature: float = 1.0,
              device: str = "cpu", batch: int = 2048) -> Tensor:
    outs = []
    for i in range(0, len(images), batch):
        u8 = torch.as_tensor(images[i:i + batch]).to(device)
        outs.append(transform_batch(u8, mode, q, k, temperature))
    return torch.cat(outs, 0)


def cache_key(ids: np.ndarray, mode: str, temperature: float, q=None, k=None) -> str:
    parts = {"ids": sha256_array(np.asarray(ids, dtype=np.int64)), "pre": "u>=128;7x7s7;grid_thermometer;active>=1",
             "mode": mode, "temperature": float(temperature) if mode in ("qk", "shuffled_keys") else None,
             "value_map": "pixel_and_position_bits", "thresholds": list(A.CONTEXT_THRESHOLDS),
             "direct_thresholds": list(A.DIRECT_THRESHOLDS), "mask": "a_i*a_j;self", "code": code_version(),
             "q": q.state_hash() if (q is not None and mode in NEEDS_ENCODERS) else None,
             "k": k.state_hash() if (k is not None and mode in NEEDS_ENCODERS) else None}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:24]


def cached_transform(images: np.ndarray, ids: np.ndarray, mode: str, q=None, k=None,
                     temperature: float = 1.0, device: str = "cpu", use_cache: bool = True):
    """Transform with an on-disk packed-bit cache. Returns ``(features bool on device, key, hit)``."""
    if mode in ("image_flat", "image_conv"):
        return transform(images, mode, device=device), None, False
    key = cache_key(ids, mode, temperature, q, k)
    path = ART / "features" / f"{key}.npz"
    nf = N_FEATURES[mode]
    if use_cache and path.exists():
        z = np.load(path)
        f = np.unpackbits(z["bits"], axis=1, count=nf).astype(bool)
        return torch.as_tensor(f).to(device), key, True
    f = transform(images, mode, q, k, temperature, device=device)
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + f".tmp{os.getpid()}.npz"
        np.savez(tmp, bits=np.packbits(f.cpu().numpy(), axis=1))
        os.replace(tmp, path)
    return f, key, False


# ---------------------------------------------------------------------- diagnostics
@torch.no_grad()
def attention_diagnostics(images: np.ndarray, mode: str, q, k, temperature: float, device: str,
                          batch: int = 2048) -> Dict:
    """Spec §16 attention diagnostics over a set of images."""
    ent, diag, mx, dev_u, n_rows = [], [], [], [], 0
    ent_multi, diag_multi = [], []
    sum_a = torch.zeros(16, 16, device=device, dtype=torch.float64)
    sum_a2 = torch.zeros(16, 16, device=device, dtype=torch.float64)
    cnt = torch.zeros(16, 16, device=device, dtype=torch.float64)
    uq_all, uk_all, act_all = [], [], []
    n_single = 0
    for i in range(0, len(images), batch):
        u8 = torch.as_tensor(images[i:i + batch]).to(device)
        _, it = transform_batch(u8, mode, q, k, temperature, intermediates=True)
        a, act = it["alpha"].double(), it["active"]
        allowed = act.unsqueeze(-1) & act.unsqueeze(-2)
        rows = act  # (B,16)
        n_act = act.sum(1)
        n_single += int((n_act == 1).sum())
        plogp = torch.where(a > 0, a * torch.log(a), torch.zeros_like(a)).sum(-1)
        e = -plogp
        dg = torch.diagonal(a, dim1=-2, dim2=-1)
        m = a.amax(-1)
        uni = allowed.double() / allowed.sum(-1, keepdim=True).clamp_min(1)
        tv = 0.5 * (a - uni).abs().sum(-1)
        multi = rows & (n_act > 1).unsqueeze(1)
        ent.append(e[rows]); diag.append(dg[rows]); mx.append(m[rows]); dev_u.append(tv[rows])
        ent_multi.append(e[multi]); diag_multi.append(dg[multi])
        n_rows += int(rows.sum())
        w = allowed.double()
        sum_a += (a * w).sum(0); sum_a2 += (a * a * w).sum(0); cnt += w.sum(0)
        if "uq" in it:
            uq_all.append(it["uq"][act]); uk_all.append(it["uk"][act])
        act_all.append(act)
    cat = lambda xs: torch.cat(xs).double() if xs else torch.zeros(0, dtype=torch.float64)
    E, Dg, Mx, TV, Em, Dm = map(cat, (ent, diag, mx, dev_u, ent_multi, diag_multi))
    var = (sum_a2 / cnt.clamp_min(1) - (sum_a / cnt.clamp_min(1)) ** 2).clamp_min(0)
    out = {"n_images": len(images), "n_valid_rows": n_rows, "images_with_one_active_token": n_single,
           "mean_entropy": float(E.mean()), "mean_effective_senders": float(torch.exp(E).mean()),
           "mean_self_mass": float(Dg.mean()), "mean_max_weight": float(Mx.mean()),
           "mean_tv_from_uniform": float(TV.mean()),
           "multi_token_rows": {"mean_entropy": float(Em.mean()) if len(Em) else None,
                                "mean_self_mass": float(Dm.mean()) if len(Dm) else None},
           "cross_image_sd_of_alpha": float(var.sqrt()[cnt > 1].mean()),
           "mean_active_tokens": float(torch.cat(act_all).sum(1).double().mean())}
    if uq_all:
        for name, U in (("q", torch.cat(uq_all)), ("k", torch.cat(uk_all))):
            sd = U.std(-1, unbiased=False)
            out[f"{name}_per_coord_var"] = U.var(0, unbiased=False).tolist()
            out[f"{name}_saturation"] = float((U.abs() >= 1).double().mean())
            out[f"{name}_zero_variance_rate"] = float((sd == 0).double().mean())
            out[f"{name}_mean_token_sd"] = float(sd.mean())
    return out


def constant_column_fraction(f: Tensor, start: int = 784) -> float:
    """Fraction of context bits that are constant over the given (training) features."""
    c = f[:, start:]
    if c.shape[1] == 0:
        return 0.0
    on = c.any(0)
    off = (~c).any(0)
    return float((~(on & off)).double().mean())
