"""Stage A: weakly labelled patch streams and the frozen Q/K encoders (spec §8)."""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from .common import seed_torch, stream
from .tm_adapter import TMAdapter
from .tokenization import TOKEN_DIM, encode_images

ENCODER_DEFAULTS = dict(M=100, T=50, s=5.0, n_states=128, epochs=10, per_image=4, batch_size=32)


def sample_patches(tokens: Tensor, active: Tensor, y: Tensor, count: int, rng: np.random.Generator
                   ) -> Tuple[Tensor, Tensor, Dict]:
    """``count`` active patches per image, without replacement when possible, else with
    replacement; all-blank images skipped. Labels = parent image digit (weak labels)."""
    act = active.cpu().numpy()
    n = act.sum(1)
    keep = np.flatnonzero(n > 0)
    act, n = act[keep], n[keep]
    keys = rng.random(act.shape)
    keys[~act] = 2.0
    order = np.argsort(keys, axis=1, kind="stable")
    r = (rng.random((len(keep), count)) * n[:, None]).astype(np.int64)  # with replacement
    with_rep = np.take_along_axis(order, r, axis=1)
    pick = np.where((n >= count)[:, None], order[:, :count], with_rep)
    repeated = int(sum(count - len(set(row)) for row in pick[n < count]))
    keep_t = torch.as_tensor(keep, device=tokens.device)
    pick_t = torch.as_tensor(pick, device=tokens.device)
    X = torch.gather(tokens[keep_t], 1, pick_t.unsqueeze(-1).expand(-1, -1, TOKEN_DIM)).reshape(-1, TOKEN_DIM)
    Y = y[keep_t].repeat_interleave(count)
    return X, Y, {"examples": int(X.shape[0]), "blank_images_skipped": int(len(active) - len(keep)),
                  "images_with_fewer_than_count": int((n < count).sum()), "repeated_samples": repeated}


def fit_encoders(images: np.ndarray, labels: np.ndarray, seed: int, device: str, cfg: Dict = None,
                 log=print) -> Tuple[TMAdapter, TMAdapter, Dict]:
    cfg = {**ENCODER_DEFAULTS, **(cfg or {})}
    u8 = torch.as_tensor(images).to(device)
    y = torch.as_tensor(labels).to(device)
    _, tokens, active = encode_images(u8)
    info: Dict = {"config": cfg, "epochs": {"q": [], "k": []}, "time_s": {}}
    encs = {}
    for name in ("q", "k"):
        tseed = seed_torch(seed, f"{name}-init")
        enc = TMAdapter(TOKEN_DIM, cfg["M"], cfg["T"], cfg["s"], n_states=cfg["n_states"],
                        feedback_mode=cfg.get("feedback_mode", "batch"), device=device)
        t0 = time.time()
        for ep in range(cfg["epochs"]):
            X, Y, st = sample_patches(tokens, active, y, cfg["per_image"], stream(seed, f"{name}-patches", ep))
            enc.fit_epoch(X, Y, shuffle=True, batch_size=cfg["batch_size"])
            st["state_sha256"] = enc.state_hash()
            info["epochs"][name].append(st)
            log(f"[encoder {name}] epoch {ep + 1}/{cfg['epochs']} examples={st['examples']} "
                f"({time.time() - t0:.0f}s)")
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        info["time_s"][name] = time.time() - t0
        info[f"{name}_torch_seed"] = tseed
        encs[name] = enc
    return encs["q"], encs["k"], info


VAL_PATCH_SEED = 271828


@torch.no_grad()
def encoder_diagnostics(q: TMAdapter, k: TMAdapter, images: np.ndarray, labels: np.ndarray, device: str) -> Dict:
    """Fixed validation patch sample (independent of the model seed): patch accuracy,
    score spread, zero-variance rate, clipping saturation. Diagnostics only (spec §8)."""
    u8 = torch.as_tensor(images).to(device)
    y = torch.as_tensor(labels).to(device)
    _, tokens, active = encode_images(u8)
    X, Y, st = sample_patches(tokens, active, y, 4, stream(VAL_PATCH_SEED, "val-patches"))
    out = {"sample": st}
    for name, enc in (("q", q), ("k", k)):
        raw = enc.class_scores(X)
        u = enc.unit_scores(X)
        if not torch.allclose(raw.clamp(-enc.T, enc.T) / enc.T, u):
            raise RuntimeError("clamped raw scores disagree with the library forward pass")
        sd = u.std(-1, unbiased=False)
        out[name] = {"patch_accuracy": float((u.argmax(1) == Y).double().mean()),
                     "per_class_patch_accuracy": [float((u.argmax(1)[Y == c] == c).double().mean()) for c in range(10)],
                     "mean_token_sd": float(sd.mean()), "zero_variance_rate": float((sd == 0).double().mean()),
                     "clip_saturation": float((raw.abs() >= enc.T).double().mean()),
                     "raw_mean": raw.mean(0).tolist(), "raw_sd": raw.std(0).tolist(),
                     "state": enc.state_manifest()}
    return out
