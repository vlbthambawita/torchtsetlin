"""``TMQKAttention``: the complete raw-image -> digit pipeline in one checkpoint (spec §7, §17).

Holds the frozen Q/K encoders (absent for uniform/self_only/image modes), the head TM and every
convention needed to reproduce predictions. ``mode`` selects the arm's feature map; the class is
also used for the baselines (``image_flat``, ``image_conv``) so that every arm is timed through
the same raw-image path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from . import attention as A
from .common import git_sha
from .features import NEEDS_ENCODERS, transform, transform_batch
from .tm_adapter import TMAdapter

CONVENTIONS = {
    "class_order": list(range(10)), "pixel_rule": "u >= 128 (uint8)", "tokens": "7x7 stride 7, i = 4r + c, row-major pixels",
    "position_bits": "[r>=1,r>=2,r>=3,c>=1,c>=2,c>=3]", "active": "any of 49 pixel bits",
    "mask": "M_ij = a_i a_j, self allowed", "alpha": "rows = receivers, columns = senders",
    "context_thresholds": list(A.CONTEXT_THRESHOLDS), "direct_thresholds": list(A.DIRECT_THRESHOLDS),
    "feature_order": "image bits (784, row-major) then context (token, feature, threshold)",
    "argmax_ties": "lowest index",
}


class TMQKAttention:
    def __init__(self, head: TMAdapter, mode: str, q: Optional[TMAdapter] = None, k: Optional[TMAdapter] = None,
                 temperature: float = 1.0, meta: Optional[Dict] = None) -> None:
        if mode in NEEDS_ENCODERS and (q is None or k is None):
            raise ValueError(f"mode {mode} needs both encoders")
        self.head, self.mode, self.temperature = head, mode, float(temperature)
        self.q, self.k = (q, k) if mode in NEEDS_ENCODERS else (None, None)
        self.meta = dict(meta or {})

    @property
    def device(self) -> str:
        return self.head.device

    @torch.no_grad()
    def features(self, u8: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        m = mode or self.mode
        return transform_batch(u8.to(self.device), m, self.q, self.k, self.temperature)

    @torch.no_grad()
    def votes_images(self, u8: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        return self.head.votes(self.features(u8, mode))

    def predict_images(self, u8: torch.Tensor) -> torch.Tensor:
        return self.votes_images(u8).argmax(1)

    @torch.no_grad()
    def votes_array(self, images: np.ndarray, mode: Optional[str] = None, batch: int = 2048) -> torch.Tensor:
        out = [self.votes_images(torch.as_tensor(images[i:i + batch]), mode) for i in range(0, len(images), batch)]
        return torch.cat(out, 0)

    def n_automata(self) -> Dict[str, int]:
        d = {"head": self.head.n_automata()}
        if self.q is not None:
            d["q"], d["k"] = self.q.n_automata(), self.k.n_automata()
        d["total"] = sum(d.values())
        return d

    def stored_bytes(self) -> int:
        return sum(a.state_manifest()["stored_bytes"] for a in (self.head, self.q, self.k) if a is not None)

    def save(self, path) -> int:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        obj = {"mode": self.mode, "temperature": self.temperature, "head": self.head.state_dict(),
               "q": self.q.state_dict() if self.q else None, "k": self.k.state_dict() if self.k else None,
               "conventions": CONVENTIONS, "meta": self.meta, "git_sha": git_sha()}
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return os.path.getsize(path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TMQKAttention":
        o = torch.load(path, map_location="cpu", weights_only=False)
        q = TMAdapter.from_state(o["q"], device) if o["q"] else None
        k = TMAdapter.from_state(o["k"], device) if o["k"] else None
        return cls(TMAdapter.from_state(o["head"], device), o["mode"], q, k, o["temperature"], o["meta"])
