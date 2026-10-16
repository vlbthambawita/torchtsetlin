"""Narrow adapter over ``torchtsetlin`` (spec §4, PLAN §2.1, D1).

Wraps ``tt.TsetlinMachine`` (flat) or ``tt.ConvTsetlinMachine`` without modifying the library.
``n_clauses`` is clauses **per class** (both polarities). ``votes`` are the library's clamped
votes (``[-T, T]``); ``class_scores`` are the raw signed sums used for diagnostics (D8).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

import torchtsetlin as tt

from .common import sha256_array

DEFAULTS = dict(n_states=128, weighted=False, feedback_mode="batch")


class TMAdapter:
    def __init__(self, n_features: Optional[int], M: int, T: int, s: float, *, kind: str = "flat",
                 n_classes: int = 10, n_states: int = 128, weighted: bool = False,
                 feedback_mode: str = "batch", patch_size: int = 7, stride: int = 1,
                 input_shape=(1, 28, 28), device: str = "cpu") -> None:
        if M < 2 or M % 2:
            raise ValueError("clauses per class must be even and >= 2")
        if T <= 0 or s < 1:
            raise ValueError("T must be positive and s >= 1")
        self.config: Dict[str, Any] = dict(n_features=n_features, M=int(M), T=int(T), s=float(s), kind=kind,
                                           n_classes=n_classes, n_states=n_states, weighted=weighted,
                                           feedback_mode=feedback_mode, patch_size=patch_size, stride=stride,
                                           input_shape=list(input_shape))
        self.device = device
        common = dict(weighted=weighted, n_states=n_states, feedback_mode=feedback_mode,
                      drop_clause_p=0.0, drop_literal_p=0.0, max_included_literals=None)
        if kind == "flat":
            self.model = tt.TsetlinMachine(n_features, n_classes, n_clauses=M, T=T, s=s, **common)
        elif kind == "conv":
            self.model = tt.ConvTsetlinMachine(n_classes, M, T=T, s=s, patch_size=patch_size, stride=stride,
                                               position_encoding=True, input_shape=tuple(input_shape), **common)
        else:
            raise ValueError(kind)
        self.model.to(device)

    # ------------------------------------------------------------------ properties
    @property
    def T(self) -> int:
        return int(self.config["T"])

    @property
    def M(self) -> int:
        return int(self.config["M"])

    def _x(self, X: Tensor) -> Tensor:
        X = X.to(self.device)
        if X.dtype != torch.bool:
            X = X.bool()
        if self.config["kind"] == "conv" and X.dim() == 3:
            X = X.unsqueeze(1)
        return X

    # ------------------------------------------------------------------ learning
    def fit_epoch(self, X: Tensor, y: Tensor, *, shuffle: bool = True, batch_size: int = 32) -> None:
        """One pass of ordinary (batched) TM feedback. Never resets the state."""
        if len(torch.unique(y)) < self.config["n_classes"]:
            raise ValueError("training stream must contain every class")
        self.model.train()
        X = self._x(X)
        y = y.to(self.device)
        n = X.shape[0]
        idx = torch.randperm(n, device=self.device) if shuffle else torch.arange(n, device=self.device)
        for i in range(0, n, batch_size):
            j = idx[i:i + batch_size]
            self.model.update(X[j], y[j])

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def votes(self, X: Tensor, batch: int = 4096) -> Tensor:
        """Clamped votes ``(B, 10)`` with prediction semantics (``eval()``)."""
        was = self.model.training
        self.model.eval()
        X = self._x(X)
        out = [self.model(X[i:i + batch]) for i in range(0, X.shape[0], batch)]
        self.model.train(was)
        return torch.cat(out, 0) if out else torch.zeros(0, self.config["n_classes"], device=self.device)

    def unit_scores(self, X: Tensor, batch: int = 4096) -> Tensor:
        """Spec §7.1 ``u = clip(s / T, -1, 1)``, which is the library's clamped vote over ``T``."""
        return self.votes(X, batch) / float(self.T)

    def signed_weights(self) -> Tensor:
        w = self.model.clause_polarity.float()
        if self.config["weighted"]:
            w = w * self.model.weights.float()
        return w

    @torch.no_grad()
    def clause_outputs(self, X: Tensor, limit: Optional[int] = None) -> Tensor:
        was = self.model.training
        self.model.eval()
        X = self._x(X if limit is None else X[:limit])
        out = self.model.evaluate_clauses(X)
        self.model.train(was)
        return out

    @torch.no_grad()
    def class_scores(self, X: Tensor, batch: int = 4096) -> Tensor:
        """Raw (unclamped) signed vote sums ``(B, 10)``, class order 0..9."""
        w = self.signed_weights()
        K, M = self.config["n_classes"], self.M
        outs = []
        for i in range(0, X.shape[0], batch):
            co = self.clause_outputs(X[i:i + batch]).float()
            outs.append((co * w).view(-1, K, M).sum(-1))
        return torch.cat(outs, 0)

    def predict(self, X: Tensor) -> Tensor:
        return self.votes(X).argmax(dim=1)  # ties -> lowest index (D11)

    # ------------------------------------------------------------------ state
    def state_hash(self) -> str:
        h = sha256_array(self.model.ta_state)
        if self.config["weighted"]:
            h = sha256_array(torch.cat([self.model.ta_state.flatten().float(), self.model.weights.float()]))
        return h

    def n_automata(self) -> int:
        return int(self.model.ta_state.numel())

    def state_manifest(self) -> Dict[str, Any]:
        m = self.model
        return {**self.config, "n_features_actual": int(m.n_features), "n_clauses_total": int(m.n_clauses_total),
                "n_literals": int(m.n_literals), "automata": self.n_automata(),
                "state_dtype": str(m.ta_state.dtype), "stored_bytes": int(m.ta_state.numel() * m.ta_state.element_size()),
                "logical_bytes_8bit": self.n_automata(), "ta_state_sha256": self.state_hash(),
                "mean_clause_size": float(m.include_count.float().mean()),
                "empty_clauses": int((m.include_count == 0).sum())}

    def state_dict(self) -> Dict[str, Any]:
        return {"config": self.config, "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}

    def load_state(self, sd: Dict[str, Any]) -> None:
        self.model.load_state_dict({k: v.to(self.device) for k, v in sd["state_dict"].items()})

    @classmethod
    def from_state(cls, sd: Dict[str, Any], device: str = "cpu") -> "TMAdapter":
        c = dict(sd["config"])
        a = cls(c.pop("n_features"), c.pop("M"), c.pop("T"), c.pop("s"), device=device, **c)
        a.load_state(sd)
        return a

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "TMAdapter":
        return cls.from_state(torch.load(path, map_location="cpu", weights_only=False), device)
