"""Thermometer Booleanization of preprocessed tabular features for the TM arms.

The library's :class:`torchtsetlin.data.ThermometerEncoder` uses the same number of thresholds
for every column; tabular data mixes binary, low-cardinality and continuous columns, so this
encoder keeps a *variable* number of unique quantile thresholds per column and remembers which
column each bit belongs to (needed to map clause literals back to features for attribution).

Bit ``b`` is ``x[:, col[b]] >= thr[b]``. Fitted on the training split only.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor


class TabularBooleanizer:
    def __init__(self, n_bits: int = 10, thresholds: Optional[List[float]] = None):
        """``thresholds``: fixed cut points shared by every column (e.g. ``[0.3]`` for MNIST
        intensities) instead of per-column quantiles; a cut is kept for a column only if the
        training data has values on both sides of it."""
        if n_bits < 1:
            raise ValueError("n_bits must be >= 1")
        self.n_bits = int(n_bits)
        self.thresholds = None if thresholds is None else [float(t) for t in thresholds]
        self.col: Optional[np.ndarray] = None  # (n_bool,) int: source column of each bit
        self.thr: Optional[np.ndarray] = None  # (n_bool,) float32 thresholds
        self._cache = {}

    @property
    def n_bool(self) -> int:
        return int(self.thr.shape[0])

    def fit(self, X: np.ndarray) -> TabularBooleanizer:
        cols, thrs = [], []
        q = np.linspace(0, 1, self.n_bits + 2)[1:-1]
        for j in range(X.shape[1]):
            v = X[:, j].astype(np.float64)
            u = np.unique(v)
            if len(u) <= 1:
                continue
            if self.thresholds is not None:
                t = np.asarray([t for t in self.thresholds if u[0] < t <= u[-1]])
            elif len(u) <= self.n_bits + 1:
                t = (u[:-1] + u[1:]) / 2.0  # every cut between observed values
            else:
                t = np.unique(np.quantile(v, q))
                t = t[(t > u[0])]  # a threshold at the minimum is always True: useless
            for tt in t:
                cols.append(j)
                thrs.append(tt)
        if not thrs:
            raise ValueError("no informative column to Booleanize")
        self.col = np.asarray(cols, dtype=np.int64)
        self.thr = np.asarray(thrs, dtype=np.float32)
        self._cache = {}
        return self

    def _tensors(self, device):
        key = str(device)
        if key not in self._cache:
            self._cache[key] = (torch.as_tensor(self.col, device=device),
                                torch.as_tensor(self.thr, device=device))
        return self._cache[key]

    def transform(self, X) -> Tensor:
        """``(N, n_features)`` float (numpy or tensor) -> ``(N, n_bool)`` bool tensor on the
        same device as a tensor input (CPU for numpy)."""
        if not isinstance(X, Tensor):
            X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        col, thr = self._tensors(X.device)
        return X.float()[:, col] >= thr

    def bit_names(self, feature_names: List[str]) -> List[str]:
        return [f"{feature_names[c]}>={t:.3g}" for c, t in zip(self.col, self.thr)]

    def state(self) -> dict:
        return {"n_bits": self.n_bits, "thresholds": self.thresholds, "col": self.col.tolist(),
                "thr": self.thr.tolist()}

    @classmethod
    def from_state(cls, st: dict) -> TabularBooleanizer:
        b = cls(st["n_bits"], st.get("thresholds"))
        b.col = np.asarray(st["col"], dtype=np.int64)
        b.thr = np.asarray(st["thr"], dtype=np.float32)
        return b
