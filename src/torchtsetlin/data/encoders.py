"""Booleanization: turning continuous, categorical and image data into Boolean features.

All encoders are ``torch.nn.Module`` objects so they can be moved to a device, saved in a
``state_dict`` and composed with :class:`torchtsetlin.data.Compose`. They follow the
scikit-learn ``fit`` / ``transform`` protocol on top of ``forward`` (``transform`` is an
alias of ``forward``; ``fit_transform`` chains both).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as TF

from ..utils import as_float_tensor

__all__ = [
    "BooleanEncoder",
    "Binarizer",
    "ThermometerEncoder",
    "OneHotEncoder",
    "BitPlaneEncoder",
    "AdaptiveThresholdEncoder",
    "ColorThermometerEncoder",
    "HypervectorEncoder",
    "PyramidEncoder",
    "Compose",
    "Flatten",
    "thermometer_thresholds",
]


class BooleanEncoder(nn.Module):
    """Base class for encoders. Subclasses implement :meth:`forward` (and optionally
    :meth:`fit`). ``transform`` is an alias of ``forward``."""

    def fit(self, x, y=None) -> BooleanEncoder:  # noqa: D401
        """Fit encoder statistics (no-op for stateless encoders)."""
        return self

    def transform(self, x) -> Tensor:
        return self.forward(x)

    def fit_transform(self, x, y=None) -> Tensor:
        return self.fit(x, y).forward(x)

    @property
    def device(self) -> torch.device:
        for b in self.buffers():
            return b.device
        return torch.device("cpu")

    def _as_float(self, x) -> Tensor:
        return as_float_tensor(x, device=self.device)

    def output_size(self, n_inputs: int) -> int:  # pragma: no cover - overridden
        raise NotImplementedError


# --------------------------------------------------------------------------------------
class Binarizer(BooleanEncoder):
    """``x > threshold`` (or ``>=``) per element. ``threshold`` may be a scalar or a tensor
    broadcastable to the input (e.g. one threshold per feature)."""

    def __init__(self, threshold: Union[float, Tensor] = 0.5, inclusive: bool = False) -> None:
        super().__init__()
        self.register_buffer("threshold", torch.as_tensor(threshold, dtype=torch.float32))
        self.inclusive = bool(inclusive)

    def forward(self, x) -> Tensor:
        x = self._as_float(x)
        return (x >= self.threshold) if self.inclusive else (x > self.threshold)

    def output_size(self, n_inputs: int) -> int:
        return n_inputs


def thermometer_thresholds(
    x: Tensor,
    n_bits: int,
    strategy: str = "quantile",
    value_range: Optional[Sequence[float]] = None,
) -> Tensor:
    """Per-feature thresholds ``(F, n_bits)`` for thermometer encoding.

    Args:
        x: ``(N, F)`` training data (only needed for ``"quantile"``/``"unique"`` or when
            ``value_range`` is ``None``).
        n_bits: thresholds per feature.
        strategy: ``"quantile"`` (equal-frequency), ``"uniform"`` (equal-width) or
            ``"unique"`` (evenly spaced ranks of the unique training values).
        value_range: optional global ``(lo, hi)`` for the uniform strategy.
    """
    x = x.to(torch.float32)
    Fn = x.shape[1]
    if strategy == "uniform":
        if value_range is not None:
            lo = torch.full((Fn,), float(value_range[0]), device=x.device)
            hi = torch.full((Fn,), float(value_range[1]), device=x.device)
        else:
            lo, hi = x.min(dim=0).values, x.max(dim=0).values
        steps = torch.arange(1, n_bits + 1, device=x.device, dtype=torch.float32) / (n_bits + 1)
        return lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * steps.unsqueeze(0)
    if strategy == "quantile":
        q = torch.arange(1, n_bits + 1, device=x.device, dtype=torch.float32) / (n_bits + 1)
        return torch.quantile(x, q, dim=0).transpose(0, 1).contiguous()
    if strategy == "unique":
        out = torch.empty(Fn, n_bits, device=x.device)
        for f in range(Fn):
            u = torch.unique(x[:, f])
            if u.numel() > 1:
                u = u[1:]  # ``x >= min`` is always True — drop it
            if u.numel() >= n_bits:
                idx = torch.linspace(0, u.numel() - 1, n_bits, device=x.device).round().long()
                out[f] = u[idx]
            else:  # pad by repeating the max (bits that are never on for training data)
                pad = u.max().repeat(n_bits - u.numel()) if u.numel() else torch.zeros(n_bits)
                out[f] = torch.cat([u, pad.to(u.device)])
        return out
    raise ValueError("strategy must be 'quantile', 'uniform' or 'unique'")


class ThermometerEncoder(BooleanEncoder):
    """Thermometer (ordinal) encoding of continuous features.

    Each feature ``v`` becomes ``n_bits`` Booleans ``[v >= t_1, ..., v >= t_n]`` for
    increasing thresholds ``t_k``. Clauses can then express ranges with two literals
    (``x>=t_a AND NOT x>=t_b``). Thresholds are learned by :meth:`fit` (quantiles by default)
    or given explicitly.

    Args:
        n_bits: thresholds per feature.
        strategy: threshold strategy (see :func:`thermometer_thresholds`).
        thresholds: explicit ``(F, n_bits)`` thresholds (skips fitting).
        value_range: ``(lo, hi)`` for the uniform strategy (e.g. ``(0, 255)`` for images).
        flatten: return ``(N, F*n_bits)`` instead of ``(N, F, n_bits)``.
    """

    thresholds: Tensor

    def __init__(
        self,
        n_bits: int = 8,
        strategy: str = "quantile",
        thresholds: Optional[Tensor] = None,
        value_range: Optional[Sequence[float]] = None,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.strategy = strategy
        self.value_range = tuple(value_range) if value_range is not None else None
        self.flatten = bool(flatten)
        if thresholds is not None:
            self.register_buffer("thresholds", torch.as_tensor(thresholds, dtype=torch.float32))
        else:
            self.register_buffer("thresholds", torch.empty(0))

    @property
    def is_fitted(self) -> bool:
        return self.thresholds.numel() > 0

    def fit(self, x, y=None) -> ThermometerEncoder:
        x = self._as_float(x)
        x2 = x.reshape(x.shape[0], -1)
        self.thresholds = thermometer_thresholds(x2, self.n_bits, self.strategy, self.value_range)
        return self

    def forward(self, x) -> Tensor:
        x = self._as_float(x)
        shape = x.shape
        x2 = x.reshape(shape[0], -1)
        if not self.is_fitted:
            if self.strategy == "uniform" and self.value_range is not None:
                self.thresholds = thermometer_thresholds(
                    x2, self.n_bits, "uniform", self.value_range
                )
            else:
                raise RuntimeError("ThermometerEncoder must be fitted first (call .fit(x)).")
        thr = self.thresholds
        if thr.shape[0] == 1 and x2.shape[1] != 1:
            thr = thr.expand(x2.shape[1], -1)
        out = x2.unsqueeze(2) >= thr.unsqueeze(0)  # (N, F, bits)
        if self.flatten:
            return out.reshape(shape[0], -1)
        return out.reshape(*shape, self.n_bits)

    def output_size(self, n_inputs: int) -> int:
        return n_inputs * self.n_bits

    def feature_names(self, names: Sequence[str]) -> List[str]:
        """Names of the Boolean outputs, e.g. ``"age>=30.0"``."""
        thr = self.thresholds
        if thr.shape[0] == 1:
            thr = thr.expand(len(names), -1)
        return [f"{n}>={float(thr[i, k]):.4g}" for i, n in enumerate(names) for k in range(self.n_bits)]

    def decode_range(self, feature: int, included_pos: Iterable[int], included_neg: Iterable[int]):
        """Range ``(lo, hi)`` implied by included positive/negated bits of one feature."""
        thr = self.thresholds[feature if self.thresholds.shape[0] > 1 else 0]
        lo = max((float(thr[k]) for k in included_pos), default=-float("inf"))
        hi = min((float(thr[k]) for k in included_neg), default=float("inf"))
        return lo, hi


class OneHotEncoder(BooleanEncoder):
    """One Boolean per category value per column. Categories are integers (fit records the
    number of categories per column, or pass ``n_categories``)."""

    def __init__(self, n_categories: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.register_buffer(
            "n_categories",
            torch.as_tensor(list(n_categories), dtype=torch.long) if n_categories else torch.empty(0, dtype=torch.long),
        )

    def fit(self, x, y=None) -> OneHotEncoder:
        x = torch.as_tensor(np.asarray(x)).long().to(self.device)
        self.n_categories = x.max(dim=0).values + 1
        return self

    def forward(self, x) -> Tensor:
        x = torch.as_tensor(np.asarray(x)).long().to(self.device)
        if self.n_categories.numel() == 0:
            raise RuntimeError("OneHotEncoder must be fitted first")
        outs = [TF.one_hot(x[:, i].clamp(0, int(n) - 1), int(n)).bool() for i, n in enumerate(self.n_categories)]
        return torch.cat(outs, dim=1)

    def output_size(self, n_inputs: int) -> int:
        return int(self.n_categories.sum())

    def feature_names(self, names: Sequence[str], categories: Optional[Sequence[Sequence[str]]] = None) -> List[str]:
        out = []
        for i, n in enumerate(self.n_categories):
            for k in range(int(n)):
                cat = categories[i][k] if categories is not None else str(k)
                out.append(f"{names[i]}={cat}")
        return out


class BitPlaneEncoder(BooleanEncoder):
    """Plain binary (place-value) encoding of non-negative integers with ``n_bits`` bits,
    most significant bit first. Images ``(N, C, H, W)`` become ``(N, C*n_bits, H, W)``."""

    def __init__(self, n_bits: int = 8, channel_dim: int = 1) -> None:
        super().__init__()
        self.n_bits = int(n_bits)
        self.channel_dim = channel_dim

    def forward(self, x) -> Tensor:
        x = torch.as_tensor(np.asarray(x)).to(self.device)
        if x.is_floating_point():
            x = x.round()
        x = x.long()
        shifts = torch.arange(self.n_bits - 1, -1, -1, device=x.device)
        if x.dim() >= 3:  # (N, C, ...) -> stack bits into the channel dimension
            bits = ((x.unsqueeze(2) >> shifts.view(1, 1, -1, *([1] * (x.dim() - 2)))) & 1).bool()
            return bits.flatten(1, 2)
        bits = ((x.unsqueeze(-1) >> shifts) & 1).bool()
        return bits.flatten(1)

    def output_size(self, n_inputs: int) -> int:
        return n_inputs * self.n_bits


class AdaptiveThresholdEncoder(BooleanEncoder):
    """Adaptive (local) thresholding for grayscale / multi-channel images, as used to
    booleanize MNIST/CIFAR for convolutional Tsetlin machines.

    ``bit = pixel > local_mean(pixel) - C`` where the local mean is computed over a
    ``block_size x block_size`` window, either uniformly (``method="mean"``) or with a
    Gaussian kernel (``method="gaussian"``, OpenCV ``ADAPTIVE_THRESH_GAUSSIAN_C``).

    Input ``(N, C, H, W)`` (or ``(N, H, W)``) -> Boolean of the same shape.
    """

    def __init__(self, block_size: int = 11, C: float = 2.0, method: str = "gaussian") -> None:
        super().__init__()
        if block_size % 2 == 0 or block_size < 3:
            raise ValueError("block_size must be odd and >= 3")
        if method not in ("gaussian", "mean"):
            raise ValueError("method must be 'gaussian' or 'mean'")
        self.block_size = int(block_size)
        self.C = float(C)
        self.method = method
        if method == "gaussian":
            sigma = 0.3 * ((block_size - 1) * 0.5 - 1) + 0.8  # OpenCV's default sigma
            ax = torch.arange(block_size, dtype=torch.float32) - (block_size - 1) / 2
            g1 = torch.exp(-(ax**2) / (2 * sigma**2))
            g1 = g1 / g1.sum()
            kernel = torch.outer(g1, g1)
        else:
            kernel = torch.full((block_size, block_size), 1.0 / block_size**2)
        self.register_buffer("kernel", kernel.view(1, 1, block_size, block_size))

    def forward(self, x) -> Tensor:
        x = self._as_float(x)
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(1)
        N, Cc, H, W = x.shape
        pad = self.block_size // 2
        xp = TF.pad(x.reshape(N * Cc, 1, H, W), (pad, pad, pad, pad), mode="replicate")
        local = TF.conv2d(xp, self.kernel).reshape(N, Cc, H, W)
        out = x > (local - self.C)
        return out.squeeze(1) if squeeze else out

    def output_size(self, n_inputs: int) -> int:
        return n_inputs


class ColorThermometerEncoder(BooleanEncoder):
    """Per-channel thermometer encoding of images: each pixel channel becomes ``n_bits``
    planes ``[v >= k * 255/(n_bits+1)]``. ``(N, C, H, W)`` -> ``(N, C*n_bits, H, W)``.

    ``value_range`` defaults to ``(0, 255)``; use ``(0, 1)`` for float images.
    """

    def __init__(self, n_bits: int = 8, value_range: Sequence[float] = (0.0, 255.0)) -> None:
        super().__init__()
        lo, hi = float(value_range[0]), float(value_range[1])
        steps = torch.arange(1, n_bits + 1, dtype=torch.float32) / (n_bits + 1)
        self.register_buffer("thresholds", lo + (hi - lo) * steps)
        self.n_bits = int(n_bits)

    def forward(self, x) -> Tensor:
        x = self._as_float(x)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        N, Cc, H, W = x.shape
        thr = self.thresholds.view(1, 1, self.n_bits, 1, 1)
        out = x.unsqueeze(2) >= thr  # (N, C, bits, H, W)
        return out.reshape(N, Cc * self.n_bits, H, W)

    def output_size(self, n_inputs: int) -> int:
        return n_inputs * self.n_bits


class HypervectorEncoder(BooleanEncoder):
    """Sparse binary hypervector encoding of discrete tokens (Halenka et al., 2024).

    Every token id ``t < vocab_size`` owns a fixed random hypervector with ``n_bits`` ones
    among ``dim`` positions. A sample (a set / sequence of token ids) is *bundled* with
    bitwise OR. Optionally each position ``p`` is *bound* by cyclically shifting the token's
    hypervector by ``p`` (``bind_position=True``), so word order is preserved.

    Input: ``(N, L)`` long tensor of token ids (padding id ``-1`` is ignored).
    Output: ``(N, dim)`` Boolean.
    """

    table: Tensor

    def __init__(self, vocab_size: int, dim: int = 1024, n_bits: int = 8, bind_position: bool = False, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        table = torch.zeros(vocab_size, dim, dtype=torch.bool)
        for t in range(vocab_size):
            idx = torch.randperm(dim, generator=g)[:n_bits]
            table[t, idx] = True
        self.register_buffer("table", table)
        self.dim = int(dim)
        self.bind_position = bool(bind_position)

    def forward(self, tokens) -> Tensor:
        tokens = torch.as_tensor(np.asarray(tokens)).long().to(self.device)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        N, L = tokens.shape
        valid = tokens >= 0
        hv = self.table[tokens.clamp(min=0)] & valid.unsqueeze(2)  # (N, L, dim)
        if self.bind_position:
            shifted = torch.empty_like(hv)
            for p in range(L):
                shifted[:, p] = torch.roll(hv[:, p], shifts=p, dims=1)
            hv = shifted
        return hv.any(dim=1)

    def output_size(self, n_inputs: int) -> int:
        return self.dim


class PyramidEncoder(BooleanEncoder):
    """Stack each pixel's neighbourhood at several resolutions into extra Boolean planes.

    A ``k x k`` patch is all a dense Tsetlin machine sees of a pixel, and some classes simply
    are not decidable from it — a patch of building interior and a patch of road look alike
    once illumination varies; telling them apart needs to know where the horizon is. This is
    the gap an encoder-decoder closes in a CNN, and it can be closed *natively* here: OR-pool
    the Boolean planes to build a pyramid, take the same ``k x k`` patch at each level, and
    resample every level back to full resolution. A clause then reads a pixel's immediate
    neighbourhood **and** the coarse structure around it, with no float parameter and no
    gradient anywhere.

    Feed the result to a model with ``patch_size=1``: the neighbourhood is already in the
    planes, and asking for a second one on top multiplies the feature count for nothing.

    Args:
        scales: downsampling factors, ascending. ``1`` is the full-resolution level.
        patch: neighbourhood side length taken at each level.
        mode: pooling mode for the pyramid (see :func:`torchtsetlin.functional.boolean_pool`).

    Shape:
        - input: ``(B, Z, H, W)`` Boolean.
        - output: ``(B, Z * patch * patch * len(scales), H, W)`` Boolean.

    Example:
        >>> enc = PyramidEncoder(scales=(1, 2, 4), patch=3)
        >>> planes = enc(x)                      # (B, Z*9*3, H, W)
        >>> model = SegmentationTsetlinMachine(4, 300, T=60, patch_size=1)
    """

    def __init__(self, scales: Sequence[int] = (1, 2, 4), patch: int = 3, mode: str = "or") -> None:
        super().__init__()
        self.scales = tuple(int(s) for s in scales)
        self.patch = int(patch)
        self.mode = str(mode)
        if min(self.scales) < 1 or self.patch < 1:
            raise ValueError("scales and patch must be >= 1")

    def forward(self, x) -> Tensor:
        from ..functional import boolean_pool

        xb = x if isinstance(x, Tensor) else torch.as_tensor(np.asarray(x))
        if xb.dim() == 3:
            xb = xb.unsqueeze(1)
        if xb.dim() != 4:
            raise ValueError("PyramidEncoder expects (B, Z, H, W)")
        H, W = xb.shape[2], xb.shape[3]
        pad = self.patch // 2
        parts = []
        for sc in self.scales:
            xs = xb if sc == 1 else boolean_pool(xb, sc, self.mode)
            f = xs.to(torch.float32)
            u = TF.unfold(TF.pad(f, (pad, pad, pad, pad)), kernel_size=self.patch)
            u = u.transpose(1, 2).reshape(xs.shape[0], xs.shape[2], xs.shape[3], -1)
            u = u.permute(0, 3, 1, 2)  # (B, Z*k*k, h, w)
            if u.shape[-2:] != (H, W):
                u = TF.interpolate(u, size=(H, W), mode="nearest")
            parts.append(u)
        return torch.cat(parts, dim=1) > 0.5

    def output_size(self, n_inputs: int) -> int:
        return n_inputs * self.patch * self.patch * len(self.scales)


class Flatten(BooleanEncoder):
    """Flatten everything but the batch dimension."""

    def forward(self, x) -> Tensor:
        x = torch.as_tensor(np.asarray(x)) if not isinstance(x, Tensor) else x
        return x.reshape(x.shape[0], -1)

    def output_size(self, n_inputs: int) -> int:
        return n_inputs


class Compose(BooleanEncoder):
    """Chain encoders / callables. ``fit`` fits each stage on the output of the previous."""

    def __init__(self, *stages) -> None:
        super().__init__()
        self.stages = nn.ModuleList([s for s in stages if isinstance(s, nn.Module)])
        self._all = list(stages)

    def fit(self, x, y=None) -> Compose:
        for s in self._all:
            if hasattr(s, "fit"):
                s.fit(x, y)
            x = s(x)
        return self

    def forward(self, x) -> Tensor:
        for s in self._all:
            x = s(x)
        return x

    def output_size(self, n_inputs: int) -> int:
        for s in self._all:
            if hasattr(s, "output_size"):
                n_inputs = s.output_size(n_inputs)
        return n_inputs
