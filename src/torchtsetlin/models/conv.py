"""Convolutional Tsetlin machines (1D and 2D) with thermometer-encoded patch positions."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as TF

from .. import functional as F
from ..utils import chunk_indices
from .base import FeedbackAccumulator
from .classifier import TsetlinMachine
from .coalesced import CoalescedTsetlinMachine
from .regression import RegressionTsetlinMachine

__all__ = [
    "ConvTsetlinMachine",
    "ConvCoalescedTsetlinMachine",
    "ConvRegressionTsetlinMachine",
    "Conv1dTsetlinMachine",
]

IntOrPair = Union[int, Tuple[int, int]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    a, b = v
    return (int(a), int(b))


class _ConvMixin:
    """Patch-based clause evaluation shared by all convolutional variants.

    Input images are Boolean tensors ``(B, Z, H, W)`` (``Z`` bit planes / channels). Every
    ``kh x kw`` window (stride ``stride``) yields a patch whose Boolean features are the
    ``Z*kh*kw`` pixels followed by thermometer-encoded patch coordinates: ``P_y - 1`` bits
    ``[y > k]`` and ``P_x - 1`` bits ``[x > k]`` where ``P_y, P_x`` are the numbers of patch
    positions along each axis. A clause is True for an image iff it matches at least one
    patch; each feedback event updates the clause from one randomly chosen matching patch
    (Granmo et al., 2019).
    """

    patch_size: Tuple[int, int]
    stride: Tuple[int, int]
    position_encoding: bool
    input_shape: Optional[Tuple[int, int, int]]

    def _init_conv(
        self,
        patch_size: IntOrPair,
        stride: IntOrPair,
        position_encoding: bool,
        input_shape: Optional[Tuple[int, int, int]],
    ) -> None:
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        self.position_encoding = bool(position_encoding)
        self.input_shape = tuple(int(v) for v in input_shape) if input_shape is not None else None  # type: ignore[assignment]
        if min(self.patch_size) < 1 or min(self.stride) < 1:
            raise ValueError("patch_size and stride must be positive")

    # ---- geometry -----------------------------------------------------------------
    def _grid(self, H: int, W: int) -> Tuple[int, int]:
        kh, kw = self.patch_size
        sh, sw = self.stride
        if H < kh or W < kw:
            raise ValueError(f"Input {H}x{W} smaller than patch {kh}x{kw}")
        return (H - kh) // sh + 1, (W - kw) // sw + 1

    def conv_features(self, input_shape: Tuple[int, int, int]) -> int:
        """Boolean features per patch for an input of shape ``(Z, H, W)``."""
        Z, H, W = input_shape
        Py, Px = self._grid(H, W)
        n = Z * self.patch_size[0] * self.patch_size[1]
        if self.position_encoding:
            n += (Py - 1) + (Px - 1)
        return n

    def _coerce_input(self, x) -> Tensor:  # type: ignore[override]
        xb = super()._coerce_input(x)  # type: ignore[misc]
        if xb.dim() == 2:  # (H, W) single image
            xb = xb.unsqueeze(0).unsqueeze(0)
        elif xb.dim() == 3:
            if self.input_shape is not None and tuple(xb.shape) == tuple(self.input_shape):
                xb = xb.unsqueeze(0)  # single (Z, H, W) image
            else:
                xb = xb.unsqueeze(1)  # (B, H, W) -> one plane
        if xb.dim() != 4:
            raise ValueError("Convolutional models expect input shaped (B, Z, H, W)")
        if self.input_shape is None:
            self.input_shape = tuple(int(v) for v in xb.shape[1:])  # type: ignore[assignment]
        elif tuple(xb.shape[1:]) != tuple(self.input_shape):
            raise ValueError(f"Expected input shape {self.input_shape}, got {tuple(xb.shape[1:])}")
        return xb

    def _infer_n_features(self, xb: Tensor) -> int:  # type: ignore[override]
        return self.conv_features(tuple(xb.shape[1:]))  # type: ignore[arg-type]

    def _patches_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        Py, Px = self._grid(xb.shape[2], xb.shape[3])
        return Py * Px

    def _chunk_elements_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        # The (B, P, C) match tensor plus, per feedback kind, the (n_sel <= B*C, P) patch
        # draw, and the worst-case gathered literal rows (B*C, 2F).
        P = self._patches_per_example(xb)
        C = self.n_clauses_total  # type: ignore[attr-defined]
        return 3 * P * C + 2 * C * self.n_literals + P * self.n_literals  # type: ignore[attr-defined]

    def _position_bits(self, Py: int, Px: int, device: torch.device) -> Tensor:
        """``(P, (Py-1)+(Px-1))`` thermometer bits for every patch position (row-major)."""
        ys = torch.arange(Py, device=device).repeat_interleave(Px)
        xs = torch.arange(Px, device=device).repeat(Py)
        ybits = ys.unsqueeze(1) > torch.arange(Py - 1, device=device).unsqueeze(0)
        xbits = xs.unsqueeze(1) > torch.arange(Px - 1, device=device).unsqueeze(0)
        return torch.cat([ybits, xbits], dim=1)

    def _encode(self, xb: Tensor) -> Tensor:  # type: ignore[override]
        """``(B, Z, H, W)`` Boolean -> literals ``(B, P, 2F)``."""
        B, Z, H, W = xb.shape
        Py, Px = self._grid(H, W)
        patches = TF.unfold(
            xb.to(self.compute_dtype), kernel_size=self.patch_size, stride=self.stride
        )  # (B, Z*kh*kw, P)
        patches = patches.transpose(1, 2)  # (B, P, Z*kh*kw)
        if self.position_encoding:
            pos = self._position_bits(Py, Px, xb.device).to(self.compute_dtype)
            patches = torch.cat([patches, pos.unsqueeze(0).expand(B, -1, -1)], dim=2)
        if patches.shape[2] != self.n_features:
            raise ValueError(
                f"Patch feature mismatch: got {patches.shape[2]}, expected {self.n_features}"
            )
        return torch.cat([patches, 1.0 - patches], dim=2)

    def _evaluate(self, literals: Tensor, empty_value: bool) -> Tuple[Tensor, Any]:  # type: ignore[override]
        B, P, L = literals.shape
        matches = F.clause_outputs(
            literals.reshape(B * P, L), self.include, self.include_count, empty_value
        ).view(B, P, -1)  # (B, P, C)
        # A clause is True for the image iff it matches at least one patch. The per-patch
        # matches travel on as the feedback context: :meth:`_feedback_counts` draws the random
        # matching patch from them, so prediction never pays for the draw.
        return matches.any(dim=1), matches

    def _feedback_counts(  # type: ignore[override]
        self,
        literals: Tensor,
        ctx: Any,
        sel_fire_i: Tensor,
        sel_nofire_i: Tensor,
        sel_fire_ii: Tensor,
        acc: FeedbackAccumulator,
    ) -> None:
        matches: Tensor = ctx
        acc.n_ib += sel_nofire_i.sum(dim=0)  # Type Ib needs no patch features
        B, P, L = literals.shape
        flat = literals.reshape(B * P, L)
        for sel, kind in ((sel_fire_i, "i"), (sel_fire_ii, "ii")):
            nz = torch.nonzero(sel, as_tuple=True)
            if nz[0].numel() == 0:
                continue
            b_idx, j_idx = nz
            # One uniformly random *matching* patch per feedback event. The draws are
            # independent per event (and so between Type Ia and Type II), because a coalesced
            # clause can receive both in the same update.
            cand = matches[b_idx, :, j_idx]  # (n_sel, P)
            p_idx = (torch.rand(cand.shape, device=cand.device) * cand).argmax(dim=1)
            rows = flat[b_idx * P + p_idx]  # (n_sel, 2F)
            if kind == "i":
                acc.n_true.index_add_(0, j_idx, rows)
                acc.n_false.index_add_(0, j_idx, 1.0 - rows)
            else:
                acc.n2.index_add_(0, j_idx, 1.0 - rows)

    # ---- dense introspection ------------------------------------------------------
    @torch.no_grad()
    def patch_clause_outputs(self, x, empty_value: Optional[bool] = None) -> Tensor:
        """Per-patch clause matches ``(B, Py, Px, C)`` — the tensor *before* the disjunction.

        :meth:`_evaluate` reduces this to ``(B, C)`` with ``matches.any(dim=1)``, which is
        what makes a convolutional Tsetlin machine translation tolerant and what stops it
        being able to localise anything. This accessor hands back the unreduced tensor, so a
        clause's matches can be plotted as a map, or fed to a second stage the way *CTM-UNet*
        stacks blocks. For actually *learning* a dense output, use
        :class:`~torchtsetlin.models.SegmentationTsetlinMachine` instead — it keeps the patch
        axis through the feedback path as well.

        Args:
            x: Boolean input batch ``(B, Z, H, W)``.
            empty_value: value of a clause with no included literals; defaults to
                ``self.training`` (``False`` when predicting).

        Returns:
            ``(B, Py, Px, C)`` bool.
        """
        if empty_value is None:
            empty_value = self.training  # type: ignore[attr-defined]
        xb = self._prepare(x)  # type: ignore[attr-defined]
        Py, Px = self._grid(xb.shape[2], xb.shape[3])
        outs = []
        for sl in chunk_indices(xb.shape[0], self._chunk_size(xb)):  # type: ignore[attr-defined]
            _, matches = self._evaluate(self._encode(xb[sl]), empty_value)  # (b, P, C)
            outs.append(matches)
        return torch.cat(outs, dim=0).view(xb.shape[0], Py, Px, -1)

    # ---- interpretation -----------------------------------------------------------
    def default_feature_names(self) -> List[str]:  # type: ignore[override]
        if self.input_shape is None:
            return super().default_feature_names()  # type: ignore[misc]
        Z, H, W = self.input_shape
        kh, kw = self.patch_size
        names = [f"p[{z},{i},{j}]" for z in range(Z) for i in range(kh) for j in range(kw)]
        if self.position_encoding:
            Py, Px = self._grid(H, W)
            names += [f"y>{k}" for k in range(Py - 1)] + [f"x>{k}" for k in range(Px - 1)]
        return names

    def clause_patch(self, clause: int) -> Tensor:
        """Visualise a clause as a ``(Z, kh, kw)`` int8 map: ``1`` = pixel must be on,
        ``-1`` = pixel must be off, ``0`` = don't care."""
        self._check_initialized()  # type: ignore[attr-defined]
        Z, _, _ = self.input_shape  # type: ignore[misc]
        kh, kw = self.patch_size
        n_pix = Z * kh * kw
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        pos = mask[:n_pix].to(torch.int8)
        neg = mask[Fn : Fn + n_pix].to(torch.int8)
        return (pos - neg).view(Z, kh, kw)

    def clause_region(self, clause: int) -> dict:
        """Decode the position literals of a clause into an inclusive patch-coordinate range
        ``{"y": (lo, hi), "x": (lo, hi)}`` (in patch grid units). ``lo > hi`` means the
        clause carries contradictory position literals and can never match."""
        self._check_initialized()  # type: ignore[attr-defined]
        Z, H, W = self.input_shape  # type: ignore[misc]
        Py, Px = self._grid(H, W)
        kh, kw = self.patch_size
        n_pix = Z * kh * kw
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        if not self.position_encoding:
            return {"y": (0, Py - 1), "x": (0, Px - 1)}

        def rng(offset: int, n: int) -> Tuple[int, int]:
            pos = mask[offset : offset + n]
            neg = mask[Fn + offset : Fn + offset + n]
            lo = 0
            hi = n  # n thermometer bits -> coordinates 0..n
            if pos.any():
                lo = int(torch.nonzero(pos).max()) + 1  # coord > k  -> coord >= k+1
            if neg.any():
                hi = int(torch.nonzero(neg).min())  # NOT (coord > k) -> coord <= k
            return lo, hi

        return {"y": rng(n_pix, Py - 1), "x": rng(n_pix + Py - 1, Px - 1)}

    def conv_extra_repr(self) -> str:
        return (
            f"patch_size={self.patch_size}, stride={self.stride}, "
            f"position_encoding={self.position_encoding}, input_shape={self.input_shape}"
        )


class ConvTsetlinMachine(_ConvMixin, TsetlinMachine):
    """Convolutional multi-class Tsetlin machine (Granmo et al., 2019).

    Args:
        n_classes: number of classes.
        n_clauses: clauses per class.
        T: vote margin.
        s: specificity.
        patch_size: ``(kh, kw)`` convolution window (int for square windows).
        stride: window stride.
        position_encoding: append thermometer-encoded patch coordinates to each patch.
        input_shape: ``(Z, H, W)`` of the Boolean images; ``None`` = infer from the first batch.
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachine`
            (``weighted``, ``n_states``, ``max_included_literals``, ``drop_clause_p``, ...).

    Shape:
        - input: ``(B, Z, H, W)`` Boolean (``(B, H, W)`` is treated as one plane).
        - output: ``(B, n_classes)`` vote sums.
    """

    def __init__(
        self,
        n_classes: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrPair = 10,
        stride: IntOrPair = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs,
    ) -> None:
        self._init_conv(patch_size, stride, position_encoding, input_shape)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_classes, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()


class ConvCoalescedTsetlinMachine(_ConvMixin, CoalescedTsetlinMachine):
    """Convolutional coalesced Tsetlin machine (shared clauses, per-output weights)."""

    def __init__(
        self,
        n_outputs: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrPair = 10,
        stride: IntOrPair = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs,
    ) -> None:
        self._init_conv(patch_size, stride, position_encoding, input_shape)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_outputs, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()


class ConvRegressionTsetlinMachine(_ConvMixin, RegressionTsetlinMachine):
    """Convolutional regression Tsetlin machine (Abeyrathna et al., 2021)."""

    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrPair = 10,
        stride: IntOrPair = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs,
    ) -> None:
        self._init_conv(patch_size, stride, position_encoding, input_shape)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()


class Conv1dTsetlinMachine(ConvTsetlinMachine):
    """1D convolutional Tsetlin machine for sequences / time series.

    Input ``(B, Z, L)`` (``Z`` Boolean channels, length ``L``) is handled as a ``(B, Z, 1, L)``
    image with a ``1 x kernel_size`` window.
    """

    def __init__(
        self,
        n_classes: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        kernel_size: int = 5,
        stride: int = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        shape3 = (input_shape[0], 1, input_shape[1]) if input_shape is not None else None
        super().__init__(
            n_classes,
            n_clauses,
            T,
            s,
            patch_size=(1, kernel_size),
            stride=(1, stride),
            position_encoding=position_encoding,
            input_shape=shape3,
            **kwargs,
        )

    def _coerce_input(self, x) -> Tensor:  # type: ignore[override]
        from ..utils import as_bool_tensor

        xb = as_bool_tensor(x, device=self.ta_state.device)
        if xb.dim() == 2:  # (B, L) -> one channel
            xb = xb.unsqueeze(1)
        if xb.dim() == 3:  # (B, Z, L) -> (B, Z, 1, L)
            xb = xb.unsqueeze(2)
        return super()._coerce_input(xb)
