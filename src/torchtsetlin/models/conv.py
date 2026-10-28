"""Convolutional Tsetlin machines (1D, 2D and 3D) with thermometer-encoded patch positions."""

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
    "Conv3dTsetlinMachine",
    "Conv3dCoalescedTsetlinMachine",
    "Conv3dRegressionTsetlinMachine",
]

IntOrPair = Union[int, Tuple[int, int]]
IntOrTriple = Union[int, Tuple[int, int, int]]


def _pair(v: IntOrPair) -> Tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    a, b = v
    return (int(a), int(b))


def _triple(v: IntOrTriple) -> Tuple[int, int, int]:
    if isinstance(v, int):
        return (v, v, v)
    a, b, c = v
    return (int(a), int(b), int(c))


def _thermometer_range(mask: Tensor, Fn: int, offset: int, n: int) -> Tuple[int, int]:
    """Inclusive coordinate range ``(lo, hi)`` admitted by the included literals of the ``n``
    thermometer bits ``[coord > k]`` stored at feature ``offset`` (``lo > hi``: never)."""
    pos = mask[offset : offset + n]
    neg = mask[Fn + offset : Fn + offset + n]
    lo = 0
    hi = n  # n thermometer bits -> coordinates 0..n
    if pos.any():
        lo = int(torch.nonzero(pos).max()) + 1  # coord > k  -> coord >= k+1
    if neg.any():
        hi = int(torch.nonzero(neg).min())  # NOT (coord > k) -> coord <= k
    return lo, hi


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
        # The largest per-example intermediate is the (P, C) block of clause evaluation, next
        # to the example's (P, 2F) literals. The patch draw and the gathered literal rows of
        # :meth:`_feedback_counts` scale with the number of feedback events, not with B*C,
        # and are bounded separately (:meth:`_event_block`). Counting their worst case here
        # (every clause selected for every example) split a batch of 16 MNIST images into 4
        # chunks and made a GPU update ~2.5x slower.
        P = self._patches_per_example(xb)
        C = self.n_clauses_total  # type: ignore[attr-defined]
        return P * C + P * self.n_literals  # type: ignore[attr-defined]

    def _event_block(self, P: int) -> int:
        """Feedback events whose random patch is drawn together: the ``(n, P)`` draw and the
        ``(n, 2F)`` gathered literal rows stay within ``max_chunk_elements``."""
        L = self.n_literals  # type: ignore[attr-defined]
        return max(1, self.max_chunk_elements // (2 * P + 2 * L))  # type: ignore[attr-defined]

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
        B, P, L = literals.shape
        if acc.fuse and B == 1:
            # Fused sequential mode (F.commit_single_example): the random matching patch of
            # every clause, drawn independently for Type Ia and Type II, as (C, 2F) literal
            # rows; the fire masks select which of them count. Dense, so it compiles.
            m = matches[0].transpose(0, 1)  # (C, P)
            p_i = (torch.rand(m.shape, device=m.device) * m).argmax(dim=1)
            p_ii = (torch.rand(m.shape, device=m.device) * m).argmax(dim=1)
            acc.single = (literals[0, p_i], literals[0, p_ii], sel_fire_i[0], sel_nofire_i[0],
                          sel_fire_ii[0])
            return
        acc.n_ib += sel_nofire_i.sum(dim=0)  # Type Ib needs no patch features
        flat = literals.reshape(B * P, L)
        E = self._event_block(P)
        for sel, kind in ((sel_fire_i, "i"), (sel_fire_ii, "ii")):
            nz = torch.nonzero(sel, as_tuple=True)
            for ev in chunk_indices(nz[0].numel(), E):
                b_idx, j_idx = nz[0][ev], nz[1][ev]
                # One uniformly random *matching* patch per feedback event. The draws are
                # independent per event (and so between Type Ia and Type II), because a
                # coalesced clause can receive both in the same update.
                cand = matches[b_idx, :, j_idx]  # (n_sel, P)
                p_idx = (torch.rand(cand.shape, device=cand.device) * cand).argmax(dim=1)
                rows = flat[b_idx * P + p_idx]  # (n_sel, 2F)
                if kind == "i":
                    acc.n_true.index_add_(0, j_idx, rows)
                    acc.n_false.index_add_(0, j_idx, 1.0 - rows)
                else:
                    acc.n2.index_add_(0, j_idx, 1.0 - rows)

    _feedback_counts._records_single = True  # type: ignore[attr-defined]

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
        return {
            "y": _thermometer_range(mask, Fn, n_pix, Py - 1),
            "x": _thermometer_range(mask, Fn, n_pix + Py - 1, Px - 1),
        }

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


# --------------------------------------------------------------------------------------
# 3D (volumetric) convolution
# --------------------------------------------------------------------------------------
class _Conv3dMixin(_ConvMixin):
    """Patch-based clause evaluation over Boolean volumes ``(B, Z, D, H, W)``.

    Every ``kd x kh x kw`` window (stride ``(sd, sh, sw)``) yields a patch whose features are
    the ``Z*kd*kh*kw`` voxels followed by three thermometers of patch coordinates
    (``[d > k]``, ``[y > k]``, ``[x > k]``). Only the *geometry* differs from the 2D mixin:
    clause evaluation (an OR over patches) and the random-matching-patch feedback are
    inherited unchanged, because they never look at how the patch axis was produced — see
    ``plans/conv3d.md`` (Theorem 4). With ``D = kd = 1`` the literal tensor is bitwise the
    one :class:`ConvTsetlinMachine` builds, so the two models train identically.

    Volumes have many patches, so a single example can exceed ``max_chunk_elements``. Then
    (or when ``patches_per_chunk`` is set) clause evaluation and the patch draw run over
    blocks of the patch axis. The OR and the draw are both exact under that split (the draw
    is equal in distribution); with a single block the 2D code path runs as is.
    """

    patch_size: Tuple[int, int, int]  # type: ignore[assignment]
    stride: Tuple[int, int, int]  # type: ignore[assignment]
    input_shape: Optional[Tuple[int, int, int, int]]  # type: ignore[assignment]
    patches_per_chunk: Optional[int]

    def _init_conv3d(
        self,
        patch_size: IntOrTriple,
        stride: IntOrTriple,
        position_encoding: bool,
        input_shape: Optional[Tuple[int, int, int, int]],
        patches_per_chunk: Optional[int],
    ) -> None:
        self.patch_size = _triple(patch_size)
        self.stride = _triple(stride)
        self.position_encoding = bool(position_encoding)
        self.input_shape = tuple(int(v) for v in input_shape) if input_shape is not None else None  # type: ignore[assignment]
        if min(self.patch_size) < 1 or min(self.stride) < 1:
            raise ValueError("patch_size and stride must be positive")
        if self.input_shape is not None and len(self.input_shape) != 4:
            raise ValueError("input_shape must be (Z, D, H, W)")
        if patches_per_chunk is not None and int(patches_per_chunk) < 1:
            raise ValueError("patches_per_chunk must be positive")
        self.patches_per_chunk = int(patches_per_chunk) if patches_per_chunk is not None else None
        self._pos_cache: Optional[Tuple[Tuple[int, int, int], torch.device, Tensor]] = None

    # ---- geometry -----------------------------------------------------------------
    def _grid(self, D: int, H: int, W: int) -> Tuple[int, int, int]:  # type: ignore[override]
        kd, kh, kw = self.patch_size
        sd, sh, sw = self.stride
        if D < kd or H < kh or W < kw:
            raise ValueError(f"Input {D}x{H}x{W} smaller than patch {kd}x{kh}x{kw}")
        return (D - kd) // sd + 1, (H - kh) // sh + 1, (W - kw) // sw + 1

    def conv_features(self, input_shape: Tuple[int, int, int, int]) -> int:  # type: ignore[override]
        """Boolean features per patch for an input of shape ``(Z, D, H, W)``."""
        Z, D, H, W = input_shape
        Pd, Ph, Pw = self._grid(D, H, W)
        kd, kh, kw = self.patch_size
        n = Z * kd * kh * kw
        if self.position_encoding:
            n += (Pd - 1) + (Ph - 1) + (Pw - 1)
        return n

    def output_grid(self) -> Tuple[int, int, int]:
        """``(Pd, Ph, Pw)`` patch-grid shape for the model's ``input_shape``."""
        if self.input_shape is None:
            raise RuntimeError("input_shape is unknown until the model has seen a batch")
        _, D, H, W = self.input_shape
        return self._grid(D, H, W)

    def _coerce_input(self, x) -> Tensor:  # type: ignore[override]
        # Skip _ConvMixin._coerce_input: it would reshape the batch to 4D.
        xb = super(_ConvMixin, self)._coerce_input(x)  # type: ignore[misc]
        if xb.dim() == 3:  # (D, H, W) single one-plane volume
            xb = xb.unsqueeze(0).unsqueeze(0)
        elif xb.dim() == 4:
            if self.input_shape is not None and tuple(xb.shape) == tuple(self.input_shape):
                xb = xb.unsqueeze(0)  # single (Z, D, H, W) volume
            else:
                xb = xb.unsqueeze(1)  # (B, D, H, W) -> one plane
        if xb.dim() != 5:
            raise ValueError("3D convolutional models expect input shaped (B, Z, D, H, W)")
        if self.input_shape is None:
            self.input_shape = tuple(int(v) for v in xb.shape[1:])  # type: ignore[assignment]
        elif tuple(xb.shape[1:]) != tuple(self.input_shape):
            raise ValueError(f"Expected input shape {self.input_shape}, got {tuple(xb.shape[1:])}")
        return xb

    def _patches_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        Pd, Ph, Pw = self._grid(xb.shape[2], xb.shape[3], xb.shape[4])
        return Pd * Ph * Pw

    def _patch_block(self, P: int) -> int:
        """Patches evaluated together; ``P`` (one block) unless one example would exceed
        ``max_chunk_elements`` or ``patches_per_chunk`` asks for less."""
        if self.patches_per_chunk is not None:
            return max(1, min(P, self.patches_per_chunk))
        C = self.n_clauses_total  # type: ignore[attr-defined]
        L = self.n_literals  # type: ignore[attr-defined]
        budget = self.max_chunk_elements  # type: ignore[attr-defined]
        if 3 * P * C + 2 * C * L + P * L <= budget:
            return P
        return max(1, min(P, (budget - 2 * C * L - P * L) // (3 * C)))

    def _chunk_elements_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        P = self._patches_per_example(xb)
        Pb = self._patch_block(P)
        if Pb >= P:
            return super()._chunk_elements_per_example(xb)
        # Float intermediates now scale with the block; the bool (B, P, C) match tensor
        # (a quarter of a float element per entry) and the literals still span every patch.
        C = self.n_clauses_total  # type: ignore[attr-defined]
        L = self.n_literals  # type: ignore[attr-defined]
        return 3 * Pb * C + (P * C) // 4 + 2 * C * L + P * L

    def _position_bits(self, Pd: int, Ph: int, Pw: int, device: torch.device) -> Tensor:  # type: ignore[override]
        """``(P, (Pd-1)+(Ph-1)+(Pw-1))`` thermometer bits for every patch (row-major)."""
        cached = self._pos_cache
        if cached is not None and cached[0] == (Pd, Ph, Pw) and cached[1] == device:
            return cached[2]
        ds = torch.arange(Pd, device=device).repeat_interleave(Ph * Pw)
        ys = torch.arange(Ph, device=device).repeat_interleave(Pw).repeat(Pd)
        xs = torch.arange(Pw, device=device).repeat(Pd * Ph)

        def therm(coord: Tensor, n: int) -> Tensor:
            return coord.unsqueeze(1) > torch.arange(n - 1, device=device).unsqueeze(0)

        table = torch.cat([therm(ds, Pd), therm(ys, Ph), therm(xs, Pw)], dim=1)
        self._pos_cache = ((Pd, Ph, Pw), device, table)
        return table

    def _encode(self, xb: Tensor) -> Tensor:  # type: ignore[override]
        """``(B, Z, D, H, W)`` Boolean -> literals ``(B, P, 2F)``."""
        B, Z, D, H, W = xb.shape
        kd, kh, kw = self.patch_size
        sd, sh, sw = self.stride
        Pd, Ph, Pw = self._grid(D, H, W)
        # torch.nn.functional.unfold is 2D only; three Tensor.unfold calls cut the windows
        # out along depth, height and width -> (B, Z, Pd, Ph, Pw, kd, kh, kw).
        windows = xb.to(self.compute_dtype).unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)
        patches = windows.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(B, Pd * Ph * Pw, Z * kd * kh * kw)
        if self.position_encoding:
            pos = self._position_bits(Pd, Ph, Pw, xb.device).to(self.compute_dtype)
            patches = torch.cat([patches, pos.unsqueeze(0).expand(B, -1, -1)], dim=2)
        if patches.shape[2] != self.n_features:  # type: ignore[attr-defined]
            raise ValueError(
                f"Patch feature mismatch: got {patches.shape[2]}, expected {self.n_features}"  # type: ignore[attr-defined]
            )
        return torch.cat([patches, 1.0 - patches], dim=2)

    # ---- patch-blocked evaluation and feedback --------------------------------------
    def _evaluate(self, literals: Tensor, empty_value: bool) -> Tuple[Tensor, Any]:  # type: ignore[override]
        B, P, L = literals.shape
        Pb = self._patch_block(P)
        if Pb >= P:
            return super()._evaluate(literals, empty_value)
        C = self.n_clauses_total  # type: ignore[attr-defined]
        matches = torch.empty(B, P, C, dtype=torch.bool, device=literals.device)
        for sl in chunk_indices(P, Pb):
            blk = literals[:, sl]
            n = blk.shape[1]
            matches[:, sl] = F.clause_outputs(
                blk.reshape(B * n, L), self.include, self.include_count, empty_value  # type: ignore[attr-defined]
            ).view(B, n, C)
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
        B, P, L = literals.shape
        Pb = self._patch_block(P)
        if Pb >= P:
            super()._feedback_counts(literals, ctx, sel_fire_i, sel_nofire_i, sel_fire_ii, acc)
            return
        matches: Tensor = ctx
        acc.n_ib += sel_nofire_i.sum(dim=0)
        flat = literals.reshape(B * P, L)
        for sel, kind in ((sel_fire_i, "i"), (sel_fire_ii, "ii")):
            nz = torch.nonzero(sel, as_tuple=True)
            if nz[0].numel() == 0:
                continue
            b_idx, j_idx = nz
            # The argmax of U * match over all patches is the best of the per-block argmaxes,
            # so carrying the running best across blocks draws a uniformly random matching
            # patch exactly as the one-block path does. Strict ">" keeps the first index on
            # ties, like argmax.
            best_val = torch.full((b_idx.numel(),), -1.0, device=flat.device)
            best_p = torch.zeros(b_idx.numel(), dtype=torch.long, device=flat.device)
            for sl in chunk_indices(P, Pb):
                cand = matches[b_idx, sl, j_idx]  # (n_sel, block)
                val, arg = (torch.rand(cand.shape, device=cand.device) * cand).max(dim=1)
                better = val > best_val
                best_val = torch.where(better, val, best_val)
                best_p = torch.where(better, arg + sl.start, best_p)
            rows = flat[b_idx * P + best_p]  # (n_sel, 2F)
            if kind == "i":
                acc.n_true.index_add_(0, j_idx, rows)
                acc.n_false.index_add_(0, j_idx, 1.0 - rows)
            else:
                acc.n2.index_add_(0, j_idx, 1.0 - rows)

    # ---- dense introspection ------------------------------------------------------
    @torch.no_grad()
    def patch_clause_outputs(self, x, empty_value: Optional[bool] = None) -> Tensor:  # type: ignore[override]
        """Per-patch clause matches ``(B, Pd, Ph, Pw, C)`` — the tensor before the OR.

        Args:
            x: Boolean input batch ``(B, Z, D, H, W)``.
            empty_value: value of a clause with no included literals; defaults to
                ``self.training`` (``False`` when predicting).

        Returns:
            ``(B, Pd, Ph, Pw, C)`` bool.
        """
        if empty_value is None:
            empty_value = self.training  # type: ignore[attr-defined]
        xb = self._prepare(x)  # type: ignore[attr-defined]
        Pd, Ph, Pw = self._grid(xb.shape[2], xb.shape[3], xb.shape[4])
        outs = []
        for sl in chunk_indices(xb.shape[0], self._chunk_size(xb)):  # type: ignore[attr-defined]
            _, matches = self._evaluate(self._encode(xb[sl]), empty_value)  # (b, P, C)
            outs.append(matches)
        return torch.cat(outs, dim=0).view(xb.shape[0], Pd, Ph, Pw, -1)

    # ---- interpretation -----------------------------------------------------------
    def default_feature_names(self) -> List[str]:  # type: ignore[override]
        if self.input_shape is None:
            return super(_ConvMixin, self).default_feature_names()  # type: ignore[misc]
        Z = self.input_shape[0]
        kd, kh, kw = self.patch_size
        names = [
            f"p[{z},{i},{j},{k}]"
            for z in range(Z)
            for i in range(kd)
            for j in range(kh)
            for k in range(kw)
        ]
        if self.position_encoding:
            Pd, Ph, Pw = self.output_grid()
            names += [f"d>{k}" for k in range(Pd - 1)]
            names += [f"y>{k}" for k in range(Ph - 1)] + [f"x>{k}" for k in range(Pw - 1)]
        return names

    def clause_patch(self, clause: int) -> Tensor:  # type: ignore[override]
        """Visualise a clause as a ``(Z, kd, kh, kw)`` int8 map: ``1`` = voxel must be on,
        ``-1`` = voxel must be off, ``0`` = don't care."""
        self._check_initialized()  # type: ignore[attr-defined]
        Z = self.input_shape[0]  # type: ignore[index]
        kd, kh, kw = self.patch_size
        n_vox = Z * kd * kh * kw
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        pos = mask[:n_vox].to(torch.int8)
        neg = mask[Fn : Fn + n_vox].to(torch.int8)
        return (pos - neg).view(Z, kd, kh, kw)

    def clause_region(self, clause: int) -> dict:  # type: ignore[override]
        """Decode the position literals of a clause into an inclusive patch-coordinate box
        ``{"d": (lo, hi), "y": (lo, hi), "x": (lo, hi)}`` (in patch grid units). ``lo > hi``
        on any axis means the clause can never match."""
        self._check_initialized()  # type: ignore[attr-defined]
        Pd, Ph, Pw = self.output_grid()
        if not self.position_encoding:
            return {"d": (0, Pd - 1), "y": (0, Ph - 1), "x": (0, Pw - 1)}
        Z = self.input_shape[0]  # type: ignore[index]
        kd, kh, kw = self.patch_size
        n_vox = Z * kd * kh * kw
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        return {
            "d": _thermometer_range(mask, Fn, n_vox, Pd - 1),
            "y": _thermometer_range(mask, Fn, n_vox + Pd - 1, Ph - 1),
            "x": _thermometer_range(mask, Fn, n_vox + Pd - 1 + Ph - 1, Pw - 1),
        }

    def conv_extra_repr(self) -> str:  # type: ignore[override]
        s = super().conv_extra_repr()
        if self.patches_per_chunk is not None:
            s += f", patches_per_chunk={self.patches_per_chunk}"
        return s


class Conv3dTsetlinMachine(_Conv3dMixin, TsetlinMachine):
    """Convolutional multi-class Tsetlin machine for volumes (3D windows).

    The 3D generalisation of :class:`ConvTsetlinMachine`: windows slide over depth, height
    and width, so one clause can find a pattern anywhere in a volume — an MRI/CT stack, or a
    ``(time, H, W)`` sequence of frames.

    Args:
        n_classes: number of classes.
        n_clauses: clauses per class.
        T: vote margin.
        s: specificity.
        patch_size: ``(kd, kh, kw)`` window (int for a cube).
        stride: window stride (int or ``(sd, sh, sw)``).
        position_encoding: append thermometer-encoded patch coordinates (depth, row, column)
            to each patch.
        input_shape: ``(Z, D, H, W)`` of the Boolean volumes; ``None`` = infer from the first
            batch.
        patches_per_chunk: evaluate at most this many patches together (bounds memory on
            large volumes). ``None`` splits the patch axis only when one example would exceed
            ``max_chunk_elements``.
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachine`.

    Shape:
        - input: ``(B, Z, D, H, W)`` Boolean (``(B, D, H, W)`` is treated as one plane).
        - output: ``(B, n_classes)`` vote sums.

    Example:
        >>> model = Conv3dTsetlinMachine(2, 40, T=15, patch_size=3, position_encoding=False)
        >>> model.update(volumes, labels)     # volumes: (B, 1, 8, 16, 16) Boolean
        >>> model.clause_patch(0).shape       # (Z, kd, kh, kw)
    """

    def __init__(
        self,
        n_classes: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrTriple = (3, 7, 7),
        stride: IntOrTriple = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        patches_per_chunk: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._init_conv3d(patch_size, stride, position_encoding, input_shape, patches_per_chunk)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_classes, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()


class Conv3dCoalescedTsetlinMachine(_Conv3dMixin, CoalescedTsetlinMachine):
    """3D convolutional coalesced Tsetlin machine (shared clauses, per-output weights).

    Arguments as :class:`Conv3dTsetlinMachine`, with ``n_outputs`` instead of ``n_classes``
    and ``n_clauses`` counting the shared clause pool.
    """

    def __init__(
        self,
        n_outputs: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrTriple = (3, 7, 7),
        stride: IntOrTriple = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        patches_per_chunk: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._init_conv3d(patch_size, stride, position_encoding, input_shape, patches_per_chunk)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_outputs, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()


class Conv3dRegressionTsetlinMachine(_Conv3dMixin, RegressionTsetlinMachine):
    """3D convolutional regression Tsetlin machine.

    Arguments as :class:`Conv3dTsetlinMachine`, without ``n_classes``.
    """

    def __init__(
        self,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrTriple = (3, 7, 7),
        stride: IntOrTriple = 1,
        position_encoding: bool = True,
        input_shape: Optional[Tuple[int, int, int, int]] = None,
        patches_per_chunk: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._init_conv3d(patch_size, stride, position_encoding, input_shape, patches_per_chunk)
        n_features = self.conv_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_clauses, T, s, **kwargs)

    def extra_repr(self) -> str:
        return self.conv_extra_repr() + ", " + super().extra_repr()
