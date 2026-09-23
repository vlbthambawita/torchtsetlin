"""Dense (per-pixel) Tsetlin machines for semantic segmentation.

A convolutional Tsetlin machine takes the **disjunction over patch positions**,
``c_j = OR_b c_j^b`` (Granmo et al., 2019): the clause fires if it matched *somewhere*. That
is what gives a CTM its translation tolerance, and it is exactly what a dense prediction
cannot afford — after the OR the model knows that a pattern occurred, not where.

The models here keep the patch axis instead. Every pixel contributes one ``kh x kw``
neighbourhood of the Boolean input planes, and that neighbourhood is an ordinary example of a
flat Tsetlin machine whose label is the pixel's class. Folding the patch axis into the batch
axis (``B' = B * P``) means the whole learning pipeline of
:class:`~torchtsetlin.models.TsetlinMachineBase` — vote sums, feedback selection, feedback
counting, :func:`torchtsetlin.functional.apply_feedback` — applies unchanged.

This is the ``CTM2D`` block of *CTM-UNet* (Liu & Abeyrathna, ISTM 2025) without the
disjunction, and the paper's ``K`` parallel machines are the ``K`` clause banks of a single
:class:`~torchtsetlin.models.TsetlinMachine`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as TF

from ..utils import as_long_tensor, chunk_indices
from .classifier import TsetlinMachine
from .coalesced import CoalescedTsetlinMachine
from .conv import IntOrPair, _pair

__all__ = [
    "SegmentationTsetlinMachine",
    "CoalescedSegmentationTsetlinMachine",
]

PositionEncoding = Union[bool, int]


class _DenseMixin:
    """Per-pixel clause evaluation shared by the segmentation models.

    Input images are Boolean tensors ``(B, Z, H, W)``. With the default
    ``padding="same"`` and ``stride=1`` there is one patch per pixel, so the output grid has
    the resolution of the input. The Boolean features of a patch are its ``Z*kh*kw`` pixels,
    optionally followed by thermometer-encoded patch coordinates.

    The learning loop differs from the flat one in a single respect: the number of examples
    per image is ``P = Py * Px``, which is large, and committing the feedback of all of them
    at once degrades fidelity to the classical algorithm the same way a very large mini-batch
    does. ``patches_per_commit`` bounds how many patches share one commit.
    """

    patch_size: Tuple[int, int]
    stride: Tuple[int, int]
    position_encoding: bool
    position_bits: Optional[int]
    padding: Tuple[int, int]
    input_shape: Optional[Tuple[int, int, int]]
    patches_per_commit: Optional[int]
    ignore_index: Optional[int]
    is_segmentation: bool = True

    def _init_dense(
        self,
        patch_size: IntOrPair,
        stride: IntOrPair,
        padding: Union[str, IntOrPair],
        position_encoding: PositionEncoding,
        input_shape: Optional[Tuple[int, int, int]],
        patches_per_commit: Optional[int],
        ignore_index: Optional[int],
    ) -> None:
        self.patch_size = _pair(patch_size)
        self.stride = _pair(stride)
        if isinstance(padding, str):
            if padding != "same":
                raise ValueError("padding must be 'same' or an int / (ph, pw) pair")
            self.padding = (self.patch_size[0] // 2, self.patch_size[1] // 2)
        else:
            self.padding = _pair(padding)
        if position_encoding is True:
            self.position_encoding, self.position_bits = True, None
        elif position_encoding is False:
            self.position_encoding, self.position_bits = False, None
        else:
            bits = int(position_encoding)
            if bits < 1:
                raise ValueError("position_encoding as an int must be >= 1")
            self.position_encoding, self.position_bits = True, bits
        self.input_shape = tuple(int(v) for v in input_shape) if input_shape is not None else None  # type: ignore[assignment]
        if patches_per_commit is not None and int(patches_per_commit) < 1:
            raise ValueError("patches_per_commit must be >= 1 or None")
        self.patches_per_commit = None if patches_per_commit is None else int(patches_per_commit)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        if min(self.patch_size) < 1 or min(self.stride) < 1 or min(self.padding) < 0:
            raise ValueError("patch_size and stride must be positive, padding non-negative")
        self._pos_cache: Optional[Tuple[Tuple[int, int], torch.device, Tensor]] = None

    # ---- geometry -----------------------------------------------------------------
    def _grid(self, H: int, W: int) -> Tuple[int, int]:
        """Number of patch positions ``(Py, Px)`` — the resolution of the output map."""
        kh, kw = self.patch_size
        sh, sw = self.stride
        ph, pw = self.padding
        Py = (H + 2 * ph - kh) // sh + 1
        Px = (W + 2 * pw - kw) // sw + 1
        if Py < 1 or Px < 1:
            raise ValueError(f"Input {H}x{W} is too small for patch {kh}x{kw} with padding {self.padding}")
        return Py, Px

    def output_shape(self, input_shape: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int]:
        """``(Py, Px)`` label-map resolution for an input of shape ``(Z, H, W)``."""
        shape = input_shape if input_shape is not None else self.input_shape
        if shape is None:
            raise RuntimeError("input_shape is unknown; pass it or run one batch first")
        return self._grid(int(shape[1]), int(shape[2]))

    def dense_features(self, input_shape: Tuple[int, int, int]) -> int:
        """Boolean features per patch for an input of shape ``(Z, H, W)``."""
        Z, H, W = input_shape
        n = int(Z) * self.patch_size[0] * self.patch_size[1]
        if self.position_encoding:
            Py, Px = self._grid(int(H), int(W))
            n += self._n_position_bits(Py) + self._n_position_bits(Px)
        return n

    def _n_position_bits(self, n_positions: int) -> int:
        full = max(0, n_positions - 1)
        return full if self.position_bits is None else min(self.position_bits, full)

    def _coerce_input(self, x) -> Tensor:  # type: ignore[override]
        xb = super()._coerce_input(x)  # type: ignore[misc]
        if xb.dim() == 2:  # (H, W) single single-plane image
            xb = xb.unsqueeze(0).unsqueeze(0)
        elif xb.dim() == 3:
            if self.input_shape is not None and tuple(xb.shape) == tuple(self.input_shape):
                xb = xb.unsqueeze(0)  # single (Z, H, W) image
            else:
                xb = xb.unsqueeze(1)  # (B, H, W) -> one plane
        if xb.dim() != 4:
            raise ValueError("Segmentation models expect input shaped (B, Z, H, W)")
        if self.input_shape is None:
            self.input_shape = tuple(int(v) for v in xb.shape[1:])  # type: ignore[assignment]
        elif tuple(xb.shape[1:]) != tuple(self.input_shape):
            raise ValueError(f"Expected input shape {self.input_shape}, got {tuple(xb.shape[1:])}")
        return xb

    def _infer_n_features(self, xb: Tensor) -> int:  # type: ignore[override]
        return self.dense_features(tuple(xb.shape[1:]))  # type: ignore[arg-type]

    def _patches_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        Py, Px = self._grid(xb.shape[2], xb.shape[3])
        return Py * Px

    def _chunk_elements_per_example(self, xb: Tensor) -> int:  # type: ignore[override]
        # Per image: the (P, 2F) literals and their complement, plus roughly eight (P, C)
        # intermediates (clause outputs, the feedback-probability draw, the Type I/II masks
        # and the three float selection tensors of _accumulate_literals). With
        # patches_per_commit set only the literals stay resident for the whole chunk, so this
        # is a conservative bound — measured, the slack costs no throughput (a 128px batch
        # runs within 1% of the relaxed budget) and it keeps the None case safe too.
        P = self._patches_per_example(xb)
        return P * (8 * self.n_clauses_total + 2 * self.n_literals)  # type: ignore[attr-defined]

    # ---- encoding -----------------------------------------------------------------
    def _position_bits_table(self, Py: int, Px: int, device: torch.device) -> Tensor:
        """``(P, n_y + n_x)`` thermometer bits for every patch position (row-major).

        At full resolution the thresholds are ``0 .. n-2`` so the bits read ``[coord > k]``,
        exactly as in :class:`~torchtsetlin.models.ConvTsetlinMachine`. When
        ``position_encoding`` is an integer the thresholds are spread evenly over the
        coordinate range instead, which keeps the feature count bounded on large images.
        """
        cached = self._pos_cache
        if cached is not None and cached[0] == (Py, Px) and cached[1] == device:
            return cached[2]

        def bits(n_positions: int) -> Tensor:
            n = self._n_position_bits(n_positions)
            coords = torch.arange(n_positions, device=device, dtype=torch.float32)
            if n == 0:
                return coords.new_zeros(n_positions, 0, dtype=torch.bool)
            if self.position_bits is None:
                thr = torch.arange(n, device=device, dtype=torch.float32)
            else:
                thr = torch.linspace(0, n_positions - 1, n + 2, device=device)[1:-1]
            return coords.unsqueeze(1) > thr.unsqueeze(0)

        ybits, xbits = bits(Py), bits(Px)
        table = torch.cat(
            [ybits.repeat_interleave(Px, dim=0), xbits.repeat(Py, 1)], dim=1
        )  # (P, n_y + n_x)
        self._pos_cache = ((Py, Px), device, table)
        return table

    def _encode(self, xb: Tensor) -> Tensor:  # type: ignore[override]
        """``(B, Z, H, W)`` Boolean -> literals ``(B*P, 2F)``, patches folded into the batch."""
        B, Z, H, W = xb.shape
        Py, Px = self._grid(H, W)
        patches = TF.unfold(
            xb.to(self.compute_dtype),  # type: ignore[attr-defined]
            kernel_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
        )  # (B, Z*kh*kw, P)
        patches = patches.transpose(1, 2).reshape(B * Py * Px, -1)
        if self.position_encoding:
            pos = self._position_bits_table(Py, Px, xb.device).to(self.compute_dtype)  # type: ignore[attr-defined]
            patches = torch.cat([patches, pos.repeat(B, 1)], dim=1)
        if patches.shape[1] != self.n_features:  # type: ignore[attr-defined]
            raise ValueError(
                f"Patch feature mismatch: got {patches.shape[1]}, expected {self.n_features}"  # type: ignore[attr-defined]
            )
        return torch.cat([patches, 1.0 - patches], dim=1)

    # ---- targets ------------------------------------------------------------------
    def _label_map_to_patches(self, y, n: int, device: torch.device) -> Tensor:
        """``(B, H, W)`` (or ``(B, Py, Px)``) label map -> ``(B, P)`` long."""
        ym = as_long_tensor(y, device=device)
        if ym.dim() == 2:
            ym = ym.unsqueeze(0)
        if ym.dim() != 3:
            raise ValueError("segmentation targets must be a label map shaped (B, H, W)")
        if ym.shape[0] != n:
            raise ValueError(f"x has {n} images but y has {ym.shape[0]} label maps")
        Py, Px = self.output_shape()
        if tuple(ym.shape[1:]) != (Py, Px):
            _, H, W = self.input_shape  # type: ignore[misc]
            if tuple(ym.shape[1:]) == (H, W):
                # Full-resolution labels for a strided output grid: take the label of the
                # pixel each patch is centred on.
                sh, sw = self.stride
                ym = ym[:, :: sh, :: sw][:, :Py, :Px]
            else:
                raise ValueError(
                    f"label map is {tuple(ym.shape[1:])}, expected {(Py, Px)} (output grid) "
                    f"or {(H, W)} (input resolution)"
                )
        return ym.reshape(n, Py * Px)

    def _valid_mask(self, y_flat: Tensor) -> Optional[Tensor]:
        if self.ignore_index is None:
            return None
        return y_flat != self.ignore_index

    def _mask_aux(self, aux: Any, valid: Tensor) -> Any:
        """Zero the weight-update bookkeeping of ignored pixels (default: nothing to do)."""
        return aux

    # ---- learning loop ------------------------------------------------------------
    def update(self, x, y, sequential: Optional[bool] = None) -> Tensor:  # type: ignore[override]
        """One learning step on a batch of images.

        Images are chunked so the encoded literals respect ``max_chunk_elements``, and within
        a chunk the feedback is committed every ``patches_per_commit`` patches.
        ``feedback_mode="sequential"`` (or ``sequential=True``) is equivalent to
        ``patches_per_commit=1``: the exact classical algorithm, one pixel at a time.

        Returns:
            Pre-update vote sums ``(B*P, n_outputs)`` in row-major ``(image, patch)`` order.
            :meth:`vote_map` reshapes those into ``(B, K, Py, Px)``.
        """
        with torch.no_grad():
            xb = self._prepare(x)  # type: ignore[attr-defined]
            B = xb.shape[0]
            y_t = self._coerce_targets(y, B, xb.device)  # type: ignore[attr-defined]
            if self.drop_granularity == "batch":  # type: ignore[attr-defined]
                self.resample_dropout()  # type: ignore[attr-defined]
            seq = (
                self.feedback_mode == "sequential"  # type: ignore[attr-defined]
                if sequential is None
                else bool(sequential)
            )
            per_commit = 1 if seq else self.patches_per_commit
            votes_all = []
            for sl in chunk_indices(B, self._chunk_size(xb)):  # type: ignore[attr-defined]
                literals = self._encode(xb[sl])
                yy = y_t[sl].reshape(-1, *y_t.shape[2:])
                n = literals.shape[0]
                step = n if per_commit is None else min(n, per_commit)
                for ps in chunk_indices(n, step):
                    acc = self._new_accumulator()  # type: ignore[attr-defined]
                    votes_all.append(self._accumulate_literals(literals[ps], yy[ps], acc))  # type: ignore[attr-defined]
                    self._commit(acc)  # type: ignore[attr-defined]
            return torch.cat(votes_all, dim=0)

    # ---- dense inference ----------------------------------------------------------
    @torch.no_grad()
    def vote_map(self, x) -> Tensor:
        """Per-pixel vote sums ``(B, n_outputs, Py, Px)``."""
        xb = self._coerce_input(x)
        if xb.dim() == 3:  # pragma: no cover - _coerce_input always returns 4D
            xb = xb.unsqueeze(0)
        B = xb.shape[0]
        Py, Px = self._grid(xb.shape[2], xb.shape[3])
        votes = self.forward(xb)  # type: ignore[attr-defined]  (B*P, K)
        return votes.view(B, Py, Px, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def predict(self, x) -> Tensor:  # type: ignore[override]
        """Predicted label map ``(B, Py, Px)``."""
        return self.vote_map(x).argmax(dim=1)

    @torch.no_grad()
    def clause_map(self, x, clauses: Optional[Sequence[int]] = None) -> Tensor:
        """Where each clause fires: ``(B, len(clauses), Py, Px)`` bool (prediction semantics)."""
        xb = self._coerce_input(x)
        B = xb.shape[0]
        Py, Px = self._grid(xb.shape[2], xb.shape[3])
        fires = self.evaluate_clauses(xb)  # type: ignore[attr-defined]  (B*P, C)
        if clauses is not None:
            fires = fires[:, torch.as_tensor(list(clauses), device=fires.device, dtype=torch.long)]
        return fires.view(B, Py, Px, -1).permute(0, 3, 1, 2).contiguous()

    # ---- interpretation -----------------------------------------------------------
    def default_feature_names(self) -> List[str]:  # type: ignore[override]
        if self.input_shape is None:
            return super().default_feature_names()  # type: ignore[misc]
        Z, H, W = self.input_shape
        kh, kw = self.patch_size
        oy, ox = kh // 2, kw // 2
        names = [
            f"p[{z},{i - oy:+d},{j - ox:+d}]" for z in range(Z) for i in range(kh) for j in range(kw)
        ]
        if self.position_encoding:
            Py, Px = self._grid(H, W)
            names += [f"y>{k}" for k in range(self._n_position_bits(Py))]
            names += [f"x>{k}" for k in range(self._n_position_bits(Px))]
        return names

    def pixel_feature_names(self, row: int, col: int) -> List[str]:
        """Feature names for the patch centred on ``(row, col)``, naming *absolute* pixels.

        ``p[1,14,7]`` reads "plane 1 of pixel (14, 7)", so a clause expression becomes a
        statement about named pixels of the image rather than about patch offsets.
        """
        if self.input_shape is None:
            raise RuntimeError("input_shape is unknown; run one batch first")
        Z, H, W = self.input_shape
        kh, kw = self.patch_size
        sh, sw = self.stride
        ph, pw = self.padding
        y0 = int(row) * sh - ph
        x0 = int(col) * sw - pw
        names = [
            f"p[{z},{y0 + i},{x0 + j}]" for z in range(Z) for i in range(kh) for j in range(kw)
        ]
        if self.position_encoding:
            Py, Px = self._grid(H, W)
            names += [f"y>{k}" for k in range(self._n_position_bits(Py))]
            names += [f"x>{k}" for k in range(self._n_position_bits(Px))]
        return names

    def clause_patch(self, clause: int) -> Tensor:
        """Visualise a clause as a ``(Z, kh, kw)`` int8 map: ``1`` = pixel must be on,
        ``-1`` = must be off, ``0`` = don't care."""
        self._check_initialized()  # type: ignore[attr-defined]
        Z, _, _ = self.input_shape  # type: ignore[misc]
        kh, kw = self.patch_size
        n_pix = Z * kh * kw
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        return (mask[:n_pix].to(torch.int8) - mask[Fn : Fn + n_pix].to(torch.int8)).view(Z, kh, kw)

    def clause_region(self, clause: int) -> dict:
        """Decode the position literals into an inclusive output-grid range
        ``{"y": (lo, hi), "x": (lo, hi)}``. ``lo > hi`` means the clause can never match."""
        self._check_initialized()  # type: ignore[attr-defined]
        Z, H, W = self.input_shape  # type: ignore[misc]
        Py, Px = self._grid(H, W)
        kh, kw = self.patch_size
        n_pix = Z * kh * kw
        Fn = int(self.n_features)  # type: ignore[attr-defined]
        mask = self.included_mask()[clause]  # type: ignore[attr-defined]
        if not self.position_encoding:
            return {"y": (0, Py - 1), "x": (0, Px - 1)}

        def rng(offset: int, n_positions: int) -> Tuple[int, int]:
            n = self._n_position_bits(n_positions)
            pos = mask[offset : offset + n]
            neg = mask[Fn + offset : Fn + offset + n]
            if self.position_bits is None:
                thr = torch.arange(n, device=mask.device, dtype=torch.float32)
            else:
                thr = torch.linspace(0, n_positions - 1, n + 2, device=mask.device)[1:-1]
            lo, hi = 0, n_positions - 1
            if bool(pos.any()):  # coord > thr  ->  coord >= floor(thr) + 1
                lo = int(torch.floor(thr[pos]).max().item()) + 1
            if bool(neg.any()):  # NOT (coord > thr)  ->  coord <= floor(thr)
                hi = int(torch.floor(thr[neg]).min().item())
            return lo, hi

        ny = self._n_position_bits(Py)
        return {"y": rng(n_pix, Py), "x": rng(n_pix + ny, Px)}

    def dense_extra_repr(self) -> str:
        pe: Any = self.position_bits if self.position_bits is not None else self.position_encoding
        parts = [
            f"patch_size={self.patch_size}",
            f"stride={self.stride}",
            f"padding={self.padding}",
            f"position_encoding={pe}",
            f"input_shape={self.input_shape}",
        ]
        if self.patches_per_commit is not None:
            parts.append(f"patches_per_commit={self.patches_per_commit}")
        if self.ignore_index is not None:
            parts.append(f"ignore_index={self.ignore_index}")
        return ", ".join(parts)


class SegmentationTsetlinMachine(_DenseMixin, TsetlinMachine):
    """Dense per-pixel Tsetlin machine with class-owned clauses.

    Each class owns ``n_clauses`` clauses (half voting for it, half against), exactly as in
    :class:`~torchtsetlin.models.TsetlinMachine`; the difference is that a *pixel*, not an
    image, is the example. Predictions are label maps.

    Args:
        n_classes: number of segmentation classes.
        n_clauses: clauses per class.
        T: vote margin.
        s: specificity.
        patch_size: ``(kh, kw)`` neighbourhood each pixel is classified from.
        stride: output-grid stride (``1`` keeps the input resolution).
        padding: ``"same"`` (default) or an explicit ``(ph, pw)`` pair.
        position_encoding: ``False``, ``True`` (one thermometer bit per output row/column) or
            an ``int`` giving the number of bits per axis. Position literals let clauses
            specialise by location — on scenes with a stable layout (a horizon, a scanner
            field of view) this matters a lot; see ``docs/concepts/segmentation.md``.
        input_shape: ``(Z, H, W)`` of the Boolean images; ``None`` = infer from the first batch.
        patches_per_commit: commit accumulated feedback every this many pixels
            (default ``128``). ``None`` commits once per image chunk — faster, and less
            faithful to the classical algorithm the larger the images are.
        ignore_index: label value that contributes no feedback and is excluded from metrics
            (the "void" class of CamVid / Cityscapes / Pascal VOC).
        class_feedback_p: optional ``(n_classes,)`` probability that a pixel of each class
            takes part in feedback — the imbalance lever for dense tasks, since a Tsetlin
            machine has no loss to weight. See
            :func:`torchtsetlin.data.balanced_class_probabilities`.
        **kwargs (Any): forwarded to :class:`~torchtsetlin.models.TsetlinMachine`
            (``weighted``, ``n_states``, ``max_included_literals``, ``drop_clause_p``, ...).

    Shape:
        - input: ``(B, Z, H, W)`` Boolean (``(B, H, W)`` is treated as one plane).
        - :meth:`forward`: ``(B*Py*Px, n_classes)`` vote sums (patches folded into the batch).
        - :meth:`vote_map`: ``(B, n_classes, Py, Px)``; :meth:`predict`: ``(B, Py, Px)``.

    Example:
        >>> model = SegmentationTsetlinMachine(4, n_clauses=200, T=60, s=10.0, patch_size=3)
        >>> model.update(x, labels)          # x: (B, Z, H, W) bool, labels: (B, H, W) long
        >>> model.eval(); pred = model.predict(x)         # (B, H, W)
    """

    class_feedback_p: Optional[Tensor]

    def __init__(
        self,
        n_classes: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrPair = 3,
        stride: IntOrPair = 1,
        padding: Union[str, IntOrPair] = "same",
        position_encoding: PositionEncoding = False,
        input_shape: Optional[Tuple[int, int, int]] = None,
        patches_per_commit: Optional[int] = 128,
        ignore_index: Optional[int] = None,
        class_feedback_p: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        self._init_dense(
            patch_size, stride, padding, position_encoding, input_shape, patches_per_commit, ignore_index
        )
        n_features = self.dense_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_classes, n_clauses, T, s, **kwargs)
        self.set_class_feedback_p(class_feedback_p)

    def set_class_feedback_p(self, p: Optional[Tensor]) -> None:
        """Set (or clear) the per-class probability that a pixel contributes feedback."""
        if p is None:
            self.class_feedback_p = None
            return
        t = torch.as_tensor(p, dtype=torch.float32, device=self.ta_state.device).view(-1)
        if t.numel() != self.n_classes:
            raise ValueError(f"class_feedback_p must have {self.n_classes} entries")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("class_feedback_p entries must lie in [0, 1]")
        self.class_feedback_p = t

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:  # type: ignore[override]
        ym = self._label_map_to_patches(y, n, device)
        ok = ym if self.ignore_index is None else ym[ym != self.ignore_index]
        if ok.numel() and (bool((ok < 0).any()) or bool((ok >= self.n_classes).any())):
            raise ValueError(f"labels must lie in [0, {self.n_classes - 1}] (or equal ignore_index)")
        return ym

    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:
        y_flat = y.reshape(-1)
        valid = self._valid_mask(y_flat)
        if valid is not None:
            y_flat = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        if self.class_feedback_p is not None:
            keep = torch.rand(y_flat.shape[0], device=y_flat.device) < self.class_feedback_p[y_flat]
            valid = keep if valid is None else (valid & keep)
        type_i, type_ii, aux = super()._select_feedback(votes, y_flat, clause_out)
        if valid is not None:
            m = valid.unsqueeze(1)
            type_i, type_ii = type_i & m, type_ii & m
        return type_i, type_ii, aux

    def extra_repr(self) -> str:
        return self.dense_extra_repr() + ", " + super().extra_repr()


class CoalescedSegmentationTsetlinMachine(_DenseMixin, CoalescedTsetlinMachine):
    """Dense per-pixel Tsetlin machine with one shared clause pool and per-class weights.

    The coalesced output layer (Glimsdal & Granmo, 2021) is a better fit than class-owned
    clauses when classes share structure — edges, textures and gradients are the same
    features whichever class they belong to — because one clause can serve every class it is
    useful for instead of being relearned per class.

    With ``multi_label=True`` the targets are a Boolean stack of masks ``(B, K, H, W)`` and
    a pixel may belong to several classes at once (overlapping structures); ``ignore_index``
    does not apply in that mode.

    Args:
        n_outputs: number of classes (or of overlapping masks in multi-label mode).
        n_clauses: total number of shared clauses.
        T: vote margin.
        s: specificity.
        **kwargs (Any): the dense arguments of
            :class:`~torchtsetlin.models.SegmentationTsetlinMachine` plus everything
            :class:`~torchtsetlin.models.CoalescedTsetlinMachine` accepts.
    """

    def __init__(
        self,
        n_outputs: int,
        n_clauses: int,
        T: float,
        s: float = 10.0,
        *,
        patch_size: IntOrPair = 3,
        stride: IntOrPair = 1,
        padding: Union[str, IntOrPair] = "same",
        position_encoding: PositionEncoding = False,
        input_shape: Optional[Tuple[int, int, int]] = None,
        patches_per_commit: Optional[int] = 128,
        ignore_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        self._init_dense(
            patch_size, stride, padding, position_encoding, input_shape, patches_per_commit, ignore_index
        )
        n_features = self.dense_features(self.input_shape) if self.input_shape else None
        super().__init__(n_features, n_outputs, n_clauses, T, s, **kwargs)
        if self.multi_label and self.ignore_index is not None:
            raise ValueError("ignore_index is not supported in multi-label mode")

    def _coerce_targets(self, y, n: int, device: torch.device) -> Tensor:  # type: ignore[override]
        if not self.multi_label:
            ym = self._label_map_to_patches(y, n, device)
            ok = ym if self.ignore_index is None else ym[ym != self.ignore_index]
            if ok.numel() and (bool((ok < 0).any()) or bool((ok >= self.n_outputs).any())):
                raise ValueError(
                    f"labels must lie in [0, {self.n_outputs - 1}] (or equal ignore_index)"
                )
            return ym
        from ..utils import as_bool_tensor

        yb = as_bool_tensor(y, device=device)
        if yb.dim() == 3:  # (K, H, W) for a single image
            yb = yb.unsqueeze(0)
        if yb.dim() != 4 or yb.shape[0] != n or yb.shape[1] != self.n_outputs:
            raise ValueError(f"multi-label targets must be shaped ({n}, {self.n_outputs}, H, W)")
        Py, Px = self.output_shape()
        if tuple(yb.shape[2:]) != (Py, Px):
            sh, sw = self.stride
            yb = yb[:, :, ::sh, ::sw][:, :, :Py, :Px]
        # (B, K, Py, Px) -> (B, P, K) so that y_t[sl].reshape(-1, K) is the folded target.
        return yb.permute(0, 2, 3, 1).reshape(n, Py * Px, self.n_outputs)

    def _select_feedback(
        self, votes: Tensor, y: Tensor, clause_out: Tensor
    ) -> Tuple[Tensor, Tensor, Any]:
        if self.multi_label:
            return super()._select_feedback(votes, y, clause_out)
        y_flat = y.reshape(-1)
        valid = self._valid_mask(y_flat)
        if valid is not None:
            y_flat = torch.where(valid, y_flat, torch.zeros_like(y_flat))
        type_i, type_ii, aux = super()._select_feedback(votes, y_flat, clause_out)
        if valid is not None:
            m = valid.unsqueeze(1)
            type_i, type_ii = type_i & m, type_ii & m
            # The coalesced weight update reads `aux` directly rather than the masked
            # feedback tensors, so ignored pixels have to be zeroed there too.
            sel_tgt, sel_neg, yy, y_neg = aux
            aux = (sel_tgt & m, sel_neg & m, yy, y_neg)
        return type_i, type_ii, aux

    @torch.no_grad()
    def predict(self, x) -> Tensor:  # type: ignore[override]
        """Label map ``(B, Py, Px)``, or a Boolean mask stack ``(B, K, Py, Px)`` in
        multi-label mode."""
        votes = self.vote_map(x)
        return votes > 0 if self.multi_label else votes.argmax(dim=1)

    def extra_repr(self) -> str:
        return self.dense_extra_repr() + ", " + super().extra_repr()
