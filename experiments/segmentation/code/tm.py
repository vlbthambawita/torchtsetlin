"""Experiment-local dense TM adapter (PLAN H1, section 7).

``ExpSegTM`` subclasses the library ``SegmentationTsetlinMachine`` and only replaces the
*order* in which encoded pixels reach the unchanged feedback path
(``_accumulate_literals`` -> ``_commit``). With ``order="library"`` it reproduces the
library ``update`` exactly (parity test in ``tests/test_adapter.py``).

Orders (per image chunk, valid pixels only unless ``order="library"``):
* ``raster``   – image-major raster order of valid pixels (library order minus void slots).
* ``perm``     – uniform random permutation of the chunk's valid pixels.
* ``disperse`` – stride-interleaved: pixels sorted by ``(y mod g, x mod g)`` phase first, so
                 consecutive pixels in a commit are ``>= g`` apart and from all images.
Sampling (``sample=(n_per_image, a, b)``) replaces the order: ``n`` valid pixels per image
are drawn *with replacement* from ``p = (1-a-b) U + a p_class + b p_boundary`` and fed in a
random order. Repeats are real feedback events.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.nn import functional as F

from torchtsetlin import SegmentationTsetlinMachine, CoalescedSegmentationTsetlinMachine
from torchtsetlin.utils import chunk_indices

IGNORE = 255


def boundary_band(y: torch.Tensor, width: int = 2, ignore: int = IGNORE) -> torch.Tensor:
    """``(B, H, W)`` labels -> valid pixels within Chebyshev ``width`` of a different valid label."""
    void = y == ignore
    k = 2 * width + 1
    hi = torch.where(void, torch.full_like(y, -1), y).float().unsqueeze(1)
    lo = torch.where(void, torch.full_like(y, 10**6), y).float().unsqueeze(1)
    mx = F.max_pool2d(F.pad(hi, (width,) * 4, value=-1.0), k, stride=1)
    mn = -F.max_pool2d(F.pad(-lo, (width,) * 4, value=-1e6), k, stride=1)
    return (mx != mn).squeeze(1) & ~void


def sample_pixels(y: torch.Tensor, n: int, a: float, b: float, gen: torch.Generator, K: int) -> torch.Tensor:
    """Per image ``n`` flat pixel indices drawn from the PLAN H1 mixture (with replacement)."""
    B, H, W = y.shape
    valid = (y != IGNORE).view(B, -1).float()
    nvalid = valid.sum(1, keepdim=True).clamp_min(1)
    p = (1 - a - b) * valid / nvalid
    if a > 0:
        yc = torch.where(y == IGNORE, torch.zeros_like(y), y).view(B, -1)
        onehot = F.one_hot(yc, K).float() * valid.unsqueeze(-1)  # (B, P, K)
        cnt = onehot.sum(1)  # (B, K)
        present = (cnt > 0).float()
        pc = present / present.sum(1, keepdim=True).clamp_min(1)  # uniform over present classes
        w = (pc / cnt.clamp_min(1)).unsqueeze(1)  # per-pixel weight of its class
        p = p + a * (onehot * w).sum(-1)
    if b > 0:
        band = boundary_band(y).view(B, -1).float()
        nb = band.sum(1, keepdim=True)
        pb = torch.where(nb > 0, band / nb.clamp_min(1), valid / nvalid)  # empty band -> uniform
        p = p + b * pb
    return torch.multinomial(p, n, replacement=True, generator=gen)


class _OrderMixin:
    """Overrides ``update`` to control which encoded pixels reach each commit."""

    order: str = "library"
    disperse_grid: int = 8

    def init_order(self, order="library", sample: Optional[Tuple[int, float, float]] = None, seed=0):
        self.order, self.sample = order, sample
        dev = self.ta_state.device
        self._gen = torch.Generator(device=dev).manual_seed(int(seed))
        self.counts = {"pixels_fed": 0, "commits": 0, "type_i": torch.zeros((), device=dev), "type_ii": torch.zeros((), device=dev)}
        self._cls_hist = torch.zeros(self.n_classes if hasattr(self, "n_classes") else self.n_outputs, device=dev)

    def _apply(self, fn, *a, **k):  # keep generator on the module's device
        out = super()._apply(fn, *a, **k)
        if hasattr(self, "_gen") and self._gen.device != self.ta_state.device:
            self.init_order(self.order, self.sample, seed=int(torch.randint(0, 2**31, (1,))))
        return out

    def _select_feedback(self, votes, y, clause_out):
        ti, tii, aux = super()._select_feedback(votes, y, clause_out)
        if hasattr(self, "counts"):
            self.counts["type_i"] += ti.sum()
            self.counts["type_ii"] += tii.sum()
        return ti, tii, aux

    def _order_indices(self, yb: torch.Tensor) -> torch.Tensor:
        """Flat indices into the chunk's ``(b*P)`` pixel axis for this chunk's feedback."""
        b, H, W = yb.shape
        P = H * W
        if self.sample is not None:
            n, a, bb = self.sample
            idx = sample_pixels(yb, n, a, bb, self._gen, self._cls_hist.numel())
            flat = (idx + torch.arange(b, device=yb.device).unsqueeze(1) * P).reshape(-1)
            return flat[torch.randperm(flat.numel(), generator=self._gen, device=yb.device)]
        valid = (yb != IGNORE).reshape(-1)
        allidx = torch.arange(b * P, device=yb.device)
        if self.order == "raster":
            return allidx[valid]
        if self.order == "perm":
            v = allidx[valid]
            return v[torch.randperm(v.numel(), generator=self._gen, device=yb.device)]
        if self.order == "disperse":
            g = self.disperse_grid
            yy = torch.arange(H, device=yb.device).view(H, 1).expand(H, W)
            xx = torch.arange(W, device=yb.device).view(1, W).expand(H, W)
            phase = ((yy % g) * g + (xx % g)).reshape(-1).repeat(b)
            img = torch.arange(b, device=yb.device).repeat_interleave(P)
            cell = ((yy // g) * ((W + g - 1) // g) + (xx // g)).reshape(-1).repeat(b)
            key = (phase * (cell.max() + 1) + cell) * b + img
            o = torch.argsort(key)
            return o[valid[o]]
        raise ValueError(self.order)

    gather: bool = False

    def _can_gather(self) -> bool:
        return self.patch_size == (1, 1) and not self.position_encoding and self.stride == (1, 1)

    def _gathered_literals(self, xb: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        """Literals of selected pixels only (patch 1): identical to ``_encode(xb)[flat_idx]``
        without materialising the whole batch (``tests/test_adapter.py`` checks equality)."""
        B, Z, H, W = xb.shape
        f = xb.permute(0, 2, 3, 1).reshape(B * H * W, Z)[flat_idx].to(self.compute_dtype)
        return torch.cat([f, 1.0 - f], dim=1)

    def update(self, x, y, sequential=None):  # type: ignore[override]
        if self.order == "library":
            return super().update(x, y, sequential)
        with torch.no_grad():
            xb = self._prepare(x)
            B = xb.shape[0]
            y_t = self._coerce_targets(y, B, xb.device)
            per_commit = self.patches_per_commit
            Py, Px = self._grid(xb.shape[2], xb.shape[3])
            if self.gather and self._can_gather():
                idx = self._order_indices(y_t.view(-1, Py, Px))
                self._feed(xb, idx, y_t.reshape(-1)[idx], per_commit)
                return None
            for sl in chunk_indices(B, self._chunk_size(xb)):
                literals = self._encode(xb[sl])
                yy = y_t[sl].reshape(-1)
                idx = self._order_indices(y_t[sl].view(-1, Py, Px))
                self.counts["pixels_fed"] += int(idx.numel())
                if idx.numel() == 0:
                    continue
                yv = yy[idx]
                self._cls_hist += torch.bincount(yv.clamp(max=self._cls_hist.numel() - 1), minlength=self._cls_hist.numel()).float()
                step = idx.numel() if per_commit is None else min(idx.numel(), per_commit)
                for ps in chunk_indices(idx.numel(), step):
                    acc = self._new_accumulator()
                    self._accumulate_literals(literals[idx[ps]], yv[ps], acc)
                    self._commit(acc)
                    self.counts["commits"] += 1
            return None

    def _feed(self, xb: torch.Tensor, idx: torch.Tensor, yv: torch.Tensor, per_commit) -> None:
        """Commit selected pixels ``idx`` in order, gathering each commit's literals on demand."""
        self.counts["pixels_fed"] += int(yv.numel())
        if yv.numel() == 0:
            return
        self._cls_hist += torch.bincount(yv.clamp(max=self._cls_hist.numel() - 1), minlength=self._cls_hist.numel()).float()
        step = yv.numel() if per_commit is None else min(yv.numel(), per_commit)
        for ps in chunk_indices(yv.numel(), step):
            acc = self._new_accumulator()
            self._accumulate_literals(self._gathered_literals(xb, idx[ps]), yv[ps], acc)
            self._commit(acc)
            self.counts["commits"] += 1


class ExpSegTM(_OrderMixin, SegmentationTsetlinMachine):
    pass


class ExpCoalescedSegTM(_OrderMixin, CoalescedSegmentationTsetlinMachine):
    pass


def state_stats(model) -> Dict:
    """Clause sizes, empty clauses and automaton saturation (PLAN H1 logging)."""
    inc = model.included_mask() if hasattr(model, "included_mask") else model.include.bool()
    sizes = inc.sum(1).float()
    st = model.ta_state
    ns = model.n_states
    return {
        "mean_clause_len": float(sizes.mean()),
        "median_clause_len": float(sizes.median()),
        "empty_clauses": int((sizes == 0).sum()),
        "n_clauses_total": int(sizes.numel()),
        "saturated_frac": float(((st <= 0) | (st >= 2 * ns - 1)).float().mean()),
        "active_literals": int(inc.sum()),
    }
