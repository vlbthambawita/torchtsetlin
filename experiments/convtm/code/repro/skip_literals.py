"""Feasibility probe for `mctm-skip`: can a conv layer take literals from two sources?

Answers the DL expert's question without touching ``src/`` (C1). Two routes, both tested:

**Route A -- channel concatenation (no subclass at all).** If the skip tensor is resampled to
the clause-map grid, concatenating along the channel axis produces an ordinary
``(B, C1+Z, H', W')`` Boolean image. ``_encode`` unfolds ``Zc*kh*kw`` features per patch and
everything downstream -- ``n_features``, ``n_literals``, ``_patches_per_example``,
``_chunk_elements_per_example`` -- is derived from the real input shape, so the chunk budget
is automatically correct. Skip literals are then *local*: a patch sees the skip bits of its
own location.

**Route B -- a subclass with global skip literals.** If the skip bits must be visible to
every patch regardless of position (the way the position-encoding bits are), ``_encode`` is
overridden to append them and ``conv_features`` **must** be overridden to match. Overriding
one without the other is caught loudly by the explicit feature-count check in
``conv.py::_encode``; it cannot fail silently.

    python code/repro/skip_literals.py          # CPU, seconds
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torchtsetlin as tt  # noqa: E402

B, C1, Z, H, W, K = 8, 16, 12, 16, 16, 3      # maps, skip planes, grid, patch


def route_a() -> None:
    maps = torch.rand(B, C1, H, W) > 0.9                     # layer-1 clause maps
    skip = torch.rand(B, Z, H, W) > 0.5                      # downsampled Boolean input
    x = torch.cat([maps, skip], dim=1)                       # (B, C1+Z, H, W)
    m = tt.ConvCoalescedTsetlinMachine(10, 64, 51.2, 10.0, patch_size=K, stride=1,
                                       position_encoding=True, input_shape=(C1 + Z, H, W))
    expect = (C1 + Z) * K * K + (H - K) + (W - K)
    print(f"A  input (Z={C1 + Z}, {H}, {W}), patch {K}x{K}")
    print(f"   n_features {m.n_features} (expected {expect}), n_literals {m.n_literals}")
    print(f"   patches/example {m._patches_per_example(x)}, "
          f"chunk at batch {B}: {m._chunk_size(x)}")
    m.update(x, torch.randint(0, 10, (B,)))
    m.eval()
    print(f"   update + predict OK -> {tuple(m.predict(x).shape)}")
    assert m.n_features == expect


class SkipConvTM(tt.ConvCoalescedTsetlinMachine):
    """Route B: append ``n_skip`` per-image Boolean literals to *every* patch.

    ``conv_features`` is overridden in lockstep with ``_encode``; that is the whole contract.
    ``_chunk_elements_per_example`` needs no change because it reads ``self.n_literals``,
    which is ``2 * conv_features(input_shape)``.

    One friction point: the parent ``_encode`` asserts its own patch width against
    ``self.n_features``, which the subclass has already widened, so it cannot be *wrapped* --
    the unfold has to be reproduced here (10 lines). That is a readability cost, not a
    capability gap, and it fails loudly rather than silently.
    """

    def __init__(self, *args, n_skip: int = 0, **kwargs):
        self._n_skip = int(n_skip)
        self._skip: torch.Tensor = None  # type: ignore[assignment]
        super().__init__(*args, **kwargs)

    def conv_features(self, input_shape):
        return super().conv_features(input_shape) + self._n_skip

    def _encode(self, xb):
        from torch.nn import functional as TF
        B_, Zc, Hh, Ww = xb.shape
        Py, Px = self._grid(Hh, Ww)
        patches = TF.unfold(xb.to(self.compute_dtype), kernel_size=self.patch_size,
                            stride=self.stride).transpose(1, 2)          # (B, P, Z*kh*kw)
        if self.position_encoding:
            pos = self._position_bits(Py, Px, xb.device).to(self.compute_dtype)
            patches = torch.cat([patches, pos.unsqueeze(0).expand(B_, -1, -1)], dim=2)
        if self._n_skip:
            sk = self._skip[:B_].to(self.compute_dtype)
            patches = torch.cat([patches, sk.unsqueeze(1).expand(B_, patches.shape[1],
                                                                 self._n_skip)], dim=2)
        if patches.shape[2] != self.n_features:
            raise ValueError(f"skip encode: {patches.shape[2]} != {self.n_features}")
        return torch.cat([patches, 1.0 - patches], dim=2)

    def set_skip(self, skip):
        self._skip = skip


def route_b() -> None:
    maps = torch.rand(B, C1, H, W) > 0.9
    skip = torch.rand(B, 20) > 0.5                            # 20 global Boolean literals
    m = SkipConvTM(10, 64, 51.2, 10.0, patch_size=K, stride=1, position_encoding=True,
                   input_shape=(C1, H, W), n_skip=20)
    m.set_skip(skip)
    expect = C1 * K * K + (H - K) + (W - K) + 20
    print(f"B  input (Z={C1}, {H}, {W}) + 20 global skip literals")
    print(f"   n_features {m.n_features} (expected {expect}), n_literals {m.n_literals}")
    print(f"   chunk at batch {B}: {m._chunk_size(maps)}")
    m.update(maps, torch.randint(0, 10, (B,)))
    m.eval()
    print(f"   update + predict OK -> {tuple(m.predict(maps).shape)}")
    assert m.n_features == expect


def route_b_broken() -> None:
    """_encode overridden, conv_features not: the library must refuse, loudly.

    It does -- as a matmul shape error inside ``clause_outputs`` rather than the library's
    own feature-count check, because that check compares against the *un*widened
    ``n_features`` and passes. Loud either way; it cannot produce a wrong answer quietly.
    """

    class Broken(tt.ConvCoalescedTsetlinMachine):
        def _encode(self, xb):
            lits = super()._encode(xb)
            pad = torch.zeros(lits.shape[0], lits.shape[1], 8, dtype=lits.dtype)
            return torch.cat([lits, pad], dim=2)

    m = Broken(10, 64, 51.2, 10.0, patch_size=K, stride=1, input_shape=(C1, H, W))
    try:
        m.update(torch.rand(B, C1, H, W) > 0.9, torch.randint(0, 10, (B,)))
        print("C  MISMATCH WAS NOT CAUGHT -- this would be a silent-failure gap")
    except (ValueError, RuntimeError) as exc:
        print(f"C  mismatched override refused loudly ({type(exc).__name__}): "
              f"{str(exc)[:70]}")


if __name__ == "__main__":
    torch.manual_seed(0)
    route_a(); print(); route_b(); print(); route_b_broken()
