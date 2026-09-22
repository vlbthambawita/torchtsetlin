"""Binarised convolution / linear layers (BinaryConnect, BNN and XNOR-Net styles).

Why this file exists
--------------------
A Tsetlin machine computes with Boolean literals and integer votes. The fairest
"same computational substrate" reference for it is not a float CNN but a network whose
weights *and* activations are 1-bit. The three published regimes we care about are:

``mode="bwn"``   BinaryConnect / Binary-Weight-Network: **weights** binarised, activations
                 stay real-valued.  [FACT: Courbariaux, Bengio & David, NIPS 2015,
                 arXiv:1511.00363]
``mode="bnn"``   Binarized Neural Network: weights **and** activations in {-1,+1}, no
                 scaling factors.  [FACT: Courbariaux, Hubara et al., arXiv:1602.02830]
``mode="xnor"``  XNOR-Net: weights and activations binarised, but each is rescaled by an
                 analytically derived L1 factor (alpha per output filter, beta/K per
                 spatial location).  [FACT: Rastegari et al., ECCV 2016, arXiv:1603.05279]

Implementation notes that matter for reproduction:

* The straight-through estimator passes the gradient through ``sign`` unchanged where
  ``|x| <= 1`` and cancels it elsewhere (the "hard tanh" STE of arXiv:1602.02830 §1.2).
  Dropping the cancellation makes BNNs diverge; it is not an optional detail.
* Latent (real-valued) weights are **clipped to [-1, 1]** after every optimiser step,
  otherwise they drift and the STE window stops carrying gradient.
* The **first convolution and the final classifier are kept full precision** in all three
  papers. We follow that and make it an explicit flag so the cost of *not* doing it can be
  measured rather than assumed.
* BatchNorm must sit between the convolution and the activation binarisation. Without it
  the pre-activation scale is arbitrary and ``sign`` throws away nearly everything.

No autograd tricks beyond the STE; everything else is ordinary PyTorch.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.autograd import Function

__all__ = [
    "SignSTE",
    "sign_ste",
    "BinaryActivation",
    "BinaryConv2d",
    "BinaryLinear",
    "clip_latent_weights",
    "binary_parameter_bits",
]


class SignSTE(Function):
    """``sign(x)`` with the hard-tanh straight-through estimator.

    Forward: ``+1`` for ``x >= 0``, ``-1`` otherwise (note ``sign(0) = +1``; torch.sign
    returns 0 there, which would silently create a third state).
    Backward: ``grad * 1[|x| <= 1]``.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:  # type: ignore[override]
        # Saving the *bool* STE window rather than x costs 1 byte/element instead of 4, and
        # the forward is built in place. This matters: the CTM-shaped arm holds a
        # (B, n_filters, 29, 29) activation, which is ~1 GB at B=128, n_filters=2000 in fp32,
        # and the naive `torch.where(x>=0, ones_like, -ones_like)` allocates three of them.
        ctx.save_for_backward(x.abs() <= 1.0)
        return (x >= 0).to(x.dtype).mul_(2.0).sub_(1.0)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        (window,) = ctx.saved_tensors
        return grad_out * window.to(grad_out.dtype)


def sign_ste(x: Tensor) -> Tensor:
    """Functional form of :class:`SignSTE`."""
    return SignSTE.apply(x)  # type: ignore[no-any-return]


class BinaryActivation(nn.Module):
    """Binarise activations to {-1, +1} (optionally with the XNOR-Net ``beta`` scale).

    ``scale=True`` reproduces XNOR-Net's input scaling: the binarised activation is
    multiplied by ``K``, the channel-averaged absolute activation smoothed over the
    convolution window. We use the per-location channel mean (the ``A`` matrix of
    arXiv:1603.05279 §3.2) without the box filter, which is the usual practical
    simplification; the box filter version is available via ``window``.
    """

    def __init__(self, scale: bool = False, window: int = 0) -> None:
        super().__init__()
        self.scale = bool(scale)
        self.window = int(window)

    def forward(self, x: Tensor) -> Tensor:
        b = sign_ste(x)
        if not self.scale:
            return b
        a = x.abs().mean(dim=1, keepdim=True)  # (B,1,H,W) channel mean of |x|
        if self.window > 1:
            k = self.window
            box = torch.full((1, 1, k, k), 1.0 / (k * k), device=x.device, dtype=x.dtype)
            a = nn.functional.conv2d(a, box, padding=k // 2)
            if a.shape[-2:] != x.shape[-2:]:
                a = a[..., : x.shape[-2], : x.shape[-1]]
        return b * a.detach()

    def extra_repr(self) -> str:
        return f"scale={self.scale}, window={self.window}"


def _binarise_weight(w: Tensor, mode: str) -> Tensor:
    """Binarise a weight tensor. ``mode`` in {'bwn', 'bnn', 'xnor'}.

    'bnn' -> plain sign. 'bwn'/'xnor' -> sign scaled by the per-output-filter mean
    absolute value ``alpha = ||W||_1 / n``, which is the optimal L2 reconstruction scale
    (arXiv:1603.05279 §3.1). BinaryConnect itself uses no scale, but every later
    binary-weight paper does; ``mode='bnn'`` is the unscaled variant.
    """
    b = sign_ste(w)
    if mode == "bnn":
        return b
    alpha = w.detach().abs().mean(dim=tuple(range(1, w.dim())), keepdim=True)
    return b * alpha


class BinaryConv2d(nn.Conv2d):
    """Conv2d whose weights are binarised in the forward pass (latent weights are real).

    ``mode`` selects the weight treatment; activation binarisation is a *separate* module
    (:class:`BinaryActivation`) so that BinaryConnect (real activations) and BNN (binary
    activations) share this layer.
    """

    def __init__(self, *args, mode: str = "bnn", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mode not in ("bwn", "bnn", "xnor"):
            raise ValueError(f"unknown binary mode {mode!r}")
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        wb = _binarise_weight(self.weight, self.mode)
        return self._conv_forward(x, wb, self.bias)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", mode={self.mode}"


class BinaryLinear(nn.Linear):
    """Linear layer with binarised weights (see :class:`BinaryConv2d`)."""

    def __init__(self, *args, mode: str = "bnn", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mode not in ("bwn", "bnn", "xnor"):
            raise ValueError(f"unknown binary mode {mode!r}")
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        wb = _binarise_weight(self.weight, self.mode)
        return nn.functional.linear(x, wb, self.bias)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", mode={self.mode}"


@torch.no_grad()
def clip_latent_weights(model: nn.Module, lo: float = -1.0, hi: float = 1.0) -> int:
    """Clamp the latent weights of every binary layer to ``[lo, hi]``.

    Call after ``optimizer.step()``. Returns the number of layers clipped.
    [FACT: arXiv:1511.00363 Algorithm 1 -- "clip(W)"].
    """
    n = 0
    for m in model.modules():
        if isinstance(m, (BinaryConv2d, BinaryLinear)):
            m.weight.clamp_(lo, hi)
            n += 1
    return n


def binary_parameter_bits(model: nn.Module) -> Tuple[int, int]:
    """``(n_binary_weights, n_float_weights)`` over the whole model.

    Used to report the model's true storage cost: a binary net holds
    ``n_binary_weights`` bits plus ``32 * n_float_weights`` bits, which is the number that
    should be compared against a Tsetlin machine's automata budget.
    """
    nb = nf = 0
    for m in model.modules():
        if isinstance(m, (BinaryConv2d, BinaryLinear)):
            nb += m.weight.numel()
            if m.bias is not None:
                nf += m.bias.numel()
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nf += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nf += sum(p.numel() for p in m.parameters())
    return nb, nf


def conv_bn_act(
    cin: int,
    cout: int,
    *,
    binary: bool,
    mode: str = "bnn",
    binary_act: bool = True,
    act_scale: bool = False,
    pool: Optional[Union[int, Tuple[int, int]]] = None,
    kernel: int = 3,
    padding: int = 1,
) -> nn.Sequential:
    """A conv block in the BNN paper's order: conv -> (pool) -> batchnorm -> activation."""
    conv: nn.Module
    if binary:
        conv = BinaryConv2d(cin, cout, kernel, padding=padding, bias=False, mode=mode)
    else:
        conv = nn.Conv2d(cin, cout, kernel, padding=padding, bias=False)
    layers: list = [conv]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    layers.append(nn.BatchNorm2d(cout))
    layers.append(BinaryActivation(scale=act_scale) if binary_act else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
