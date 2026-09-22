"""CNN architectures for the CIFAR-10 baselines of `experiments/convtm`.

Every builder takes ``in_ch`` so the identical architecture can be trained on

* ``in_ch=3``  — standard normalised float CIFAR-10, and
* ``in_ch=12`` — the thermometer-4 Boolean tensor the Tsetlin machines see
                 (3 colour channels x 4 thresholds; verified against
                 `experiments/mctm/.cache/cifar10_therm4.pt`, shape (50000, 12, 32, 32) bool).

Holding the architecture, optimiser and schedule fixed and changing only the input is what
makes `cnn-boolean` a *control*: the difference is the cost of Booleanization and nothing
else. Nothing in the TM image literature measures that number, which is why it is here.

Architecture provenance
-----------------------
``cnn_small``       our own 6-conv VGG-style net; the "simple CNN" reference point. Not from
                    a paper -- reported as ours, with measured params/MACs.
``vgg_cifar``       VGG-16 configuration D of [FACT: Simonyan & Zisserman, ICLR 2015,
                    arXiv:1409.1556 Table 1] with BatchNorm and a CIFAR head. The paper is
                    ImageNet-only; the CIFAR adaptation is the community-standard one.
``resnet_cifar``    the 6n+2 CIFAR ResNet of [FACT: He et al., CVPR 2016, arXiv:1512.03385
                    §4.2]: 3x3 stem, 3 stages of n basic blocks at 16/32/64 channels,
                    option-A (zero-padded) identity shortcuts, global average pool.
                    ``n=3 -> ResNet-20``, ``n=9 -> ResNet-56``.
``resnet18_cifar``  the 4-stage ImageNet BasicBlock ResNet-18 with the CIFAR stem (3x3 s1,
                    no max-pool). Not in He et al.; the standard CIFAR adaptation.
``ctm_shaped``      **the CTM's own architecture, trained by SGD.** One convolution layer
                    (patch = the CTM's patch, stride = the CTM's stride), a pointwise
                    non-linearity, a global pool over patch positions, and a linear vote.
                    ``pool='max'`` is exactly the CTM's OR over patches; ``pool='mean'`` is
                    the counting pool of PLAN.md §9.2(1); ``binary=True`` makes filters and
                    activations 1-bit, i.e. the closest differentiable analogue of a clause.
                    This arm separates *architecture* from *learning algorithm* and is the
                    cheapest decisive control in the whole programme.
``bnn_vgg``         the VGG-like binary net of [FACT: Courbariaux et al. arXiv:1511.00363 §3
                    and arXiv:1602.02830 §2]: 2x128C3-MP2-2x256C3-MP2-2x512C3-MP2-
                    1024FC-1024FC-10.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .binary import (
    BinaryActivation,
    BinaryConv2d,
    BinaryLinear,
    binary_parameter_bits,
    conv_bn_act,
)

__all__ = [
    "cnn_small",
    "vgg_cifar",
    "resnet_cifar",
    "resnet18_cifar",
    "ctm_shaped",
    "bnn_vgg",
    "mlp",
    "linear_probe",
    "count_parameters",
    "count_macs",
    "model_bits",
    "width_for_params",
]


# --------------------------------------------------------------------------------------
# simple / VGG-style
# --------------------------------------------------------------------------------------
def cnn_small(
    in_ch: int = 3,
    n_classes: int = 10,
    width: int = 64,
    dropout: float = 0.0,
    batchnorm: bool = True,
) -> nn.Module:
    """A 6-conv + 1-linear net: [w,w]-P-[2w,2w]-P-[4w,4w]-P-GAP-FC.

    ``width=64`` gives ~1.2 M parameters. ``width`` is the knob used to build the
    parameter-matched arm (see :func:`width_for_params`).
    """
    w = width

    def blk(cin: int, cout: int, pool: bool) -> List[nn.Module]:
        layers: List[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1, bias=not batchnorm)]
        if batchnorm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.ReLU(inplace=True))
        if pool:
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        return layers

    body: List[nn.Module] = []
    body += blk(in_ch, w, False) + blk(w, w, True)
    body += blk(w, 2 * w, False) + blk(2 * w, 2 * w, True)
    body += blk(2 * w, 4 * w, False) + blk(4 * w, 4 * w, True)
    return nn.Sequential(
        *body,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        nn.Linear(4 * w, n_classes),
    )


_VGG_CFG: Dict[str, Sequence[Union[int, str]]] = {
    # [FACT: arXiv:1409.1556 Table 1] configurations D (VGG-16) and E (VGG-19),
    # convolution stacks only; the 4096-4096-1000 ImageNet head is replaced by a
    # CIFAR head (see vgg_cifar).
    "vgg11": (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"),
    "vgg16": (64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
              512, 512, 512, "M", 512, 512, 512, "M"),
    "vgg19": (64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
              512, 512, 512, 512, "M", 512, 512, 512, 512, "M"),
}


def vgg_cifar(
    in_ch: int = 3,
    n_classes: int = 10,
    cfg: str = "vgg16",
    batchnorm: bool = True,
    dropout: float = 0.5,
) -> nn.Module:
    """VGG-16/19 conv stack + a 512-512-10 CIFAR head. ~14.7 M params for vgg16."""
    layers: List[nn.Module] = []
    c = in_ch
    for v in _VGG_CFG[cfg]:
        if v == "M":
            layers.append(nn.MaxPool2d(2))
            continue
        v = int(v)
        layers.append(nn.Conv2d(c, v, 3, padding=1, bias=not batchnorm))
        if batchnorm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(nn.ReLU(inplace=True))
        c = v
    head: List[nn.Module] = [nn.Flatten(), nn.Linear(512, 512), nn.ReLU(inplace=True)]
    if dropout > 0:
        head.append(nn.Dropout(dropout))
    head.append(nn.Linear(512, n_classes))
    return nn.Sequential(*layers, *head)


# --------------------------------------------------------------------------------------
# ResNets
# --------------------------------------------------------------------------------------
class _ShortcutPadA(nn.Module):
    """He et al. option (A): stride-2 subsample + zero-pad the new channels.

    This is the shortcut used for the CIFAR results of arXiv:1512.03385 §4.2 and keeps
    ResNet-20 at 0.27 M parameters. Option (B) (1x1 projection) is used by resnet18_cifar.
    """

    def __init__(self, stride: int, pad: int) -> None:
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, :, :: self.stride, :: self.stride]
        return nn.functional.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1, shortcut: str = "A") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            if shortcut == "A":
                self.short = _ShortcutPadA(stride, cout - cin)
            else:
                self.short = nn.Sequential(
                    nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)
                )

    def forward(self, x: Tensor) -> Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))


class _ResNet(nn.Module):
    def __init__(
        self,
        blocks: Sequence[int],
        channels: Sequence[int],
        in_ch: int,
        n_classes: int,
        stem: int,
        shortcut: str,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, stem, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        stages: List[nn.Module] = []
        cin = stem
        for i, (n, c) in enumerate(zip(blocks, channels)):
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                stages.append(BasicBlock(cin, c, stride, shortcut))
                cin = c
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def resnet_cifar(in_ch: int = 3, n_classes: int = 10, depth: int = 20) -> nn.Module:
    """The 6n+2 CIFAR ResNet of He et al. ``depth`` in {20, 32, 44, 56, 110}.

    [FACT: arXiv:1512.03385 §4.2] 0.27 M (20), 0.46 M (32), 0.66 M (44), 0.85 M (56),
    1.7 M (110) parameters.
    """
    if (depth - 2) % 6 != 0:
        raise ValueError("CIFAR ResNet depth must be 6n+2")
    n = (depth - 2) // 6
    return _ResNet([n, n, n], [16, 32, 64], in_ch, n_classes, stem=16, shortcut="A")


def resnet18_cifar(in_ch: int = 3, n_classes: int = 10, width: int = 64) -> nn.Module:
    """ResNet-18 (4 stages x 2 BasicBlocks, 64/128/256/512) with the CIFAR stem.

    ~11.17 M parameters at ``width=64``. Not from He et al.; the standard CIFAR adaptation
    replaces the 7x7 stride-2 stem + max-pool with a 3x3 stride-1 conv, because 32x32
    inputs cannot afford the early 4x downsample.
    """
    ch = [width, 2 * width, 4 * width, 8 * width]
    return _ResNet([2, 2, 2, 2], ch, in_ch, n_classes, stem=width, shortcut="B")


# --------------------------------------------------------------------------------------
# The CTM-shaped CNN: one conv layer, global pool over patches, linear vote
# --------------------------------------------------------------------------------------
class CtmShaped(nn.Module):
    """A CNN with exactly the convolutional Tsetlin machine's computational shape.

    CTM:  clause_j(image) = OR over patches p of  AND over literals ( patch_p )
          class score     = sum_j (+/-1) * clause_j
    Here: filter_j(image) = POOL over patches p of  act( w_j . patch_p + b_j )
          class score     = W . filter_outputs

    With ``pool='max'`` and a binary activation this is the CTM up to the learning
    algorithm: the OR over patch positions *is* a max over binary indicators. With
    ``pool='mean'`` it is the counting pool of PLAN.md §9.2(1). Comparing the two, at equal
    filter count, prices the pooling choice on a substrate where we can train exactly.
    """

    def __init__(
        self,
        in_ch: int = 12,
        n_classes: int = 10,
        n_filters: int = 2000,
        patch: Union[int, Tuple[int, int]] = 4,
        stride: int = 1,
        pool: str = "max",
        binary: bool = False,
        act_scale: bool = False,
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if pool not in ("max", "mean", "logsumexp", "topk"):
            raise ValueError(f"unknown pool {pool!r}")
        k = patch if isinstance(patch, tuple) else (patch, patch)
        self.pool = pool
        self.binary = bool(binary)
        conv_cls = BinaryConv2d if binary else nn.Conv2d
        kw = {"mode": "xnor" if act_scale else "bnn"} if binary else {}
        self.conv = conv_cls(in_ch, n_filters, k, stride=stride, bias=False, **kw)  # type: ignore[arg-type]
        self.bn = nn.BatchNorm2d(n_filters) if batchnorm else nn.Identity()
        self.act: nn.Module = BinaryActivation(scale=act_scale) if binary else nn.ReLU(inplace=True)
        lin_cls = BinaryLinear if binary else nn.Linear
        lkw = {"mode": "xnor" if act_scale else "bnn"} if binary else {}
        self.fc = lin_cls(n_filters, n_classes, bias=True, **lkw)  # type: ignore[arg-type]

    def features(self, x: Tensor) -> Tensor:
        """The pre-pool activation ``(B, C, P)`` -- one value per (image, filter, patch).

        Exposed so that the per-filter *match count* can be measured: the CNN-side analogue
        of a Tsetlin clause's ``|M_j|`` (how many of the P patch positions the clause matches
        on a given image). See ``CnnArm.match_count_stats``.
        """
        return self.act(self.bn(self.conv(x))).flatten(2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)  # (B, C, P)
        if self.pool == "max":
            h = h.amax(dim=2)
        elif self.pool == "mean":
            h = h.mean(dim=2)
        elif self.pool == "logsumexp":
            h = torch.logsumexp(h, dim=2) - torch.log(
                torch.tensor(float(h.shape[2]), device=h.device)
            )
        else:  # topk: a soft counting pool -- mean of the k strongest patches
            kk = max(1, h.shape[2] // 100)
            h = h.topk(kk, dim=2).values.mean(dim=2)
        return self.fc(h)


def ctm_shaped(in_ch: int = 12, n_classes: int = 10, **kw) -> nn.Module:
    return CtmShaped(in_ch=in_ch, n_classes=n_classes, **kw)


# --------------------------------------------------------------------------------------
# Binary VGG (BinaryConnect / BNN / XNOR)
# --------------------------------------------------------------------------------------
def bnn_vgg(
    in_ch: int = 3,
    n_classes: int = 10,
    mode: str = "bnn",
    binary_act: bool = True,
    binary_first_last: bool = False,
    width: int = 128,
    fc_width: int = 1024,
) -> nn.Module:
    """2x128C3-MP2-2x256C3-MP2-2x512C3-MP2-1024FC-1024FC-10, binarised.

    [FACT: arXiv:1511.00363 §3 / arXiv:1602.02830 §2 -- the CIFAR-10 architecture used by
    both BinaryConnect and BNN.] ~14 M parameters.

    ``mode='bwn', binary_act=False`` -> BinaryConnect (binary weights, real activations).
    ``mode='bnn', binary_act=True``  -> BNN (binary weights and activations).
    ``mode='xnor', binary_act=True`` -> XNOR-Net style with L1 scaling factors.

    ``binary_first_last=False`` keeps the first conv and the classifier in full precision,
    which is what all three papers do.
    """
    w, act_scale = width, (mode == "xnor")
    first_bin = binary_first_last
    layers: List[nn.Module] = [
        conv_bn_act(in_ch, w, binary=first_bin, mode=mode, binary_act=binary_act,
                    act_scale=act_scale),
        conv_bn_act(w, w, binary=True, mode=mode, binary_act=binary_act,
                    act_scale=act_scale, pool=2),
        conv_bn_act(w, 2 * w, binary=True, mode=mode, binary_act=binary_act,
                    act_scale=act_scale),
        conv_bn_act(2 * w, 2 * w, binary=True, mode=mode, binary_act=binary_act,
                    act_scale=act_scale, pool=2),
        conv_bn_act(2 * w, 4 * w, binary=True, mode=mode, binary_act=binary_act,
                    act_scale=act_scale),
        conv_bn_act(4 * w, 4 * w, binary=True, mode=mode, binary_act=binary_act,
                    act_scale=act_scale, pool=2),
        nn.Flatten(),
    ]
    flat = 4 * w * 4 * 4
    lin = BinaryLinear if True else nn.Linear
    layers += [
        lin(flat, fc_width, bias=False, mode=mode),
        nn.BatchNorm1d(fc_width),
        BinaryActivation() if binary_act else nn.ReLU(inplace=True),
        lin(fc_width, fc_width, bias=False, mode=mode),
        nn.BatchNorm1d(fc_width),
        BinaryActivation() if binary_act else nn.ReLU(inplace=True),
    ]
    if binary_first_last:
        layers += [BinaryLinear(fc_width, n_classes, bias=False, mode=mode),
                   nn.BatchNorm1d(n_classes)]
    else:
        layers += [nn.Linear(fc_width, n_classes), nn.BatchNorm1d(n_classes)]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# flat readers (the encoding-ceiling probes)
# --------------------------------------------------------------------------------------
def mlp(
    in_ch: int = 5832,
    n_classes: int = 10,
    spatial: Tuple[int, int] = (1, 1),
    hidden: Sequence[int] = (2048, 2048),
    dropout: float = 0.3,
    batchnorm: bool = True,
    binary: bool = False,
    mode: str = "bnn",
) -> nn.Module:
    """Flatten -> [Linear-BN-ReLU-Dropout]* -> Linear. No spatial structure exploited.

    **Why this exists.** The HOG Booleanization of the reference scripts computes ONE
    descriptor over the whole 32x32 window, so it is a flat 5 832-bit vector with no spatial
    layout left to convolve over (`code/hog.py`). A convolutional reader is therefore not
    merely unnecessary but undefined for it, and `cnn-boolean` on HOG bits -- which DR-003
    Decision 2 promoted to the *primary* HOG evidence -- has to be an MLP.

    It is also the only architecture that can read HOG bits and thermometer bits under the
    *same* inductive bias: giving the thermometer a convolution and HOG an MLP would confound
    "which encoding carries more information" with "which reader exploits more structure".
    Hence `cnn-hog-mlp` / `cnn-therm8-mlp` / `cnn-float-mlp` are an architecture-matched
    triple, and the convolutional arms sit beside them as the best-effort reading.

    ``binary=True`` binarises the hidden layers (weights, and activations via
    :class:`BinaryActivation`), giving the binary-substrate reading of the same encoding.
    """
    in_dim = int(in_ch) * int(spatial[0]) * int(spatial[1])
    lin = (lambda a, b: BinaryLinear(a, b, bias=False, mode=mode)) if binary else (
        lambda a, b: nn.Linear(a, b, bias=not batchnorm))
    layers: List[nn.Module] = [nn.Flatten()]
    prev = in_dim
    for h in hidden:
        # The first layer always stays full precision: every binary-net paper in the
        # bibliography keeps the first and last layers real [FACT: arXiv:1602.02830 S2].
        first = prev == in_dim
        layers.append(nn.Linear(prev, h, bias=not batchnorm) if (binary and first)
                      else lin(prev, h))
        if batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(BinaryActivation() if binary else nn.ReLU(inplace=True))
        if dropout > 0 and not binary:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def linear_probe(in_ch: int = 5832, n_classes: int = 10,
                 spatial: Tuple[int, int] = (1, 1)) -> nn.Module:
    """Flatten -> Linear. Multinomial logistic regression on the bits.

    The cheapest possible ceiling probe: whatever a linear map over the encoding can reach is
    a floor on what the encoding carries, and the MLP-minus-linear gap says how much of the
    encoding's information is only accessible non-linearly. Costs minutes, not hours.
    """
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(int(in_ch) * int(spatial[0]) * int(spatial[1]), n_classes))


# --------------------------------------------------------------------------------------
# capacity accounting
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    return int(sum(p.numel() for p in ps if p.requires_grad or not trainable_only))


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: Tuple[int, int, int], device: str = "cpu") -> int:
    """Multiply-accumulates for one forward pass, counted over Conv2d and Linear only.

    This is the CNN-side analogue of the TM's `literals_evaluated_per_image` and goes into
    the record's `capacity` block. BatchNorm/pooling/activation cost is ignored, as is
    conventional.
    """
    total = [0]
    hooks = []

    def conv_hook(m, inp, out):
        cin = m.in_channels // m.groups
        total[0] += int(out.numel() / out.shape[0]) * cin * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        total[0] += m.in_features * m.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval().to(device)
    model(torch.zeros(1, *input_shape, device=device))
    for h in hooks:
        h.remove()
    model.train(was_training)
    return total[0]


def model_bits(model: nn.Module) -> Dict[str, int]:
    """Storage cost of the model in bits, splitting 1-bit from 32-bit parameters.

    A Tsetlin machine's automata budget (`n_clauses_total * 2 * n_features` states) is the
    quantity this must be compared against, so the record carries both.
    """
    nb, nf = binary_parameter_bits(model)
    if nb == 0:  # a fully float model
        nf = count_parameters(model, trainable_only=False)
    return {"binary_weights": nb, "float_weights": nf, "total_bits": nb + 32 * nf}


def width_for_params(
    builder, target: int, in_ch: int = 3, lo: int = 4, hi: int = 512, **kw
) -> Tuple[int, int]:
    """Smallest ``width`` whose parameter count is >= ``target``; returns (width, params).

    Used to build the parameter-matched arm against a TM's automata budget. Monotone in
    width, so a plain scan is enough and is easier to audit than a bisection.
    """
    best = (lo, count_parameters(builder(in_ch=in_ch, width=lo, **kw)))
    for w in range(lo, hi + 1):
        p = count_parameters(builder(in_ch=in_ch, width=w, **kw))
        best = (w, p)
        if p >= target:
            break
    return best
