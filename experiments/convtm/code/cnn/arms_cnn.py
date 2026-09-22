"""CNN arm registry and the arm object that `code/run_arm.py` drives.

`code/CONTRACT.md` says an arm object implements::

    fit_epoch(xtr, ytr, batch_size) -> None
    predict(x) -> LongTensor
    capacity() -> dict
    diagnostics() -> dict

:class:`CnnArm` implements exactly that, so a CNN baseline and a Tsetlin arm are
interchangeable to the runner. `code/arms.py` picks the CNN arms up with
:func:`register_cnn_arms`.

Design decisions worth knowing about
------------------------------------
* **Augmentation is applied per batch, by the arm, using the harness's own function.**
  `data.load_float(augment=True)` hands back a callable rather than pre-augmented tensors;
  the arm calls it on each batch, on the batch's own device. `data._augment_batch` is
  channel-agnostic, so `cnn-boolean-aug` augments the *Boolean planes* with exactly the
  transformation the float arms get -- which is also precisely the operation PLAN.md
  §9.2(5) proposes for the TM side. One implementation, three families, no drift.
* **Selection is on validation.** The arm keeps the best-val weights in CPU memory and
  `restore_selected()` puts them back before the single test evaluation. The arm never
  sees test data during training; `predict` is generic and the driver decides what to feed.
* **No AMP for binary arms.** The sign STE and the latent-weight clip both assume fp32;
  autocast around them changes the binarisation threshold behaviour.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from . import models as M
from .binary import BinaryConv2d, BinaryLinear, clip_latent_weights

__all__ = ["CnnArm", "CNN_ARMS", "build_cnn_arm", "register_cnn_arms", "DEFAULTS"]


# --------------------------------------------------------------------------------------
# defaults
# --------------------------------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    "arch": "cnn_small",
    "booleanization": None,   # None -> float CIFAR-10; else 'therm4' / 'therm8' / ...
    "bool_scale": "pm1",      # how Boolean planes enter the net: '01' or 'pm1'
    "in_ch": 3,
    "n_classes": 10,
    "epochs": 120,
    "batch_size": 128,
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "schedule": "cosine",     # 'cosine' | 'he-step' | 'step' | 'const'
    "warmup_epochs": 0,
    "lr_min": 0.0,
    "augment": True,
    "loss": "ce",             # 'ce' | 'hinge' (squared hinge, the BNN paper's loss)
    "label_smoothing": 0.0,
    "amp": True,
    "grad_clip": 0.0,
    # Inference batch. The CTM-shaped arms hold a (B, n_filters, 29, 29) activation --
    # 3.4 GB at B=512, n_filters=2000 in fp32 -- so the eval batch is a real setting, not a
    # detail. `None` means 512.
    "eval_batch_size": None,
    # Spatial extent of the input. `None` -> 32x32. The driver overwrites it from the loaded
    # tensor, so this is a default, never an assumption: HOG is (5832, 1, 1).
    "input_hw": None,
    "arch_kw": {},
}


def _harness_augment() -> Optional[Callable[[Tensor], Tensor]]:
    """`code/data.py::_augment_batch` if importable, else ``None``.

    Reflect-pad 4 + random 32x32 crop + horizontal flip, on the batch's own device. Note
    this is *reflect* padding; He et al. (arXiv:1512.03385 §4.2) zero-pad. The difference is
    worth a line in the record's notes but not a second implementation.
    """
    try:
        import sys as _sys
        import os as _os

        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from data import _augment_batch  # type: ignore

        return _augment_batch
    except Exception:
        return None


def _build_model(cfg: Dict[str, Any]) -> nn.Module:
    arch, kw = cfg["arch"], dict(cfg.get("arch_kw") or {})
    in_ch, nc = int(cfg["in_ch"]), int(cfg["n_classes"])
    if arch in ("mlp", "linear"):
        # The flat readers need the spatial extent, because HOG arrives as (N, 5832, 1, 1)
        # and a thermometer as (N, 24, 32, 32). `input_hw` is set from the data by the
        # driver, so an arm can never be built against an assumed shape.
        kw.setdefault("spatial", tuple(cfg.get("input_hw") or (32, 32)))
        if arch == "mlp":
            return M.mlp(in_ch=in_ch, n_classes=nc, **kw)
        return M.linear_probe(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "cnn_small":
        return M.cnn_small(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "vgg":
        return M.vgg_cifar(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "resnet_cifar":
        return M.resnet_cifar(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "resnet18":
        return M.resnet18_cifar(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "ctm_shaped":
        return M.ctm_shaped(in_ch=in_ch, n_classes=nc, **kw)
    if arch == "bnn_vgg":
        return M.bnn_vgg(in_ch=in_ch, n_classes=nc, **kw)
    raise ValueError(f"unknown arch {arch!r}")


# --------------------------------------------------------------------------------------
# the arm
# --------------------------------------------------------------------------------------
class CnnArm:
    """A trainable CNN wrapped in the harness's arm protocol."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        device: str = "cuda",
        augment_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.cfg = {**DEFAULTS, **cfg}
        self.device = torch.device(device)
        # Prefer the harness's own augmentation so every arm in the programme -- TM and CNN,
        # float and Boolean -- applies the *same* transformation. Falling back to the local
        # copy only matters if data.py is unavailable (e.g. a standalone smoke test).
        self.augment_fn = augment_fn or _harness_augment() or self._augment
        self.model = _build_model(self.cfg).to(self.device)
        self.epochs_done = 0
        self.steps_done = 0
        self._best_val = -1.0
        self._best_state: Optional[Dict[str, Tensor]] = None
        self._best_epoch = -1
        self._train_s = 0.0
        self._steps_per_epoch: Optional[int] = None
        self._has_binary = any(
            isinstance(m, (BinaryConv2d, BinaryLinear)) for m in self.model.modules()
        )
        if self._has_binary:
            self.cfg["amp"] = False
        self.opt = self._make_optimizer()
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(self.cfg["amp"]))

    # -- optimisation ------------------------------------------------------------------
    def _make_optimizer(self) -> torch.optim.Optimizer:
        c = self.cfg
        params = [p for p in self.model.parameters() if p.requires_grad]
        if c["optimizer"] == "adam":
            return torch.optim.Adam(params, lr=c["lr"], weight_decay=c["weight_decay"])
        if c["optimizer"] == "adamw":
            return torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
        return torch.optim.SGD(
            params,
            lr=c["lr"],
            momentum=c["momentum"],
            nesterov=bool(c["nesterov"]),
            weight_decay=c["weight_decay"],
        )

    def _lr_at(self, epoch_frac: float) -> float:
        """Learning rate as a function of (fractional) epoch. Stepped per iteration."""
        c = self.cfg
        base, E = float(c["lr"]), float(c["epochs"])
        if c["warmup_epochs"] and epoch_frac < c["warmup_epochs"]:
            return base * (epoch_frac + 1e-8) / float(c["warmup_epochs"])
        sched = c["schedule"]
        if sched == "const":
            return base
        if sched == "cosine":
            t = min(1.0, max(0.0, (epoch_frac - c["warmup_epochs"]) / max(1e-8, E - c["warmup_epochs"])))
            lo = float(c["lr_min"])
            return lo + 0.5 * (base - lo) * (1.0 + math.cos(math.pi * t))
        if sched == "he-step":
            # [FACT: arXiv:1512.03385 §4.2] lr 0.1, /10 at 32k and 48k of 64k iterations,
            # i.e. at 50% and 75% of training.
            if epoch_frac >= 0.75 * E:
                return base * 0.01
            if epoch_frac >= 0.5 * E:
                return base * 0.1
            return base
        if sched == "step":  # /10 at 50%, 75%, 90%
            for frac, mult in ((0.9, 0.001), (0.75, 0.01), (0.5, 0.1)):
                if epoch_frac >= frac * E:
                    return base * mult
            return base
        raise ValueError(f"unknown schedule {self.cfg['schedule']!r}")

    # -- data plumbing -----------------------------------------------------------------
    def _prep(self, x: Tensor) -> Tensor:
        """Move a batch to the device and give it the dtype the net wants."""
        x = x.to(self.device, non_blocking=self.device.type == "cuda")
        if x.dtype == torch.bool:
            x = x.float()
            if self.cfg["bool_scale"] == "pm1":
                x = x * 2.0 - 1.0
        elif x.dtype != torch.float32:
            x = x.float()
        return x

    def _augment(self, x: Tensor) -> Tensor:
        """Random crop from 4-pixel zero padding + random horizontal flip, on device.

        [FACT: arXiv:1512.03385 §4.2 -- "4 pixels are padded on each side, and a 32x32 crop
        is randomly sampled from the padded image or its horizontal flip"]. Implemented
        batched: one crop offset and one flip decision per image.
        """
        B, _, H, W = x.shape
        pad = 4
        xp = nn.functional.pad(x, (pad, pad, pad, pad))
        oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
        ar = torch.arange(H, device=x.device)
        rows = (oy[:, None] + ar[None, :])                      # (B,H)
        cols = (ox[:, None] + ar[None, :])                      # (B,W)
        idx_b = torch.arange(B, device=x.device)[:, None, None]
        out = xp[idx_b, :, rows[:, :, None], cols[:, None, :]]  # (B,H,W,C)
        out = out.permute(0, 3, 1, 2).contiguous()
        flip = torch.rand(B, device=x.device) < 0.5
        out[flip] = out[flip].flip(-1)
        return out

    def _loss(self, logits: Tensor, y: Tensor) -> Tensor:
        if self.cfg["loss"] == "hinge":
            # Squared hinge on +/-1 targets. [FACT: arXiv:1602.02830 §2 -- BNN is trained
            # with a square hinge loss rather than cross-entropy.]
            t = -torch.ones_like(logits)
            t.scatter_(1, y.view(-1, 1), 1.0)
            return torch.clamp(1.0 - logits * t, min=0.0).pow(2).mean()
        return nn.functional.cross_entropy(
            logits, y, label_smoothing=float(self.cfg["label_smoothing"])
        )

    # -- the arm protocol --------------------------------------------------------------
    def fit_epoch(self, xtr: Tensor, ytr: Tensor, batch_size: Optional[int] = None) -> Dict[str, float]:
        """One pass over the training set. Returns {'train_acc', 'train_loss', 'lr'}."""
        bs = int(batch_size or self.cfg["batch_size"])
        n = int(xtr.shape[0])
        self._steps_per_epoch = max(1, math.ceil(n / bs))
        self.model.train()
        perm = torch.randperm(n, device=xtr.device)
        correct = total = 0
        loss_sum = 0.0
        t0 = time.time()
        last_lr = 0.0
        for i in range(0, n, bs):
            sl = perm[i : i + bs]
            xb = self._prep(xtr[sl])
            yb = ytr[sl].to(self.device, non_blocking=self.device.type == "cuda")
            if self.cfg["augment"]:
                xb = self.augment_fn(xb)
            frac = self.epochs_done + (i / max(1, n))
            last_lr = self._lr_at(frac)
            for g in self.opt.param_groups:
                g["lr"] = last_lr
            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(self.cfg["amp"])):
                logits = self.model(xb)
                loss = self._loss(logits.float(), yb)
            self.scaler.scale(loss).backward()
            if self.cfg["grad_clip"]:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["grad_clip"]))
            self.scaler.step(self.opt)
            self.scaler.update()
            if self._has_binary:
                clip_latent_weights(self.model)
            self.steps_done += 1
            loss_sum += float(loss.detach()) * yb.numel()
            correct += int((logits.detach().argmax(1) == yb).sum())
            total += int(yb.numel())
        self.epochs_done += 1
        self._train_s += time.time() - t0
        return {
            "train_acc": correct / max(1, total),
            "train_loss": loss_sum / max(1, total),
            "lr": last_lr,
        }

    def _eval_bs(self, batch_size: Optional[int] = None) -> int:
        return int(batch_size or self.cfg.get("eval_batch_size") or 512)

    @torch.no_grad()
    def predict(self, x: Tensor, batch_size: Optional[int] = None) -> Tensor:
        """Class predictions. Always in `eval()` mode -- BatchNorm statistics matter.

        Halves the batch and retries on CUDA OOM: a wide single-layer arm can train fine at
        its own batch size and still blow up on the larger inference batch, and silently
        losing a 40-minute run to that would be absurd.
        """
        self.model.eval()
        bs = self._eval_bs(batch_size)
        while True:
            try:
                out: List[Tensor] = []
                for i in range(0, int(x.shape[0]), bs):
                    xb = self._prep(x[i : i + bs])
                    with torch.amp.autocast("cuda", enabled=bool(self.cfg["amp"])):
                        out.append(self.model(xb).float().argmax(1).cpu())
                if bs != self._eval_bs(batch_size):
                    self.cfg["eval_batch_size"] = bs  # remember, don't rediscover each epoch
                return torch.cat(out)
            except torch.OutOfMemoryError:
                if bs <= 8:
                    raise
                bs //= 2
                torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor, batch_size: Optional[int] = None) -> float:
        p = self.predict(x, batch_size)
        return float((p == y.cpu()).float().mean())

    # -- validation-based selection ----------------------------------------------------
    def maybe_select(self, val_acc: float) -> bool:
        """Remember the weights if this is the best validation accuracy so far."""
        if val_acc > self._best_val:
            self._best_val = float(val_acc)
            self._best_epoch = self.epochs_done
            self._best_state = {
                k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()
            }
            return True
        return False

    def restore_selected(self) -> int:
        """Load the validation-selected weights. Returns the selected epoch (1-based)."""
        if self._best_state is None:
            return self.epochs_done
        self.model.load_state_dict({k: v.to(self.device) for k, v in self._best_state.items()})
        return self._best_epoch

    @property
    def selected_epoch(self) -> int:
        return self._best_epoch if self._best_epoch > 0 else self.epochs_done

    @property
    def best_val(self) -> float:
        return self._best_val

    # -- reporting ---------------------------------------------------------------------
    def capacity(self, input_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        hw = tuple(self.cfg.get("input_hw") or (32, 32))
        shape = input_shape or (int(self.cfg["in_ch"]), int(hw[0]), int(hw[1]))
        bits = M.model_bits(self.model)
        macs = M.count_macs(self.model, shape, device="cpu")
        self.model.to(self.device)
        return {
            "n_parameters": M.count_parameters(self.model, trainable_only=False),
            "n_trainable": M.count_parameters(self.model, trainable_only=True),
            "macs_per_image": int(macs),
            "binary_weights": bits["binary_weights"],
            "float_weights": bits["float_weights"],
            "state_bytes": int(math.ceil(bits["total_bits"] / 8)),
            # TM-comparable names so the report's capacity table lines up across families.
            # 0, not null: a CNN has no automata and no clauses, and the
            # record validator does integer comparisons on these keys.
            "n_automata": 0,
            "n_clauses_total": 0,
            "literals_evaluated_per_image": int(macs),
        }

    @torch.no_grad()
    def throughput(self, x: Tensor, batch_size: Optional[int] = None,
                   reps: int = 3) -> float:
        """Inference images/second on this machine, measured not estimated.

        Reported at `diagnostics['throughput_batch']`, because throughput without a
        batch size is not a number."""
        self.model.eval()
        batch_size = self._eval_bs(batch_size)
        xb = self._prep(x[:batch_size])
        with torch.amp.autocast("cuda", enabled=bool(self.cfg["amp"])):
            self.model(xb)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(reps):
            with torch.amp.autocast("cuda", enabled=bool(self.cfg["amp"])):
                self.model(xb)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return reps * batch_size / max(1e-9, time.time() - t0)

    @torch.no_grad()
    def match_count_stats(self, x: Tensor, batch_size: int = 32) -> Dict[str, Any]:
        """Distribution of the per-(image, filter) **match count** over patch positions.

        This is the CNN-side analogue of a Tsetlin clause's ``|M_j|`` -- the number of the
        ``P`` patch positions at which clause ``j`` matches image ``x``. It exists because
        the whole counting-pool argument turns on it: if a unit fires at essentially one
        position per image, a count carries no more information than an indicator and
        replacing the OR with a count can buy nothing.

        **Definition, stated precisely so it is not over-read.** A filter is taken to
        "match" position ``p`` when its post-BatchNorm, post-activation value is > 0. For the
        binary variant that is exactly ``sign(.) = +1``; for the ReLU variant it is the
        natural thresholded reading, but it is a *thresholded real value*, not a logical
        match, and it is not the same object as a TM clause's conjunction being satisfied.
        This measures the gradient-trained substrate, not a Tsetlin machine.

        Measured on whatever split is passed -- the caller passes **validation**, never test.
        Exact statistics via a full histogram over 0..P, not a sampled estimate.
        """
        if not isinstance(self.model, M.CtmShaped):
            return {}
        self.model.eval()
        P = None
        hist: Optional[Tensor] = None
        for i in range(0, int(x.shape[0]), batch_size):
            xb = self._prep(x[i : i + batch_size])
            h = self.model.features(xb)  # (B, C, P)
            if P is None:
                P = int(h.shape[2])
                hist = torch.zeros(P + 1, dtype=torch.float64, device=h.device)
            counts = (h > 0).sum(dim=2).reshape(-1)  # (B*C,) in 0..P
            hist += torch.bincount(counts, minlength=P + 1).to(torch.float64)  # type: ignore[operator]
            del h, counts
        assert hist is not None and P is not None
        h_cpu = hist.cpu()
        total = float(h_cpu.sum())
        firing = h_cpu.clone()
        firing[0] = 0.0                       # condition on the clause firing at all
        n_fire = float(firing.sum())

        def quantile(counts: Tensor, q: float) -> float:
            n = float(counts.sum())
            if n <= 0:
                return float("nan")
            c = torch.cumsum(counts, 0)
            return float(torch.searchsorted(c, torch.tensor(q * n)).item())

        idx = torch.arange(P + 1, dtype=torch.float64)
        return {
            "definition": "count of patch positions with post-BN post-activation value > 0",
            "split": "validation",
            "n_patches": P,
            "fire_frac": (n_fire / total) if total else float("nan"),
            # over (image, filter) pairs where the filter fires at least once:
            "median_given_firing": quantile(firing, 0.5),
            "mean_given_firing": (float((idx * firing).sum()) / n_fire) if n_fire else float("nan"),
            "p95_given_firing": quantile(firing, 0.95),
            # the theorist's falsifier threshold: is the OR ever an OR over >= 5 terms?
            "frac_ge5_given_firing": (float(firing[5:].sum()) / n_fire) if n_fire else float("nan"),
            "frac_eq1_given_firing": (float(firing[1]) / n_fire) if n_fire else float("nan"),
            "median_all_pairs": quantile(h_cpu, 0.5),
        }

    def diagnostics(self) -> Dict[str, Any]:
        """CNN-side analogues of the TM diagnostics, plus what only a CNN has."""
        with torch.no_grad():
            ws = [
                m.weight.detach()
                for m in self.model.modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))
            ]
            wabs = torch.cat([w.abs().flatten() for w in ws]) if ws else torch.zeros(1)
            dead = 0
            nfilt = 0
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    norms = m.weight.detach().flatten(1).norm(dim=1)
                    dead += int((norms < 1e-3 * norms.mean().clamp(min=1e-12)).sum())
                    nfilt += int(norms.numel())
        return {
            "epochs_done": self.epochs_done,
            "steps_done": self.steps_done,
            "train_s": round(self._train_s, 2),
            "weight_abs_median": float(wabs.median()),
            "dead_filter_frac": (dead / nfilt) if nfilt else 0.0,
            "has_binary_layers": self._has_binary,
            "amp": bool(self.cfg["amp"]),
        }


# --------------------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------------------
@dataclass
class CnnArmSpec:
    family: str
    paper: Optional[str]
    doc: str
    defaults: Dict[str, Any] = field(default_factory=dict)

    def build(self, cfg: Dict[str, Any], device: str = "cuda") -> CnnArm:
        return CnnArm({**self.defaults, **cfg}, device=device)


def _float(**kw) -> Dict[str, Any]:
    return {"booleanization": None, "in_ch": 3, **kw}


# Planes per Booleanization. Must agree with `code/data.py::BOOLEANIZATIONS`; the driver
# overrides `in_ch` from the data anyway, so a wrong entry here is loud, not silent -- but an
# `else: 24` fallback (what this was) is silently wrong for `adaptive` (3) and catastrophically
# wrong for `hog` (5832), which is now the primary HOG evidence per DR-003 Decision 2.
PLANES: Dict[str, int] = {"therm4": 12, "therm8": 24, "adaptive": 3, "hog": 5832}
# Spatial extent per Booleanization. HOG's reference configuration uses a 32x32 window, i.e.
# ONE window over the whole image, so the descriptor is flat: (N, 5832, 1, 1).
SPATIAL: Dict[str, Tuple[int, int]] = {"hog": (1, 1)}


def _boolean(name: str = "therm4", **kw) -> Dict[str, Any]:
    """Boolean input: therm4 gives Z=12 planes (3 colour channels x 4 thresholds)."""
    out: Dict[str, Any] = {"booleanization": name, "in_ch": PLANES.get(name, 24), **kw}
    if name in SPATIAL:
        out.setdefault("input_hw", SPATIAL[name])
    return out


CNN_ARMS: Dict[str, CnnArmSpec] = {
    # ---------------- float baselines, with augmentation --------------------------------
    "cnn-small": CnnArmSpec(
        family="baseline", paper=None,
        doc="6-conv VGG-style net (~1.2M params) on float CIFAR-10 with crop+flip.",
        defaults=_float(arch="cnn_small", arch_kw={"width": 64}, augment=True, epochs=120),
    ),
    "cnn-vgg": CnnArmSpec(
        family="baseline", paper="Simonyan & Zisserman 2015, arXiv:1409.1556 (CIFAR adaptation)",
        doc="VGG-16-BN conv stack + 512-512-10 head, float CIFAR-10, crop+flip.",
        defaults=_float(arch="vgg", arch_kw={"cfg": "vgg16"}, augment=True, epochs=160,
                        weight_decay=5e-4),
    ),
    "cnn-resnet18": CnnArmSpec(
        family="baseline", paper="He et al. 2016, arXiv:1512.03385 (CIFAR adaptation of ResNet-18)",
        doc="ResNet-18 with a 3x3 stride-1 CIFAR stem (~11.2M params), crop+flip.",
        defaults=_float(arch="resnet18", augment=True, epochs=160, weight_decay=5e-4),
    ),
    "cnn-resnet20": CnnArmSpec(
        family="baseline", paper="He et al. 2016, arXiv:1512.03385 §4.2",
        doc="The 0.27M-param CIFAR ResNet-20, He's own recipe (step schedule, wd 1e-4).",
        defaults=_float(arch="resnet_cifar", arch_kw={"depth": 20}, augment=True,
                        epochs=182, schedule="he-step", weight_decay=1e-4, nesterov=False),
    ),
    "cnn-resnet56": CnnArmSpec(
        family="baseline", paper="He et al. 2016, arXiv:1512.03385 §4.2",
        doc="The 0.85M-param CIFAR ResNet-56, He's own recipe.",
        defaults=_float(arch="resnet_cifar", arch_kw={"depth": 56}, augment=True,
                        epochs=182, schedule="he-step", weight_decay=1e-4, nesterov=False),
    ),

    # ---------------- the same nets WITHOUT augmentation --------------------------------
    # The TM arms start without augmentation; an augmented CNN is not a like-for-like
    # comparison. These arms are the like-for-like ones.
    "cnn-small-noaug": CnnArmSpec(
        family="baseline", paper=None,
        doc="cnn-small with no augmentation -- the like-for-like reference for the TM arms.",
        defaults=_float(arch="cnn_small", arch_kw={"width": 64}, augment=False, epochs=120),
    ),
    "cnn-vgg-noaug": CnnArmSpec(
        family="baseline", paper=None,
        doc="cnn-vgg with no augmentation.",
        defaults=_float(arch="vgg", arch_kw={"cfg": "vgg16"}, augment=False, epochs=160),
    ),
    "cnn-resnet18-noaug": CnnArmSpec(
        family="baseline", paper=None,
        doc="cnn-resnet18 with no augmentation.",
        defaults=_float(arch="resnet18", augment=False, epochs=160),
    ),
    "cnn-resnet20-noaug": CnnArmSpec(
        family="baseline", paper=None,
        doc="cnn-resnet20 with no augmentation.",
        defaults=_float(arch="resnet_cifar", arch_kw={"depth": 20}, augment=False,
                        epochs=182, schedule="he-step", weight_decay=1e-4, nesterov=False),
    ),

    # ---------------- the Booleanization control ----------------------------------------
    # Identical architecture and recipe to cnn-small / cnn-resnet18; only the input differs.
    # The gap IS the cost of Booleanization.
    "cnn-boolean": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-small on the 12-plane thermometer-4 Boolean tensor the TM sees, no aug.",
        defaults=_boolean("therm4", arch="cnn_small", arch_kw={"width": 64},
                          augment=False, epochs=120),
    ),
    "cnn-boolean-aug": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-boolean with crop+flip applied to the Boolean planes (PLAN §9.2.5).",
        defaults=_boolean("therm4", arch="cnn_small", arch_kw={"width": 64},
                          augment=True, epochs=120),
    ),
    "cnn-boolean-therm8": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-boolean on thermometer-8 (24 planes) -- does more bits recover the loss?",
        defaults=_boolean("therm8", arch="cnn_small", arch_kw={"width": 64},
                          augment=False, epochs=120),
    ),
    "cnn-boolean-resnet18": CnnArmSpec(
        family="control", paper=None,
        doc="ResNet-18 on the Boolean tensor -- Booleanization cost at high capacity.",
        defaults=_boolean("therm4", arch="resnet18", augment=False, epochs=160),
    ),

    # ---------------- binary computation ------------------------------------------------
    "cnn-binary": CnnArmSpec(
        family="control", paper="Courbariaux, Hubara et al. 2016, arXiv:1602.02830",
        doc="BNN: binary weights AND activations, VGG-like 2x128-2x256-2x512-1024-1024-10.",
        defaults=_float(arch="bnn_vgg", arch_kw={"mode": "bnn", "binary_act": True},
                        augment=False, epochs=200, optimizer="adam", lr=3e-3,
                        weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                        batch_size=128),
    ),
    "cnn-binary-aug": CnnArmSpec(
        family="control", paper="Courbariaux, Hubara et al. 2016, arXiv:1602.02830",
        doc="cnn-binary with crop+flip (the papers train without augmentation).",
        defaults=_float(arch="bnn_vgg", arch_kw={"mode": "bnn", "binary_act": True},
                        augment=True, epochs=200, optimizer="adam", lr=3e-3,
                        weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                        batch_size=128),
    ),
    "cnn-binaryconnect": CnnArmSpec(
        family="control", paper="Courbariaux, Bengio & David 2015, arXiv:1511.00363",
        doc="BinaryConnect: binary WEIGHTS, real activations -- separates the two effects.",
        defaults=_float(arch="bnn_vgg", arch_kw={"mode": "bwn", "binary_act": False},
                        augment=False, epochs=200, optimizer="adam", lr=3e-3,
                        weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                        batch_size=128),
    ),
    "cnn-binary-xnor": CnnArmSpec(
        family="control", paper="Rastegari et al. 2016, arXiv:1603.05279",
        doc="XNOR-style: binary weights and activations with L1 scaling factors.",
        defaults=_float(arch="bnn_vgg", arch_kw={"mode": "xnor", "binary_act": True},
                        augment=False, epochs=200, optimizer="adam", lr=3e-3,
                        weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                        batch_size=128),
    ),
    "cnn-binary-boolean": CnnArmSpec(
        family="control", paper=None,
        doc="BNN on the Boolean tensor: binary weights, binary activations, Boolean input. "
            "The closest CNN there is to a Tsetlin machine's substrate.",
        defaults=_boolean("therm4", arch="bnn_vgg",
                          arch_kw={"mode": "bnn", "binary_act": True},
                          augment=False, epochs=200, optimizer="adam", lr=3e-3,
                          weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                          batch_size=128),
    ),

    # ---------------- the CTM-shaped controls -------------------------------------------
    # One conv layer + a pool over patch positions + a linear vote. Same computational
    # shape as a convolutional Tsetlin machine; trained by SGD. Separates architecture from
    # learning algorithm, which nothing else in the programme does.
    "cnn-ctmshape-max": CnnArmSpec(
        family="control", paper=None,
        doc="1 conv layer (4x4 patches) -> global MAX over patches -> linear. The CTM's "
            "architecture with SGD; max over patches IS the CTM's OR.",
        defaults=_boolean("therm4", arch="ctm_shaped",
                          arch_kw={"n_filters": 2000, "patch": 4, "stride": 1, "pool": "max"},
                          augment=False, epochs=60, lr=0.05, weight_decay=1e-4,
                          batch_size=64, eval_batch_size=64),
    ),
    "cnn-ctmshape-sum": CnnArmSpec(
        family="control", paper=None,
        doc="As cnn-ctmshape-max but MEAN over patches -- the counting pool of PLAN §9.2.1, "
            "priced on a substrate we can train exactly.",
        defaults=_boolean("therm4", arch="ctm_shaped",
                          arch_kw={"n_filters": 2000, "patch": 4, "stride": 1, "pool": "mean"},
                          augment=False, epochs=60, lr=0.05, weight_decay=1e-4,
                          batch_size=64, eval_batch_size=64),
    ),
    "cnn-ctmshape-binary": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-ctmshape-max with 1-bit filters and activations: the tightest upper bound "
            "on what a single-layer CTM architecture can reach.",
        defaults=_boolean("therm4", arch="ctm_shaped",
                          arch_kw={"n_filters": 2000, "patch": 4, "stride": 1,
                                   "pool": "max", "binary": True},
                          augment=False, epochs=100, optimizer="adam", lr=3e-3,
                          weight_decay=0.0, loss="hinge", amp=False, batch_size=32,
                          eval_batch_size=32),
    ),
    "cnn-ctmshape-deep": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-ctmshape-max with a 10x10 patch -- prices receptive field at fixed depth "
            "(PLAN §9.2.4).",
        defaults=_boolean("therm4", arch="ctm_shaped",
                          arch_kw={"n_filters": 2000, "patch": 10, "stride": 1, "pool": "max"},
                          augment=False, epochs=60, lr=0.05, weight_decay=1e-4,
                          batch_size=64, eval_batch_size=64),
    ),

    # ---------------- the encoding ceiling: architecture-matched flat readers -------------
    # DR-003 Decision 2: HOG holds 11 664 literals/clause and OOMs a 24 GB card at 40 000
    # clauses, so the two-curve C-2 design is not runnable and `cnn-boolean` on HOG bits is
    # the PRIMARY HOG evidence. It is also the theorist's own named falsifier for the
    # encoding-ceiling claim.
    #
    # HOG is FLAT -- one 32x32 window, 5 832 bits, no spatial layout (code/hog.py) -- so its
    # reader must be an MLP. To keep "which encoding carries more class information" separate
    # from "which reader exploits more structure", the thermometer and the raw float get the
    # SAME MLP. That triple is the measurement; the convolutional arms above are the
    # best-effort reading of each encoding and are reported beside it, never instead of it.
    "cnn-hog-mlp": CnnArmSpec(
        family="control", paper=None,
        doc="2x2048 MLP on the 5832 HOG bits of the reference scripts -- the primary HOG "
            "evidence (DR-003 Dec. 2) and the ceiling of what that encoding carries.",
        defaults=_boolean("hog", arch="mlp", arch_kw={"hidden": (2048, 2048), "dropout": 0.3},
                          augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                          weight_decay=1e-2, schedule="cosine", batch_size=256, amp=True),
    ),
    "cnn-hog-linear": CnnArmSpec(
        family="control", paper=None,
        doc="Multinomial logistic regression on the 5832 HOG bits -- the cheapest floor on "
            "what the encoding carries, and the linear/non-linear split when read with "
            "cnn-hog-mlp.",
        defaults=_boolean("hog", arch="linear", augment=False, epochs=40, optimizer="adamw",
                          lr=3e-3, weight_decay=1e-3, schedule="cosine", batch_size=256),
    ),
    "cnn-therm8-mlp": CnnArmSpec(
        family="control", paper=None,
        doc="The same 2x2048 MLP on the flattened 24x32x32 thermometer-8 tensor -- HOG's "
            "architecture-matched counterpart, and the TM flagship's own encoding.",
        defaults=_boolean("therm8", arch="mlp", arch_kw={"hidden": (2048, 2048), "dropout": 0.3},
                          augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                          weight_decay=1e-2, schedule="cosine", batch_size=256, amp=True),
    ),
    "cnn-therm8-linear": CnnArmSpec(
        family="control", paper=None,
        doc="Logistic regression on the flattened thermometer-8 tensor -- HOG's linear twin.",
        defaults=_boolean("therm8", arch="linear", augment=False, epochs=40, optimizer="adamw",
                          lr=3e-3, weight_decay=1e-3, schedule="cosine", batch_size=256),
    ),
    "cnn-float-mlp": CnnArmSpec(
        family="control", paper=None,
        doc="The same 2x2048 MLP on normalised float pixels -- the no-Booleanization end of "
            "the architecture-matched triple.",
        defaults=_float(arch="mlp", arch_kw={"hidden": (2048, 2048), "dropout": 0.3},
                        augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                        weight_decay=1e-2, schedule="cosine", batch_size=256, amp=True),
    ),
    "cnn-float-linear": CnnArmSpec(
        family="control", paper=None,
        doc="Logistic regression on float pixels -- the classic ~40% CIFAR-10 floor, "
            "reproduced here so the linear row of the encoding table is complete.",
        defaults=_float(arch="linear", augment=False, epochs=40, optimizer="adamw",
                        lr=3e-3, weight_decay=1e-3, schedule="cosine", batch_size=256),
    ),
    "cnn-therm4-mlp": CnnArmSpec(
        family="control", paper=None,
        doc="The same 2x2048 MLP on thermometer-4 -- completes the bit-depth row of the "
            "encoding table (therm4 is what ctm-small and the screens use).",
        defaults=_boolean("therm4", arch="mlp", arch_kw={"hidden": (2048, 2048), "dropout": 0.3},
                          augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                          weight_decay=1e-2, schedule="cosine", batch_size=256, amp=True),
    ),
    "cnn-adaptive-mlp": CnnArmSpec(
        family="control", paper=None,
        doc="The same 2x2048 MLP on the 3-plane adaptive-threshold encoding -- the cheapest "
            "Booleanization in the registry, i.e. the bottom of the encoding range.",
        defaults=_boolean("adaptive", arch="mlp", arch_kw={"hidden": (2048, 2048),
                                                          "dropout": 0.3},
                          augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                          weight_decay=1e-2, schedule="cosine", batch_size=256, amp=True),
    ),
    "cnn-hog-mlp-binary": CnnArmSpec(
        family="control", paper="Courbariaux, Hubara et al. 2016, arXiv:1602.02830",
        doc="cnn-hog-mlp with binary hidden weights and activations -- HOG bits read on a "
            "binary substrate, i.e. the closest differentiable thing to a TM over HOG.",
        defaults=_boolean("hog", arch="mlp",
                          arch_kw={"hidden": (2048, 2048), "dropout": 0.0, "binary": True},
                          augment=False, epochs=100, optimizer="adam", lr=3e-3,
                          weight_decay=0.0, schedule="cosine", loss="hinge", amp=False,
                          batch_size=128),
    ),
    "cnn-hog-mlp-wide": CnnArmSpec(
        family="control", paper=None,
        doc="One wide hidden layer (4096) on the HOG bits instead of two of 2048 -- the "
            "depth-vs-width control for the HOG reader. Not in any job file: it runs only if "
            "cnn-hog-mlp looks capacity-limited, i.e. if its train accuracy fails to saturate.",
        defaults=_boolean("hog", arch="mlp", arch_kw={"hidden": (4096,), "dropout": 0.3},
                          augment=False, epochs=60, optimizer="adamw", lr=1e-3,
                          weight_decay=1e-2, schedule="cosine", batch_size=256),
    ),

    # ---------------- budget-matched -----------------------------------------------------
    # width 45 is not a round number: it is the smallest cnn_small width whose parameter count
    # reaches 556 514, the MEASURED `capacity.included_literals_total` of the flagship TM
    # [MEASURED: results/ctm-therm5-preflight_seed0.json] -- i.e. the number of literals that
    # arm actually keeps, which is the honest analogue of a CNN parameter. The TM's other two
    # budgets are reported beside it rather than matched, because matching them is absurd in
    # opposite directions: its `state_bytes` is 104.6 MB (26.2 M automata), 2.3x MORE storage
    # than ResNet-18's 11.2 M fp32 parameters; and its 436 M included literals per image sit
    # between cnn-small (165 M MACs) and ResNet-18 (~557 M MACs).
    "cnn-matched": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-small at width 45 (577 765 params) -- matched to the flagship TM's 556 514 "
            "included literals. Float input, no augmentation.",
        defaults=_float(arch="cnn_small", arch_kw={"width": 45}, augment=False, epochs=120),
    ),
    "cnn-matched-boolean": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-matched on thermometer-8 -- matched budget AND the flagship TM's own input. "
            "This is the single fairest CNN/TM comparison the programme can make.",
        defaults=_boolean("therm8", arch="cnn_small", arch_kw={"width": 45},
                          augment=False, epochs=120),
    ),
    "cnn-matched-therm4": CnnArmSpec(
        family="control", paper=None,
        doc="cnn-matched on thermometer-4 -- the budget-matched control for the ctm-small "
            "family, which runs therm4.",
        defaults=_boolean("therm4", arch="cnn_small", arch_kw={"width": 45},
                          augment=False, epochs=120),
    ),
}


def build_cnn_arm(name: str, cfg: Optional[Dict[str, Any]] = None, device: str = "cuda") -> CnnArm:
    if name not in CNN_ARMS:
        raise KeyError(f"unknown CNN arm {name!r}; known: {sorted(CNN_ARMS)}")
    return CNN_ARMS[name].build(cfg or {}, device=device)


def register_cnn_arms(arms: Dict[str, Any], arm_spec_cls: Optional[type] = None) -> Dict[str, Any]:
    """Merge the CNN arms into `code/arms.py::ARMS`.

    `arms.py` calls this with its own ``ArmSpec`` class; if it passes ``None`` the local
    :class:`CnnArmSpec` is registered instead (it exposes the same ``build``/``family``/
    ``paper``/``doc``/``defaults`` surface).
    """
    for name, spec in CNN_ARMS.items():
        if arm_spec_cls is None:
            arms[name] = spec
        else:
            arms[name] = arm_spec_cls(
                family=spec.family,
                paper=spec.paper,
                build=(lambda s: lambda cfg: s.build(cfg))(spec),
                defaults=dict(spec.defaults),
                doc=spec.doc,
            )
    return arms
