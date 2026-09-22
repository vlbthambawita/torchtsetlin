"""CIFAR-10 loading, the fixed 45000/5000 split, and the Booleanization registry.

Implements ``code/CONTRACT.md``. Every arm in the programme -- TM and CNN alike -- gets its
data from here, so that the split is provably identical across arms: :func:`split_hash` goes
into every result record and the aggregator refuses records whose hashes disagree.

Three rules this module exists to enforce:

1. **One split, forever.** ``SPLIT_SEED = 1234``, stratified, 45000 train / 5000 val, carved
   from the official torchvision train order. The official 10000-image test set is untouched
   and is never subsetted.
2. **Never re-download, never re-encode what is already cached.** ``therm4`` / ``therm8``
   reuse ``.cache/cifar10_therm{4,8}.pt`` (written by ``experiments/mctm``); they are in the
   raw torchvision train order and are re-split here, not re-encoded.
3. **No ``.to("cpu", non_blocking=True)``, ever.** Encoders hold CPU-side thresholds; a
   non-blocking device-to-host copy into pageable memory returns before the data lands and
   the encoder thresholds garbage -- ~40% wrong bits, no error (repo ``CLAUDE.md``). All
   host-bound movement in this file is plain blocking ``.cpu()``.
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor

import torchtsetlin as tt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.environ.get("CONVTM_DATA", os.path.join(ROOT, ".data"))
CACHE = os.environ.get("CONVTM_CACHE", os.path.join(ROOT, ".cache"))

SPLIT_SEED = 1234
N_VAL = 5000
N_CLASSES = 10

# CIFAR-10 channel statistics used by every float (CNN) arm.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

_RAW_CACHE: Dict[str, Dict[str, Tensor]] = {}


# --------------------------------------------------------------------------- raw
def raw_cifar10(root: Optional[str] = None) -> Dict[str, Tensor]:
    """``{'xtr': uint8 (50000,3,32,32), 'ytr', 'xte': uint8 (10000,3,32,32), 'yte'}``.

    Read from ``.data`` (a symlink to ``experiments/mctm/.data``) with
    ``download=False`` -- this never reaches the network. Memoised per process.
    """
    root = root or DATA
    if root in _RAW_CACHE:
        return _RAW_CACHE[root]
    from torchvision.datasets import CIFAR10

    out = {}
    for train, tag in ((True, "tr"), (False, "te")):
        ds = CIFAR10(root, train=train, download=False)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # (N,3,32,32) uint8
        out[f"x{tag}"] = x
        out[f"y{tag}"] = torch.tensor(ds.targets, dtype=torch.int64)
    _RAW_CACHE[root] = out
    return out


# --------------------------------------------------------------------------- splits
def _class_permutations(labels: Tensor, seed: int) -> Dict[int, Tensor]:
    """A fixed per-class permutation of ``labels``' positions. Deterministic in ``seed``
    and independent of GPU state (an explicit CPU generator, never the global RNG)."""
    g = torch.Generator().manual_seed(int(seed))
    perms = {}
    for c in range(N_CLASSES):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        perms[c] = idx[torch.randperm(idx.numel(), generator=g)]
    return perms


def split_indices(seed: int = SPLIT_SEED, n_val: int = N_VAL) -> Tuple[Tensor, Tensor]:
    """Stratified deterministic ``(train_idx, val_idx)`` into the official 50000 train set.

    ``n_val // 10`` images per class go to validation; both index tensors are returned
    sorted so that the boolean tensors stay in a cache-friendly order and the hash below is
    insensitive to the internal permutation order.
    """
    if n_val % N_CLASSES:
        raise ValueError(f"n_val must be divisible by {N_CLASSES}")
    y = raw_cifar10()["ytr"]
    per = n_val // N_CLASSES
    perms = _class_permutations(y, seed)
    val = torch.cat([perms[c][:per] for c in range(N_CLASSES)]).sort().values
    trn = torch.cat([perms[c][per:] for c in range(N_CLASSES)]).sort().values
    assert trn.numel() + val.numel() == y.numel()
    assert torch.isin(trn, val).sum() == 0
    return trn, val


def split_hash(seed: int = SPLIT_SEED, n_val: int = N_VAL) -> str:
    """sha1 of the two index tensors. Goes into every result record; two arms whose
    ``data.split_hash`` differ are not comparable and the aggregator says so."""
    trn, val = split_indices(seed, n_val)
    h = hashlib.sha1()
    for t in (trn, val):
        h.update(t.to(torch.int64).numpy().tobytes())
    return h.hexdigest()


def subset_indices(train_idx: Tensor, n: int, seed: int = SPLIT_SEED) -> Tensor:
    """Stratified **nested** subsets of ``train_idx`` for the sample-efficiency curve.

    Nested: ``subset(1000) subset-of subset(5000) subset-of subset(10000)``, because the
    per-class order is one fixed permutation and ``n`` only moves the cut. That is what makes
    the curve a curve rather than three unrelated experiments.
    """
    if n >= train_idx.numel():
        return train_idx
    if n % N_CLASSES:
        raise ValueError(f"subset size must be divisible by {N_CLASSES}")
    y = raw_cifar10()["ytr"][train_idx]
    per = n // N_CLASSES
    g = torch.Generator().manual_seed(int(seed) + 7919)  # a different stream from the split
    out = []
    for c in range(N_CLASSES):
        pos = (y == c).nonzero(as_tuple=True)[0]
        pos = pos[torch.randperm(pos.numel(), generator=g)][:per]
        out.append(train_idx[pos])
    return torch.cat(out).sort().values


# --------------------------------------------------------------------------- label maps
# CIFAR-2: the vehicle/animal merge of CIFAR-10.
#
# **This is an assumption, and it is recorded as one** (ARMS.md A9). CSC-TM reports a
# "CIFAR-2" row and does not define it [FACT: arXiv:2301.08190 Table 3 + fn. 6 -- neither the
# merge, nor the Booleanization, nor the convolution window is stated for that row]. The
# vehicle/animal merge is the reading the theorist proposes and the one implemented here; it
# is NOT verified against the paper or against reference code, because neither states it.
# [HYPOTHESIS] any other 4-6 or 5-5 merge would give a different absolute accuracy.
#
# Why that is nevertheless safe for the experiment these arms exist for: the quantity is the
# budget DELTA between `ctm-small-cifar2-unc` and `ctm-small-cifar2-b32`, and both arms share
# whatever this map is. Only the comparison of our ABSOLUTE CIFAR-2 number against CSC-TM's
# 94.18% depends on the merge being theirs, and the report must not make that comparison
# without saying so.
#
# 0 airplane, 1 automobile, 8 ship, 9 truck  -> 0 (vehicle)
# 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse -> 1 (animal)
CIFAR2_MAP = (0, 0, 1, 1, 1, 1, 1, 1, 0, 0)

TASKS: Dict[str, Dict[str, object]] = {
    "cifar10": {"n_classes": 10, "map": None,
                "doc": "the official 10 classes"},
    "cifar2": {"n_classes": 2, "map": CIFAR2_MAP,
               "doc": "vehicle (airplane/automobile/ship/truck) = 0 vs animal = 1"},
}


def remap_labels(y: Optional[Tensor], task: str) -> Optional[Tensor]:
    """Apply a task's label map. ``cifar10`` is the identity.

    **The split is NOT re-derived.** ``split_indices`` stratifies on the *original ten*
    labels and is untouched by this, so every task shares one ``split_hash`` and a CIFAR-2
    arm is directly comparable, image for image, with a CIFAR-10 one. A 2-class stratified
    re-split would have been a different 45000/5000 partition and would have broken exactly
    the comparison these arms exist to make.
    """
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; have {sorted(TASKS)}")
    m = TASKS[task]["map"]
    if y is None or m is None:
        return y
    lut = torch.tensor(m, dtype=y.dtype, device=y.device)
    return lut[y]


# --------------------------------------------------------------------------- booleanization
def _therm(n_bits: int) -> Callable[[Tensor], Tensor]:
    """Per-channel colour thermometer, ``(N,3,32,32) uint8 -> (N,3*n_bits,32,32) bool``.

    The exact recipe behind ``.cache/cifar10_therm{4,8}.pt``: ``ToTensor``-scaled floats in
    ``[0,1]`` against ``ColorThermometerEncoder(n_bits, value_range=(0,1))``. Verified
    bit-identical to the cache by ``code/audit_data.py``.
    """

    def enc(x_u8: Tensor) -> Tensor:
        e = tt.data.ColorThermometerEncoder(n_bits=n_bits, value_range=(0.0, 1.0))
        outs = []
        for i in range(0, x_u8.shape[0], 5000):  # cpu, chunked: the float blow-up is 4x
            outs.append(e(x_u8[i : i + 5000].float() / 255.0))
        return torch.cat(outs)

    return enc


def _adaptive(block_size: int = 11, C: float = 2.0 / 255.0) -> Callable[[Tensor], Tensor]:
    """Gaussian adaptive thresholding per channel, ``-> (N,3,32,32) bool``.

    ``AdaptiveThresholdEncoder`` with OpenCV's default geometry. ``C`` is expressed in the
    encoder's own input units, so on ``[0,1]`` floats the classical ``C=2`` (of 255) is
    ``2/255``. This is the "adaptive thresholding" Booleanization of the CTM image papers;
    it is 3 planes against thermometer-4's 12, i.e. a 4x smaller literal space.
    """

    def enc(x_u8: Tensor) -> Tensor:
        e = tt.data.AdaptiveThresholdEncoder(block_size=block_size, C=C, method="gaussian")
        outs = []
        for i in range(0, x_u8.shape[0], 5000):
            outs.append(e(x_u8[i : i + 5000].float() / 255.0))
        return torch.cat(outs)

    return enc


def _hog(**kw) -> Callable[[Tensor], Tensor]:
    """HOG -> Boolean, from ``code/hog.py`` (LG-008), at the reference specification.

    32x32 window (the whole image, so this specialist is **flat**), 12x12 blocks, 4x4 stride,
    4x4 cells, **18 signed** orientation bins, gamma correction, L2-Hys 0.2, thresholded at
    ``>= 0.1``: 5 832 bits, emitted as ``(N, 5832, 1, 1)`` so the convolutional builder
    expresses it exactly with ``patch_size=1``.
    """
    import hog as hog_mod

    return hog_mod.make(**kw)


# name -> encoder callable. Open registry: call register_booleanization() to add more.
BOOLEANIZATIONS: Dict[str, Callable[[Tensor], Tensor]] = {
    "therm4": _therm(4),
    "therm8": _therm(8),
    "adaptive": _adaptive(),
    "hog": _hog(),
}
BOOLEANIZATION_DOC: Dict[str, str] = {
    "therm4": "per-channel colour thermometer, 4 bits/channel -> Z=12",
    "therm8": "per-channel colour thermometer, 8 bits/channel -> Z=24",
    "adaptive": "Gaussian adaptive threshold per channel (block 11, C=2/255) -> Z=3",
    "hog": "HOG per the reference scripts (32x32 window, 12x12/4x4/4x4, 18 signed bins, "
            "gamma, >=0.1) -> 5832 flat bits, emitted as (N,5832,1,1)",
}
# therm4/therm8 already exist as full-dataset caches written by experiments/mctm; they are
# re-split here and must never be re-encoded (0.7 / 1.5 GB and ~4 CPU-minutes each).
_MCTM_CACHE = {"therm4": "cifar10_therm4.pt", "therm8": "cifar10_therm8.pt"}


def register_booleanization(name: str, fn: Callable[[Tensor], Tensor], doc: str) -> None:
    """Add a Booleanization to the registry (``hog``, ``edges``, ...). The cache file is
    derived from ``name``, so a name must never be reused for a different encoding."""
    if name in BOOLEANIZATIONS and BOOLEANIZATIONS[name] is not fn:
        raise ValueError(f"Booleanization {name!r} is already registered; pick a new name")
    BOOLEANIZATIONS[name] = fn
    BOOLEANIZATION_DOC[name] = doc


def _boolean_full(name: str) -> Dict[str, Tensor]:
    """Full 50000+10000 Boolean dataset in raw torchvision order, from cache or encoded once."""
    if name not in BOOLEANIZATIONS:
        raise KeyError(f"unknown Booleanization {name!r}; have {sorted(BOOLEANIZATIONS)}")
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, _MCTM_CACHE.get(name, f"cifar10_{name}.pt"))
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
    else:
        raw = raw_cifar10()
        fn = BOOLEANIZATIONS[name]
        blob = {"xtr": fn(raw["xtr"]).cpu(), "ytr": raw["ytr"].clone(),
                "xte": fn(raw["xte"]).cpu(), "yte": raw["yte"].clone()}
        torch.save(blob, path)
    raw = raw_cifar10()
    # The cache carries its own labels; if they ever drift from the raw torchvision order
    # every split in the programme is silently wrong, so this is checked, not assumed.
    if not bool((blob["ytr"] == raw["ytr"]).all()) or not bool((blob["yte"] == raw["yte"]).all()):
        raise RuntimeError(f"cache {path} is not in raw CIFAR-10 order -- refusing to split it")
    return blob


def load_boolean(name: str = "therm4", device="cuda",
                 splits: Sequence[str] = ("train", "val", "test"),
                 subset: Optional[int] = None, task: str = "cifar10") -> Dict[str, object]:
    """Boolean CIFAR-10 on ``device`` under the fixed split.

    Returns ``{'xtr','ytr','xva','yva','xte','yte','name','Z','split_hash','n_*'}``; entries
    for splits not requested are ``None``. ``subset=n`` replaces the training split by the
    nested stratified subset of size ``n`` (validation and test are never subsetted).

    ``splits=('train','val')`` is what ``run_arm.py`` uses during training: the test tensor
    is not even materialised until selection is over.
    """
    blob = _boolean_full(name)
    trn, val = split_indices()
    if subset is not None:
        trn = subset_indices(trn, subset)
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; have {sorted(TASKS)}")
    out: Dict[str, object] = {"name": name, "Z": int(blob["xtr"].shape[1]),
                              "split_hash": split_hash(),
                              "n_train": int(trn.numel()), "n_val": int(val.numel()),
                              "n_test": int(blob["xte"].shape[0]),
                              "task": task, "dataset": task,
                              "n_classes": int(TASKS[task]["n_classes"])}
    for k in ("xtr", "ytr", "xva", "yva", "xte", "yte"):
        out[k] = None
    if "train" in splits:
        out["xtr"] = blob["xtr"][trn].to(device)
        out["ytr"] = remap_labels(blob["ytr"][trn], task).to(device)
    if "val" in splits:
        out["xva"] = blob["xtr"][val].to(device)
        out["yva"] = remap_labels(blob["ytr"][val], task).to(device)
    if "test" in splits:
        out["xte"] = blob["xte"].to(device)
        out["yte"] = remap_labels(blob["yte"], task).to(device)
    return out


# --------------------------------------------------------------------------- float (CNN)
def load_float(device="cuda", augment: bool = False,
               splits: Sequence[str] = ("train", "val", "test"),
               subset: Optional[int] = None, task: str = "cifar10") -> Dict[str, object]:
    """Normalised float32 CIFAR-10 under the *same* split, for the CNN baselines.

    ``augment=True`` adds an ``'augment'`` entry: a callable ``(B,3,32,32) -> (B,3,32,32)``
    applying random crop (32, pad 4, reflect) + horizontal flip **on the batch's own
    device**. It is applied per batch by the training loop, not baked into the tensors, so
    the same tensors serve the augmented and unaugmented arms.
    """
    raw = raw_cifar10()
    trn, val = split_indices()
    if subset is not None:
        trn = subset_indices(trn, subset)
    mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)

    def prep(x_u8: Tensor) -> Tensor:
        return (x_u8.to(device).float().div_(255.0) - mean) / std

    out: Dict[str, object] = {"name": "float", "Z": 3, "split_hash": split_hash(),
                              "n_train": int(trn.numel()), "n_val": int(val.numel()),
                              "n_test": int(raw["xte"].shape[0]),
                              "task": task, "dataset": task,
                              "n_classes": int(TASKS[task]["n_classes"]),
                              "mean": CIFAR_MEAN, "std": CIFAR_STD}
    for k in ("xtr", "ytr", "xva", "yva", "xte", "yte"):
        out[k] = None
    if "train" in splits:
        out["xtr"] = prep(raw["xtr"][trn])
        out["ytr"] = remap_labels(raw["ytr"][trn], task).to(device)
    if "val" in splits:
        out["xva"] = prep(raw["xtr"][val])
        out["yva"] = remap_labels(raw["ytr"][val], task).to(device)
    if "test" in splits:
        out["xte"] = prep(raw["xte"])
        out["yte"] = remap_labels(raw["yte"], task).to(device)
    if augment:
        out["augment"] = _augment_batch
    return out


def _augment_batch(x: Tensor) -> Tensor:
    """Random crop 32 (pad 4, reflect) + horizontal flip, on ``x``'s own device."""
    B = x.shape[0]
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    oy = torch.randint(0, 9, (B,), device=x.device)
    ox = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (oy.view(B, 1) + ar.view(1, 32))
    cols = (ox.view(B, 1) + ar.view(1, 32))
    bi = torch.arange(B, device=x.device).view(B, 1, 1)
    out = xp[bi, :, rows.view(B, 32, 1), cols.view(B, 1, 32)]  # (B,32,32,C)
    out = out.permute(0, 3, 1, 2).contiguous()
    flip = torch.rand(B, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out


if __name__ == "__main__":  # a one-line summary, and a smoke test of every path
    trn, val = split_indices()
    y = raw_cifar10()["ytr"]
    print(f"split_hash={split_hash()}  train={trn.numel()} val={val.numel()}")
    print("  val class counts ", torch.bincount(y[val]).tolist())
    for n in (1000, 5000, 10000):
        s = subset_indices(trn, n)
        print(f"  subset {n}: {s.numel()} imgs, classes {torch.bincount(y[s]).tolist()}")
    a, b = subset_indices(trn, 1000), subset_indices(trn, 5000)
    print("  nested:", bool(torch.isin(a, b).all()))
