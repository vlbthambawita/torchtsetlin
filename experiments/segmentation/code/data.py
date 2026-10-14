"""Dataset adapters, split manifests and resized caches for the segmentation programme.

Kvasir-SEG and ACDC roots are read-only (PLAN section 4); everything derived is written under
``experiments/segmentation/{manifests,artifacts/cache}``.

Conventions
-----------
* Labels are ``uint8``; ``IGNORE = 255`` marks letterbox padding (excluded from feedback and
  metrics). Kvasir: 0 background, 1 polyp. ACDC: 0 bg, 1 RV, 2 MYO, 3 LV.
* Resized caches keep the geometry needed to undo the letterbox (``geom`` rows
  ``(H0, W0, top, left, h, w)``) so score maps can be restored to the native grid.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS = os.path.join(HERE, "manifests")
CACHE = os.path.join(HERE, "artifacts", "cache")
KVASIR_ROOT = "/work/vajira/DATA/Kvasir_seg_original/Kvasir-SEG"
ACDC_ROOT = "/work/vajira/DATA/ACDC/ACDC/database"
IGNORE = 255
SPLIT_SEED = 1234


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------
def letterbox_geometry(H0: int, W0: int, S: int) -> Tuple[int, int, int, int]:
    """Aspect-preserving fit of ``H0 x W0`` into ``S x S``: returns ``(top, left, h, w)``."""
    scale = S / max(H0, W0)
    h, w = max(1, round(H0 * scale)), max(1, round(W0 * scale))
    return (S - h) // 2, (S - w) // 2, h, w


def restore_scores(scores: torch.Tensor, geom) -> torch.Tensor:
    """``(K, S, S)`` score map on the letterboxed grid -> ``(K, H0, W0)`` on the native grid.

    Crops the padding and resizes the *scores* bilinearly; argmax is taken afterwards by the
    caller (labels are never interpolated)."""
    H0, W0, top, left, h, w = [int(v) for v in geom]
    crop = scores[:, top : top + h, left : left + w].unsqueeze(0).float()
    return torch.nn.functional.interpolate(crop, size=(H0, W0), mode="bilinear", align_corners=False)[0]


# ----------------------------------------------------------------------------------
# Kvasir-SEG
# ----------------------------------------------------------------------------------
def _dhash(img: Image.Image, n: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.float32)
    return (g[:, 1:] > g[:, :-1]).reshape(-1)


def kvasir_manifest(dhash_max: int = 12) -> Dict:
    """Pair images/masks, hash files, group duplicates and near duplicates, split 70/15/15.

    Grouping rule (frozen before any model was fitted, DECISIONS D-002): exact SHA-256
    duplicates, plus 256-bit difference-hash pairs with Hamming distance ``<= dhash_max``,
    joined transitively (union-find). Image content only; masks/bboxes are not used.
    """
    path = os.path.join(MANIFESTS, "kvasir_split.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    imgs = sorted(glob.glob(os.path.join(KVASIR_ROOT, "images", "*.jpg")))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in imgs]
    masks = [os.path.join(KVASIR_ROOT, "masks", s + ".jpg") for s in stems]
    assert all(os.path.exists(m) for m in masks), "unpaired image"
    sha = [sha256_file(p) for p in imgs]
    msha = [sha256_file(p) for p in masks]
    sizes, hashes = [], []
    for p, m in zip(imgs, masks):
        im, mk = Image.open(p), Image.open(m)
        assert im.size == mk.size, (p, im.size, mk.size)
        sizes.append(im.size)
        hashes.append(_dhash(im))
    H = np.stack(hashes).astype(np.uint8)
    dist = (H[:, None, :] != H[None, :, :]).sum(-1)
    parent = list(range(len(imgs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    near_pairs = []
    for i in range(len(imgs)):
        for j in range(i + 1, len(imgs)):
            if sha[i] == sha[j] or dist[i, j] <= dhash_max:
                near_pairs.append((stems[i], stems[j], int(dist[i, j])))
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(imgs)):
        groups.setdefault(find(i), []).append(i)
    glist = sorted(groups.values(), key=lambda g: stems[g[0]])
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(glist))
    n = len(imgs)
    targets = {"train": 0.70 * n, "val": 0.15 * n}
    split = {"train": [], "val": [], "final": []}
    for gi in order:
        g = glist[gi]
        if len(split["train"]) + len(g) <= targets["train"] + 0.5:
            split["train"] += g
        elif len(split["val"]) + len(g) <= targets["val"] + 0.5:
            split["val"] += g
        else:
            split["final"] += g
    man = {
        "dataset": "Kvasir-SEG",
        "root": KVASIR_ROOT,
        "split_seed": SPLIT_SEED,
        "grouping": f"sha256 exact + dhash256 hamming<={dhash_max}, transitive",
        "n_images": n,
        "n_groups": len(glist),
        "multi_member_groups": [[stems[i] for i in g] for g in glist if len(g) > 1],
        "near_pairs": near_pairs,
        "counts": {k: len(v) for k, v in split.items()},
        "splits": {k: sorted(stems[i] for i in v) for k, v in split.items()},
        "image_sha256": dict(zip(stems, sha)),
        "mask_sha256": dict(zip(stems, msha)),
        "sizes": dict(zip(stems, [list(s) for s in sizes])),
    }
    _save_json(man, path)
    return man


def load_kvasir_mask(stem: str) -> np.ndarray:
    """Native-resolution binary mask; JPEG decoded as grayscale and thresholded at >= 128."""
    m = np.asarray(Image.open(os.path.join(KVASIR_ROOT, "masks", stem + ".jpg")).convert("L"))
    return (m >= 128).astype(np.uint8)


def kvasir_cache(S: int, split: str) -> Dict:
    """Letterboxed ``S x S`` RGB uint8 images, labels (255 = padding) and geometry."""
    path = os.path.join(CACHE, f"kvasir_{S}_{split}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    man = kvasir_manifest()
    stems = man["splits"][split]
    X = np.zeros((len(stems), 3, S, S), np.uint8)
    Y = np.full((len(stems), S, S), IGNORE, np.uint8)
    geom = np.zeros((len(stems), 6), np.int32)
    for i, s in enumerate(stems):
        im = Image.open(os.path.join(KVASIR_ROOT, "images", s + ".jpg")).convert("RGB")
        W0, H0 = im.size
        top, left, h, w = letterbox_geometry(H0, W0, S)
        X[i, :, top : top + h, left : left + w] = np.asarray(im.resize((w, h), Image.LANCZOS)).transpose(2, 0, 1)
        m = Image.fromarray(load_kvasir_mask(s))
        Y[i, top : top + h, left : left + w] = np.asarray(m.resize((w, h), Image.NEAREST))
        geom[i] = (H0, W0, top, left, h, w)
    out = {"x": torch.from_numpy(X), "y": torch.from_numpy(Y), "geom": torch.from_numpy(geom), "ids": stems}
    os.makedirs(CACHE, exist_ok=True)
    torch.save(out, path)
    return out


# ----------------------------------------------------------------------------------
# ACDC
# ----------------------------------------------------------------------------------
def _info(pdir: str) -> Dict[str, str]:
    out = {}
    with open(os.path.join(pdir, "Info.cfg")) as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def acdc_manifest() -> Dict:
    """80/20 patient split of the training cohort stratified by group (16/4 per group);
    the 50 testing patients are the sealed final set."""
    path = os.path.join(MANIFESTS, "acdc_split.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    rng = np.random.default_rng(SPLIT_SEED)
    pats = {}
    for cohort in ("training", "testing"):
        for pdir in sorted(glob.glob(os.path.join(ACDC_ROOT, cohort, "patient*"))):
            info = _info(pdir)
            pid = os.path.basename(pdir)
            frames = []
            for ph in ("ED", "ES"):
                fr = int(info[ph])
                img = os.path.join(pdir, f"{pid}_frame{fr:02d}.nii.gz")
                gt = os.path.join(pdir, f"{pid}_frame{fr:02d}_gt.nii.gz")
                assert os.path.exists(img) and os.path.exists(gt), (img, gt)
                frames.append({"phase": ph, "frame": fr, "img": img, "gt": gt,
                               "img_sha256": sha256_file(img), "gt_sha256": sha256_file(gt)})
            pats[pid] = {"cohort": cohort, "group": info["Group"], "frames": frames}
    split = {"train": [], "val": [], "final": []}
    for grp in sorted({p["group"] for p in pats.values()}):
        ids = sorted(k for k, p in pats.items() if p["cohort"] == "training" and p["group"] == grp)
        perm = rng.permutation(len(ids))
        split["val"] += [ids[i] for i in perm[:4]]
        split["train"] += [ids[i] for i in perm[4:]]
    split["final"] = sorted(k for k, p in pats.items() if p["cohort"] == "testing")
    man = {
        "dataset": "ACDC",
        "root": ACDC_ROOT,
        "split_seed": SPLIT_SEED,
        "classes": ["background", "RV", "MYO", "LV"],
        "counts": {k: len(v) for k, v in split.items()},
        "splits": {k: sorted(v) for k, v in split.items()},
        "patients": pats,
    }
    _save_json(man, path)
    return man


def normalize_volume(v: np.ndarray) -> np.ndarray:
    """Per-volume input-only normalisation: clip to nonzero finite 1st/99th percentiles -> [0, 1]."""
    v = v.astype(np.float32)
    fin = v[np.isfinite(v) & (v != 0)]
    if fin.size == 0:
        return np.zeros_like(v)
    lo, hi = np.percentile(fin, [1, 99])
    if hi <= lo:
        return np.zeros_like(v)
    return np.clip((np.nan_to_num(v) - lo) / (hi - lo), 0, 1)


def load_acdc_volume(frame: Dict):
    import nibabel as nib

    im, gt = nib.load(frame["img"]), nib.load(frame["gt"])
    assert im.shape == gt.shape, frame["img"]
    assert np.allclose(im.affine, gt.affine, atol=1e-3), frame["img"]
    spacing = tuple(float(z) for z in im.header.get_zooms()[:3])
    return normalize_volume(np.asarray(im.dataobj)), np.asarray(gt.dataobj).astype(np.uint8), spacing


def acdc_cache(S: int, split: str) -> Dict:
    """Slices of the ED/ES volumes, letterboxed to ``S x S`` (axis 2 is the slice axis).

    In-plane array axes (0, 1) are kept as ``(H0, W0)``; no transpose, no through-plane
    interpolation. ``vol`` maps every slice to its ``(patient, phase)`` volume index."""
    path = os.path.join(CACHE, f"acdc_{S}_{split}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    man = acdc_manifest()
    X, Y, G, vol, sl, vols = [], [], [], [], [], []
    for pid in man["splits"][split]:
        for fr in man["patients"][pid]["frames"]:
            v, m, spacing = load_acdc_volume(fr)
            H0, W0, D = v.shape
            assert set(np.unique(m)).issubset({0, 1, 2, 3}), (pid, np.unique(m))
            top, left, h, w = letterbox_geometry(H0, W0, S)
            vi = len(vols)
            vols.append({"patient": pid, "phase": fr["phase"], "group": man["patients"][pid]["group"],
                         "shape": [H0, W0, D], "spacing": list(spacing)})
            for z in range(D):
                xi = np.zeros((S, S), np.float32)
                xi[top : top + h, left : left + w] = np.asarray(
                    Image.fromarray(v[:, :, z]).resize((w, h), Image.BILINEAR))
                yi = np.full((S, S), IGNORE, np.uint8)
                yi[top : top + h, left : left + w] = np.asarray(Image.fromarray(m[:, :, z]).resize((w, h), Image.NEAREST))
                X.append(np.clip(xi, 0, 1))
                Y.append(yi)
                G.append((H0, W0, top, left, h, w))
                vol.append(vi)
                sl.append(z)
    out = {
        "x": torch.from_numpy((np.stack(X)[:, None] * 255).round().astype(np.uint8)),
        "y": torch.from_numpy(np.stack(Y)),
        "geom": torch.from_numpy(np.asarray(G, np.int32)),
        "vol": torch.tensor(vol),
        "slice": torch.tensor(sl),
        "vols": vols,
    }
    os.makedirs(CACHE, exist_ok=True)
    torch.save(out, path)
    return out


def acdc_native_masks(split: str) -> List[np.ndarray]:
    """Native ground-truth volumes in the order of ``acdc_cache(..)['vols']``."""
    man = acdc_manifest()
    out = []
    for pid in man["splits"][split]:
        for fr in man["patients"][pid]["frames"]:
            out.append(load_acdc_volume(fr)[1])
    return out


def load_split(dataset: str, S: int, split: str) -> Dict:
    if split == "final" and os.environ.get("SEG_UNSEAL") != "1":
        raise PermissionError("final split is sealed; set SEG_UNSEAL=1 only after DECISIONS freeze")
    return kvasir_cache(S, split) if dataset == "kvasir" else acdc_cache(S, split)


if __name__ == "__main__":
    km = kvasir_manifest()
    print("kvasir", km["counts"], "groups", km["n_groups"], "multi", len(km["multi_member_groups"]))
    am = acdc_manifest()
    print("acdc", am["counts"])
    for S in (128, 256):
        for sp in ("train", "val"):
            d = kvasir_cache(S, sp)
            print("kvasir", S, sp, tuple(d["x"].shape))
            d = acdc_cache(S, sp)
            print("acdc", S, sp, tuple(d["x"].shape))
