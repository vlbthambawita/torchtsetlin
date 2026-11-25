"""Kvasir-SEG as an instance dataset on the 128 px letterbox grid (PLAN_KVASIR §4).

* Split: ``manifests/kvasir_split.json``, copied from the segmentation programme (700 / 150 / 150,
  duplicate groups, SHA-256 per file). ``verify()`` checks the local copy of the data against it.
* Instances: native resolution, mask >= 128, 8-connected components, components under 0.1 % of
  the image area dropped; the id map is resized with nearest neighbour.
* Cache: ``.cache/data/kvasir_128.pt`` holds, per split, ``x`` uint8 (N, 3, S, S), ``inst`` uint8
  (N, S, S) with 0 background, 1..k instances and 255 on letterbox padding, and the geometry.
* The training pool is train + val (850 images: 0..699 train, 700..849 val), so that the
  cross-validation folds and the out-of-fold halves are index sets of one array.
* Augmentation: the eight symmetries of the square are exact tensor operations (``sym``,
  ``sym_boxes``), applied on the device per batch; nothing is interpolated.

Run ``python data.py`` to verify the data, build the cache, the panel and anchor manifests.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from common import CACHE, IGNORE, MANIFESTS, S, read_json, sha256_file, write_json

# The folder that holds images/ and masks/. KV_DATA wins; otherwise the first known location that
# exists (the V100 server, then the build machine). A folder that only holds a nested Kvasir-SEG/
# is accepted too.
DATA_DIRS = ("/work/vajira/DATA/Kvasir-SEG", "/work/vajira/DATA/Kvasir_seg_original/Kvasir-SEG")


def _data_dir():
    cands = [os.environ["KV_DATA"]] if os.environ.get("KV_DATA") else list(DATA_DIRS)
    for c in cands:
        for p in (Path(c), Path(c) / "Kvasir-SEG"):
            if (p / "images").is_dir() and (p / "masks").is_dir():
                return p
    return Path(cands[0])


KV_DATA = _data_dir()
CACHE_FILE = CACHE / "data" / f"kvasir_{S}.pt"
SPLITS = {"train": "train", "val": "val", "test": "final"}        # our name -> manifest name
MIN_AREA_FRAC = 0.001
KMAX = 16                                                         # most instances kept per image
PANEL_N, PANEL_SEED, FOLD_SEED = 16, 1234, 1234


def manifest():
    m = read_json(MANIFESTS / "kvasir_split.json")
    if m is None:
        raise SystemExit("manifests/kvasir_split.json is missing")
    return m


def verify(verbose=True):
    """Every image and mask of the local copy must match the SHA-256 of the manifest."""
    m = manifest()
    bad = []
    for kind, table in (("images", m["image_sha256"]), ("masks", m["mask_sha256"])):
        for stem, h in table.items():
            p = KV_DATA / kind / f"{stem}.jpg"
            if not p.exists():
                bad.append(f"missing {p}")
            elif sha256_file(p) != h:
                bad.append(f"hash differs {p}")
    if bad:
        raise SystemExit(f"Kvasir-SEG at {KV_DATA} does not match the manifest ({len(bad)} files), e.g. {bad[:3]}.\n"
                         "Set KV_DATA to the folder that holds images/ and masks/.")
    if verbose:
        print(f"data: {len(m['image_sha256'])} images and masks at {KV_DATA} match the manifest")


def letterbox_geometry(H0, W0, size=S):
    scale = size / max(H0, W0)
    h, w = max(1, round(H0 * scale)), max(1, round(W0 * scale))
    return (size - h) // 2, (size - w) // 2, h, w


def native_instances(mask):
    """Instance id map (0 background, 1..k) of a native binary mask, and the dropped count."""
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return lab.astype(np.uint8), 0
    area = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = area >= MIN_AREA_FRAC * mask.size
    remap = np.zeros(n + 1, np.int64)
    remap[1:][keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[lab].astype(np.uint8), int((~keep).sum())


def groups_of(m):
    """Duplicate-group id per image stem (the bootstrap and the folds resample groups)."""
    gid, nxt = {}, 0
    for members in m["multi_member_groups"]:
        for s in members:
            gid[s] = nxt
        nxt += 1
    out = {}
    for split in SPLITS.values():
        for s in m["splits"][split]:
            if s not in gid:
                gid[s] = nxt
                nxt += 1
            out[s] = gid[s]
    return out


def build(force=False):
    if CACHE_FILE.exists() and not force:
        return
    from PIL import Image
    m = manifest()
    gid = groups_of(m)
    out, stats = {}, {}
    for name, key in SPLITS.items():
        stems = m["splits"][key]
        X = np.zeros((len(stems), 3, S, S), np.uint8)
        INST = np.full((len(stems), S, S), IGNORE, np.uint8)
        geom = np.zeros((len(stems), 6), np.int32)
        dropped = vanished = 0
        for i, s in enumerate(stems):
            im = Image.open(KV_DATA / "images" / f"{s}.jpg").convert("RGB")
            W0, H0 = im.size
            top, left, h, w = letterbox_geometry(H0, W0)
            X[i, :, top:top + h, left:left + w] = np.asarray(im.resize((w, h), Image.LANCZOS)).transpose(2, 0, 1)
            mask = np.asarray(Image.open(KV_DATA / "masks" / f"{s}.jpg").convert("L")) >= 128
            lab, d = native_instances(mask)
            dropped += d
            small = np.asarray(Image.fromarray(lab).resize((w, h), Image.NEAREST))
            ids = [v for v in np.unique(small) if v != 0]
            vanished += int(lab.max()) - len(ids)
            ids = ids[:KMAX]
            remap = np.zeros(256, np.uint8)
            remap[ids] = np.arange(1, len(ids) + 1)
            INST[i, top:top + h, left:left + w] = remap[small]
            geom[i] = (H0, W0, top, left, h, w)
        out[name] = {"x": torch.from_numpy(X), "inst": torch.from_numpy(INST), "geom": torch.from_numpy(geom),
                     "stems": stems, "group": torch.tensor([gid[s] for s in stems])}
        n_inst = [int(v) for v in np.where(INST == IGNORE, 0, INST).reshape(len(stems), -1).max(1)]
        stats[name] = {"images": len(stems), "instances": int(sum(n_inst)), "dropped_tiny_components": dropped,
                       "vanished_at_working_resolution": vanished,
                       "instances_per_image": {str(k): n_inst.count(k) for k in sorted(set(n_inst))}}
    tmp = CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, tmp)
    os.replace(tmp, CACHE_FILE)
    # the instance statistics of the sealed split are not written anywhere (rule C3)
    write_json({k: v for k, v in stats.items() if k != "test"}, MANIFESTS / "instances.json")


# --------------------------------------------------------------------------- symmetries
def sym(t, k):
    """Symmetry ``k`` (0..7) of the square on the last two axes: ``k % 4`` quarter turns, then a
    horizontal flip for ``k >= 4``."""
    t = torch.rot90(t, k % 4, dims=(-2, -1))
    return torch.flip(t, dims=(-1,)) if k >= 4 else t


def sym_inv(t, k):
    if k >= 4:
        t = torch.flip(t, dims=(-1,))
    return torch.rot90(t, -(k % 4), dims=(-2, -1))


def sym_boxes(b, k, size=S):
    """Boxes ``(..., 4)`` as ``x1, y1, x2, y2`` (pixel edges) under symmetry ``k``."""
    x1, y1, x2, y2 = b.unbind(-1)
    for _ in range(k % 4):                           # rot90: new_x = y, new_y = size - x
        x1, y1, x2, y2 = y1, size - x2, y2, size - x1
    if k >= 4:
        x1, x2 = size - x2, size - x1
    return torch.stack([x1, y1, x2, y2], -1)


def boxes_of(inst):
    """Tight boxes of the instances of one id map ``(S, S)``: ``(k, 4)`` float, pixel edges."""
    n = int(inst[inst != IGNORE].max()) if (inst != IGNORE).any() else 0
    out = []
    for j in range(1, n + 1):
        ys, xs = torch.nonzero(inst == j, as_tuple=True)
        if ys.numel() == 0:
            out.append([0.0, 0.0, 0.0, 0.0])
        else:
            out.append([xs.min().item(), ys.min().item(), xs.max().item() + 1.0, ys.max().item() + 1.0])
    return torch.tensor(out, dtype=torch.float32, device=inst.device).reshape(-1, 4)


def valid_rect(inst):
    """The rectangle not covered by letterbox padding, ``x1, y1, x2, y2`` (pixel edges)."""
    ys, xs = torch.nonzero(inst != IGNORE, as_tuple=True)
    return torch.tensor([xs.min().item(), ys.min().item(), xs.max().item() + 1.0, ys.max().item() + 1.0],
                        device=inst.device)


# --------------------------------------------------------------------------- loaded data
class Data:
    """The cache on a device. ``x``/``inst``/``boxes`` index the pool (train then val); the
    sealed split is ``test_x`` / ``test_inst`` and is only read by the panel and the final job."""

    def __init__(self, device="cpu"):
        if not CACHE_FILE.exists():
            raise SystemExit("data cache missing: run ./setup.sh (or python code/data.py)")
        d = torch.load(CACHE_FILE, weights_only=False)
        self.device = torch.device(device)
        self.n_train, self.n_val = d["train"]["x"].shape[0], d["val"]["x"].shape[0]
        self.x = torch.cat([d["train"]["x"], d["val"]["x"]]).to(self.device)
        self.inst = torch.cat([d["train"]["inst"], d["val"]["inst"]]).to(self.device)
        self.group = torch.cat([d["train"]["group"], d["val"]["group"]])
        self.stems = d["train"]["stems"] + d["val"]["stems"]
        self.geom = torch.cat([d["train"]["geom"], d["val"]["geom"]])
        self.test_x, self.test_inst = d["test"]["x"].to(self.device), d["test"]["inst"].to(self.device)
        self.test_group, self.test_stems, self.test_geom = d["test"]["group"], d["test"]["stems"], d["test"]["geom"]
        self.boxes = [boxes_of(i) for i in self.inst]
        self.rect = torch.stack([valid_rect(i) for i in self.inst])
        self._test_boxes = None

    # the sealed split's labels are materialised only when asked for
    @property
    def test_boxes(self):
        if self._test_boxes is None:
            self._test_boxes = [boxes_of(i) for i in self.test_inst]
        return self._test_boxes

    def image_sets(self, fold=None):
        """``(train_idx, val_idx)`` into the pool for ``fold`` = ``None`` (canonical split) or a
        dict with ``cv`` (0..4: that fold is validation) and/or ``oof`` (0/1: that half of the
        training images is left out, to make out-of-fold proposals for it)."""
        fold = fold or {}
        n = self.n_train + self.n_val
        if fold.get("cv") is None:
            tr, va = torch.arange(self.n_train), torch.arange(self.n_train, n)
        else:
            f = cv_folds(self.group)
            va = torch.nonzero(f == int(fold["cv"]), as_tuple=True)[0]
            tr = torch.nonzero(f != int(fold["cv"]), as_tuple=True)[0]
        if fold.get("oof") is not None:
            half = oof_halves(self.group)
            tr = tr[half[tr] != int(fold["oof"])]
        return tr, va

    def panel(self, split):
        p = read_json(MANIFESTS / "panel.json")
        if split == "test":
            idx = [self.test_stems.index(s) for s in p["test"]]
            return self.test_x[idx], self.test_inst[idx], p["test"]
        idx = [self.stems.index(s) for s in p["val"]]
        return self.x[idx], self.inst[idx], p["val"]


def _group_assign(group, k, seed):
    g = torch.unique(group)
    perm = g[torch.randperm(len(g), generator=torch.Generator().manual_seed(seed))]
    fold_of = {int(v): i % k for i, v in enumerate(perm)}
    return torch.tensor([fold_of[int(v)] for v in group])


def cv_folds(group, k=5):
    """Fold id per pool image; duplicate groups stay together."""
    return _group_assign(group, k, FOLD_SEED)


def oof_halves(group):
    return _group_assign(group, 2, FOLD_SEED + 1)


def build_panel():
    """16 test and 16 validation images: the first of a seeded permutation of the sorted file
    names. No label is consulted (PLAN_KVASIR §4, rule C8)."""
    path = MANIFESTS / "panel.json"
    if path.exists():
        return
    m = manifest()
    out = {}
    for name, key in (("test", "final"), ("val", "val")):
        stems = sorted(m["splits"][key])
        perm = torch.randperm(len(stems), generator=torch.Generator().manual_seed(PANEL_SEED))
        out[name] = [stems[i] for i in perm[:PANEL_N].tolist()]
    out["rule"] = f"first {PANEL_N} of torch.randperm(seed={PANEL_SEED}) over the sorted stems of the split"
    write_json(out, path)


if __name__ == "__main__":
    verify()
    build(force="--force" in sys.argv)
    build_panel()
    st = read_json(MANIFESTS / "instances.json")
    for k, v in st.items():
        print(f"data: {k}: {v['images']} images, {v['instances']} instances, per image {v['instances_per_image']}")
    print(f"data: cache {CACHE_FILE}")
