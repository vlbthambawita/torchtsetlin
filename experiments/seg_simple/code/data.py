"""Kvasir-SEG for the programme: manifest check, resized cache, encodings, symmetries, floors.

    python code/data.py            verify the files against the manifest and build the caches

The split is ``manifests/kvasir_split.json`` (700 / 150 / 150), a copy of the split of
``experiments/mask_rcnn_tm``. Images are resized to S x S (area filter); a mask pixel is polyp when
at least half of its area is polyp.

Encodings give ``(B, 3 colours, D, H, W)`` Boolean; ``planes`` flattens them colour-major, the
layout of ``tt.data.ColorThermometerEncoder``:
    T4, T8   thermometer, plane d of a colour is ``value / 255 >= (d + 1) / (D + 1)``
    B8       bit planes of the raw 8-bit value, most significant bit first
"""

import os
from pathlib import Path

import numpy as np
import torch
from common import CACHE, MANIFESTS, S, read_json, sha256_file, write_json

MANIFEST_SHA256 = "915e22eb119e5888"                 # first 16 hex digits of manifests/kvasir_split.json
DATA_DIRS = ("/work/vajira/DATA/Kvasir-SEG", "/work/vajira/DATA/Kvasir_seg_original/Kvasir-SEG")
SPLITS = ("train", "val", "test")
DEPTH = {"T4": 4, "T8": 8, "B8": 8}
N_PANEL = 8


def data_dir():
    """The folder that holds images/ and masks/. SS_DATA wins; a nested Kvasir-SEG/ is accepted."""
    cands = [os.environ["SS_DATA"]] if os.environ.get("SS_DATA") else list(DATA_DIRS)
    for c in cands:
        for p in (Path(c), Path(c) / "Kvasir-SEG"):
            if (p / "images").is_dir() and (p / "masks").is_dir():
                return p
    raise SystemExit(f"Kvasir-SEG not found in {cands}: set SS_DATA to the folder that holds images/ and masks/")


def manifest():
    p = MANIFESTS / "kvasir_split.json"
    if not sha256_file(p).startswith(MANIFEST_SHA256):
        raise SystemExit(f"{p} is not the split of the plan (checksum differs)")
    return read_json(p)


def verify(verbose=True):
    m, root, bad = manifest(), data_dir(), []
    for kind, key in (("images", "image_sha256"), ("masks", "mask_sha256")):
        for stem, h in m[key].items():
            p = root / kind / f"{stem}.jpg"
            if not p.exists() or sha256_file(p) != h:
                bad.append(f"{kind}/{stem}.jpg")
    if bad:
        raise SystemExit(f"Kvasir-SEG at {root} does not match the manifest ({len(bad)} files), e.g. {bad[:3]}")
    if verbose:
        print(f"data: {len(m['image_sha256'])} images and masks at {root} match the manifest")


def cache_path(size=S):
    return CACHE / "data" / f"kvasir_{size}.pt"


def build(size=S, force=False):
    """Write the resized cache (runners never build it: setup.sh does)."""
    out = cache_path(size)
    if out.exists() and not force:
        return out
    from PIL import Image
    m, root = manifest(), data_dir()
    stems, split = [], {}
    for sp in SPLITS:
        names = m["splits"]["final" if sp == "test" else sp]          # the manifest calls the test split "final"
        split[sp] = list(range(len(stems), len(stems) + len(names)))
        stems += names
    x = np.zeros((len(stems), 3, size, size), np.uint8)
    y = np.zeros((len(stems), size, size), np.uint8)
    for i, s in enumerate(stems):
        im = Image.open(root / "images" / f"{s}.jpg").convert("RGB").resize((size, size), Image.BOX)
        x[i] = np.asarray(im).transpose(2, 0, 1)
        mk = Image.fromarray(((np.asarray(Image.open(root / "masks" / f"{s}.jpg").convert("L")) >= 128) * 255)
                             .astype(np.uint8)).resize((size, size), Image.BOX)
        y[i] = np.asarray(mk) >= 128
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    torch.save({"stems": stems, "split": split, "x": torch.from_numpy(x), "y": torch.from_numpy(y), "size": size}, tmp)
    os.replace(tmp, out)
    print(f"data: wrote {out} ({len(stems)} images at {size} px, polyp share {y.mean():.3f})")
    return out


class Data:
    """The three splits as uint8 tensors on ``device``: ``x[split]`` (N, 3, S, S), ``y[split]`` (N, S, S)."""

    def __init__(self, device="cpu", size=S, limit=None):
        p = cache_path(size)
        if not p.exists():
            raise SystemExit(f"{p} is missing: run ./setup.sh (or python code/data.py)")
        d = torch.load(p, weights_only=False)
        self.size, self.stems, self.x, self.y = size, {}, {}, {}
        for sp in SPLITS:
            idx = d["split"][sp][:limit] if limit else d["split"][sp]
            self.stems[sp] = [d["stems"][i] for i in idx]
            self.x[sp] = d["x"][idx].to(device)
            self.y[sp] = d["y"][idx].long().to(device)

    def panel(self):
        """Indices of the fixed validation images shown in the panels."""
        n = len(self.stems["val"])
        return [int(i) for i in np.linspace(0, n - 1, min(N_PANEL, n)).round()]


# --------------------------------------------------------------------------- encodings
_THR = {}


def encode(x, enc):
    """uint8 images ``(B, 3, H, W)`` -> Boolean ``(B, 3, D, H, W)``."""
    if enc == "B8":
        sh = torch.arange(7, -1, -1, device=x.device).view(1, 1, 8, 1, 1)
        return ((x.unsqueeze(2).to(torch.int16) >> sh) & 1).bool()
    key = (enc, x.device)
    if key not in _THR:
        import torchtsetlin as tt
        _THR[key] = tt.data.ColorThermometerEncoder(n_bits=DEPTH[enc], value_range=(0.0, 1.0)).thresholds.to(x.device)
    return (x.float() / 255.0).unsqueeze(2) >= _THR[key].view(1, 1, -1, 1, 1)


def planes(x, enc):
    """uint8 images -> Boolean planes ``(B, 3 * D, H, W)``, colour-major."""
    return encode(x, enc).flatten(1, 2)


# --------------------------------------------------------------------------- augmentation (arm sym)
def sym(t, k):
    """Symmetry ``k`` in 0..7 of the square on the last two axes: ``k % 4`` quarter turns, then a flip for k >= 4."""
    t = torch.rot90(t, k % 4, (-2, -1))
    return t.flip(-1) if k >= 4 else t


def sym_inv(t, k):
    t = t.flip(-1) if k >= 4 else t
    return torch.rot90(t, -(k % 4), (-2, -1))


def augment(x, y, gen):
    """One random symmetry per image, the same for the image and its mask. Returns ``(x, y, ks)``."""
    ks = torch.randint(0, 8, (x.shape[0],), generator=gen).tolist()
    return (torch.stack([sym(x[i], k) for i, k in enumerate(ks)]),
            torch.stack([sym(y[i], k) for i, k in enumerate(ks)]), ks)


# --------------------------------------------------------------------------- floors (plan section 6)
def floors(data):
    """Predictions that use no image content. Written to ``results/floors.json`` by setup."""
    from models import Confusion
    freq = data.y["train"].float().mean(0)                                   # polyp frequency per position
    out = {}
    preds = {"all_background": lambda n: torch.zeros(n, *freq.shape, dtype=torch.long, device=freq.device),
             "position_prior": lambda n: (freq > 0.5).long().expand(n, -1, -1)}
    best_t, best = 0.5, -1.0
    for t in [i / 40 for i in range(1, 40)]:
        c = Confusion().add((freq > t).long().expand(len(data.y["val"]), -1, -1), data.y["val"])
        if c.metrics()["mIoU"] > best:
            best, best_t = c.metrics()["mIoU"], t
    preds["position_prior_best_threshold"] = lambda n: (freq > best_t).long().expand(n, -1, -1)
    for name, f in preds.items():
        out[name] = {sp: Confusion().add(f(len(data.y[sp])), data.y[sp]).metrics() for sp in ("val", "test")}
    out["position_prior_best_threshold"]["threshold"] = best_t
    out["polyp_share"] = {sp: float(data.y[sp].float().mean()) for sp in SPLITS}
    return out


if __name__ == "__main__":
    verify()
    for size in (S, 128):
        build(size)
    d = Data()
    print("data: splits", {sp: len(d.stems[sp]) for sp in SPLITS})
    from common import RESULTS
    fl = floors(d)
    write_json(fl, RESULTS / "floors.json")
    for k, v in fl.items():
        if k != "polyp_share":
            print(f"floor {k:32s} val mIoU {v['val']['mIoU']:.3f}  polyp IoU {v['val']['IoU_polyp']:.3f}")
