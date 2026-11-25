"""Training views of the images for one epoch (PLAN_KVASIR §4).

    none  the images as cached
    sym   one random symmetry of the square per image (exact, on the device)   [default]
    alb   sym, then an AlbumentationsX pipeline on the uint8 image, its instance map and boxes;
          ``alb-photo`` / ``alb-geo`` run only the photometric / geometric part

A view holds, for the chosen images: ``x`` uint8 (N, 3, S, S), ``inst`` (N, S, S), the tight
``boxes`` of the instances (recomputed from the transformed map), the valid rectangle and, when
given, the proposals carried through the same transform. Validation, panels and the sealed
evaluation always use ``none``.

Library: ``pip install "albumentationsx[headless]"`` (import name ``albumentations``, AGPL-3.0,
Python >= 3.11) or ``albumentations==2.0.8`` on an older Python. Telemetry and the update check
are switched off before the import.
"""

import os

os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"
os.environ["ALBUMENTATIONS_OFFLINE"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import IGNORE, S  # noqa: E402
from data import boxes_of, sym, sym_boxes, valid_rect  # noqa: E402

MIN_PIXELS = 16                                      # 0.1 % of the 128 x 128 grid


class View:
    def __init__(self, x, inst, boxes, rect, props=None, index=None):
        self.x, self.inst, self.boxes, self.rect, self.props, self.index = x, inst, boxes, rect, props, index

    def __len__(self):
        return self.x.shape[0]


def telemetry_off():
    return all(os.environ.get(k) == "1" for k in ("ALBUMENTATIONS_NO_TELEMETRY", "ALBUMENTATIONS_OFFLINE"))


def _names(cls):
    import inspect
    return set(inspect.signature(cls.__init__).parameters)


def build_alb(part="all"):
    """The pipeline of PLAN_KVASIR §4. ``part``: all | photo | geo.

    AlbumentationsX renamed several arguments (``*_limit`` -> ``*_range``, ``BboxParams(format=)``
    -> ``coord_format=``, ``clip=`` -> ``clip_after_transform=``); the names are chosen from the
    installed signatures, so the same pipeline runs on ``albumentations`` 2.0.8."""
    import warnings

    import albumentations as A
    import cv2
    x = "brightness_range" in _names(A.RandomBrightnessContrast)      # AlbumentationsX argument names
    tf = []
    if part in ("all", "geo"):
        tf.append(A.Affine(scale=(0.8, 1.25), translate_percent=(-0.1, 0.1), rotate=(-20, 20), keep_ratio=True,
                           interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                           fill=0, fill_mask=IGNORE, p=1.0))
    if part in ("all", "photo"):
        if x:
            tf += [A.RandomBrightnessContrast(brightness_range=(-0.2, 0.2), contrast_range=(-0.2, 0.2), p=0.8),
                   A.HueSaturationValue(hue_shift_range=(-10, 10), sat_shift_range=(-20, 20), val_shift_range=(-10, 10), p=0.8),
                   A.GaussianBlur(blur_range=(3, 5), p=0.2)]
        else:
            tf += [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                   A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.8),
                   A.GaussianBlur(blur_limit=(3, 5), p=0.2)]
    kw = {"telemetry": False} if "telemetry" in _names(A.Compose) else {}
    bn = _names(A.BboxParams)
    bkw = {"coord_format" if "coord_format" in bn else "format": "pascal_voc", "label_fields": ["idx"],
           "min_visibility": 0.0, "min_area": 0.0, "clip_after_transform" if "clip_after_transform" in bn else "clip": True}
    warnings.filterwarnings("ignore", message="Got processor for bboxes")
    return A.Compose(tf, bbox_params=A.BboxParams(**bkw), **kw)


def _seed(tf, seed):
    if hasattr(tf, "set_random_seed"):
        tf.set_random_seed(int(seed))
    else:                                            # very old releases
        import random
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))


def alb_one(tf, x, inst, props, seed):
    """One image through the pipeline. Padding stays 0 in the image and 255 in the id map;
    instances left with fewer than ``MIN_PIXELS`` pixels are removed. Proposals keep their order
    (a box that leaves the image becomes a zero box)."""
    _seed(tf, seed)
    pad = inst == IGNORE
    img = np.ascontiguousarray(x.transpose(1, 2, 0))
    boxes = [] if props is None else np.clip(props, 0, S).tolist()
    keep = [i for i, b in enumerate(boxes) if b[2] - b[0] >= 1 and b[3] - b[1] >= 1]
    out = tf(image=img, mask=inst, bboxes=[boxes[i] for i in keep], idx=keep)
    xi, mi = out["image"].copy(), out["mask"].copy()
    xi[mi == IGNORE] = 0
    ids, cnt = np.unique(mi[(mi != 0) & (mi != IGNORE)], return_counts=True)
    remap = np.zeros(256, np.uint8)
    remap[IGNORE] = IGNORE
    nxt = 1
    for v, c in zip(ids, cnt):
        if c >= MIN_PIXELS:
            remap[v] = nxt
            nxt += 1
    mi = remap[mi]
    del pad
    p = None
    if props is not None:
        p = np.zeros((len(boxes), 4), np.float32)
        for b, i in zip(out["bboxes"], out["idx"]):
            p[int(i)] = b[:4]
    return xi.transpose(2, 0, 1), mi, p


def make_view(data, idx, aug="sym", seed=0, epoch=0, props=None):
    """The view of pool images ``idx`` for ``(seed, epoch)``. ``props``: ``(n_pool, P, 4)``
    proposals in the frame of the cached images, or ``None``."""
    dev = data.device
    idx = torch.as_tensor(idx)
    x, inst = data.x[idx.to(dev)], data.inst[idx.to(dev)]
    pr = None if props is None else props[idx].to(dev)
    if aug == "none":
        return View(x, inst, [data.boxes[i] for i in idx.tolist()], data.rect[idx.to(dev)], pr, idx)
    g = torch.Generator().manual_seed(1_000_003 * int(seed) + 7919 * int(epoch) + 17)
    ks = torch.randint(0, 8, (len(idx),), generator=g)
    xo, io = torch.empty_like(x), torch.empty_like(inst)
    po = None if pr is None else torch.empty_like(pr)
    for k in range(8):
        sel = torch.nonzero(ks == k, as_tuple=True)[0].to(dev)
        if sel.numel():
            xo[sel], io[sel] = sym(x[sel], k), sym(inst[sel], k)
            if po is not None:
                po[sel] = sym_boxes(pr[sel], k)
    if aug == "sym":
        boxes = [sym_boxes(data.boxes[i], int(k)) for i, k in zip(idx.tolist(), ks.tolist())]
        rect = torch.stack([sym_boxes(data.rect[i], int(k)) for i, k in zip(idx.tolist(), ks.tolist())])
        return View(xo, io, boxes, rect, po, idx)
    if not aug.startswith("alb"):
        raise ValueError(f"unknown augmentation level {aug!r}")
    tf = build_alb({"alb": "all", "alb-photo": "photo", "alb-geo": "geo"}[aug])
    xn, inn = xo.cpu().numpy(), io.cpu().numpy()
    pn = None if po is None else po.cpu().numpy()
    xs, ins, ps = [], [], []
    for j, i in enumerate(idx.tolist()):
        a, b, c = alb_one(tf, xn[j], inn[j], None if pn is None else pn[j],
                          seed=(int(seed) * 100_003 + int(epoch)) * 1_009 + i)
        xs.append(a)
        ins.append(b)
        ps.append(c)
    x2 = torch.from_numpy(np.stack(xs)).to(dev)
    i2 = torch.from_numpy(np.stack(ins)).to(dev)
    p2 = None if pn is None else torch.from_numpy(np.stack(ps)).to(dev)
    ok = (i2 != IGNORE).flatten(1).any(1)
    assert bool(ok.all()), "an augmented image has no valid pixel left"
    return View(x2, i2, [boxes_of(m) for m in i2], torch.stack([valid_rect(m) for m in i2]), p2, idx)
