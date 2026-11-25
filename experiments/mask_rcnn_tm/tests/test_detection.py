"""Gate K0a and the module tests of the detection machinery (CPU, synthetic fixtures only)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from detection import boxes as B  # noqa: E402
from detection import evaluate as E  # noqa: E402
from detection import rois as R  # noqa: E402


def disc(size, cx, cy, r):
    y, x = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def test_anchor_layout_and_shape_bits():
    a = B.Anchors(stride=8)
    assert len(a) == 16 * 16 * 24 and a.A == 24 and len(B.Anchors()) == 32 * 32 * 24
    w, h = a.boxes[:, 2] - a.boxes[:, 0], a.boxes[:, 3] - a.boxes[:, 1]
    assert torch.allclose((w * h).sqrt()[: a.A].reshape(8, 3)[:, 0], torch.tensor(B.SCALES, dtype=torch.float32), atol=1e-3)
    assert torch.allclose((h / w)[: a.A].reshape(8, 3)[0], torch.tensor(B.RATIOS), atol=1e-4)
    bits = a.shape_bits()
    assert bits.shape == (24, 10) and len({tuple(r.tolist()) for r in bits}) == 24
    assert torch.equal(a.cell[: a.A], torch.zeros(a.A, dtype=torch.long)) and a.shape[a.A + 3] == 3


def test_assignment_rules():
    a = B.Anchors(stride=8)
    valid = torch.ones(len(a), dtype=torch.bool)
    gt = torch.tensor([[30.0, 30.0, 70.0, 70.0], [100.0, 100.0, 107.0, 109.0]])
    lab, m = B.assign(a.boxes, valid, gt)
    iou = B.box_iou(a.boxes, gt)
    assert (lab[iou.max(1).values >= 0.7] == 1).all()
    assert lab[iou[:, 1].argmax()] == 1 and m[iou[:, 1].argmax()] == 1        # best anchor of a small box
    assert (lab[(iou.max(1).values < 0.3)] != 1).sum() >= (iou.max(1).values < 0.3).sum() - 2
    valid[:] = False
    assert (B.assign(a.boxes, valid, gt)[0] == -1).all()                       # padding is ignored
    lab0, _ = B.assign(a.boxes, torch.ones(len(a), dtype=torch.bool), gt[:0])
    assert (lab0 == 0).all()


def test_deltas_round_trip_and_bins():
    g = torch.Generator().manual_seed(0)
    ref = torch.tensor([[10.0, 20.0, 50.0, 80.0]]).repeat(200, 1)
    d = (torch.rand(200, 4, generator=g) - 0.5) * torch.tensor([1.0, 1.0, 1.4, 1.4])
    gt = B.decode(ref, d)
    assert torch.allclose(B.encode(ref, gt), d, atol=1e-5)
    for ordinal in (True, False):
        bins = B.DeltaBins(0.1, ordinal)
        assert bins.n_outputs == (44 if ordinal else 48)
        t, oor = bins.targets(d)
        assert oor == 0
        dec = bins.decode(t.float() * 2 - 1)
        assert (dec - d).abs().max() <= 0.05 + 1e-5                            # bin centre is within half a bin
    t, oor = B.DeltaBins(0.1).targets(torch.tensor([[2.0, -2.0, 0.0, 0.0]]))
    assert oor == 1 and t[0, :9].all() and not t[0, 9:18].any()                # end bins


def test_proposals_clip_filter_nms():
    boxes = torch.tensor([[0.0, 0.0, 40.0, 40.0], [1.0, 1.0, 41.0, 41.0], [60.0, 60.0, 61.0, 90.0],
                          [50.0, 50.0, 200.0, 200.0], [5.0, 5.0, 30.0, 30.0]])
    scores = torch.tensor([0.9, 0.8, 0.95, 0.7, 0.99])
    valid = torch.tensor([True, True, True, True, False])
    rect = torch.tensor([0.0, 0.0, 128.0, 128.0])
    b, s, idx = B.proposals(scores, boxes, valid, rect, post=10)
    assert idx.tolist() == [0, 3]                                              # 2: thin, 1: suppressed, 4: invalid
    assert b[1].tolist() == [50.0, 50.0, 128.0, 128.0]


def test_jitter_hits_the_iou_range():
    g = torch.Generator().manual_seed(1)
    boxes = torch.tensor([[40.0, 40.0, 90.0, 80.0]]).repeat(500, 1)
    j, iou = R.jitter(boxes, torch.tensor([0.0, 0.0, 128.0, 128.0]), g)
    real = torch.diag(B.box_iou(j, boxes))
    assert torch.allclose(real, iou, atol=1e-4)
    assert (real > 0.45).float().mean() > 0.97 and (real < 0.93).float().mean() > 0.97
    assert real.std() > 0.08                                                   # spread over the range
    bg = R.background_boxes(64, boxes[:1], torch.tensor([0.0, 0.0, 128.0, 128.0]), g)
    assert len(bg) > 32 and B.box_iou(bg, boxes[:1]).max() < 0.5


def test_crop_matches_the_image_and_is_zero_outside():
    x = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128) % 251
    c = R.crop(x, torch.zeros(1, dtype=torch.long), torch.tensor([[16.0, 32.0, 48.0, 64.0]]), 32)
    assert torch.equal(c[0, 0], x[0, 0, 32:64, 16:48])                         # same scale: exact
    c = R.crop(torch.full((1, 1, 128, 128), 200.0), torch.zeros(1, dtype=torch.long),
               torch.tensor([[-20.0, 10.0, 20.0, 50.0]]), 32)
    assert (c[0, 0, :, :14] == 0).all() and (c[0, 0, :, 18:] == 200).all()     # left half is outside


def test_paste_inverts_the_mask_crop():
    m = disc(128, 60.0, 64.0, 16.0)
    box = R.mask_boxes(m[None])
    assert box.tolist() == [[44.0, 48.0, 76.0, 80.0]]
    for size, floor in ((28, 0.9), (32, 0.9), (16, 0.85)):
        t = R.crop(m[None, None].float(), torch.zeros(1, dtype=torch.long), box, size, mode="mask")[:, 0]
        back = R.paste(t.float() * 2 - 1, box, 128)
        assert R.mask_iou(back, m[None]).item() >= floor
    rect = torch.zeros(128, 128, dtype=torch.bool)
    rect[20:52, 30:62] = True
    b = R.mask_boxes(rect[None])
    t = R.crop(rect[None, None].float(), torch.zeros(1, dtype=torch.long), b, 32, mode="mask")[:, 0]
    assert torch.equal(R.paste(t.float() * 2 - 1, b, 128)[0], rect)            # box-aligned rectangle: exact
    out = R.paste(torch.ones(1, 32, 32), torch.tensor([[100.0, 100.0, 160.0, 160.0]]), 128)
    assert out[0, 100:, 100:].all() and out.sum() == 28 * 28                   # clipped, never wraps


def fixture(seed, n_img=30):
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n_img):
        k, n = rng.integers(0, 4), rng.integers(0, 8)
        iou = rng.random((n, k)) * (rng.random((n, k)) > 0.4)
        images.append((np.round(rng.random(n), 1), iou))                       # tied scores on purpose
    return images


def test_ground_truth_as_predictions_is_perfect_and_order_free():
    images = [(np.array([0.9, 0.8]), np.eye(2)), (np.array([0.7]), np.ones((1, 1)))]
    r = E.average_precision(images)
    assert r["AP"] == pytest.approx(1.0) and r["AP50"] == pytest.approx(1.0)
    f = fixture(0)
    base = E.average_precision(f)
    rng = np.random.default_rng(1)
    shuffled = []
    for s, iou in f:                                                           # distinct scores first
        s = s + np.arange(len(s)) * 1e-6
        p = rng.permutation(len(s))
        shuffled.append(((s, iou), (s[p], iou[p])))
    a = E.average_precision([u for u, _ in shuffled])
    b = E.average_precision([v for _, v in shuffled])
    assert a["AP"] == pytest.approx(b["AP"], abs=1e-12) and base["AP"] >= 0


def test_parity_with_pycocotools():
    pytest.importorskip("pycocotools")
    import contextlib
    import io

    from pycocotools import mask as mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rng = np.random.default_rng(3)
    S, gts, dts, mine_m, mine_b = 64, [], [], [], []
    ann = 1
    imgs = []
    for img in range(25):
        imgs.append({"id": img, "height": S, "width": S})
        gm, dm, ds = [], [], []
        for _ in range(rng.integers(0, 4)):
            m = disc(S, rng.uniform(10, 54), rng.uniform(10, 54), rng.uniform(4, 12)).numpy()
            gm.append(m)
        for _ in range(rng.integers(0, 6)):
            if gm and rng.random() < 0.7:
                base = gm[rng.integers(len(gm))]
                m = np.roll(base, (rng.integers(-3, 4), rng.integers(-3, 4)), (0, 1))
            else:
                m = disc(S, rng.uniform(10, 54), rng.uniform(10, 54), rng.uniform(4, 12)).numpy()
            dm.append(m)
            ds.append(float(rng.random()))
        for m in gm:
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            gts.append({"id": ann, "image_id": img, "category_id": 1, "segmentation": rle, "area": float(m.sum()),
                        "bbox": mu.toBbox(rle).tolist(), "iscrowd": 0})
            ann += 1
        for m, s in zip(dm, ds):
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            dts.append({"image_id": img, "category_id": 1, "segmentation": rle, "score": s, "bbox": mu.toBbox(rle).tolist()})
        g = torch.from_numpy(np.array(gm).reshape(-1, S, S))
        d = torch.from_numpy(np.array(dm).reshape(-1, S, S))
        mine_m.append((np.array(ds), R.mask_iou(d, g).numpy()))
        mine_b.append((np.array(ds), B.box_iou(R.mask_boxes(d).double(), R.mask_boxes(g).double()).numpy()))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {"images": imgs, "annotations": gts, "categories": [{"id": 1}]}
        coco.createIndex()
        res = coco.loadRes(dts)
        for kind, mine in (("segm", mine_m), ("bbox", mine_b)):
            ev = COCOeval(coco, res, kind)
            ev.params.areaRng, ev.params.areaRngLbl = [[0, 1e10]], ["all"]
            ev.evaluate()
            ev.accumulate()
            p = ev.eval["precision"][:, :, 0, 0, -1]                           # (T, R), maxDets 100
            ref = [float(p[t].mean()) for t in range(10)]
            got = E.average_precision(mine)
            assert np.allclose(got["per_thr"], ref, atol=1e-6), kind
            assert got["AP"] == pytest.approx(float(np.mean(ref)), abs=1e-6)


def test_recall_scores_and_semantic():
    images = [(np.array([0.9, 0.1, 0.5]), np.array([[0.2, 0.0], [0.8, 0.0], [0.0, 0.6]]))]
    r = E.recall_at(images, ks=(1, 2, 3))
    assert r == {1: 0.0, 2: 0.5, 3: 1.0}
    s, tie = E.distinct_scores([(1.0, 0.5, 0, 0), (1.0, 0.7, 0, 1), (2.0, 0.0, 1, 0)])
    assert s.tolist() == [1 / 3, 2 / 3, 1.0] and tie == pytest.approx(2 / 3)
    t = np.zeros((8, 8), np.uint8)
    t[2:6, 2:6] = 1
    t[0] = 255
    sem = E.semantic([E.confusion(t == 1, t)])
    assert sem["mIoU"] == 1.0 and sem["Dice"] == 1.0
    lo, hi = E.bootstrap_ap(fixture(5), np.arange(30), n_boot=50)
    assert lo <= E.average_precision(fixture(5))["AP50"] <= hi
