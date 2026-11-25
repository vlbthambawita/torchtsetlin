"""COCO-style average precision from per-image IoU matrices, recall@k, semantic read-outs.

``average_precision`` follows ``pycocotools`` (greedy matching in score order to the unmatched
ground truth of highest IoU, 101 recall points, precision envelope, at most 100 detections per
image, one class, no crowd regions); ``tests/test_detection.py`` pins the parity. Working from
IoU matrices keeps the sufficient statistics small, so the bootstrap can recompute AP for
resampled images (``bootstrap_ap``).
"""

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


def _as2d(scores, iou):
    """``iou`` as ``(n, k)`` float64, also when ``n`` or ``k`` is zero."""
    iou = np.asarray(iou, np.float64)
    k = iou.shape[1] if iou.ndim == 2 else 0
    return iou.reshape(len(scores), k)


def _match(scores, iou, thr):
    """Greedy COCO matching of one image at one threshold: Boolean ``tp`` per detection, in the
    order of ``scores`` sorted descending (stable)."""
    order = np.argsort(-scores, kind="mergesort")[:MAX_DETS]
    k = iou.shape[1]
    taken = np.zeros(k, bool)
    tp = np.zeros(len(order), bool)
    for r, d in enumerate(order):
        best, m = min(thr, 1 - 1e-10), -1
        for g in range(k):
            if taken[g] or iou[d, g] < best:
                continue
            best, m = iou[d, g], g
        if m >= 0:
            taken[m] = True
            tp[r] = True
    return scores[order], tp


def average_precision(images, thrs=IOU_THRS):
    """``images``: list of ``(scores (n,), iou (n, k))`` per image (``k`` ground-truth objects).
    Returns ``{"AP", "AP50", "AP75", "per_thr"}``; -1 when there is no ground truth."""
    npig = sum(_as2d(*im).shape[1] for im in images)
    if npig == 0:
        return {"AP": -1.0, "AP50": -1.0, "AP75": -1.0, "per_thr": []}
    per = []
    for t in thrs:
        sc, tp = [], []
        for scores, iou in images:
            if len(scores) == 0:
                continue
            s, m = _match(np.asarray(scores, np.float64), _as2d(scores, iou), t)
            sc.append(s)
            tp.append(m)
        if not sc:
            per.append(0.0)
            continue
        sc, tp = np.concatenate(sc), np.concatenate(tp)
        order = np.argsort(-sc, kind="mergesort")
        tp = tp[order]
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        rc = ctp / npig
        pr = ctp / (ctp + cfp + np.spacing(1))
        for i in range(len(pr) - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        ok = inds < len(pr)
        q[ok] = pr[inds[ok]]
        per.append(float(q.mean()))
    out = {"AP": float(np.mean(per)), "per_thr": per}
    for name, t in (("AP50", 0.5), ("AP75", 0.75)):
        j = np.where(np.isclose(thrs, t))[0]
        out[name] = per[int(j[0])] if len(j) else -1.0
    return out


def recall_at(images, ks=(1, 5, 10, 30), thr=0.5):
    """Fraction of ground-truth objects covered (IoU >= ``thr``) by the ``k`` best detections of
    their image. ``images`` as in ``average_precision``."""
    out, n = {k: 0 for k in ks}, 0
    for scores, iou in images:
        iou = _as2d(scores, iou)
        n += iou.shape[1]
        order = np.argsort(-np.asarray(scores), kind="mergesort")
        for k in ks:
            if len(order) and iou.shape[1]:
                out[k] += int((iou[order[:k]].max(0) >= thr).sum())
    return {k: (v / n if n else 0.0) for k, v in out.items()}


def confusion(pred, target, ignore=255):
    """2 x 2 confusion matrix (rows = truth) of Boolean ``pred`` against a label map."""
    p, t = np.asarray(pred).reshape(-1).astype(np.int64), np.asarray(target).reshape(-1).astype(np.int64)
    ok = t != ignore
    return np.bincount(np.minimum(t[ok], 1) * 2 + p[ok], minlength=4).reshape(2, 2)


def semantic(cms):
    """mIoU (mean over background and polyp, pooled over images, as the segmentation programme)
    and Dice of the polyp class (pooled, and the mean over images) from per-image confusions."""
    cms = np.asarray(cms, np.float64).reshape(-1, 2, 2)
    cm = cms.sum(0)
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
    den = cms[:, 1, 1] * 2 + cms[:, 0, 1] + cms[:, 1, 0]
    per_image = np.where(den > 0, 2 * cms[:, 1, 1] / np.maximum(den, 1), 1.0)
    return {"mIoU": float(np.nanmean(iou)), "IoU_polyp": float(iou[1]),
            "Dice": float(2 * cm[1, 1] / max(2 * cm[1, 1] + cm[0, 1] + cm[1, 0], 1)),
            "Dice_per_image": float(per_image.mean())}


def distinct_scores(keys):
    """Decision D-3: detections ordered lexicographically by ``keys`` (a list of tuples, most
    important key first, ending in image id and RoI index) get distinct floats in (0, 1], so the
    evaluator never has to break a tie. Returns ``(scores, tie_fraction)`` where the tie fraction
    is the share of detections whose first key is shared with another detection."""
    n = len(keys)
    if n == 0:
        return np.zeros(0), 0.0
    order = sorted(range(n), key=lambda i: keys[i])
    scores = np.zeros(n)
    scores[order] = (np.arange(n) + 1) / n
    first = np.asarray([k[0] for k in keys], np.float64)
    _, inv, cnt = np.unique(first, return_inverse=True, return_counts=True)
    return scores, float((cnt[inv] > 1).mean())


def bootstrap_ap(images, groups, n_boot=1000, seed=0, key="AP50"):
    """Percentile interval of AP under resampling of duplicate groups."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        vals.append(average_precision([images[i] for g in pick for i in by[g]])[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
