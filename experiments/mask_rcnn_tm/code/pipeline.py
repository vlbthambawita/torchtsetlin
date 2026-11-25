"""End-to-end inference with trained heads, and its evaluation (PLAN_KVASIR §5, §10).

    proposals   objectness of every anchor -> top 1000 -> deltas -> clip -> NMS 0.7 -> keep k
    detections  RoI classifier margin per proposal -> box refinement -> NMS 0.5 -> top 100
    masks       mask head on the final boxes -> paste-back at threshold 0

Scores are unclipped margins (decision D-3): detections are ordered by (classifier margin,
objectness margin, mean mask margin, image, region) and mapped to distinct floats. There is no
score threshold. ``oracle`` replaces one component by the ground truth (gate E4b of PLAN.md).
"""

import numpy as np
import torch
from common import IGNORE, MODELS, S
from detection import boxes as B
from detection import evaluate as EV
from detection import rois as R
from heads import HEADS, chunks

PRE_NMS, NMS_PROP, NMS_DET, MAX_DETS = 1000, 0.7, 0.5, 100
MIN_PIXELS = 16


def best_path(tag):
    return MODELS / tag / "best.pt"


def load_head(ref, data, device):
    """``(head, team)`` of a finished run (its tag), ready to predict."""
    ck = torch.load(best_path(ref), map_location=device, weights_only=False)
    head = HEADS[ck["job"]["head"]](ck["job"], data, device, inference=True)
    team = head.team()
    team.load_state(ck["states"])
    team.eval()
    return head, team


class Pipeline:
    """``refs``: ``{"rpn_obj": tag, "rpn_delta": ..., "cls": ..., "box": ..., "mask": ...,
    "dense": ...}``; any of them may be missing or ``None``. Options:

    ``rank``    ``cls`` (default when a classifier is given) or ``obj``
    ``refine``  ``deltas`` (the box head), ``mask`` (tight box of the predicted mask) or ``none``
    ``mask_from``  ``mask`` (the RoI head) or ``dense`` (the dense map cut to the box)
    """

    def __init__(self, data, device, refs, rank=None, refine="deltas", mask_from="mask", post=30, live=None):
        self.d, self.dev, self.post = data, torch.device(device), post
        self.h = {k: load_head(v, data, device) for k, v in refs.items() if v}
        for k, v in (live or {}).items():            # a head that is still training (panels)
            self.h[k] = v
        self.rank = rank or ("cls" if "cls" in self.h else "obj")
        self.refine = refine if (refine != "deltas" or "box" in self.h) else "none"
        self.mask_from = mask_from if mask_from in self.h else ("mask" if "mask" in self.h else None)

    # ------------------------------------------------------------------ stage 1
    @torch.no_grad()
    def propose(self, x, inst, rect, post=None):
        """Per image ``(boxes, objectness margin)``, best first."""
        post = post or self.post
        oh, ot = self.h["rpn_obj"]
        sc = oh.score(ot, x)
        feats = None
        if "rpn_delta" in self.h:
            dh, dt = self.h["rpn_delta"]
            feats = [dh.feats(s, x) for s in dh.specs]
        out = []
        for j in range(x.shape[0]):
            valid = oh.anchors.valid(inst[j])
            s = torch.where(valid, sc[j], torch.full_like(sc[j], float("-inf")))
            top = s.topk(min(PRE_NMS, int(valid.sum()))).indices
            boxes = oh.anchors.boxes[top]
            if feats is not None:
                boxes = B.decode(boxes, dh.deltas(dt, [dh.rows(f, j, top) for f in feats]))
            boxes = B.clip(boxes, rect[j])
            ok = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
            boxes, st = boxes[ok], s[top][ok]
            keep = B.nms(boxes, st, NMS_PROP)[:post]
            out.append((boxes[keep], st[keep]))
        return out

    # ------------------------------------------------------------------ stage 2
    @torch.no_grad()
    def detect(self, x, inst, rect, gt_boxes=None, oracle=(), proposals=None):
        """Detections of a batch: per image a dict with ``boxes``, ``masks`` (Boolean, full
        image), and the score keys ``cls``, ``obj``, ``mmask``."""
        Bn = x.shape[0]
        if "proposals" in oracle:
            props = [(g, torch.zeros(len(g), device=self.dev)) for g in gt_boxes]
        else:
            props = proposals if proposals is not None else self.propose(x, inst, rect)
        bidx = torch.cat([torch.full((len(p[0]),), j, dtype=torch.long, device=self.dev) for j, p in enumerate(props)])
        boxes, obj = torch.cat([p[0] for p in props]), torch.cat([p[1] for p in props])
        iou_gt = m_gt = None
        if gt_boxes is not None and oracle:
            iou_gt, m_gt = torch.zeros(len(boxes), device=self.dev), torch.zeros(len(boxes), dtype=torch.long, device=self.dev)
            for j in range(Bn):
                sel = bidx == j
                if gt_boxes[j].shape[0] and sel.any():
                    iou_gt[sel], m_gt[sel] = B.box_iou(boxes[sel], gt_boxes[j]).max(1)
        if "class" in oracle:
            mcls = torch.where(iou_gt >= 0.5, 1e6, -1e6).float()
        elif self.rank == "cls":
            ch, ct = self.h["cls"]
            mcls = ch.margins(ct, x, bidx, boxes)
        else:
            mcls = obj.clone()
        if "box" in oracle:
            for j in range(Bn):
                sel = torch.nonzero((bidx == j) & (iou_gt >= 0.5), as_tuple=True)[0]
                boxes[sel] = gt_boxes[j][m_gt[sel]]
        elif self.refine == "deltas":
            bh, bt = self.h["box"]
            boxes = B.clip(bh.refine(bt, x, inst, bidx, boxes), rect[bidx])
            ok = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
            bidx, boxes, obj, mcls = bidx[ok], boxes[ok], obj[ok], mcls[ok]
            if iou_gt is not None:
                iou_gt, m_gt = iou_gt[ok], m_gt[ok]
        # NMS per image in the order (classifier margin, objectness)
        order = torch.argsort(obj, descending=True, stable=True)
        order = order[torch.argsort(mcls[order], descending=True, stable=True)]
        rank = torch.empty(len(order), device=self.dev)
        rank[order] = torch.arange(len(order), 0, -1, device=self.dev, dtype=torch.float32)
        keep = []
        for j in range(Bn):
            sel = torch.nonzero(bidx == j, as_tuple=True)[0]
            keep.append(sel[B.nms(boxes[sel], rank[sel], NMS_DET)[:MAX_DETS]])
        keep = torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long, device=self.dev)
        bidx, boxes, obj, mcls = bidx[keep], boxes[keep], obj[keep], mcls[keep]
        # masks
        if "mask" in oracle:
            m_k, i_k = m_gt[keep], iou_gt[keep]
            gm = (inst[bidx] == (m_k + 1).view(-1, 1, 1).to(inst.dtype)) & (i_k >= 0.5).view(-1, 1, 1)
            masks = gm & R.paste(torch.ones(len(boxes), 2, 2, device=self.dev), boxes, S)
            mmask = torch.zeros(len(boxes), device=self.dev)
        elif self.mask_from == "dense":
            dh, dt = self.h["dense"]
            dm = dh.predict(dt, x)
            inside = R.paste(torch.ones(len(boxes), 2, 2, device=self.dev), boxes, S)
            masks = (dm[bidx] > 0) & inside
            mmask = (dm[bidx] * masks).flatten(1).sum(1) / masks.flatten(1).sum(1).clamp(min=1)
        elif self.mask_from == "mask":
            mh, mt = self.h["mask"]
            mg = mh.predict(mt, x, bidx, boxes)
            masks = R.paste(mg, boxes, S)
            pos = mg > 0
            mmask = (mg * pos).flatten(1).sum(1) / pos.flatten(1).sum(1).clamp(min=1)
        else:                                        # no mask head: the filled box
            masks = R.paste(torch.ones(len(boxes), 2, 2, device=self.dev), boxes, S)
            mmask = torch.zeros(len(boxes), device=self.dev)
        masks &= (inst[bidx] != IGNORE)
        if self.refine == "mask" and "box" not in oracle:
            tight = R.mask_boxes(masks)
            has = masks.flatten(1).any(1)
            boxes = torch.where(has.view(-1, 1), tight, boxes)
        out = []
        for j in range(Bn):
            sel = bidx == j
            out.append({"boxes": boxes[sel], "masks": masks[sel], "cls": mcls[sel], "obj": obj[sel], "mmask": mmask[sel]})
        return out


@torch.no_grad()
def dense_cc(head, team, x, inst):
    """Baseline (v): the dense semantic map, 8-connected components as instances, scored by
    their mean margin. Returns detections in the format of ``Pipeline.detect`` plus the map."""
    from scipy import ndimage
    dm = head.predict(team, x)
    sem = (dm > 0) & (inst != IGNORE)
    out = []
    for j in range(x.shape[0]):
        lab, n = ndimage.label(sem[j].cpu().numpy(), structure=np.ones((3, 3)))
        lab = torch.from_numpy(lab).to(x.device)
        masks, score = [], []
        for k in range(1, n + 1):
            m = lab == k
            if int(m.sum()) >= MIN_PIXELS:
                masks.append(m)
                score.append(dm[j][m].mean())
        masks = torch.stack(masks) if masks else torch.zeros(0, S, S, dtype=torch.bool, device=x.device)
        sc = torch.stack(score) if score else torch.zeros(0, device=x.device)
        out.append({"boxes": R.mask_boxes(masks), "masks": masks, "cls": sc, "obj": torch.zeros_like(sc),
                    "mmask": sc.clone(), "semantic": sem[j]})
    return out


def run(detector, x, inst, rect, gt_boxes, batch=8):
    """Apply ``detector(x, inst, rect, gt_boxes) -> list of detections`` in batches."""
    out = []
    for c in chunks(x.shape[0], batch):
        out += detector(x[c], inst[c], rect[c], [gt_boxes[i] for i in c])
    return out


def evaluate(dets, inst, gt_boxes, threshold=None):
    """Mask and box AP, recall, tie fraction and the semantic read-out of a list of detections.

    Returns ``(metrics, stats)``; ``stats`` holds the per-image sufficient statistics (scores,
    IoU matrices, confusion matrices) the bootstrap needs. The semantic prediction is the
    union of the detections whose classifier margin is at least ``threshold``; with
    ``threshold=None`` the threshold that maximises Dice is searched (do that on validation
    data only and pass the value when scoring anything else)."""
    keys, where = [], []
    for i, d in enumerate(dets):
        for r in range(len(d["cls"])):
            keys.append((float(d["cls"][r]), float(d["obj"][r]), float(d["mmask"][r]), -i, -r))
            where.append(i)
    scores, tie = EV.distinct_scores(keys)
    where = np.asarray(where, np.int64)
    im_mask, im_box, gts = [], [], []
    for i, d in enumerate(dets):
        k = int(gt_boxes[i].shape[0])
        gm = torch.stack([inst[i] == j + 1 for j in range(k)]) if k else torch.zeros(0, S, S, dtype=torch.bool, device=inst.device)
        gts.append(gm)
        s = scores[where == i]
        im_mask.append((s, R.mask_iou(d["masks"], gm).cpu().numpy().reshape(len(s), k)))
        im_box.append((s, B.box_iou(d["boxes"].double(), gt_boxes[i].double()).cpu().numpy().reshape(len(s), k)))
    m, b = EV.average_precision(im_mask), EV.average_precision(im_box)
    rec = EV.recall_at(im_box)
    target = [np.where(t == IGNORE, IGNORE, t > 0).astype(np.uint8) for t in inst.cpu().numpy()]

    def cms_at(thr):
        out = []
        for i, d in enumerate(dets):
            if "semantic" in d:
                pred = d["semantic"]
            else:
                sel = d["cls"] >= thr
                pred = d["masks"][sel].any(0) if bool(sel.any()) else torch.zeros(S, S, dtype=torch.bool)
            out.append(EV.confusion(pred.cpu().numpy(), target[i]))
        return out

    if threshold is None:
        allc = np.asarray([k[0] for k in keys])
        cands = np.unique(np.quantile(allc, np.linspace(0, 1, 41))) if len(allc) else np.asarray([0.0])
        best = max(cands, key=lambda t: EV.semantic(cms_at(t))["Dice"])
        threshold = float(best)
    cms = cms_at(threshold)
    sem = EV.semantic(cms)
    metrics = {"mask_AP": m["AP"], "mask_AP50": m["AP50"], "mask_AP75": m["AP75"], "box_AP": b["AP"],
               "box_AP50": b["AP50"], "box_AP75": b["AP75"], **{f"recall@{k}": v for k, v in rec.items()},
               "tie_fraction": tie, "semantic_threshold": threshold, "mIoU": sem["mIoU"], "Dice": sem["Dice"],
               "Dice_per_image": sem["Dice_per_image"], "detections_per_image": len(keys) / max(len(dets), 1)}
    stats = {"mask": [(s.tolist(), i.tolist()) for s, i in im_mask], "box": [(s.tolist(), i.tolist()) for s, i in im_box],
             "cms": np.asarray(cms).tolist()}
    return metrics, stats
