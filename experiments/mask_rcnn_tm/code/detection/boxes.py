"""Anchors, assignment, box deltas and their bins, proposals."""

import math

import torch
from torchvision.ops import box_iou, nms

SCALES = (10, 14, 20, 28, 40, 56, 80, 112)
RATIOS = (0.5, 1.0, 2.0)
STRIDE = 4                                           # stride 8 misses gate K0b on the 128 px grid (DECISIONS.md)
DELTA_RANGE = (0.5, 0.5, 0.7, 0.7)                   # dx, dy, dw, dh are binned on [-r, r]


class Anchors:
    """Anchors of a ``size`` x ``size`` image, ordered ``(cell, shape)`` with cells row-major and
    shape index ``scale_idx * len(ratios) + ratio_idx``. ``ratio`` is height / width."""

    def __init__(self, size=128, stride=STRIDE, scales=SCALES, ratios=RATIOS, device="cpu"):
        self.size, self.stride, self.scales, self.ratios = size, stride, tuple(scales), tuple(ratios)
        self.n_side = size // stride
        self.n_cells, self.A = self.n_side ** 2, len(scales) * len(ratios)
        c = (torch.arange(self.n_side, dtype=torch.float32) + 0.5) * stride
        cy, cx = torch.meshgrid(c, c, indexing="ij")
        wh = torch.tensor([[s / math.sqrt(r), s * math.sqrt(r)] for s in scales for r in ratios])
        cx, cy = cx.reshape(-1, 1), cy.reshape(-1, 1)
        w, h = wh[:, 0].view(1, -1), wh[:, 1].view(1, -1)
        self.boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1).reshape(-1, 4).to(device)
        self.cell = torch.arange(self.n_cells).view(-1, 1).expand(-1, self.A).reshape(-1).to(device)
        self.shape = torch.arange(self.A).view(1, -1).expand(self.n_cells, -1).reshape(-1).to(device)
        self.centre = torch.cat([cx, cy], 1).to(device)          # (cells, 2)
        self.device = torch.device(device)

    def __len__(self):
        return self.boxes.shape[0]

    def shape_bits(self):
        """``(A, n_scales - 1 + n_ratios)`` Boolean: thermometer of the scale index, one-hot of
        the ratio. A clause can say "scale at least 40" with one literal."""
        ns, nr = len(self.scales), len(self.ratios)
        a = torch.arange(self.A)
        si, ri = a // nr, a % nr
        therm = si.view(-1, 1) > torch.arange(ns - 1).view(1, -1)
        onehot = ri.view(-1, 1) == torch.arange(nr).view(1, -1)
        return torch.cat([therm, onehot], 1).to(self.device)

    def valid(self, inst, ignore=255):
        """Anchors whose centre is not on letterbox padding: ``(n_anchors,)`` Boolean."""
        c = self.centre.long()
        ok = inst[c[:, 1], c[:, 0]] != ignore
        return ok[self.cell]

    def describe(self):
        return {"size": self.size, "stride": self.stride, "scales": list(self.scales), "ratios": list(self.ratios),
                "A": self.A, "anchors": len(self)}


def assign(anchors, valid, gt, hi=0.7, lo=0.3):
    """Label per anchor: 1 positive (IoU >= ``hi`` or the best valid anchor of a ground-truth
    box), 0 negative (max IoU < ``lo``), -1 ignored. Returns ``(labels, matched_gt)``."""
    n = anchors.shape[0]
    labels = torch.full((n,), -1, dtype=torch.long, device=anchors.device)
    matched = torch.zeros(n, dtype=torch.long, device=anchors.device)
    if gt.shape[0] == 0:
        labels[valid] = 0
        return labels, matched
    iou = box_iou(anchors, gt)
    best, matched = iou.max(1)
    labels[best < lo] = 0
    labels[best >= hi] = 1
    iv = iou * valid.unsqueeze(1)
    top = iv.argmax(0)                               # the best valid anchor of each box
    labels[top] = 1
    matched[top] = torch.arange(gt.shape[0], device=anchors.device)
    labels[~valid] = -1
    return labels, matched


def sample_labels(labels, n, pos_frac, gen):
    """Indices of at most ``n`` anchors (or RoIs), at most ``pos_frac`` of them positive."""
    pos = torch.nonzero(labels == 1, as_tuple=True)[0]
    neg = torch.nonzero(labels == 0, as_tuple=True)[0]
    n_pos = min(len(pos), int(n * pos_frac))
    n_neg = min(len(neg), n - n_pos)
    pp = torch.randperm(len(pos), generator=gen)[:n_pos].to(pos.device)
    pn = torch.randperm(len(neg), generator=gen)[:n_neg].to(neg.device)
    return pos[pp], neg[pn]


def encode(ref, gt):
    """Deltas ``dx, dy, dw, dh`` of ``gt`` against the reference boxes (no weights)."""
    wa, ha = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    xa, ya = ref[:, 0] + wa / 2, ref[:, 1] + ha / 2
    w, h = (gt[:, 2] - gt[:, 0]).clamp(min=1e-3), (gt[:, 3] - gt[:, 1]).clamp(min=1e-3)
    x, y = gt[:, 0] + w / 2, gt[:, 1] + h / 2
    return torch.stack([(x - xa) / wa, (y - ya) / ha, torch.log(w / wa), torch.log(h / ha)], 1)


def decode(ref, d):
    wa, ha = ref[:, 2] - ref[:, 0], ref[:, 3] - ref[:, 1]
    xa, ya = ref[:, 0] + wa / 2, ref[:, 1] + ha / 2
    x, y = xa + d[:, 0] * wa, ya + d[:, 1] * ha
    w, h = wa * torch.exp(d[:, 2].clamp(max=4.0)), ha * torch.exp(d[:, 3].clamp(max=4.0))
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)


class DeltaBins:
    """Deltas as Boolean outputs (decision D-4). ``ordinal``: one output per threshold between
    bins ("is the delta above this threshold?"), decoded by counting the positive outputs.
    Otherwise one output per bin, decoded by arg-max. Out-of-range targets go to the end bins."""

    def __init__(self, width=0.1, ordinal=True):
        self.width, self.ordinal = float(width), bool(ordinal)
        self.n_bins = [int(round(2 * r / width)) for r in DELTA_RANGE]
        self.sizes = [n - 1 for n in self.n_bins] if ordinal else list(self.n_bins)
        self.n_outputs = sum(self.sizes)

    def bin_index(self, d):
        r = torch.tensor(DELTA_RANGE, device=d.device)
        n = torch.tensor(self.n_bins, device=d.device)
        return torch.minimum(((d + r) / self.width).floor().clamp(min=0).long(), n - 1)

    def targets(self, d):
        """``(N, 4)`` deltas -> ``(N, n_outputs)`` Boolean targets, and the out-of-range count."""
        b = self.bin_index(d)
        r = torch.tensor(DELTA_RANGE, device=d.device)
        out_of_range = int((d.abs() > r).any(1).sum())
        cols = []
        for i, n in enumerate(self.n_bins):
            t = torch.arange(n - 1 if self.ordinal else n, device=d.device).view(1, -1)
            cols.append(b[:, i:i + 1] > t if self.ordinal else b[:, i:i + 1] == t)
        return torch.cat(cols, 1), out_of_range

    def decode(self, votes):
        """``(N, n_outputs)`` vote sums (or margins) -> ``(N, 4)`` deltas at the bin centres."""
        out, o = [], 0
        for i, size in enumerate(self.sizes):
            v = votes[:, o:o + size]
            o += size
            b = (v > 0).sum(1) if self.ordinal else v.argmax(1)
            out.append(-DELTA_RANGE[i] + (b.float() + 0.5) * self.width)
        return torch.stack(out, 1)


def clip(boxes, rect):
    """Clip to the valid rectangle ``x1, y1, x2, y2`` (one rectangle, or one per box)."""
    rect = rect.expand(boxes.shape[0], 4) if rect.dim() == 1 else rect
    return torch.stack([torch.maximum(torch.minimum(boxes[:, 0], rect[:, 2]), rect[:, 0]),
                        torch.maximum(torch.minimum(boxes[:, 1], rect[:, 3]), rect[:, 1]),
                        torch.maximum(torch.minimum(boxes[:, 2], rect[:, 2]), rect[:, 0]),
                        torch.maximum(torch.minimum(boxes[:, 3], rect[:, 3]), rect[:, 1])], 1)


def proposals(scores, boxes, valid, rect, pre=1000, nms_thr=0.7, post=100, min_side=2.0):
    """Top ``pre`` valid boxes by score -> clip -> drop sides under ``min_side`` -> NMS -> keep
    ``post``. Returns ``(boxes, scores, index into the input)``, best first."""
    s = torch.where(valid, scores.float(), torch.full_like(scores, float("-inf"), dtype=torch.float32))
    k = min(pre, int(valid.sum()))
    top = s.topk(k).indices
    b = clip(boxes[top], rect)
    ok = ((b[:, 2] - b[:, 0]) >= min_side) & ((b[:, 3] - b[:, 1]) >= min_side)
    top, b = top[ok], b[ok]
    keep = nms(b, s[top], nms_thr)[:post]
    return b[keep], s[top][keep], top[keep]


__all__ = ["Anchors", "DeltaBins", "assign", "sample_labels", "encode", "decode", "clip", "proposals", "box_iou", "nms"]
