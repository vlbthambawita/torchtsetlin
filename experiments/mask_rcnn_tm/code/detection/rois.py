"""Regions: jittered and background boxes, crops, mask targets and paste-back."""

import torch
import torch.nn.functional as TF
from torchvision.ops import box_iou, roi_align

PAD = 64                                             # zero margin around the image for crops


def scale_boxes(b, factor):
    w, h = (b[:, 2] - b[:, 0]) * factor, (b[:, 3] - b[:, 1]) * factor
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def jitter(boxes, rect, gen, lo=0.5, hi=0.9, tries=48):
    """One perturbed copy of each box whose IoU with it is close to a uniform draw from
    ``[lo, hi]`` (clipped to ``rect``). Candidates are random shifts and scalings; the one whose
    IoU is closest to the target is returned, with its IoU."""
    n, dev = boxes.shape[0], boxes.device
    if n == 0:
        return boxes, torch.zeros(0, device=dev)
    target = lo + (hi - lo) * torch.rand(n, generator=gen).to(dev)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    r = torch.rand(n, tries, 4, generator=gen).to(dev) * 2 - 1
    amp = torch.rand(n, tries, 1, generator=gen).to(dev)             # small and large perturbations
    r = r * amp
    cx = (boxes[:, 0] + boxes[:, 2]).view(n, 1) / 2 + r[..., 0] * 0.4 * w.view(n, 1)
    cy = (boxes[:, 1] + boxes[:, 3]).view(n, 1) / 2 + r[..., 1] * 0.4 * h.view(n, 1)
    ww = w.view(n, 1) * torch.exp(r[..., 2] * 0.5)
    hh = h.view(n, 1) * torch.exp(r[..., 3] * 0.5)
    cand = torch.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], -1)
    rc = rect.view(1, 1, 4) if rect.dim() == 1 else rect.view(n, 1, 4)
    cand = torch.stack([cand[..., 0].maximum(rc[..., 0]), cand[..., 1].maximum(rc[..., 1]),
                        cand[..., 2].minimum(rc[..., 2]), cand[..., 3].minimum(rc[..., 3])], -1)
    ix1, iy1 = torch.maximum(cand[..., 0], boxes[:, None, 0]), torch.maximum(cand[..., 1], boxes[:, None, 1])
    ix2, iy2 = torch.minimum(cand[..., 2], boxes[:, None, 2]), torch.minimum(cand[..., 3], boxes[:, None, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area = (cand[..., 2] - cand[..., 0]).clamp(min=0) * (cand[..., 3] - cand[..., 1]).clamp(min=0)
    iou = inter / (area + (w * h).view(n, 1) - inter).clamp(min=1e-6)
    pick = (iou - target.view(n, 1)).abs().argmin(1)
    ar = torch.arange(n, device=dev)
    return cand[ar, pick], iou[ar, pick]


def background_boxes(n, gt, rect, gen, max_iou=0.5, lo=8.0, hi=112.0):
    """``n`` random boxes inside ``rect`` whose IoU with every ground-truth box is below
    ``max_iou``. Half of the draws are placed near a ground-truth box (hard negatives)."""
    dev = rect.device
    m = 6 * n
    u = torch.rand(m, 5, generator=gen).to(dev)
    side = lo * (hi / lo) ** u[:, 0]
    ratio = 2.0 ** (u[:, 1] * 2 - 1)
    w, h = side / ratio.sqrt(), side * ratio.sqrt()
    cx = rect[0] + u[:, 2] * (rect[2] - rect[0])
    cy = rect[1] + u[:, 3] * (rect[3] - rect[1])
    if gt.shape[0]:
        near = u[:, 4] < 0.5
        g = gt[torch.randint(0, gt.shape[0], (m,), generator=gen).to(dev)]
        gw, gh = g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
        cx = torch.where(near, (g[:, 0] + g[:, 2]) / 2 + (u[:, 2] * 2 - 1) * gw, cx)
        cy = torch.where(near, (g[:, 1] + g[:, 3]) / 2 + (u[:, 3] * 2 - 1) * gh, cy)
    b = torch.stack([(cx - w / 2).maximum(rect[0]), (cy - h / 2).maximum(rect[1]),
                     (cx + w / 2).minimum(rect[2]), (cy + h / 2).minimum(rect[3])], 1)
    ok = ((b[:, 2] - b[:, 0]) >= 4) & ((b[:, 3] - b[:, 1]) >= 4)
    if gt.shape[0]:
        ok &= box_iou(b, gt).max(1).values < max_iou
    return b[ok][:n]


def crop(x, batch_idx, boxes, size, factor=1.0, mode="image"):
    """Crop ``boxes`` (scaled by ``factor`` about their centres) from ``x`` ``(B, C, H, W)`` and
    resize to ``size`` x ``size`` (RoIAlign with the adaptive sampling grid: exact at scale 1,
    averaged when a box is larger than the crop). Outside the image the crop is zero.

    ``mode="image"`` returns rounded uint8-valued floats; ``mode="mask"`` a Boolean ``> 0.5``."""
    b = scale_boxes(boxes, factor) if factor != 1.0 else boxes
    xf = TF.pad(x.float(), (PAD, PAD, PAD, PAD))
    rois = torch.cat([batch_idx.view(-1, 1).float(), b + PAD], 1)
    out = roi_align(xf, rois, output_size=(size, size), spatial_scale=1.0, sampling_ratio=0, aligned=True)
    return out > 0.5 if mode == "mask" else out.round_()


def paste(margins, boxes, size):
    """Paste-back: the soft margin maps ``(R, m, m)`` are resized bilinearly to their boxes and
    thresholded at 0. Returns ``(R, size, size)`` Boolean masks; nothing outside a box is set."""
    R = margins.shape[0]
    if R == 0:
        return torch.zeros(0, size, size, dtype=torch.bool, device=margins.device)
    c = torch.arange(size, device=margins.device, dtype=torch.float32) + 0.5
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-3).view(R, 1)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-3).view(R, 1)
    u = (c.view(1, -1) - boxes[:, 0:1]) / w          # (R, size) position inside the box, 0..1
    v = (c.view(1, -1) - boxes[:, 1:2]) / h
    inside = ((u >= 0) & (u <= 1)).view(R, 1, size) & ((v >= 0) & (v <= 1)).view(R, size, 1)
    grid = torch.stack([(2 * u - 1).view(R, 1, size).expand(R, size, size),
                        (2 * v - 1).view(R, size, 1).expand(R, size, size)], -1)
    s = TF.grid_sample(margins.float().unsqueeze(1), grid, mode="bilinear", padding_mode="border",
                       align_corners=False)[:, 0]
    return (s > 0) & inside


def mask_boxes(masks):
    """Tight boxes of Boolean masks ``(R, H, W)``; an empty mask gives a zero box."""
    R = masks.shape[0]
    out = torch.zeros(R, 4, device=masks.device)
    rows, cols = masks.any(2), masks.any(1)
    H, W = masks.shape[1], masks.shape[2]
    ar_h = torch.arange(H, device=masks.device).view(1, -1)
    ar_w = torch.arange(W, device=masks.device).view(1, -1)
    has = rows.any(1)
    out[:, 0] = torch.where(cols, ar_w, torch.full_like(ar_w, W)).min(1).values
    out[:, 1] = torch.where(rows, ar_h, torch.full_like(ar_h, H)).min(1).values
    out[:, 2] = torch.where(cols, ar_w, torch.full_like(ar_w, -1)).max(1).values + 1
    out[:, 3] = torch.where(rows, ar_h, torch.full_like(ar_h, -1)).max(1).values + 1
    out[~has] = 0
    return out


def mask_iou(a, b):
    """IoU matrix ``(n, k)`` of Boolean masks ``(n, H, W)`` and ``(k, H, W)``."""
    af, bf = a.flatten(1).double(), b.flatten(1).double()
    inter = af @ bf.T
    union = af.sum(1, keepdim=True) + bf.sum(1).view(1, -1) - inter
    return inter / union.clamp(min=1.0)
