"""The heads of the pipeline (PLAN_KVASIR §6-§8): how each one builds its inputs, what one
training epoch is, its fixed validation set and early-stopping metric, and its prediction panel.

    rpn_obj    polyp / not, per (cell, anchor shape) row            monitor: mean recall@{1,5,10,30}
    rpn_delta  44 ordinal box-delta outputs at positive anchors     monitor: mean IoU of the decoded box
    cls        polyp / background per RoI crop                      monitor: balanced accuracy
    box        44 ordinal delta outputs per foreground RoI crop     monitor: mean IoU after refinement
    mask       inside / outside per pixel of a 32 x 32 RoI crop     monitor: mean mask IoU per RoI
    dense      polyp / background per image pixel (baseline v-a)    monitor: mIoU

Every head trains its members on rows or crops gathered from a chunk of images, shuffled across
the chunk, and committed in mini-batches (training mechanism of TM B.0: flat models on gathered
rows). Stage-2 heads take their regions from the ground truth (``src="gt"``: steps S1-S2) or
from cached proposals (``src=<proposal cache>``: step S6).
"""

import boolenc as E
import numpy as np
import torch
from augment import make_view
from common import CACHE, IGNORE, VAL_SEED, S
from detection import boxes as B
from detection import evaluate as EV
from detection import rois as R
from members import Team

GROUP = 64                                           # images whose rows are shuffled together


def chunks(n, size):
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def load_props(name):
    p = CACHE / "props" / f"{name}.pt"
    if not p.exists():
        raise SystemExit(f"proposal cache {p} is missing (run the `props` job first)")
    return torch.load(p, weights_only=False)


class Head:
    name = ""
    n_out = 0
    batch = 128

    def __init__(self, job, data, device, inference=False):
        """``inference=True`` builds the head only to predict with (the pipeline): no training
        images, no proposal cache, no validation set."""
        self.job, self.d, self.dev = job, data, torch.device(device)
        self.specs = job["members"]
        self.batch = int(job.get("batch", self.batch))
        self.seed = int(job.get("seed", 0))
        self.aug = job.get("aug", "sym")
        self.src = job.get("src", "gt")
        self.running = [0.0, 0]
        self.inference = inference
        self.props = None
        if inference:
            return
        self.tr, self.va = data.image_sets(job.get("fold"))
        if job.get("subset"):                        # smoke runs
            self.tr, self.va = self.tr[: int(job["subset"])], self.va[: max(8, int(job["subset"]) // 4)]
        self.props = None if self.src == "gt" else load_props(self.src)
        self.val_view = make_view(data, self.va, "none", props=None if self.props is None else self.props["boxes"])
        self.setup()

    def setup(self):
        """Fixed validation examples (not needed for inference)."""

    # ---------------------------------------------------------------- shared machinery
    def team(self):
        return Team(self.specs, self.n_out, self.job.get("combine", "single"), self.dev)

    def view(self, epoch):
        return make_view(self.d, self.tr, self.aug, self.seed, epoch,
                         props=None if self.props is None else self.props["boxes"])

    def commit(self, team, Xs, y, gen):
        """Shuffle the gathered examples and update in mini-batches."""
        n = y.shape[0]
        perm = torch.randperm(n, generator=gen).to(self.dev)
        for i in range(0, n, self.batch):
            idx = perm[i:i + self.batch]
            yb = y[idx]
            v = team.update([X[idx] for X in Xs], yb)
            if v is not None:
                v = v.float()
                hit = ((v > 0) == yb) if self.n_out else ((v[:, 1] > v[:, 0]) == yb.bool())
                self.running[0] += float(hit.float().mean()) * len(idx)
                self.running[1] += len(idx)
        return n

    def train_epoch(self, team, epoch):
        team.train()
        self.running = [0.0, 0]
        gen = torch.Generator().manual_seed(7_919 * self.seed + 104_729 * epoch + 1)
        view = self.view(epoch)
        n = 0
        for g in chunks(len(view), GROUP):
            Xs, y = self.gather(view, g, gen, train=True)
            if y is not None and y.shape[0]:
                n += self.commit(team, Xs, y, gen)
        out = {"train_examples": n}
        if self.running[1]:
            out["train_acc_running"] = self.running[0] / self.running[1]
        return out

    def gather(self, view, g, gen, train):           # pragma: no cover - per head
        raise NotImplementedError

    def validate(self, team):                        # pragma: no cover - per head
        raise NotImplementedError

    def panel(self, team, x, inst):                  # pragma: no cover - per head
        raise NotImplementedError


# ============================================================================== RPN
class RpnBase(Head):
    def __init__(self, job, data, device, inference=False):
        self.anchors = B.Anchors(S, device=torch.device(device), **job.get("anchors", {}))
        self.bits = self.anchors.shape_bits()
        self.per_shape = bool(job.get("per_shape"))  # control arm RPN-A: one model per anchor shape
        super().__init__(job, data, device, inference)

    def feats(self, spec, x):
        """``(B, cells, F)``: the pyramid rows at the stride grid."""
        py = E.pyramid(E.planes(x, spec["enc"]), spec["scales"], spec["patch"])
        o, st = self.anchors.stride // 2, self.anchors.stride
        r = py[:, :, o::st, o::st]
        return r.permute(0, 2, 3, 1).reshape(x.shape[0], self.anchors.n_cells, -1)

    def rows(self, feats, b, a_idx):
        """Rows of anchors ``a_idx`` of image ``b``: the cell row plus the shape bits."""
        f = feats[b, self.anchors.cell[a_idx]]
        return f if self.per_shape else torch.cat([f, self.bits[self.anchors.shape[a_idx]]], 1)

    def all_rows(self, feats):
        Bn, C, F = feats.shape
        A = self.anchors.A
        f = feats[:, :, None, :].expand(Bn, C, A, F)
        if self.per_shape:
            return f.reshape(-1, F)
        return torch.cat([f, self.bits[None, None].expand(Bn, C, A, -1)], -1).reshape(-1, F + self.bits.shape[1])

    def assigned(self, view, i):
        valid = self.anchors.valid(view.inst[i])
        hi, lo = self.job.get("iou_hi", 0.7), self.job.get("iou_lo", 0.3)
        lab, matched = B.assign(self.anchors.boxes, valid, view.boxes[i], hi, lo)
        return lab, matched, valid


class RpnObj(RpnBase):
    name = "rpn_obj"

    def team(self):
        if not self.per_shape:
            return super().team()
        return Team(self.specs * self.anchors.A, 0, "composite", self.dev)

    def gather(self, view, g, gen, train):
        Xs, ys, sh = [[] for _ in self.specs], [], []
        for sub in chunks(len(g), 16):
            ii = [g[j] for j in sub]
            feats = [self.feats(s, view.x[ii]) for s in self.specs]
            for j, i in enumerate(ii):
                lab, _, _ = self.assigned(view, i)
                pos, neg = B.sample_labels(lab, 256, 0.5, gen)
                a = torch.cat([pos, neg])
                for m, f in enumerate(feats):
                    Xs[m].append(self.rows(f, j, a))
                ys.append(torch.cat([torch.ones_like(pos), torch.zeros_like(neg)]))
                sh.append(self.anchors.shape[a])
        self._shape = torch.cat(sh)
        return [torch.cat(X) for X in Xs], torch.cat(ys)

    def commit(self, team, Xs, y, gen):
        if not self.per_shape:
            return super().commit(team, Xs, y, gen)
        perm = torch.randperm(y.shape[0], generator=gen).to(self.dev)
        X, y, shape = Xs[0][perm], y[perm], self._shape[perm]
        for a in range(self.anchors.A):
            sel = torch.nonzero(shape == a, as_tuple=True)[0]
            for i in range(0, len(sel), self.batch):
                idx = sel[i:i + self.batch]
                team.members[a].update(X[idx], y[idx])
        return int(y.shape[0])

    @torch.no_grad()
    def score(self, team, x):
        """Objectness margin of every anchor: ``(B, n_anchors)``."""
        team.eval()
        if self.per_shape:
            f = self.feats(self.specs[0], x)                                   # (B, cells, F)
            out = torch.stack([team.member_margin(a, f.reshape(-1, f.shape[-1])) for a in range(self.anchors.A)], 1)
            return out.reshape(x.shape[0], -1)
        m = sum(team.member_margin(i, self.all_rows(self.feats(s, x))) for i, s in enumerate(self.specs))
        return m.reshape(x.shape[0], -1)

    def raw_proposals(self, team, view, post=30):
        """Per image ``(boxes, scores, anchor index)`` from the raw anchors (no deltas)."""
        out = []
        for sub in chunks(len(view), 8):
            sc = self.score(team, view.x[sub])
            for j, i in enumerate(sub):
                out.append(B.proposals(sc[j], self.anchors.boxes, self.anchors.valid(view.inst[i]), view.rect[i], post=post))
        return out

    def validate(self, team):
        v = self.val_view
        props = self.raw_proposals(team, v)
        images = [(p[1].cpu().numpy(), B.box_iou(p[0].double(), v.boxes[i].double()).cpu().numpy())
                  for i, p in enumerate(props)]
        rec = EV.recall_at(images)
        out = {f"recall@{k}": r for k, r in rec.items()}
        out["monitor"] = float(np.mean(list(rec.values())))
        return out

    def panel(self, team, x, inst):
        items = []
        view = make_fixed_view(x, inst)
        for p in self.raw_proposals(team, view, post=5):
            items.append({"pred_boxes": p[0].cpu(), "scores": p[1].cpu()})
        return items


class RpnDelta(RpnBase):
    name = "rpn_delta"
    batch = 64

    def __init__(self, job, data, device, inference=False):
        self.bins = B.DeltaBins(**job.get("bins", {}))
        self.n_out = self.bins.n_outputs
        self.out_of_range = 0
        super().__init__(job, data, device, inference)

    def setup(self):
        X, y, ref, gt = self._positives(self.val_view, None)
        self._val = (X, ref, gt)
        self.out_of_range = 0

    def _positives(self, view, gen, g=None):
        Xs, ys, refs, gts = [[] for _ in self.specs], [], [], []
        g = list(range(len(view))) if g is None else g
        for sub in chunks(len(g), 16):
            ii = [g[j] for j in sub]
            feats = [self.feats(s, view.x[ii]) for s in self.specs]
            for j, i in enumerate(ii):
                lab, matched, _ = self.assigned(view, i)
                pos = torch.nonzero(lab == 1, as_tuple=True)[0]
                if pos.numel() == 0:
                    continue
                ref, gt = self.anchors.boxes[pos], view.boxes[i][matched[pos]]
                t, oor = self.bins.targets(B.encode(ref, gt))
                self.out_of_range += oor
                for m, f in enumerate(feats):
                    Xs[m].append(self.rows(f, j, pos))
                ys.append(t)
                refs.append(ref)
                gts.append(gt)
        if not ys:
            return [None] * len(self.specs), None, None, None
        return [torch.cat(X) for X in Xs], torch.cat(ys), torch.cat(refs), torch.cat(gts)

    def gather(self, view, g, gen, train):
        Xs, y, _, _ = self._positives(view, gen, g=g)
        return Xs, y

    @torch.no_grad()
    def deltas(self, team, Xs):
        team.eval()
        return self.bins.decode(team.margin(Xs))

    def validate(self, team):
        Xs, ref, gt = self._val
        dec = B.decode(ref, self.deltas(team, Xs))
        after = torch.diag(B.box_iou(dec, gt))
        before = torch.diag(B.box_iou(ref, gt))
        return {"monitor": float(after.mean()), "iou_anchor": float(before.mean()),
                "iou_gain": float((after - before).mean()), "iou75": float((after >= 0.75).float().mean())}

    def panel(self, team, x, inst):
        items = []
        view = make_fixed_view(x, inst)
        for i in range(len(view)):
            Xs, _, ref, gt = self._positives(view, None, g=[i])
            if ref is None:
                items.append({})
                continue
            dec = B.decode(ref, self.deltas(team, Xs))
            best = torch.diag(B.box_iou(ref, gt)).argsort(descending=True)[:3]
            items.append({"ref_boxes": ref[best].cpu(), "pred_boxes": dec[best].cpu()})
        return items


# ============================================================================== RoI heads
def make_fixed_view(x, inst):
    """A view of panel images (no augmentation), with their boxes."""
    from augment import View
    from data import boxes_of, valid_rect
    return View(x, inst, [boxes_of(m) for m in inst], torch.stack([valid_rect(m) for m in inst]))


class RoiBase(Head):
    n_fg, n_bg = 4, 12                               # regions per image from the ground truth (S1-S2)
    n_prop, fg_frac = 64, 0.25                       # regions per image from proposals (S6)

    def crop_input(self, spec, x, bidx, boxes, inst=None):
        c = R.crop(x, bidx, boxes, int(spec["crop"]), float(spec.get("factor", 1.0)))
        if spec["conv"] == "conv3d":
            return E.volume(c, spec["enc"])
        p = E.planes(c, spec["enc"])
        if spec.get("validity"):
            v = R.crop((inst != IGNORE)[:, None].float(), bidx, boxes, int(spec["crop"]), float(spec.get("factor", 1.0)),
                       mode="mask")
            p = torch.cat([p, v], 1)
        return p

    def regions(self, view, i, gen, fg_only=False, n_fg=None, n_bg=None, append_gt=True):
        """Regions of image ``i``: ``(boxes, label, matched gt index)``; label 1 = polyp."""
        gt, rect, dev = view.boxes[i], view.rect[i], self.dev
        n_fg = self.n_fg if n_fg is None else n_fg
        n_bg = self.n_bg if n_bg is None else n_bg
        if view.props is None:
            if gt.shape[0]:
                pick = torch.randint(0, gt.shape[0], (n_fg,), generator=gen).to(dev)
                fg, _ = R.jitter(gt[pick], rect, gen)
            else:
                pick, fg = torch.zeros(0, dtype=torch.long, device=dev), gt
            if fg_only:
                return fg, torch.ones(len(fg), dtype=torch.long, device=dev), pick
            bg = R.background_boxes(n_bg, gt, rect, gen)
            return (torch.cat([fg, bg]), torch.cat([torch.ones(len(fg)), torch.zeros(len(bg))]).long().to(dev),
                    torch.cat([pick, torch.zeros(len(bg), dtype=torch.long, device=dev)]))
        p = view.props[i]
        p = p[((p[:, 2] - p[:, 0]) >= 2) & ((p[:, 3] - p[:, 1]) >= 2)]
        cand = torch.cat([p, gt]) if gt.shape[0] and append_gt else p    # ground-truth boxes appended
        if gt.shape[0] and cand.shape[0]:
            iou, m = B.box_iou(cand, gt).max(1)
        else:
            iou, m = torch.zeros(len(cand), device=dev), torch.zeros(len(cand), dtype=torch.long, device=dev)
        lab = (iou >= 0.5).long()
        if fg_only:
            sel = torch.nonzero(lab == 1, as_tuple=True)[0]
            sel = sel[torch.randperm(len(sel), generator=gen)[:n_fg].to(dev)]
            return cand[sel], lab[sel], m[sel]
        pos, neg = B.sample_labels(lab, self.n_prop, self.fg_frac, gen)
        sel = torch.cat([pos, neg])
        return cand[sel], lab[sel], m[sel]

    def collect(self, view, g, gen, **kw):
        bidx, boxes, labs, gts = [], [], [], []
        for j, i in enumerate(g):
            b, lab, m = self.regions(view, i, gen, **kw)
            bidx.append(torch.full((len(b),), j, dtype=torch.long, device=self.dev))
            boxes.append(b)
            labs.append(lab)
            gts.append(m)
        return torch.cat(bidx), torch.cat(boxes), torch.cat(labs), torch.cat(gts)

    def fixed_val(self, **kw):
        """The validation regions, drawn once with the fixed seed (the same for every run)."""
        v = self.val_view
        gen = torch.Generator().manual_seed(VAL_SEED)
        if v.props is not None and not kw.get("fg_only"):
            # proposals as they come at inference: the best 30, no ground-truth box appended
            bidx, boxes, labs, gts = [], [], [], []
            for i in range(len(v)):
                p = v.props[i][:30]
                p = p[((p[:, 2] - p[:, 0]) >= 2) & ((p[:, 3] - p[:, 1]) >= 2)]
                gt = v.boxes[i]
                iou, m = (B.box_iou(p, gt).max(1) if gt.shape[0] and len(p) else
                          (torch.zeros(len(p), device=self.dev), torch.zeros(len(p), dtype=torch.long, device=self.dev)))
                bidx.append(torch.full((len(p),), i, dtype=torch.long, device=self.dev))
                boxes.append(p)
                labs.append((iou >= 0.5).long())
                gts.append(m)
            return torch.cat(bidx), torch.cat(boxes), torch.cat(labs), torch.cat(gts)
        return self.collect(v, list(range(len(v))), gen, append_gt=False, **kw)


class RoiCls(RoiBase):
    name = "cls"
    batch = 64

    def setup(self):
        self._val = self.fixed_val()

    def gather(self, view, g, gen, train):
        bidx, boxes, lab, _ = self.collect(view, g, gen)
        x = view.x[g]
        return [self.crop_input(s, x, bidx, boxes) for s in self.specs], lab

    @torch.no_grad()
    def margins(self, team, x, bidx, boxes):
        team.eval()
        out = []
        for c in chunks(len(boxes), 1024):
            out.append(team.margin([self.crop_input(s, x, bidx[c], boxes[c]) for s in self.specs]))
        return torch.cat(out) if out else torch.zeros(0, device=self.dev)

    def validate(self, team):
        bidx, boxes, lab, _ = self._val
        m = self.margins(team, self.val_view.x, bidx, boxes)
        pred = m > 0
        tpr = float(pred[lab == 1].float().mean()) if (lab == 1).any() else 0.0
        tnr = float((~pred[lab == 0]).float().mean()) if (lab == 0).any() else 0.0
        return {"monitor": (tpr + tnr) / 2, "acc_polyp": tpr, "acc_background": tnr,
                "n_polyp": int((lab == 1).sum()), "n_background": int((lab == 0).sum())}

    def panel(self, team, x, inst):
        view = make_fixed_view(x, inst)
        gen = torch.Generator().manual_seed(VAL_SEED + 1)
        bidx, boxes, lab, _ = self.collect(view, list(range(len(view))), gen, n_fg=3, n_bg=5)
        m = self.margins(team, x, bidx, boxes)
        items = []
        for i in range(len(view)):
            sel = bidx == i
            pos = sel & (m > 0)
            items.append({"pred_boxes": boxes[pos].cpu(), "scores": m[pos].cpu(), "ref_boxes": boxes[sel & (m <= 0)].cpu()})
        return items


class RoiBox(RoiBase):
    name = "box"
    batch = 32
    n_fg = 8

    def __init__(self, job, data, device, inference=False):
        self.bins = B.DeltaBins(**job.get("bins", {}))
        self.n_out = self.bins.n_outputs
        self.out_of_range = 0
        super().__init__(job, data, device, inference)

    def setup(self):
        self._val = self.fixed_val(fg_only=True)

    def gather(self, view, g, gen, train):
        bidx, boxes, _, m = self.collect(view, g, gen, fg_only=True)
        if not len(boxes):
            return None, None
        gt = torch.cat([view.boxes[i] for i in g])
        off = torch.tensor([0] + [len(view.boxes[i]) for i in g], device=self.dev).cumsum(0)[:-1]
        t, oor = self.bins.targets(B.encode(boxes, gt[off[bidx] + m]))
        self.out_of_range += oor
        x, inst = view.x[g], view.inst[g]
        return [self.crop_input(s, x, bidx, boxes, inst) for s in self.specs], t

    @torch.no_grad()
    def refine(self, team, x, inst, bidx, boxes):
        team.eval()
        out = []
        for c in chunks(len(boxes), 512):
            mg = team.margin([self.crop_input(s, x, bidx[c], boxes[c], inst) for s in self.specs])
            out.append(B.decode(boxes[c], self.bins.decode(mg)))
        return torch.cat(out) if out else boxes

    def validate(self, team):
        v = self.val_view
        bidx, boxes, _, m = self._val
        gt = torch.stack([v.boxes[int(b)][int(k)] for b, k in zip(bidx, m)])
        new = self.refine(team, v.x, v.inst, bidx, boxes)
        after, before = torch.diag(B.box_iou(new, gt)), torch.diag(B.box_iou(boxes, gt))
        return {"monitor": float(after.mean()), "iou_before": float(before.mean()),
                "iou_gain": float((after - before).mean()), "iou75": float((after >= 0.75).float().mean())}

    def panel(self, team, x, inst):
        view = make_fixed_view(x, inst)
        gen = torch.Generator().manual_seed(VAL_SEED + 1)
        bidx, boxes, _, _ = self.collect(view, list(range(len(view))), gen, fg_only=True, n_fg=2)
        new = self.refine(team, x, inst, bidx, boxes) if len(boxes) else boxes
        return [{"ref_boxes": boxes[bidx == i].cpu(), "pred_boxes": new[bidx == i].cpu()} for i in range(len(view))]


class RoiMask(RoiBase):
    name = "mask"
    batch = 128
    n_fg = 2
    size, pixels = 32, 128

    def __init__(self, job, data, device, inference=False):
        self.size = int(job.get("mask_size", 32))
        self.pixels = int(job.get("pixels", 128))
        super().__init__(job, data, device, inference)

    def setup(self):
        self._val = self.fixed_val(fg_only=True, n_fg=3)

    def feats(self, spec, x, bidx, boxes):
        """``(R, F, m, m)``: the OR pyramid of the crop, plus the position bits."""
        c = R.crop(x, bidx, boxes, self.size, 1.0)
        py = E.pyramid(E.planes(c, spec["enc"]), spec["scales"], spec["patch"])
        if spec.get("pos", True):
            pos = E.position_bits(self.size, self.size, self.dev)
            py = torch.cat([py, pos[None].expand(py.shape[0], -1, -1, -1)], 1)
        return py

    def targets(self, inst, bidx, boxes, m):
        """The matched instance mask cropped to the region box: ``(R, m, m)`` Boolean."""
        gm = (inst[bidx] == (m + 1).view(-1, 1, 1).to(inst.dtype))[:, None].float()
        return R.crop(gm, torch.arange(len(boxes), device=self.dev), boxes, self.size, 1.0, mode="mask")[:, 0]

    def gather(self, view, g, gen, train):
        bidx, boxes, _, m = self.collect(view, g, gen, fg_only=True)
        if not len(boxes):
            return None, None
        x, inst = view.x[g], view.inst[g]
        n = self.size * self.size
        Xs, ys = [[] for _ in self.specs], []
        for c in chunks(len(boxes), 32):
            pix = torch.randint(0, n, (len(c), self.pixels), generator=gen).to(self.dev)
            ys.append(self.targets(inst, bidx[c], boxes[c], m[c]).reshape(len(c), n).gather(1, pix).reshape(-1))
            for k, s in enumerate(self.specs):
                f = self.feats(s, x, bidx[c], boxes[c])
                f = f.permute(0, 2, 3, 1).reshape(len(c), n, -1)
                Xs[k].append(f.gather(1, pix[:, :, None].expand(-1, -1, f.shape[-1])).reshape(-1, f.shape[-1]))
        return [torch.cat(X) for X in Xs], torch.cat(ys).long()

    @torch.no_grad()
    def predict(self, team, x, bidx, boxes):
        """Soft margin maps ``(R, m, m)`` of the regions."""
        team.eval()
        out = []
        for c in chunks(len(boxes), 32):
            Xs = []
            for s in self.specs:
                f = self.feats(s, x, bidx[c], boxes[c])
                Xs.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]))
            out.append(team.margin(Xs).reshape(len(c), self.size, self.size))
        return torch.cat(out) if out else torch.zeros(0, self.size, self.size, device=self.dev)

    def region_iou(self, masks, inst, bidx, m):
        gm = inst[bidx] == (m + 1).view(-1, 1, 1).to(inst.dtype)
        inter = (masks & gm).flatten(1).sum(1).float()
        union = (masks | gm).flatten(1).sum(1).float()
        return inter / union.clamp(min=1)

    def validate(self, team):
        v = self.val_view
        bidx, boxes, _, m = self._val
        masks = R.paste(self.predict(team, v.x, bidx, boxes), boxes, S)
        iou = self.region_iou(masks, v.inst, bidx, m)
        filled = R.paste(torch.ones(len(boxes), 2, 2, device=self.dev), boxes, S)
        fb = self.region_iou(filled, v.inst, bidx, m)
        return {"monitor": float(iou.mean()), "iou_filled_box": float(fb.mean()),
                "iou_minus_filled_box": float((iou - fb).mean()), "iou50": float((iou >= 0.5).float().mean()),
                "n_regions": int(len(boxes))}

    def panel(self, team, x, inst):
        view = make_fixed_view(x, inst)
        items = []
        for i in range(len(view)):
            b = view.boxes[i]
            if not len(b):
                items.append({"pred_mask": torch.zeros(S, S, dtype=torch.bool)})
                continue
            bidx = torch.full((len(b),), i, dtype=torch.long, device=self.dev)
            masks = R.paste(self.predict(team, x, bidx, b), b, S)
            items.append({"pred_mask": masks.any(0).cpu(), "ref_boxes": b.cpu()})
        return items


class Dense(Head):
    name = "dense"
    batch = 128
    pixels = 512

    def __init__(self, job, data, device, inference=False):
        self.pixels = int(job.get("pixels", 512))
        super().__init__(job, data, device, inference)

    def feats(self, spec, x):
        return E.pyramid(E.planes(x, spec["enc"]), spec["scales"], spec["patch"])

    def gather(self, view, g, gen, train):
        Xs, ys = [[] for _ in self.specs], []
        for sub in chunks(len(g), 8):
            ii = [g[j] for j in sub]
            inst = view.inst[ii].reshape(len(ii), -1)
            w = (inst != IGNORE).float()
            pix = torch.multinomial(w.cpu(), self.pixels, replacement=False, generator=gen).to(self.dev)
            ys.append((inst.gather(1, pix) > 0).reshape(-1))
            for k, s in enumerate(self.specs):
                f = self.feats(s, view.x[ii])
                f = f.permute(0, 2, 3, 1).reshape(len(ii), S * S, -1)
                Xs[k].append(f.gather(1, pix[:, :, None].expand(-1, -1, f.shape[-1])).reshape(-1, f.shape[-1]))
        return [torch.cat(X) for X in Xs], torch.cat(ys).long()

    @torch.no_grad()
    def predict(self, team, x, step=1):
        """Margin maps ``(B, S / step, S / step)`` (``step`` > 1 evaluates a pixel sub-grid)."""
        team.eval()
        out = []
        for c in chunks(x.shape[0], 4):
            Xs = []
            for s in self.specs:
                f = self.feats(s, x[c])[:, :, ::step, ::step]
                Xs.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]))
            out.append(team.margin(Xs).reshape(len(c), S // step, S // step))
        return torch.cat(out)

    def validate(self, team):
        """mIoU on every second pixel of the validation images (the full grid is scored by the
        evaluation jobs)."""
        v = self.val_view
        pred = self.predict(team, v.x, step=2) > 0
        t = v.inst[:, ::2, ::2]
        cms = [EV.confusion(p.cpu().numpy(), np.where(tt_ == IGNORE, IGNORE, (tt_ > 0)).astype(np.uint8))
               for p, tt_ in zip(pred, t.cpu().numpy())]
        sem = EV.semantic(cms)
        return {"monitor": sem["mIoU"], "dice": sem["Dice"], "iou_polyp": sem["IoU_polyp"]}

    def panel(self, team, x, inst):
        pred = (self.predict(team, x) > 0) & (inst != IGNORE)
        return [{"pred_mask": p.cpu()} for p in pred]


HEADS = {h.name: h for h in (RpnObj, RpnDelta, RoiCls, RoiBox, RoiMask, Dense)}
