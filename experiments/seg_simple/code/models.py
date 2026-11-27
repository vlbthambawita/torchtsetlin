"""The Tsetlin machines of the programme behind one interface (``Net``), and the metrics.

    dense   tt.SegmentationTsetlinMachine: every pixel is an example of its k x k patch (the tutorial's
            model; plan section 5.1). ``sampled`` trains on 256 random pixels per image and epoch
            (the others carry ``ignore_index``).
    pool    the same machine with patch 1 on pyramid planes (``data.pool_planes``), trained on pixels
            sampled from all training images and mixed before they are committed (phase ``improve``)
    win2d   tt.ConvTsetlinMachine on the 9 x 9 window of a pixel, label = the centre pixel
    win3d   tt.Conv3dTsetlinMachine on the same window as a volume (3 colours, D planes, 9, 9)

A job dict (``jobs.py``) holds the settings: model, n, k, kd, enc, pos, t_ratio, aug, bal, order,
sampled, px. Randomness comes from the global torch generators, which ``train.py`` checkpoints.
"""

import torch
import torchtsetlin as tt
from common import IGNORE, SAMPLED, WINDOW
from data import augment, encode, planes, pool_planes, sample_pixels

BATCH_IMAGES = 8                                     # the tutorial's batch (dense models)
BATCH_WINDOWS = 64                                   # window models (DECISIONS D2: 256 made the clauses collapse)
EVAL_IMAGES = {"dense": 32, "pool": 16, "win2d": 4, "win3d": 2}  # images per evaluation chunk
POOL_IMAGES = 50                                     # images encoded at a time when the pixel pool is built
COMMIT = 128                                         # pixels per commit (dense and pool models)


class Confusion:
    """Binary confusion matrix over pixels; ``IGNORE`` labels are skipped."""

    def __init__(self):
        self.m = None

    def add(self, pred, target):
        keep = target != IGNORE
        c = torch.bincount((target[keep] * 2 + pred[keep]).flatten(), minlength=4).view(2, 2).double()
        self.m = c if self.m is None else self.m + c
        return self

    def metrics(self):
        m = self.m.cpu()
        tn, fp, fn, tp = m[0, 0].item(), m[0, 1].item(), m[1, 0].item(), m[1, 1].item()
        iou_p = tp / max(tp + fp + fn, 1.0)
        iou_b = tn / max(tn + fp + fn, 1.0)
        return {"mIoU": (iou_p + iou_b) / 2, "IoU_polyp": iou_p, "IoU_background": iou_b,
                "Dice": 2 * tp / max(2 * tp + fp + fn, 1.0), "pixel_acc": (tp + tn) / max(tp + tn + fp + fn, 1.0)}


def margin(job):
    return max(1, round(job["t_ratio"] * job["n"]))


class Net:
    def __init__(self, job, device):
        self.job, self.dev, self.kind, self.enc = job, torch.device(device), job["model"], job["enc"]
        n, k, T = job["n"], job["k"], margin(job)
        kw = dict(weighted=True, n_states=100)
        if self.kind == "dense":
            self.m = tt.SegmentationTsetlinMachine(
                2, n_clauses=n, T=T, s=10.0, patch_size=k, position_encoding=8 if job["pos"] else False,
                patches_per_commit=COMMIT, pixel_order=job["order"], ignore_index=IGNORE if job["sampled"] else None, **kw)
        elif self.kind == "pool":
            self.m = tt.SegmentationTsetlinMachine(2, n_clauses=n, T=T, s=10.0, patch_size=1, position_encoding=False,
                                                   patches_per_commit=COMMIT, **kw)
        elif self.kind == "win2d":
            self.m = tt.ConvTsetlinMachine(2, n, T=T, s=10.0, patch_size=k, position_encoding=True, **kw)
        elif self.kind == "win3d":
            self.m = tt.Conv3dTsetlinMachine(2, n, T=T, s=10.0, patch_size=(job["kd"], k, k), position_encoding=True, **kw)
        else:
            raise ValueError(self.kind)
        self.m.to(self.dev)
        self.T = float(T)
        self._pad = {}

    # ---------------------------------------------------------------- windows
    def _padded(self, x, key):
        """Encoded images with a zero border of half a window, cached per split."""
        if key not in self._pad or self._pad[key].shape[0] != x.shape[0]:
            p = WINDOW // 2
            self._pad[key] = torch.nn.functional.pad(encode(x, self.enc), (p, p, p, p))
        return self._pad[key]

    def _windows(self, P, b, yy, xx):
        ar = torch.arange(WINDOW, device=P.device)
        w = P[b[:, None, None], :, :, (yy[:, None] + ar)[:, :, None], (xx[:, None] + ar)[:, None, :]]
        w = w.permute(0, 3, 4, 1, 2)                                         # (M, 3, D, 9, 9)
        return w.flatten(1, 2) if self.kind == "win2d" else w

    def _planes(self, x):
        return pool_planes(x, self.enc, self.job["scales"], self.job["pos"])

    # ---------------------------------------------------------------- training
    @torch.no_grad()
    def _pool_epoch(self, x, y):
        """Sampled pixels of all images, mixed, then committed ``COMMIT`` at a time.

        The library's ``update`` commits the pixels of one image chunk in order, so the pixels reach
        the machine's own feedback path (``_accumulate_literals``, ``_commit``) directly here
        (LIBRARY_GAPS SS-4). Nothing of the feedback is changed."""
        job, conf, m = self.job, Confusion(), self.m
        if m.input_shape is None:
            self.votes(x[:1])                                                # lazy shapes
        feats, labs = [], []
        for i in range(0, x.shape[0], POOL_IMAGES):
            xb, yb = x[i:i + POOL_IMAGES], y[i:i + POOL_IMAGES]
            if job["aug"] == "sym":
                xb, yb, _ = augment(xb, yb, None)
            idx = sample_pixels(yb, job["pool_n"], job["pool_a"], job["pool_b"])           # (B, n)
            f = self._planes(xb).flatten(2).transpose(1, 2)                                # (B, H*W, Z)
            feats.append(f.gather(1, idx.unsqueeze(-1).expand(-1, -1, f.shape[-1])).flatten(0, 1))
            labs.append(yb.flatten(1).gather(1, idx).flatten())
        f, lab = torch.cat(feats), torch.cat(labs)
        order = torch.randperm(lab.numel(), device=self.dev)
        for i in range(0, lab.numel(), COMMIT):
            j = order[i:i + COMMIT]
            fj = f[j].to(m.compute_dtype)
            acc = m._new_accumulator()
            v = m._accumulate_literals(torch.cat([fj, 1.0 - fj], 1), lab[j], acc)
            m._commit(acc)
            conf.add(v.argmax(1), lab[j])
        return conf

    def train_epoch(self, x, y):
        """One pass over the training images. Returns the running confusion (pre-update predictions)."""
        self.m.train()
        if self.kind == "pool":
            return self._pool_epoch(x, y)
        conf, N, (H, W) = Confusion(), x.shape[0], y.shape[-2:]
        if self.kind == "dense":
            job = self.job
            if job["bal"] and getattr(self, "_bal", None) is None:
                self._bal = tt.data.balanced_class_probabilities(y, 2, floor=0.05)
                self.m.set_class_feedback_p(self._bal.to(self.dev))
            perm = torch.randperm(N, device=self.dev)
            for i in range(0, N, BATCH_IMAGES):
                idx = perm[i:i + BATCH_IMAGES]
                xb, yb = x[idx], y[idx]
                if job["aug"] == "sym":
                    xb, yb, _ = augment(xb, yb, None)
                if job["sampled"]:
                    keep = torch.rand(yb.shape[0], H * W, device=self.dev).argsort(1)[:, :SAMPLED]
                    lab = torch.full((yb.shape[0], H * W), IGNORE, device=self.dev, dtype=yb.dtype)
                    yb = lab.scatter(1, keep, yb.flatten(1).gather(1, keep)).view_as(yb)
                v = self.m.update(planes(xb, self.enc), yb)
                conf.add(v.argmax(1), yb.flatten())
            return conf
        P = self._padded(x, "train")
        pos = torch.rand(N, H * W, device=self.dev).argsort(1)[:, :SAMPLED]      # pixels of each image
        b = torch.arange(N, device=self.dev).repeat_interleave(SAMPLED)
        pos = pos.flatten()
        order = torch.randperm(b.numel(), device=self.dev)
        for i in range(0, b.numel(), BATCH_WINDOWS):
            j = order[i:i + BATCH_WINDOWS]
            bb, yy, xx = b[j], pos[j] // W, pos[j] % W
            lab = y[bb, yy, xx]
            v = self.m.update(self._windows(P, bb, yy, xx), lab)
            conf.add(v.argmax(1), lab)
        return conf

    # ---------------------------------------------------------------- prediction
    @torch.no_grad()
    def votes(self, x, key=None):
        """Vote maps ``(N, 2, H, W)`` of uint8 images, in prediction mode."""
        self.m.eval()
        out, N, (H, W) = [], x.shape[0], x.shape[-2:]
        step = EVAL_IMAGES[self.kind]
        if self.kind == "dense":
            for i in range(0, N, step):
                out.append(self.m.vote_map(planes(x[i:i + step], self.enc)).float())
        elif self.kind == "pool":
            for i in range(0, N, step):
                out.append(self.m.vote_map(self._planes(x[i:i + step])).float())
        else:
            P = self._padded(x, key) if key else torch.nn.functional.pad(encode(x, self.enc), (WINDOW // 2,) * 4)
            g = torch.arange(H * W, device=self.dev)
            for i in range(0, N, step):
                n = min(step, N - i)
                b = torch.arange(i, i + n, device=self.dev).repeat_interleave(H * W)
                p = g.repeat(n)
                v = self.m(self._windows(P, b, p // W, p % W)).float()
                out.append(v.view(n, H, W, 2).permute(0, 3, 1, 2))
        self.m.train()
        return torch.cat(out)

    def predict(self, x, key=None):
        return self.votes(x, key).argmax(1)

    # ---------------------------------------------------------------- state
    def state(self):
        return {k: v.clone() for k, v in self.m.state_dict().items()}

    def load(self, state, x=None):
        if getattr(self.m, "ta_state", None) is None or self.m.ta_state.numel() == 0:
            self.votes(x[:1])                                                # lazy shapes before loading
        self.m.load_state_dict(state)

    def info(self):
        c = self.m.include_count.float()
        return {"n_features": int(self.m.ta_state.shape[1] // 2), "clauses_total": int(self.m.ta_state.shape[0]),
                "literals_per_clause": float(c.mean()), "empty_clauses": int((c == 0).sum()),
                "included_literals": int(c.sum())}
