"""The Tsetlin machines of the programme behind one interface (``Net``), and the metrics.

    dense   tt.SegmentationTsetlinMachine: every pixel is an example of its k x k patch (the tutorial's
            model; plan section 5.1). ``sampled`` trains on 256 random pixels per image and epoch
            (the others carry ``ignore_index``).
    pool    the same machine with patch 1 on pyramid planes (``data.pool_planes``), trained on pixels
            sampled from all training images and mixed before they are committed (phase ``improve``)
    cell    tt.ConvTsetlinMachine that classifies a cell of c x c pixels from a w x w window around it
            (phase ``coarse``); its prediction is the cell map, enlarged to the image
    multi   one group of clauses per neighbourhood size (3 x 3, 5 x 5, ... 25 x 25) around the pixel; the groups
            learn from one joint vote sum (phase ``multi``)
    win2d   tt.ConvTsetlinMachine on the 9 x 9 window of a pixel, label = the centre pixel
    win3d   tt.Conv3dTsetlinMachine on the same window as a volume (3 colours, D planes, 9, 9)

A job dict (``jobs.py``) holds the settings: model, n, k, kd, enc, pos, t_ratio, aug, bal, order,
sampled, px. Randomness comes from the global torch generators, which ``train.py`` checkpoints.
"""

import torch
import torchtsetlin as tt
from common import IGNORE, SAMPLED, WINDOW
from data import augment, encode, planes, pool_planes, position_planes, sample_pixels

BATCH_IMAGES = 8                                     # the tutorial's batch (dense models)
BATCH_WINDOWS = 64                                   # window models (DECISIONS D2: 256 made the clauses collapse)
EVAL_IMAGES = {"dense": 32, "pool": 16, "cell": 8, "multi": 1, "win2d": 4, "win3d": 2}  # images per evaluation chunk
POOL_IMAGES = 50                                     # images encoded at a time when the pixel pool is built
COMMIT = 128                                         # pixels per commit (dense, pool and multi models)
MULTI_CHUNK = 8192                                   # pixels per prediction chunk of the multi models


def joint_wide():
    """The joint-feedback team of ``experiments/mask_rcnn_tm`` (members that learn from one vote sum)."""
    import importlib.util
    import sys
    if "joint_wide" not in sys.modules:
        from common import REPO
        spec = importlib.util.spec_from_file_location("joint_wide", REPO / "experiments" / "mask_rcnn_tm" / "code" / "joint_wide.py")
        sys.modules["joint_wide"] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["joint_wide"]


class MultiSize:
    """Groups of clauses that read neighbourhoods of different sizes around one pixel.

    Group g is a machine on the ``k_g x k_g`` neighbourhood of the pixel (all planes) and the position
    bits, given to it as a 1 x 1 image with that many planes. With ``joint`` the groups learn from the
    sum of their votes against the sum of their margins, which is one machine whose clauses differ in
    what they can read. Without it every group learns alone and the votes are only added to predict."""

    def __init__(self, job, T_of, device):
        jw = joint_wide()
        self.sizes, self.kmax, self.pos = list(job["sizes"]), max(job["sizes"]), job["pos"]
        self.members = [jw.JointConv(2, c, T=T_of(c), s=10.0, patch_size=1, position_encoding=False, weighted=True,
                                     n_states=100).joint_setup(COMMIT).to(device) for c in job["counts"]]
        self.joint = bool(job["joint"]) and len(self.members) > 1
        self.team = jw.JointTeam([f"k{k}" for k in self.sizes], self.members, "sum") if self.joint else None
        self._posp = None

    def inputs(self, P, b, yy, xx):
        """One ``(M, Z_g, 1, 1)`` Boolean tensor per group for pixels ``(yy, xx)`` of images ``b``.
        ``P``: planes with a zero border of ``kmax // 2``."""
        K = self.kmax
        ar = torch.arange(K, device=P.device)
        win = P[b[:, None, None], :, (yy[:, None] + ar)[:, :, None], (xx[:, None] + ar)[:, None, :]].permute(0, 3, 1, 2)
        if self.pos:
            H = P.shape[-1] - 2 * (K // 2)
            if self._posp is None or self._posp.shape[-1] != H:
                self._posp = position_planes(H, H, P.device)
            pos = self._posp[:, yy, xx].T
        out = []
        for k in self.sizes:
            o = (K - k) // 2
            f = win[:, :, o:o + k, o:o + k].flatten(1)
            out.append((torch.cat([f, pos], 1) if self.pos else f)[:, :, None, None])
        return out

    def update(self, xs, y):
        """One commit; returns the pre-update score of the team (learning semantics)."""
        if self.joint:
            return self.team.update(xs, y)
        return sum(m.update(x, y) for m, x in zip(self.members, xs))

    @torch.no_grad()
    def library_votes(self, xs):
        """The team's score through the members' own ``forward`` (prediction mode). Slow: tests only."""
        return sum(m(x).float() for m, x in zip(self.members, xs))

    @torch.no_grad()
    def prepare(self):
        """Read the clauses once for ``votes`` (call again after any update)."""
        self._prep = []
        for m in self.members:
            inc = m.included_mask()
            F = inc.shape[1] // 2
            pos, neg = inc[:, :F].float(), inc[:, F:].float()
            self._prep.append(((pos - neg).T.contiguous(), pos.sum(1), m.include_count > 0,
                               m._signed_weights().float().view(m.n_classes, m.n_clauses)))

    @torch.no_grad()
    def votes(self, xs):
        """The same score as ``library_votes`` from one matrix product per group: a clause fires when none
        of its included literals is false (and it is not empty). ``forward`` of a member costs about
        35 ms per call whatever its size, which is minutes per split at 256 px."""
        out = 0.0
        for (D, npos, nonempty, w), x in zip(self._prep, xs):
            failed = npos.unsqueeze(0) - x[:, :, 0, 0].float() @ D           # false positive literals + true negated ones
            fires = ((failed < 0.5) & nonempty.unsqueeze(0)).float()
            out = out + (fires.view(len(x), *w.shape) * w).sum(2)
        return out

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    def state_dict(self):
        return {f"{i}.{k}": v for i, m in enumerate(self.members) for k, v in m.state_dict().items()}

    def load_state_dict(self, state):
        for i, m in enumerate(self.members):
            m.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(f"{i}.")})

    def info(self):
        cs = [m.include_count.float() for m in self.members]
        allc = torch.cat(cs)
        out = {"n_features": max(int(m.ta_state.shape[1] // 2) for m in self.members),
               "clauses_total": int(allc.numel()), "literals_per_clause": float(allc.mean()),
               "empty_clauses": int((allc == 0).sum()), "included_literals": int(allc.sum())}
        for k, c in zip(self.sizes, cs):
            out[f"literals_per_clause_k{k}"], out[f"empty_clauses_k{k}"] = float(c.mean()), int((c == 0).sum())
        return out


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
        elif self.kind == "multi":
            self.m = MultiSize(job, lambda c: max(1, round(job["t_ratio"] * c)), self.dev)
        elif self.kind in ("win2d", "cell"):
            self.m = tt.ConvTsetlinMachine(2, n, T=T, s=10.0, patch_size=k, position_encoding=True, **kw)
        elif self.kind == "win3d":
            self.m = tt.Conv3dTsetlinMachine(2, n, T=T, s=10.0, patch_size=(job["kd"], k, k), position_encoding=True, **kw)
        else:
            raise ValueError(self.kind)
        if self.kind != "multi":
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

    # ---------------------------------------------------------------- cells (phase coarse)
    def _cell_padded(self, x, key):
        """Encoded planes ``(N, Z, H + 2p, W + 2p)`` with a zero border, so that the window of the cell whose
        top-left pixel is (y, x) starts at (y, x). Cached per split."""
        if key not in self._pad or self._pad[key].shape[0] != x.shape[0]:
            p = (self.job["win"] - self.job["csize"]) // 2
            self._pad[key] = torch.nn.functional.pad(planes(x, self.enc), (p, p, p, p))
        return self._pad[key]

    def _cell_windows(self, P, b, yy, xx):
        """``(M, Z [+ 14], w, w)`` windows of the cells with top-left pixels ``(yy, xx)`` of images ``b``."""
        w, c = self.job["win"], self.job["csize"]
        ar = torch.arange(w, device=P.device)
        win = P[b[:, None, None], :, (yy[:, None] + ar)[:, :, None], (xx[:, None] + ar)[:, None, :]].permute(0, 3, 1, 2)
        if self.job["pos"]:                              # where the cell is in the image, the same in every pixel
            thr = torch.arange(1, 8, device=P.device) * (P.shape[-1] - w + c) / 8.0
            pos = torch.cat([(yy[:, None] + c / 2) > thr, (xx[:, None] + c / 2) > thr], 1)
            win = torch.cat([win, pos[:, :, None, None].expand(-1, -1, w, w)], 1)
        return win

    def cell_labels(self, y):
        """``(N, H, W)`` masks -> ``(N, H / c, W / c)``: a cell is polyp when at least half of its pixels are."""
        return (torch.nn.functional.avg_pool2d(y.float().unsqueeze(1), self.job["csize"]).squeeze(1) >= 0.5).long()

    def oracle(self, y):
        """The true cell map enlarged to the image: the best a model of this cell size can do."""
        c = self.job["csize"]
        return self.cell_labels(y).repeat_interleave(c, 1).repeat_interleave(c, 2)

    @torch.no_grad()
    def _cell_epoch(self, x, y):
        """As many cells per image as the grid has, at random offsets, mixed over all images."""
        c, conf, N, (H, W) = self.job["csize"], Confusion(), x.shape[0], y.shape[-2:]
        P = self._cell_padded(x, "train")
        share = torch.nn.functional.avg_pool2d(y.float().unsqueeze(1), c, stride=1).squeeze(1)       # every offset
        M = N * (H // c) * (W // c)
        b = torch.randint(0, N, (M,), device=self.dev)
        yy = torch.randint(0, H - c + 1, (M,), device=self.dev)
        xx = torch.randint(0, W - c + 1, (M,), device=self.dev)
        lab = (share[b, yy, xx] >= 0.5).long()
        for i in range(0, M, BATCH_WINDOWS):
            j = slice(i, i + BATCH_WINDOWS)
            v = self.m.update(self._cell_windows(P, b[j], yy[j], xx[j]), lab[j])
            conf.add(v.argmax(1), lab[j])
        return conf

    # ---------------------------------------------------------------- multi-size groups (phase multi)
    def _multi_padded(self, x, key):
        if key not in self._pad or self._pad[key].shape[0] != x.shape[0]:
            self._pad[key] = torch.nn.functional.pad(planes(x, self.enc), (self.m.kmax // 2,) * 4)
        return self._pad[key]

    @torch.no_grad()
    def _multi_epoch(self, x, y):
        """Sampled pixels of all images (as the pool models), mixed, one joint commit per ``COMMIT`` pixels."""
        job, conf, N, W = self.job, Confusion(), x.shape[0], y.shape[-1]
        P = self._multi_padded(x, "train")
        idx = torch.cat([sample_pixels(y[i:i + POOL_IMAGES], job["pool_n"], job["pool_a"], job["pool_b"])
                         for i in range(0, N, POOL_IMAGES)])                                  # (N, n)
        b = torch.arange(N, device=self.dev).repeat_interleave(idx.shape[1])
        order = torch.randperm(b.numel(), device=self.dev)
        b, pix = b[order], idx.flatten()[order]
        yy, xx = pix // W, pix % W
        lab = y[b, yy, xx]
        for i in range(0, b.numel() - COMMIT + 1, COMMIT):                                    # full commits only
            j = slice(i, i + COMMIT)
            v = self.m.update(self.m.inputs(P, b[j], yy[j], xx[j]), lab[j])
            conf.add(v.argmax(1), lab[j])
        return conf

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
        if self.kind == "cell":
            return self._cell_epoch(x, y)
        if self.kind == "multi":
            return self._multi_epoch(x, y)
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
        elif self.kind == "multi":
            P = self._multi_padded(x, key) if key else torch.nn.functional.pad(planes(x, self.enc), (self.m.kmax // 2,) * 4)
            g = torch.arange(H * W, device=self.dev)
            self.m.prepare()
            for i in range(N):
                b = torch.full((MULTI_CHUNK,), i, device=self.dev, dtype=torch.long)
                v = torch.cat([self.m.votes(self.m.inputs(P, b[:c.numel()], c // W, c % W)) for c in g.split(MULTI_CHUNK)])
                out.append(v.view(1, H, W, 2).permute(0, 3, 1, 2))
        elif self.kind == "cell":
            c = self.job["csize"]
            P = self._cell_padded(x, key) if key else torch.nn.functional.pad(planes(x, self.enc), ((self.job["win"] - c) // 2,) * 4)
            gy, gx = torch.meshgrid(torch.arange(0, H, c, device=self.dev), torch.arange(0, W, c, device=self.dev), indexing="ij")
            gy, gx = gy.flatten(), gx.flatten()
            for i in range(0, N, step):
                n = min(step, N - i)
                b = torch.arange(i, i + n, device=self.dev).repeat_interleave(gy.numel())
                v = self.m(self._cell_windows(P, b, gy.repeat(n), gx.repeat(n))).float()
                v = v.view(n, H // c, W // c, 2).permute(0, 3, 1, 2)
                out.append(v.repeat_interleave(c, 2).repeat_interleave(c, 3))
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
        m0 = self.m.members[0] if self.kind == "multi" else self.m
        if getattr(m0, "ta_state", None) is None or m0.ta_state.numel() == 0:
            if self.kind == "multi":                                         # lazy shapes before loading
                z = torch.zeros(1, dtype=torch.long, device=self.dev)
                P = torch.nn.functional.pad(planes(x[:1], self.enc), (self.m.kmax // 2,) * 4)
                self.m.library_votes(self.m.inputs(P, z, z, z))
            else:
                self.votes(x[:1])
        self.m.load_state_dict(state)

    def info(self):
        if self.kind == "multi":
            return self.m.info()
        c = self.m.include_count.float()
        return {"n_features": int(self.m.ta_state.shape[1] // 2), "clauses_total": int(self.m.ta_state.shape[0]),
                "literals_per_clause": float(c.mean()), "empty_clauses": int((c == 0).sum()),
                "included_literals": int(c.sum())}
