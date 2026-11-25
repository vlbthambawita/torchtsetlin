"""W&B tracking and prediction panels (PLAN_KVASIR §9). Tracking never breaks training (C10):
every call is wrapped, and on failure the run goes on with its JSON record only.

Project ``TM_KVASIR`` (``KV_WANDB_PROJECT``), entity ``simulamet_mlc`` (``KV_WANDB_ENTITY``).
``KV_WANDB=offline`` writes local runs for ``wandb sync``; ``KV_WANDB=0`` turns tracking off.

A panel shows, for the fixed images of ``manifests/panel.json``, the image, the real mask and the
prediction (a mask, or boxes for the heads that predict boxes) with an error map: hits green,
false alarms red, misses blue. Panels are also written to ``figures/panels/<tag>/``.
"""

import os

import numpy as np
from common import CACHE, FIGURES, IGNORE, S

ENTITY = os.environ.get("KV_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("KV_WANDB_PROJECT", "TM_KVASIR")
MODE = os.environ.get("KV_WANDB", "1")
LABELS = {0: "background", 1: "polyp"}


def run_id(tag):
    return f"kv-{tag}".replace("+", "_").replace(".", "_")[:120]


class Tracker:
    def __init__(self, job, config, enabled=True):
        self.run, self.url = None, None
        if not enabled or MODE == "0":
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(job["tag"]), name=job["tag"],
                group=job.get("group") or job.get("head") or job.get("kind"), job_type=job["phase"],
                tags=[t for t in (job.get("head"), job.get("stage"), job.get("aug"), job.get("combine")) if t],
                config=config, resume="allow", dir=str(CACHE), mode="offline" if MODE == "offline" else "online",
                settings=wandb.Settings(init_timeout=180, console="off"))
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")
            self.url = getattr(self.run, "url", None)
        except Exception as e:                       # noqa: BLE001
            self._fail("init", e)

    def _fail(self, what, e):
        print(f"[wandb] {what} failed, tracking off: {type(e).__name__}: {e}", flush=True)
        self.run = None

    def log(self, epoch, scalars):
        if self.run is None:
            return
        try:
            self.run.log({"epoch": epoch, **{k: v for k, v in scalars.items() if isinstance(v, (int, float))}})
        except Exception as e:                       # noqa: BLE001
            self._fail("log", e)

    def panel(self, epoch, split, x, inst, items, grid_path):
        if self.run is None:
            return
        try:
            import wandb
            imgs = [wandb_image(x[i], inst[i], items[i]) for i in range(len(items))]
            self.run.log({"epoch": epoch, f"panel/{split}": imgs,
                          f"panel_grid/{split}": wandb.Image(str(grid_path), caption=f"{split}, epoch {epoch}")})
        except Exception as e:                       # noqa: BLE001
            self._fail("panel", e)

    def summary(self, values):
        if self.run is None:
            return
        try:
            for k, v in values.items():
                self.run.summary[k] = v
        except Exception as e:                       # noqa: BLE001
            self._fail("summary", e)

    def finish(self):
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as e:                       # noqa: BLE001
            self._fail("finish", e)


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _box_data(boxes, class_id, scores=None):
    out = []
    for j, b in enumerate(_np(boxes).reshape(-1, 4)):
        d = {"position": {"minX": float(b[0]), "minY": float(b[1]), "maxX": float(b[2]), "maxY": float(b[3])},
             "domain": "pixel", "class_id": class_id}
        if scores is not None and j < len(scores):
            d["scores"] = {"margin": float(_np(scores)[j])}
        out.append(d)
    return out


def gt_boxes_of(inst):
    inst = _np(inst)
    out = []
    for j in range(1, int(np.where(inst == IGNORE, 0, inst).max()) + 1):
        ys, xs = np.nonzero(inst == j)
        if len(ys):
            out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(out, np.float32).reshape(-1, 4)


def wandb_image(x, inst, item):
    """One image with the real mask and the prediction as overlays that can be switched."""
    import wandb
    img = _np(x).transpose(1, 2, 0)
    inst = _np(inst)
    masks = {"real mask": {"mask_data": ((inst > 0) & (inst != IGNORE)).astype(np.uint8), "class_labels": LABELS}}
    boxes = {}
    if item.get("pred_mask") is not None:
        masks["predicted mask"] = {"mask_data": _np(item["pred_mask"]).astype(np.uint8), "class_labels": LABELS}
    if item.get("pred_boxes") is not None:
        boxes["predicted boxes"] = {"box_data": _box_data(item["pred_boxes"], 1, item.get("scores")), "class_labels": LABELS}
        boxes["real boxes"] = {"box_data": _box_data(gt_boxes_of(inst), 1), "class_labels": LABELS}
    return wandb.Image(img, masks=masks, boxes=boxes or None)


def render(x, inst, items, title, path):
    """The grid figure: one row per image; columns image, real mask, prediction, error map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    n = len(items)
    has_mask = any(it.get("pred_mask") is not None for it in items)
    cols = ["image", "real mask", "predicted mask" if has_mask else "predicted boxes"] + (["error map"] if has_mask else [])
    fig, axes = plt.subplots(n, len(cols), figsize=(1.7 * len(cols), 1.7 * n + 0.5), squeeze=False)
    for i, it in enumerate(items):
        img, lab = _np(x[i]).transpose(1, 2, 0), _np(inst[i])
        real = (lab > 0) & (lab != IGNORE)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(real, cmap="gray", vmin=0, vmax=1)
        ax = axes[i, 2]
        if has_mask:
            pred = _np(it["pred_mask"]).astype(bool) if it.get("pred_mask") is not None else np.zeros((S, S), bool)
            ax.imshow(pred, cmap="gray", vmin=0, vmax=1)
            err = np.zeros((S, S, 3), np.uint8)
            err[pred & real] = (60, 180, 75)
            err[pred & ~real] = (220, 50, 50)
            err[~pred & real] = (60, 110, 230)
            err[lab == IGNORE] = (70, 70, 70)
            axes[i, 3].imshow(err)
        else:
            ax.imshow(img)
            for b in gt_boxes_of(lab):
                ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="lime", lw=1.2))
        for key, colour, style in (("ref_boxes", "yellow", ":"), ("pred_boxes", "red", "-")):
            if it.get(key) is not None:
                for b in _np(it[key]).reshape(-1, 4):
                    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=colour, lw=1.0, ls=style))
        for a in axes[i]:
            a.set_xticks([])
            a.set_yticks([])
            a.set_xlim(0, S)
            a.set_ylim(S, 0)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=8)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def panel_path(tag, split, epoch):
    name = "final" if epoch is None else f"epoch_{int(epoch):04d}"
    return FIGURES / "panels" / tag / f"{split}_{name}.png"
