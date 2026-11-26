"""W&B tracking and prediction panels (plan section 8). Tracking never breaks training: every call
is wrapped, and on failure the run goes on with its JSON record only.

Project ``TM_SEG_SIMPLE`` (``SS_WANDB_PROJECT``), entity ``simulamet_mlc`` (``SS_WANDB_ENTITY``).
``SS_WANDB=offline`` writes local runs for ``wandb sync``; ``SS_WANDB=0`` turns tracking off.

A panel shows the fixed validation images in three rows: image, true mask, predicted mask. In the
last row hits are green, false alarms red and misses blue. Panels are also written to
``figures/panels/<tag>/``.
"""

import os

import numpy as np
from common import CACHE, FIGURES

ENTITY = os.environ.get("SS_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("SS_WANDB_PROJECT", "TM_SEG_SIMPLE")
MODE = os.environ.get("SS_WANDB", "1")


def run_id(tag):
    return f"ss-{tag}".replace("+", "_").replace(".", "_")[:120]


class Tracker:
    def __init__(self, job, config, enabled=True):
        self.run, self.url = None, None
        if not enabled or MODE == "0":
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(job["tag"]), name=job["tag"], group=job["group"],
                job_type=job["phase"], tags=[str(t) for t in (job.get("model"), job.get("enc"), job.get("aug")) if t],
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

    def image(self, epoch, key, path, caption):
        if self.run is None:
            return
        try:
            import wandb
            self.run.log({"epoch": epoch, key: wandb.Image(str(path), caption=caption)})
        except Exception as e:                       # noqa: BLE001
            self._fail("image", e)

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


def panel_path(tag, epoch):
    return FIGURES / "panels" / tag / f"epoch_{epoch:04d}.png"


def render(x, y, pred, path, title=""):
    """``x`` (N, 3, H, W) uint8, ``y`` and ``pred`` (N, H, W) in {0, 1}: one column per image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, y, pred = (np.asarray(t.detach().cpu()) for t in (x, y, pred))
    n = len(x)
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5.2), squeeze=False)
    for i in range(n):
        err = np.zeros((*y[i].shape, 3), np.float32)
        err[(pred[i] == 1) & (y[i] == 1)] = (0.2, 0.8, 0.3)
        err[(pred[i] == 1) & (y[i] == 0)] = (0.9, 0.2, 0.2)
        err[(pred[i] == 0) & (y[i] == 1)] = (0.2, 0.4, 0.95)
        for r, im in enumerate((x[i].transpose(1, 2, 0), y[i], err)):
            ax[r, i].imshow(im, cmap="gray", vmin=0, vmax=1 if r == 1 else None, interpolation="nearest")
            ax[r, i].set_xticks([])
            ax[r, i].set_yticks([])
    for r, name in enumerate(("image", "true mask", "prediction")):
        ax[r, 0].set_ylabel(name)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
