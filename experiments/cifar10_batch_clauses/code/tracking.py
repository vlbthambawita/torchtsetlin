"""Progress mirror to Weights & Biases (``simulamet_mlc/TM_CIFAR``), plan §6.

One W&B run per training run, fixed id ``cbc-<arm>-<tag>``, ``resume="allow"``, so a run resumed
from its checkpoint continues the same W&B run. Group = the arm (smoke, pilot, grid,
control_T, control_seq). Every epoch logs the curve row against the ``epoch`` axis.

W&B is a mirror, not the record: ``results/<arm>/*.json`` are the source of every reported number.
Any W&B failure is printed once and then ignored, so it can never stop a training run. Disable
with ``--no-wandb`` or ``CBC_WANDB=0``.
"""

import os

from common import CACHE

# The project lives on CoreWeave's W&B (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
ENTITY = os.environ.get("CBC_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("CBC_WANDB_PROJECT", "TM_CIFAR")
SCALARS = ("val_acc", "best_val_acc", "epochs_since_improvement", "test_acc", "train_acc_running",
           "train_s", "val_s", "test_s", "examples_per_s", "commits", "peak_gpu_mib", "mean_clause_len",
           "frac_empty_clauses", "vote_saturation", "elapsed_s")


def run_id(arm, tag):
    return f"cbc-{arm}-{tag}"


class Tracker:
    def __init__(self, arm, tag, config, tags, enabled=True):
        self.run = None
        if not enabled or os.environ.get("CBC_WANDB", "1") == "0":
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(arm, tag), name=tag, group=arm,
                job_type=arm, tags=tags, config=config, resume="allow", dir=str(CACHE),
                settings=wandb.Settings(init_timeout=180, console="off"))
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")
        except Exception as e:                      # noqa: BLE001 - tracking must never break training
            self._fail("init", e)

    def _fail(self, what, e):
        print(f"[wandb] {what} failed, tracking off: {type(e).__name__}: {e}", flush=True)
        self.run = None

    @property
    def url(self):
        return getattr(self.run, "url", None)

    def epoch(self, row):
        if self.run is None:
            return
        try:
            d = {"epoch": row["epoch"], **{k: row[k] for k in SCALARS if k in row}}
            self.run.log(d)
        except Exception as e:                      # noqa: BLE001
            self._fail("log", e)

    def end(self, summary):
        if self.run is None:
            return
        try:
            self.run.summary.update({k: v for k, v in summary.items() if v is not None})
            self.run.finish(exit_code=0 if summary.get("status") == "done" else 143)
        except Exception as e:                      # noqa: BLE001
            self._fail("finish", e)
