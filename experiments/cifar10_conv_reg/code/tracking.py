"""Progress mirror to Weights & Biases (``simulamet_mlc/TM_CIFAR``), plan §9.

One W&B run per training run, fixed id ``ccr-<phase>-<tag>``, name ``ccr-<tag>``, group
``ccr-<phase>``, ``resume="allow"`` (a run resumed from its checkpoint continues the same W&B
run). Every epoch logs the curve row against the ``epoch`` axis; every 20 epochs the three
figures (``viz/panel``, ``viz/health``, ``viz/curves``).

W&B is a mirror, not the record: ``results/<phase>/*.json`` are the source of every reported
number. Any W&B failure is printed once and then ignored, so it can never stop a training run.
Disable with ``--no-wandb`` or ``CCR_WANDB=0``.
"""

import os

from common import CACHE

# The project lives on CoreWeave's W&B (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
ENTITY = os.environ.get("CCR_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("CCR_WANDB_PROJECT", "TM_CIFAR")
SCALARS = ("val_acc", "best_val_acc", "epochs_since_improvement", "test_acc", "train_acc_running",
           "train_eval_acc", "gap", "train_s", "val_s", "test_s", "train_eval_s", "aug_wait_s",
           "examples_per_s", "commits", "peak_gpu_mib", "mean_clause_len", "frac_empty_clauses",
           "vote_saturation", "elapsed_s", "panel_acc", "frac_never_firing", "frac_firing_over_half",
           "frac_top_sum_at_T", "median_margin_over_T", "max_weight", "median_weight")


def run_id(phase, tag):
    return f"ccr-{phase}-{tag}"


class Tracker:
    def __init__(self, phase, tag, config, tags, enabled=True):
        self.run = None
        if not enabled or os.environ.get("CCR_WANDB", "1") == "0":
            return
        try:
            import wandb
            self.wandb = wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(phase, tag), name=f"ccr-{tag}", group=f"ccr-{phase}",
                job_type=phase, tags=["conv_reg", *tags], config=config, resume="allow", dir=str(CACHE),
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

    def epoch(self, row, images=None):
        """``images``: {key: (png path, caption)} logged with this epoch."""
        if self.run is None:
            return
        try:
            d = {"epoch": row["epoch"], **{k: row[k] for k in SCALARS if row.get(k) is not None}}
            for key, (path, caption) in (images or {}).items():
                d[key] = self.wandb.Image(str(path), caption=caption)
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
