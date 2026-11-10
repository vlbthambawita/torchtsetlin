"""Progress mirror to Weights & Biases (``simulamet_mlc/torchtsetlin``), plan §5.

One W&B run per training run, fixed id ``mbc-<tag>`` (smoke: ``mbc-smoke-<tag>``), ``resume="allow"``,
so a run resumed from its checkpoint continues the same W&B run. Group ``mnist_batch_clauses``
(smoke runs: ``mnist_batch_clauses_smoke``), ``job_type`` = the arm. Every epoch logs the curve row
against the ``epoch`` axis (``commits`` is logged too, as an alternative x-axis).

W&B is a mirror, not the record: ``results/<arm>/*.json`` are the source of every reported number.
Any W&B failure is printed once and then ignored, so it can never stop a training run. Disable
with ``--no-wandb`` or ``MBC_WANDB=0``; change the target with ``MBC_WANDB_ENTITY`` /
``MBC_WANDB_PROJECT``.
"""

import os

from common import CACHE

ENTITY = os.environ.get("MBC_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("MBC_WANDB_PROJECT", "torchtsetlin")
GROUP = "mnist_batch_clauses"
METRICS = ("val_acc", "best_val_acc", "epochs_since_improvement", "test_acc", "train_acc_running",
           "train_s", "val_s", "test_s", "examples_per_s", "commits", "peak_gpu_mib",
           "mean_clause_len", "frac_empty_clauses", "vote_saturation", "elapsed_s")


def run_id(arm, tag):
    return f"mbc-{'smoke-' if arm == 'smoke' else ''}{tag}"


class Tracker:
    def __init__(self, arm, tag, config, tags, enabled=True):
        self.run = None
        if not enabled or os.environ.get("MBC_WANDB", "1") == "0":
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(arm, tag), name=tag,
                group=GROUP + ("_smoke" if arm == "smoke" else ""), job_type=arm, tags=tags,
                config=config, resume="allow", dir=str(CACHE),
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
            self.run.log({"epoch": row["epoch"], **{k: row[k] for k in METRICS if k in row}})
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
