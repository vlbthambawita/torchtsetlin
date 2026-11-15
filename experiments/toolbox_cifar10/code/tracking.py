"""Progress reporting to Weights & Biases (``simulamet_mlc/torchtsetlin``).

One W&B run per training run, with a fixed id derived from the tag (``tb-<tag>``, ``/`` -> ``-``), so a
run resumed from its checkpoint continues the same W&B run (``resume="allow"``). Group
``toolbox_cifar10``; config = the record's hyper-parameters plus ``spec`` and ``tag``. Every epoch logs
``test_acc``, ``train_acc_running``, ``train_s``, ``test_s``, ``peak_gib`` against the ``epoch`` axis.
At the end the summary gets the last-25 statistics and ``status``; an interrupted run finishes with
exit code 143 (shown as failed/crashed). Epochs trained before tracking existed are loaded once with
``code/wandb_backfill.py``.

W&B is a mirror, not the record: ``results/*.json`` stay the source of every reported number
(README.md, R2). Any W&B failure is printed once and then ignored, so it can never stop a training
run. Disable with ``--no-wandb`` or ``TB_WANDB=0``; change the target with ``TB_WANDB_ENTITY`` /
``TB_WANDB_PROJECT``.
"""

import os

from common import CACHE

ENTITY = os.environ.get("TB_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("TB_WANDB_PROJECT", "torchtsetlin")
GROUP = "toolbox_cifar10"
METRICS = ("test_acc", "train_acc_running", "train_s", "test_s", "peak_gib")


def run_id(tag):
    return "tb-" + tag.replace("/", "-")


class Tracker:
    def __init__(self, tag, spec, hp, enabled=True, env=None):
        self.run = None
        if not enabled or os.environ.get("TB_WANDB", "1") == "0":
            return
        try:
            import wandb
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(tag), name=tag, group=GROUP,
                job_type="diagnostic" if tag.startswith("diagnostics/") else "specialist",
                tags=[spec, f"c{hp['clauses_per_class']}", f"b{hp['batch_size']}"],
                config=dict(hp, spec=spec, tag=tag, env=env or {}), resume="allow",
                dir=str(CACHE), settings=wandb.Settings(init_timeout=180, console="off"))
            self.run.define_metric("epoch")
            self.run.define_metric("*", step_metric="epoch")
        except Exception as e:                      # noqa: BLE001 - tracking must never break training
            self._fail("init", e)

    def _fail(self, what, e):
        print(f"[wandb] {what} failed, tracking off: {type(e).__name__}: {e}", flush=True)
        self.run = None

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
