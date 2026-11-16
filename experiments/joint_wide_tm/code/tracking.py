"""Progress mirror to Weights & Biases (``simulamet_mlc/TM_fMNIST``, shared with fmnist_bitplanes).

One W&B run per training run, fixed id ``jw-<phase>-<arm>-<tag>``, ``resume="allow"``; group
``jw-<arm>``, tag ``joint_wide`` (filters this programme apart from the ``fm-`` runs there).
W&B is a mirror, not the record. Any failure is printed once and ignored. Disable with
``--no-wandb`` or ``JW_WANDB=0``.
"""

import os

from common import CACHE

ENTITY = os.environ.get("JW_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("JW_WANDB_PROJECT", "TM_fMNIST")
SCALARS = ("val_acc", "best_val_acc", "test_acc", "train_acc_running", "train_s", "eval_s",
           "examples_per_s", "peak_gpu_mib", "elapsed_s")


def run_id(phase, arm, tag):
    return f"jw-{phase}-{arm}-{tag}"


class Tracker:
    def __init__(self, phase, arm, tag, config, tags, enabled=True):
        self.run = None
        if not enabled or os.environ.get("JW_WANDB", "1") == "0":
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=ENTITY, project=PROJECT, id=run_id(phase, arm, tag), name=f"jw-{arm}-{tag}",
                group=f"jw-{arm}", job_type=phase, tags=["joint_wide"] + tags, config=config,
                resume="allow", dir=str(CACHE), settings=wandb.Settings(init_timeout=180, console="off"))
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
            for name, m in row.get("members", {}).items():
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        d[f"member/{name}/{k}"] = v
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
