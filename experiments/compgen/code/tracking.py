"""Progress mirror to Weights & Biases (``simulamet_mlc/TM_COMPGEN``).

One W&B run per training run, fixed id ``cg-<phase>-<tag>``, ``resume="allow"``, group ``cg-<arm>``.
W&B is a mirror, not the record: ``results/<phase>/*.json`` hold every reported number. A W&B
failure is printed once and then ignored. Disable with ``--no-wandb`` or ``CG_WANDB=0``.
"""

import os

from common import CACHE

os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
ENTITY = os.environ.get("CG_WANDB_ENTITY", "simulamet_mlc")
PROJECT = os.environ.get("CG_WANDB_PROJECT", "TM_COMPGEN")


class Tracker:
    def __init__(self, phase, arm, tag, config, tags, enabled=True):
        self.run = None
        if not enabled or os.environ.get("CG_WANDB", "1") == "0" or phase in ("smoke", "price", "resume_check"):
            return
        try:
            import wandb
            CACHE.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(entity=ENTITY, project=PROJECT, id=f"cg-{phase}-{tag}", name=f"cg-{tag}",
                                  group=f"cg-{arm}", job_type=phase, tags=["compgen", *tags], config=config,
                                  resume="allow", dir=str(CACHE),
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
            self.run.log({k: v for k, v in row.items() if isinstance(v, (int, float))})
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
