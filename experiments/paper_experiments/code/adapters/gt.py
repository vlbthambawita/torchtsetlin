"""green-tsetlin 1.1.0. Its Trainer owns the epoch loop, so this adapter implements run_all()
instead of fit_epoch(): one train() call (re-calling it re-seeds every epoch, config_map a.12),
load_best_state=False (otherwise it restores the best-on-test state), early_exit_acc=2.0.
Per-epoch train time is GT's own train_time_of_epochs; test time is the gap from the end of
training to the eval hook, which also contains GT's swap of the eval data into its input block."""
import time

import numpy as np

from .base import Adapter as _Base


class Adapter(_Base):
    def version(self):
        from importlib.metadata import version
        return version("green-tsetlin")

    def build(self, x, y, n_classes):
        import green_tsetlin as gt
        self.gt = gt
        self.model = gt.TsetlinMachine(n_literals=int(np.prod(x.shape[1:])), n_classes=n_classes,
                                       **self.lib_cfg["kwargs"])

    def run_all(self, xtr, ytr, xte, yte, epochs, on_epoch):
        tcfg = self.lib_cfg["trainer"]
        seed = tcfg.get("fixed_seed", self.seed + 1)       # GT seed 0 is not reproducible
        stamps, preds = [], []

        def hook(y_true, y_hat):
            stamps.append(time.perf_counter())
            preds.append(np.array(y_hat))
            return float((np.asarray(y_true) == np.asarray(y_hat)).mean())

        trainer = self.gt.Trainer(self.model, feedback_type=tcfg["feedback_type"],
                                  n_jobs=self.threads, n_epochs=epochs, seed=seed,
                                  early_exit_acc=2.0, load_best_state=False,
                                  progress_bar=False, fn_eval_score=hook)
        trainer.set_train_data(xtr.reshape(len(xtr), -1).astype(np.uint8), ytr.astype(np.uint32))
        trainer.set_eval_data(xte.reshape(len(xte), -1).astype(np.uint8), yte.astype(np.uint32))
        t_start = time.perf_counter()
        res = trainer.train()
        train_t = res["train_time_of_epochs"]
        prev = t_start
        for e, (tr, st, p) in enumerate(zip(train_t, stamps, preds), start=1):
            on_epoch(e, tr, st - prev - tr, p, first_includes_setup=(e == 1))
            prev = st
        self.gt_seed = seed
        return res

    def state_nbytes(self):
        st = self.model._state
        return int(sum(getattr(st, k).nbytes for k in ("c", "w") if hasattr(st, k)))

    def save(self, path):
        self.model.save_state(str(path) + ".npz")
        import os
        os.replace(str(path) + ".npz", path)
