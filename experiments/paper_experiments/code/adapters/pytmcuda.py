"""PyTsetlinMachineCUDA 0.1.9 (PyCUDA kernels). One process per run (uninitialised last TA
chunk, config_map a.4)."""
import importlib

import numpy as np

from .base import Adapter as _Base


class Adapter(_Base):
    def version(self):
        from importlib.metadata import version
        return version("PyTsetlinMachineCUDA")

    def build(self, x, y, n_classes):
        # its curand generator is seeded from np.random at import (config_map a.11)
        np.random.seed(self.seed + 1)
        mod = importlib.import_module("PyTsetlinMachineCUDA.tm")
        self.model = getattr(mod, self.lib_cfg["cls"])(**self.lib_cfg["kwargs"])
        self.regression = self.cfg["task"] == "regression"
        self.batch_size = self.lib_cfg.get("batch_size", 100)

    def _y(self, y):
        return y.astype(np.float64) if self.regression else y.astype(np.uint32)

    def fit_epoch(self, x, y, epoch):
        self.model.fit(x.astype(np.uint32), self._y(y), epochs=1, incremental=True,
                       batch_size=self.batch_size)

    def predict(self, x):
        return self.model.predict(x.astype(np.uint32))

    def sync(self):
        import pycuda.driver as cuda
        try:
            cuda.Context.synchronize()
        except Exception:
            pass

    def state_nbytes(self):
        st = self.model.__getstate__()
        return int(st["ta_state"].nbytes + st["clause_weights"].nbytes)
