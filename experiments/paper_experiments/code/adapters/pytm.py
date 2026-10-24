"""pyTsetlinMachine 0.6.6 (single-threaded C). Also used for pyTsetlinMachineParallel."""
import importlib

import numpy as np

from .base import Adapter as _Base

MODULE = "pyTsetlinMachine.tm"


class Adapter(_Base):
    module = MODULE

    def version(self):
        from importlib.metadata import version
        return version(self.module.split(".")[0])

    def build(self, x, y, n_classes):
        mod = importlib.import_module(self.module)
        self.model = getattr(mod, self.lib_cfg["cls"])(**self.lib_cfg["kwargs"])
        self.regression = self.cfg["task"] == "regression"

    def _y(self, y):
        return y.astype(np.float64) if self.regression else y.astype(np.uint32)

    def fit_epoch(self, x, y, epoch):
        # the README pattern; the first call with incremental=True creates the machine
        self.model.fit(x.astype(np.uint32), self._y(y), epochs=1, incremental=True)

    def predict(self, x):
        return self.model.predict(x.astype(np.uint32))

    def state_nbytes(self):
        try:
            st = self.model.get_state()
        except Exception:
            return None
        flat = st if isinstance(st, (list, tuple)) else [st]
        tot = 0
        for item in flat:
            for a in (item if isinstance(item, (list, tuple)) else [item]):
                if isinstance(a, np.ndarray):
                    tot += a.nbytes
        return tot
