"""tmu 0.8.3 (C via CFFI on CPU, PyCUDA on GPU). numpy 1.26 venv (config_map a.12)."""
import numpy as np

from .base import Adapter as _Base, ndarray_bytes

CLASSES = {
    "TMClassifier": "tmu.models.classification.vanilla_classifier",
    "TMCoalescedClassifier": "tmu.models.classification.coalesced_classifier",
    "TMRegressor": "tmu.models.regression.vanilla_regressor",
}


class Adapter(_Base):
    def version(self):
        from importlib.metadata import version
        return version("tmu")

    def build(self, x, y, n_classes):
        import importlib
        tmu_seed = 2 * self.seed + 1           # seed=0 hangs in tmu 0.8.3 (config_map a.11)
        np.random.seed(tmu_seed)                # its GPU curand seeds from global np.random
        mod = importlib.import_module(CLASSES[self.lib_cfg["cls"]])
        kw = dict(self.lib_cfg["kwargs"])
        kw["platform"] = "GPU" if self.device.startswith("cuda") else "CPU"
        kw["seed"] = tmu_seed
        self.model = getattr(mod, self.lib_cfg["cls"])(**kw)
        self.regression = self.cfg["task"] == "regression"
        self._checked = False

    def _y(self, y):
        return y.astype(np.float64) if self.regression else y.astype(np.uint32)

    def _check_platform(self):
        # tmu silently falls back to the CPU clause bank if PyCUDA fails (config_map a.12)
        banks = getattr(self.model, "clause_banks", None)
        bank = banks[0] if banks else getattr(self.model, "clause_bank", None)
        name = type(bank).__name__
        if self.device.startswith("cuda") and "CUDA" not in name:
            raise RuntimeError(f"tmu fell back to {name} on a GPU run")
        self.clause_bank_class = name
        self._checked = True

    def fit_epoch(self, x, y, epoch):
        self.model.fit(x.astype(np.uint32), self._y(y))     # one shuffled epoch
        if not self._checked:
            self._check_platform()

    def predict(self, x):
        return self.model.predict(x.astype(np.uint32))

    def sync(self):
        if self.device.startswith("cuda"):
            try:
                import pycuda.driver as cuda
                cuda.Context.synchronize()
            except Exception:
                pass

    def state_nbytes(self):
        return ndarray_bytes(self.model, {"clause_bank", "clause_bank_ind", "weights"})
