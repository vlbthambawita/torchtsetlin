"""torchtsetlin (main env). Data resident on the device; one shuffled pass of update() calls per
epoch; model.eval() before predicting (empty-clause semantics, CLAUDE.md)."""
import numpy as np

from .base import Adapter as _Base


class Adapter(_Base):
    def version(self):
        import torchtsetlin
        return torchtsetlin.__version__

    def build(self, x, y, n_classes):
        import torch
        import torchtsetlin as tt
        self.torch = torch
        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.dev = torch.device(self.device)
        if self.dev.type == "cuda":
            torch.zeros(1, device=self.dev)             # create the context first
            torch.cuda.reset_peak_memory_stats(self.dev)
        self.gen = torch.Generator(device="cpu").manual_seed(self.seed)
        ttc = self.cfg["tt"]
        self.batch_size = ttc["batch_size"]
        kw = dict(self.lib_cfg["kwargs"], feedback_mode=ttc["feedback_mode"])
        cls = self.lib_cfg["cls"]
        model = self.cfg["model"]
        n_features = int(np.prod(x.shape[1:]))
        self.regression = self.cfg["task"] == "regression"
        if cls == "TsetlinMachine":
            m = tt.TsetlinMachine(n_features, n_classes, **kw)
        elif cls == "CoalescedTsetlinMachine":
            m = tt.CoalescedTsetlinMachine(n_features, n_classes, **kw)
        elif cls == "ConvTsetlinMachine":
            m = tt.ConvTsetlinMachine(n_classes, **kw)
        elif cls == "RegressionTsetlinMachine":
            m = tt.RegressionTsetlinMachine(n_features, y_range=(float(y.min()), float(y.max())), **kw)
        else:
            raise KeyError(cls)
        self.model = m.to(self.dev)
        self.conv = model == "conv"
        self._xcache = {}

    def _x(self, x):
        key = id(x)
        if key not in self._xcache:
            t = self.torch.as_tensor(np.ascontiguousarray(x), dtype=self.torch.bool)
            if self.conv:
                t = t.reshape(len(x), 1, *x.shape[1:])
            self._xcache[key] = t.to(self.dev)
        return self._xcache[key]

    def _y(self, y):
        key = ("y", id(y))
        if key not in self._xcache:
            dt = self.torch.float32 if self.regression else self.torch.long
            self._xcache[key] = self.torch.as_tensor(y, dtype=dt).to(self.dev)
        return self._xcache[key]

    def fit_epoch(self, x, y, epoch):
        xt, yt = self._x(x), self._y(y)
        self.model.train()
        perm = self.torch.randperm(len(xt), generator=self.gen).to(self.dev)
        for i in range(0, len(xt), self.batch_size):
            idx = perm[i:i + self.batch_size]
            self.model.update(xt[idx], yt[idx])

    def predict(self, x):
        self.model.eval()
        assert not self.model.training
        with self.torch.no_grad():
            out = []
            xt = self._x(x)
            for i in range(0, len(xt), 4096):
                out.append(self.model.predict(xt[i:i + 4096]))
            p = self.torch.cat(out)
        return p.float().cpu().numpy() if self.regression else p.cpu().numpy()

    def sync(self):
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize(self.dev)

    def extra_memory(self):
        if self.dev.type == "cuda":
            return {"torch_max_allocated_mib": self.torch.cuda.max_memory_allocated(self.dev) / 2 ** 20,
                    "torch_max_reserved_mib": self.torch.cuda.max_memory_reserved(self.dev) / 2 ** 20}
        return {}

    def state_nbytes(self):
        sd = self.model.state_dict()
        return int(sum(v.numel() * v.element_size() for v in sd.values()))

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)
