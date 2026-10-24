"""Common adapter interface. An adapter turns a frozen cell config (configs.py) into calls of
one library's own public API, the way that library's demos call it."""
import pickle
from pathlib import Path

import numpy as np


class Adapter:
    #: library-specific per-run seed derived from run index k (config_map a.11)
    def __init__(self, cfg, threads=1, device="cpu", seed=0):
        self.cfg = cfg
        self.lib_cfg = cfg["lib_cfg"]
        self.threads = threads
        self.device = device
        self.seed = seed
        self.model = None

    def version(self):
        return None

    def build(self, x, y, n_classes):          # construct the model (timed as build_s)
        raise NotImplementedError

    def fit_epoch(self, x, y, epoch):           # exactly one training epoch
        raise NotImplementedError

    def predict(self, x):                       # labels (classification) or values (regression)
        raise NotImplementedError

    def sync(self):
        pass

    def extra_memory(self):
        return {}

    def state_nbytes(self):
        """Bytes of the learned state in the library's own storage format."""
        return None

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def model_size(self, path):
        out = {"state_bytes": self.state_nbytes()}
        try:
            self.save(path)
            out["saved_file_bytes"] = Path(path).stat().st_size
        except Exception as e:                  # e.g. a library that cannot pickle its model
            out["save_error"] = repr(e)
        return out


def ndarray_bytes(obj, names, depth=0, seen=None):
    """Sum nbytes of numpy arrays stored under attribute / key `names`, walking objects."""
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 5:
        return 0
    seen.add(id(obj))
    tot = 0
    items = (obj.items() if isinstance(obj, dict)
             else enumerate(obj) if isinstance(obj, (list, tuple))
             else vars(obj).items() if hasattr(obj, "__dict__") else ())
    for k, v in items:
        if isinstance(v, np.ndarray):
            if k in names:
                tot += v.nbytes
        elif isinstance(v, (dict, list, tuple)) or hasattr(v, "__dict__"):
            tot += ndarray_bytes(v, names, depth + 1, seen)
    return tot
