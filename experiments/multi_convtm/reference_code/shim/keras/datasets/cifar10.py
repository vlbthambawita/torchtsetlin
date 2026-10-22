"""Stand-in for keras.datasets.cifar10 so the authors' scripts run without TensorFlow.

Returns exactly what keras returns: ((x_train, y_train), (x_test, y_test)) with uint8 images of
shape (N, 32, 32, 3) in RGB and uint8 labels of shape (N, 1), read from the official
python-pickle release in data/cifar-10-batches-py (tarball md5 c58f30108f718f92721af3b95e74349a).
"""
import os
import pickle

import numpy as np

_ROOT = os.environ.get("CIFAR10_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..",
                                                    "..", "data", "cifar-10-batches-py"))


def _batch(name):
    with open(os.path.join(_ROOT, name), "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    x = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return np.ascontiguousarray(x), np.asarray(d[b"labels"], dtype=np.uint8).reshape(-1, 1)


def load_data():
    parts = [_batch(f"data_batch_{i}") for i in range(1, 6)]
    xtr = np.concatenate([p[0] for p in parts])
    ytr = np.concatenate([p[1] for p in parts])
    return (xtr, ytr), _batch("test_batch")
