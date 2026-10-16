"""C1 and spec test 37 (label-shuffle leakage control)."""
import numpy as np
import torch

from experiments.transformer_like.code.common import src_diff
from experiments.transformer_like.code.features import transform
from experiments.transformer_like.code.tm_adapter import TMAdapter
from experiments.transformer_like.code.training import fit_classifier


def test_library_untouched():
    assert src_diff() == "", "C1: src/torchtsetlin must stay read-only in this programme"


def test_label_shuffle_control(smoke, tiny_encoders):  # spec 37
    q, k, _ = tiny_encoders
    tr, va = smoke
    Xtr = transform(tr.images, "qk", q, k, 1.0)
    Xva = transform(va.images, "qk", q, k, 1.0)
    y = np.random.default_rng(0).permutation(tr.y)
    torch.manual_seed(0)
    head = TMAdapter(3424, 20, 10, 5.0)
    fit = fit_classifier(head, Xtr, torch.as_tensor(y), Xva, torch.as_tensor(va.y), max_epochs=3,
                         patience=10, log=lambda s: None)
    assert fit["best_val_accuracy"] < 0.25
