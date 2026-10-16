"""Spec tests 27, 30, 31, 36 and the encoder sampler."""
import numpy as np
import torch

from experiments.transformer_like.code.common import stream
from experiments.transformer_like.code.features import transform
from experiments.transformer_like.code.models import TMQKAttention
from experiments.transformer_like.code.patch_training import sample_patches
from experiments.transformer_like.code.tm_adapter import TMAdapter
from experiments.transformer_like.code.tokenization import encode_images
from experiments.transformer_like.code.training import fit_classifier


def _pipeline(smoke, enc, mode="qk", M=20, epochs=2):
    q, k, _ = enc
    tr, va = smoke
    Xtr = transform(tr.images, mode, q, k, 1.0)
    Xva = transform(va.images, mode, q, k, 1.0)
    torch.manual_seed(0)
    head = TMAdapter(Xtr.shape[1], M, M // 2, 5.0)
    fit = fit_classifier(head, Xtr, torch.as_tensor(tr.y), Xva, torch.as_tensor(va.y), max_epochs=epochs,
                         patience=10, log=lambda s: None)
    return TMQKAttention(head, mode, q, k, 1.0), fit


def test_encoders_frozen_during_head_training(smoke, tiny_encoders):  # spec 27
    q, k, _ = tiny_encoders
    hq, hk = q.state_hash(), k.state_hash()
    model, _ = _pipeline(smoke, tiny_encoders)
    model.predict_images(torch.as_tensor(smoke[1].images[:20]))
    assert q.state_hash() == hq and k.state_hash() == hk


def test_extreme_images(smoke, tiny_encoders):  # spec 30
    model, _ = _pipeline(smoke, tiny_encoders, epochs=1)
    u8 = torch.stack([torch.zeros(28, 28, dtype=torch.uint8), torch.full((28, 28), 255, dtype=torch.uint8)])
    v = model.votes_images(u8)
    assert torch.isfinite(v).all() and model.predict_images(u8).shape == (2,)


def test_pipeline_save_load(smoke, tiny_encoders, tmp_path):  # spec 31
    model, _ = _pipeline(smoke, tiny_encoders, epochs=1)
    model.save(tmp_path / "m.pt")
    m2 = TMQKAttention.load(tmp_path / "m.pt")
    u8 = torch.as_tensor(smoke[1].images)
    assert torch.equal(model.votes_images(u8), m2.votes_images(u8))
    for mode in ("uniform", "self_only"):
        mm = TMQKAttention(model.head, mode)
        mm.save(tmp_path / f"{mode}.pt")
        assert TMQKAttention.load(tmp_path / f"{mode}.pt").q is None


def test_earliest_best_checkpoint_restored(smoke, tiny_encoders):
    model, fit = _pipeline(smoke, tiny_encoders, epochs=3)
    accs = [h["val_accuracy"] for h in fit["history"]]
    assert fit["best_epoch"] == int(np.argmax(accs)) + 1  # argmax = earliest maximum
    va = smoke[1]
    acc = float((model.predict_images(torch.as_tensor(va.images)) == torch.as_tensor(va.y)).double().mean())
    assert acc == fit["best_val_accuracy"]


def test_overfit_100_images(smoke):  # spec 36 (diagnostic, seeded)
    tr, _ = smoke
    X = torch.as_tensor(tr.images[:100]) >= 128
    X = X.reshape(100, -1)
    y = torch.as_tensor(tr.y[:100])
    torch.manual_seed(0)
    head = TMAdapter(784, 200, 20, 5.0)
    for _ in range(30):
        head.fit_epoch(X, y, batch_size=10)
    assert (head.predict(X) == y).double().mean() > 0.9


def test_patch_sampler(smoke):
    tr, _ = smoke
    u8 = torch.as_tensor(tr.images[:200])
    _, tokens, active = encode_images(u8)
    slot = int(active[1].nonzero()[0])
    active[0] = False
    active[1] = False
    active[1, slot] = True
    y = torch.as_tensor(tr.y[:200])
    X, Y, st = sample_patches(tokens, active, y, 4, stream(0, "t"))
    assert st["blank_images_skipped"] == 1 and X.shape == (199 * 4, 55)
    assert (X[:4] == tokens[1, slot]).all() and st["repeated_samples"] >= 3
    # every sampled patch is active and comes from its parent image
    assert (X[:, :49].any(1)).all()
    assert torch.equal(Y, y[1:].repeat_interleave(4))
    X2, _, _ = sample_patches(tokens, active, y, 4, stream(0, "t"))
    assert torch.equal(X, X2)
