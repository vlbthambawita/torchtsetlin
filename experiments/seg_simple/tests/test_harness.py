"""Tests T1 to T6 of the plan (section 7) and checks of the encodings, windows and job list."""

import shutil

import pytest
import torch

import jobs as JOBS
from common import IGNORE, WINDOW, model_dir, read_json, record_path
from data import Data, augment, cache_path, encode, floors, manifest, planes, sym, sym_inv
from models import Confusion, Net

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
needs_data = pytest.mark.skipif(not cache_path().exists(), reason="data cache missing: run ./setup.sh")


@needs_data
def test_T1_data_splits():
    m, d = manifest(), Data()
    assert [len(d.stems[s]) for s in ("train", "val", "test")] == [700, 150, 150]
    assert d.stems["test"] == m["splits"]["final"]
    names = d.stems["train"] + d.stems["val"] + d.stems["test"]
    assert len(set(names)) == 1000
    for s in d.y:
        assert set(d.y[s].unique().tolist()) <= {0, 1}
        assert d.x[s].shape[1:] == (3, 64, 64) and d.x[s].dtype == torch.uint8


def test_T2_metrics_match_the_library():
    import torchtsetlin as tt
    g = torch.Generator().manual_seed(0)
    pred, tgt = torch.randint(0, 2, (6, 32, 32), generator=g), torch.randint(0, 2, (6, 32, 32), generator=g)
    m = Confusion().add(pred[:3], tgt[:3]).add(pred[3:], tgt[3:]).metrics()
    assert abs(m["mIoU"] - float(tt.metrics.mean_iou(pred, tgt, n_classes=2))) < 1e-6
    assert abs(m["pixel_acc"] - float(tt.metrics.pixel_accuracy(pred, tgt))) < 1e-6
    tgt2 = tgt.clone()
    tgt2[:, :4] = IGNORE
    m2 = Confusion().add(pred, tgt2).metrics()
    assert abs(m2["mIoU"] - float(tt.metrics.mean_iou(pred, tgt2, n_classes=2, ignore_index=IGNORE))) < 1e-6


def _clean(*tags):
    for t in tags:
        shutil.rmtree(model_dir(t), ignore_errors=True)
        record_path(t).unlink(missing_ok=True)


@needs_data
@pytest.mark.parametrize("kw", [dict(n=40, k=3, aug="sym"), dict(model="win3d", n=40, k=3, kd=2, enc="T8")])
def test_T3_resume_gives_the_same_run(kw):
    import train
    whole = JOBS.J("smoke", "t", limit=16, max_epochs=4, prefix="t3whole-", **kw)
    part = JOBS.J("smoke", "t", limit=16, max_epochs=2, prefix="t3part-", **kw)
    _clean(whole["tag"], part["tag"])
    try:
        assert train.run(whole, DEV, wandb_on=False) == 0
        assert train.run(part, DEV, wandb_on=False) == 0
        assert train.run({**part, "max_epochs": 4}, DEV, wandb_on=False) == 0          # resumes after epoch 2
        a, b = read_json(record_path(whole["tag"])), read_json(record_path(part["tag"]))
        assert len(b["history"]) == 4
        assert [r["val_mIoU"] for r in a["history"]] == [r["val_mIoU"] for r in b["history"]]
        assert a["test"]["mIoU"] == b["test"]["mIoU"] and a["model"]["included_literals"] == b["model"]["included_literals"]
    finally:
        _clean(whole["tag"], part["tag"])


@needs_data
def test_T4_all_background_floor():
    d = Data()
    f = floors(d)
    share = float(d.y["val"].float().mean())
    assert abs(f["all_background"]["val"]["mIoU"] - (1 - share) / 2) < 1e-9
    assert f["all_background"]["val"]["IoU_polyp"] == 0.0
    assert f["position_prior_best_threshold"]["val"]["mIoU"] >= f["position_prior"]["val"]["mIoU"]


def test_T5_job_list():
    n = {ph: [j for j in JOBS.jobs(ph)[0] if j["kind"] == "train"] for ph in JOBS.PHASES}
    assert len(n["main"]) == 54 and len(n["side"]) == 113 and len(n["wide"]) == 18 and len(n["conv3d"]) == 69
    tags = [j["tag"] for j in JOBS.all_jobs(JOBS.PHASES)[0]]
    assert len(tags) == len(set(tags))
    cells = {(j["n"], j["k"]) for j in n["main"]}
    assert len(cells) == 18 and set(JOBS.CLAUSE_LINE) | set(JOBS.PATCH_LINE) | set(JOBS.GRID) == cells
    for (m, k) in cells:
        assert sorted(j["seed"] for j in n["main"] if (j["n"], j["k"]) == (m, k)) == [0, 1, 2]
    known = set(tags)
    for j in JOBS.all_jobs(JOBS.PHASES)[0]:
        assert set(j["needs"]) <= known, j["tag"]                      # every team member is a job
    keys = {JOBS.price_key(j) for ph in ("main", "side", "wide") for j in n[ph]}
    assert keys == {JOBS.price_key(j) for j in n["price"]}             # every run has a price
    assert {JOBS.price_key(j) for j in n["conv3d"]} == {JOBS.price_key(j) for j in n["price3d"]}
    c = [JOBS.cost(j) for j in JOBS.jobs("main")[0]]
    assert c == sorted(c)                                              # cheap jobs first


def test_T6_augmentation():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 256, (8, 3, 16, 16), generator=g, dtype=torch.uint8)
    y = (x[:, 0] > 127).long()
    seen = set()
    for k in range(8):
        assert torch.equal(sym_inv(sym(x, k), k), x) and torch.equal(sym_inv(sym(y, k), k), y)
        seen.add(tuple(sym(x[0], k).flatten().tolist()))
    assert len(seen) == 8                                              # eight different symmetries
    ks = []
    for _ in range(100):
        xa, ya, k = augment(x, y, g)
        ks += k
        assert torch.equal(ya, (xa[:, 0] > 127).long())                # image and mask move together
        for i, kk in enumerate(k):
            assert torch.equal(sym_inv(xa[i], kk), x[i])
    assert set(ks) == set(range(8))


def test_encodings():
    import torchtsetlin as tt
    x = torch.arange(256, dtype=torch.uint8).view(1, 1, 16, 16).expand(2, 3, 16, 16).contiguous()
    for enc, bits in (("T4", 4), ("T8", 8)):
        lib = tt.data.ColorThermometerEncoder(n_bits=bits, value_range=(0.0, 1.0))(x.float() / 255.0)
        assert torch.equal(planes(x, enc), lib.bool())
        e = encode(x, enc)
        assert e.shape == (2, 3, bits, 16, 16)
        assert (e[:, :, :-1] >= e[:, :, 1:]).all()                      # a thermometer: plane d+1 implies plane d
    b = encode(x, "B8").long()
    assert torch.equal((b * (2 ** torch.arange(7, -1, -1)).view(1, 1, 8, 1, 1)).sum(2), x.long())


def test_windows_are_centred_and_zero_padded():
    x = torch.randint(0, 256, (3, 3, 12, 12), dtype=torch.uint8, device=DEV)
    for model, kd in (("win2d", None), ("win3d", 2)):
        net = Net(JOBS.J("x", "x", model=model, n=10, k=3, kd=kd, enc="B8"), DEV)
        P = net._padded(x, "t")
        b, yy, xx = torch.tensor([0, 2, 1], device=DEV), torch.tensor([0, 5, 11], device=DEV), torch.tensor([0, 7, 11], device=DEV)
        w = net._windows(P, b, yy, xx).view(3, 3, 8, WINDOW, WINDOW)
        e, c = encode(x, "B8"), WINDOW // 2
        for i in range(3):
            assert torch.equal(w[i, :, :, c, c], e[b[i], :, :, yy[i], xx[i]])
        assert not w[0, :, :, :c, :].any() and not w[2, :, :, c + 1:, :].any()        # outside the image
        assert torch.equal(w[1, :, :, c - 1, c + 2], e[2, :, :, 4, 9])


@needs_data
def test_votes_cover_every_pixel_and_do_not_depend_on_chunking():
    d = Data(DEV, limit=6)
    for kw in (dict(n=20, k=3), dict(model="win2d", n=20, k=3, enc="T8")):
        torch.manual_seed(0)
        net = Net(JOBS.J("x", "x", **kw), DEV)
        net.train_epoch(d.x["train"], d.y["train"])
        v = net.votes(d.x["val"])
        assert v.shape == (6, 2, 64, 64)
        assert torch.equal(v[4:5], net.votes(d.x["val"][4:5]))
        assert torch.equal(net.predict(d.x["val"]), v.argmax(1))
