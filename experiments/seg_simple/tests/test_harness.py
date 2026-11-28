"""Tests T1 to T6 of the plan (section 7) and checks of the encodings, windows and job list."""

import shutil

import pytest
import torch

import jobs as JOBS
from common import IGNORE, WINDOW, model_dir, read_json, record_path
from data import (Data, augment, boundary_band, cache_path, encode, floors, manifest, planes, pool_planes, position_planes,
                  sample_pixels, sym, sym_inv)
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
    assert len(n["improve"]) == 15 and {j["model"] for j in n["improve"]} == {"pool"}
    assert len(n["coarse"]) == 30 and len({j["cell"] for j in n["coarse"]}) == 10
    assert len(n["multi"]) == 27 and len({j["cell"] for j in n["multi"]}) == 9
    assert all(j["n"] == sum(j["counts"]) and j["k"] == max(j["sizes"]) and j["px"] == 256 for j in n["multi"])
    assert JOBS.multi_job()["tag"] == "multi.k3-5-9-17-25.n80x5.T4.px256__s0" and JOBS.multi_job(sizes=(9,))["counts"] == [400]
    assert JOBS.coarse_job()["tag"] == "cell.c16.w64.k5.T4.n400.px256__s0"
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


def test_pool_planes_match_the_dense_model():
    """Scale 1 with position bits holds the bits the dense model sees for patch 3 (other order)."""
    import torchtsetlin as tt
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8, device=DEV)
    for enc in ("T4", "AG", "T4+AG"):
        assert planes(x, enc).shape[1] == 3 * JOBS.DEPTH[enc]
        j = JOBS.J("x", "x", model="pool", n=10, enc=enc, scales=JOBS.WIDE)
        assert pool_planes(x, enc, JOBS.WIDE).shape == (2, JOBS.features(j), 32, 32)
    m = tt.SegmentationTsetlinMachine(2, n_clauses=4, T=2, patch_size=3, position_encoding=8).to(DEV)
    lit = m._encode(m._prepare(planes(x, "T4")))                                 # (B*P, 2F), patch bits then position bits
    F = lit.shape[1] // 2
    mine = pool_planes(x, "T4", (1,)).flatten(2).transpose(1, 2).flatten(0, 1).float()
    assert mine.shape[1] == F
    assert torch.equal(mine[:, -16:], lit[:, F - 16:F])                           # the position bits, bit for bit
    assert torch.equal(mine[:, :-16].sum(1), lit[:, :F - 16].sum(1))             # the same 3 x 3 neighbourhood
    assert position_planes(32, 32, DEV).shape == (16, 32, 32)


def test_sampled_pixels_follow_the_mixture():
    y = torch.zeros(4, 32, 32, dtype=torch.long, device=DEV)
    y[:3, 10:14, 10:14] = 1                                                      # 16 polyp pixels of 1024; image 3 has none
    band = boundary_band(y)
    assert band[0, 9, 9] and band[0, 12, 12] and not band[0, 20, 20] and not band[3].any()
    torch.manual_seed(0)
    idx = sample_pixels(y, 4000, 1.0, 0.0)
    lab = y.flatten(1).gather(1, idx)
    assert abs(lab[:3].float().mean().item() - 0.5) < 0.03 and lab[3].sum() == 0  # classes present get equal shares
    idx = sample_pixels(y, 4000, 0.0, 1.0)
    assert band.flatten(1).gather(1, idx)[:3].all()                              # only boundary pixels
    assert idx[3].unique().numel() > 900                                         # no boundary: uniform
    uni = y.flatten(1).gather(1, sample_pixels(y, 4000, 0.0, 0.0))[:3].float().mean().item()
    assert abs(uni - 16 / 1024) < 0.01


@needs_data
def test_pool_model_learns_and_resumes():
    import train
    kw = dict(model="pool", n=40, enc="T4+AG", scales=JOBS.WIDE, limit=16)
    whole = JOBS.J("smoke", "t", max_epochs=4, prefix="t7whole-", **kw)
    part = JOBS.J("smoke", "t", max_epochs=2, prefix="t7part-", **kw)
    _clean(whole["tag"], part["tag"])
    try:
        assert train.run(whole, DEV, wandb_on=False) == 0
        assert train.run(part, DEV, wandb_on=False) == 0
        assert train.run({**part, "max_epochs": 4}, DEV, wandb_on=False) == 0
        a, b = read_json(record_path(whole["tag"])), read_json(record_path(part["tag"]))
        assert [r["val_mIoU"] for r in a["history"]] == [r["val_mIoU"] for r in b["history"]]
        assert a["model"]["n_features"] == JOBS.features(whole) == 826
        d = Data(DEV, limit=6)
        net = Net(whole, DEV)
        v = net.votes(d.x["val"])
        assert v.shape == (6, 2, 64, 64) and torch.equal(v[4:5], net.votes(d.x["val"][4:5]))
    finally:
        _clean(whole["tag"], part["tag"])


def test_cell_windows_labels_and_votes():
    x = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8, device=DEV)
    y = torch.zeros(3, 32, 32, dtype=torch.long, device=DEV)
    y[0, :8, :4] = 1                                                             # exactly half of cell (0, 0)
    y[1, 8:16, 8:11] = 1                                                         # less than half of cell (1, 1)
    job = JOBS.coarse_job(n=10, csize=8, win=16, k=3, pos=True, px=32)
    net = Net(job, DEV)
    lab = net.cell_labels(y)
    assert lab.shape == (3, 4, 4) and lab[0, 0, 0] == 1 and lab.sum() == 1
    assert torch.equal(net.oracle(y)[0, :8, :8], torch.ones(8, 8, dtype=torch.long, device=DEV))
    P = net._cell_padded(x, "t")
    assert P.shape == (3, 12, 40, 40)
    b, yy, xx = torch.tensor([0, 2, 1], device=DEV), torch.tensor([0, 24, 9], device=DEV), torch.tensor([0, 24, 17], device=DEV)
    w = net._cell_windows(P, b, yy, xx)
    assert w.shape == (3, 12 + 14, 16, 16)
    e = planes(x, "T4")
    for i in range(3):                                                           # the cell sits in the middle of its window
        assert torch.equal(w[i, :12, 4:12, 4:12], e[b[i], :, yy[i]:yy[i] + 8, xx[i]:xx[i] + 8])
    assert not w[0, :12, :4].any() and not w[1, :12, 12:].any()                  # outside the image
    assert not w[0, 12:].any() and w[1, 12:, 3, 5].sum() == 12                   # position planes: centre (4, 4) and (28, 28) of 32
    assert w[2, 12:19, 0, 0].sum() == 3 and w[2, 19:, 0, 0].sum() == 5           # centre (13, 21) of 32: > 4, 8, 12 and > 4 .. 20
    v = net.votes(x)
    assert v.shape == (3, 2, 32, 32) and torch.equal(v[1:2], net.votes(x[1:2]))
    assert torch.equal(v[:, :, ::8, ::8].repeat_interleave(8, 2).repeat_interleave(8, 3), v)          # constant inside a cell


@needs_data
def test_cell_model_resumes():
    import train
    kw = dict(n=40, px=64, csize=8, win=16, k=3, limit=16, phase="smoke", group="t")
    whole, part = JOBS.coarse_job(max_epochs=4, prefix="t8whole-", **kw), JOBS.coarse_job(max_epochs=2, prefix="t8part-", **kw)
    _clean(whole["tag"], part["tag"])
    try:
        assert train.run(whole, DEV, wandb_on=False) == 0
        assert train.run(part, DEV, wandb_on=False) == 0
        assert train.run({**part, "max_epochs": 4}, DEV, wandb_on=False) == 0
        a, b = read_json(record_path(whole["tag"])), read_json(record_path(part["tag"]))
        assert [r["val_mIoU"] for r in a["history"]] == [r["val_mIoU"] for r in b["history"]]
        assert a["cell_oracle"]["val"]["mIoU"] > 0.7 > a["val"]["mIoU"]
    finally:
        _clean(whole["tag"], part["tag"])


def test_multi_groups_read_centred_neighbourhoods():
    x = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8, device=DEV)
    net = Net(JOBS.multi_job(px=32, sizes=(3, 9), counts=(10, 10)), DEV)
    P, e = net._multi_padded(x, "t"), planes(x, "T4")
    b, yy, xx = torch.tensor([0, 1, 1], device=DEV), torch.tensor([0, 20, 31], device=DEV), torch.tensor([0, 10, 31], device=DEV)
    i3, i9 = net.m.inputs(P, b, yy, xx)
    assert i3.shape == (3, 12 * 9 + 16, 1, 1) and i9.shape == (3, 12 * 81 + 16, 1, 1)
    w3, w9 = i3[:, :108, 0, 0].view(3, 12, 3, 3), i9[:, :972, 0, 0].view(3, 12, 9, 9)
    assert torch.equal(w9[1], e[1, :, 16:25, 6:15]) and torch.equal(w3[1], e[1, :, 19:22, 9:12])
    assert torch.equal(w9[:, :, 3:6, 3:6], w3)                                   # the small neighbourhood is the centre of the large
    assert not w9[0, :, :4].any() and not w9[2, :, 5:].any()                     # outside the image
    assert torch.equal(i3[:, 108:], i9[:, 972:])                                 # the same position bits in every group
    assert i3[0, 108:].sum() == 0 and i3[2, 108:].sum() == 16
    y = (x[:, 0] > 127).long()
    net.job["pool_n"] = 256
    for _ in range(3):                                                           # some learning, so that clauses differ
        net.train_epoch(x, y)
    v = net.votes(x)
    assert v.shape == (2, 2, 32, 32) and torch.equal(v[1:2], net.votes(x[1:2])) and v.abs().sum() > 0
    net.m.eval()                                                                 # the fast votes are the library's votes
    g = torch.arange(32 * 32, device=DEV)
    lib = net.m.library_votes(net.m.inputs(P, torch.zeros_like(g), g // 32, g % 32))
    assert torch.equal(lib.view(32, 32, 2).permute(2, 0, 1), v[0])


@needs_data
@pytest.mark.parametrize("joint", [True, False])
def test_multi_model_resumes(joint):
    import train
    kw = dict(px=64, sizes=(3, 9), counts=(20, 20), pool_n=64, joint=joint, limit=16, phase="smoke", group="t")
    whole, part = JOBS.multi_job(max_epochs=4, prefix="t9whole-", **kw), JOBS.multi_job(max_epochs=2, prefix="t9part-", **kw)
    _clean(whole["tag"], part["tag"])
    try:
        assert train.run(whole, DEV, wandb_on=False) == 0
        assert train.run(part, DEV, wandb_on=False) == 0
        assert train.run({**part, "max_epochs": 4}, DEV, wandb_on=False) == 0
        a, b = read_json(record_path(whole["tag"])), read_json(record_path(part["tag"]))
        assert [r["val_mIoU"] for r in a["history"]] == [r["val_mIoU"] for r in b["history"]]
        assert a["model"]["clauses_total"] == 80 and "literals_per_clause_k9" in a["model"] and "floors" in a
    finally:
        _clean(whole["tag"], part["tag"])
