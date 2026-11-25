"""Tests of the harness before any training (PLAN_KVASIR §11): T3, T7-T12 and the pipeline's
oracle path. They need the data cache (``python code/data.py``) and skip without it. T1, T2, T4
and T5 of PLAN.md test the dense and regression classes, which this harness does not use (see
DECISIONS.md); T6 is the pricing phase."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("KV_WANDB", "0")

import augment  # noqa: E402
import data as D  # noqa: E402
import members as M  # noqa: E402
from common import IGNORE, S  # noqa: E402

import torchtsetlin as tt  # noqa: E402

needs_data = pytest.mark.skipif(not D.CACHE_FILE.exists(), reason="data cache missing: python code/data.py")
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def data():
    return D.Data(DEVICES[-1])


def test_t3_scoring_refuses_training_mode():
    team = M.Team([{"conv": "flat", "clauses": 20, "T": 10}], 0, "single", "cpu")
    x = torch.rand(64, 30) > 0.5
    team.update([x], torch.randint(0, 2, (64,)))
    team.train()
    with pytest.raises(RuntimeError, match="training mode"):
        team.margin([x])
    team.eval()
    assert team.margin([x]).shape == (64,)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("graphs", [True, False])
def test_t7_one_joint_member_is_the_plain_model(device, graphs, monkeypatch):
    monkeypatch.setenv("TORCHTSETLIN_CUDA_GRAPHS", "1" if graphs else "0")
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(20, 128, 60, generator=g) > 0.5).to(device)
    y = torch.randint(0, 2, (20, 128), generator=g).to(device)
    kw = dict(weighted=True, **M.TM_KW)

    def run(joint):
        torch.manual_seed(7)
        if joint:
            m = M.JointFlat(60, 2, 40, 15, 10.0, **kw)
            m.joint_setup(128)
            m = m.to(device)
            team = M.FlatJointTeam(["m0"], [m], "sum")
        else:
            m = tt.TsetlinMachine(60, 2, 40, 15, 10.0, **kw).to(device)
        for i in range(20):
            torch.manual_seed(100 + i)
            team.update([x[i]], y[i]) if joint else m.update(x[i], y[i])
        return m.ta_state.clone(), getattr(m, "clause_weights", torch.zeros(1)).clone()

    a, b = run(False), run(True)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@needs_data
def test_t8_symmetries_invert_and_move_boxes_with_masks(data):
    inst = data.inst[700:740]
    for k in range(8):
        assert torch.equal(D.sym_inv(D.sym(inst, k), k), inst)
        assert torch.equal(D.sym_inv(D.sym(data.x[:8], k), k), data.x[:8])
        for j in range(len(inst)):
            assert torch.equal(D.boxes_of(D.sym(inst[j], k)), D.sym_boxes(data.boxes[700 + j], k))
    v = augment.make_view(data, torch.arange(32), "sym", seed=3, epoch=5)
    for j in range(32):
        assert torch.equal(D.boxes_of(v.inst[j]), v.boxes[j]) and torch.equal(D.valid_rect(v.inst[j]), v.rect[j])


@needs_data
def test_t12_albumentations_keeps_image_masks_and_boxes_aligned(data):
    assert augment.telemetry_off()
    idx = torch.arange(48)
    props = torch.stack([torch.cat([b[:1], torch.zeros(1, 4, device=b.device)]) if len(b) else torch.zeros(2, 4, device=data.device)
                         for b in data.boxes]).cpu()
    a = augment.make_view(data, idx, "alb", seed=1, epoch=2, props=props)
    b = augment.make_view(data, idx, "alb", seed=1, epoch=2, props=props)
    c = augment.make_view(data, idx, "alb", seed=1, epoch=3, props=props)
    assert torch.equal(a.x, b.x) and torch.equal(a.inst, b.inst) and not torch.equal(a.x, c.x)     # (seed, epoch, image)
    assert (a.x.permute(0, 2, 3, 1)[a.inst == IGNORE] == 0).all()                                 # padding stays black
    for j in range(len(idx)):
        assert torch.equal(D.boxes_of(a.inst[j]), a.boxes[j])
        for bx in a.boxes[j]:
            assert (bx[2] - bx[0]) >= 1 and (bx[3] - bx[1]) >= 1 and bx.min() >= 0 and bx.max() <= S
        if len(a.boxes[j]) == len(data.boxes[j]) == 1:                                            # the carried proposal follows its polyp
            p, t = a.props[j][0], a.boxes[j][0]
            if (p[2] - p[0]) > 4:
                assert (p[0] <= t[0] + 2) and (p[1] <= t[1] + 2) and (p[2] >= t[2] - 2) and (p[3] >= t[3] - 2)
    disc = torch.zeros(S, S, dtype=torch.uint8)
    yy, xx = torch.meshgrid(torch.arange(S), torch.arange(S), indexing="ij")
    disc[(xx - 60) ** 2 + (yy - 64) ** 2 <= 400] = 1
    tf = augment.build_alb("geo")
    xi, mi, _ = augment.alb_one(tf, np.full((3, S, S), 120, np.uint8), disc.numpy(), None, seed=5)
    ys, xs = np.nonzero(mi == 1)
    assert len(ys) > 400 and (xi.transpose(1, 2, 0)[mi == IGNORE] == 0).all()


@needs_data
def test_oracle_pipeline_is_perfect(data):
    """With every component replaced by the ground truth the evaluator must return AP = 1."""
    from pipeline import Pipeline, evaluate, run
    v = augment.make_view(data, torch.arange(700, 850), "none")
    pipe = Pipeline(data, data.device, {})
    dets = run(lambda x, inst, rect, gt: pipe.detect(x, inst, rect, gt_boxes=gt, oracle=("proposals", "class", "box", "mask")),
               v.x, v.inst, v.rect, v.boxes)
    m, _ = evaluate(dets, v.inst, v.boxes)
    assert m["mask_AP"] > 0.99 and m["box_AP"] > 0.99 and m["Dice"] > 0.999 and m["recall@30"] > 0.99


def run_py(script, *args, env=None):
    e = dict(os.environ, KV_WANDB="0", **(env or {}))
    return subprocess.run([sys.executable, script, *args], cwd=CODE, env=e, capture_output=True, text=True)


@needs_data
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_t9_t10_resume_and_panels_without_tracking(tmp_path):
    """A run stopped after an epoch and resumed ends with the same history as an uninterrupted
    one (CUDA graphs off), and writes its panels with tracking off."""
    import shutil

    from common import model_dir, read_json, record_path
    tag = "smoke-mask__T4.p3x8.n300.T60__s0"
    hist = []
    for interrupted in (False, True):
        shutil.rmtree(model_dir(tag), ignore_errors=True)
        record_path(tag).unlink(missing_ok=True)
        if interrupted:
            r = run_py("train.py", "--tag", tag, "--max-epochs", "1", "--no-graphs", "--no-wandb")
            assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
            record_path(tag).unlink()                 # the 1-epoch run is "done"; continue it to 3
        r = run_py("train.py", "--tag", tag, "--max-epochs", "3", "--no-graphs", "--no-wandb")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rec = read_json(record_path(tag))
        assert rec["status"] == "done" and rec["epochs"] == 3
        hist.append([h["val_monitor"] for h in rec["history"]])
        assert all(Path(p).exists() for p in rec["panels"]) and len(rec["panels"]) == 2
    assert hist[0] == hist[1]
    shutil.rmtree(model_dir(tag), ignore_errors=True)
    record_path(tag).unlink(missing_ok=True)


def test_t10_the_panel_path_computes_no_metric():
    import train
    for fn in (train.log_panels, train.panel_items):
        src = inspect.getsource(fn)
        for word in ("validate", "evaluate(", "average_precision", "recall_at", "semantic("):
            assert word not in src, f"{fn.__name__} mentions {word}"


@needs_data
def test_t11_concurrent_builds_leave_the_cache_alone():
    before = (D.CACHE_FILE.stat().st_mtime_ns, D.CACHE_FILE.stat().st_size)
    ps = [subprocess.Popen([sys.executable, "-c", "import data; data.build(); data.build_panel(); d = data.Data('cpu'); print(d.x.shape[0])"],
                           cwd=CODE, stdout=subprocess.PIPE, text=True) for _ in range(4)]
    outs = [p.communicate()[0].strip() for p in ps]
    assert outs == ["850"] * 4 and all(p.returncode == 0 for p in ps)
    assert (D.CACHE_FILE.stat().st_mtime_ns, D.CACHE_FILE.stat().st_size) == before
