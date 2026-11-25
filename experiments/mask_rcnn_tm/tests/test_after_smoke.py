"""Checks that need the models of the smoke test (``setup.sh`` runs them after it): the K3
comparison and the sealed-evaluation code path. The sealed path is exercised on *validation*
images put in the place of the sealed split, so no test label is read here."""

import os
import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
os.environ.setdefault("KV_WANDB", "0")

import data as D  # noqa: E402
import jobs as J  # noqa: E402
from common import is_done, read_json, record_path  # noqa: E402

SMOKE = {j["tag"]: j for j in J.smoke_jobs()}
ready = D.CACHE_FILE.exists() and all(is_done(t) for t in SMOKE) and torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not ready, reason="needs the finished smoke test and a GPU")


@pytest.fixture(scope="module")
def data():
    return D.Data("cuda:0")


def test_k3_compares_the_arms_on_the_same_regions(data):
    import evals
    t = [k for k in SMOKE if k.startswith("smoke-mask__")]
    a = next(k for k in t if k == "smoke-mask__T4.p3x8.n300.T60__s0")
    c = next(k for k in t if "p5x4" in k and "joint" not in k and "comp" not in k)
    iv = next(k for k in t if "grad" in k)
    dense = next(k for k in SMOKE if k.startswith("smoke-dense"))
    out = evals.do_k3({"arms": {"A": [a], "C": [c], "B": [dense], "iv": [iv]}}, data, data.device)
    assert set(out["mean_iou"]) == {"A", "B", "C", "iv", "filled_box"} and out["n_regions"] > 300
    assert all(0.0 <= v <= 1.0 for v in out["mean_iou"].values())
    lo, hi = out["A-B"]["ci95"]
    assert lo <= out["A-B"]["mean"] <= hi and out["mask_head_for_pipeline"] in ("mask", "dense")


def test_sealed_path_scores_all_and_never_viewed_images(data, monkeypatch):
    import evals
    n = 40                                           # validation images stand in for the sealed split
    idx = torch.arange(data.n_train, data.n_train + n)
    monkeypatch.setattr(data, "test_x", data.x[idx.to(data.device)])
    monkeypatch.setattr(data, "test_inst", data.inst[idx.to(data.device)])
    monkeypatch.setattr(data, "test_group", data.group[idx])
    panel = read_json(D.MANIFESTS / "panel.json")["test"]
    monkeypatch.setattr(data, "test_stems", panel[:4] + [f"x{i}" for i in range(n - 4)])
    monkeypatch.setattr(data, "_test_boxes", [data.boxes[i] for i in idx.tolist()])
    for src in ("e2e-smoke-full__s0", "e2e-smoke-dense_cc__s0", "smoke-base-unet__s0", "smoke-base-mrcnn_iis__s0"):
        monkeypatch.setattr(evals, "stats_path", lambda tag: Path(os.environ.get("TMPDIR", "/tmp")) / f"kv_test_{os.getpid()}_{tag}.json")
        out = evals.do_sealed({"tag": f"sealed-test-{src}", "source": src}, data, data.device)
        assert out["n_images"] == n and out["n_never_viewed"] == n - 4
        assert out["metrics"]["semantic_threshold"] == read_json(record_path(src))["metrics"]["semantic_threshold"]
        assert 0.0 <= out["metrics"]["mask_AP50"] <= 1.0 and "Dice" in out["metrics_never_viewed"]
