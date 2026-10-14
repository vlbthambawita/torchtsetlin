"""Hand cases for the evaluator (PLAN section 6 / 9 required checks)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
import metrics as M  # noqa: E402


def _sq(n=12, a=3, b=8, k=1):
    y = np.zeros((n, n), np.int64)
    y[a:b, a:b] = k
    return y


def test_perfect_mask():
    y = _sq()
    s = M.summarize(M.confusion(y, y, 2)[None], {1: M.boundary_counts(y, y, 2, 1)[None]}, 2)
    assert s["miou"] == 1.0 and s["bf1_t1_mean"] == 1.0


def test_absent_class_excluded_false_positive_scores_zero():
    y = np.zeros((6, 6), np.int64)
    p = y.copy()
    s = M.summarize(M.confusion(p, y, 3)[None], {}, 3)
    assert s["absent_classes"] == [1, 2] and s["miou"] == 1.0
    p[0, 0] = 2
    s = M.summarize(M.confusion(p, y, 3)[None], {}, 3)
    assert s["iou"][2] == 0.0 and s["absent_classes"] == [1]


def test_ignore_excluded():
    y = _sq()
    y[0, :] = M.IGNORE
    p = _sq()
    p[0, :] = 1  # wrong only where ignored
    assert M.summarize(M.confusion(p, y, 2)[None], {}, 2)["miou"] == 1.0


def test_class_swap_breaks_class_aware_boundary():
    y = np.zeros((12, 12), np.int64)
    y[2:6, 2:6] = 1
    y[6:10, 6:10] = 2
    p = y.copy()
    p[y == 1], p[y == 2] = 2, 1
    bc = M.boundary_counts(p, y, 3, 1)
    bf = M.boundary_f1_from_counts(bc)
    assert bf[1] < 0.5 and bf[2] < 0.5  # class-blind BF1 would be 1.0 here


def test_shifted_edge_tolerance():
    y = _sq(20, 5, 15)
    p = _sq(20, 6, 16)  # shift by one pixel
    b1 = M.boundary_f1_from_counts(M.boundary_counts(p, y, 2, 1))[1]
    b0 = M.boundary_f1_from_counts(M.boundary_counts(p, y, 2, 0))[1]
    assert b1 == 1.0 and b0 < 1.0


def test_empty_masks():
    y = np.zeros((8, 8), np.int64)
    bf = M.boundary_f1_from_counts(M.boundary_counts(y, y, 2, 1))
    assert np.isnan(bf[1])
    p = _sq(8, 2, 5)
    bf = M.boundary_f1_from_counts(M.boundary_counts(p, y, 2, 1))
    assert bf[1] == 0.0


def test_boundary_near_void_excluded():
    y = _sq()
    y[:, :4] = M.IGNORE
    p = _sq()
    c = M.boundary_counts(p, y, 2, 1)
    # the left edge of the square (column 3) is void-adjacent and removed from both sides
    assert c[1, 0] == c[1, 1] and c[1, 2] == c[1, 3]


def test_3d_dice_and_hd95():
    g = np.zeros((10, 10, 4), np.uint8)
    g[2:7, 2:7, 1:3] = 1
    assert M.dice_3d(g, g, 1) == 1.0 and M.hd95_3d(g, g, 1, (1, 1, 1)) == 0.0
    assert M.dice_3d(g, g, 2) is None and M.hd95_3d(g, g, 2, (1, 1, 1)) is None
    p = np.roll(g, 1, axis=0)
    assert abs(M.hd95_3d(p, g, 1, (2.0, 1, 1)) - 2.0) < 1e-6  # 1 voxel along a 2 mm axis


def test_bootstrap_zero_difference():
    rng = np.random.default_rng(0)
    cms = rng.integers(0, 50, (20, 2, 2))
    st = {"ids": list(range(20)), "cm": cms.tolist()}
    r = M.paired_bootstrap([st, st], [st, st], 2, n_boot=50)
    assert r["diff"] == 0.0 and r["ci95"] == [0.0, 0.0]


def test_contrast_bootstrap_identity_and_sign():
    import stats as ST

    rng = np.random.default_rng(1)
    a = [{"ids": list(range(30)), "cm": rng.integers(1, 40, (30, 2, 2)).tolist()} for _ in range(3)]
    b = []
    for s in a:  # b = a with more true positives everywhere -> positive difference
        cm = np.asarray(s["cm"]).copy()
        cm[:, 1, 1] += 20
        b.append({"ids": s["ids"], "cm": cm.tolist()})
    r0 = ST.contrast([a, a], [1, -1], 2, n_boot=50)
    assert r0["estimate"] == 0.0 and r0["ci95"] == [0.0, 0.0]
    r = ST.contrast([a, b], [-1, 1], 2, n_boot=200)
    assert r["estimate"] > 0 and r["ci95"][0] > 0
