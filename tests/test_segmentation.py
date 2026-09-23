"""Dense (per-pixel) segmentation models, metrics and the context helpers around them."""

import io

import pytest
import torch

import torchtsetlin as tt
from torchtsetlin.data import make_scenes, make_segmentation_shapes


def _rng_state(device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    return torch.random.get_rng_state()


def _train(model, x, y, epochs=8, bs=8):
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=x.device)
        for i in range(0, x.shape[0], bs):
            j = perm[i : i + bs]
            model.update(x[j], y[j])
    model.eval()
    return model


# ------------------------------------------------------------------------------ learning
def test_dense_learns_solid_regions(device):
    """A disc interior and a ring interior look identical pixel by pixel, so the model has
    to use the neighbourhood. If mIoU does not clear chance, the feedback path is broken."""
    x, y = make_segmentation_shapes(120, size=16, seed=0, device=device)
    xt, yt = make_segmentation_shapes(60, size=16, seed=1, device=device)
    m = tt.SegmentationTsetlinMachine(
        3, n_clauses=100, T=30, s=5.0, patch_size=5, weighted=True, n_states=100
    ).to(device)
    _train(m, x, y, epochs=10)
    pred = m.predict(xt)
    assert tt.metrics.pixel_accuracy(pred, yt) > 0.88
    assert tt.metrics.mean_iou(pred, yt, n_classes=3) > 0.55


def test_position_literals_help_when_layout_is_stable(device):
    """On scenes with a horizon the vertical position is real information, and the model
    should be able to use it — this pins that the position literals reach the clauses."""
    rgb, lab = make_scenes(60, 32, seed=0, device=device)
    enc = tt.data.ColorThermometerEncoder(n_bits=2, value_range=(0.0, 1.0)).to(device)
    x = enc(rgb).float()
    scores = {}
    for pos in (False, 8):
        tt.seed_everything(0)
        m = tt.SegmentationTsetlinMachine(
            4, n_clauses=120, T=40, s=10.0, patch_size=3, position_encoding=pos,
            weighted=True, n_states=100,
        ).to(device)
        _train(m, x[:45], lab[:45], epochs=6)
        scores[pos] = tt.metrics.mean_iou(m.predict(x[45:]), lab[45:], n_classes=4)
    assert scores[8] > scores[False]


def test_coalesced_segmentation_learns(device):
    x, y = make_segmentation_shapes(90, size=16, seed=0, device=device)
    m = tt.CoalescedSegmentationTsetlinMachine(
        3, n_clauses=200, T=30, s=5.0, patch_size=5, n_states=100
    ).to(device)
    _train(m, x, y, epochs=8)
    assert tt.metrics.pixel_accuracy(m.predict(x), y) > 0.80


def test_multi_label_segmentation(device):
    """Overlapping masks: a pixel may belong to several structures at once."""
    x, y = make_segmentation_shapes(60, size=16, seed=0, device=device)
    masks = torch.stack([(y == 1), (y == 2)], dim=1)  # (B, 2, H, W)
    m = tt.CoalescedSegmentationTsetlinMachine(
        2, n_clauses=150, T=25, s=5.0, patch_size=5, multi_label=True, n_states=100
    ).to(device)
    _train(m, x, masks, epochs=6)
    pred = m.predict(x)
    assert pred.shape == masks.shape and pred.dtype == torch.bool
    assert float((pred == masks).float().mean()) > 0.80


# ------------------------------------------------------------------------------- geometry
def test_padding_preserves_resolution(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=5).to(device)
    x = torch.randint(0, 2, (2, 3, 13, 17), device=device).bool()
    m.update(x, torch.randint(0, 3, (2, 13, 17), device=device))
    m.eval()
    assert m.vote_map(x).shape == (2, 3, 13, 17)
    assert m.predict(x).shape == (2, 13, 17)
    assert m.output_shape() == (13, 17)


def test_stride_subsamples_the_label_map(device):
    """With stride 2 the output grid halves; full-resolution labels are subsampled to it."""
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3, stride=2).to(device)
    x = torch.randint(0, 2, (2, 1, 16, 16), device=device).bool()
    y_full = torch.randint(0, 3, (2, 16, 16), device=device)
    m.update(x, y_full)  # accepts input-resolution labels
    m.eval()
    assert m.output_shape() == (8, 8)
    assert m.predict(x).shape == (2, 8, 8)
    m.update(x, y_full[:, ::2, ::2])  # and output-grid labels
    with pytest.raises(ValueError, match="label map"):
        m.update(x, torch.randint(0, 3, (2, 5, 5), device=device))


def test_valid_padding(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3, padding=0).to(device)
    x = torch.randint(0, 2, (2, 1, 12, 12), device=device).bool()
    m.update(x, torch.randint(0, 3, (2, 12, 12), device=device))
    m.eval()
    assert m.output_shape() == (10, 10)
    assert m.predict(x).shape == (2, 10, 10)


def test_feature_count_matches_geometry(device):
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, position_encoding=4, input_shape=(2, 16, 16)
    ).to(device)
    assert m.n_features == 2 * 3 * 3 + 4 + 4
    plain = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=(3, 5), input_shape=(2, 16, 16)
    ).to(device)
    assert plain.n_features == 2 * 3 * 5


def test_single_image_and_single_plane_inputs(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (4, 12, 12), device=device).bool()  # (B, H, W) -> one plane
    m.update(x, torch.randint(0, 3, (4, 12, 12), device=device))
    m.eval()
    assert m.predict(x).shape == (4, 12, 12)
    assert m.predict(x[0]).shape == (1, 12, 12)  # single (H, W) image


# ------------------------------------------------------------------------------- feedback
def test_ignore_index_produces_no_feedback(device):
    """Every label ignored -> the automata must not move at all."""
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, ignore_index=255, weighted=True
    ).to(device)
    x = torch.randint(0, 2, (4, 1, 12, 12), device=device).bool()
    m.update(x, torch.zeros(4, 12, 12, dtype=torch.long, device=device))  # initialise
    before = m.ta_state.clone()
    w_before = m.weights.clone()
    m.update(x, torch.full((4, 12, 12), 255, dtype=torch.long, device=device))
    assert torch.equal(m.ta_state, before)
    assert torch.equal(m.weights, w_before)


def test_ignore_index_partially(device):
    """Ignored pixels are dropped, the rest still learn."""
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, ignore_index=255
    ).to(device)
    x = torch.randint(0, 2, (4, 1, 12, 12), device=device).bool()
    y = torch.randint(0, 3, (4, 12, 12), device=device)
    m.update(x, y)
    before = m.ta_state.clone()
    y2 = y.clone()
    y2[:, 6:, :] = 255
    m.update(x, y2)
    assert not torch.equal(m.ta_state, before)


def test_coalesced_ignore_index_freezes_weights(device):
    m = tt.CoalescedSegmentationTsetlinMachine(
        3, 40, T=10, s=3.0, patch_size=3, ignore_index=255
    ).to(device)
    x = torch.randint(0, 2, (4, 1, 12, 12), device=device).bool()
    m.update(x, torch.zeros(4, 12, 12, dtype=torch.long, device=device))
    before, w_before = m.ta_state.clone(), m.weights.clone()
    m.update(x, torch.full((4, 12, 12), 255, dtype=torch.long, device=device))
    assert torch.equal(m.ta_state, before)
    assert torch.equal(m.weights, w_before), "coalesced weights ignored the mask"


def test_class_feedback_p_zero_silences_a_class(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (4, 1, 12, 12), device=device).bool()
    m.update(x, torch.zeros(4, 12, 12, dtype=torch.long, device=device))
    m.set_class_feedback_p(torch.zeros(3))
    before = m.ta_state.clone()
    m.update(x, torch.randint(0, 3, (4, 12, 12), device=device))
    assert torch.equal(m.ta_state, before)
    with pytest.raises(ValueError):
        m.set_class_feedback_p(torch.ones(2))


def test_patches_per_commit_changes_commit_count(device):
    """One commit per `patches_per_commit` pixels, so the automata move more often."""
    calls = {"n": 0}
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, patches_per_commit=16, input_shape=(1, 8, 8)
    ).to(device)
    real = m._commit
    m._commit = lambda acc: (calls.__setitem__("n", calls["n"] + 1), real(acc))[1]
    x = torch.randint(0, 2, (2, 1, 8, 8), device=device).bool()
    m.update(x, torch.randint(0, 3, (2, 8, 8), device=device))
    assert calls["n"] == 2 * 64 // 16


def test_sequential_mode_commits_per_pixel(device):
    calls = {"n": 0}
    m = tt.SegmentationTsetlinMachine(
        3, 10, T=10, s=3.0, patch_size=3, input_shape=(1, 6, 6), feedback_mode="sequential"
    ).to(device)
    real = m._commit
    m._commit = lambda acc: (calls.__setitem__("n", calls["n"] + 1), real(acc))[1]
    x = torch.randint(0, 2, (1, 1, 6, 6), device=device).bool()
    m.update(x, torch.randint(0, 3, (1, 6, 6), device=device))
    assert calls["n"] == 36


# ----------------------------------------------------------------------------- inference
def test_vote_map_matches_forward_and_predict(device):
    m = tt.SegmentationTsetlinMachine(4, 30, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (3, 2, 10, 10), device=device).bool()
    m.update(x, torch.randint(0, 4, (3, 10, 10), device=device))
    m.eval()
    folded = m(x)  # (B*P, K)
    assert folded.shape == (3 * 100, 4)
    vm = m.vote_map(x)
    assert torch.equal(vm, folded.view(3, 10, 10, 4).permute(0, 3, 1, 2))
    assert torch.equal(m.predict(x), vm.argmax(dim=1))


def test_train_eval_mode_empty_clause_semantics(device):
    """An empty clause is True while learning and False when predicting; forgetting
    model.eval() silently changes every pixel."""
    m = tt.SegmentationTsetlinMachine(3, 21, T=10, s=3.0, patch_size=3, input_shape=(1, 8, 8)).to(device)
    x = torch.randint(0, 2, (2, 1, 8, 8), device=device).bool()
    assert int(m.include_count.sum()) == 0  # fresh model: every clause is empty
    m.train()
    lits = m._encode(m._coerce_input(x))
    assert m.clause_outputs(lits).all(), "empty clauses must be True while learning"
    train_votes = m(x)
    m.eval()
    assert not m.clause_outputs(lits).any(), "empty clauses must be False when predicting"
    assert float(m(x).abs().sum()) == 0.0  # nothing fires -> no votes
    # 21 clauses/class split 11 positive / 10 negative, so training mode nets +1 per class
    assert not torch.equal(train_votes, m(x))


def test_prediction_is_rng_free(device):
    """Dense prediction must never consume randomness — no patch is drawn outside feedback."""
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (4, 1, 10, 10), device=device).bool()
    m.update(x, torch.randint(0, 3, (4, 10, 10), device=device))
    m.eval()
    before = _rng_state(device)
    first = m.vote_map(x)
    assert torch.equal(_rng_state(device), before), "vote_map() consumed randomness"
    assert torch.equal(m.vote_map(x), first)


def test_chunking_does_not_change_predictions(device):
    """Tiny chunk budgets must only change memory use, never the answer."""
    m = tt.SegmentationTsetlinMachine(3, 40, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (6, 1, 12, 12), device=device).bool()
    m.update(x, torch.randint(0, 3, (6, 12, 12), device=device))
    m.eval()
    full = m.vote_map(x)
    m.max_chunk_elements = 2**12  # forces one image per chunk
    assert m._chunk_size(m._coerce_input(x)) < 6
    assert torch.equal(m.vote_map(x), full)


def test_clause_map_localises_clauses(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (2, 1, 10, 10), device=device).bool()
    m.update(x, torch.randint(0, 3, (2, 10, 10), device=device))
    m.eval()
    assert m.clause_map(x).shape == (2, 10, 10, 60)[:1] + (60, 10, 10)
    assert m.clause_map(x, [0, 5]).shape == (2, 2, 10, 10)


# ------------------------------------------------------------------------ (de)serialising
def test_save_load_roundtrip(device):
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, position_encoding=4, weighted=True
    ).to(device)
    x = torch.randint(0, 2, (4, 2, 12, 12), device=device).bool()
    m.update(x, torch.randint(0, 3, (4, 12, 12), device=device))
    m.eval()
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    clone = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, position_encoding=4, weighted=True,
        input_shape=(2, 12, 12),
    ).to(device)
    clone.load_state_dict(torch.load(buf, weights_only=True))
    clone.eval()
    assert torch.equal(clone.vote_map(x), m.vote_map(x))


# --------------------------------------------------------------------------------- metrics
def test_segmentation_metrics_basics():
    t = torch.tensor([[[0, 0, 1], [0, 1, 1], [2, 2, 1]]])
    p = t.clone()
    p[0, 0, 0] = 1
    assert tt.metrics.pixel_accuracy(p, t) == pytest.approx(8 / 9)
    ious = tt.metrics.iou(p, t, n_classes=3)
    assert ious[0] == pytest.approx(2 / 3)  # 2 of {0,0,0} kept, union 3
    assert ious[2] == pytest.approx(1.0)
    assert tt.metrics.mean_iou(p, t, n_classes=3) == pytest.approx(float(ious.mean()))
    d = tt.metrics.dice(p, t, n_classes=3)
    assert d[2] == pytest.approx(1.0)


def test_iou_is_nan_for_absent_classes():
    t = torch.zeros(1, 4, 4, dtype=torch.long)
    ious = tt.metrics.iou(t, t, n_classes=3)
    assert ious[0] == pytest.approx(1.0)
    assert torch.isnan(ious[1:]).all(), "absent classes must be NaN, not 0"
    assert tt.metrics.mean_iou(t, t, n_classes=3) == pytest.approx(1.0)


def test_metrics_ignore_index():
    t = torch.tensor([[[0, 1], [255, 255]]])
    p = torch.tensor([[[0, 1], [1, 1]]])
    assert tt.metrics.pixel_accuracy(p, t, ignore_index=255) == pytest.approx(1.0)
    assert tt.metrics.pixel_accuracy(p, t) < 1.0
    cm = tt.metrics.segmentation_confusion_matrix(p, t, n_classes=2, ignore_index=255)
    assert int(cm.sum()) == 2


def test_metrics_accept_vote_maps():
    votes = torch.randn(2, 4, 6, 6)
    target = torch.randint(0, 4, (2, 6, 6))
    assert tt.metrics.pixel_accuracy(votes, target) == pytest.approx(
        tt.metrics.pixel_accuracy(votes.argmax(1), target)
    )


def test_boundary_f1():
    def stripe(split):
        t = torch.zeros(1, 16, 16, dtype=torch.long)
        t[:, :, split:] = 1
        return t

    truth = stripe(8)
    assert tt.metrics.boundary_f1(truth, truth, tolerance=2) == pytest.approx(1.0)
    assert tt.metrics.boundary_f1(stripe(9), truth, tolerance=2) == pytest.approx(1.0)
    assert tt.metrics.boundary_f1(stripe(11), truth, tolerance=2) < 1.0
    # a prediction with no edge at all is a total miss, not an undefined score
    flat = torch.zeros(1, 16, 16, dtype=torch.long)
    assert tt.metrics.boundary_f1(flat, truth, tolerance=2) == 0.0


def test_segmentation_report_renders():
    t = torch.randint(0, 3, (2, 8, 8))
    txt = tt.metrics.segmentation_report(t, t, class_names=["a", "b", "c"])
    assert "mean IoU" in txt and "pixel accuracy" in txt and "a" in txt


# --------------------------------------------------------------------------------- context
def test_boolean_pool():
    x = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    x[0, 0, 0, 0] = True
    x[0, 0, 2, 2] = x[0, 0, 2, 3] = True
    assert tt.functional.boolean_pool(x, 2).flatten().tolist() == [True, False, False, True]
    assert not tt.functional.boolean_pool(x, 2, "and").any()
    assert torch.equal(tt.functional.boolean_pool(x, 1), x)
    ones = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    assert tt.functional.boolean_pool(ones, 2, "and").flatten().tolist() == [True]


def test_thermometer_cast_is_monotone():
    v = torch.tensor([[[[-30.0, 0.0], [10.0, 50.0]]]])
    out = tt.functional.thermometer_cast(v, (-20, 0, 20))
    assert out.shape == (1, 3, 2, 2)
    assert out[0, :, 0, 0].tolist() == [False, False, False]  # -30 clears nothing
    assert out[0, :, 1, 1].tolist() == [True, True, True]  # 50 clears everything
    assert out[0, :, 0, 1].tolist() == [True, True, False]  # 0 clears -20 and 0


def test_pyramid_encoder_shape_and_context():
    enc = tt.data.PyramidEncoder(scales=(1, 2, 4), patch=3)
    x = torch.rand(2, 3, 16, 16) > 0.5
    out = enc(x)
    assert out.shape == (2, enc.output_size(3), 16, 16)
    assert out.dtype == torch.bool
    # the full-resolution level's centre tap is the pixel itself
    centre = out[:, 4]  # plane 0, offset (0, 0) of a 3x3 patch
    assert torch.equal(centre, x[:, 0])


def test_balanced_class_probabilities():
    y = torch.tensor([0] * 90 + [1] * 10)
    p = tt.data.balanced_class_probabilities(y, 2)
    assert p[1] == pytest.approx(1.0)
    assert p[0] == pytest.approx(10 / 90, abs=1e-6)
    p = tt.data.balanced_class_probabilities(y, 2, floor=0.5)
    assert p[0] == pytest.approx(0.5)


def test_patch_clause_outputs_is_the_unreduced_match_tensor(device):
    """The conv accessor must agree with the disjunction the model actually takes."""
    m = tt.ConvTsetlinMachine(2, 10, T=10, s=3.0, patch_size=3, input_shape=(1, 8, 8)).to(device)
    x = torch.randint(0, 2, (4, 1, 8, 8), device=device).bool()
    m.update(x, torch.randint(0, 2, (4,), device=device))
    m.eval()
    pm = m.patch_clause_outputs(x)
    assert pm.shape == (4, 6, 6, 20)
    assert torch.equal(pm.any(dim=1).any(dim=1), m.evaluate_clauses(x))


# ---------------------------------------------------------------------------------- data
def test_make_segmentation_shapes():
    x, y = make_segmentation_shapes(5, size=16, seed=0)
    assert x.shape == (5, 1, 16, 16) and x.dtype == torch.bool
    assert y.shape == (5, 16, 16) and y.dtype == torch.long
    assert set(y.unique().tolist()) <= {0, 1, 2}


def test_make_scenes():
    rgb, lab = make_scenes(4, 32, seed=0)
    assert rgb.shape == (4, 3, 32, 32) and 0.0 <= float(rgb.min()) and float(rgb.max()) <= 1.0
    assert lab.shape == (4, 32, 32)
    assert set(lab.unique().tolist()) <= set(range(len(tt.data.SCENE_CLASSES)))
    with pytest.raises(ValueError):
        make_scenes(1, 8)


# ------------------------------------------------------------------------------- trainer
def test_trainer_segmentation(device):
    x, y = make_segmentation_shapes(60, size=16, seed=0, device=device)
    m = tt.SegmentationTsetlinMachine(
        3, n_clauses=60, T=25, s=5.0, patch_size=5, weighted=True, n_states=100
    ).to(device)
    m.class_names = ["background", "disc", "ring"]
    trainer = tt.Trainer(m, device=device, batch_size=8, verbose=False)
    assert trainer.task == "segmentation"
    hist = trainer.fit((x, y), epochs=3, val_data=(x, y))
    assert "val_mean_iou" in hist.keys()
    assert "val_iou_disc" in hist.keys()
    assert 0.0 <= hist.epochs[-1]["batch_accuracy"] <= 1.0
    res = trainer.test((x, y))
    assert set(res) >= {"test_pixel_accuracy", "test_mean_iou", "test_mean_dice"}
    assert trainer.predict(x).shape == (60, 16, 16)


def test_trainer_segmentation_multilabel(device):
    x, y = make_segmentation_shapes(40, size=16, seed=0, device=device)
    masks = torch.stack([(y == 1), (y == 2)], dim=1)
    m = tt.CoalescedSegmentationTsetlinMachine(
        2, 100, T=25, s=5.0, patch_size=5, multi_label=True, n_states=100
    ).to(device)
    trainer = tt.Trainer(m, device=device, batch_size=8, verbose=False)
    res = trainer.evaluate((x, masks))
    assert set(res) == {"hamming_accuracy", "micro_f1"}


# ------------------------------------------------------------------------ interpretability
def test_explain_pixel(device):
    x, y = make_segmentation_shapes(40, size=16, seed=0, device=device)
    m = tt.SegmentationTsetlinMachine(
        3, 60, T=25, s=5.0, patch_size=5, weighted=True, n_states=100
    ).to(device)
    _train(m, x, y, epochs=5)
    e = tt.interpret.explain_pixel(m, x[0], 8, 8, target=int(y[0, 8, 8]))
    assert e.row == 8 and e.col == 8
    assert e.prediction == int(m.predict(x[:1])[0, 8, 8])
    assert len(e.votes) == 3
    if e.matched:
        # absolute naming puts real image coordinates in the rule
        assert "p[0," in e.matched[0].expression
    assert str(e).startswith("pixel (8, 8)")
    with pytest.raises(IndexError):
        tt.interpret.explain_pixel(m, x[0], 99, 0)


def test_explain_pixel_rejects_flat_models(device):
    m = tt.TsetlinMachine(8, 2, 10, T=10, s=3.0).to(device)
    with pytest.raises(TypeError):
        tt.interpret.explain_pixel(m, torch.zeros(8, device=device), 0, 0)


def test_clause_patch_and_region(device):
    m = tt.SegmentationTsetlinMachine(
        3, 20, T=10, s=3.0, patch_size=3, position_encoding=4, input_shape=(2, 16, 16)
    ).to(device)
    assert m.clause_patch(0).shape == (2, 3, 3)
    r = m.clause_region(0)
    assert r["y"] == (0, 15) and r["x"] == (0, 15)  # fresh clause constrains nothing
    names = m.pixel_feature_names(5, 7)
    assert names[0] == "p[0,4,6]"  # top-left of the 3x3 patch centred on (5, 7)
    assert len(names) == m.n_features


# ------------------------------------------------------------------------------ validation
def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="padding"):
        tt.SegmentationTsetlinMachine(3, 20, T=10, padding="valid")
    with pytest.raises(ValueError, match="patches_per_commit"):
        tt.SegmentationTsetlinMachine(3, 20, T=10, patches_per_commit=0)
    with pytest.raises(ValueError, match="position_encoding"):
        tt.SegmentationTsetlinMachine(3, 20, T=10, position_encoding=0)
    with pytest.raises(ValueError, match="multi-label"):
        tt.CoalescedSegmentationTsetlinMachine(
            3, 20, T=10, multi_label=True, ignore_index=255
        )


def test_rejects_bad_labels(device):
    m = tt.SegmentationTsetlinMachine(3, 20, T=10, s=3.0, patch_size=3).to(device)
    x = torch.randint(0, 2, (2, 1, 8, 8), device=device).bool()
    with pytest.raises(ValueError, match="labels must lie"):
        m.update(x, torch.full((2, 8, 8), 7, dtype=torch.long, device=device))
    with pytest.raises(ValueError, match="label maps"):
        m.update(x, torch.zeros(3, 8, 8, dtype=torch.long, device=device))
