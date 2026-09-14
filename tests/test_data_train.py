
import pytest
import torch

import torchtsetlin as tt
from torchtsetlin.data import (
    AdaptiveThresholdEncoder,
    Binarizer,
    BitPlaneEncoder,
    BooleanTensorDataset,
    ColorThermometerEncoder,
    Compose,
    Flatten,
    HypervectorEncoder,
    OneHotEncoder,
    ThermometerEncoder,
    make_2d_noisy_xor,
    make_noisy_xor,
    make_parity,
    make_shapes,
    make_xor,
)


def test_synthetic_datasets():
    x, y = make_xor(10)
    assert x.shape == (10, 2) and y.tolist() == (x[:, 0] ^ x[:, 1]).long().tolist()
    x, y = make_noisy_xor(20, noise=0.5, seed=1)
    assert x.dtype == torch.bool and y.dtype == torch.long
    x, y = make_parity(20)
    assert set(y.tolist()) <= {0, 1}
    x, y = make_2d_noisy_xor(50, size=4)
    assert x.shape == (50, 1, 4, 4)
    x, y = make_shapes(10, size=6)
    assert x.shape == (10, 1, 6, 6) and x.sum(dim=(1, 2, 3)).min() >= 4


def test_thermometer_encoder():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    enc = ThermometerEncoder(3, strategy="uniform").fit(x)
    out = enc(x)
    assert out.shape == (5, 3) and out.dtype == torch.bool
    assert out.sum(1).tolist() == sorted(out.sum(1).tolist())  # monotone
    assert enc.output_size(1) == 3
    assert len(enc.feature_names(["v"])) == 3
    q = ThermometerEncoder(4, strategy="quantile").fit(torch.randn(200, 3))
    assert q(torch.randn(5, 3)).shape == (5, 12)
    assert q(torch.randn(5, 3), ).shape == (5, 12)
    u = ThermometerEncoder(2, strategy="unique").fit(torch.tensor([[1.0], [1.0], [2.0]]))
    assert u.thresholds.shape == (1, 2)
    with pytest.raises(RuntimeError):
        ThermometerEncoder(3)(torch.randn(2, 2))
    fixed = ThermometerEncoder(2, strategy="uniform", value_range=(0, 10))
    assert fixed(torch.tensor([[5.0, 9.0]])).int().tolist() == [[1, 0, 1, 1]]


def test_other_encoders():
    assert Binarizer(0.5)(torch.tensor([[0.2, 0.7]])).tolist() == [[False, True]]
    oh = OneHotEncoder().fit(torch.tensor([[0, 1], [2, 0]]))
    assert oh(torch.tensor([[2, 1]])).int().tolist() == [[0, 0, 1, 0, 1]]
    bp = BitPlaneEncoder(4)
    assert bp(torch.tensor([[9]])).int().tolist() == [[1, 0, 0, 1]]
    assert bp(torch.zeros(2, 3, 4, 4)).shape == (2, 12, 4, 4)
    ad = AdaptiveThresholdEncoder(5, 2, "mean")
    img = torch.zeros(1, 1, 8, 8)
    img[0, 0, 2:6, 2:6] = 255
    out = ad(img)
    assert out.shape == (1, 1, 8, 8) and bool(out[0, 0, 3, 3]) and not bool(out[0, 0, 0, 0])
    with pytest.raises(ValueError):
        AdaptiveThresholdEncoder(4)
    ct = ColorThermometerEncoder(4)
    assert ct(torch.zeros(2, 3, 5, 5)).shape == (2, 12, 5, 5)
    hv = HypervectorEncoder(10, dim=64, n_bits=4, bind_position=True)
    out = hv(torch.tensor([[1, 2, -1]]))
    assert out.shape == (1, 64) and 4 <= int(out.sum()) <= 8
    comp = Compose(ThermometerEncoder(2, strategy="uniform"), Flatten())
    assert comp.fit_transform(torch.randn(4, 3)).shape == (4, 6)
    assert comp.output_size(3) == 6


def test_boolean_tensor_dataset():
    ds = BooleanTensorDataset([[1, 0], [0, 1]], [0, 1])
    assert ds.x.dtype == torch.bool and ds.y.dtype == torch.long and len(ds) == 2
    dr = BooleanTensorDataset([[1, 0]], [0.5], regression=True)
    assert dr.y.dtype == torch.float32


def test_trainer_fit_evaluate_predict(tmp_path, device):
    x, y = make_noisy_xor(1500, noise=0.4, seed=0)
    xt, yt = make_noisy_xor(500, noise=0.0, seed=1)
    m = tt.TsetlinMachine(12, 2, 20, T=15, s=3.9, n_states=50)
    ckpt = tmp_path / "best.pt"
    csv = tmp_path / "log.csv"
    calls = []
    trainer = tt.Trainer(
        m,
        device=device,
        batch_size=10,
        verbose=False,
        callbacks=[
            tt.EarlyStopping(patience=100),
            tt.ModelCheckpoint(str(ckpt)),
            tt.CSVLogger(str(csv)),
            tt.StateSummaryLogger(),
            tt.LambdaCallback(on_epoch_end=lambda e, logs: calls.append(e)),
            tt.HyperparameterSchedule("s", lambda e: 3.9),
        ],
        metrics={"margin": lambda votes, y: float(tt.metrics.vote_margin(votes).mean())},
    )
    hist = trainer.fit((x, y), epochs=12, val_data=(xt, yt))
    assert len(hist) == 12 and calls == list(range(12))
    assert hist["val_accuracy"][-1] > 0.7
    assert "val_margin" in hist.keys() and "included_literals_mean" in hist.keys()
    assert ckpt.exists() and csv.exists()
    assert trainer.test((xt, yt))["test_accuracy"] > 0.7
    pred = trainer.predict((xt, yt))
    assert pred.shape == (500,)
    votes = trainer.predict(xt, return_votes=True)
    assert votes.shape == (500, 2)
    best, epoch = hist.best("val_accuracy")
    assert 0 <= epoch < 12 and best >= hist["val_accuracy"][-1] - 1e-9


def test_trainer_dataloader_and_early_stop():
    from torch.utils.data import DataLoader, TensorDataset

    x, y = make_noisy_xor(400, seed=0)
    loader = DataLoader(TensorDataset(x, y), batch_size=20, shuffle=True)
    m = tt.TsetlinMachine(12, 2, 10, T=5, n_states=20)
    tr = tt.Trainer(m, verbose=False, callbacks=[tt.EarlyStopping(monitor="val_accuracy", patience=1, restore_best=True)])
    hist = tr.fit(loader, epochs=50, val_data=DataLoader(TensorDataset(x, y), batch_size=100))
    assert len(hist) < 50  # early stopping kicked in
    assert tr.evaluate(loader)["accuracy"] >= 0.0


def test_trainer_regression_and_multilabel():
    x = torch.randint(0, 2, (300, 6)).bool()
    yr = x[:, :2].float().sum(1)
    r = tt.RegressionTsetlinMachine(6, 20, T=10, y_range=(0, 2))
    h = tt.Trainer(r, verbose=False).fit((x, yr), epochs=2, val_data=(x, yr))
    assert {"val_mae", "val_rmse", "val_r2", "batch_mae"} <= set(h.epochs[-1])
    ml = tt.CoalescedTsetlinMachine(6, 2, 10, T=5, multi_label=True)
    ym = x[:, :2]
    h = tt.Trainer(ml, verbose=False).fit((x, ym), epochs=2, val_data=(x, ym))
    assert "val_hamming_accuracy" in h.epochs[-1]
    assert tt.evaluate(ml, (x, ym))["hamming_accuracy"] >= 0
    assert tt.predict(ml, x).shape == (300, 2)


def test_metrics():
    votes = torch.tensor([[2.0, -1.0], [-1.0, 3.0], [1.0, 0.0]])
    y = torch.tensor([0, 1, 1])
    assert abs(tt.metrics.accuracy(votes, y) - 2 / 3) < 1e-6
    cm = tt.metrics.confusion_matrix(votes, y)
    assert cm.tolist() == [[1, 0], [1, 1]]
    prf = tt.metrics.precision_recall_f1(cm)
    assert prf["recall"][0] == 1.0
    assert "macro avg" in tt.metrics.classification_report(votes, y, ["a", "b"])
    rm = tt.metrics.regression_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert abs(rm["mae"] - 0.5) < 1e-6
    curve = tt.metrics.trustworthiness_curve(votes, y, n_levels=3)
    assert curve["accuracy"].shape == (3,)
    assert 0 <= tt.metrics.expected_calibration_error(torch.softmax(votes, 1), y) <= 1


def test_interpret(device):
    x, y = make_noisy_xor(400, seed=0, device=device)
    m = tt.TsetlinMachine(12, 2, 10, T=5, n_states=20).to(device)
    for _ in range(5):
        m.update(x, y)
    act = tt.interpret.clause_activity(m, x, y)
    assert act.shape == (2, 20)
    cp = tt.interpret.clause_precision(m, x, y)
    assert cp["precision"].shape == (20,)
    gi = tt.interpret.global_feature_importance(m)
    assert gi.shape == (2, 24)
    li = tt.interpret.local_feature_importance(m, x[:3])
    assert li.shape == (3, 24)
    agg = tt.interpret.aggregate_literal_importance(gi[0], groups=[0] * 6 + [1] * 6)
    assert agg.shape == (2,)
    ex = tt.interpret.explain(m, x[0])
    assert ex.prediction in (0, 1) and isinstance(str(ex), str)
    assert tt.interpret.rules(m) == m.rules()


def test_viz_smoke():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    x, y = make_noisy_xor(200, seed=0)
    m = tt.TsetlinMachine(12, 2, 10, T=5, n_states=20, weighted=True)
    h = tt.Trainer(m, verbose=False).fit((x, y), epochs=2, val_data=(x, y))
    tt.viz.plot_history(h)
    tt.viz.plot_confusion_matrix(tt.metrics.confusion_matrix(m(x), y), ["a", "b"], normalize=True)
    tt.viz.plot_clause_memory(m, 0)
    tt.viz.plot_ta_states(m, clauses=[0, 1])
    tt.viz.plot_literal_frequency(m, top_k=5)
    tt.viz.plot_feature_importance(tt.interpret.global_feature_importance(m, 0), m.literal_names())
    tt.viz.plot_votes(m(x), y)
    tt.viz.plot_trustworthiness(tt.metrics.trustworthiness_curve(m(x), y))
    tt.viz.plot_clause_weights(m)
    xs, ys = make_shapes(50)
    c = tt.ConvTsetlinMachine(2, 4, T=2, patch_size=3)
    c.update(xs, ys)
    tt.viz.plot_conv_clause(c, 0)
    tt.viz.plot_conv_clauses(c, [0, 1, 2, 3], ncols=2)
    matplotlib.pyplot.close("all")
