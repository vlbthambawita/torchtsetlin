# Booleanization

Tsetlin machines consume Boolean features. Turning raw data into Booleans is the single most
important modelling decision, because the literals *are* the vocabulary of the learned rules.
`torchtsetlin.data` provides encoders that follow the scikit-learn `fit` / `transform`
protocol and are `nn.Module`s (they move to the GPU and are saved in `state_dict()`).

## Continuous features: thermometer encoding

`ThermometerEncoder(n_bits, strategy)` turns each value `v` into bits `[v >= t_1, …, v >= t_n]`
for increasing thresholds. Two literals then express a range: `age>=30 AND NOT age>=50`.

```python
enc = tt.data.ThermometerEncoder(n_bits=8, strategy="quantile").fit(x_train_float)
xb = enc(x_train_float)                       # (N, F * 8) bool
model.feature_names = enc.feature_names(column_names)   # readable rules
```

Strategies: `"quantile"` (equal-frequency, robust default), `"uniform"` (equal-width, pass
`value_range=(0, 255)` for images), `"unique"` (evenly spaced ranks of the unique training
values, like tmu's `StandardBinarizer`).

## Categorical features

`OneHotEncoder` produces one Boolean per category (`menopause=ge40`), as in the book's breast
cancer example. `BitPlaneEncoder` gives the binary digits of integers when you want compact
codes (`even numbers` are `NOT bit0`).

## Images

* `Binarizer(0.3)` — fixed threshold (the original MNIST recipe: `pixel > 0.3`).
* `AdaptiveThresholdEncoder(block_size=11, C=2, method="gaussian")` — local thresholding
  (OpenCV's `ADAPTIVE_THRESH_GAUSSIAN_C`), used for MNIST/Fashion-MNIST in the convolutional
  TM papers.
* `ColorThermometerEncoder(n_bits=8)` — per-channel thermometer, `(N, 3, H, W)` →
  `(N, 24, H, W)`, the standard CIFAR encoding for TM composites.
* `BitPlaneEncoder(8)` — 8 place-value planes per channel.

`load_mnist_boolean` / `load_torchvision_boolean` apply any of these to a torchvision
dataset and return Boolean tensors ready for the GPU.

## Text and sets

Bag-of-words presence bits are the classical text encoding. `HypervectorEncoder` maps
token ids to sparse random hypervectors and bundles them with OR (optionally binding word
position with a cyclic shift), a compact alternative for large vocabularies.

## Composition

```python
pipeline = tt.data.Compose(tt.data.ThermometerEncoder(4), tt.data.Flatten())
xb = pipeline.fit_transform(x)
pipeline.output_size(n_columns)
```

## Decoding rules

`ThermometerEncoder.decode_range` and `ConvTsetlinMachine.clause_region` translate included
bits back to intervals, and `interpret.aggregate_literal_importance` sums literal
importances over the bits of one original feature.
