# Changelog

## 0.1.3 (2026-09-15)

- **Fixed / faster (convolutional models):** the random matching patch used for Recognize and
  Reject feedback was drawn in `_ConvMixin._evaluate`, which builds a `(B, P, C)` `rand` tensor
  on *every* call — including `forward()` / `predict()` / `evaluate_clauses()`, where the draw is
  discarded. `_evaluate` now returns the per-patch match tensor as its feedback context and
  `_feedback_counts` draws from it, over the selected feedback events only. Prediction is
  RNG-free: on a 400-clause 10x10-window MNIST config, `predict()` went 885 -> 180 ms and
  `update()` 914 -> 216 ms per 64 images (4.9x / 4.2x). Each feedback event now also draws its
  own patch; a `ConvCoalescedTsetlinMachine` clause that takes Type Ia and Type II in the same
  update (16% of feedback pairs in a 4-output probe, 43% multi-label) was sharing one. Learning
  is unchanged for `ConvTsetlinMachine`, where the two feedback types never coincide.
- Verified the convolutional implementation against chapter 4 of *An Introduction to Tsetlin
  Machines* — patch layout, the OR over patches, the thermometer position bits (`P-1` per axis,
  bit `k` = `coord >= k+1`) and Recognize / Erase / Reject feedback all match the theory. Three
  new tests pin those semantics against a brute-force reference.
- Examples: three new convolutional scripts — `examples/shapes_conv.py` (2-D clauses vs a flat
  machine, and what `position_encoding` costs on a translation-invariant task),
  `examples/conv1d_ramps.py` (`Conv1dTsetlinMachine` on thermometer-encoded signals) and
  `examples/conv_regression_blobs.py` (`ConvRegressionTsetlinMachine`). `examples/mnist_conv.py`
  gained `--stride`, `--no-position-encoding`, `--coalesced` and `--max-samples`, keeps the
  encoder on the compute device, and prints the clauses it learned. New `examples/README.md`
  indexes all of them.
- Docs: new *Convolution on generated data* example page (`docs/examples/convolution.md`) in
  the Examples nav, a runnable-examples table on the Convolution concept page, and measured
  `--stride` / `--coalesced` / `position_encoding` comparisons on the MNIST example page.

## 0.1.2 (2026-09-15)

- **Fixed (silent data corruption on GPU):** `as_bool_tensor` / `as_long_tensor` /
  `as_float_tensor` moved tensors with `non_blocking=True` regardless of direction. A
  *device-to-host* copy issued that way returns before the transfer completes, so a CPU-resident
  `BooleanEncoder` handed a CUDA tensor thresholded stale memory and returned roughly 40% wrong
  bits — with no error, just a quietly worse model. All three now route through
  `utils._to_device`, which uses `non_blocking=True` only when the destination is CUDA;
  `Trainer` used the same pattern and is fixed too.
- Added: two notebooks working through the Tsetlin-machine segmentation literature —
  `examples/notebooks/04_convolutional_regression_tsetlin.ipynb` (C-RTM, ICMLT 2021) and
  `examples/notebooks/05_ctm_unet_segmentation.ipynb` (CTM-UNet, ISTM 2025), the latter
  including a dense per-pixel Tsetlin segmentation head built on `TsetlinMachine`.

## 0.1.1 (2026-09-14)

- Fixed: the 0.1.0 wheel on PyPI was missing the `torchtsetlin.data` subpackage (it had been
  excluded from git by an over-broad `.gitignore` rule), which made `import torchtsetlin` fail.
  0.1.0 should not be used.
- The publish workflow now runs the test-suite and smoke-imports the built wheel before uploading.

## 0.1.0 (2026-09-14)

First public release.

- Models: `TsetlinMachine` (multi-class, optional integer clause weights),
  `CoalescedTsetlinMachine` (shared clauses, multi-class / multi-label),
  `RegressionTsetlinMachine`, convolutional variants (`ConvTsetlinMachine`,
  `ConvCoalescedTsetlinMachine`, `ConvRegressionTsetlinMachine`, `Conv1dTsetlinMachine`).
- Learning options: specificity `s`, vote margin `T`, memory depth `n_states`, boosted
  true-positive feedback, clause size constraint (`max_included_literals`), drop-clause and
  drop-literal regularisation, focused negative sampling, batched or exact sequential feedback.
- Data preparation: thermometer / one-hot / bit-plane / adaptive-threshold / colour thermometer
  / hypervector encoders, synthetic datasets, torchvision helpers.
- Training: `Trainer` with `DataLoader` support, callbacks (early stopping, checkpoints, CSV
  logging, hyper-parameter schedules), metrics.
- Interpretation: rule extraction, clause activity/precision, closed-form global and local
  feature importance, per-example explanations.
- Visualisation: memory plots, automata heat-maps, convolutional clause patches, confusion
  matrices, trustworthiness curves.
- Documentation site (MkDocs) and examples.
