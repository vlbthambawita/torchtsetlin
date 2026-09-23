# Changelog

## 0.2.0 (2026-09-23)

**Semantic segmentation.** Dense, per-pixel prediction is now a supported task rather than
something to rebuild in a notebook.

- **New models:** `SegmentationTsetlinMachine` and `CoalescedSegmentationTsetlinMachine`
  (`models/segmentation.py`). A convolutional TM takes the disjunction over patch positions —
  `matches.any(dim=1)` — which gives it translation tolerance and makes a dense output
  impossible: afterwards it knows a pattern occurred, not where. The dense models keep the
  patch axis and fold it into the batch axis instead (`B' = B·P`), so vote sums, Type I /
  Type II selection, feedback counting and `apply_feedback` all apply unchanged. `predict`
  returns a `(B, H, W)` label map, `vote_map` a `(B, K, H, W)` vote surface, and `forward`
  the folded `(B·P, K)` votes that keep `Trainer` / `metrics` / `interpret` working.
  `padding="same"` with `stride=1` preserves the input resolution; `ignore_index` handles the
  void class of CamVid / Cityscapes / VOC; multi-label mode takes `(B, K, H, W)` mask stacks.
- **`patches_per_commit` (default 128).** One batch of 8 images at 32×32 is 8 192
  pixel-examples, and committing their feedback at once is a mini-batch of 8 192 — far past
  where batched feedback stops matching the classical algorithm. Measured over 3 seeds on
  `make_segmentation_shapes`: mIoU 0.749 at 128 against 0.527 at 4 096, and the *variance*
  grows with the commit size too (sd 0.009 → 0.073), so a large-commit run can look fine once
  and collapse the next time. `feedback_mode="sequential"` is `patches_per_commit=1`.
- **Context, measured.** A `k × k` patch cannot decide classes that need global structure. Two
  TM-native fixes ship: thermometer-encoded patch-position literals (`position_encoding`,
  now also an `int` to bound the feature count on large images) and `data.PyramidEncoder`, an
  OR-pooled multi-scale stack built on the new `functional.boolean_pool` and
  `functional.thermometer_cast`. On generated CamVid-like scenes, building IoU goes 0.505 with
  neither → 0.699 (pyramid) → 0.894 (position) → 0.921 (both), and mIoU 0.798 → 0.959. No
  float parameter and no gradient anywhere.
- **Imbalance:** `class_feedback_p` plus `data.balanced_class_probabilities`. A Tsetlin machine
  has no loss to reweight, so the lever is which pixels produce feedback. It is a rescue, not
  a default — it lifts a pyramid-only run 0.874 → 0.927 mIoU but *costs* 0.05 once position
  literals are on.
- **Metrics:** `pixel_accuracy`, `iou`, `mean_iou`, `dice`, `boundary_f1`,
  `segmentation_confusion_matrix`, `segmentation_metrics` and `segmentation_report`, all
  accepting vote maps, label maps or folded votes. Absent classes get `NaN` IoU rather than
  `0`, and `mean_iou` skips them. `confusion_matrix` gained `ignore_index`.
- **`Trainer`** detects a dense model, logs pixel accuracy / mIoU / mean Dice / per-class IoU,
  and accumulates the confusion matrix batch by batch instead of keeping `(B, K, H, W)` vote
  tensors for a whole test set.
- **Interpretability:** `interpret.explain_pixel` lists the clauses that decided one pixel,
  decoded over **absolute image coordinates** (`p[1,21,13]` = "plane 1 of pixel (21, 13)"), so
  a dense prediction can be audited pixel by pixel rather than attributed by a saliency
  heuristic. Plus `model.clause_map`, `model.pixel_feature_names`, `clause_patch` and
  `clause_region` for the dense geometry.
- **Data:** `make_scenes` (CamVid-like street scenes with a wavy horizon, per-class texture and
  CamVid-like class imbalance), `make_segmentation_shapes` (a fast toy where the classes are
  only separable from the neighbourhood), `load_segmentation_boolean` and
  `load_voc_segmentation_boolean` — the latter resize masks with **nearest** interpolation,
  since bilinear invents class indices that do not exist.
- **Visualisation:** `viz.plot_segmentation` (image / truth / prediction / error panels) and
  `viz.plot_vote_map`, plus `viz.segmentation_palette`.
- **`_ConvMixin.patch_clause_outputs`** exposes the per-patch match tensor `(B, Py, Px, C)` that
  previously had no public accessor — useful for plotting where a clause fires and for
  stacking blocks, though learning a dense output still needs the segmentation models.
- **Benchmarks:** a `segmentation` suite in `bench_device.py` reporting **pixels** per second,
  with a figure, table and page section. Measured on a 16-core CPU: ~35 000 px/s training and
  ~1 090 000 px/s predicting at the default commit size; `patches_per_commit=None` is 4.2×
  faster and measurably worse.
- **Docs:** new `concepts/segmentation.md`, `guides/segmentation.md` and
  `examples/segmentation.md`, with every number above reproduced from
  `examples/segmentation_scenes.py`. They also report the *row-prior baseline* — a classifier
  that sees no image content and still scores 0.876 pixel accuracy on these scenes — because
  pixel accuracy on a layout-driven dataset is a weak claim.
- Internal: `TsetlinMachineBase._accumulate_chunk` split into `_accumulate_literals` so models
  that encode once and commit several times do not re-encode. No behaviour change.

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
