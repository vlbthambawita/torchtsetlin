# Changelog

## 0.4.0 (2026-10-22)

**Training now matches the reference implementation (tmu) on the TM Composites benchmark.**
Reproducing *TMComposites* (Granmo, 2023) on CIFAR-10 turned up two library gaps. Both are fixed
here. With 0.4.0, torchtsetlin reproduces 8 of the 9 rows of the paper's Table 2, against 1 of 9
with 0.3.2 (`experiments/multi_convtm/REPORT.md`). Two behaviour changes, so check results you
compare against 0.3.x:

- **Behaviour change: the clause budget is soft again by default** (`budget_mode="soft"`, MC-2).
  `max_included_literals` now follows the reference implementation (tmu): the budget only gates
  Type Ia, so a clause may overshoot it and is then pruned by Type Ib until it shrinks back.
  - **Why**: the 0.3.x hard cap reverted every inclusion beyond the budget, so no clause ever
    exceeded it and the Type Ia -> Type Ib switch never fired. Clauses froze at exactly the budget.
  - **Effect** on the TM Composites CIFAR-10 reproduction (batch 50, 3 seeds): the 3x3 / 4x4
    colour-thermometer specialists go from about 51% to 61.5-61.7%, matching tmu. The
    four-specialist composite goes from 68.6% to 73.3% (`experiments/multi_convtm/REPORT.md`).
  - **Compatibility**: `budget_mode="hard"` restores 0.3.x. Every model trained with
    `max_included_literals` learns differently, so results measured with 0.3.x and a budget are not
    directly comparable. Experiments in `experiments/` that recorded results under 0.3.x pass no
    `budget_mode`, so re-running them now uses the soft rule.
- **Classifier predictions no longer clamp vote sums** (MC-1). Learning still clamps each class
  sum to `[-T, T]`, because the feedback probability is defined on it. `forward()` and
  `predict()` used the clamped sum too, so every class that reached `+T` tied for the argmax,
  and ties went to the lowest class index. `forward()` now returns the raw class sums, as the
  reference implementations do.
  - Measured on the TM Composites CIFAR-10 specialists, the clamp had cost 0.6-3.9 pp of
    test accuracy (`experiments/multi_convtm/REPORT.md`).
  - `predict_proba()` and `confidence()` still clamp, so the probabilities they return are
    unchanged.
  - `RegressionTsetlinMachine` keeps the clamp (new class attribute
    `clamp_prediction_votes = True`), because its output is the clamped sum rescaled to the
    target range.
  - This is a behaviour change: a classifier whose class sums exceed `T` predicts differently
    from 0.3.2, almost always better.

## 0.3.2 (2026-10-19)

**A segmentation memory fix, and documentation for the Tsetlin-machine literature.** There are no
API changes, and every non-segmentation model behaves exactly as in 0.3.1.

- **Segmentation: committed updates are budgeted per commit window** (LG-001). With
  `patches_per_commit` set, the chunk budget now counts only the `(P, 2L)` literals per image plus
  one commit window, rather than every intermediate per image. Before, 128-px images with a few
  hundred clauses were processed one image per chunk, so a commit could never mix pixels from
  different images. `forward()` and `patches_per_commit=None` keep the per-image bound.
- **Segmentation: opt-in `pixel_order="shuffle"`** permutes a chunk's pixels before committing.
  Returned votes stay in row-major `(image, patch)` order. The default is unchanged.
- **Docs: new *Benchmarks → Literature results* page.** It lists every dataset and reported result
  from 55 Tsetlin-machine papers (128 datasets, 3,295 numbers). Filtering, grouping and
  best-per-group ranking run as SQL in the browser over a SQLite catalog. The page also has a
  best-per-dataset leaderboard, dataset and paper tables, and a SQL console. The catalog's source
  of truth is `benchmarks/literature/papers/*.json`, rebuilt by
  `benchmarks/literature/build_catalog.py`. The same data is served by a
  [Hugging Face Space](https://huggingface.co/spaces/vlbthambawita/TsetlinMachines).
- **Docs:** the changelog page renders again; it was missing the `pymdownx.snippets` extension.
  The site now uses the white paper's orange palette, with an automata-grid logo and favicon
  (`scripts/make_logo.py`). The concept figures are recoloured to match, and the current release
  is shown site-wide.

## 0.3.1 (2026-10-12)

**3D convolution.** Clauses can now slide through volumes: MRI/CT stacks, or frame
sequences with time as depth.

- **New models:** `Conv3dTsetlinMachine`, `Conv3dCoalescedTsetlinMachine` and
  `Conv3dRegressionTsetlinMachine` take `(B, Z, D, H, W)` Boolean volumes with a
  `kd × kh × kw` window and depth/row/column thermometer position literals. Clause evaluation
  and the random-matching-patch feedback are the 2D code, unchanged. With `D = kd = 1`, the
  literals and the whole training run are bit-identical to `ConvTsetlinMachine`.
- **Patch-axis blocking:** when one volume would exceed `max_chunk_elements`, the patch axis
  is split into blocks (or set `patches_per_chunk`). The output is identical, and the patch
  draw has the same distribution.
- `clause_patch` returns `(Z, kd, kh, kw)`, `clause_region` adds a `"d"` range, and
  `patch_clause_outputs` returns `(B, Pd, Ph, Pw, C)`. `viz.plot_conv_clause(s)` draws 3D
  clauses slice by slice and takes `depth=` to show one slice.
- `data.make_3d_shapes`: two shapes that differ only in the order of their slices.
  `examples/shapes_conv3d.py` gets 1.000 on it, against 0.83–0.98 for the 2D model on
  slices stacked as channels and 0.73–0.80 for a flat machine.
- Docs: a "3D volumes" section with two new figures in *Convolution*, a `conv3d` benchmark
  suite on the CPU-vs-GPU page, and the design and proofs in `plans/conv3d.md`.

## 0.3.0 (2026-10-01)

**The clause-size budget is now a hard limit.** This is a behaviour change: models trained with
`max_included_literals` will learn differently from 0.2.0.

- **`max_included_literals` caps clause growth in every feedback mode** (issue #1). Before, the
  budget was only checked at the start of a commit. One batched commit aggregates many Type Ia
  and Type II events, and Type II was never gated, so a clause under budget could end the
  commit far over it: 41% of rules exceeded a budget of 6, and segmentation models averaged
  about 200 literals against a budget of 32. Now, after each commit, a clause that would exceed
  the budget keeps only a random subset of its newly included literals and the rest return to
  just below the include boundary. Oversized clauses still get forgetting-only feedback and can
  only shrink. Results measured with a budget under 0.2.0 are not directly comparable.
- **Fixed: `SegmentationTsetlinMachine` crashed on GPU when `class_feedback_p` was set before
  `.to(device)`** (issue #1). It is now a non-persistent buffer, so it moves with the model;
  checkpoints are unchanged and 0.2.0 checkpoints still load.

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
