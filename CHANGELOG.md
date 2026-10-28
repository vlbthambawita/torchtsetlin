# Changelog

## 0.6.0 (2026-10-28)

**The CUDA path is rebuilt around two Triton kernels and CUDA graphs** (gaps G15-G24 of the
low-level kernel audit, `experiments/paper_experiments/LIBRARY_GAPS.md`). What a model learns is
unchanged in distribution, and CUDA-graph replays are bit-identical to eager execution. Same GPU
(RTX 3080), per `update()` against 0.5.1: flat MNIST batch 32 4.7 -> 1.1 ms, batch 100 5.1 -> 1.7
ms; IMDb-sized flat model 20.7 -> 6.6 ms; convolutional MNIST batch 16 5.0 -> 1.5 ms, batch 100
22.7 -> 7.6 ms; Noisy XOR sequential 17x and 2-D Noisy XOR sequential 12x faster; convolutional
prediction 4.7x faster.

- **Fused commit kernel** (`torchtsetlin._triton.fused_commit`; G15, G16, G17, G22). A batched
  CUDA update that fits one chunk records its events instead of count tensors; one kernel computes
  the Type Ia / Type II counts per tile (tensor cores for flat models, a gather of each clause's
  random matching patch for convolutional ones), samples them (inverse CDF, walked arithmetically
  for small counts), and writes the state, the include cache, the bit-packed include mask and the
  include counts. No `(C, 2F)` count buffer is allocated, saving 12 bytes per automaton.
- **Fused convolutional evaluation** (`_triton.conv_eval`; G20, G21, G24). Clauses are evaluated
  on bit-packed literals with the OR over patches in registers; while learning the kernel also
  draws each clause's random matching patch for Type Ia and, independently, Type II. Conv literals
  are float16 on CUDA. `ConvTsetlinMachine.patch_clause_outputs` keeps the dense evaluation.
- **No host synchronisation in an update** (G18). Labels already on a CUDA device are checked by
  a device-side assertion (`utils.class_targets`); labels from the host are checked there before
  the copy. Feedback selection uses `scatter_`, the CDF table is cached per device, and the
  table width comes from the batch size instead of a read-back.
- **CUDA graphs** (G19). On CUDA, an update shape seen for the third time (sequential mode: blocks
  of 64 examples) is captured and replayed. Off with `model.cuda_graphs = False` or
  `TORCHTSETLIN_CUDA_GRAPHS=0`; not used with clause or literal dropout.
- `functional.commit_dense` and `functional.commit_single_example` update `include_count` in
  place; `commit_dense` takes `max_events` and no longer sums the include counts inside the
  compiled kernel (half its time). `TORCHTSETLIN_TRITON=0` disables the Triton kernels.
- Start-up: the first update of a process takes 2.2 s cold / 0.7 s warm (was 5.2 / 2.1 s, G23).

## 0.5.1 (2026-10-28)

**Faster GPU training, and two CPU slow paths removed** (gaps G10-G14 of the cross-library
benchmark, `experiments/paper_experiments/LIBRARY_GAPS.md`). Learning is unchanged in
distribution; clause evaluation on CUDA is bit-identical. Times are per `update()` on an RTX 3090
/ Ryzen 9 3950X after warm-up, MNIST-sized models (20k clauses).

- **Fused dense commit on CUDA** (G10). A batched GPU update applies its feedback with one
  `torch.compile` kernel (`functional.commit_dense`) that draws the binomials inside the kernel
  (inverse CDF by bisection over a fixed table), writes the state and the include cache, and
  zeroes the counts for reuse, instead of ~20 full passes over the automata. Flat MNIST: batch 32
  7.2 -> 3.9 ms, batch 100 7.6 -> 4.1 ms. Hard budgets and counts above 255 per automaton keep the
  eager commit.
- **Clauses are evaluated in float16 on CUDA** (G11). The include cache is `float16` on CUDA
  (tensor-core matmul). The evaluation only tests whether a count of violated literals is zero,
  which half precision decides exactly, so clause outputs, votes and every random draw are
  unchanged. `compute_dtype` still sets the literals and the feedback counts; the CPU keeps it.
- **Convolutional batches are no longer split into tiny chunks** (G12). The chunk size counted
  the worst-case random-patch draw (every clause selected for every example), which split a
  batch of 16 MNIST images into 4 chunks. It now follows the largest real intermediate, the
  `(P, C)` evaluation block, and the patch draw is bounded separately by blocks of events
  (`_event_block`). With G10/G11, conv MNIST batch 16: 9.6 -> 4.1 ms on CUDA.
- **Sequential mode on CUDA: small models and convolutional models use the fused kernel** (G13).
  Whether to compile is decided on the work of the whole `update()` call, not one step, and
  `ConvTsetlinMachine` records each example's per-clause random matching patch for
  `functional.commit_single_example`, which now accepts `(C, 2F)` literals (and `literals_ii`).
  Noisy XOR: 3.1 -> 0.55 ms per example; 2-D Noisy XOR (conv): 3.6 -> 0.73 ms.
- **CPU: two slow paths removed** (G14). Clause-major evaluation is used only for up to 256 rows:
  for the thousands of patch rows of a convolutional batch it was 3x slower (conv MNIST batch 16,
  16 threads: 619 -> 258 ms per update). Counts beyond the CPU kernels' 64 x 32 table (large
  batches, small `s`) now bisect the wide table instead of falling back to the eager geometric
  sampler (coalesced MNIST, 2000 clauses, s=3, batch 100: 2047 -> 110 ms).

## 0.5.0 (2026-10-24)

**Learning follows the reference implementations, and training is much faster.** A
configuration audit against pyTsetlinMachine, pyTsetlinMachineParallel, PyTsetlinMachineCUDA,
tmu and green-tsetlin (`experiments/paper_experiments/LIBRARY_GAPS.md`) found nine gaps; eight
are closed here, and CPU training is the one still open (14x faster, but still slower than
pyTsetlinMachine on one core). G1, G3 and G4 change what a model learns, so results measured
with 0.4.0 on weighted, regression or focused-sampling models are not directly comparable.

- **Behaviour change: clause weights floor at 1, not 0** (G1). `TsetlinMachine(weighted=True)`
  and `RegressionTsetlinMachine(weighted=True)` used to let Type II drive a weight to 0, which
  silences the clause. pyTsetlinMachine and tmu keep every weight at least 1.
- **Behaviour change: regression feedback probability is the squared error** (G3),
  `((v* - v)/T)**2`, as in pyTsetlinMachine, pyTsetlinMachineParallel and tmu. It was `|v* - v|/T`.
- **Behaviour change: focused negative sampling uses the reference rule** (G4). A non-target class
  is drawn with probability proportional to `(T + clip(v_k, -T, T)) / 2T`, as in tmu and
  green-tsetlin. It was proportional to `v_k - min(v) + 1`. `functional.sample_negative_classes`
  takes a new `T` argument and raises without it when `focused=True`.
- **Behaviour change: automaton states are 8-bit by default** (G6). `state_dtype=None` picks
  `uint8` when `2 * n_states - 1 <= 255` (the default `n_states=128`), else `int16` / `int32`,
  so a model and its checkpoint are 4x smaller, the same 8 bits per automaton the reference
  implementations use. Commits form the new state in a wider buffer, so narrow dtypes cannot
  wrap. Checkpoints saved with int32 states load unchanged. Cast before doing arithmetic on
  `ta_state` yourself (`model.ta_state.long() - N`).
- **Faster CPU training and sequential mode** (G8, G9). On the CPU and in sequential mode the
  commit is row-sparse (`functional.apply_feedback_rows`): Type Ia / Type II counts are kept only
  for the clauses that matched and were selected, Type Ib forgetting is drawn by geometric gaps
  (`functional.sample_forget_events`, exact), and the include caches are refreshed only where the
  state changed. The count buffers are reused between updates. Same distribution as before.
  MNIST-sized model (20k clauses, batch 32): CPU 2.2x faster with 16 threads (and it now scales
  with threads, 2.7x from 1 to 16), GPU sequential mode 2.8x faster. Batched GPU training keeps
  the dense commit and is unchanged.
- **`torch.compile` kernels for the update** (G8, G9). On the CPU, Type Ib forgetting and the
  binomial draws of the row-sparse commit run as exact inverse-CDF kernels with the random
  numbers drawn inside the kernel, and clause evaluation is computed clause-major. In sequential
  mode on CUDA, flat models (`TsetlinMachine`, `CoalescedTsetlinMachine`,
  `RegressionTsetlinMachine`) commit each example with one fused kernel
  (`functional.commit_single_example`). MNIST-sized model, one epoch from initialisation: CPU
  batch 32 with 16 threads 1,259 s -> 91 s, GPU sequential mode 406 s -> 36 s (including
  compilation). The compiled paths are used only for large enough work, never with an explicit
  `generator=`, and fall back to the eager code (with one warning) where `torch.compile` does not
  work. `functional.set_compile(False)` or `TORCHTSETLIN_COMPILE=0` turns them off. Compiled and
  eager runs consume random numbers differently, so a seeded run reproduces only under the same
  setting. The first update of a session pays the compilation (seconds; cached on disk).
- `max_weight` caps clause weights (G2): `TsetlinMachine`, `RegressionTsetlinMachine` and
  (on the magnitude) `CoalescedTsetlinMachine`.
- `clamp_prediction_votes` is now a constructor argument (G7), for pyTsetlinMachine's clamped
  argmax.
- Documented (G5): a clause over the literal budget still gets its weight updated, as in tmu;
  pyTsetlinMachine skips that increment.

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
