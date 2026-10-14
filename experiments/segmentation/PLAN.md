# Segmentation improvement experiments

Status: **executed (compressed) on 2026-10-13 — see [report/REPORT.md](report/REPORT.md) and
[DECISIONS.md](DECISIONS.md) for what was run, cut and changed.** Written 2026-10-13 against
`torchtsetlin` v0.3.1, revision `f5168bdf656d597461a5ef14f463afd1eee90eeb`.
Task: [segimporvements_task.md](../../ideas/segimporvements_task.md).
The [earlier plan](../../ideas/segmentation_plan.md) records the already implemented
segmentation API; this program investigates improvements before library integration.

## 1. Objective and scope

Determine which changes improve dense TM segmentation, why they improve it, and their
cost relative to a tuned current TM and a trained U-Net. Aim for an interpretable,
competitive quality–resource tradeoff, without assuming that a TM must beat every CNN.
Separate three outcomes: better TM learning, better input features, and better structured
inference. An improvement in post-processing alone is not evidence of better clause learning.

All proposed mechanisms below are **hypotheses**. Code observations are identified as
facts with source links. Numerical budgets and gates are preregistered starting proposals,
not performance predictions. There are no new measurements in this document.

Research code, configurations, tests, and results will live under `experiments/segmentation/`.
Keep `src/torchtsetlin/` unchanged during experiments. Record defects or missing hooks in a
future `LIBRARY_GAPS.md` with a minimal reproduction, impact, and fix sketch. Integration
is a later, separately scoped change after the evidence review in section 10.
Preserve unrelated working-tree changes; record the initial source-tree hash and compare
it at each gate instead of assuming the entire repository starts clean.

Primary scope: supervised, 2D, single-label semantic segmentation on the user-selected
Kvasir-SEG and ACDC datasets. ACDC is trained slice by slice and evaluated after reconstructing
its annotated volumes; learned 3D models, instance segmentation, clinical evaluation, and
general-purpose differentiable TMs are separate work. Synthetic diagnostics remain small
mechanism checks. Kvasir-SEG is the primary real-data benchmark; ACDC tests the method on
a different modality, with models trained separately on each dataset.

## 2. What the code already does, and what remains uncertain

| Observation from current code | Implication for the experiments |
|---|---|
| [`_DenseMixin._encode`](../../src/torchtsetlin/models/segmentation.py) unfolds local patches and folds pixels into the example axis. It does not OR away pixel location. | Dense output already works. Replacing image-level OR pooling is not a new contribution here. |
| `SegmentationTsetlinMachine` shares clause rules over positions; a pixel's output depends on its supplied features. | Shared local rules exist, but the model has no built-in learned encoder–decoder or interaction between predicted labels. |
| [`PyramidEncoder`](../../src/torchtsetlin/data/encoders.py) supplies fixed multiscale neighborhoods. [`boolean_pool`](../../src/torchtsetlin/functional.py) supports OR and AND. Position literals and class feedback probabilities also exist. | Compare proposed context mechanisms against these stronger baselines, not only a 3×3 head. |
| Dense `update` walks encoded pixels in image/raster order and commits every `patches_per_commit`; the default is 128. | Correlated commits and aggregation size are distinct possible learning limitations. Test their interaction at equal pixel exposure. |
| `class_feedback_p` suppresses feedback for selected classes after dense encoding/evaluation. | It is neither rare-class oversampling nor a guarantee of less feature-extraction work. Count accepted feedback and measured cost. |
| Dense encoding materializes unfolded patches and their complementary literals for each image chunk. Image chunking cannot split a single image's unfold. The pyramid expands feature planes at full resolution. | A small commit size does not bound all memory allocations. High-resolution feasibility needs an explicit audit and potentially tiled extraction. |
| `_coerce_input` fixes the input shape after initialization. | Variable-size/tiled experiments need a tested adapter; changing image shape on an initialized model is not automatically supported. |
| Classifier votes are normally clipped to `[-T, T]`. Empty-clause prediction depends on evaluation mode. | Measure clipping/ties; use `model.eval()` for evaluation and feature caches. Vote values are not automatically calibrated probabilities. |
| [`boundary_f1`](../../src/torchtsetlin/metrics.py) compares boundaries of label maps without matching boundary class identity. | Keep it for continuity, but add class-aware boundary evaluation for this program. Swapping class names can preserve its score. |

These observations do **not** prove that the implementation underperforms CNNs on real data.
The [existing synthetic example](../../docs/examples/segmentation.md) illustrates useful
context mechanisms but also a strong row-position prior. Its historical numbers are not
new results and do not replace the controlled baselines below.

The architecture comparison is motivated by U-Net's contracting/expanding context paths
and localization design, not by an assumption that stacking arbitrary TM blocks reproduces
end-to-end CNN learning. [U-Net paper](https://arxiv.org/abs/1505.04597).

## 3. Questions and prioritized hypotheses

Let `N = BHW`, `F` be Boolean features per pixel, `C` total clauses across all classes/stages,
and `K` semantic classes. A clause is a conjunction of selected literals; its weighted vote
contributes to a per-pixel class score `v_i(k)`. Adding features increases both the automaton
bank, proportional to `2FC`, and the dense feature/evaluation workload.

| ID | Hypothesis and mechanism | Primary diagnostic | Priority |
|---|---|---|---|
| H1 | Decorrelated commits and deliberate pixel exposure improve learning stability and rare/boundary classes. | Quality versus accepted feedback, commit size, and time. | First |
| H2 | Occupancy and relative contrast retain spatial evidence lost by OR pooling or absolute intensity thresholds. | Context ambiguity, illumination shifts, bit saturation. | First |
| H3 | Coarse-to-fine learned vote features resolve ambiguity beyond a local receptive field while fine inputs retain edges. | Long-range task, object size, real-data mIoU. | Second |
| H4 | An edge-weighted Potts energy improves consistency without erasing small regions. | mIoU, class-aware boundary F1, thin-object recall. | Second |
| H5 | Anchored diffusion of score distributions offers a stable, cheaper alternative to discrete refinement. | Same cached unaries as H4; quality–latency curve. | Second |
| H6 | Learned relational messages add information beyond fixed smoothing and larger receptive fields. | Connectivity task at matched context/capacity. | Conditional |
| H7 | Topology-aware feedback sampling reduces breaks/holes on tasks with a justified structural prior. | Connectivity and topology errors alongside overlap. | Conditional |

The execution priority is an estimate of diagnostic value and implementation cost, not a
decision that any method works. Explore H1/H2 first; do not build a large combined model
before measuring its components.

### H1: feedback order, exposure, and imbalance

First permute the same valid pixels within each image batch before committing; compare
raster, uniform permutation, and spatially dispersed ordering at `q ∈ {32, 128, 512}`
pixels/commit. Use `q=1` and one-commit-per-image only on tiny diagnostic subsets. Keep
the image order, image batch size, total pixels, and dropout behavior controlled.
Undo permutations before interpreting returned vote maps. Record image chunk boundaries:
changing `max_chunk_elements` can also change commit boundaries and therefore learning.

Then sample a fixed number of valid training pixels using the normalized mixture

`p(i) = (1-a-b)/N_valid + a p_class(i) + b p_boundary(i)`,

where `a,b ≥ 0`, `a+b ≤ 1`; `p_class` first chooses a present class uniformly and then a
pixel of that class, and `p_boundary` is uniform on a training-label boundary band.
Use `(a,b) = (0,0), (0.5,0), (0,0.5), (0.25,0.25)` initially. If the boundary set is
empty, redistribute its mass to uniform sampling. Void pixels have probability zero.
Sampling with replacement means repeats are real feedback events and must be counted.

Controls: existing `class_feedback_p`, uniform sampling with the same accepted count,
and all pixels. Report accuracy both per epoch and per accepted pixel/commit budget.
This changes the training distribution; do not call it an unbiased estimate of the
original TM update, and do not insert inverse-probability weights into integer automata
without a separate derivation. Disable class thinning when testing an isolated sampler.

Prototype as an experiment-local dense adapter; reuse parent feedback operations. First
validate an identity/raster adapter against the library with matched RNG and commit
boundaries. For sampling efficiency, gather selected patches before expanding all
literals; compare against the simpler fully encoded reference. Reject a claimed speedup
if it merely performs fewer accepted updates than its control.

Failure condition: benefit disappears at matched exposure/time, or better mean IoU hides
a repeated rare-class or boundary regression. Log per-commit class entropy, spatial
distance, feedback counts, clause sizes, empty clauses, and state saturation.

### H2: multiscale occupancy and contrast literals

For a Boolean plane with independent occupancy probability `p`, OR pooling over `m`
pixels has probability `1-(1-p)^m` of being true. This simple model predicts saturation
as the window grows; real pixels are correlated, so measure actual plane entropy and
occupancy rather than assuming independence holds. AND pooling has the complementary
failure mode on sparse signals.

Replace or supplement OR with occupancy literals:

`z_(r,t)(i) = 1[mean_(j in W_r(i)) x(j) >= t]`,

starting with `t ∈ {0.25, 0.5, 0.75}`. Compare OR, AND, majority-only, and the three
occupancy thresholds at scales `(1,2,4)` and, only after screening, `(1,2,4,8)`.
Deduplicate identical channels at scale 1 and count retained features. Relative contrast
literals use `1[I(i)-mean_W I >= t]`; start with `t ∈ {-0.1,0,0.1}` on normalized RGB
for Kvasir-SEG or normalized single-channel intensity for ACDC.
Under an additive brightness shift without clipping these differences are invariant;
gamma, clipping, and multiplicative changes require separate tests.

Keep a fine-resolution branch. Occupancy should not blur every feature available to the
classifier. Record feature names including scale, channel, statistic, and threshold so
clause explanations stay meaningful. Any fitted thresholds use training images only.

Controls: current OR pyramid, AND pyramid, larger single patch, and the same extra feature
count allocated to ordinary local thresholds. Run both fixed-clause and approximately
fixed-automaton-byte comparisons; extra features are extra capacity. Use the same feature
sets with a linear pixel classifier to test whether improvement requires TM clauses.

Failure condition: a gain vanishes under the capacity control, or robustness improves
only at the expense of thin objects. Optional morphology (opening/closing) and quantized
Haar differences may follow a successful occupancy result; neither starts as another
unbounded search family.

### H3: a supervised, coarse-to-fine logical decoder

Start with two stages, not a full deep U-Net. A coarse dense TM at scale 4 predicts
semantic votes. Upsample thresholded normalized votes and concatenate them with original
fine-resolution features; train a fine TM on pixel labels:

`z_i = concat(local_boolean_features(i), upsample(1[v_coarse/T >= tau])(i))`.

Use `tau ∈ {-0.5,0,0.5}` initially, with the exact clipping and threshold convention
recorded. These are vote features, not asserted class probabilities. Coarse targets use
the majority valid label in each cell, ignoring all-void cells; compare center targets
as an ablation because majority targets can remove small classes. Keep the fine labels
unchanged. Evaluate coarse logits with nearest and bilinear upsampling, thresholding
after interpolation for the latter; do not interpolate categorical label IDs.

Train stages with their own ordinary supervised TM feedback, freeze the coarse stage,
then train the fine stage. There is no gradient through clauses and no claim of joint
optimization. A CNN Dice loss attached to the output does not train a discrete TM by itself.

Prevent a train/inference mismatch in stacked features: generate coarse training votes
out of fold using three image/group folds within the training split. Each fold model and
its encoder must exclude its held-out images. Freeze upstream hyperparameters before
cross-fitting. Train the deployment coarse model on all training images and evaluate
the final stack on validation; report the out-of-fold versus deployment feature shift.
Include all fold fits in compute totals. Ground-truth coarse maps may be used only as
an explicitly labeled oracle diagnostic, never as a competitive arm.

Controls: fixed pyramid, coarse-only upsampling, fine-only model, shuffled coarse vote
maps, and larger flat receptive field. Compare equal total `2FC` state budgets summed
over stages; split clause/state budget explicitly. Ablate fine skip inputs and vote
quantization. Only expand to three scales if two stages beat both fine-only and the
fixed pyramid on the long-range diagnostic and real validation data.

Failure condition: gains require oracle features, vanish against a matched fixed pyramid,
or training/inference feature mismatch dominates. This is inspired by the role of context
in [U-Net](https://arxiv.org/abs/1505.04597); it is a new experimental TM construction,
not a reproduction of U-Net's training method.

### H4: Potts-model spatial refinement

Treat the frozen TM votes as unary evidence for labels on a four-neighbor pixel graph:

`E(y) = sum_i U_i(y_i) + lambda sum_(i,j in edges) w_ij 1[y_i != y_j]`,

`U_i(k) = -v_i(k)/T`,

`w_ij = exp(-||I_i-I_j||²/(2 sigma²))`.

Count each undirected edge once. This is a ferromagnetic Potts-style preference for
agreement, weakened across image edges. Start with these normalized score unaries; a
separate calibrated arm may use `-log p_i(k)`. Do not equate raw votes with probabilities.
Fit any calibration on held-out training calibration groups or out-of-fold training
predictions; evaluate calibration on validation.

Initialize with unary argmin (same tie rule as raw TM). Use deterministic red/black
coordinate descent on the four-neighbor grid, retaining the current label on equal
energy. Each color set contains no adjacent vertices, so exact conditional minimization
cannot increase this energy. Stop when unchanged or after 10 sweeps; log energy and label
changes after each half-sweep. A local optimum is not a global optimum. Validate on tiny
binary graphs by exhaustive enumeration, and optionally compare an exact binary graph-cut
solver. Do not imply that multiclass coordinate descent is exact.

Initial grid: `lambda ∈ {0,0.1,0.3,1}` and `sigma ∈ {0.05,0.15}` with RGB or MRI
intensity in `[0,1]`; tune modalities independently because their edge scales differ;
deduplicate the `lambda=0` arm. Controls: unweighted Potts smoothing, majority filtering,
raw TM, and the same refinement on CNN predictions. This tests whether gains are generic
post-processing. Report changed-pixel rate and newly introduced errors.

Failure condition: IoU increases while boundaries/thin structures repeatedly degrade,
or additional latency exceeds its declared resource tier. If local smoothing cannot
resolve semantic ambiguity, proceed to learned context rather than increasing smoothing
until small objects vanish. CRFs provide established structured-prediction motivation;
this local solver is not the fully connected method in
[Krähenbühl and Koltun](https://arxiv.org/abs/1210.5644).

### H5: diffusion with a unary anchor

The physical analogy is diffusion of evidence within regions, restrained at strong image
edges. Fix nonnegative edge conductances from the input image, `L = D-W`, and solve

`min_Q 0.5 ||Q-P||_F² + (lambda/2) tr(Q^T L Q)`.

Here `P_i = softmax((v_i/T)/temperature)` is an operational score distribution, not a
claim of calibration. Use `temperature ∈ {0.5,1,2}`, selected on validation; use the same
unary/calibration options in H4 when comparing score representations. The solution solves
`(I + lambda L)Q = P`. The identity term anchors the solution and prevents unconstrained
diffusion from erasing all information as iteration count grows.

Prototype the update `Q <- Q - eta[(Q-P)+lambda LQ]` on the four-neighbor graph with
`eta <= 1/(1+2 lambda d_max)`, where `d_max` is maximum weighted degree. This conservative
bound gives a stable step for the quadratic objective; it also keeps the update a
nonnegative mixture when initialized at `P`. Verify probability sums, positivity, energy
decrease, and agreement with a direct tiny-system solve. If `d_max=0`, use `eta=1`.
Stop at relative linear-system residual `<1e-4` or 100 iterations; report nonconvergence.

Compare `lambda ∈ {0,0.1,1,10}`, fixed iteration latency caps, uniform conductance, H4,
and raw predictions. Keep the image-derived conductances fixed initially. Recomputing
them from predictions creates a nonlinear process and would need its own stability audit.

Failure condition: smoothing is indistinguishable from a cheap local average, or lacks a
quality–latency advantage. This proposed anchored graph problem draws on edge-preserving
diffusion, but is not a reproduction of the nonlinear Perona–Malik PDE.
[Perona and Malik](https://authors.library.caltech.edu/records/1p8h5-5x870).

### H6: relational messages (conditional)

Use a coarse grid with directional edges. At each node encode local observations and
quantized previous-round neighbor votes; train a new dense TM for each round on node
labels, freezing earlier rounds. Direction-specific slots preserve geometry. Use 1, 2,
and 4 rounds, a fine input skip, and the same out-of-fold discipline as H3. Track feature
provenance to decode a rule such as “neighbor above has strong road evidence.”

Controls: H3, fixed diffusion with the same communication radius, and a larger patch.
Match total state bytes and report propagation radius. Stop if gains are explained by
context size alone or if recursive rules become impractically large to explain.
[GraphTM](https://arxiv.org/abs/2507.14874) establishes related logical message-passing
work; this wrapper would not constitute a GraphTM reproduction. Any reproduction must
pin its implementation and separately verify the dense node-supervision contract.

### H7: topology-aware feedback (conditional)

On generated vessels, rings, and branching paths, measure connected components `beta_0`,
holes `beta_1`, path connectivity, and clDice alongside Dice/IoU. Fix foreground
4-connectivity and background 8-connectivity. Betti counts alone do not locate errors.

At an epoch boundary, predict the training set and prioritize mislabeled pixels whose
correction toward the training label reduces
`|beta_0(pred)-beta_0(target)| + |beta_1(pred)-beta_1(target)|`.
Compute candidate corrections on small maps first; cache the selected set for the next
epoch. Mix 25% priority sampling with 75% uniform sampling and retain ordinary supervised
TM feedback. Count the topology-selection cost. Compare against uniform, boundary-only,
and equally sized generic error sampling. This is a heuristic sampling policy, not
backpropagation of a topological loss or a guarantee of topology preservation.

Training labels may guide sampling; inference must not see target topology. Do not impose
“one component” or “no holes” on every polyp or cardiac slice. Generalization tests must
include valid disconnected structures, empty foreground, and varying component counts.
After synthetic controls, ACDC is an optional anatomy-specific extension: evaluate per-class
topology by slice level and phase, and do not force apical/basal slices to have a mid-ventricular
myocardial ring. Polyp/cardiac regions are not generally tubular networks, so clDice remains
a synthetic-curve diagnostic rather than the main real-data metric.
Use [clDice](https://arxiv.org/abs/2003.07311) as a relevant connectivity-aware metric;
its neural-network loss does not directly update Tsetlin automata.

## 4. Dataset ladder and leakage controls

| Tier | Dataset and split | Purpose |
|---|---|---|
| D0 | Existing `make_segmentation_shapes` and `make_scenes`; independent train/validation/final seeds with 1,000/200/200 images per generator, 32×32. Preserve a small 16×16 fixture. | Check correctness and reproduce qualitative baseline behavior. |
| D1 | New controlled generators: long-range identical-local-patch ambiguity, 1–3-pixel curves/branches, rings/holes, small rare objects, position/texture decoupling; 1,000/200/200 images per family, 64×64. | Identify mechanisms and expose shortcuts. |
| D2 | Kvasir-SEG: fixed 70/15/15 train/validation/final split with `split_seed=1234`, grouped as described below; nominally 700/150/150 images. | Primary binary polyp segmentation benchmark; 128×128 screens, 256×256 finalists. |
| D3 | ACDC: split the existing 100 training patients into 80 train / 20 validation, stratified by `Info.cfg` group with `split_seed=1234`; reserve the existing 50 testing patients for final evaluation. | Four-class cardiac MRI segmentation; 2D slice models, patient/phase volume evaluation. |

Use these existing directories read-only; write manifests and derived caches under this
experiment directory, not into the dataset roots:

| Dataset | Configured root | Files used |
|---|---|---|
| Kvasir-SEG | `/work/vajira/DATA/Kvasir_seg_original/Kvasir-SEG` | `images/*.jpg`, matching `masks/*.jpg`; pair by exact filename stem. |
| ACDC | `/work/vajira/DATA/ACDC` | Extracted `ACDC/database/training/patient*/` and `ACDC/database/testing/patient*/`; no need to extract `data.zip` again. |

Local inventory checked 2026-10-13: Kvasir has 1,000 images and 1,000 masks with matching
stems. ACDC has 100 training and 50 testing patient directories, with respectively 200
and 100 ground-truth file paths. Its training set has 20 patients in each of `DCM`, `HCM`,
`MINF`, `NOR`, and `RV`. This was a file/metadata inventory, not an evaluation of held-out
masks. Hash the files and save the exact split IDs before running experiments.

**Kvasir-SEG protocol.** Segment background `0` versus polyp `1`. The local masks are JPEG
images: decode as grayscale and apply the fixed threshold `>=128` before nearest-neighbor
mask resizing. Do not use `>0`, which would turn JPEG compression artifacts into foreground.
Check every image/mask pair for matching dimensions and audit a development-only sample
of decoded masks. The bounding boxes in `kavsir_bboxes.json` derive from annotations and
must not define input crops or inference features. The data's image/mask convention is
described in the [Kvasir-SEG paper](https://web-backend.simula.no/sites/default/files/publications/files/mmm_2020_kvasir_seg_debesh.pdf).

The local Kvasir root has no split manifest or patient metadata file. Before splitting,
group exact duplicates and reviewed near-duplicates using image content only; use any
reliable patient/procedure identifiers if independently available. Keep groups intact,
target 70/15/15 proportions, and record actual counts if grouping prevents exact counts.
Without patient identifiers this is an image/group-held-out experiment, not a demonstrated
patient-independent result. Freeze duplicate-group rules before fitting models; do not
change groups or choose the split seed to improve scores. Final images and labels become
sealed after manifest creation. Keep 2D aspect ratio with antialiased RGB letterboxing;
padding receives an experiment ignore label and is excluded from feedback and metrics.

**ACDC protocol.** Use the annotated ED/ES frames identified by `Info.cfg` and matching
`patientNNN_frameXX.nii.gz` / `patientNNN_frameXX_gt.nii.gz` pairs. Use class IDs `0`
background, `1` right-ventricular cavity, `2` myocardium, and `3` left-ventricular cavity;
verify this semantic mapping in the data adapter against release documentation. A local
training-mask check found IDs `{0,1,2,3}`. Keep both phases, all slices, augmentations, and
out-of-fold features of a patient in one split/fold. Use 16 training and 4 validation
patients from each of the five groups. Group labels are split/analysis metadata, not
segmentation inputs. Do not use unannotated cine frames as extra training data in the
primary comparison. The [ACDC dataset description](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
and [training protocol](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html)
describe the patient cohorts; cite Bernard et al.,
[IEEE TMI 2018](https://doi.org/10.1109/TMI.2018.2837502), as requested by the bundled
`MANDATORY_CITATION.md`.

Load NIfTI geometry with a pinned reader (e.g. nibabel); verify image/mask affine, shape,
orientation, and spacing. Preserve the short-axis slice axis and record all axis transforms.
Use all annotated slices, including empty-foreground slices. Normalize each image volume
independently: clip to its nonzero finite intensity 1st/99th percentiles, map to `[0,1]`,
and handle empty/constant volumes deterministically. This input-only rule applies unchanged
at inference and uses no mask-defined crop or intensity region. Encode one grayscale
channel for the TM and use one input channel for the CNN.

Train 128×128 screening and 256×256 finalist slices using aspect-preserving in-plane
resizing and padding; retain geometry to restore predictions. Do not interpolate along
the slice axis. Resize intensity images continuously and masks only with nearest-neighbor.
At evaluation undo padding/resizing on score maps, then take argmax on the original grid
and reconstruct each ED/ES volume. Check restoration with synthetic landmarks. The current
spatial-refinement arms operate within slices; through-plane refinement is a separate
ablation. Local test masks are available but must remain inaccessible to tuning, calibration,
feature fitting, and checkpoint selection. Report this as local held-out evaluation, not
a new challenge-server submission.

Create D1 long-range pairs whose local input neighborhoods are identical but whose center
labels depend on a distant visible cue; keep paired variants in the same split. Verify
local-feature collisions directly. A deterministic local-only predictor cannot distinguish
such a pair. Add an oracle that receives the distant cue to verify the task is solvable.
The curves, objects, and layout shifts must preserve valid label semantics. Do not use a
rotation of a street scene as an unqualified realistic-distribution test.

Freeze independent RNG streams for data generation, split selection, initialization,
sampling, augmentation, and bootstrapping. Default training seeds: `0,1,2` for comparisons;
add `3,4` for finalists and all their direct controls. Share dataset/split manifests across
models. Split before cropping/augmenting; related frames, subjects, scenes, or duplicates
stay in one group. In particular, ACDC out-of-fold stacking/calibration must split by patient,
never by slice or phase. Threshold fitting, class frequencies, calibration, feature selection,
and stacked-feature training never use final evaluation images or labels.

During development, evaluate clean validation and fixed validation stress sets: brightness
offsets, gamma changes, noise, object-size changes, and layout changes. Freeze severities
and a separate final stress manifest before selection. Add MRI-specific intensity bias/noise
stress cases for ACDC; color perturbations apply only to Kvasir. Masks change only under geometric
transforms, with newly uncovered pixels ignored. Report each severity, not a selected best.

## 5. Baselines, tuning, and experiment matrix

Required baselines:

- B0: majority class, train-derived row prior, and per-pixel color/intensity linear classifier.
- B1: current dense TM with patch 3 and no position information.
- B2: tuned current TM: patch `{3,5}` or OR pyramid `(1,2,4)` with patch 1 at the head;
  position `{off,8}` and existing class-feedback control as ablations.
- B3: coalesced dense TM sharing clauses, matched by total state bytes, not clauses/class.
- B4: compact U-Net trained from scratch on RGB (Kvasir) or grayscale (ACDC); widths `{16,32}`, four resolution levels,
  skip connections, AdamW, masked cross-entropy and an optional masked Dice term.
- B5: same U-Net on the exact Boolean input planes as the TM, plus a local CNN with the
  same receptive field as B1. These separate encoding limitations from context/learning.

A pretrained CNN is an optional external-data reference, in a separately labeled track.
Do not use it to claim equal training-data budgets. An optional hybrid frozen CNN encoder
plus TM head can diagnose representation limits but cannot substantiate a pure-TM claim.
Give B4 the same spatial post-processing options as the TM when evaluating H4/H5.

Use `n_states=100`, `compute_dtype=float32`, no dropout, and fixed image batch size as
initial TM controls. Tune clauses/class `{100,300,600}`, `T/clauses_per_class ∈ {0.1,0.2,0.5}`,
`s ∈ {3,10,20}`, and literal caps `{16,32,None}` using a seeded, bounded search. In the
coalesced model `T` must be tuned independently because the clause allocation differs.
Include a weighted/unweighted control. Do not run this entire Cartesian product.

For CNNs search learning rate `{1e-4,3e-4,1e-3}`, weight decay `{0,1e-4}`, widths above,
and Dice weight `{0,0.5}`. Share augmentations, image resolution, valid labels, and data
exposure. Use a common maximum training-time budget per trial; record actual epochs and
pixels processed. Tune each family adequately instead of forcing identical optimizers.

Search rule: at most 12 seeded configurations per required trainable family per development
dataset, one screening seed; repeat the best two on three seeds. A one-seed screen is
triage only, not evidence of superiority. Use the same validation criterion, maximum
time, and trial allowance for competing families; include failed runs and tuning cost.
For novel H1/H2 arms first use frozen B2 hyperparameters to isolate effects, then allow the
same limited retuning as their matched control. Keep a manifest of every proposed trial.

| Wave | Experiments | Required comparisons and deliverable |
|---|---|---|
| E0 | Tiny correctness fixtures, D0 baselines, feature/memory profiling at 32/64/128/256 pixels. | Evaluation audit, feasible run sizes, baseline variation and cost. |
| E1 | H1 order × commit size; sampling mixtures; H2 pooling/contrast. D0 and relevant D1 families. | Separate order, sampling, representation, and capacity effects. |
| E2 | B0–B5 on D2; top two E1 changes; H3 two-stage stack. | Real-data baseline table and out-of-fold stacking audit. |
| E3 | H4 and H5 on cached B2/H3/B4 unaries. H6/H7 only if their triggering diagnostics fail. | Spatial quality–cost comparison and error-category analysis. |
| E4 | Combine at most two supported components; D2 and D3, five seeds. Establish and tune B0–B5 and the selected recipes on ACDC train/validation before freezing its comparison. | Frozen finalist comparison, interaction/deletion ablations, final evaluation; separate training per dataset. |

For a combined model run baseline, A, B, A+B with the same seeds and budget tier; report
the interaction `score(A+B)-score(A)-score(B)+score(base)`. Also remove each component from
the final model. Keep H4/H5 refinement parameters fixed when asking whether H3 improved
unary evidence. Never silently increase clauses, image resolution, and data exposure
at the same time as introducing a mechanism.

## 6. Measurements and statistical decisions

Primary quality metric: dataset-level mIoU from the accumulated confusion matrix over
valid pixels. Report per-class IoU, macro Dice, foreground IoU/Dice on Kvasir-SEG, rare/small-object
recall, and pixel accuracy as a secondary metric. Include background in mIoU and give
foreground-only summaries separately. An absent class with zero union is excluded and
its absence is reported; a class with false positives has nonzero union and scores zero.
Report per-image distributions as well as aggregate scores. Use restored native-resolution
predictions for final real-data quality metrics; give resized-grid screening metrics separate
names so they are not mixed in a comparison.

For Kvasir also report mean per-image foreground Dice/IoU; pooled foreground IoU is a
different statistic. For ACDC report each patient's reconstructed 3D Dice for RV, myocardium,
and LV separately at ED and ES, plus the patient-macro foreground mean. Keep dataset-level
mIoU for the shared acceptance gate, with foreground Dice as a required ACDC guardrail
(mean decline no greater than 0.01). Add 3D HD95 in millimeters using native voxel spacing;
state the symmetric surface-distance convention. Both-empty classes are undefined/excluded
with counts, while exactly one empty segmentation scores zero Dice and has undefined HD95
with a failure count; never silently remove such failures from the report. Report phase
and pathology-group breakdowns. ACDC confidence intervals resample patients with both phases
and all slices together, never slices as independent cases.

For boundaries, compute per-class binary boundaries and match only within the same class.
Report F1 at tolerance 1 and 2 pixels at the declared evaluation resolution, plus a
resolution-normalized tolerance for cross-resolution comparisons. Exclude boundaries
whose evaluation neighborhoods intersect void/padding, and report remaining coverage.
Validate perfect masks, class swaps, shifted edges, empty masks, and ignore bands by hand.
Topology uses D1 connected-component/hole errors and clDice with a pinned skeletonizer.
Report missing/empty structures separately; avoid hiding undefined scores in averages.

Efficiency: end-to-end train time, time to a fixed validation quality target, preprocessing
and cross-fitting time, inference median/p95 latency, images/s and pixels/s at batch 1 and
a throughput batch, CPU peak RSS, GPU peak allocated/reserved bytes, checkpoint bytes,
automaton states, active literals, and included-rule lengths. For a stack count all stages;
for coalesced models include the weight matrix. Record original and resized resolution.
Boolean inputs do not imply bit-packed execution or 1-bit automata in this implementation.
For ACDC distinguish slices/s from complete ED/ES volumes/s and report end-to-end time
per patient, including NIfTI loading and native-grid reconstruction.

Benchmark on an otherwise idle declared device with fixed CPU threads, warmup, CUDA
synchronization around timed GPU work, and repeated measurements. Include extraction,
transfer, refinement, and tile stitching. Give cached-model-only timing separately. Do
not run competing timing jobs on the same GPU. OOM/timeout is a recorded outcome, not
a missing row. Hardware availability and timings must be discovered in E0; earlier
research programs' hardware descriptions are not a current resource guarantee.

Use two explicit budget views: fixed training wall time, and a state/memory-matched TM
comparison; also report accuracy versus inference latency across models. A CNN parameter
is not equivalent to an automaton state. A quality-first curve may exceed a budget, but
must state the cost. Efficiency claims need measured numbers, not operation counts alone.

Use paired training seeds and paired evaluation images/groups. Report mean ± sample SD
over seeds and a 95% paired hierarchical bootstrap interval for the score difference:
resample seed pairs, then image/group IDs with replacement, and recompute each metric
from selected predictions/confusion counts. Use 2,000 bootstrap draws, storing its seed.
Never treat neighboring pixels as independent observations or apply pixel-level McNemar
as if they were independent. Five seeds give limited precision; report inconclusive
intervals honestly. Validation screens are exploratory; only the frozen primary comparison
is confirmatory. Label secondary comparisons exploratory or correct for multiplicity.

Proposed final acceptance rules, fixed before E4:

- **Quality improvement:** at least +0.02 absolute mIoU versus tuned B2, paired interval
  lower bound above zero on Kvasir-SEG; confirmation of the sign on ACDC with its own tuned
  B2 and the foreground-Dice guardrail above. No mean class-aware boundary-F1 decline greater than 0.01,
  and investigate every per-class IoU decline greater than 0.03.
- **Efficiency improvement:** at least 2× end-to-end inference throughput or at least 40%
  peak-memory reduction, with the lower confidence bound of the paired mIoU difference
  above -0.01. State which resource improved and disclose the others.
- **CNN competitiveness:** report the actual gap and resource tradeoff. A claim of quality
  parity requires a preregistered 0.02 mIoU noninferiority margin whose paired lower bound
  exceeds -0.02, plus the boundary guardrail. Closing some of the gap is a separate claim.

Select **one** primary acceptance route and resource tier for each finalist before final
evaluation; do not choose quality or efficiency after inspecting the final results.
These thresholds express project priorities, not mathematical laws. They can be revised
after E0 cost/variance estimates, with rationale and a dated decision before candidate
selection. A domain-specific winner may remain an optional experiment even if it fails
the general-library gate. Never change a threshold to rescue a final result.

## 7. Memory and implementation feasibility

Before each new arm estimate and measure: `2FC` state elements, include caches, feedback
accumulators, encoded literals of order `2NF`, clause intermediates of order `NC`, and
vote maps of order `NK`. Use actual tensor dtypes and multiplicities from the implementation;
these orders alone are not peak-memory estimates. Preserve exact integer clause arithmetic;
do not switch to float16 merely to fit more literals without checking exactness bounds.

Start with the simple full-image reference at small resolution. If memory blocks D2/D3,
implement experiment-local tiled/gathered extraction before scaling up clause counts.
For local inference, retain a halo covering the effective receptive field and discard
halo outputs. For pyramids align pooling to the global origin, including padding and
rounding at each scale. For positions use global coordinates, never tile-local coordinates.
For a hierarchy include the full composed receptive field. Validate odd dimensions,
image corners, and seam pixels against full-image reference output.

Iterative spatial solvers need halo exchange each iteration or a global graph solve;
independent tile refinement is generally a different algorithm. For training, preserve
global pixel order and commit boundaries when testing numerical equivalence. If the
adapter changes order, compare it as an H1 arm, not as a transparent memory optimization.
Do not change an initialized model's recorded input shape to bypass its public contract.
Use a fixed-size adapter or experiment subclass with explicit geometry and parity checks.

## 8. Phases, budgets, and stopping rules

Budget units below are device-hours, summed across devices, including tuning, preprocessing
on the accelerator, out-of-fold fits, and failed runs. They are provisional caps, not
promises about runtime; re-estimate from a short E0 pilot. Wall time depends on available
hardware. CPU time and engineering time are logged separately.

| Phase | Suggested effort / compute cap | Gate |
|---|---|---|
| P0 / E0 | 2–3 working days; 4 device-hours | G0: masks, splits, metrics, training/eval semantics, recording, and feasible memory validated. |
| P1 / E1 | 3–5 days; 12 device-hours | G1: reject unsupported mechanisms; promote at most two cheap changes with three-seed validation evidence. |
| P2 / E2 | 5–8 days; 32 device-hours | G2: real-data baselines complete; determine whether learned context beats the strongest fixed-context control. |
| P3 / E3 | 3–6 days; 16 device-hours | G3: refinement/conditional mechanisms earn their complexity; select at most two final recipes. |
| P4 / E4 | 5–8 days; 48 device-hours | G4: five-seed controls, interactions, robustness, locked final evaluation, reproducible report. |

Total initial cap: 112 device-hours, roughly 4–6 working weeks of implementation and
analysis. Running every optional branch is not required. Start with G0 only; do not launch
the entire search unattended. These provisional caps must be re-estimated in E0 for the
selected datasets: ACDC patients contain many slices, so patient counts do not equal the
number of training examples. Include volume loading, restoration, and surface metrics in
cost estimates. Within each phase, a runner must refuse new jobs once the
remaining cap cannot accommodate the estimated run; save checkpoints and an incomplete
status rather than producing partial success rows. Use pilots to allocate the bounded
trial list, reduce scope if necessary, and document any revised budget before extending it.

G0 stops on invalid evaluation or infeasible memory until those issues are understood.
G1/G2 may conclude that ordinary tuning explains the apparent gap; report that result.
Drop a mechanism after its bounded screen and three-seed check show no practical signal
or a dominated quality–cost curve. H6 requires unresolved long-range/relational errors
after H3; H7 requires topology errors and a domain that justifies the prior. Failure of
these triggers removes their branches rather than delaying the whole program.

Do not inspect sealed final labels until all configs, checkpoints/selection procedures,
post-processing settings, seeds, metrics, and primary hypotheses are frozen in a decision
record. Evaluate each final checkpoint once on the sealed set. Choose epochs using
validation mIoU with deterministic tie-breaking, not final results. A failure after
unsealing is a result; further tuning needs a new development/final-evaluation protocol.

## 9. Reproducible deliverables and implementation backlog

Only this plan exists initially. The following tree is an implementation backlog, not a
claim that commands or scripts are already available:

```text
experiments/segmentation/
  PLAN.md
  code/{data,features,baselines,sampling,hierarchy,spatial,metrics,record,run,report}.py
  tests/                       # experiment-specific correctness checks
  configs/                     # fixed arm definitions and search manifests
  manifests/                   # splits, labels, source/dataset hashes
  results/                     # immutable per-run JSON records
  artifacts/                   # checkpoints, predictions, representative failure masks
  report/                      # generated tables, plots, and final report
  LIBRARY_GAPS.md
  DECISIONS.md                 # dated choices; link supporting run IDs
```

First implementation tasks, in dependency order:

1. Implement Kvasir image/mask pairing and ACDC NIfTI/ED/ES adapters with the exact roots
   in section 4. Write run-record validation, source/environment capture, split manifests, and an
   evaluator independent of the training loop. Define label/void conventions and metric
   fixtures before adding candidate models.
2. Wrap B0–B5 in one train/validate interface with `fit`, `predict_votes`, state save/load,
   and identical valid-pixel masks; add fixed-seed smoke configs and memory profiling.
3. Validate the identity dense adapter and implement H1/H2 plus the D1 mechanism datasets.
4. Add out-of-fold feature caching and H3. Implement H4/H5 against tiny analytical cases
   before applying them to cached real predictions.
5. Run the phase gates; implement H6/H7 only when justified. Generate final tables directly
   from admissible records and preserve negative findings.

Each run record includes: schema version, run/parent IDs, status and failure reason;
complete resolved config and argv; git SHA plus dirty diff/artifact hash; library/source
tree hash; Python/torch/CUDA/package versions and device/threads; dataset version/checksum,
label map, split-manifest hash; all RNG seeds; encoder/calibrator fit IDs; stage/fold
provenance; clauses/features/state dtypes; image/patch/commit sizes; sampled and accepted
feedback counts; per-epoch validation metrics; selected checkpoint and selection rule;
confusion matrices/per-image sufficient statistics; timings and memory; checkpoint and
prediction artifact hashes. Store undefined metrics as JSON `null` with a reason.
Interrupted/OOM runs must not be summarized as completed experiments.

Audit records should allow a reviewer to reconstruct a number, including selection and
preprocessing, rather than only rerun a final architecture. Keep per-image/group predictions
or sufficient counts for paired statistics. Store anonymized dataset identifiers, not
unnecessary copies of external images. Generate every measured table/figure from records;
link run IDs in decisions. Hand-authored numeric gates remain clearly labeled as proposals.

Required experiment checks: metric hand cases; source-hash guard; split/group disjointness;
mask alignment; JPEG binary decoding; ACDC patient/phase isolation and NIfTI geometry
round trips; native-grid volume reconstruction; `eval()` prediction; save/load and cache provenance; identity adapter
parity; no label access at inference; coarse out-of-fold exclusion; H4 energy monotonicity;
H5 direct-solve/stability check; tiling equivalence if implemented. Learning comparisons
belong in run records, not fragile CI accuracy assertions.

Final report: baseline and ablation tables, quality–time/memory curves, boundary/rare-class
breakdowns, shift results, calibration if used, clause explanations with full feature
provenance, fixed-ID qualitative successes and failures, negative results, limitations,
and an explicit recommendation for each candidate: reject, keep experimental, or propose
integration. Show raw TM decisions alongside refined decisions: a clause explanation of
the unary head is not an explanation of a spatially changed final label.

## 10. Library integration criteria

Prepare a concrete integration proposal only after G4 and the section 6 acceptance route.
It must identify the smallest reusable abstraction supported by the results: encoder,
sampling control, optional structured predictor, or separately named hierarchical model.
Avoid replacing defaults merely because a complex recipe won one dataset.

The proposal must include independent reproduction from saved configs/checkpoints, cost
and limitations, CPU/CUDA behavior where available, deterministic evaluation, serializable
encoder/calibration state, backward compatibility, interpretability behavior, and a minimal
example. Before a later library change ships, run the existing segmentation and package
tests, appropriate new behavior tests, lint, and documentation checks. No unvalidated
prototype enters the library as part of this planning task.

If no candidate clears the gate, the completed deliverable is a reproducible negative
report and a diagnosis of whether representation, spatial context, learning dynamics,
or compute was the binding limitation. That is preferable to exposing an unsupported API.

## 11. Related work and evidence boundaries

Sources checked on 2026-10-13. These establish mechanisms and relevant comparisons; they
do not establish that any proposed adaptation improves this library.

| Source | Use in this program | Limit of the evidence |
|---|---|---|
| [Convolutional Tsetlin Machine](https://arxiv.org/abs/1905.09688), Granmo et al. | Background on shared logical filters. | Image classification results do not establish dense segmentation quality. |
| [U-Net](https://arxiv.org/abs/1505.04597), Ronneberger et al. | Context/localization architecture and trained CNN reference. | Our stagewise TM stack has a different learning rule. |
| [CTM-UNet](https://doi.org/10.1109/ISTM67926.2025.00013), Liu and Abeyrathna; [authors' repository](https://github.com/yiliu-coding/CTM-UNet) | Related TM segmentation architecture; possible reproduction after an implementation audit. | Local PDF section IV-D and Table I concern binary sky versus background; do not present these as full multiclass CamVid results. |
| [Fully connected CRF](https://arxiv.org/abs/1210.5644), Krähenbühl and Koltun | Structured unary/pairwise inference reference. | Our initial local Potts solver is a simpler, different method. |
| [Anisotropic diffusion](https://authors.library.caltech.edu/records/1p8h5-5x870), Perona and Malik | Edge-dependent smoothing motivation. | Our anchored linear graph system differs from their nonlinear diffusion. |
| [GraphTM](https://arxiv.org/abs/2507.14874), Granmo et al. | Logical message passing and deeper relational rules. | Published graph/image-classification evidence is not evidence for our dense decoder. |
| [clDice](https://arxiv.org/abs/2003.07311), Shit et al. | Connectivity-aware evaluation of tubular structures. | A metric or differentiable CNN loss does not supply a TM feedback rule. |

The [local CTM-UNet PDF](../../source_documents/papers/03_segmentation/CTM-UNet_Integrating_Convolutional_Tsetlin_Machines_with_U-Net_Architecture_for_Semantic_Segmentation.pdf)
also describes its loss as monitoring rather than driving TM training (section III-E).
Audit which components actually learn before reproducing it. The authors' repository
currently describes work in progress and shows a classic binary U-Net script in its root;
do not assume a complete runnable CTM implementation is available from the README alone.
Avoid carrying forward older repository claims that there is “only one” relevant paper;
this targeted literature check is not an exhaustive systematic review.
