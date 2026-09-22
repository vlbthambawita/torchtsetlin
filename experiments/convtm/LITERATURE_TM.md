# LITERATURE_TM.md — every convolution-relevant Tsetlin machine approach

**Owner**: `tm-theorist` · **Phase**: P1 · **Date**: 2026-09-20
**Scope**: every approach in `source_documents/papers/**` and `source_documents/book_chapters/**` that
bears on convolution in Tsetlin machines, or on CIFAR-10 accuracy for TMs.

Claim tags per CHARTER: `[FACT: source]` citable, `[MEASURED: id]` a record in `results/` or
`experiments/mctm/results/`, `[HYPOTHESIS]` everything else. Untagged text reads as `[HYPOTHESIS]`.

---

## 0. The headline finding of this review

**PLAN.md §2's target ladder rests on a number that the primary literature does not support, and it
is 14.7 points too low.**

| PLAN.md says | What the primary sources say |
|---|---|
| "Published CTM best on CIFAR-10 is **60.7%**" (T0/T2) | 60.7% traces to a **2020 University of Agder Master's thesis** on a **modified** CIFAR-10, cited at third hand. `[FACT: Grønningsæter et al. 2024 §V + ref 14]` |
| "the honest single-model bar" = 60.7% | The honest single-model bar is **75.4%** — 5×5 Colour Thermometers CTM, 64 000 clauses `[FACT: Grønningsæter et al. 2024, Table IV]`; with Drop Clause, **75.1%** at 60 000 clauses `[FACT: Sharma et al. AAAI 2023, Table 2]`; **vanilla CTM alone reaches 69.3%** at that budget `[FACT: Sharma et al. AAAI 2023, Table 2, p=0 column]` |
| TM SOTA **82.8%** = "an ensemble of specialised TMs" | Confirmed, and it is larger than implied: **22 TM Specialists, each 64 000 clauses, 250 epochs** `[FACT: Grønningsæter et al. 2024, Tables III–IV]` |

Consequences for the programme, stated without choosing a method (C3):

1. **T0 as written ("reproduce vanilla CTM at 55–61%") is not a reproduction target** — it is a
   reproduction of an unpublished thesis on a dataset we do not have. A *checkable* T0 exists:
   Table IV of the Toolbox paper gives 22 specialist × 6 clause-budget cells, the lowest of which
   (2 000 clauses) costs ~0.3–0.6 GPU-h/seed on this machine `[HYPOTHESIS: cost model, §6]`.
2. **T2 ("beat the best single-model TM") means beating 75.4%, not 60.7%.** That is a materially
   harder objective and the orchestrator should re-baseline the ladder before P3 commits.
3. The 82.8% composite, reproduced faithfully, is ≈ 4 600 GPU-h at 3 seeds — **15–20× the whole
   programme budget** `[HYPOTHESIS: §6]`. Its 2023 ancestor at **75.1%** (4 specialists × 2 000
   clauses × 100 epochs) costs ≈ **5 GPU-h at 3 seeds** and *is* reproducible here
   `[FACT: Granmo 2023, Table 1–2]` `[HYPOTHESIS: cost model]`.

---

## 1. The table

Columns: method · paper (file + id) · reported CIFAR-10 accuracy · the exact configuration behind
that number · what is mechanistically novel · can `torchtsetlin` express it today ·
estimated re-implementation effort.

`—` in the accuracy column means *the paper reports no CIFAR-10 number*; what it does report is
stated. Effort is engineer-days of new code in `experiments/convtm/code/`, excluding GPU time.

| # | Method | Paper (file · id) | CIFAR-10 accuracy | Exact configuration behind that number | Mechanistically novel | Expressible in torchtsetlin today? | Re-impl. effort |
|---|---|---|---|---|---|---|---|
| **A1** | **Convolutional TM (CTM)** — the base construction | `01_foundations/2019_The_Convolutional_Tsetlin_Machine` · arXiv:1905.09688 | **— none.** Reports MNIST 99.40 (peak) / 99.33 ± 0.0 (mean), K-MNIST 96.31/96.08, F-MNIST 91.50/91.18, 2D Noisy XOR 100.0 `[FACT: §4, Table 4]`. CIFAR-10 is named only as **future work** `[FACT: §5]` | 8 000 clauses **per class**, T=10 000, s=5.0 (MNIST) / 10.0 (K-, F-MNIST), window W=10, Z=1 bit-plane, adaptive Gaussian thresholding (block 11, C=2), integer clause weighting on, 250 epochs, results = mean of last 100 epochs, DGX-2 `[FACT: Table 1 + §4]` | Clause as convolution filter; clause output = **OR over patch matches**; feedback contrasts against **one uniformly random matching patch**; patch coordinates appended as thermometer bits | **Yes** — `ConvTsetlinMachine` is a direct implementation (`src/torchtsetlin/models/conv.py:237`). Differences in §5 below | 0 d (exists) |
| **A2** | **Integer-weighted clauses** (used by every later CIFAR-10 result) | `01_foundations/2019_The_Weighted_Tsetlin_Machine` · arXiv:1911.12607; and CTM §2.5 | — (MNIST/IMDb/Connect-4 only) | WTM: w ← w·(1+γ) on Type I, w ← w/(1+γ) on Type II, **real-valued**, γ a learning rate, weights init 1.0 `[FACT: §3.4, Eqs 12–17]` | One clause replaces many; compresses the clause pool | **Partially.** `weighted=True` implements the **integer additive** variant (±1 per Type Ia/II event, clamped ≥0) — `models/classifier.py:173-187`. The 2019 **multiplicative real-valued** rule is *not* expressible → `LG-006` | 0 d integer; 1 d real-valued (local subclass) |
| **A3** | **Coalesced TM (CoTM)** — shared clause pool, per-output integer weights | `01_foundations/2021_Coalesced_Multi_Output_TMs` · arXiv:2108.07594 | **— no accuracy number.** CIFAR-10 appears only as an *imbalance-robustness* study reporting class-wise F1 `[FACT: §4, Table 7]`. Accuracy is reported for MNIST/F-MNIST/K-MNIST | F-MNIST 71.99 → 89.66 at 50 clauses/class (22 Kb); CoTM ≈ TM above 1K clauses/class; 3× faster to peak on MNIST at 8K `[FACT: abstract]` | Clause sharing across outputs; weight sign selects polarity per output; SSL weight learning | **Yes** — `ConvCoalescedTsetlinMachine` (`models/conv.py:278`). Weight update is additive integer, unclamped, sign-as-polarity (`models/coalesced.py:160-181`) | 0 d (exists) |
| **A4** | **CoTM on CIFAR-10 (the best-specified CoTM number anywhere in this bibliography)** | `01_foundations/2025_The_TM_Goes_Deep_Graphs` · arXiv:2507.14874, Table 3 (as the GraphTM's baseline) | **66.42 ± 0.19** `[FACT: Table 3]` | 80 000 clauses, **8×8** convolution window, 30 epochs, T=15 000, s=20.0, **max included literals 32**, adaptive Gaussian thresholding only, accuracy = mean of final 5 epochs `[FACT: Table 16 + §3.2]` | — (it is the control) | **Yes**, exactly — this is `ConvCoalescedTsetlinMachine(n_outputs=10, n_clauses=80000, T=15000, s=20, patch_size=8, max_included_literals=32)` | 0 d |
| **A5** | **Graph TM (GraphTM)** — hypervector symbols + message passing | `01_foundations/2025_The_TM_Goes_Deep_Graphs` · arXiv:2507.14874 | **70.28 ± 0.17** `[FACT: Table 3]` | 80 000 clauses, 2 500 symbols, 8×8 window, hypervector size 128, **depth = 1**, 30 epochs, T=15 000, s=20.0, budget 32, input = adaptive Gaussian thresholding **and** 8-bin colour thermometer as two views `[FACT: Table 16, §3.2]` | Deep/nested clauses through message passing; arbitrary graph input; multi-view input in one model | **No.** Requires hypervector symbol encoding + message passing; `HypervectorEncoder` is for discrete tokens, not this. → `LG-009` | 8–15 d |
| **A6** | **Drop Clause (DC)** — clause dropout per epoch | `01_foundations/2021_Drop_Clause...` arXiv:2105.14506 and `2023_Drop_Clause_AAAI_camera_ready` | **75.1 ± 0.4** at p=0.5; **69.3 at p=0** (vanilla CTM), 70.5 (p=.1), 73.2 (p=.25), 72.6 (p=.75) `[FACT: AAAI 2023, Table 2]` | 60 000 clauses, T=48 000, s=10.0, adaptive Gaussian thresholding, CTM. **Convolution window size, epochs and the clause/class convention are not stated** `[FACT: AAAI 2023, "Datasets" §]` — recorded as an omission (§4) | Drop a random 1−p subset of clauses for a whole epoch; +5.8 points and −47% training time on CIFAR-10 `[FACT: §"Enhanced..."]` | **Yes** — `drop_clause_p=0.5, drop_granularity="epoch"` + `resample_dropout()` once per epoch (`models/base.py:388-400`) | 0.5 d (epoch-granularity plumbing in the harness) |
| **A7** | **Clause-size constraint (CSC-TM)** | `01_foundations/2023_Building_Concise_Logical_Patterns` · arXiv:2301.08190 | **— CIFAR-2 only.** 94.24 at budget 64 (34.1 literals used) vs 94.18 unconstrained (60.4 literals) `[FACT: Table 3]` | CIFAR-2: 8 000 clauses, T=6 000, s=10.0 `[FACT: fn. 6]`. MNIST w/conv: 99.33 at budget 16 vs 99.28 unconstrained | Oversized clauses receive only forgetting: **Type I is modified, Type II is unchanged** `[FACT: §2]` | **Yes**, faithfully (`functional.apply_feedback`, `max_included_literals`, `functional.py:232-242`). **CORRECTED 2026-09-21**: it *does* bind — at convergence, budget 32 gives 32.14 ± 0.02 literals batched and 31.89 ± 0.01 sequential `[MEASURED: ctm-small-T80-b32, ctm-small-seq-b32]`. The "does not bind" entry came from a budget-8, few-epoch probe. **This is the programme's largest own-measured effect: +9.46 pp batched, +8.75 pp sequential on CIFAR-10, against the paper's own +0.06 on CIFAR-2** — see THEORY.md §5.5 | 0 d, **done** |
| **A8** | **TM Composites** — confidence-normalised plug-and-play ensemble | `02_image_classification/2023_TMComposites` · arXiv:2309.04801 | **75.1** (composite of 4) `[FACT: Table 2]`. Members: HOG 63.5, 4×4 Colour Therm. 62.7, 3×3 Colour Therm. 62.1, 10×10 Thresholding 57.0 | **2 000 weighted clauses each**, budget 32 literals, 100 epochs; 10×10 Thresholding T=500 s=10.0; 3×3/4×4 Colour Therm. T=1 500 s=2.5; HOG (32×32, unweighted) T=50 s=10.0 `[FACT: Table 1]`. §4 prose says "8 000 weighted clauses **per class**" for the *confidence* study — the clause convention is ambiguous (§4) | α_t = max−min class sum **over the evaluated set**, then ŷ = argmax_i Σ_t c^i_{t}/α_t `[FACT: Eqs 6–8]`. Note this normalisation is **transductive** | **Yes, in harness code** — it is inference-only aggregation over independently trained models. No library change needed | 1 d |
| **A9** | **Optimised Toolbox / TM Composites at scale — current TM SOTA** | `02_image_classification/2024_Optimized_Toolbox...` · arXiv:2406.00704 | **82.8 ± 0.01** (composite) `[FACT: Table IV]`. Best **single** specialist **75.4 ± 0.09**. Composite at 32k = 82.7, 16k = 82.2, 8k = 81.5, 4k = 80.6, 2k = 79.5 | **22 specialists × 64 000 clauses × 250 epochs**, each specialist's (window, T, s, weighted) from an Optuna TPE search run at 2 000 clauses and **reused unchanged** at all budgets `[FACT: Tables II–IV]`. "Augmented" = training set doubled with horizontal flips (100 000 images, same epoch count — **not a matched data budget**) | 7 Booleanizations: Canny, HOG, adaptive Gaussian, adaptive mean, Otsu, colour thermometers, **adaptive (multilevel) colour thermometers** | **Partially.** `AdaptiveThresholdEncoder` (gaussian/mean) and `ColorThermometerEncoder` exist (`data/encoders.py:268,314`). Otsu, Canny, HOG and adaptive multilevel thermometers do **not** → `LG-008` (harness-local is fine) | 2 d encoders + 1 d composite |
| **A10** | **Uncertainty quantification on CIFAR-10** | `01_foundations/2025_Uncertainty_Quantification_in_the_TM` · arXiv:2507.04175 | Cites 82.8 as SOTA; contributes no new accuracy `[FACT: §3.2]` | — | Class-sum → calibrated probability; clause-count-vs-difficulty analysis | Yes (`predict_proba`, `confidence`, `functional.confidence_from_votes`) | 0 d |
| **A11** | **Convolutional Regression TM (C-RTM)** | `03_segmentation/3468891.3468901` · ICMLT 2021 | — (72 synthetic image-regression datasets) | — | Confirms the same patch aggregation: **c_j = ⋁_{b=1..B} c_j^b** `[FACT: §3.1, Eq. 4]`. Establishes that **no published TM uses anything but OR pooling** | Yes (`ConvRegressionTsetlinMachine`) | 0 d |
| **A12** | **CTM-UNet** — stacked CTM blocks with **local** max-pooling | `03_segmentation/CTM-UNet...` · ISTM 2025 | — (CamVid segmentation, mIoU) | 3×3 patches, clauses per block ≈ 4× channels: [256, 512, 1024, 2048], T=50, 300 epochs, 8-bit per-channel binarisation (H×W×3 → H×W×24), Adam + 1×1 conv fusion (a **hybrid**, not a pure TM) | **The only published construction that keeps clause outputs spatially resolved**: V^(l) = CTM2D(B^(l)) is a vote *map*, then `MaxPool`, re-binarise, next block `[FACT: Algorithm 1]` | **Partially.** The per-patch match tensor exists internally (`models/conv.py:146`) but has **no public accessor** → `LG-005`. Reachable today only via `functional.clause_outputs` + the private `_encode` | 3–5 d |
| **A13** | **Multi-task convolutional TM over RGB** | `02_image_classification/2025_Transparent_Logic_...RGB` · arXiv:2510.01906 | **— none.** MNIST 98.5%, CelebA 86.56 F1 (vs ResNet50 88.07) `[FACT: abstract]` | — | Local interpretation + global class representation for convolutional clauses (the clause↔pixel mapping problem) | Yes — `ConvCoalescedTsetlinMachine(multi_label=True)` + `clause_patch` / `clause_region` (`models/conv.py:190,203`) | 1 d (interpretation only) |
| **A14** | **Hyperdimensional vectors for TMs (HVTM)** | `02_image_classification/2024_Effects_of_Hyperdimensional_Vectors` · arXiv:2406.02648 | **— none** (MNIST, IMDb, TREC, chemical structures) | — | Symbols → sparse binary hypervectors, bundled; "hyper literals"/"hyper clauses" | **Partially** — `HypervectorEncoder` handles discrete tokens, not image patches | 4–6 d |
| **A15** | **Knowledge distillation into TMs** | `02_image_classification/2025_Knowledge_Distillation_in_TMs` · arXiv:2504.01798 (MSc thesis) | **— none** (MNIST + text) | — | Clause-transfer initialisation of the student from a ranked teacher; teacher class-probability targets | **Partially** — clause transfer = copying `ta_state` rows (a buffer, fully accessible). Soft targets need a new feedback policy (a local `_select_feedback` override) | 3 d |
| **A16** | **Sparse TM (active literals)** | `01_foundations/2024_The_Sparse_TM` · arXiv:2405.02375 | **— none** (NLP) | — | Clauses start empty; literals materialise on demand; lower-bound state removes a TA | **No** — changes the TA storage model | 6–10 d |
| **A17** | **Contracting TM (absorbing automata)** | `01_foundations/2023_Contracting_TM_with_Absorbing_Automata` · arXiv:2310.11481 | **— none** (NLP) | — | Absorbing Include/Exclude states; three action lists; the model contracts | **No** — changes the TA state machine | 6–10 d |
| **A18** | **Fuzzy-Pattern TM (FPTM)** | `01_foundations/2025_Fuzzy_Pattern_TM` · arXiv:2508.08350 | **— none.** F-MNIST 92.18 (2 clauses), 93.19 (20), **94.68 (8 000)** vs Composite TM 93.00 (8 000); IMDb 90.15 with **one clause per class** `[FACT: abstract]` | — | **Replaces the all-or-nothing AND with a graded match**: a clause still votes, at reduced strength, when some literals fail. Claims ~400× clause reduction on F-MNIST | **Partially.** `functional.clause_violations` already returns the *count* of failing literals (`functional.py:52`), so a graded clause output is computable; the **feedback rule** would need a local override | 4–6 d |
| **A19** | **Pre-sorted TM (genetic K-medoid)** | `02_image_classification/2024_Pre_Sorted_TM` · arXiv:2403.09680 | **— none** ("MNIST-level" problems: up to +10% accuracy, ~383× training-time reduction) | — | Cluster the data first, align K independent TMs by maximising Hamming distance | **Yes, in harness code** (it is a data-routing wrapper around ordinary TMs) | 2–3 d |
| **A20** | **CTM over frozen CNN features** (transfer learning + CTM) | `04_medical_clinical/2026_Enhanced_Cervical_Cancer_Classification_Convolutional_TM` · DiscoverAI 2026 | — (cervical cytology) | — | InceptionV3 features → Booleanize → CTM. The only published "good representation + TM" pipeline in this bibliography | **Yes, in harness code** (an encoder in front of an existing model) | 1 d |
| **A21** | **Hardware CTM/ConvCoTM accelerators** | `05_hardware_vision_accelerators/*` (6 papers) | **No TM CIFAR-10 accuracy of their own.** All cite 82.8% as the state of the art; the 86% / 91.3% / 91.7% figures in these papers are **BNN/CNN** accelerators, not TMs `[FACT: 2501.19347 §II]` | 65 nm ConvCoTM ASIC, 2.7 mm², MNIST | Energy/latency, not accuracy | n/a (out of scope, PLAN §1.3) | — |
| **A22** | **This repo's own stacked CTM** | `experiments/mctm/report2` | `final_test_acc` **41.80 ± 0.98** (3 seeds) `[MEASURED: experiments/mctm/results/mctm-random-calib_seed{0,1,2}]`; PLAN quotes 42.6 as `best_test_acc` | 640 clauses total, therm-4, 2 layers (random layer 1 + 20% firing-rate calibration), 30 epochs, batch 50, 1 265 664 automata | Label-free per-clause firing-rate calibration over a **random** layer 1 | Yes (it was built on this library) | 0 d (re-run under the new protocol) |

### Reference points that are *not* rows above but belong in the ladder

| Claim | Value | Source |
|---|---|---|
| Vanilla CTM, 60 000 clauses, adaptive Gaussian | **69.3%** | `[FACT: Sharma et al., AAAI 2023, Table 2, p=0]` |
| Best single-model TM on CIFAR-10 | **75.4 ± 0.09%** (5×5 Colour Thermometers, 64 000 clauses, 250 ep) | `[FACT: Grønningsæter et al. 2024, Table IV]` |
| Best TM on CIFAR-10 of any kind | **82.8 ± 0.01%** (22-specialist composite, 64 000 clauses each) | `[FACT: Grønningsæter et al. 2024, Table IV]` |
| "60.7%" | CTM on a **modified** CIFAR-10, UiA MSc thesis 2020, cited at third hand; primary document not reachable from this machine | `[FACT: Grønningsæter et al. 2024 §V, ref. 14]` |
| A single conv layer here, 640 clauses, therm-4, 30 ep | **35.14 ± 0.74%** | `[MEASURED: experiments/mctm/results/single-matched_seed{0,1,2}]` |

---

## 2. Grouped by *what they change*

This grouping, not the table, is what the team should reason over.

### 2.1 The clause / feedback mechanism
**Nobody has changed it for images since 2019.**
- The CTM's three mechanism choices — patch OR, one random matching patch per feedback event,
  thermometer position bits — are stated once in 2019 `[FACT: arXiv:1905.09688 §3]`, restated
  verbatim in the 2021 C-RTM `[FACT: Eq. 4]`, in the 2023 Drop Clause paper `[FACT: "TM and CTM" §]`,
  and in Chapter 4 of the book `[FACT: Ch. 4 §4.3–4.4]`. No variant is proposed anywhere in this
  bibliography.
- The only mechanism edits that *are* published are orthogonal to convolution:
  **Drop Clause** (A6, a regulariser, +5.8 pts on CIFAR-10), **clause-size constraint** (A7,
  published as interpretability/power at ≈0 pts on images — **but measured here at +9.46 pp on
  convolutional CIFAR-10 and +8.75 pp under the exact sequential algorithm**
  `[MEASURED: ctm-small-T80-{unc,b32}, ctm-small-seq-{unc,b32}]`, THEORY.md §5.5),
  **integer weighting** (A2, compression),
  **absorbing / sparse automata** (A16–A17, memory), and **fuzzy clause evaluation**
  (A18, no image-convolution evaluation).
- `[HYPOTHESIS]` The convolution mechanism of the CTM is therefore the least-explored large surface
  in TM image classification, and it is the surface this programme is aimed at. Falsifier: a paper
  outside this bibliography that varies the patch aggregation. I searched the local corpus for
  `pooling|max-pool|stride|dilat|multi-scale|receptive field|two-layer|deeper CTM|hierarch` and found
  no TM image paper that does.

### 2.2 The patch / pooling structure
- **OR pooling is universal** (A1, A11, A12 inner blocks). `[FACT: 1905.09688 §3; 3468891 Eq. 4;
  Book Ch. 4 §4.6]`
- **Position encoding** is thermometer bits on the patch coordinates, giving a clause a rectangular
  region constraint `[FACT: 1905.09688 §3; Book Ch. 4 §4.5]`. It is on by default in the library and
  the papers, and **it is not free**: `[MEASURED: experiments/mctm/results/mctm-nopos_seed{0,1,2}]`
  a 2-layer stack with position encoding **off** collapses to chance (10.00 ± 0.02%), while the
  same stack with it on reaches 12.56%; on translation-invariant toys it is the opposite
  (PLAN §3.2). Nobody has measured what it costs a *single* layer on CIFAR-10.
- **Window size is the most-varied and least-explained hyperparameter in the literature**: 10×10
  (CTM 2019, Drop Clause, Composites Thresholding), 8×8 (GraphTM/CoTM), 5×5 (Toolbox's best
  specialist), 3×3 (CTM-UNet, Composites Colour Therm.), 32×32 = whole image (HOG specialist).
  The Toolbox searched it over `[1, 32]` with Optuna and reports different optima per Booleanization
  `[FACT: Tables II–III]` — but never at a matched automata budget.
- **Local pooling exists in exactly one paper** (A12, CTM-UNet: per-position vote maps → MaxPool →
  re-binarise), and that paper is a backprop hybrid on a segmentation task.

### 2.3 The Booleanization
This is where all the measured progress is. At a *fixed* 2 000 clauses, the Toolbox's specialists
span **48.7% (10×10 Canny) to 65.4% (4×4 Augmented Colour Thermometers)** — a **16.7-point** range
from the encoding alone `[FACT: Table IV]`. For comparison, Drop Clause buys 5.8 points and the
clause-size constraint buys ~0 **as published** — `[MEASURED: ctm-small-T80-{unc,b32}_seed{0,1,2}]`
it buys **+9.46 pp** in this programme's own 2 000-clause 4×4 configuration, which is the largest
own-measured effect here and the reason A7 moved from "control" to "default on every arm"
(DR-001 Decision 2, DR-003 Decision 1, DR-004 Decision 2).
- Ordering at 2 000 / 64 000 clauses `[FACT: Table IV]`: Colour Thermometers 64.5 / **75.4** >
  HOG 64.2 / 67.5 (saturates) > Adaptive Colour Thermometers 62.0 / 73.3 > Adaptive Gaussian 57.9 /
  70.2 ≈ Adaptive Mean 57.8 / 68.4 > Otsu 56.9 / 65.0 > Canny 48.7 / 60.6.
- **HOG saturates and the thermometers do not.** HOG goes 64.2 → 67.5 from 2k to 64k and is flat
  from 32k; 5×5 Colour Thermometers goes 64.5 → 75.4 and is **still climbing at 64 000 clauses**
  `[FACT: Table IV]`. `[HYPOTHESIS]` The clause-scaling curve saturates at the information ceiling
  of the Booleanization, not at a ceiling of the clause mechanism. Falsifier: a Booleanization whose
  curve flattens while a CNN on the same bits keeps improving.
- The 2019 CTM used **1 bit per pixel** (adaptive Gaussian on greyscale) `[FACT: Table 1, |Z|=1]`.
  Every CIFAR-10 result since uses ≥3 planes and usually 24 (3 channels × 8 thermometer levels).

### 2.4 The ensemble structure
- **TM Composites** (A8/A9) is a *post-hoc, inference-only* ensemble: independently trained TMs,
  class sums range-normalised per model, summed, argmax `[FACT: 2309.04801 Eqs 6–8]`. No joint
  training, no fine-tuning.
- Gain over the best member: **75.1 vs 63.5 = +11.6** at 2 000 clauses `[FACT: 2309.04801 Table 2]`;
  **82.8 vs 75.4 = +7.4** at 64 000 clauses `[FACT: 2406.00704 Table IV]`. The gain *shrinks* as the
  members get bigger.
- **Protocol caveat**: α_t in Eq. 7 is the max−min class sum over the *evaluated input set* X, so the
  normaliser is computed on the test set. `[HYPOTHESIS]` This is transductive and mildly leaky; our
  reproduction must fix α_t on validation and report both. Falsifier: reproduce with α_t from
  validation and show the composite number is unchanged within the seed band.
- **GraphTM's CIFAR-10 gain is an ensemble-of-views effect, by the authors' own words**: "This
  improvement is likely due to the GraphTM's ability to incorporate multiple views of each image …
  the CoTM was trained using only adaptive Gaussian thresholding" `[FACT: 2507.14874 §3.2]`, and its
  CIFAR-10 run uses **depth = 1** `[FACT: Table 16]`. The 3.86-point gap is therefore *not*
  evidence that depth helps on CIFAR-10.

### 2.5 The training scheme
- **Epoch counts are large**: 250 (CTM 2019, Toolbox), 100 (Composites 2023), 30 (GraphTM/CoTM).
- **Reported accuracy is usually an average over the tail of training, not a validation-selected
  point**: "mean of the last 100 of 250 epochs" `[FACT: 1905.09688 §4]`, "mean of the last 25 epochs"
  `[FACT: 2406.00704 §IV]`, "final five epochs" `[FACT: 2507.14874 §3.2]`, "peak … averaged over 100
  runs" `[FACT: Drop Clause AAAI]`. **None of them uses a validation split.** Under PLAN §7.1
  (validation-selected, test touched once) our numbers are expected to be *slightly lower* than the
  same configuration reported in a paper. `[HYPOTHESIS]` The bias is ≤1 point for a converged CTM;
  it must be measured once, at G3, by also logging the paper's own statistic.
- **Hyperparameters are transferred across budgets without re-tuning**: the Toolbox searched at
  2 000 clauses and reused those (T, s, window) at 4k–64k `[FACT: §IV-B]`, scaling only T
  ("we change the clause size and scale the feedback threshold (T) thereafter").
- **Augmentation is nearly absent**: the only instance is the Toolbox's static horizontal-flip
  doubling, and **it is not budget-matched** (100 000 vs 50 000 images at the same epoch count). Its
  effect changes sign by Booleanization at 64k clauses: Canny +1.8, Adaptive Mean +1.8, HOG +1.0,
  Otsu +0.7, Adaptive Gaussian +0.1, **5×5 Colour Thermometers −1.3, 5×5 Adaptive Colour
  Thermometers −0.4** `[FACT: Table IV]`, with variance rising 5–15× for the augmented thermometer
  arms.

---

## 3. What the literature does *not* report (omissions are findings)

1. **The Drop Clause paper gives no convolution window and no epoch count** for its CIFAR-10 runs
   `[FACT: AAAI 2023, "Datasets"]`. The headline 75.1% is therefore not reproducible from the paper
   alone. This is the clearest instance of the rule in this round's brief: an accuracy without its
   configuration.
2. **"Clauses" is ambiguous throughout.** Granmo 2023 Table 1 says "2K weighted clauses" while §4 of
   the same paper says "8 000 weighted clauses per class" `[FACT: 2309.04801 Table 1 vs §4]`. The
   CTM 2019 paper is explicit ("#Class Clauses", 8 000 **per class**) `[FACT: Table 1]`. A factor of
   10 in the clause budget is the difference between a 2-hour and a 20-hour run, and between a fair
   and an unfair budget match. **Every arm we register must state clauses *total*.**
3. **No TM image paper reports a validation split**, a seed count, or a paired significance test.
   Spreads are sample variance over training epochs, not over seeds.
4. **No published matched-budget comparison exists** between any two CTM variants: Drop Clause
   compares p values at fixed clauses (fair), but GraphTM vs CoTM differs in input views *and*
   architecture, and the Composites' member-vs-composite comparison differs by 4× in total clauses.
5. **Nobody has published a CTM ablation of the OR pool, the random-patch rule, or position encoding
   on CIFAR-10.** `[MEASURED: experiments/mctm/results/mctm-nopos_*, mctm-nopool_*, mctm-pool4_*]`
   is, as far as this bibliography goes, the only such data that exists — and it is for a 2-layer
   stack at 640 clauses.
6. **No TM paper measures sample efficiency against a CNN** under one protocol, despite the claim
   being common.

---

## 4. Clause-budget conventions — the thing that will break our comparisons

| Source | What "clauses" means | Evidence |
|---|---|---|
| CTM 2019 | **per class** (Table 1 row is "#Class Clauses"); 8 000 → 80 000 total | `[FACT: Table 1, Table 3]` |
| CSC-TM 2023 | "8000 clauses per class" | `[FACT: fn. 6]` |
| Composites 2023 | Table 1 unqualified ("2K weighted clauses"); §4 says "per class" | `[FACT: Table 1 vs §4]` — **ambiguous** |
| Toolbox 2024 | unqualified ("Clauses 2 000 … 64 000") | `[FACT: Tables II–IV]` — **ambiguous** |
| GraphTM 2025 | unqualified; CoTM shares one pool, so 80 000 is almost certainly **total** | `[FACT: Table 16]` |
| `torchtsetlin` | `TsetlinMachine(n_clauses=…)` is **per class**; `CoalescedTsetlinMachine(n_clauses=…)` is **total** | `models/classifier.py:82`, `models/coalesced.py` |

`[HYPOTHESIS]` The Toolbox/Composites numbers are **total** (they use the CAIR `tmu` coalesced
classifier). Falsifier: run both conventions at 2 000 and see which matches Table IV's 2 000 column
(79.5% composite / 64.5% for 5×5 colour thermometers). **This is a 30-minute experiment and it must
be done before any budget-matched claim is made.**

---

## 5. Where `torchtsetlin` differs from the papers (our re-implementations inherit these)

Full derivation in `THEORY.md` §7. Summary:

| # | Difference | Library | Papers | Consequence |
|---|---|---|---|---|
| D1 | Feedback granularity | `feedback_mode="batch"` by default: all events of a mini-batch aggregated, applied once (`models/base.py:308-321`) | Strictly sequential, one example at a time `[FACT: Book Ch. 4 §4.4 step 5 "Goto 1"]` | Fidelity degrades with batch size; 10–50 tracks sequential, 200 does not (PLAN §3.2). **Arms are only comparable at equal batch size.** |
| D2 | Clause-size budget | Type Ia → Type Ib for oversized clauses; **Type II ungated** (`functional.py:232-242` vs `253-257`) | Identical — CSC-TM changes Type I only `[FACT: 2301.08190 §2]`, and names the resulting boundary literals itself | The library is *faithful*. **CORRECTED 2026-09-21**: the budget binds at convergence (32.14 ± 0.02 at budget 32, mode-independent); the ungated Type II produces a residual *fringe* of literals at exactly the include boundary, which is CSC-TM's own explanation of budget overshoot and is, per THEORY.md §5.5.2 C5.3, what a tight budget strips first |
| D3 | Clause weights | Additive integer ±1, clamped ≥0 (`classifier.py:187`) | 2019 WTM: multiplicative real-valued `[FACT: Eqs 12–17]`; CTM 2019 §2.5 and `tmu`: additive integer | Matches what the CIFAR-10 SOTA actually used; the 2019 rule is unavailable. LG-006 |
| D4 | Random patch draw | Drawn **per feedback event**, independently for Type Ia and Type II (`conv.py:161-176`) | "one of the patches, randomly selected among the patches that made the clause evaluate to 1" `[FACT: 1905.09688 §3]` — silent on sharing | Deliberate and correct for coalesced models, where one clause can receive both kinds in one update (`conv.py:166-168`). A faithfulness question only for coalesced arms. |
| D5 | TA initialisation | `init="boundary"` by default: every TA at N−1 (all excluded, every clause empty) | "A TM is initialized by **randomly** setting the states" `[FACT: 1911.12607 §2.3]`; `init="random"` reproduces the N−1/N variant | Empty clauses evaluate **True** while learning, so at epoch 0 every clause fires. Cheap to test both. |
| D6 | Per-patch clause outputs | Computed (`conv.py:140-146`) but not exposed | n/a | Blocks count-pooling, spatial diagnostics and CTM-UNet-style stacking from public API. LG-005 |
| D7 | Empty-clause semantics | Keyed off `self.training` (`base.py:230-238, 270`) | Same `[FACT: 1905.09688 §2.2]` | Faithful — but `model.eval()` is mandatory on every evaluation path (CHARTER failure mode 1). |
| D8 | Chunk budget | `max_chunk_elements=2**27` default; conv cost model is `3PC + 2C·2F + P·2F` per example (`conv.py:106-111`) | n/a | At 64 000 clauses / 10×10 / 24 planes the per-example budget is ~7.3×10⁸ elements → **chunk size 1**, i.e. a 50× kernel-launch penalty at batch 50. The constructor arg exists; the **default** must be raised for large arms. LG-004 |

---

## 6. Cost model — what the literature's configurations actually cost here

Calibrated on PLAN §3.4 (`640 clauses, 4×4, therm-4, 50 000 images → 4.6 s/epoch`), giving
k = 3.45×10⁻¹³ s per clause-patch-literal MAC. Cross-check: predicts **11.6 s/epoch** for
640 clauses at 9×9 where §3.4 measured **11.2** `[HYPOTHESIS: model, to be replaced by P0's
measurement]`. Cost ≈ k · N · P · 2F · C · epochs, with P = (33−k)² and F = Z·k² + 2(32−k).

| Configuration (45 000 train) | C | P | F | s/epoch | h/seed | h × 3 seeds |
|---|---|---|---|---|---|---|
| `ctm-vanilla` small: 640 cl, 4×4, therm-4, 30 ep | 640 | 841 | 248 | 4.1 | 0.03 | 0.1 |
| `ctm-vanilla` mid: 8 000 cl, 4×4, therm-4, 30 ep | 8 000 | 841 | 248 | 52 | 0.43 | 1.3 |
| **CoTM as GraphTM's baseline: 80 000 cl, 8×8, adaptive, 30 ep** | 80 000 | 625 | 240 | 372 | **3.1** | **9.3** |
| same, scaled to 8 000 cl | 8 000 | 625 | 240 | 37 | 0.31 | 0.9 |
| Toolbox 5×5 colour-therm-8, 2 000 cl, 60 ep | 2 000 | 784 | 654 | 32 | 0.53 | 1.6 |
| Toolbox 5×5 colour-therm-8, 8 000 cl, 60 ep | 8 000 | 784 | 654 | 127 | 2.1 | 6.4 |
| Toolbox 5×5 colour-therm-8, 32 000 cl, 60 ep | 32 000 | 784 | 654 | 509 | 8.5 | 25.4 |
| **Toolbox 5×5 colour-therm-8, 64 000 cl, 250 ep (the SOTA member)** | 64 000 | 784 | 654 | 1 018 | **70.7** | **212** |
| Composites 2023 10×10 thresholding, 2 000 cl, 100 ep | 2 000 | 529 | 344 | 11.3 | 0.31 | 0.9 |
| Drop Clause scale: 60 000 cl, 10×10, adaptive, 60 ep | 60 000 | 529 | 344 | 339 | 5.6 | 16.9 |
| P5 screen: 10 000 imgs, 2 000 cl, 4×4 therm-8, 15 ep | 2 000 | 841 | 440 | 5.1 | 0.02 | 0.1 |

Memory (`ta_state` + `include` cache + 3 accumulators, fp32/int32) stays under **1.6 GiB** even at
64 000 clauses — **time, not memory, is the binding constraint**, and the 10 GB card is not the
limit anyone expected. `[MEASURED: LIBRARY_GAPS LG-004]` independently confirms the memory side:
peak GPU memory stays around 1 GB across 640→8 000 clauses and 4×4→10×10 patches.

### 6.1 SUPERSEDED — use the measured price list

*(Revised 2026-09-20 after P0.)* The estimates above are superseded by
`[MEASURED: results/tables_p0.md]`:

    s/epoch = (clauses_total / 1000) x coef(k) x F(k,Z)/F(k,12) ,  F(k,Z) = Z k^2 + 2(32-k)
    coef = {3: 2.6, 4: 3.7, 5: 4.8, 6: 5.8, 8: 7.3, 10: 10.1, 16: 16.5}   (45 000 train, default chunk)

Every calibration cell was measured at **therm-4, i.e. Z = 12**; the `F` ratio is this document's
inference and the engineer should confirm it, because it is load-bearing (±3.6×). Two corrections
follow, in opposite directions:

* **The model above was ~1.8× optimistic** at the therm-4 configurations
  (`[MEASURED: calib_grid_gpu0]` `ctm-vanilla` @8 000 / 10×10 / 60 ep = ~82 min/seed vs ~45 predicted),
  most likely because the default chunk budget serialises the mini-batch (`LG-004`: chunk size 2 of 50).
* **But the papers' own Booleanization is far cheaper than therm-4.** Adaptive Gaussian thresholding
  per channel is Z = 3, and `F(10,3)/F(10,12) = 344/1244 = 0.28`, so a *paper-faithful*
  arm is **3.6× cheaper** at 10×10 than the therm-4 cell that was calibrated.

Revised anchor costs on the measured list:

| | cost |
|---|---|
| **82.8% faithfully** (22 specialists × 64 000 clauses × 250 epochs, each at its own window and encoding) | **≈ 440 GPU-h at 1 seed, ≈ 1 300 at 3** — still 4–10× the programme envelope |
| **75.1% faithfully** (Granmo 2023: 4 specialists × 2 000 clauses × 100 epochs) | **≈ 5 GPU-h at 3 seeds** |
| **66.42 ± 0.19 faithfully** (GraphTM's CoTM baseline: 80 000 clauses, 8×8, adaptive Gaussian, 30 ep) | **≈ 4.3 GPU-h at 3 seeds** |
| `ctm-vanilla` @80 000 = 8 000/class (the 2019 paper's own budget), 10×10, adaptive, 60 ep | **≈ 11.2 GPU-h at 3 seeds** (not 31.6 — that figure is for therm-4) |

**No conclusion in this document changes**; the 82.8% composite remains out of reach and the
75.1% composite and the 66.42% CoTM remain reachable, now by a wider margin. Also
`[MEASURED: results/tables_p0.md]`: raising `max_chunk_elements` cuts 8 k/10×10 from 77.7 to
55.4 s/epoch, 16 k from 151.1 to 103.7 and 40 k from 329.4 to 270.3 — **roughly 30% of the P3
envelope is recoverable** once the chunk-budget neutrality check lands.

**The two anchors, costed:**
- **82.8% faithfully** = 22 specialists × 64 000 clauses × 250 epochs ≈ **1 550 GPU-h at 1 seed**,
  **≈ 4 600 at 3**. Against a 220–300 GPU-h programme budget: out of reach by 15–20×.
- **75.1% faithfully** (Granmo 2023: 4 specialists × 2 000 clauses × 100 epochs, budget 32) ≈
  **1.6 GPU-h/seed, ≈ 5 GPU-h at 3 seeds**. In reach.
- **66.42% faithfully** (GraphTM's CoTM baseline, the best-specified single-model number here) ≈
  **9.3 GPU-h at 3 seeds**. In reach, on the 3090.

---

## 7. Library expressibility — one-line verdicts

**Expressible today, no new code**: A1 CTM, A3/A4 CoTM, A2 integer weights, A6 Drop Clause,
A7 clause-size constraint, A10 confidence, A11 C-RTM, A13 multi-task/multi-label + clause
interpretation, A22 this repo's stack.
**Expressible in harness code only**: A8/A9 composites and their Booleanizations (Otsu, Canny, HOG,
adaptive multilevel thermometers), A19 pre-sorting, A20 CNN-feature CTM, count-pooling and
multi-scale patches (by subclassing `_evaluate`/`_votes`, which `models/conv.py::_ConvMixin` shows is
the supported extension point).
**Not expressible without changing `src/` (forbidden by C1)**: A5 GraphTM, A16 Sparse TM,
A17 Contracting TM, A2's real-valued weights, and a public per-patch clause output.
Gaps recorded as `LG-006` … `LG-010` in `LIBRARY_GAPS.md` (LG-003/004/005 were filed concurrently by `research-engineer`).

---

## 8. CONFIGURATION BLOCK — everything needed to reproduce the Optimized Toolbox (arXiv:2406.00704)

**Written 2026-09-20 for `research-engineer`, who is blocked on it.** Two sources, and they are not
interchangeable:

* **The paper** `[FACT: arXiv:2406.00704]` — authoritative for the *reported accuracies* and for the
  post-Optuna hyperparameters in Table III.
* **The reference implementation** `[FACT: github.com/cair/An-Optimized-Toolbox-for-Advanced-Image-Processing-with-Tsetlin-Machine-Composites, retrieved 2026-09-20]`
  and `[FACT: github.com/cair/tmu, tmu/models/classification/vanilla_classifier.py]` — authoritative
  for *structure*: model class, encoder parameters, literal budget, and protocol. **Several script
  defaults predate the paper's Optuna search and disagree with Table III — those are flagged below.
  Use Table III for hyperparameters, the scripts for everything else.**

### 8.1 The clause convention — SETTLED, and it is **per class**

Two independent confirmations:

1. `[FACT: arXiv:2406.00704 §II-B]` — *"A TM utilizes a group of n conjunctive clauses **for each
   class**. Here, n is a hyper-parameter set by the user. Half of the conjunctive clauses are
   assigned positive polarity (+), while the remaining half receive negative polarity (−)"*, and
   Eq. (2) is `ŷ = argmax_i ( Σ_{j=1..n/2} C^{i,+}_j(x) − Σ_{j=1..n/2} C^{i,−}_j(x) )`.
2. `[FACT: cair/tmu vanilla_classifier.py]` — every script instantiates
   `TMClassifier(number_of_clauses=…)` from `tmu.models.classification.vanilla_classifier`, which
   calls `self.clause_banks.populate(list(range(self.number_of_classes)))` — **one clause bank of
   `number_of_clauses` clauses per class** — with
   `positive_clauses = [1]*(n//2) + [0]*(n//2)` inside each bank, and
   `transform()` returning shape `(N, n_classes × number_of_clauses)`.

> **Therefore: every clause count in the Toolbox's Tables II, III and IV is PER CLASS.
> "2 000 clauses" = 20 000 total. "8 000" = 80 000 total. "64 000" = 640 000 total.
> The 82.8% composite holds 22 × 640 000 = 14.08 million clauses.**

**The model is class-owned with fixed per-clause polarity — `ConvTsetlinMachine`, not
`ConvCoalescedTsetlinMachine`.** This confirms `ARMS.md` A2 and settles A1. `T` is compared against a
*single class's* vote sum, so `ARMS.md` convention 2 (`T_ratio` × `n_clauses/n_classes` for a
class-owned model) is the right normalisation — only the absolute counts move.

**M0 is now a confirmation, not a discovery.** It should still run (2.1 GPU-h) but its result is
predicted: 20 000 total clauses reproduces the paper's 2 000-clause column, 2 000 total does not.

### 8.2 Per-specialist configuration (Table III, post-Optuna — authoritative for hyperparameters)

All 22 specialists: **250 epochs, 2 000 clauses per class (= 20 000 total)**. `max_included_literals`
is **absent from Table III** but is **32 in every reference script** — see §8.7, and note
`[MEASURED: ctm-small-T80-{unc,b32}_seed{0,1,2}]` that this is worth **+9.46 pp** here (47.77 ± 0.32 vs 38.31 ± 1.05 at a corrected `T`), and **+8.75 pp** under the exact sequential algorithm `[MEASURED: ctm-small-seq-{unc,b32}_seed{0,1,2}]` — so it is not an artefact of our batching. THEORY.md §5.5. (The superseded +7.50 pp figure was measured at the inert `T = 160`.)

| TM Specialist | window | weighted | **T** | **s** | T / (clauses per class) |
|---|---|---|---|---|---|
| 10×10 Canny Edge Detection | 10×10 | True | 1 500 | 10.0 | 0.75 |
| 5×5 Adaptive Gaussian Thresholding | 5×5 | True | 2 500 | **3.5** | 1.25 |
| 10×10 Adaptive Mean Thresholding | 10×10 | True | **500** | 10.0 | 0.25 |
| 10×10 Otsu's Thresholding | 10×10 | True | 3 000 | 10.0 | 1.50 |
| 3×3 / 4×4 / 5×5 Colour Thermometers | 3×3 / 4×4 / 5×5 | True | 3 000 | 5.0 | 1.50 |
| 3×3 / 4×4 / 5×5 Adaptive Colour Thermometers | 3×3 / 4×4 / 5×5 | True | 3 000 | 5.0 | 1.50 |
| Histogram of Oriented Gradients | **32×32 = no convolution** | **False** | **50** | 10.0 | 0.025 |

Each also has an "Augmented" twin with **identical** hyperparameters (§8.7).
`T / (clauses per class)` spans **0.025 → 1.50**, which is the concrete form of the round's agreed
finding that `T_ratio = 0.8` is not a safe default.

**Script/paper disagreements (use Table III):** `CIFAR10AdaptiveThresholdingGaussian.py` defaults to
`T=1500, s=10.0, patch_size=10` against Table III's `T=2500, s=3.5, 5×5` — the script predates the
Optuna search. Canny (`T=1500, s=10, patch 10`), Otsu (`T=3000, s=10, patch 10`), Adaptive Mean
(`T=500, s=10, patch 10`), 5×5 Colour Thermometers (`T=3000, s=5.0, patch 5`) and HOG
(`T=50, s=10, weighted False`) all agree with Table III.

### 8.3 Target accuracies `X` for the engineer's pre-flight decision rule

The paper's cheapest measured cell is **2 000 per class = 20 000 total**. There is **no 8 000-total
cell in the paper** — its "8,000" column is 80 000 total. Both readings, plus a down-extrapolation:

| encoding | window | **@ 20 000 total** (paper's "2,000") | **@ 80 000 total** (paper's "8,000") | local slope pp/doubling | @ 8 000 total *(extrapolated, weaker)* |
|---|---|---|---|---|---|
| 5×5 Colour Thermometers | 5×5 | **64.5** | **69.1** | 2.30 | ~61.5 |
| 4×4 Colour Thermometers | 4×4 | 64.6 | 68.9 | 2.15 | ~61.8 |
| 3×3 Colour Thermometers | 3×3 | 63.6 | 68.4 | 2.40 | ~60.4 |
| HOG (flat) | 32×32 | 64.2 | 66.9 | 1.35 | ~62.4 |
| 5×5 Adaptive Colour Therm. | 5×5 | 62.0 | 67.2 | 2.60 | ~58.6 |
| 4×4 Adaptive Colour Therm. | 4×4 | 61.3 | 66.9 | 2.80 | ~57.6 |
| 3×3 Adaptive Colour Therm. | 3×3 | 60.8 | 65.3 | 2.25 | ~57.8 |
| 5×5 Adaptive Gaussian | 5×5 | 57.9 | 63.0 | 2.55 | ~54.5 |
| 10×10 Adaptive Mean | 10×10 | 57.8 | 62.3 | 2.25 | ~54.8 |
| 10×10 Otsu | 10×10 | 56.9 | 61.2 | 2.15 | ~54.1 |
| 10×10 Canny | 10×10 | 48.7 | 53.4 | 2.35 | ~45.6 |

**Recommendation: run the pre-flight at 20 000 total clauses**, which is the paper's own measured
cell and needs no extrapolation. With **5×5 Colour Thermometers** the target is `X = 64.5`.

**Epoch adjustment to `X`** (from §8.6): subtract **1.2 pp at 50 epochs**, **0.5 pp at 100 epochs**,
**0 at 250**. So a 50-epoch, 20 000-clause, 5×5-colour-thermometer pre-flight scores against
**X ≈ 63.3**, and the decision rule reads: ≥60.3 proceed · 55.3–60.3 hyperparameter fault ·
<55.3 implementation fault.

**Two further adjustments the engineer should apply before calling a `GAP`:** the paper trains on the
full 50 000 and reports the mean of the last 25 epochs on the test set (§8.7); we train on 45 000 and
select on validation. Log the paper's estimator alongside (C-5), and expect our number to sit at or
slightly below `X`.

### 8.4 HOG — the exact specification (it is an OpenCV descriptor, not skimage)

`[FACT: CIFAR10HistogramOfGradients.py]`:

```python
hog = cv2.HOGDescriptor(
    (32, 32),   # winSize             -- the whole image
    (12, 12),   # blockSize
    (4, 4),     # blockStride
    (4, 4),     # cellSize
    18,         # nbins
    1,          # derivAperture
    -1.0,       # winSigma
    0,          # histogramNormType   (L2-Hys)
    0.2,        # L2HysThreshold
    True,       # gammaCorrection
    64,         # nlevels
    True,       # signedGradient      -- 0..360 deg, NOT the usual 0..180
)
X[i] = hog.compute(image_uint8_HWC_RGB) >= 0.1      # Booleanization: elementwise >= 0.1
```

* Feature length = blocks × cells-per-block × bins = `((32−12)/4+1)² × (12/4)² × 18` =
  `36 × 9 × 18` = **5 832 Boolean features**.
* **`signedGradient=True` and `nbins=18`** are both non-default and both matter — unsigned 9-bin HOG
  is the common choice and would give 2 916 features and different content.
* `hog.compute` takes the **uint8 3-channel image directly**; OpenCV takes the per-pixel max-magnitude
  channel internally. Do not greyscale first.
* **`patch_size = 0` in the script and no `patch_dim` is passed to `TMClassifier`: the HOG specialist
  is a FLAT (non-convolutional) TM over 5 832 features.** Table III's "32×32 convolution window" means
  "no convolution". It is also the only **unweighted** specialist, with `T = 50`.
* This is `LG-008`'s largest missing encoder. A pure-torch re-implementation is possible but a
  `cv2.HOGDescriptor` call is exact and cheap — prefer it, and cache.

### 8.5 The other Booleanizations — exact parameters

| encoding | reference code | library equivalent |
|---|---|---|
| **Colour thermometers** | `X[:,:,:,:,z] = img >= (z+1)*255/(resolution+1)`, `resolution = 8`, reshaped to `(N,32,32,24)` channel-major | **`ColorThermometerEncoder(n_bits=8)` is bit-identical** — same thresholds `255k/9`, same `>=`, same channel-major ordering (`encoders.py:314-336`) |
| **Adaptive Gaussian** | `cv2.adaptiveThreshold(ch, 1, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, blockSize=3, C=2)`, **per RGB channel** | `AdaptiveThresholdEncoder(block_size=3, C=2.0, method="gaussian")` — **note `block_size=3`, not the library default 11.** (The 2019 CTM paper used 11 for MNIST `[FACT: arXiv:1905.09688 §4]`; the Toolbox uses 3 for CIFAR-10) |
| **Adaptive Mean** | `cv2.adaptiveThreshold(ch, 1, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize=11, C=2)`, per channel | `AdaptiveThresholdEncoder(block_size=11, C=2.0, method="mean")` — matches the library default |
| **Otsu** | `cv2.threshold(ch, 0, 1, THRESH_BINARY + THRESH_OTSU)`, per channel | **absent** (LG-008); trivial in torch (per-channel between-class-variance argmax) |
| **Canny** | `cv2.GaussianBlur(ch, (3,3), 0)` then `cv2.Canny(blur, 100, 200) != 0`, per channel | **absent** (LG-008); use OpenCV |
| **Adaptive colour thermometers** | `MultilevelThresholdingThermometers.py`: recursive mean±`k`σ split, `k = 0.7` with schedule `k ← k·(i+1)`, `n = 8` thresholds, then thermometer-encode with the learned thresholds | **absent** (LG-008); ~30 lines |

All three thresholding encoders process **each RGB channel independently as a greyscale image** →
Z = 3 planes. Colour thermometers → Z = 24.

### 8.6 250 epochs is an **unexamined default**, and it costs ≤ 0.5 pp to run 100 instead

`[FACT: Fig. 3 of arXiv:2406.00704, digitised from the embedded vector paths]`. Validation: the
digitised finals reproduce Table IV's composite row to **≤ 0.3 pp** (79.7/80.5/81.7/81.9/82.8/82.7
against the published 79.5/80.6/81.5/82.2/82.7/82.8).

| clauses/class | ep 5 | ep 10 | ep 25 | **ep 50** | ep 75 | **ep 100** | ep 150 | ep 250 | first epoch within 1.0 pp | within 0.5 pp |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 000 | 73.7 | 75.5 | 77.7 | **78.5** | 79.1 | **79.2** | 79.5 | 79.7 | 40 | 63 |
| 4 000 | 74.5 | 76.5 | 78.8 | **79.4** | 80.2 | **80.0** | 80.2 | 80.5 | 47 | 58 |
| 8 000 | 75.5 | 77.5 | 79.4 | **80.5** | 80.8 | **81.2** | 81.5 | 81.7 | 42 | 62 |
| 16 000 | 75.8 | 77.9 | 79.7 | **81.1** | 81.5 | **81.9** | 81.9 | 81.9 | 46 | 74 |
| 32 000 | 75.9 | 78.4 | 80.2 | **81.6** | 81.9 | **82.4** | 82.5 | 82.8 | 57 | 79 |
| 64 000 | 76.8 | 78.5 | 80.8 | **81.6** | 82.2 | **82.4** | 82.8 | 82.7 | 47 | 72 |

> **Answer: 250 epochs is a default, not a requirement.** Running **100 epochs costs 0.3–0.5 pp**,
> which is **inside the programme's 1.00 pp seed band** — i.e. free, for a 60% saving. Running **50
> epochs costs 1.1–1.2 pp**, which is *at* the band: acceptable for screens, ladders and pre-flights,
> not for a headline reproduction. **Recommendation: 100 epochs for any arm whose number is reported
> as a reproduction; 50 for ladders, screens and pre-flights, with the −1.2 pp correction applied to
> the target.**
>
> Caveat `[HYPOTHESIS]`: Fig. 3 is the *composite*. An individual specialist may converge differently,
> although a composite is a sum of specialists so it is a reasonable proxy. **Falsifier**: our own
> per-epoch curve for one specialist at 20 000 clauses showing >1 pp still to gain after epoch 100.
> It costs nothing — the curve is already logged.

### 8.7 Protocol facts from the reference code — four of them change how we score

1. **`max_included_literals = 32` in every one of the 22 scripts**, and **Table III does not mention
   it**. Combined with `[MEASURED: ctm-small-T80-{unc,b32}]` (**+9.46 pp**, 47.77 ± 0.32 vs
   38.31 ± 1.05; **+8.75 pp** sequential),
   this means the paper's numbers all carry a literal budget and **a reproduction without one is not
   a reproduction**. A clause-size budget is a *default for every arm*, not a control (see the
   ROUND-1 addendum).
2. **No validation split.** Every script fits on the full 50 000 training images and calls
   `tm.predict(X_test)` each epoch, saving per-epoch class sums for the 10 000 **test** images; the
   paper reports the mean of the last 25 epochs. Confirms C-5, and it is the exact size of the
   protocol penalty we must log alongside.
3. **The composite's normaliser is computed on the test set.** `α_t = max−min class sum` is taken
   over the saved test-set class sums, so the published composition (Eq. 4–5) is **transductive**.
   This is open question **O-1**; our reproduction must fix `α_t` on validation and report both.
4. **Augmentation = static horizontal-flip doubling of the training set**
   (`augmented_images.append(horizontal_flip(image))`), **at the same 250 epochs** — confirming the
   data-budget confound behind **C-3**. The augmented twins use identical hyperparameters.
