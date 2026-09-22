# ROUND-1 — tm-theorist position

**Question**: which existing approaches do we re-implement, in what order, and why those?
**Deliverables backing this paper**: `LITERATURE_TM.md`, `THEORY.md`, `LIBRARY_GAPS.md` LG-006…LG-010.

## Answer in one sentence

Re-implement the **three axes that account for essentially all published CIFAR-10 progress** —
Booleanization, clause count, and confidence-weighted composition — plus the **two single-model
mechanisms that have a checkable CIFAR-10 number** (vanilla CTM and Drop Clause), and **drop every
arm whose paper reports no CIFAR-10 result**; because since 2019 nobody has changed the convolution
mechanism itself, which is simultaneously the reason the P3 list should not be a paper-enumeration
and the reason this programme has somewhere to go.

---

## Claims

### On the two anchor numbers (this is the finding that changes the plan)

- `[FACT: Grønningsæter et al. 2024, arXiv:2406.00704 §V and ref. 14]` **60.7% is not a published
  CTM result.** It traces to *E. Mathisen and H. S. Smørvik, "Analysis of binarization techniques and
  Tsetlin machine architectures targeting image classification", MSc thesis, University of Agder,
  2020*, which "utilized a **modified** CIFAR-10 dataset". It is cited at third hand, the
  modification is not described in any paper we hold, and the thesis itself is not reachable from
  this machine. Its clause budget, window, T, s and epoch count are unknown to us.
- `[FACT: arXiv:1905.09688 §5]` **The Convolutional TM paper reports no CIFAR-10 number at all.** It
  reports MNIST 99.40 / K-MNIST 96.31 / F-MNIST 91.50 / 2D Noisy XOR 100.0 peak, and names CIFAR-10
  only as future work, together with "deeper CTMs".
- `[FACT: Sharma et al., AAAI 2023, Table 2]` **A vanilla CTM reaches 69.3% on CIFAR-10** at 60 000
  clauses, T = 48 000, s = 10.0, adaptive Gaussian thresholding. With Drop Clause p = 0.5 it reaches
  **75.1 ± 0.4%**. (The paper states no convolution window and no epoch count — an omission that
  makes its headline not reproducible from the paper alone.)
- `[FACT: arXiv:2406.00704, Table IV]` **The best single-model TM on CIFAR-10 is 75.4 ± 0.09%** —
  5×5 Colour Thermometers, **64 000 clauses**, 250 epochs, T = 3 000, s = 5.0, weighted.
- `[FACT: arXiv:2406.00704, Tables III–IV]` **82.8 ± 0.01% is an ensemble**, confirmed: a TM
  Composite of **22 TM Specialists, each with 64 000 clauses, each trained 250 epochs**, over seven
  Booleanizations, half of them on a flip-doubled training set. Composite at 32 000 clauses is 82.7,
  at 2 000 clauses 79.5.
- `[FACT: arXiv:2309.04801, Eqs 6–8 and Tables 1–2]` The composition rule is inference-only: each
  member's class sums are divided by `α_t = max−min class sum **over the evaluated set**` and
  summed. **`α_t` is computed on the set being classified**, so the published protocol is
  transductive. Its 2023 instance — 4 specialists × **2 000** weighted clauses, literal budget 32,
  100 epochs — reaches **75.1%** against a best member of 63.5%.
- **Therefore**: `[FACT, composed from the three rows above]` PLAN §2's ladder is mis-set. T0's
  "55–61%" is not a reproduction target; T2's "beat 60.7%" understates the single-model bar by
  **14.7 points**. `[HYPOTHESIS]` The right ladder is T0 = match the Toolbox's own clause-scaling
  column at our budget (±3 points, PLAN §7.5), T2 = **75.4%**, T3 = 82.8%.
  **Falsifier**: the Mathisen & Smørvik thesis becomes reachable and shows a standard CIFAR-10.

### On what the literature has actually changed

- `[FACT: arXiv:1905.09688 §3; Book Ch. 4 §4.3–4.6; Abeyrathna et al. 2021 Eq. 4; Sharma et al. AAAI
  2023 "TM and CTM"]` **The CTM's convolution mechanism — patch OR, one uniformly random matching
  patch per feedback event, thermometer position bits — is unchanged in every paper in this
  bibliography.** I swept the whole corpus for `pooling|max-pool|stride|dilat|multi-scale|receptive
  field|two-layer|deeper CTM|hierarch`; the only construction that varies it is CTM-UNet
  `[FACT: ISTM 2025, Algorithm 1]`, which keeps clause outputs spatially resolved and max-pools
  between blocks — on CamVid segmentation, as a backprop hybrid, with no CIFAR-10 number.
- `[FACT: arXiv:2406.00704 Table IV]` **Booleanization is the largest measured lever**: at a fixed
  2 000 clauses the specialists span **48.7% (10×10 Canny) to 65.4% (4×4 augmented colour
  thermometers)** — a 16.7-point range from the encoding alone.
- `[FACT: arXiv:2406.00704 Table IV]` **Clause count is the second**: 5×5 colour thermometers go
  64.5 → 75.4 from 2 000 to 64 000 clauses (**+10.9**), and are **still climbing at 64 000**
  (+1.7 from 32k). HOG, by contrast, saturates flat at 67.5 by 32 000.
- `[FACT: arXiv:2309.04801 Table 2; arXiv:2406.00704 Table IV]` **Composition is the third**: +11.6
  over the best member at 2 000 clauses, +7.4 at 64 000. The gain shrinks as members grow.
- `[FACT: Sharma et al. AAAI 2023 Table 2]` Drop Clause is the **only** published single-model
  mechanism worth more than a point on CIFAR-10: **+5.8**, and it halves epoch time.
- `[FACT: arXiv:2301.08190 Table 3]` The clause-size constraint is worth **+0.06** on CIFAR-2
  (94.24 at budget 64 vs 94.18 unconstrained) and **+0.05** on MNIST-with-convolution. It is an
  interpretability and hardware mechanism, not an accuracy mechanism — yet it is part of two SOTA
  recipes (`[FACT: arXiv:2309.04801 Table 1]` budget 32; `[FACT: arXiv:2507.14874 Table 16]`
  budget 32; the Optimized Toolbox's Table III states none), so it belongs in the arm list as a
  *control*, not as a candidate.
- `[FACT: arXiv:2507.14874 §3.2 and Table 16]` **GraphTM's CIFAR-10 advantage is not a depth
  effect.** Its CIFAR-10 run uses **depth = 1**, and the authors attribute the +3.86 points over
  their CoTM baseline to "the GraphTM's ability to incorporate multiple views of each image … the
  CoTM was trained using only adaptive Gaussian thresholding". The comparison is unmatched in input.

### On the theory (full derivations in `THEORY.md`)

- `[FACT: derivation, THEORY.md §2]` A convolutional TM layer is **exactly** a linear model over a
  learned Boolean dictionary of existential conjunctive patch predicates:
  `ŷ = argmax_i ⟨w_i, Φ(x)⟩` with `Φ_j(x) = [∃ patch b : φ_b(x) ⊨ π_j]`.
- `[FACT: derivation, THEORY.md §3.3–3.4]` Consequently one layer cannot express XOR of two
  presences, and cannot express any *relative* spatial relation between patterns that do not fit in
  one window — **at any clause budget**, because more clauses only add more coordinates of the same
  existential kind. "A present and B absent" *is* reachable; so is the conjunction of two
  absolute-position presences.
- `[FACT: derivation, THEORY.md §5.2]` The feedback rule makes **specificity free**: a clause's vote
  is 1 whether it matches 1 patch or 841, and only `|M_j| = 0` is punished (by Type Ib). The learning
  pressure is therefore exactly "match ≥ 1 patch of as many target images as possible and 0 patches
  of as many negative images as possible", which drives clauses toward knife-edge detectors.
- `[MEASURED: experiments/mctm, PLAN.md §3.1]` Corroborated: trained layer-1 clauses fire on
  **0.14% of patch positions** — on an 841-position grid, **≈ 1.2 matching patches per firing
  clause**. **The OR in `⋁_b c_j^b` is, in the trained regime, almost never an OR over more than one
  term.**
- `[HYPOTHESIS]` Therefore **a counting pool is a no-op at the densities the standard feedback rule
  produces, and can only pay off jointly with a density / firing-rate controller.** This is a
  prediction against PLAN §9.2 item 1 tested in isolation, and it links item 1 to item 3.
  **Falsifier**: median `|M_j|` ≥ 5 for a firing clause in a ≥ 8 000-clause colour-thermometer CTM.
  The measurement is free (the `(B,P,C)` tensor is already computed at `conv.py:140-146`).
- `[HYPOTHESIS]` **The clause-scaling curve saturates at the information ceiling of the
  Booleanization, not at a ceiling of the clause mechanism** (HOG flat at 67.5 by 32k; colour
  thermometers still rising at 64k). **Falsifier**: `cnn-boolean` on HOG bits substantially beating
  67.5%, which would show the ceiling is the TM's, not the encoding's.
- `[FACT: THEORY.md §4.1]` At 64 000 clauses and a 10×10 window over 24 planes the CTM holds
  **313 M include/exclude bits**, against ResNet-18's ≈ 358 M parameter bits. The CIFAR-10 gap is
  therefore `[HYPOTHESIS]` not a raw storage gap but a gap in what those bits may express.
- `[FACT: THEORY.md §6.2]` Growing the window grows the automata budget **quadratically** while the
  supply of training patches **falls**; compute peaks near `k ≈ 16` on 32×32 and collapses at
  `k = 32` (which is a flat TM). This is `[HYPOTHESIS]` why every published CIFAR-10 window sits in
  3–10 and why the one 32×32 "window" in the literature (HOG) is applied to an already-pooled
  descriptor.

### On configuration ambiguities that will silently break matched-budget claims

- `[FACT: arXiv:1905.09688 Table 1; arXiv:2301.08190 fn. 6]` "Clauses" means **per class** in the
  CTM and CSC-TM papers. `[FACT: arXiv:2309.04801 Table 1 vs §4]` TM Composites uses both
  conventions **in the same paper**. `[FACT: arXiv:2406.00704; arXiv:2507.14874]` the Toolbox and
  GraphTM are unqualified. A factor of 10 in the automata budget is at stake (`ARMS.md` A1,
  `LIBRARY_GAPS.md` LG-010).
- `[FACT: LG-010]` **`T_ratio = 0.8` is not a safe default**: measured across the bibliography,
  `T / (total clauses)` spans **0.025 to 1.5**. Every arm must carry its paper's own `T` and `s`.
- `[FACT: LITERATURE_TM.md §2.5]` **No TM image paper uses a validation split.** Reported accuracy
  is the mean of the last 25 / 100 epochs, or a peak. Under PLAN §7.1 our numbers will be *lower*
  for the same configuration; `[HYPOTHESIS]` by ≤ 1 point for a converged CTM. **Falsifier /
  control**: log the paper's own statistic alongside the validation-selected one at G3.
- `[FACT: arXiv:2406.00704 §IV-A and Table IV]` The Toolbox's "Augmented" arms train on 100 000
  images for the same 250 epochs — **an unmatched data budget** — and the effect changes sign by
  Booleanization at 64 000 clauses (Canny +1.8, adaptive mean +1.8, HOG +1.0, adaptive Gaussian
  +0.1, **5×5 colour thermometers −1.3**). PLAN §9.2 item 5 calls augmentation "possibly the highest
  value per GPU-hour in the programme"; the only data we have says its sign depends on the encoding.

### On cost — the constraint that decides the list

- `[HYPOTHESIS: LITERATURE_TM.md §6, calibrated on PLAN §3.4, then corrected ×1.8 against
  [MEASURED: calib_grid_gpu0]]` **A faithful 82.8% is ≈ 2 800 GPU-h at one seed** (22 specialists ×
  64 000 clauses × 250 epochs) — **10–15× the entire programme budget.** It is not reproducible here
  and no plan should assume it.
- `[HYPOTHESIS: same model]` **A faithful 75.1% (TM Composites 2023) is ≈ 9 GPU-h at 3 seeds**
  (4 specialists × 2 000 clauses × 100 epochs). It *is* reproducible here, and it is 14.4 points
  above the anchor the plan currently uses.
- `[HYPOTHESIS: same model]` **A faithful 66.42% (the GraphTM paper's CoTM baseline, the
  best-specified single-model CIFAR-10 configuration in the whole bibliography — 80 000 clauses,
  8×8, T = 15 000, s = 20.0, budget 32, 30 epochs, adaptive Gaussian) is ≈ 17 GPU-h at 3 seeds** on
  the 3090. It is the one literature number we can reproduce *exactly as published*.
- `[MEASURED: LIBRARY_GAPS LG-004]` Memory is not the constraint: peak stays ~1 GB from 640 to
  8 000 clauses; the 10 GB card is under-used and the default chunk budget, not the card, is what
  serialises large arms.

---

## The shortlist (my ranking, with the drops)

Ordering rule per PLAN §8 P3: trust-anchor first, then cheapest-per-bit-of-information. Costs are
`[HYPOTHESIS]` from the ×1.8-corrected model unless marked `[MEASURED]`; each is 3 seeds unless
stated.

| # | Arm | Family | Published target to reproduce against | Why it is on the list | Cost (GPU-h) |
|---|---|---|---|---|---|
| **1** | `ctm-vanilla` | existing | **NOT 60.7%.** Toolbox 5×5-adaptive-Gaussian column: 57.9 @2k / 63.0 @8k / 68.2 @32k `[FACT: Table IV]`; Drop Clause p=0: **69.3 @60k** `[FACT: AAAI 2023]` | T0. Nothing downstream means anything until this reproduces | **4.1** `[MEASURED: calib_grid_gpu0]` |
| **2** | **M0: clause-convention probe** | control | Toolbox 2 000-clause column (64.5 for 5×5 colour thermometers) | Settles `ARMS.md` A1 — per class or total — which is a 10× budget factor. 2 runs, 1 seed | **0.5** |
| **3** | `ctm-boolean-*` (therm4, therm8, ct3/ct4/ct5, adaptive-gauss, adaptive-mean, otsu, canny, hog) | existing | **22 published cells**, Toolbox Table IV @2 000 + Composites Table 2 | **The highest information per GPU-hour in P3**: it validates the harness against 22 numbers at the cheapest budget, and it measures the single largest lever (16.7-point spread). 10 configs × 1 seed, then 4 × 3 seeds for the composite members | **35** |
| **4** | `tm-composite` | existing | **75.1%** `[FACT: arXiv:2309.04801 Table 2]` | The reachable SOTA reproduction; inference-only over arm 3's members. Report `α_t` from **validation** and from test, since the published rule is transductive | **~0** |
| **5** | `ctm-coalesced` | existing | **66.42 ± 0.19** @80 000 cl, 8×8, T=15 000, s=20.0, budget 32, 30 ep `[FACT: arXiv:2507.14874 Table 3 + 16]` | The only fully specified single-model CIFAR-10 configuration in the bibliography; reproduce it *exactly*, plus an 8 000-clause point for the scaling curve | **20** |
| **6** | `ctm-dropclause` p ∈ {0, .25, .5, .75} | existing | 69.3 / 73.2 / 75.1 / 72.6 `[FACT: AAAI 2023 Table 2]` | The only published single-model mechanism worth >1 point; the *shape* of the p-curve is a stronger reproduction test than any single number, and at p=0.5 it costs about half of vanilla per epoch (`[FACT: AAAI 2023]` 61.83 s → 32.58 s on CIFAR-10) | **12** |
| **7** | `ctm-capacity` (2k/4k/8k/16k/32k × 2 Booleanizations) | existing/control | Toolbox Table IV columns | Answers the programme's most important question — capacity-limited or mechanism-limited — *and* is a 10-cell reproduction check at the same time. PLAN P4.1 pulled forward because it is free here | **25** |
| **8** | `ctm-weighted` vs unweighted | control | — (no paper isolates it) | Every SOTA config is weighted; without this control we cannot attribute anything | **4** |
| **9** | `ctm-clausesize` {16, 32, 64, None} + a `feedback_mode="sequential"` control | existing/control | CSC-TM: budget ≈ **no accuracy effect** on images `[FACT: Table 3]` | Budget 32 is in the Composites and GraphTM recipes (the Toolbox states none), so it must be characterised; and this arm **is** the LG-003 probe — a reproduction that is also a library finding | **3** |
| **10** | `ctm-multiview` (CoTM over concatenated adaptive-Gaussian + colour-thermometer planes) | existing/control | GraphTM's **+3.86 over CoTM** `[FACT: arXiv:2507.14874 Table 3]` | **This replaces the GraphTM arm.** It tests the authors' own stated explanation of their CIFAR-10 gain, at ~1% of the implementation cost and with no library change | **4** |
| **11** | `ctm-patch` (k ∈ {3,4,5,8,10,16} at matched **automata** budget) | control | — none; that is the point | The literature's windows range 3×3–32×32 with no matched-budget justification `[FACT: LITERATURE_TM §2.2]`; `THEORY.md` §6.2 predicts a unimodal curve. Cheap and never done | **8** |
| **12** | `mctm-calibrated` | existing | this repo's **41.80 ± 0.98** `[MEASURED: experiments/mctm/results/mctm-random-calib_seed*]` | The T1 rung; re-run under validation-selection so the report can quote it honestly | **1.5** |
| — | matched-budget + matched-RF controls for every multi-component arm | control | — | Mandated by PLAN §8 P3 | folded in |

**Total ≈ 117 GPU-h** — at the top of PLAN's 80–120 band. **Cut order if it overruns**: 11 → move to
P4; 7 → one Booleanization instead of two (−12); 6 → p ∈ {0, 0.5} only (−6). Do not cut 1–5.

### Drops from PLAN §9.1, with reasons

| Dropped | Reason |
|---|---|
| `ctm-graph-deep` (GraphTM) | `[FACT: arXiv:2507.14874 Table 16]` its CIFAR-10 run is at **depth 1**, and `[FACT: §3.2]` its authors attribute the +3.86 to multi-view input, not depth. 8–15 engineer-days and a whole new model (LG-009) to reproduce an effect we can buy for 4 GPU-h as arm 10. **Replaced by `ctm-multiview`.** |
| `ctm-hypervector` | `[FACT: arXiv:2406.02648]` reports **no CIFAR-10 number** (MNIST, IMDb, TREC, chemical structures). Nothing to reproduce against; `HypervectorEncoder` targets discrete tokens, not image patches. |
| `ctm-distill` | `[FACT: arXiv:2504.01798]` is an MSc thesis with **no CIFAR-10 result and no convolutional component**. Not a P3 reproduction. **Keep as a P5 candidate**, where the useful version is a CNN teacher, not a TM teacher. |
| `ctm-sparse`, `ctm-absorbing` | `[FACT: arXiv:2405.02375; arXiv:2310.11481]` both are memory/throughput mechanisms validated on NLP; **neither reports CIFAR-10**; neither is expressible without changing the TA state machine, which C1 forbids (LG-009). |
| `ctm-multitask-rgb` | `[FACT: arXiv:2510.01906]` reports MNIST 98.5% and CelebA 86.56 F1, **not CIFAR-10**; its contribution is an interpretation methodology. **Keep for P6/P7's interpretability deliverable**, not as an accuracy arm. |

### One addition that is not on PLAN §9.1 and that I argue hardest for

`ctm-cnnfeature` — a CTM over **Booleanized frozen CNN features** (the architecture of
`[FACT: 2026_Enhanced_Cervical_Cancer_Classification_Convolutional_TM]`, InceptionV3 → Booleanize →
CTM; here, frozen `cnn-small`/ResNet-18 features from P2). It has no CIFAR-10 number to reproduce,
so it is a **diagnostic arm, not a reproduction arm** — but it is the experiment that decomposes the
whole programme's question, and it costs ~3 GPU-h. See measurement **M2** below.

---

## What I need measured (ranked, with cost estimate)

| Rank | Measurement | Why it is ranked here | Cost |
|---|---|---|---|
| **M1** | **`ctm-cnnfeature`: CTM (2 000 and 8 000 clauses) over Booleanized frozen CNN features, against the same CTM over `therm8` pixels, at matched clause budget** | **The single measurement I most want.** It splits the CTM↔CNN gap into *representation* and *mechanism* in one run. If the CTM over ResNet features reaches ≥ 85%, the convolution mechanism is not the bottleneck and the programme should re-aim at Booleanization; if it stalls near 70%, the mechanism is the bottleneck and §3's capacity results are the story. Nothing else in the plan answers this. Depends on P2 | **~3 GPU-h** |
| **M2** | **`|M_j(x)|` histogram — how many patches a firing clause actually matches** — logged as a per-epoch diagnostic for every conv arm from P3 onward | Free (the `(B,P,C)` tensor is already computed, `conv.py:140-146`). It settles whether OR-pooling is degenerate, and therefore whether count-pooling, multi-scale patches and any "pool" candidate in P5 have anything to work with. Must be added to `code/diagnostics.py` **before** P3 runs, or P5 will have to re-run P3 to get it | **~0** (+0.5 d engineer) |
| **M3** | **M0, the clause-convention probe**: `ctm-vanilla` and `ctm-coalesced` at 2 000 *total* and 2 000 *per class*, 1 seed, against the Toolbox's 2 000-clause column | Every budget-matched claim in the report depends on it. Two runs | **0.5 GPU-h** |
| **M4** | **The Booleanization family at 2 000 clauses against Toolbox Table IV** (arm 3, 1 seed pass) | Validates the harness against 22 published cells at the cheapest budget, and measures the largest lever. This is how T0 should actually be judged, not against 60.7% | **~16 GPU-h** |
| **M5** | **Clause-count scaling to the largest budget we can afford, on the best and worst Booleanization** (arm 7) | Decides capacity-limited vs mechanism-limited — PLAN's own "most important question in the programme". Published comparison exists for every point | **~25 GPU-h** |
| **M6** | **`ctm-clausesize` at batch 50 vs batch 5 vs `feedback_mode="sequential"`** (arm 9) | Reproduction + LG-003 probe + the batched-feedback fidelity control in one. Also gives the `[MEASURED]` id that LG-003's "Impact here" is currently missing | **3 GPU-h** |
| **M7** | **Window sweep at matched automata budget** (arm 11) | `THEORY.md` §6.2 predicts unimodal in `k`; the literature has never matched budgets across windows | **8 GPU-h** |
| **M8** | **Validation-selection bias control**: for `ctm-vanilla`, log both the validation-selected test accuracy *and* the paper's own statistic (mean of last 25 epochs) | Without it, every "GAP" verdict in P3 is confounded with our protocol change. Costs nothing but a logging line | **~0** |
| **M9** | **Object-multiplicity probe** (`THEORY.md` §3.5) — tile a 16×16 crop 1×/2×/4× and compare CTM class-sum change against CNN logit change | Tests Corollary 3 directly. Only worth running after M2 says whether `|M_j|` is degenerate | **0.2 GPU-h** |

---

## What would change my mind

1. **The Mathisen & Smørvik thesis turns out to describe a standard CIFAR-10** with a configuration
   we can afford. Then 60.7% is a legitimate T0 and the ladder stands as written. I could not reach
   the document from this machine; someone with network access should try once.
2. **`ctm-vanilla` lands more than 5 points below the Toolbox's matching clause-count cell.** Then my
   cost model is wrong by more than the ×1.8 correction already applied, the whole 117 GPU-h plan is
   fiction, and the list must be rebuilt around arms 1–5 only.
3. **M5 shows the clause-scaling curve still climbing steeply at the largest budget we can afford.**
   Then the honest recommendation is "more clauses", PLAN §9.2 item 9 applies, arms 6–11 become a
   footnote, and my claim that the gap is expressive rather than capacity-driven is wrong.
4. **M2 shows median `|M_j| ≥ 5` for firing clauses at realistic budgets.** Then my §5.2 argument
   that OR-pooling is degenerate in the trained regime is wrong, count-pooling should be screened on
   its own merits, and `THEORY.md` §3.5/G1 needs rewriting.
5. **M1 shows a CTM over frozen CNN features stalling near 70%.** Then Booleanization is *not* where
   the gap lives, the whole "Booleanization is the biggest lever" reading of the literature is an
   artefact of nobody having tried a better representation, and the programme should go straight at
   the clause mechanism.
6. **`ctm-multiview` fails to recover the GraphTM's +3.86.** Then the GraphTM's advantage is *not*
   the multi-view input its authors claim, something in the graph formulation is doing real work,
   and the 8–15-day GraphTM arm should be reconsidered for P6.
7. **Any member of the team shows me a TM paper that varies the patch aggregation on an image
   benchmark.** My central claim — that the convolution mechanism has been untouched since 2019 —
   rests on a corpus sweep of 45 local PDFs plus the book chapters. It is a claim about *this
   bibliography*, and I would withdraw it for one counter-example.

---

## Explicit disagreements to put into the crossfire

- **With PLAN §2**: T0 and T2 are set from a number that no primary source supports (claim 1). I am
  not proposing a method change — this is a `[FACT]` about the literature — but the orchestrator
  should re-baseline the ladder in `DR-001` before P3 commits compute.
- **With PLAN §9.2 item 5** ("augmentation … may be the single highest-value-per-GPU-hour experiment
  in the programme"): the only data we have `[FACT: arXiv:2406.00704 Table IV]` says static flip
  augmentation *hurts* the best specialist by 1.3 points and raises its variance 15×, while helping
  the edge/threshold specialists — and the published comparison is not data-budget-matched. I would
  rank it below M1–M6. **Resolved by**: an augmentation arm at **matched data budget** (same images
  seen), which the literature has never run.
- **With PLAN §9.2 item 1** ("does a counting pool carry strictly more information?"): yes in
  principle, but `[MEASURED: experiments/mctm]` + `THEORY.md` §5.2 predict it is a null result unless
  paired with density control. **Resolved by M2**, which costs nothing.
- **With PLAN §9.2 item 8** ("thermometer Booleanization treats channels independently"): true of the
  *encoder*, false of the *clause* — a k×k window sees all `Z` planes jointly, so cross-channel
  conjunctions are available today. The real question is whether clauses *use* them.
  **Resolved by** the G7 diagnostic in `THEORY.md` §7: the fraction of included literals spanning
  ≥ 2 colour channels. Free.

---

## Crossfire response

Read `rounds/ROUND-1/crossfire.md`. Two of my claims move, one of my sentences is withdrawn, and one
of my falsifiers was badly specified. Details below, then the re-priced shortlist.

## 0. Accept / reject ledger

**Accept without reservation**: all of section A (my own findings, independently verified);
`ctm-vanilla` = class-owned (C-4.2); `T_ratio = 0.8` is unsafe; memory is never the constraint and
the 3080 is a slower card, not a small one; the 1.00 pp seed band for 3-seed means; the
chunk-budget neutrality rule; binary computation is not the bottleneck (the DL expert's BWN/BNN
decomposition — it is new to me and it sharpens `THEORY.md` §7 G6: if binary *weights* cost ~0.2 pp
and binary *activations* cost 12.4, then a TM's all-or-nothing clause is the analogue of the
expensive half, and G5 rises in importance).

**Accept with correction**: C-1 (§1), C-2 (§2).

**Reject**: the dl-expert's "OR gives the vote `C` bits, counting gives ~`9.7C`" — the correct figure
from the measured distribution is **×1.71**, not ×9.7 (§1.2). And the dl-expert's "saturates below
70% by 8–16 k clauses" — `[FACT: arXiv:2406.00704 Table IV]` puts 5×5 colour thermometers at
**69.1% at 8 k, 71.7% at 16 k, 73.7% at 32 k, 75.4% at 64 k**. That is a published table; the
prediction is contradicted by it unless the Toolbox's clause counts are per class (which M0
settles for 1.6 GPU-h).

**New finding I am adding to the round**: §7 — composition is worth **+5.0 points at matched
*total* clause budget**, and the advantage shrinks with budget. Neither position paper said this and
it is derivable from published tables.

---

## 1. C-1 — the counting pool

### 1.1 My falsifier did not fire, and I am not going to hide behind that

Pre-registered: median `|M_j|` ≥ 5 falsifies. Measured median is **2**. Formally my prediction
stands. I do not rest on it, because **my falsifier was mis-specified, and so was the DL expert's**.

`[MEASURED: results/ctm-small_seed0.json → diagnostics.match_count]` the full quantiles are
q25 = 1, **median = 2**, q75 = 6, p95 = 24, max = 433, mean 5.99, P(=1 | fires) = 0.354,
P(≥5 | fires) = 0.303, fire_frac = 0.0698.

Both conditions were stated about a distribution **that the intervention itself changes**. The
measured histogram is the equilibrium of a model *trained under OR pooling*. A count pool changes
the training objective, so it changes the density it would be judged on. Neither "median ≥ 5" nor
"median ≈ 1" could have settled anything. That is a methodological error on my part and it is worth
recording for the rest of the programme: **a pre-registered falsifier stated over a quantity that
the candidate mechanism alters is not a falsifier.**

### 1.2 What the distribution does say, and the arithmetic that corrects both of us

My `THEORY.md` §5.2 mechanism — *specificity is free, only `|M| = 0` is punished* — **is visible**:
the mode is 1, 35.4% of firing events have nothing to count, and the median is 2. My *sentence*
"almost never an OR over more than one term" is **withdrawn**: 64.6% of firing events have ≥2 and
30.3% have ≥5. The distribution is long-tailed, not degenerate, and the mean is tail-dominated.

**What I missed: Type Ib is the density floor.** A clause that narrows too far stops firing on target
images and is punished by Type Ib forgetting `[FACT: functional.py:250-251; Book Ch. 4 §4.4
step 4(a)(ii)]`. The equilibrium is a balance between Type II (narrow, to kill false positives) and
Type Ib (broad, to keep firing on targets), and a mode-at-1 heavy tail is exactly what that balance
produces. §5.2 stated only the first pressure. `THEORY.md` is being corrected.

**The information claim, done properly.** The vote coordinate's entropy, not the count's range, is
what a pooling change buys, and it is gated by the firing rate:

| | fire_frac | H(indicator) | H(\|M\| given firing) | H(count) | **gain** |
|---|---|---|---|---|---|
| **TM clause** (`ctm-small`) | 0.0698 | 0.365 bits | ≈3.71 bits | 0.625 bits | **×1.71** |
| **CNN filter** (`cnn-ctmshape-*`) | 0.0096 | 0.078 bits | ≈7.84 bits | 0.153 bits | **×1.96** |

(`H(|M| | fires)` from a point-mass-at-1 + geometric mixture matched to the measured `P(=1)` and mean;
it reproduces the measured p95 to within 2 points. Script: scratchpad `price.py`. **Request to the
engineer**: log the full `|M|` histogram, not only quantiles — it is free and it turns this estimate
into a measurement.)

So: **the "×9.7" is wrong by 5.7×.** But notice the more interesting consequence — **1.71 and 1.96
are close.** The information ratio does not discriminate between the two substrates, so it cannot
predict the TM's gain in *either* direction. **Both of us were using the wrong lens.** I withdraw
"counting has nothing to count" as an information argument, and I reject "×9.7 ⇒ 5–12 points" as one.

### 1.3 The transfer objection that does survive — and it is structural, not statistical

The CNN's 12.2-point `max` → `sum` gap confounds two things: what the *forward pass* sees, and
where the *gradient goes* (max routes it to one spatial position per filter per image; sum to all).
A Tsetlin machine has no gradient — but it has an exact analogue of the first behaviour:

> `[FACT: Granmo et al. 2019 §3; Book Ch. 4 §4.4 step 4(a)(i); src/torchtsetlin/models/conv.py:169-171]`
> **the TM already updates from exactly one uniformly random matching patch, and it does so
> regardless of how the clause output is pooled.**

Changing the pool therefore transfers only the forward half of the CNN's mechanism. To transfer the
credit half you must change the *draw*, not the pool. That is a separate, cheap, and as far as I can
tell **unpublished** candidate:

> **multi-patch feedback**: draw κ matching patches per feedback event instead of 1, and accumulate
> from all κ. It is a few lines in a harness subclass of `_feedback_counts`; the draw already exists.

This is the point the orchestrator invited me to press, and pressing it correctly changes what the
CNN proxy means: **it is an upper bound on what pooling alone can transfer, and the arm that tests
the rest of it is a feedback arm, not a pooling arm.**

### 1.4 Verdict and the arm structure I would screen

**Verdict: (b) — admissible only as a paired candidate, screened as a set.** Not standalone on
present evidence, not dead. The reason is not information; it is that **a count pool with threshold
κ ≥ 2 makes narrowing costly, which is density control applied through the forward pass.** A count
arm and a density arm are confounded *by construction*, so a count arm screened without a
density-only control cannot attribute its own result.

Screen set (PLAN §8 P5.2 protocol: 10 000 images, 15 epochs, 1 seed, small budget, `screen/` only):

| id | arm | what it is |
|---|---|---|
| **S0** | `pool-or` | the screened vanilla — control |
| **S1** | `pool-density` | OR pool + per-clause firing-rate controller — **density-only control** |
| **S2** | `pool-thermo` | **thermometer-count pool**: vote coordinates `[\|M\|≥1], [≥2], [≥4], [≥8]`, each with its own weight. **Zero extra automata**, 4× weight memory |
| **S3** | `pool-thermo + density` | the pairing |
| **S4** | `pool-sum` | graded: contribution `w·min(\|M\|, cap)` — the direct analogue of the CNN's `-sum` |
| **S5** | `feedback-kpatch` | **OR pool unchanged**; draw κ = 4 matching patches per feedback event — the analogue of dense credit (§1.3) |

Promotion rules, all against the measured band:
- counting is **standalone** iff (S2 or S4) beats S0 **and** beats S1, both by more than the band;
- counting is **paired-only** iff S3 beats max(S1, S2) by more than the band while S2 ≈ S0;
- counting is **dead** iff S2, S3, S4 all ≤ S1 + band;
- **if S5 ≥ S4, the effect was credit assignment, not pooling**, and the candidate is re-framed.

Log the `|M|` distribution for every arm: the whole argument is that it **moves**.

**Why S2 rather than a single threshold κ.** A single κ bets on one operating point. The thermometer
version subsumes every monotone function of `|M|` as a linear combination of its coordinates, costs
**zero extra automata** (so it is budget-matched to S0 on the axis that matters), and **its learned
weights are the answer**: if the weights on `[≥2], [≥4], [≥8]` come out near zero, counting is dead
and we know it from a single run. It is the most informative single arm in the set.

**Cost**: all six at screen scale ≈ **0.05 GPU-h**. The cost is engineering (~2 days), all in harness
subclasses of `_evaluate` / `_votes` / `_feedback_counts`, with the LG-005 private-API caveat.

---

## 2. C-2 — capacity or mechanism. **I withdraw my framing and keep my fact; the disagreement dissolves into a slope.**

### 2.1 The fact stands

`[FACT: arXiv:2406.00704 Table IV]`, 5×5 colour thermometers: 2 k → **64.5**, 4 k → 66.8,
8 k → **69.1**, 16 k → **71.7**, 32 k → 73.7, 64 k → **75.4**. There is no flat ceiling at 8–16 k for
a good encoding. HOG *does* flatten (67.5 by 32 k and unchanged at 64 k), which is why I said the
ceiling is encoding-dependent, and I still say that.

### 2.2 What I withdraw

"Saturation is at the Booleanization's information ceiling, therefore buy clauses and encodings" is
the wrong conclusion, because **not saturated ≠ a route**. The per-doubling deltas are
**+2.3, +2.3, +2.6, +2.0, +1.7** — an average of **+2.18 pp per doubling**, decaying.

Extrapolated at the last measured slope (+1.7), priced on the engineer's measured coefficients:

| target | clauses needed | GPU-h / seed |
|---|---|---|
| **82.8%** (match the composite as a *single model*) | ≈ 1.3 × 10⁶ | ≈ **800** |
| **85%** (small CNN, PLAN §2 T4) | ≈ 3.2 × 10⁶ | ≈ **1 960** |
| **94%** (ResNet-18, T5) | ≈ 1.3 × 10⁸ | ≈ **77 000** |

**The clause axis is logarithmically open and practically closed.** Both positions were half right:
the DL expert is wrong about the *shape* (it does not flatten) and right about the *consequence*
(capacity is not how a TM reaches CNN parity). I was right about the shape and wrong to read it as a
route.

### 2.3 The reframing I propose as the round's output

Stop asking "does it saturate". Ask **"what is the slope of accuracy against log₂(clauses), and does
this mechanism shift or steepen it?"**

- A mechanism earns its place iff it **shifts the curve up** (intercept) or **steepens it** (slope)
  relative to `ctm-vanilla` at matched automata budget.
- The published bar is quantitative and cheap to beat-or-miss: **a candidate must be worth more than
  two doublings (≈ +4 pp at matched budget), or it is cheaper to buy clauses.**
- Two points per arm give a slope. This is measurable at *small* budgets, which makes it affordable,
  and it is a far better report statistic than any single accuracy.

**I propose this as the form of DR-002 and as the primary comparison statistic for P5/P6.**

### 2.4 What I would conclude in each direction (as asked)

| our ladder shows | conclusion | consequence for the programme |
|---|---|---|
| ≈ +2 pp/doubling, still rising at 32–64 k (**I expect this**) | TMs are **not capacity-saturated, but capacity return is logarithmic**; parity by clause count needs 10³–10⁴× more automata than anyone has run | P5 proceeds as an *accuracy* contribution **at matched budget**; every candidate scored as a curve shift; the report's headline becomes the slope, not a point |
| flattens below 70% by 8–16 k (DL expert right) | this **contradicts a published table**, so the first response is a `GAP` audit of our T, s, epochs and clause convention — **not** a conclusion | only after that audit closes do I accept "mechanism-limited"; then P5 is justified on the strongest possible grounds and the report leads with the reproduction failure as a finding about the field |
| steeper than +2.18 pp/doubling | clause count **is** a route; PLAN §9.2 item 9 applies | mechanism work demoted to efficiency and interpretability; I would say so plainly and the report's claim changes shape |

The clean falsifier for "the ceiling is the encoding's, not the TM's" remains **`cnn-boolean` on HOG
bits beating 67.5%**. It is a P2 by-product and costs nothing extra.

---

## 3. C-3 — augmentation. Yes, a matched arm resolves it; here is the design and two mechanisms.

Accepted. The confound is real and mine to fix: `[FACT: arXiv:2406.00704 §IV-A]` the augmented arms
train on 100 000 images for the **same** 250 epochs. Three conditions at **equal examples seen**:

- **A** — no augmentation, `N` images, `E` epochs.
- **B** — static flip doubling, `2N` images, `E/2` epochs. *Reproduces the Toolbox design at matched
  data budget and isolates the confound.*
- **C** — on-the-fly random flip + crop, `N` images, `E` epochs. *Only C tests the mechanism.*

I add a **2×2 with `position_encoding`**, because I have a mechanism and it is free to test:

1. `[HYPOTHESIS]` **A horizontal flip mirrors the patch *and* its x-coordinate.** A position-encoded
   clause that learned `x > 12` must be relearned at `x < 17` for the flipped copy
   `[FACT: conv.py:113-119, 203-228]`. **Flip augmentation should therefore cost more with position
   encoding on.** Falsifier: no interaction between the aug and position-encoding factors.
2. `[HYPOTHESIS]` **A clause memorises an exact bit conjunction, so flipping doubles the patch
   vocabulary the pool must cover.** The cost should scale with the entropy of the patch
   distribution — high for 24-plane colour thermometers, low for 1-bit Canny — which **predicts the
   sign flip in Table IV** (Canny +1.8, adaptive mean +1.8, HOG +1.0, adaptive Gaussian +0.1,
   5×5 colour thermometers **−1.3**). Falsifier: the augmentation penalty does not correlate with
   measured patch-distribution entropy across the four cells. The entropy is free to compute.

Run the 2×2 at **two** encodings — one where the Toolbox says flip helps (adaptive mean 10×10) and
one where it hurts (5×5 colour thermometers) — because the sign flip is the thing to explain.
**4.6 GPU-h for 8 cells at 3 seeds.** I move augmentation **up** my ranking on the strength of the
two mechanisms, but still below the C-2 ladder and M1.

---

## 4. C-4 — T0. **The engineer's 31.6 GPU-h is an artefact of the Booleanization, not of the clause count.**

`[MEASURED: results/tables_p0.md]` every calibration cell is **therm4 (Z = 12)**. But the papers with
CIFAR-10 numbers use **adaptive Gaussian thresholding per channel (Z = 3)**
`[FACT: Sharma et al. AAAI 2023; arXiv:2507.14874 §3.2]`. Cost scales as `P · 2F` with
`F = Z k² + 2(32−k)`, so at 10×10:

    F(10, 3) / F(10, 12) = 344 / 1244 = 0.28  →  the faithful encoding is 3.6x CHEAPER.

| arm | cost, therm4 (as priced) | cost, **adaptive Gaussian (the papers' encoding)** |
|---|---|---|
| `ctm-vanilla` @8 k, 10×10, 60 ep, 3 seeds | 3.9 GPU-h | **1.1 GPU-h** |
| `ctm-vanilla` @80 k, 10×10, 60 ep, 3 seeds | 31.6 GPU-h | **11.2 GPU-h** |
| **`ctm-coalesced` @80 k, 8×8, 30 ep, 3 seeds** | — | **4.3 GPU-h** |

Three consequences:

1. **The 80 000-clause arm is affordable** — 11.2 GPU-h for 3 seeds, not 31.6. And note the alignment:
   80 000 total for a class-owned model **is 8 000 per class**, which is exactly the 2019 paper's own
   budget `[FACT: arXiv:1905.09688 Table 1, "#Class Clauses"]`.
2. **But the arm that should carry an 80 k budget is `ctm-coalesced`, not `ctm-vanilla`.** At
   80 000 clauses / 8×8 / adaptive Gaussian / 30 epochs / T = 15 000 / s = 20.0 / budget 32 it has an
   **exact published target with a standard deviation — 66.42 ± 0.19**
   `[FACT: arXiv:2507.14874 Tables 3 and 16]` — against our 1.00 pp band, and it costs **4.3 GPU-h**.
   The 80 k `ctm-vanilla` has no published number of its own; its nearest "target" is a Toolbox row
   produced by a coalesced `tmu` model.
3. **T0 is discharged by the ladder, not by any single arm.** `ctm-vanilla` at 2/4/8/16/32 k matching
   the Toolbox's own column within §7.5 tolerance at **≥3 of 5 points**, with the C-5 control applied,
   is a strictly stronger reproduction than one cell — and it settles C-2 at the same time. Five
   cells cost ~2.9 GPU-h at adaptive Gaussian.

**Answer**: run 8 k first (1.1 GPU-h); discharge T0 on the ladder; run the exact `ctm-coalesced` 80 k
reproduction (4.3 GPU-h, 3 seeds) because it is the only exactly specified CIFAR-10 configuration in
the bibliography; run `ctm-vanilla` @80 k (11.2 GPU-h) **after** M0 settles the clause convention,
as the paper-faithful point — not before.

**Also worth ~30% of the P3 envelope**: `[MEASURED: results/tables_p0.md]` raising
`max_chunk_elements` cuts 8 k/10×10 from 77.7 to 55.4 s/epoch, 16 k from 151.1 to 103.7, and 40 k
from 329.4 to 270.3. Once the neutrality check lands, roughly a third of P3's compute is recoverable
at zero risk to accuracy. I would prioritise that check.

---

## 5. C-5 — the protocol control. Confirmed, with one correction to my own claim.

**Confirmed.** Record **three** statistics per `existing`-family arm, all from the same recorded curve
and all free:

1. **val-selected test accuracy** — the protocol number, and **the only one used for selection or any
   A-vs-B comparison**;
2. **mean of the last 25 epochs** — the Toolbox and Composites estimator;
3. **peak over epochs** — the Drop Clause and CTM-2019 estimator.

**Reproduction verdicts under §7.5 are evaluated against the paper's own estimator**, not against (1).
Comparing our val-selected number to their last-25-mean compares two different estimators, and that
difference is the systematic penalty.

**Correction to my earlier claim.** I said the penalty was a ≤1 pp *bias*. `[MEASURED:
results/calibration_seednoise.json]` says it is mostly **variance**: val at last epoch
0.36676 ± 0.00549 vs val at selected epoch 0.37088 ± 0.00639 — selection gains **+0.41 pp** with a
similar spread. And the 3-seed band is **1.00 pp** against a **3.00 pp** tolerance, so **it is the
seed band, not the protocol, that eats a third of the tolerance**. The protocol control costs zero
and removes the remaining ambiguity, so it should simply be done; it is not a threat to the budget.

**One caveat to record**: `[MEASURED: calibration_seednoise]` selected epochs were 29, 8, 14, 13 and
**2**, with 7–15 of 30 epochs within 0.5 pp of the best. `selected_epoch` must never be reported as
"when the model was best". For `existing` arms run at the papers' 100–250 epochs, the last-25-mean is
the **lower-variance** estimator of the same quantity, which is another reason it is the fairer target.

---

## 6. C-6 — `ctm-clausesize`. Control, not candidate; it does **not** have to bind; but stop extrapolating from budget 8.

**Confirmed: a control.** `[FACT: arXiv:2301.08190 Table 3]` the mechanism is worth **+0.06** on
CIFAR-2 and **+0.05** on MNIST-with-convolution. Reproducing a +0.06 effect at ~10× cost through
sequential feedback, on a *different dataset* from ours, is not a defensible use of the envelope.

**It does not have to bind — but the present evidence does not say that it fails to bind at the
budget the SOTA actually uses.** `[MEASURED: code/repro/lg003.py, LG-003]` budget **8** gives median
**10.5** at batch 50 — a **31% overshoot**. The SOTA recipes use **budget 32**
`[FACT: arXiv:2309.04801 Table 1; arXiv:2507.14874 Table 16]`. If the overshoot is proportional,
budget 32 → median ≈ 42, which is close enough that the recipe is reproducible as published and
**C-6 evaporates**. Nobody has checked.

**So: one measurement decides whether C-6 is a problem at all.** Budgets {16, 32, 64} at batch 50,
2 000 clauses, 4×4, 30 epochs, 3 seeds — record the *achieved* median clause size.
**0.6 GPU-h.** Plus one `feedback_mode="sequential"` control at the same small scale (**0.6 GPU-h**
with the ~10× factor) to pin the accuracy cost of the non-binding.

**Registry rule** (this is how "what reproduced means" gets handled — by disclosure, not by 10×
compute): every arm records **which dial produced its clause size** — budget under batched feedback,
budget under sequential feedback, specificity `s`, or an explicit density controller — and the report
states it. That closes the concern in C-6 at 1.2 GPU-h instead of 10×.

---

## 7. The thing neither position paper said: **composition wins at matched *total* clause budget, and the win shrinks with budget**

Derivable from `[FACT: arXiv:2406.00704 Table IV]` alone, and nobody in the literature has made the
comparison. The composite of 22 members at `n` clauses each holds `22n` total clauses. Comparing it
against the single-model 5×5 colour-thermometer curve **at the same total**:

| composite | total clauses | composite acc | single model at that total | **Δ** |
|---|---|---|---|---|
| 22 × 2 000 | 44 000 | **79.5** | ~74.5 *(interpolated between the measured 32 k and 64 k points)* | **+5.0** |
| 22 × 4 000 | 88 000 | **80.6** | ~76.2 | **+4.4** |
| 22 × 8 000 | 176 000 | **81.5** | ~77.9 | **+3.6** |
| 22 × 16 000 | 352 000 | 82.2 | ~79.6 *(extrapolated — weaker)* | +2.6 |

Only the first row is pure interpolation; the rest degrade into extrapolation and I flag them. The
defensible statement is:

> `[FACT, derived from Table IV]` **At 44 000–88 000 total clauses, composition over diverse
> Booleanizations beats a single model of the same total clause budget by +4.4 to +5.0 points, and
> the advantage shrinks monotonically as the budget grows.**

**Composition is therefore the most automata-efficient mechanism in the literature by a clear margin,
and it is nearly free to implement.** But it is **confounded with encoding diversity**, and the
control that separates them has never been run:

> **`ctm-composite-samenc`**: a composite of `N` members all on the **same** encoding (different
> seeds only) versus `N` members on `N` **different** encodings, at equal total clauses.

`[HYPOTHESIS]` **same-encoding composition captures ≤ 2 of the 5 points — diversity, not ensembling,
is doing the work.** **Falsifier**: the same-encoding composite lands within 1.00 pp of the diverse
one. **Cost 3.0 GPU-h.** This is the single most decision-relevant cheap arm I can name, and it is
now arm 15 below.

---

## 8. Final ranked P3 shortlist, priced on the measured coefficients

Price model: `s/epoch = (C_total/1000) × coef(k) × F(k,Z)/F(k,12)`, `F = Z k² + 2(32−k)`, with
`coef = {3:2.6, 4:3.7, 5:4.8, 6:5.8, 8:7.3, 10:10.1, 16:16.5}` from
`[MEASURED: results/tables_p0.md]` (all cells therm4 ⇒ Z = 12), default chunk budget, 45 000 train.
**Request to the engineer**: add a `Z` column to the price list — the `F` ratio is my inference, and
it is load-bearing for every adaptive-Gaussian and colour-thermometer arm (±3.6× either way).

| # | Arm | Target | Config | Seeds | GPU-h |
|---|---|---|---|---|---|
| **1** | `ctm-vanilla` @8 k | Toolbox adaptive-Gauss column, 63.0 @8 k | 10×10, adaptive (Z=3), 60 ep | 3 | **1.1** |
| **2** | **M0 clause-convention probe** | Toolbox 2 000 column (64.5) | 2 k total vs 20 k total, 5×5 CT | 1 | **1.6** |
| **3** | `ctm-boolean-*` — 10 encodings @2 k | **22 published cells** (Table IV @2 k + Composites Table 2) | per-paper (k, T, s, weighted), 100 ep | 1 | **3.6** |
| **4** | `ctm-boolean-*` — 4 composite members | Composites Table 2 members (57.0–63.5) | as above | 3 | **4.6** |
| **5** | `tm-composite` | **75.1%** — α from **validation** and from test, both reported (O-1) | inference only | — | **0** |
| **6** | **`ctm-coalesced` @80 k — the exact reproduction** | **66.42 ± 0.19** | 8×8, adaptive, T=15 000, s=20, budget 32, 30 ep | 3 | **4.3** |
| **7** | `ctm-coalesced` @8 k | scaling point for 6 | as above | 3 | **0.4** |
| **8** | **`ctm-capacity` ladder — settles C-2** | Table IV columns, both encodings | 2/4/8/16/32/**64 k**, colour-therm 5×5 **and** adaptive 10×10, 60 ep; +2 extra seeds at CT 8 k and 32 k | 1 (+3 at 2 anchors) | **36.2** |
| **9** | `ctm-dropclause` p ∈ {.25,.5,.75} | shape 73.2 / 75.1 / 72.6 | vanilla config; p=0 is arm 1 | 3 | **2.7** |
| **10** | `ctm-vanilla` @80 k (= 8 k/class, the 2019 budget) | paper-faithful point; run **after** M0 | 10×10, adaptive, 60 ep | 3 | **11.2** |
| **11** | `ctm-patch` window sweep, **matched automata** | none — that is the point; `THEORY.md` §6.2 predicts unimodal in k | k ∈ {3,4,5,8,10,16}, 30 ep | 1, +3 at best two | **11.0** |
| **12** | `ctm-weighted` / unweighted control | none | vanilla config | 3 | **1.1** |
| **13** | `ctm-augment` 2×2 × 2 encodings (§3) | resolves C-3 | A/B/C at matched data budget × position-encoding | 3 | **4.6** |
| **14** | `ctm-multiview` CoTM | GraphTM's **+3.86 over CoTM** | 8 k, 8×8, adaptive+CT4 (Z=15), 30 ep | 3 | **1.8** |
| **15** | **`ctm-composite-samenc`** (§7) | separates ensembling from diversity | 6 same-encoding members @2 k | 1 | **3.0** |
| **16** | `ctm-clausesize` {16,32,64} + sequential control (§6) | achieved median size at the SOTA budget | 2 k, 4×4, 30 ep | 3 | **1.2** |
| **17** | `ctm-cnnfeature` (**M1**) | none — the decomposition arm | flat TM over Booleanized frozen CNN features, 2 budgets | 3 | **0.5** |
| **18** | `mctm-calibrated` re-run | this repo's 41.80 ± 0.98 | 640 cl, 4×4, 30 ep | 3 | **0.1** |
| | **matched-budget + matched-RF controls** (PLAN §8 P3) | | | | folded in |
| | **TOTAL** | | | | **≈ 88** |
| | **+20% contingency** (Z-scaling risk, per-paper hyperparameters) | | | | **≈ 106** |

Fits the ~120 GPU-h envelope with ~14 GPU-h of margin, **before** the ~30% recoverable from the
chunk-budget neutrality check (§4). The C-1 screen set (S0–S5, §1.4) is P5 and costs ~0.05 GPU-h.

**What changed from my earlier list**: my previous estimate was 1.8× pessimistic *and* it priced the
paper-faithful encoding wrong in the other direction. The net effect is that the whole list, plus the
80 k reproduction, plus a capacity ladder to 64 k, now fits — which it did not before.

## 9. Cut order, first to last

1. **#11 `ctm-patch` (11.0)** → P4. It has no published target and `THEORY.md` §6.2 predicts its
   shape; it is the most deferrable.
2. **#10 `ctm-vanilla` @80 k (11.2)** → run only if #1 and #8 both reproduce. Its published target is
   weaker than #6's, which is already in the list at a third of the cost.
3. **#8's 64 k points (9.5 + 3.0)** → the ladder still settles C-2 at 32 k; the 64 k cells only pin
   the last doubling.
4. **#15 (3.0)** → P5, where it belongs if composition becomes a candidate rather than a baseline.
5. **#13's second encoding (1.1)** → keep the colour-thermometer 2×2, which is the surprising case.

I would **not** cut #1–#8 or #16–#18. #8 is the round's decisive measurement and #17 costs 0.5 GPU-h.

## 10. Corrections I am making to `THEORY.md`

- **§5.2 point 3** — "almost never an OR over more than one term" is withdrawn and replaced with the
  measured distribution; **Type Ib added as the density floor** the original argument omitted.
- **§5.2 point 4** — the prediction is restated: not "count pooling is a no-op" but "a count pool with
  κ ≥ 2 is density control applied through the forward pass, and is therefore confounded with an
  explicit controller"; falsifier updated to the S0–S5 screen.
- **§7 G1** — prediction and falsifier rewritten around the measured quantiles and the entropy
  arithmetic; **G5 promoted** on the strength of the DL expert's binary-activation decomposition
  (binary weights ≈ free, binary activations ≈ −12.4 on ImageNet — the all-or-nothing clause is the
  analogue of the expensive half).
- **§4.2** — the per-doubling slope (+2.18 average, +1.7 last) and the extrapolated cost of parity
  added, replacing the bare "still climbing" reading.

---

## Addendum — configuration block delivered, and the shortlist repriced

Written 2026-09-20 in response to the orchestrator's four specifics. The extraction is in
`LITERATURE_TM.md` **§8** (new). Headlines here; the engineer should work from §8.

### A. The one finding that changes everything: **the Toolbox's clause counts are PER CLASS**

Two independent confirmations, so this is not an inference:

1. `[FACT: arXiv:2406.00704 §II-B]` — *"A TM utilizes a group of n conjunctive clauses **for each
   class**"*, with Eq. (2) summing `j = 1..n/2` positive minus `j = 1..n/2` negative **per class i**.
2. `[FACT: cair/tmu vanilla_classifier.py]` — every reference script uses
   `TMClassifier(number_of_clauses=…)`, which calls
   `clause_banks.populate(range(number_of_classes))`: **one bank of `number_of_clauses` clauses per
   class**, `positive_clauses = [1]*(n//2)+[0]*(n//2)` inside each bank, `transform()` returning
   `(N, n_classes × number_of_clauses)`.

> **"2 000 clauses" = 20 000 total. "8 000" = 80 000 total. "64 000" = 640 000 total.
> The 82.8% composite holds 22 × 640 000 = 14.08 million clauses.**

The model is **class-owned with fixed polarity** — `ConvTsetlinMachine`, confirming `ARMS.md` A2 and
settling A1. `T` is compared against one class's vote sum, so `ARMS.md` convention 2 (`T_ratio` ×
`n_clauses/n_classes`) is the right normalisation; only the absolute counts move. **M0 is now a
confirmation, not a discovery** — its result is predicted, and it should still run because a
prediction that cheap deserves a check.

**Consequences I did not want but have to report**: the Booleanization family and the capacity ladder
must run at **20 000 total** clauses to be scoreable against the paper's cheapest measured cell.
That is a 10× increase on the arms that carry the reproduction, and it is what consumed the headroom
I reported last time. The list below absorbs it.

### B. Item 2 — the target `X`. There is **no 8 000-total cell in the paper**

Its "8,000" column is 80 000 total. Full table in §8.3; the operative rows:

| encoding | @ **20 000 total** (paper's "2,000") | @ **80 000 total** (paper's "8,000") | @ 8 000 total *(extrapolated)* |
|---|---|---|---|
| **5×5 Colour Thermometers** | **64.5** | **69.1** | ~61.5 |
| HOG (flat) | 64.2 | 66.9 | ~62.4 |
| 5×5 Adaptive Gaussian | 57.9 | 63.0 | ~54.5 |
| 10×10 Canny | 48.7 | 53.4 | ~45.6 |

**Run the pre-flight at 20 000 total, 5×5 colour thermometers, 50 epochs — `X = 64.5`, minus the
epoch correction below → `X ≈ 63.3`.** Decision rule: **≥ 60.3 proceed · 55.3–60.3 hyperparameter
fault · < 55.3 implementation fault.** No extrapolation, the paper's own cell, **2.1 GPU-h**.

### C. Item 3 — HOG. It is an **OpenCV** descriptor with two non-default settings, and it is **flat**

`cv2.HOGDescriptor((32,32), blockSize=(12,12), blockStride=(4,4), cellSize=(4,4), nbins=18,
derivAperture=1, winSigma=-1.0, histogramNormType=0, L2HysThreshold=0.2, gammaCorrection=True,
nlevels=64, signedGradient=True)`, Booleanized as `hog.compute(img) >= 0.1`.
**5 832 features.** `nbins=18` and `signedGradient=True` are both non-default; unsigned 9-bin HOG
would give 2 916 features and different content. It takes the uint8 3-channel image directly — do
**not** greyscale first. And `patch_size = 0` with no `patch_dim` passed: **the HOG specialist is a
flat, non-convolutional, unweighted TM with `T = 50`.** Table III's "32×32 convolution window" means
"no convolution". Exact parameters for the other five encoders are in §8.5 — note **adaptive Gaussian
uses `blockSize=3`, not the library's default 11**, and **`ColorThermometerEncoder(n_bits=8)` is
bit-identical to the reference**.

### D. Item 4 — **250 epochs is an unexamined default. 100 epochs costs ≤ 0.5 pp.**

I digitised Fig. 3's embedded vector paths; the recovered finals reproduce Table IV's composite row
to **≤ 0.3 pp**, so the extraction is validated. Full table in §8.6.

| clauses/class | ep 25 | **ep 50** | **ep 100** | ep 250 | first epoch within 1.0 pp |
|---|---|---|---|---|---|
| 2 000 | 77.7 | **78.5** | **79.2** | 79.7 | 40 |
| 8 000 | 79.4 | **80.5** | **81.2** | 81.7 | 42 |
| 64 000 | 80.8 | **81.6** | **82.4** | 82.7 | 47 |

> **100 epochs costs 0.3–0.5 pp — inside the 1.00 pp seed band, i.e. free, for a 60% saving.
> 50 epochs costs 1.1–1.2 pp — at the band: fine for ladders, screens and pre-flights with the
> −1.2 pp correction, not for a headline reproduction.**

`[HYPOTHESIS]` this transfers from the composite to individual specialists; falsifier is our own
per-epoch curve for one specialist showing >1 pp still to gain after epoch 100, and it costs nothing
because the curve is already logged. **This is the single largest budget saving in the round and it
rests on a measurement, not a preference.**

### E. C-6 — **I was wrong about the direction, and I withdraw the recommendation**

I said the overshoot was probably proportional and that budget 32 would therefore *nearly* bind. The
engineer measured it: the overshoot is **absolute**, so the budget **holds** at 16/32/64 at batch 50
with no sequential feedback and no 10× cost. And `[MEASURED: ctm-small-budget32]` a budget of 32 is
worth **+7.50 pp** (44.58 ± 0.55 vs 37.08 ± 0.61, seeds disjoint, 4.8× fewer literals per image,
+1.7% wall-clock).

**I accept this without reservation and withdraw "`ctm-clausesize` is a control worth +0.06".** Two
things I can now add that make it worse for my earlier position:

1. `[FACT: all 22 reference scripts]` **`max_included_literals = 32` is set in every one of them, and
   Table III does not mention it.** The paper's numbers *all* carry a literal budget. So a
   reproduction without one is not a reproduction, and our un-budgeted baseline was not merely
   mis-tuned — it was a different algorithm from the one being scored against.
2. This is the third undocumented-default finding in this paper family (budget 32; no validation
   split; transductive `α_t`). `[HYPOTHESIS]` **the modal cause of a T0 shortfall in this programme
   is an undocumented default, not an implementation fault** — which is an argument for the
   engineer's cheap-debug band being wide, and for reading the reference code before declaring any
   `GAP`. I would put that in DR-001 as a standing rule.

**Revised status**: a clause-size budget is a **default that belongs on every arm** (`budget = 32`
unless a paper says otherwise), and `ctm-clausesize` becomes a small sweep {16, 32, 64, none} to
locate *our* optimum rather than to reproduce a +0.06 effect. The 10×-cost sequential control is
**dropped entirely** — it was only needed to make a non-binding budget bind, and the budget binds.

### F. The Z repricing — accepted, and the per-class finding dominates it anyway

Accepted: my `F`-ratio predicts literal counts exactly but not cost, because per-batch overhead does
not scale with `F`. I have repriced on the engineer's measured savings — **1.5× at 4×4, 1.8× at 5×5,
2.9× at 10×10** (midpoints) — and kept a **1.6× penalty** for 24-plane colour thermometers, itself
compressed from the naïve `F` ratio of 1.85 by the same overhead logic. **That CT penalty is an
estimate and it is now load-bearing for half the list — one measured therm4-vs-colour-thermometer
cell at 5×5 would retire it.**

For the record, the two corrections run opposite ways and the second is much larger: the Z
correction makes adaptive arms **1.5–2.3× more expensive than I said**, and the per-class finding
makes every paper-scored arm **10× more expensive**. Net, my "≈88 GPU-h" was optimistic by ~2.3×.

### G. Both engineer cuts: **agreed, and I can strengthen both**

**Cut 1 — the 80 000-clause 10×10 `ctm-vanilla`.** Agreed, and it was already #2 in my cut order. The
per-class finding makes the case stronger, not weaker: the 2019 paper's budget is 8 000 *per class*
= 80 000 total, so an 80 000-total arm is the 2019 configuration — and the 2019 paper reports **no
CIFAR-10 number**, so there is still nothing to score it against. The capacity ladder passes through
80 000 total anyway, which is *also* the paper's "8,000" cell, so the ladder subsumes it twice over.
**Dropped.**

**Cut 2 — the faithful 64 000-clause composite.** Agreed, and the engineer's 400–600 GPU-h for one
seed is **an order of magnitude too low**, because 64 000 is per class: the faithful composite is
**22 × 640 000 = 14.08 M clauses**, i.e. **≈ 4 000–6 000 GPU-h for one seed**. It is not reproducible
here by a factor of 30–50, not 4. **Dropped, and I would state the corrected figure in DR-001** — it
is the cleanest possible justification for reproducing the 2023 composite (75.1%) instead.

This also revises `THEORY.md` §4.3: the clause counts needed for parity are **10× larger** than I
reported — ≈ 1.3 × 10⁷ total clauses for 82.8% as a single model, ≈ 3 × 10⁷ for small-CNN parity.
The conclusion ("logarithmically open, practically closed") is 10× stronger. §4.4's composition
result is **unchanged**, because both sides of that comparison scale by the same factor.

### H. Repriced P3 shortlist — two tiers, **117.4 GPU-h**

Pricing: measured `coef = {3:2.6, 4:3.7, 5:4.8, 6:5.8, 8:7.3, 10:10.1}` s/epoch per 1 000 **total**
clauses at therm-4; ÷ {4×4: 1.5, 5×5: 1.8, 8×8: 2.4, 10×10: 2.9} for adaptive-type encodings;
× 1.6 for colour thermometers *(estimate — please measure one cell)*; HOG flat.

**Tier 1 — scoreable against a published number. 86.4 GPU-h. Do not cut.**

| arm | config | target | GPU-h |
|---|---|---|---|
| **PF** pre-flight | 5×5 CT, **20 000 total**, 50 ep, 1 seed | **X ≈ 63.3** (§B) | **2.1** |
| 1 `ctm-vanilla` | 5×5 adaptive Gaussian, 20 000 total, 60 ep | 57.9 − 1.0 | 2.7 |
| 3 Booleanization family | 10 encodings, 20 000 total, 50 ep, 1 seed | **11 published cells** (§8.3) | 14.0 |
| 4 composite members | 3 CT + HOG, 20 000 total, 100 ep, 3 seeds | Composites Table 2 members | 31.1 |
| 5 `tm-composite` | α from validation **and** test (O-1) | **75.1%** | 0 |
| 6 `ctm-coalesced` @80 000 total | 8×8 adaptive, T=15 000, s=20, budget 32, 30 ep | **66.42 ± 0.19** | 6.1 |
| 7 `ctm-coalesced` @8 000 total | scaling point for 6 | — | 0.6 |
| 8 **capacity ladder** | 5/10/20/40/80 k total, 5×5 CT **and** 10×10 adaptive, 60 ep, 1 seed | Table IV at the 20 k and 80 k points; **settles C-2** | 28.8 |
| 16 `ctm-clausesize` | {16, 32, 64, none} @2 000 total — *our* optimum, no sequential control | — | 0.6 |
| 17 `ctm-cnnfeature` (**M1**) | flat TM over Booleanized frozen CNN features, 2 budgets | — | 0.4 |
| 18 `mctm-calibrated` | re-run under the protocol | 41.80 ± 0.98 | 0.1 |

**Tier 2 — mechanism and control, no published target. 31.0 GPU-h. Cut from the bottom.**

| arm | config | GPU-h | cut order |
|---|---|---|---|
| 13 `ctm-augment` 2×2 (aug × position-encoding) | 5×5 CT, 8 000 total, 60 ep, 3 seeds | 12.3 | **1st** |
| 15 `ctm-composite-samenc` | 6 same-encoding members, 8 000 total | 7.9 | **2nd** |
| 8c 2 extra seeds at the 20 000 ladder anchor | | 5.1 | **3rd** |
| 9 `ctm-dropclause` p ∈ {.25,.5,.75} | 8 000 total, 60 ep, 3 seeds | 3.2 | 4th (drop p=.75 first) |
| 14 `ctm-multiview` CoTM | 8 000 total, 8×8 | 1.5 | keep |
| 12 weighted / unweighted control | 8 000 total | 1.1 | keep |

**`ctm-patch` (window sweep, ~10.7 GPU-h) moves out of P3 to P4 entirely** — it has no published
target and `THEORY.md` §6.2 predicts its shape, so it is diagnostic rather than reproductive.

**Three standing changes to every arm, from §8.7 and §E**: `max_included_literals = 32` unless the
paper says otherwise; **100 epochs for reproductions, 50 for ladders and screens** with the −1.2 pp
correction; and log the paper's own estimator (last-25-epoch mean, peak) alongside the
validation-selected number.

**One request back**: a single measured therm-4-versus-colour-thermometer cell at 5×5. The 1.6×
penalty is my estimate and it prices roughly half of Tier 1; if the true figure is 1.85× the list is
~8 GPU-h heavier, and if it is 1.3× it is ~8 lighter.

---

## Addendum 2 — the clause-size budget, accounted for and pre-registered

The orchestrator asked whether `THEORY.md` §5.2 accounts for the +7.50 pp, and whether it predicts
where the optimum budget sits. **Answer: §5.2 as written did not — it had the sign of the dominant
force wrong — and it now does. The corrected account makes four predictions, registered below before
the sweep runs.** Full derivation in `THEORY.md` §5.4 (new) and §5.2 point 2c (new).

### What §5.2 got wrong

§5.2 said the feedback rule makes specificity free and drives clauses toward knife-edge detectors.
"Free" is not "driven", and I never said what *drives* composition. The forces are asymmetric in a
way I missed:

* **adding literals is deterministic** — Type Ia memorises every **True** literal of the chosen patch
  with probability 1 under `boost_true_positive` (the library default *and* the reference default)
  `[FACT: functional.py:230, 245]`, and Type II pushes every **False** excluded literal one step
  toward inclusion with no `s`-gate at all `[FACT: functional.py:253-257]`;
* **removing literals is probabilistic** at rate `1/s` — and that is the *only* removal force
  `[FACT: functional.py:250-251]`.

A deterministic inflow against a `1/s` outflow has a **wide** equilibrium, not a narrow one, and the
measurement says exactly that: `[MEASURED: results/ctm-small_seed*.json]` an unconstrained clause
settles at **156.4 of the 248 literals it could possibly include — 63%**. That is not a pattern, it
is **a memorised patch**, matching 5.1 × 10⁻⁴ of patches and firing on 7% of images.

### Why the cap buys 7.50 pp, quantitatively

| | clause length | % of 248 includable | firing rate | per-literal selectivity `−ln q` | accuracy |
|---|---|---|---|---|---|
| unconstrained | 156.4 | **63.1%** | 0.0695 | 0.0485 | 37.08 ± 0.61 |
| budget 32 | 32.5 | 13.1% | **0.1586** | **0.1923** | **44.58 ± 0.55** |

The cap does not merely shorten clauses — **it forces a choice of which literals survive**, and the
survivors are **4.0× more selective**. The clause ends up 4.8× shorter, built from individually rarer
literals, and still matching **3.8× more** patches.

**And it beats `s` for a structural reason**: `s` is a per-literal, per-event Bernoulli(1/s) removal,
while the cap is a **deterministic, per-clause, per-commit** conversion of Type Ia into Type Ib
`[FACT: functional.py:232-242]`. Under batched feedback the deterministic inflow aggregates over the
whole mini-batch before a single commit, so a probabilistic outflow is out-raced and a deterministic
gate is not. **The cap is the only deterministic forgetting force in the algorithm.**

### Four pre-registered predictions (details and falsifiers in `THEORY.md` §5.4.4)

| | prediction | falsifier |
|---|---|---|
| **P1** | The optimum is **at or below 32 and the curve is flat beneath it**: `acc(32) > acc(64) > acc(none)`, with `acc(16)` inside the 1.00 pp band of `acc(32)`. Reason: extra coverage is spent inside images the clause already fires on — `\|M\|` given firing already rises 6.1 → 10.2 as the cap tightens | `acc(64) > acc(32)`, or `acc(16)` more than 1.00 pp below `acc(32)` |
| **P2** | The budget's **entire** effect is mediated by firing rate: accuracy plotted against *achieved* firing rate collapses the `{budget} × {s} × {density controller}` points onto one curve | at matched firing rate, the budget arm and the `s` arm differ by more than the seed band |
| **P3** | The +7.50 pp is **largely a batched-feedback artefact**: under `feedback_mode="sequential"` the gap shrinks to **≤ +1 pp**, consistent with CSC-TM's +0.06 at an unconstrained length of 60.4 against our 156.4 | sequential also shows ≥ +5 pp |
| **P4** | `b*` is **roughly absolute, not proportional to encoding width**: it varies by < 2× while `n_features` varies 2.6× (248 → 654) and `P` varies 1.6× (841 → 529), because the coverage relation is only *logarithmic* in patch count | `b*` tracks `n_features` proportionally — which would also mean the Toolbox left accuracy on the table by using a single budget of 32 across all seven encodings |

**P3 is the one I most want run**, and it is cheap at `ctm-small` scale. It decides whether §5.4.3
enters the report as a **caveat about our approximation** or as a **finding about Tsetlin machines**
— and those are very different claims.

### A second, independent fault in the same baseline — please fix before the sweep

`[MEASURED: results/ctm-small_seed*.json hp]` `ctm-small` has 2 000 total clauses / 10 classes = 200
per class = **100 positive**, unweighted → **maximum achievable class sum = 100**, against a
configured **`T = 160`**. The vote can never reach the margin, so the feedback probability never
anneals: **0.478** unconstrained, **0.450** at budget 32. **The margin is doing nothing in either
arm** — which is exactly the failure `ARMS.md` convention 2 warns about.

One-line fix: **`T_ratio` must be taken against the positive half, `n_clauses_per_class / 2`** (for
an unweighted class-owned model; with `weighted=True`, against the summed positive weights).
`[HYPOTHESIS]` a reachable `T` moves the optimum budget **up**, because a lower required firing rate
tolerates longer clauses. **Until that is fixed, the budget sweep and the `T` setting are
confounded**, and P1 is a claim about `ctm-small as configured`, not about convolutional TMs. This
also means the +7.50 pp itself is measured at an inoperative margin — it will very likely survive a
correct `T`, but the size of it may not.

### Two notes on process, not on method

1. **Items 1–4 and the repricing were delivered before the session limit**, not after it:
   `LITERATURE_TM.md` **§8** (lines 322–511) carries the clause convention, per-specialist `T`/`s`,
   the target table, the HOG spec and the epoch answer; this file's **Addendum** (lines 687–868)
   carries the repriced two-tier list at **117.4 GPU-h**. Nothing there needs redoing.
2. **`DR-001`'s "~33 GPU-h for Tiers 0–1" and my 117.4 GPU-h differ by 3.5×**, and I flag it as a
   budget risk rather than reopening a settled decision. The gap is almost certainly the **per-class
   clause convention** (§8.1): any arm scored against the Toolbox's table must run at **20 000 total
   clauses**, not 2 000, and that is a 10× multiplier on the Booleanization family, the composite
   members and the capacity ladder. If DR-001's figure was set before §8.1 landed, Tier 0–1 is
   under-funded by roughly that factor and it will surface as the pre-flight over-running. Worth one
   check against the numbers in this file's Addendum §H before the queue is launched.
