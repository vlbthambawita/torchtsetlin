# Arm registry

Nothing runs that is not registered here first. This table is how the orchestrator budgets compute
and how the report's method table is written.

`family`: `baseline` (CNN), `existing` (re-implemented from a paper), `candidate` (ours),
`control` (matched-budget or ablation control).
`status`: `planned` | `implemented` | `running` | `done` | `GAP` (reproduction failed) | `dropped`.

Source of truth for the builders and defaults is `code/arms.py`; this file is the human index.
`python code/arms.py` prints the live registry.

| arm | family | what it is (one sentence) | source | command line | cost (GPU-min x seeds) | status |
|---|---|---|---|---|---|---|
| `ctm-vanilla` | existing | The 2019 convolutional TM: class-owned clauses, a clause fires on an image iff it matches at least one patch, feedback applied at one randomly chosen matching patch. **80 000 clauses = 8 000 per class**, the paper's own reading [FACT: LG-010 A1]. | Granmo et al. 2019, arXiv:1905.09688 | `python code/run_arm.py --arm ctm-vanilla --seed K --eval-n 2000 --m2-n 500` (80000 clauses, 10x10, 60 epochs) | **632 s/epoch → 10.5 h/seed → 31.6 GPU-h x 3** (3090 only) [MEASURED: calib_large_gpu0] | implemented, `config_status=provisional`; **ROUND-1 recommends cutting it** — no published CIFAR-10 number exists at 10x10 to score it against |
| `ctm-vanilla-8k` | existing | The same arm at 8 000 clauses in **total** — the unqualified reading several later papers use, at a tenth of the budget. Settles LG-010 A1 by measurement. | Granmo et al. 2019, arXiv:1905.09688 | `python code/run_arm.py --arm ctm-vanilla-8k --seed K` | 78 s/epoch → 1.3 h/seed → **3.9 GPU-h x 3** at 10x10; **1.6 GPU-h x 3 at 5x5** [MEASURED: calib_grid_gpu0, calib_ladder5x5_gpu0] | implemented, `config_status=provisional` |
| `ctm-small` | control | A cheap scaled-down `ctm-vanilla` (same model class and code path, 2000 clauses, 4x4 patches) used for the P0 seed-noise band, screens and calibration. | — | `python code/run_arm.py --arm ctm-small --seed K` (2000 clauses, 4x4, 30 epochs) | **6 min x 5 = 30 GPU-min** (3080) [MEASURED: ctm-small_seed0..4, 355-369 s/seed] | done — **test 37.08 +- 0.61 pp**, the programme's seed-noise band |
| `ctm-coalesced-small` | control | `ctm-small`'s clause budget in a single shared, weighted clause pool — the matched-budget control for what coalescing buys. | Glimsdal & Granmo 2021, arXiv:2108.07594 | `python code/run_arm.py --arm ctm-coalesced-small --seed K` | ~6 min x 3 (3080) [estimate from ctm-small] | implemented |
| `cnn-ctmshape-max` | control | A CNN with the CTM's exact computational shape — one conv layer (2000 filters, 4x4 patches, stride 1) over the same 12-plane therm4 Boolean tensor, **global max** over the 841 patch positions, linear vote — trained by SGD. Max over binary indicators *is* the CTM's OR, so this isolates architecture from learning algorithm. | — (ours) | `python code/cnn/train.py --arm cnn-ctmshape-max --seed 0 --screen --device cuda:1` | ~2 min x 1 (3080) | done |
| `cnn-ctmshape-sum` | control | As `cnn-ctmshape-max` but **global mean** over patch positions — the counting pool of PLAN §9.2.1, priced on a substrate that can be trained exactly. The pair is a **CNN-side proxy for a TM-side question**; see the confound recorded in both screen records' `notes`. | — (ours) | `python code/cnn/train.py --arm cnn-ctmshape-sum --seed 0 --screen --device cuda:1` | ~2 min x 1 (3080) | done |

| `ctm-therm5` | existing | The Toolbox's best single specialist: 5×5 colour thermometers (n_bits=8), weighted, T=3000, s=5.0, budget 32. **`config_status="paper"`** — fully specified; one quantified deviation (50–100 epochs vs 250, worth 1.1–1.2 / 0.3–0.5 pp, corrected in the target). | Grønningsæter et al. 2024, arXiv:2406.00704 Table IV | `python code/run_arm.py --arm ctm-therm5 --seed K --epochs 50` | **2.51 h/seed @20 000 total**; 4.80 @40 000; 8.97 @80 000 [MEASURED: calib_perclass_gpu1] | T0 pre-flight running |
| `ctm-therm5c` | control | `ctm-therm5` on the **coalesced** builder. Retained as a control only — A7 is settled in favour of class-owned by the reference code, so this is no longer a reproduction candidate. | same | `--arm ctm-therm5c` | as above | implemented |
| `ctm-small-cifar2-unc` | control | `ctm-small` on a **CIFAR-2** vehicle/animal merge of the identical split, with **no** clause-size budget. Therm-4, 4x4, s=10, 2 000 total clauses, 30 epochs; `T` set by the same rule as its CIFAR-10 sibling (0.8 x the achievable class sum, which at 2 classes is 500, so **T = 400**, not 80). | — (ours); the question is CSC-TM's, arXiv:2301.08190 Table 3 | `python code/queue_runner.py --gpu 1 --jobs jobs/p3_cifar2.txt --seeds 0 1 2` | [HYPOTHESIS] **0.15-0.25 h/seed**, NOT the 0.104 h of `ctm-small`: at 2 classes the feedback rule touches the whole clause pool per example instead of a fifth of it (see the job file). To be measured from seed 0. | implemented, queued (wave 2) |
| `ctm-small-cifar2-b32` | control | The same at clause-size budget 32. The **pair** is the only experiment that separates the *bloat* and *headroom* explanations of why we measure +9.46 pp where CSC-TM measured +0.06 (THEORY.md 5.5.6). | same | same job file | same | implemented, queued (wave 2) |
| `ctm-hog` | existing | The HOG specialist: `cv2.HOGDescriptor` at the reference settings (32×32 window, 12×12/4×4/4×4, **18 signed** bins, gamma, `>= 0.1`) → **5 832 flat bits**, run as a flat TM via `patch_size=1`, unweighted, T=50, budget 32. | Granmo et al. 2023 arXiv:2309.04801 Table 1 | `python code/run_arm.py --arm ctm-hog --seed K --epochs 50` | **2.04 h/seed @20 000 total — the ONLY reachable point; 40 000 and 80 000 OOM a 24 GB card** | implemented |

P3 adds the rest of the shortlist (PLAN §9.1) after `DR-001`; the funding *order* is `DR-003`
Decision 3.

### Screen records (PLAN §8 P5 protocol: 10 000 train images, 15 epochs, 1 seed, val-selected)

Screens live in `screen/`, never in `results/`, and are never quoted as results. `--screen` applies
the protocol in the harness rather than leaving it to the command line, so a record in `screen/` is
guaranteed to have been produced under it.

| screen record | arm | purpose |
|---|---|---|
| `screen/cnn-ctmshape-max_seed0.json` | `cnn-ctmshape-max` | OR-pool baseline for the counting-pool argument |
| `screen/cnn-ctmshape-sum_seed0.json` | `cnn-ctmshape-sum` | counting-pool comparison at identical filters/recipe |

---

## Harness conventions that every arm inherits

These are decisions, not defaults. They are recorded here because they change what a number means.

1. **`hp.n_clauses` is always the TOTAL clause count.** A class-owned model divides it by
   `n_classes`; a coalesced model uses it directly. Without this convention "8000 clauses"
   means 8000 or 80000 depending on the arm and every clause-budget comparison in the report
   is wrong. Both `n_clauses_total` and `n_clauses_per_class` are recorded.
2. **Every arm with a paper carries that paper's own `T` and `s`.** There is no safe shared
   default: `T / (total clauses)` spans **0.025 to 1.5** across the bibliography [FACT: LG-010 A3].
   The `T_ratio = 0.8` fallback survives only for arms with no paper. An arm that used the fallback
   records `hp.config_status = "provisional"` and `hp.T_source = "ratio-default"`, the runner
   prints a warning, and `record.admissible_as_reproduction()` refuses it as evidence for a
   "reproduced" or `GAP` verdict. `validate()` still accepts it — a provisional run is a valid
   record, just not a reproduction.
3. **Every arm records its batch size and is compared only at equal batch size** — batched feedback
   tracks the sequential algorithm at 10-50 and degrades noticeably at 200 (repo `CLAUDE.md`).
4. **Selection is `argmax(val_acc)`**, enforced by `record.validate()`, which recomputes it and
   refuses a record that disagrees. Test is evaluated once, after the training loop has ended, from
   a restored best-validation snapshot.
5. **A median clause length above half of `n_features` is a defect signature, not an
   observation.** [MEASURED: ctm-small-budget32] Our unconstrained baseline sat at 175 of 248
   features and left **7.7 percentage points** on the floor; capping at 32 recovered them while
   evaluating **5.4× fewer literals per image**. Every arm carries a clause-size budget unless it
   has a stated reason not to, and any arm breaching the threshold is mis-tuned until shown
   otherwise.
6. **Diagnostics are collected during the run** (clause length, negation fraction, firing rate,
   dead/saturated/used fractions, and **M2**, the per-clause match-count distribution), per epoch,
   for every clause-based arm. `validate()` requires M2 for any arm with more than one patch.
7. **`max_chunk_elements` is part of the measurement, not of the model.** It changes throughput by
   1.2-1.5x and peak memory by up to 13x with no change to the algorithm (LG-004), so it is
   recorded in `hp` and arms are compared only at equal values. [MEASURED: ctm-small-chunk29] It is **accuracy-neutral** (+0.18 pp on 3-seed means, inside the
   1.00 pp band), so it is released for use. But [MEASURED: calib_ladder5x5_gpu0] it is worth
   1.2-1.4x only in the 2 000-8 000 clause range and 1.15x at 80 000, so there is **no
   programme-wide default**: use it for screens, record it, compare only at equal values.

## Open ambiguities (to settle in P1, before any `GAP` verdict is issued)

| # | Ambiguity | Current harness reading | Why it is load-bearing |
|---|---|---|---|
| A1 | Is a paper's "number of clauses" per class or in total? | **SETTLED** [FACT: LG-010 A1]: the 2019 paper's column is "#Class Clauses", so per class. `hp.n_clauses` stays the *total* by convention, and `ctm-vanilla` is registered at 80 000. `ctm-vanilla-8k` measures the alternative reading. | a factor of 10 in the automata budget, and **29 GPU-hours** in P3 |
| A2 | `ctm-vanilla` = class-owned `ConvTsetlinMachine` or shared-pool `ConvCoalescedTsetlinMachine`? | **SETTLED** [FACT: LG-010 A2]: class-owned. | using the coalesced model for "vanilla CTM" reproduces the wrong paper and breaks matched-budget comparisons against `ctm-coalesced` |
| A3 | `T` and `s` for CIFAR-10 in each paper | **OPEN, and no default is safe** [FACT: LG-010 A3]: measured `T/clauses` spans 0.025-1.5. Per-paper values go into each `ArmSpec.defaults`; until then `config_status=provisional`. | a reproduction cannot be called a `GAP` while its hyperparameters are ours and not the paper's |
| A4 | Patch position encoding: thermometer (library) or one-hot (some papers) | **SETTLED** [FACT: LG-010 A4]: thermometer. | changes `n_features` and therefore the automata budget |
| A9 | What **is** "CIFAR-2"? | **OPEN, and no source settles it.** CSC-TM reports a CIFAR-2 row and defines neither the merge, the Booleanization, nor the convolution window for it [FACT: arXiv:2301.08190 Table 3 + fn. 6 — the window is stated for "MNIST w/conv." and never for CIFAR-2]. `data.py::CIFAR2_MAP` implements the **vehicle (airplane/automobile/ship/truck) vs animal** merge, on the ten-class stratified split, un-re-stratified, so `split_hash` is shared with every other arm. This is a convention we adopt, **not** a citation. | It decides whether our CIFAR-2 **absolute** accuracy may be set beside CSC-TM's 94.18%. It does **not** affect the quantity the arms exist to measure — the budget Δ between `ctm-small-cifar2-unc` and `-b32`, which is taken between two arms sharing whatever the map is. **The report may quote our Δ against theirs; it may not quote our accuracy against theirs without this caveat in the same sentence.** |
| A10 | `T` when the task changes `n_classes` | **DECIDED, engineer, 2026-09-21**: the *rule* is held fixed, not the numeral. A class-owned model's vote capacity is `n_clauses / n_classes / 2`, so 2 000 clauses give a capacity of 100 at ten classes and 500 at two. `ctm-small-cifar2-*` therefore run `T = 0.8 x capacity = 400`, matching `ctm-small-T80`'s ratio rather than its `T`. | Holding `T = 80` would have multiplied the reachable vote by 5 while holding the margin fixed — i.e. changed the one hyperparameter DR-002/DR-003 measured as load-bearing (an unreachable `T` cost 2 pp of the effect). The alternative reading — hold `T` and per-class clause count, i.e. 400 total clauses — is defensible and *not* run; it changes the clause budget, which is the axis of the comparison. |

---

# Measured price list (ROUND-1) — **always quote a cost with its Z**

> **Restored 2026-09-20 after a concurrent-edit collision on this file; see `AUDIT.md` A20.**
> `ARMS.md` now has at least two authors. **Append, do not rewrite.**

Every cost quoted in this programme before ROUND-1 silently assumed **Z = 12** (thermometer-4).
The encoding changes cost by up to 3×, so **a budget without a Z attached is not a budget**.
`Z = 3` adaptive Gaussian thresholding · `Z = 12` thermometer-4 · `Z = 24` thermometer-8.

Literal count is exactly `F = Z·k² + 2(32−k)` (verified to the digit against every record), but
**cost is _not_ proportional to `F`** — fixed per-batch overhead does not scale with it, so the
saving from a small `Z` is smaller than `F` implies and the gap closes as the arm grows.

### 5×5 window — the one the P3 shortlist uses. RTX 3090, 45 000 train, batch 50, s/epoch.

| clauses | Z=3 | Z=12 | Z=24 | measured Z12→Z3 | `F` model claims |
|---|---|---|---|---|---|
| 2 000 | 5.7 | 9.8 | 17.8 | **1.73×** | 2.74× |
| 8 000 | 18.8 | 32.7 | 59.4 | **1.74×** | 2.74× |
| 32 000 | 84.6 | 133.8 | 221.6 | **1.58×** | 2.74× |
| 64 000 | 125.5 | 262.2 | 395.5 | **2.09×** | 2.74× |

At **4×4** the Z12→Z3 saving is **1.41–1.62×**; at **10×10**, **2.70–3.10×**. It grows with window
(the `2(32−k)` position bits are Z-independent and dominate at small `k` — at 4×4 they are 56 of
adaptive's 104 literals) and with clause count (fixed overhead amortises). Peak memory falls
1.7–2.6× at Z=3.

### Other windows, Z = 12 (from P0)

| clauses | 4×4 | 8×8 | 10×10 |
|---|---|---|---|
| 640 | 3.2 | 6.5 | 9.1 |
| 2 000 | 7.7 | 15.8 | 20.3 |
| 8 000 | 29.9 | 58.2 | 77.7 |
| 16 000 | — | — | 151 |
| 40 000 | — | — | 329 |
| 80 000 | — | — | 632 |

**Rule of thumb**: `s/epoch ≈ (clauses/1000) × coef`, with `coef` = 3.7 (4×4), 7.3 (8×8), 10.1
(10×10) at Z=12, and **2.9 (Z=3) / 4.4 (Z=12) / 7.0 (Z=24) at 5×5**. Within 10% to 16 000 clauses;
above that the chunk budget collapses to 1–2 examples and the rule over-predicts.

**Z is not a cost dial.** [FACT] The corpus reports a **16.7-point accuracy spread across
Booleanizations at fixed 2 000 clauses**; HOG plateaus at 67.5% where colour thermometers reach
75.4%. An arm's Z matches the paper it reproduces, never what is cheapest, and an arm priced at
Z=3 may not quote an accuracy target from a Z=12 paper.

### What the literature-scale targets actually cost

| target | configuration | GPU-h / seed |
|---|---|---|
| Best published **single model**, 75.4% | 64 000 clauses, 5×5, 250 ep | **18.2** (Z=12) / **8.7** (Z=3) |
| Vanilla CTM 69.3% (AAAI 2023) | 60 000 clauses, 250 ep | ~17 (Z=12) |
| **TM Composites 82.8%** | 22 specialists × 64 000 × 250 ep | **400–600 — not reproducible on this machine** |
| Composite at 2 000 clauses, 79.5% | 22 specialists × 2 000 × 250 ep | **15–27, one seed — affordable** |

## The seed-noise band — the number every comparison is judged against

[MEASURED: calibration_seednoise.json] `ctm-small`, 5 seeds, full protocol:
**test 37.08 ± 0.61 pp**. **Two 3-seed means must differ by ≥ 1.00 percentage points before the
difference is a difference.** Of that band, 0.55 pp is genuine seed-to-seed variation at a fixed
epoch and only ~0.09 pp is added by `argmax(val_acc)` selection (AUDIT A13).

Also measured (AUDIT A19): the **chunk budget** shifts a 3-seed mean by +0.18 pp and the **GPU**
by +0.28 pp — both inside the band, so both are released; but the same seed on a different card
diverges at epoch 1, so **`device + seed` is the reproducibility key, not `seed`**.

## Clause-size budget — the ROUND-1 correction

[MEASURED: lg003_budget_sweep, ctm-small-budget32] `max_included_literals` **does** bind at batch
50 for budgets ≥ 16 (it overshoots by +2.5 literals only at budget 8), and at the SOTA recipes'
budget of 32 a full-scale 30-epoch run lands at median 33. **No sequential feedback is needed.**
Applying it to `ctm-small` is worth **+7.7 points of test accuracy** while evaluating **5.4×
fewer literals per image**, because the unconstrained baseline was sitting at a median clause
length of 175 of 248 available features. `CHARTER.md` silent-failure mode 2 and `LIBRARY_GAPS.md`
LG-003 both overstated this; LG-003 has been narrowed, `CHARTER.md` is the orchestrator's to fix.

---

# Price list at the CORRECTED per-class clause convention (post-DR-001)

`[FACT: arXiv:2406.00704 §II-B; cair/tmu vanilla_classifier.py]` the papers' "clauses" is **per
class**, so their "2 000" is **20 000 total** and their "64 000" is **640 000 total**. Every
price below is at the corrected convention. **There is no 8 000-total cell in the paper at all.**

`[MEASURED: calib_perclass_gpu1]` RTX 3090, 45 000 train, batch 50, s/epoch:

| total clauses | (per class) | 5×5 therm8 | 50 ep, 1 seed | HOG (flat, 5 832 bits) | 50 ep, 1 seed |
|---|---|---|---|---|---|
| 20 000 | 2 000 | **180.7** | **2.51 h** | **146.6** (9.5 GB) | **2.04 h** |
| 40 000 | 4 000 | 345.5 | 4.80 h | **OOM on 24 GB** | — |
| 80 000 | 8 000 | 645.8 | 8.97 h | **OOM on 24 GB** | — |
| 640 000 | 64 000 | ~5 200 (est.) | ~72 h | — | — |

Two hard walls, both measured, both new:

* **The paper's best cell (64 000/class = 640 000 total, 75.4%) is unreachable** — roughly 72
  GPU-h *per seed* at 50 epochs and ~10× the memory of the 80 000-total cell. It joins the
  22-specialist composite as cite-only.
* **HOG caps at 20 000 total — one point, not a curve.** With 5 832 features it holds **11 664
  literals per clause**, and the feedback accumulator alone is four tensors of `(C, 2F)`. At
  20 000 clauses that is 233 M automata and 9.5 GB reserved; 40 000 OOMs a 24 GB card. The
  published HOG plateau begins at 32 000 **per class** = 640 000 total, i.e. **32× beyond what
  fits**, so we cannot observe HOG's saturation directly at all.

**Do not reach for `compute_dtype=float16` to buy headroom here.** `clause_violations` sums up to
`n_literals` 0/1 terms — 11 664 for HOG — and float16 represents integers exactly only to 2 048,
so it would silently miscount violations on exactly the arms that need the memory. `state_dtype
= torch.int16` is safe (states are ≤ 255) and saves half of `ta_state`, but not enough alone.

## Clause-size budget at a correct `T` — the effect grew

[MEASURED: ctm-small-T80-{b32,unc}, 3 seeds each] `T` was inert in the original measurement
(AUDIT A21). Re-measured at `T = 80` (0.8 × the achievable class sum of 100):

| | unconstrained | budget 32 | effect |
|---|---|---|---|
| broken `T = 160` | 37.08 ± 0.61 | 44.58 ± 0.55 | +7.50 pp |
| **correct `T = 80`** | **38.31 ± 1.05** | **47.77 ± 0.32** | **+9.46 pp** |

Seed intervals disjoint (47.44 > 39.50). **The effect is not a `T` artefact — correcting `T`
strengthened it by 2 pp.** Fixing `T` alone was worth +1.23 pp unconstrained and +3.19 pp at
budget 32, so the budget and the margin are complements, not substitutes.

---

## Per-epoch diagnostics every clause arm now records (added 2026-09-21)

Schema and cost are in `code/CONTRACT.md`; the mechanism they test is THEORY.md §5.5.2. Here so
that a reader of this file knows what is in a record without opening one.

| field | one line | added |
|---|---|---|
| `clause_len` / `negation_fraction` / `empty_clause_frac` | read off the automata | P0 |
| `firing_rate` / `clauses_used_frac` / `clauses_ever_fired_frac` | image-level, one eval-mode pass over a 2 000-image probe | P0 |
| `match_count` (**M2**) | exact `|M_j(x)|` histogram, conv arms only | P0 |
| `ta_state` (**D-TA**) | TA-state histogram over *included* literals: `frac_at_boundary` (`== N`, an **upper bound** on the Type-II fringe), `frac_deep` (`>= 1.5N`), `frac_saturated`, depth quantiles | P3 |
| `firing_rate_eligible` (**D-FIRE**) | `P(fire | Type-I-eligible)`, with `1/(1+s)` and the ratio to it in the field. **The global `firing_rate` cannot substitute** — it mixes polarities | P3 |
| `order` (**D-ORDER**) | top-`k`-prefix match probability by TA-state depth, `k ∈ {8,16,32,all}`, with `frac_tied_at_k`. **Sampled**: 256 images × 2 000 clauses by default, both in `hp` | P3 |

[MEASURED: timed against a contended 21.25 s `ctm-small` epoch on the 3090] the three P3 additions
cost **1.85% of an epoch** (D-TA 0.003 s, D-FIRE +0.015 s, D-ORDER 0.373 s); the P0 probes cost
5.2%. None of them consumes a CUDA RNG draw, so switching them on does not perturb training —
verified by `ta_state` hash on both devices (AUDIT A24).
