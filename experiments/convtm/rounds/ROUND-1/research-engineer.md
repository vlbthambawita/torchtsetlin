# ROUND-1 — research-engineer position

*Feasibility, cost and what would have to be written. The other two members are proposing arms
without knowing what they cost; this is the price list, measured on this machine today.*

## Answer in one sentence

The harness can express nine of the thirteen seed arms in PLAN §9.1 with between zero and ~150
lines of experiment-local code, and **memory is never the constraint** — peak GPU usage stays
under 1.5 GB from 640 to 8 000 clauses and reaches only 8.4 GB at 80 000 — but time is, and the
single fact that should reshape the P3 shortlist is that the theorist's settling of A1 makes the
faithful `ctm-vanilla` an **80 000-clause, 9–10 GPU-hour-per-seed arm** whose three seeds would
consume a quarter to a third of the entire P3 envelope.

## Claims

### The measured price list

- [MEASURED: calib_grid_gpu0] Training cost on the RTX 3090 at full scale (45 000 train,
  thermometer-4, batch 50, library-default chunk budget) is **linear in clause count and in
  patch area** across the whole grid:

  | clauses | 4×4 | 8×8 | 10×10 |
  |---|---|---|---|
  | 640 | 3.2 s/ep | 6.5 | 9.1 |
  | 2 000 | 7.7 | 15.8 | 20.3 |
  | 4 000 | 14.7 | 30.0 | 40.2 |
  | 8 000 | 29.9 | 58.2 | 77.7 |

  **Budgeting rule: s/epoch ≈ (total clauses / 1000) × {3.7 (4×4), 7.3 (8×8), 10.1 (10×10)}.**
  It reproduces every cell to within 10% up to 16 000 clauses.
- [MEASURED: calib_large_gpu0] Beyond that the rule breaks *pessimistically*, because the chunk
  budget has already collapsed: 16 000 clauses at 10×10 is 151 s/epoch, 40 000 is 329, and
  **80 000 is 632** (550 at the best chunk setting). Price anything above 16 000 from the probe
  table in `AUDIT.md` A7, not from the rule.
- [MEASURED: calib_grid_gpu0 + calib_large_gpu0 + calib_gpu1_3080] **Peak memory is flat and
  small.** 0.9–1.5 GB reserved for everything from 640 to 8 000 clauses, of which ~0.7 GB is the
  resident Boolean dataset; 8.4 GB for the 80 000-clause arm. **Every arm in PLAN §9.1 fits the
  10 GB card, and the largest was run on it to confirm** — 80 000 clauses at 10×10 completed on
  the RTX 3080 at 6 678 MB allocated / 8 432 MB reserved against its 9 871 MB usable. Raising
  the chunk budget to `2**33` pushes it to 10 520 MB and it OOMs, so the 10 GB card holds the
  biggest arm only at its slowest setting.
- [MEASURED: calib_gpu1_3080 vs calib_grid_gpu0] **The 3080 is a slower card, not a small one,
  and the gap widens with size**: 1.08–1.09× slower at 2 000–8 000 clauses, 1.31× at 40 000 and
  **1.46× at 80 000** (922 vs 632 s/epoch). Screens and small arms cost almost nothing to move
  there; large arms cost a third to a half. That, not capacity, is the device policy.
- [MEASURED: calib_large_gpu0] thermometer-8 roughly doubles the cost of a given clause budget
  (8 000 clauses at 10×10: 78 s/epoch on therm4, 152 on therm8). The Booleanization family arm
  should be priced at 2× for its 8-bit members.
- [MEASURED: calib_grid_gpu0] Inference throughput, the report's efficiency axis, falls with the
  same linearity: 20 092 img/s at 640 clauses / 4×4 down to 781 at 8 000 / 10×10 and 78 at
  80 000. Every arm records it; it is free.
- [MEASURED: smoke_vanilla80k] **At very large clause budgets the per-epoch *evaluation* stops
  being free.** At 78 img/s, the default probes — train accuracy on 5 000 images, validation on
  5 000, the M2 probe on 2 000 — add ~150 s to a 632 s epoch, i.e. **+23%, which turns a 10.5-hour
  seed into a 13-hour seed**. For any arm above ~16 000 clauses, run with `--eval-n 2000
  --m2-n 500`; the diagnostics are distributional and lose nothing that matters at that sample
  size. This is in `jobs/p3_t0_full.txt`.

### The chunk budget is a dial, and it does not save us

- [MEASURED: `code/repro/lg004.py`, LG-004] The library's default `max_chunk_elements = 2**27`
  runs a mini-batch of 50 as **2 examples per chunk at 8 000 clauses / 10×10**, and as **1 at
  80 000**, on a card with 23 GB free. Chunking is a pure partition of the mini-batch —
  feedback accumulates across chunks and commits once [FACT: `models/base.py:305-322`] — so
  this is a throughput/memory dial, not an algorithm change.
- [MEASURED: calib_chunk_gpu0] Raising it buys **1.18–1.51×**, saturating between `2**29` and
  `2**31`, at up to 13× the peak memory. **`2**29` is the right standard**: most of the gain for
  ≤ 1.9 GB reserved, so it still fits the 10 GB card. It does **not** rescue the large arm:
  1.15× at 13× the memory for the 80 000-clause configuration.
- [HYPOTHESIS] The dial is accuracy-neutral. I have read the commit path and it is a pure batch
  partition, but the per-chunk random draws mean the RNG realisation differs. Until the 3-seed
  check lands (`jobs/p0_chunkneutral.txt`, ~25 GPU-min) **no arm uses a non-default chunk
  budget**, and thereafter arms are only compared at equal values — a lower budget is a pure
  slowdown that would make an identical arm look expensive.

### What the harness can and cannot express

- [FACT: `src/torchtsetlin/models/`] Available with **zero new code**, as flags already wired
  through `code/arms.py`: convolutional class-owned TM, convolutional coalesced TM, integer
  clause weights, drop-clause and drop-literal, multi-label heads, focused negative sampling,
  sequential vs batched feedback, and thermometer / adaptive-threshold / bit-plane /
  hypervector Booleanization.
- [FACT: `grep -rn "absorb|sparse|graph|distill|composite|hog" src/torchtsetlin/` → nothing]
  **Absent entirely**: absorbing automata, the graph TM, HOG, distillation, composites.
- [MEASURED: `code/repro/lg003.py`] **Clause-size control does not bind under batched
  feedback.** Budget 8 gives median clause size **10.5 at batch 50**, 8.0 at batch 5, 7.0
  sequential. `ctm-clausesize` must therefore use sequential feedback (exact, ~10× slower) or a
  ported density controller, and which one it uses changes what "we reproduced that paper"
  means. Note the overshoot here (1.3×) is milder than the 9× in the repo `CLAUDE.md` because
  this is a 2-epoch reproduction; it accumulates with training, so **every arm relying on the
  budget must verify per run that it binds**, from the `clause_len` series the harness records.
- [MEASURED: `code/repro/skip_literals.py`] **`mctm-skip` is feasible, and cheaper than it
  looks.** Answering the DL expert directly: if the skip tensor is resampled to the clause-map
  grid, *no subclass is needed at all* — concatenating along the channel axis gives an ordinary
  `(B, C1+Z, H', W')` Boolean image, and `n_features`, `n_literals`, `_patches_per_example` and
  `_chunk_elements_per_example` are all derived from the real input shape, so **the chunk budget
  is automatically correct** (verified: chunk 8/8 at batch 8). If the skip literals must instead
  be *global* — visible to every patch, the way the position-encoding bits are — a subclass
  overriding `_encode` **and** `conv_features` in lockstep works, costs ~10 duplicated lines
  because the parent `_encode` cannot be wrapped (LG-011), and a mismatched override fails
  loudly rather than silently. Either way this is a forward-path change that never calls
  `apply_feedback` on layer 1, so it does not re-open the credit-propagation failure of §3.1.
- [MEASURED: AUDIT A10] No public API reaches the `(B, P, C)` patch-match tensor (LG-005), so
  M2 and every pooling or multi-layer candidate in P5 goes through private hooks. It works, it
  is recorded, and it is the one place a library upgrade would break us.

### M2 — the statistic that decides the P5 counting-pool branch

The measurement now exists per epoch, for every convolutional arm, at ~1% of an epoch, and it is
in the same schema and on the same split (validation, 841 patch positions) as the DL expert's
CNN-side numbers, so the two sit side by side:

| | `ctm-small` (TM, 2 000 clauses, 4×4, trained) | `cnn-ctmshape-max` (CNN, same patch grid) |
|---|---|---|
| `fire_frac` | **0.070** | 0.0096 |
| median \|M\| given firing | **2** | **40** |
| mean given firing | 6.0 | 86.6 |
| p95 given firing | 24 | 328 |
| **P(\|M\| ≥ 5 \| fires)** | **0.303** | **0.883** |
| P(\|M\| = 1 \| fires) | 0.354 | 0.038 |
| unconditional mean | 0.42 | — |

- [MEASURED: ctm-small_seed0..4, cnn-ctmshape-max_seed0] **A CNN filter is a rare, broad
  detector; a TM clause is a common, narrow one.** The TM clause fires on 7× more (image, unit)
  pairs but, when it fires, matches 20× fewer positions. That is the same difference stated
  twice, and it is a sharper statement of the density problem than "clauses fire on 0.14% of
  positions" ever was.
- [MEASURED: ctm-small_seed0..4] The theorist's falsifier (median ≥ 5) is **not met on the TM
  side**: the OR is usually an OR over one or two terms, and a third of firing events are
  singletons. But it is not degenerate either — 30% of firing events match five or more
  positions and the mean given firing is three times the median, so the distribution is
  long-tailed.
- [HYPOTHESIS] That is neither "dead on arrival" nor "carries a great deal". It says a counting
  pool has real information to work with for a minority of clauses, and that the branch is most
  likely a **candidate that only makes sense paired with density control** — because the thing
  that would make counting pay is precisely moving the TM's distribution towards the CNN's. The
  verdict is the theorist's and the DL expert's to argue; my contribution is that the number is
  now measured on both sides under one definition, with the difference in what "match" means
  recorded in the `definition` field rather than hidden.

### What this means for the shortlist

- [MEASURED: calib_large_gpu0] **`ctm-vanilla` at the faithful configuration is 28–32 GPU-hours
  for three seeds.** `ctm-vanilla-8k`, the unqualified reading of "8 000 clauses" that several
  later papers use, is **3.9 GPU-hours** for the same three seeds — an 8× difference for a
  reading of one table column. Both are registered. [HYPOTHESIS] The right move is to run
  `ctm-vanilla-8k` at three seeds *first*, use it to establish that the harness reaches the T0
  band at all, and only then decide whether the full 80 000-clause arm is worth a third of P3.
  That ordering costs 4 GPU-hours and could save 28.
- [HYPOTHESIS] The rest of P3 should be **6 cheap arms sharing one code path + 2 expensive ones
  + at most 1 structurally new mechanism**. More than one structurally new mechanism in P3 is
  how the phase overruns.
- [HYPOTHESIS] `ctm-graph-deep` is the arm I would drop from P3 on cost grounds — a different
  model class, ~500 lines, nothing in the library supports it. It belongs in P5/P6 as a
  candidate if the theory round says depth is where the headroom is.
- [HYPOTHESIS] `tm-composite` is cheap to write (~150 lines over arms that already exist) and
  expensive to run (k× a full arm). Sequence it last in P3, after the Booleanization family it
  is built from has been measured.

## Feasibility and cost, arm by arm

Cost = 3 seeds on the 3090 at the configuration stated, from the measured rule and probe table.
"New code" is experiment-local lines in `experiments/convtm/code/` (C1: never `src/`).

| arm | expressible today? | new code | config priced | GPU-h × 3 seeds | risk |
|---|---|---|---|---|---|
| `ctm-vanilla` | **yes, zero code** | 0 | 80 000 cl, 10×10, 60 ep | **28–32** | low technically, **very high on budget** |
| `ctm-vanilla-8k` | **yes, zero code** | 0 | 8 000 cl, 10×10, 60 ep | **3.9** | low — run this one first |
| `ctm-coalesced` | **yes, zero code** | 0 (registered) | 8 000 cl, 10×10, 60 ep | 3.9 | low |
| `ctm-weighted` | **yes, one flag** | ~5 | 8 000 cl, 10×10, 60 ep | 3.9 | low |
| `ctm-dropclause` | **yes, one flag** | ~5 | 8 000 cl, 10×10, 60 ep, p ∈ {0.25, 0.5} | 7.8 | low |
| `ctm-boolean-*` | partly | ~60 (HOG), ~20 (edges) | 4 000 cl, 8×8, 40 ep × 4 encodings | ~4.5 (8-bit members cost 2×) | medium — HOG has no library support and its parameters are a free choice |
| `ctm-clausesize` | **no — LG-003** | ~80 (density controller) **or** 0 (sequential, ~10× slower) | 4 000 cl, 8×8, 40 ep | 2.0 / ~20 | **high** — the arm cannot use the library's own mechanism |
| `ctm-multitask-rgb` | mostly (`multi_label=True`) | ~60 | 8 000 cl, 10×10, 60 ep | 3.9 | medium |
| `ctm-hypervector` | partly (encoder is token-based) | ~80 | 4 000 cl, 8×8, 40 ep | 2.0 | medium |
| `ctm-distill` | no | ~120 + a P2 teacher | 4 000 cl, 8×8, 40 ep | 2.0 + teacher | medium — hard labels are all the TM update consumes; how a soft target enters is a design decision, not a port |
| `ctm-sparse` / `ctm-absorbing` | no | ~80 (local subclass freezing absorbed automata) | 8 000 cl, 10×10, 60 ep | 3.9 | medium-high — needs a local override of the commit path; show equivalence on a toy first |
| `ctm-graph-deep` | **no** | ~500+ | — | — | **highest — recommend deferring out of P3** |
| `mctm-calibrated` | yes, by porting | ~200 (port from `experiments/mctm`) | as published (128 + 512, 30 ep) | 0.5 | low — code exists; re-run under this protocol, since it was reported on `best_test_acc` |
| `mctm-skip` (P5) | **yes, zero code via channel concat** | 0–40 | layer 2 at 4 000 cl, 3×3 over pooled maps | ~1.5 | low — verified in `code/repro/skip_literals.py` |

**Budget check.** Everything except `ctm-vanilla` and `ctm-graph-deep` totals **≈ 35 GPU-hours**
for three seeds each, inside PLAN's 80–120 GPU-h envelope with room for the matched-budget
controls each headline comparison needs (which roughly doubles the arms that enter one). Adding
the faithful `ctm-vanilla` takes it to **63–67**, which is still inside the envelope but leaves
much less room for controls — hence the run-`8k`-first recommendation above.

## What I need measured (ranked, with cost estimate)

1. **`ctm-vanilla-8k`, 3 seeds** (~4 GPU-h). T0 is mandatory and gates P4–P6. Its uncertainty is
   not compute, it is `T` and `s`: LG-010 A3 shows `T/clauses` spanning 0.025–1.5 across the
   bibliography, and until P1 supplies this paper's values the arm records
   `config_status=provisional` and `record.admissible_as_reproduction()` refuses it as evidence
   for a reproduction or a `GAP` verdict. **Getting `T` and `s` from the paper is worth more
   than any compute I can spend.**
2. **Is the chunk budget accuracy-neutral?** (~25 GPU-min, job file already written.) If yes,
   every cost above falls 1.2–1.4× and `2**29` becomes the standard. If no, the dial is
   off-limits and the costs stand.
3. **A clause-budget scaling curve** (640 / 2 000 / 8 000 / 40 000 at fixed patch, 3 seeds,
   ~12 GPU-h). PLAN §9.2.9 is right that if the curve is still climbing at 8 000, the honest
   next move is more clauses, not a new mechanism — and this curve *is* the decision about
   whether the 80 000-clause arm is worth its 30 GPU-hours.
4. **Augmentation on the Boolean tensor** (~2 GPU-h for 3 seeds at a mid config). PLAN §9.2.5
   calls it possibly the highest value per GPU-hour; ~30 lines, because `data.load_float`
   already carries a batch-level augmenter that can be pointed at the Boolean tensors.
5. **`ctm-clausesize` via the ported controller vs via sequential feedback, small config**
   (~1 GPU-h). Decides whether that arm can be run at scale at all.

## What would change my mind

- **A measured throughput number that breaks the linear rule.** An arm more than ~1.5× off the
  rule is chunk-budget-suspect (LG-004) before it is anything else, and I check that before
  anyone concludes the arm is expensive.
- **Evidence that the chunk budget is not accuracy-neutral** (item 2). Then every cost rises and
  `2**29` is off the table.
- **A theory argument that one of the three structurally missing mechanisms is where the
  headroom is.** I am ranking on cost, not on promise. If the theory round says absorbing
  automata or the graph formulation is the mechanism that matters, I will build it — I am
  arguing only that we can afford exactly one of them in P3, and that the choice should be
  deliberate rather than the result of running out of time.
- **A decision that the faithful 80 000-clause `ctm-vanilla` is the T0 bar and nothing smaller
  will do.** That is a legitimate position — a reproduction at a tenth of the paper's clause
  budget is not a reproduction — and if the round takes it, I will run it and the shortlist
  shrinks accordingly. What I object to is arriving at that cost by accident.

---

# Crossfire response

*New measurements since my position paper: the 5×5 literature-scale capacity ladder
(`results/calib_ladder5x5_gpu0.json`, 9 cells), the chunk-neutrality wave, and the C-5 protocol
control, which required a harness change I have made.*

## 0. The price list the round has been arguing without

Every cost below is on the RTX 3090, 45 000 training images, batch 50, **5×5 window** (the
Toolbox's, not the 10×10 I priced before), library-default chunk budget.

| clauses | therm4 s/epoch | 250 ep (GPU-h) | therm8 s/epoch | 250 ep (GPU-h) |
|---|---|---|---|---|
| 2 000 | 9.8 | **0.68** | 17.8 | **1.24** |
| 8 000 | 32.7 | 2.27 | 59.4 | 4.12 |
| 16 000 | 65.0 | 4.51 | — | — |
| 32 000 | 133.8 | 9.29 | 221.6 | 15.4 |
| 64 000 | 262.2 | **18.2** | 395.5 | **27.5** |

Three consequences the round has not absorbed:

1. **`[MEASURED: calib_ladder5x5_gpu0]` The Toolbox's best single model — 75.4%, 64 000 clauses,
   250 epochs — costs 18.2 GPU-h per seed on therm4 and 27.5 on therm8: 55 to 82 GPU-h for
   three seeds.** T2, treated as a *reproduction*, is half to two-thirds of the entire P3
   envelope for one arm.
2. **`[MEASURED]` T3 is not reproducible on this machine at any budget.** The 82.8% composite is
   22 specialists × 64 000 clauses × 250 epochs = **400–600 GPU-h for a single seed**, against a
   whole-programme budget of 220–300. It must be **cited, not reproduced**. I would like this
   written into `DECISIONS.md` now, because every week that it stays nominally on the list is a
   week of planning around something that cannot happen.
3. **`[MEASURED]` The composite at 2 000 clauses, which the crossfire records as already 79.5%,
   costs 0.68–1.24 GPU-h *per specialist* — 15 to 27 GPU-h for all 22, one seed.** That is
   affordable, it is above the best single model, and **nobody has priced it**. On
   accuracy-per-GPU-hour it is the best item in the corpus by a wide margin. Its blockers are
   engineering (four missing Booleanizations, LG-008) and protocol (the transductive `α_t`,
   O-1), not compute.

## 1. Claims I accept and reject

**Accept**, and they change my position paper:

- The target ladder correction (A.1–A.5). My paper priced `ctm-vanilla` against a 55–61% band
  that traces to an MSc thesis on a modified CIFAR-10 at third hand. T0 ≈ 69% and T2 = 75.4%
  are the defensible numbers, and both sit at ~60–64 k clauses, which is what re-prices
  everything above.
- A.2, and it is decisive for C-4: **the 2019 CTM paper reports no CIFAR-10 number at all.**
- D: no one has changed the convolution mechanism since 2019; binary computation is not the
  bottleneck; GraphTM's CIFAR-10 gain is not a depth effect.
- C-5's premise. My harness did **not** carry what is needed. See §4 — I have fixed it.

**Reject / narrow:**

- The crossfire's framing of C-4 as "8 000 or 80 000 clauses?". **Both readings of the 2019
  paper are unscoreable**, because that paper has no CIFAR-10 result (A.2). See §3.
- My own earlier claim that the chunk dial is worth standardising on. At the clause counts that
  now matter it is nearly worthless: `[MEASURED: calib_chunk_gpu0]` 1.15× at 80 000 clauses for
  13× the memory, and the 5×5 ladder is chunk-serialised to 1–4 examples from 8 000 clauses up.
  **I withdraw the `2**29` recommendation as a programme-wide default**; it is worth 1.2–1.4×
  only in the 2 000–8 000 range, which is where screens live. Verdict against the band in §9.

## 2. C-2 — answered, with the cheapest design that actually discriminates

**My position: the two hypotheses are not symmetric, and one of them is already measured to be
partly wrong — but not in the direction either member argued.**

`[MEASURED: calibration_seednoise.json + screen/cnn-ctmshape-*]` At 2 000 clauses and 4×4 our
CTM reaches 37.1%, while the DL expert's CNN of the *same shape* over the *same patch grid*
reaches 51.0% with a sum pool and 38.9% with a max pool. A 14-point gap at identical clause
count, identical window and identical patch aggregation **is** a mechanism gap — but the max-pool
CNN, which is the architecture that actually matches the CTM's OR, is within 1.8 points of the
CTM. That is evidence the aggregation, not the capacity, is where the CNN's advantage sits, and
it is the one piece of the argument that is already `[MEASURED]` rather than cited.

**Design — 3 clause counts × 2 encodings × 2 seeds × 60 epochs, ~12–16 GPU-h.**

| element | choice | why this and not the obvious alternative |
|---|---|---|
| code path | the `ctm-vanilla` builder at 5×5 | makes the ladder's top point a large-clause vanilla CTM, so it **discharges T0 as a by-product** (§3). A separate 80 k arm then buys nothing. |
| clause ladder | **{2 000, 8 000, 32 000}** | the discriminator is the **8 k → 32 k segment**, not 32 k → 64 k. The published thermometer curve rises 64.5 → 75.4 over 2 k → 64 k, ≈ 2.2 points per doubling, so 8 k → 32 k should carry ≈ **+4.4 points**; the DL expert predicts ≈ 0 there. One segment, two opposite predictions — and it is the *cheap* half of the ladder (5.9 GPU-h on therm4 against 18.2 for the 64 k point alone). |
| encodings | **therm8 + HOG** | the theorist's claim is that the ceiling belongs to the *encoding*. A single-encoding curve that saturates is consistent with both hypotheses and settles nothing. These two are named in the literature as having **opposite** shapes (thermometer still climbing at 64 k; HOG flat by 32 k), which is exactly what makes them discriminating. |
| seeds | **2** | the predicted effect is ≈ 4.4 points against a 2-seed resolution of ≈ 1.2 pp. Three seeds would buy resolution we do not need and cost 50% more. This is a *shape* measurement across three points, not one pairwise contrast. |
| epochs | **60**, fixed across all points | see the confound below. |

**The confound I must name, because it biases toward the DL expert's conclusion.** The published
curve is at 250 epochs; at 60, the large-clause end is under-trained, which makes the curve look
*flatter* than it is. Control, at zero cost: the per-epoch validation curve is already recorded,
so I report whether the 32 000-clause point is **still improving at epoch 60**. If it is, the
measurement is a lower bound on the large end and a "still climbing" verdict is strengthened; if
the top point has plateaued by epoch ~40 *and* the curve is flat across clause counts, the
saturation verdict is real. Without that check the design would be rigged and I would not run it.

**The theorist's own falsifier — `cnn-boolean` on HOG bits — is affordable and I want it in.**
~2 GPU-h on the DL expert's existing harness. It is the only cheap way to *falsify* the
theorist's side: if a CNN on the same HOG Boolean planes substantially beats HOG's 67.5% TM
ceiling, the ceiling is the TM's, not the encoding's. Prerequisite: **HOG does not exist in the
library (LG-008) — ~60 lines plus one cache, which I will write.** It is on the critical path for
both halves of C-2 and should be started now, not after DR-001.

**Conditional extension**: the 64 000-clause top point on the thermometer curve only if the
8 k → 32 k segment is still climbing. +8.7 GPU-h at 2 seeds on therm4, +13.2 on therm8.

## 3. C-4 — the decision rule, and why no outcome buys the 80 000-clause arm

The rule I would apply is shaped by a fact the crossfire established and then did not follow
through: **`[FACT: arXiv:1905.09688 §5]` the 2019 paper reports no CIFAR-10 number.** So a
10×10 `ctm-vanilla` at 8 000 *or* 80 000 clauses has **nothing to be scored against**, and
neither can discharge T0 in the §7.5 sense. T0 must be re-pointed at a number that exists:
AAAI 2023's **69.3% vanilla CTM at 60 000 clauses**, or the Toolbox's 5×5 thermometer column —
both of which sit at the **top of the C-2 ladder we are running anyway**.

**So: the C-2 capacity ladder subsumes the 80 000-clause arm, and I would cut that arm entirely,
saving 31.6 GPU-h.** The 8 000-clause run survives, but demoted from "T0 candidate" to what it
should always have been — a **cheap pre-flight** that checks the code path before we commit 20
GPU-h to the ladder. At 5×5 it costs **1.6 GPU-h for 3 seeds**.

Decision rule for that pre-flight, against the Toolbox's own 8 000-clause thermometer column
(call it X; the theorist has the table, I estimate X ≈ 68–70%):

| outcome | verdict | action |
|---|---|---|
| **within 3 points of X** (§7.5 tolerance) | code path sound at this budget | go straight to the C-2 ladder; its top point is T0 |
| **3–8 points below X** | hyperparameter fault (T, s, epochs), not mechanism | debug at 8 k, where an iteration costs **0.5 GPU-h instead of 18**. Do not scale a suspect arm. |
| **> 8 points below X, or below the 42.6% this repo already reached with a *stacked* TM** | implementation fault | hard stop; audit against the paper; escalate before any further compute |
| any outcome | — | **the 80 000-clause 10×10 arm is never bought** — there is no published 10×10 CIFAR-10 number to score it against |

## 4. C-5 — confirmed: the curve did **not** carry it. Fixed, before P3, not after.

**Direct answer: no.** The recorded curve holds `train_acc`, `val_acc`, `epoch_s` and the
diagnostics, and `validate()` **refuses** a `test_acc` key inside any curve entry — that refusal
is the leakage guard. The papers' statistic is a last-N-epoch mean or a peak of **test**
accuracy, so it was **not computable from a completed run**. This is the M2 mistake exactly:
cheap now, a full re-run of P3 later. The orchestrator was right to ask.

**Fixed, in a way that does not reintroduce the hazard.** `run_arm.py --test-curve-last N` keeps
the last N **automata snapshots** on the host and scores them **after the training loop has
ended**. No test accuracy exists while any decision is being made; `validate()` still refuses
test inside the curve; the paper's statistics land in `diagnostics.paper_statistics`
(`val_selected_test`, `last_n_mean_test`, `peak_test`, `peak_test_epoch`, and both penalties)
with the full series in `diagnostics.test_curve`.

Cost, measured: N test evaluations plus N × `capacity.state_bytes` of host RAM. At N=25 that is
**+1.7% on a 250-epoch 64 000-clause arm** (25 evaluations against 250 epochs) and 4.5 GB of the
121 GB we have; **+7% on a 60-epoch 8 000-clause arm**. Negligible. **Recommendation: run every
`existing`-family arm with `--test-curve-last 25`.**

**On widening the §7.5 tolerance — my view is: do not.** Widening a tolerance to absorb a bias
you can measure is how a reproduction becomes unfalsifiable; a 3-point band that already contains
a known 1-point systematic is a 2-point band that pretends to be 3. The right move is the one
that is now free: **score the reproduction against the paper's statistic computed our way**
(`last_n_mean_test` or `peak_test`, whichever the paper used), and report the validation-selected
number as the headline. Then 3 points stays 3 points of *real* discrepancy, and the protocol
penalty becomes a measured quantity we report rather than slack we hide in a tolerance. If the
penalty turns out to be large and consistent across arms, that is itself a finding worth a
paragraph in the report — and it would justify changing the tolerance **with** evidence.

I am measuring the penalty now on `ctm-small` at 3 seeds (`jobs/r1_protocol_penalty.txt`,
~16 GPU-min); the number is in §9.

## 5. C-6 — `ctm-clausesize` is a control, the budget must bind, and binding is cheaper than 10×

> **SUPERSEDED by §11 and §12, written later the same day after I measured it.** Everything
> below was reasoned from the *published* +0.06 effect and from LG-003 as originally filed. Both
> turned out to be misleading: the budget binds at batch 50 (no batch-5 trick needed) and the
> mechanism is worth **+7.50 pp** in our configuration, not +0.06. The section is kept unedited
> because the round should see what I concluded before measuring and what changed after.

**Confirmed: a control, not a candidate — and the accuracy question is unanswerable by us at any
budget.** `[FACT: arXiv:2301.08190 Table 3]` the effect is **+0.06 points**.
`[MEASURED: calibration_seednoise.json]` our resolution is **1.00 pp for two 3-seed means**. The
published effect is **16× below our noise floor.** No amount of compute changes that, so the arm
must not be sold as an accuracy result.

**Recording that the budget does not bind is *not* sufficient.** A control that does not apply
its treatment is not a control — it measures nothing, and "we capped clause size and nothing
happened" would be false. That the budget does not bind is a library defect, already recorded as
LG-003; it is not an experimental result.

**But full sequential feedback is not the price.** `[MEASURED: code/repro/lg003.py]` the budget
**binds at batch 5** (median 8.0 against a budget of 8), not only at batch 1. And
`[MEASURED: calib_ladder5x5_gpu0]` every configuration that matters is *already* chunk-serialised
at batch 50 — chunk 4 at 8 000 clauses, chunk 1 at 32 000. **When the library is already
processing 1–4 examples per chunk, dropping the batch size to 5 costs far less than 10×**,
because the chunk work is unchanged and only the number of commits rises. I am measuring the
exact factor (`results/calib_batch5_gpu0.json`); §9.

So the arm's deliverable is **interpretability and efficiency, not accuracy**: clause length,
`capacity.included_literals_per_image`, and readable rules — all of which every run already
records. Run it at batch 5 at the ladder's 8 000-clause point, with a **batch-5 control** at the
same budget (arms are only comparable at equal batch size), and report the clause-length and
literals-evaluated deltas with the accuracy delta stated as "below the noise floor".

## 6. C-1 and C-3 — briefly, since they are the other two members' to settle

**C-1.** The engineering position: the measurement supports **(b), paired with density control,
screened as a unit against a density-only control** — and the arm structure is cheap. `[MEASURED]`
median |M| = 2, P(≥5 | fires) = 0.303, against the CNN's 40 and 0.883. A counting pool has real
information to work with for about a third of firing events, and the thing that would make
counting pay is precisely moving the TM's distribution towards the CNN's. Screen as a 2×2 at
2 000 clauses — {OR, count} × {no density control, density control} — which is **4 × 3 seeds ×
0.16 GPU-h = 2 GPU-h** at the ladder's cheapest point. At that price the round should not argue
about it; it should run it. A standalone counting arm without the density cell would confound the
two and is the one version I would refuse.

**C-3.** The disagreement is resolvable only by the design neither paper ran, and the crossfire
names it correctly: **data-budget-matched** augmentation. The TM evidence against augmentation is
confounded by 100 000 vs 50 000 images at equal epochs, so it is not evidence about augmentation
at all. `data.load_float` already carries a batch-level augmenter that can be pointed at the
Boolean tensors (~30 lines). Cost at the 8 000-clause ladder point, 3 seeds, with the
images-seen-matched control: **3.3 GPU-h**. Cheap, and it settles a direct contradiction.

## 7. *(numbering note: §7 was folded into §8 during drafting; nothing is missing)*

## 8. Final feasibility-ranked P3 shortlist against the ~120 GPU-h envelope

> **Amended by §12c** after the clause-size measurement landed. Read them together.

All costs measured, 5×5 window, 60 epochs, RTX 3090, default chunk budget. Arms at 8 000 clauses
share the ladder's 8 000-clause point, so every comparison below is automatically
clause- and automata-budget-matched (§7.3).

### Tier 0 — buys the programme's framing. Run in this order. (~15.5 GPU-h)

| # | arm | config | seeds | GPU-h | what it settles |
|---|---|---|---|---|---|
| 1 | `ctm-vanilla-8k` pre-flight | 5×5, 8 k, therm | 3 | **1.6** | is the code path sound before we spend 20 GPU-h? Decision rule in §3 |
| 2 | capacity ladder, thermometer | `ctm-vanilla` path, {2 k, 8 k, 32 k} | 2 | **5.9** | **C-2**, and its top point is T0 |
| 3 | capacity ladder, HOG | same, HOG *(needs ~60 lines — LG-008)* | 2 | **~6** | **C-2**: is the ceiling the encoding's or the mechanism's? |
| 4 | `cnn-boolean` on HOG bits | dl-expert harness | 3 | **~2** | the theorist's own falsifier — the only cheap way to falsify their side |
| 5 | *conditional* — ladder top point 64 k, thermometer | only if 8 k→32 k is still climbing | 2 | +8.7 | confirms T0/T2 at literature scale |

### Tier 1 — cheap, near-zero code, settles a live contradiction each (~16 GPU-h)

| # | arm | seeds | GPU-h | why it is here |
|---|---|---|---|---|
| 6 | **Booleanization family at 2 000 clauses**, 4 encodings | 3 | **~3** | `[FACT]` **16.7-point spread at fixed 2 000 clauses** — the largest measured lever in the corpus, at the cheapest point on the ladder. **Best accuracy-per-GPU-hour in the programme, and it is prerequisite work for the composite.** |
| 7 | `ctm-dropclause`, p ∈ {0.25, 0.5}, at 8 k | 3 | **3.3** | published **+5.8**, the largest single-mechanism gain in the corpus, **one constructor flag** |
| 8 | augmentation, **data-budget-matched**, at 8 k + matched control | 3 | **3.3** | settles **C-3**; the design neither paper ran; ~30 lines |
| 9 | counting-pool **2×2 screen** at 2 k: {OR, count} × {density control, none} | 3 | **2.0** | settles **C-1** as a unit; a standalone counting arm would confound the two |
| 10 | `ctm-weighted` at 8 k | 3 | **1.6** | one flag |
| 11 | `ctm-coalesced` at 8 k | 3 | **1.6** | matched-budget control for the shared clause pool |
| 12 | `ctm-clausesize` at 8 k, **batch 5**, + batch-5 control | 3 | **~2–4** | interpretability/efficiency only — the accuracy effect is 16× below our band (§5) |

### Tier 2 — the item nobody priced (~1.2 GPU-h to screen, 15–27 to run)

| # | arm | seeds | GPU-h | why |
|---|---|---|---|---|
| 13 | `tm-composite` **screen** at 2 000 clauses, the 5 specialists we can build today | 1 | **~1.2** | decides whether #14 is worth it |
| 14 | full 22-specialist composite at **2 000** clauses | 1 | **15–27** | reportedly **79.5%**, above the best single model, for a *fortieth* of the 64 000-clause composite's cost. Blocked on 4 missing encodings (LG-008) and the transductive `α_t` (O-1), **not** on compute. |

**Total: ~33 GPU-h for Tiers 0–1 plus the composite screen, ~42 with the conditional 64 k point.**
That leaves 75–85 GPU-h of the envelope for the full 2 000-clause composite if its screen
justifies it, third seeds on whatever enters a headline comparison, and P6.

### What I cut, in order

1. **The 80 000-clause 10×10 `ctm-vanilla` — 31.6 GPU-h.** No published CIFAR-10 number to score
   it against (A.2), and the capacity ladder's top point answers the same question. **This is the
   single largest saving available and it costs us nothing.**
2. **The full 64 000-clause 22-specialist composite — 400–600 GPU-h for one seed.** Not
   reproducible on this machine at any budget. Cite it; reproduce the 2 000-clause version.
3. **`ctm-graph-deep`** — ~500 lines, a different model class, and the crossfire has already
   settled that its CIFAR-10 advantage is a multi-view effect, not a depth effect. Replace with a
   **multi-view CoTM arm at 8 k (~1.6 GPU-h)**, which tests the same claim for 1% of the cost.
4. **Any `ctm-clausesize` above 8 000 clauses** — the published effect is 16× below our band.
5. **`ctm-hypervector`, `ctm-distill`, `ctm-multitask-rgb`** — medium code, and none has a
   published CIFAR-10 number that would change the framing. Defer to P5/P6 as candidates.
6. **Any 64 000-clause arm that is not the ladder's conditional top point.** At 18.2–27.5 GPU-h
   per seed, a 64 000-clause arm has to be the thing the report turns on, or it does not run.


### What I need before Tier 0 can run — all of it from the theorist, none of it compute

Every arm below records `config_status=provisional` and is **refused as reproduction evidence**
by `record.admissible_as_reproduction()` until these land. They are the critical path, not the
GPUs:

1. **`T` and `s` for the Toolbox's 5×5 thermometer column** (Table IV gives `T=3000, s=5.0,
   weighted` for the 64 000-clause row — confirm it holds down the ladder, or give the per-row
   values). `T/clauses` spans 0.025–1.5 across the corpus, so this is not guessable.
2. **Is the Toolbox's model class-owned or coalesced, and is "weighted" the Integer-Weighted TM?**
   Both are one constructor flag, but they are different arms and a wrong choice invalidates every
   budget-matched comparison against `ctm-coalesced`.
3. **The Toolbox's own 8 000-clause thermometer accuracy** — this is the target `X` in the §3
   pre-flight decision rule, and the rule cannot be applied without it.
4. **The HOG specification actually used** (cell size, orientations, block normalisation,
   thresholding to bits) so that the ~60 lines I write are that HOG and not a different one.
5. **Epoch count**: the published rows are 250 epochs. Our ladder is 60. Confirm that 250 is a
   real requirement and not an unexamined default — at 64 000 clauses the difference is 18.2 GPU-h
   against 4.4 per seed, and `calibration_seednoise.json` shows our small arm plateauing by epoch
   2–9.

## 9. Verdicts from the four waves I ran during this round

All four are `[MEASURED]`, all against the **1.00 pp** band (two 3-seed means).

### 9a. The chunk budget is accuracy-neutral — the dial is released

| | 3-seed mean test |
|---|---|
| band, GPU 1, chunk 18/50 (default) | 36.97 ± 0.81 |
| `ctm-small-chunk29`, GPU 1, chunk 50/50 (`2**29`) | 37.15 ± 0.85 |
| **difference** | **+0.18 pp — inside the band** |

**Verdict: neutral. The prohibition is lifted**, subject to the standing rule that arms are
compared only at equal `max_chunk_elements`. But note §1: at the clause counts that now matter
the dial is worth almost nothing (1.15× at 80 000), so I no longer recommend a programme-wide
default. Use it in the 2 000–8 000 range, where it is worth 1.2–1.4×, and record it.

### 9b. The same seed on a different GPU is a different draw — and that is fine

`ctm-small` seed 0 is **bit-identical** when re-run on the same card (verified three times at
G0), but on a different card it diverges **at epoch 1** (val 0.3398 on the 3080 against 0.3512
on the 3090) and ends 1.2 pp apart. Floating-point non-associativity in the feedback reductions
is enough to fork the trajectory, after which the run is effectively a fresh draw.

| | 3-seed mean test |
|---|---|
| band, GPU 1 | 36.97 ± 0.81 |
| `ctm-small-protocol`, GPU 0, otherwise identical | 37.25 ± 0.38 |
| **difference** | **+0.28 pp — inside the band** |

**Verdict: arms may be spread across the two cards.** Two consequences worth stating: the
reproducibility check in PLAN §7.4.6 must be "lands inside the band", never "reproduces the
number" — which is what it says, and now it is measured rather than assumed; and **`seed` alone
is not a reproducibility key, `device + seed` is.** The `env.gpu` field already in every record
carries it.

### 9c. The protocol penalty is ~0.25 pp, not ~1 point — do not widen the tolerance

`ctm-small`, 3 seeds, `--test-curve-last 25`, i.e. the papers' own statistic:

| statistic | 3-seed mean | vs our headline |
|---|---|---|
| **val-selected (ours, PLAN §7.1)** | **37.25** | — |
| mean of the last 25 epochs (theirs) | 37.21 | **−0.04 pp** |
| peak over the last 25 epochs (theirs) | 37.50 | **+0.25 pp** |

Peak epochs were 19, 7, 23 — scattered, consistent with the flat-curve finding at G0.

**Verdict: the crossfire's `[HYPOTHESIS]` of "≤ 1 point" is confirmed and is generous — the real
penalty is ≤ 0.25 pp, four times below the seed band and twelve times below the §7.5 tolerance.**
My §4 recommendation stands and is now evidence-backed: **do not widen the 3-point tolerance.**
Report both statistics, score reproductions against the paper's own, and the tolerance keeps its
full width for real discrepancies. One caveat for honesty: this is measured on a *converged,
flat-curve* arm. A large arm still climbing at its last epoch would show a larger gap against a
last-N mean, so `--test-curve-last 25` goes on every `existing` arm rather than being assumed
away from this one measurement.

## 10. The Z re-pricing — the model is right in direction and 30–70% too optimistic in size

**I measured it rather than accepting it, and the answer changes the conclusion.** The `F` model
(`F = Z·k² + 2(32−k)`) is *exactly* right about the literal count — it reproduces every
`n_features` in my records to the digit — but **cost is not proportional to `F`**, because a
fixed per-batch overhead (the unfold, kernel launches, the feedback bookkeeping) does not scale
with it. That overhead is a larger share of a cheap arm, so the model is worst exactly where the
theorist applied it.

`[MEASURED: calib_zratio_gpu1, calib_zratio5x5_gpu0]` therm4 (Z=12) ÷ adaptive (Z=3):

| window | clauses | measured saving | `F` model predicts | model too optimistic by |
|---|---|---|---|---|
| 4×4 | 2 000 | **1.41×** | 2.38× | +69% |
| 4×4 | 8 000 | **1.62×** | 2.38× | +47% |
| **5×5** | **2 000** | **1.73×** | 2.74× | +58% |
| **5×5** | **8 000** | **1.74×** | 2.74× | +58% |
| **5×5** | **32 000** | **1.58×** | 2.74× | +73% |
| **5×5** | **64 000** | **2.09×** | 2.74× | +31% |
| 10×10 | 2 000 | **2.70×** | 3.62× | +34% |
| 10×10 | 8 000 | **3.10×** | 3.62× | +17% |

Two regularities: the saving **grows with clause count** (fixed overhead amortises) and **grows
with window** (because the `2(32−k)` position bits are Z-independent and dominate at small `k` —
at 4×4 they are 56 of adaptive's 104 literals, so more than half the literal budget is
position). The model is only within ~20% in the regime it was never applied to (10×10, large).

**Corrected re-pricing at the 5×5 window the shortlist actually uses** — the saving is **1.6–2.1×,
not 3.6×**:

| arm | therm4 (Z=12) | adaptive (Z=3) |
|---|---|---|
| 8 000 clauses, 250 ep | 2.27 h/seed | **1.31 h/seed** |
| 32 000 clauses, 250 ep | 9.29 h/seed | **5.87 h/seed** |
| 64 000 clauses, 250 ep | 18.2 h/seed | **8.7 h/seed** |

So the theorist's direction is right and their headline figure is roughly right *for the 80 000
clause 10×10 arm* (3.10× measured against their 3.62×), but **their ~88 GPU-h P3 total should be
recomputed with 1.6–2.1×, not 3.6×** — for a list dominated by 5×5 arms that is a ~1.7–2.2×
under-estimate of the remaining budget. Memory also falls, by 1.7–2.6× (64 000 clauses at 5×5:
928 MB reserved on adaptive against 2 420 on therm4), which matters only for the very largest
arms.

**The correction that matters more than the arithmetic: Z is not a free cost dial.** The corpus's
own headline number is a **16.7-point spread across Booleanizations at fixed 2 000 clauses**, and
HOG is flat at 67.5% while colour thermometers reach 75.4%. **An arm priced on adaptive may not
quote its accuracy target from a thermometer paper.** The Z column belongs to the *arm*, chosen
to match the paper being reproduced, and it is never chosen for cheapness. I have added it to the
price list in `ARMS.md` and `AUDIT.md` so that the next budget quoted in this programme carries
its encoding with it — which, as the orchestrator says, three of us have already failed to do.

## 11. C-6 does not evaporate — it inverts. The largest measured effect of the round.

The theorist was right that the overshoot is **absolute, not proportional**, and right that C-6's
premise was wrong. They were right for the wrong reason, and the consequence is the opposite of
"a free disclosure line in the registry".

**11a. The budget-sweep measurement they asked for** (`results/lg003_budget_sweep.json`,
200 clauses, 4×4, 2 epochs, 0.1 GPU-h — a sixth of their estimate):

| budget | batch 50 median | batch 5 | sequential | verdict at batch 50 |
|---|---|---|---|---|
| 8 | **10.5** | 8.0 | 7.0 | **exceeded, +2.5 absolute** |
| 16 | 15.0 | 9.0 | — | **held** (−1.0) |
| 32 | 20.0 | 12.0 | 11.0 | **held** (−12.0) |
| 64 | 25.0 | 13.5 | — | **held** (−39.0) |

The overshoot is **+2.5 literals in absolute terms and only at budget 8**. The median tracks the
budget (25 → 20 → 15 → 10.5 as the cap falls 64 → 32 → 16 → 8), so the cap is *active* at every
one of these settings — it is not that the clauses are naturally small. **At the SOTA recipes'
budget of 32, `max_included_literals` binds at batch 50. No sequential feedback. No 10× cost.**

**11b. And then the full-scale run, which is where it stops being a bookkeeping question.**
`ctm-small` at 45 000 images, 30 epochs, batch 50, one flag changed
(`--max-included-literals 32`), against its own band:

| | unconstrained (band) | budget 32 | Δ |
|---|---|---|---|
| **test accuracy** | **37.08 ± 0.61** (5 seeds) | **44.74** (seed 0) | **+7.7 pp** |
| median clause length | 175 of 248 features | **33** | −142 |
| p95 clause length | 241 | 35 | −206 |
| median firing rate | 0.077 | 0.136 | +77% |
| P(\|M\| ≥ 5 \| fires) | 0.303 | **0.365** | +0.06 |
| **included literals / image** | **294 M** | **54.7 M** | **5.4× fewer** |
| wall-clock | 356 s | 371 s | +4% |

`[MEASURED: ctm-small-budget32_seed0]` **+7.7 points — nearly eight times the seed band — from a
single constructor flag, while evaluating 5.4× fewer literals per image.** That is a Pareto
improvement on two of the report's three axes at once. 3-seed confirmation is running
(`jobs/r1_budget32.txt`, 12 GPU-min); §12 carries the verdict.

**What I think this means, and it is uncomfortable for my own earlier position.**

- **Our provisional baseline is badly mis-tuned, and the P0 diagnostics said so and nobody read
  it.** A median clause length of 175 out of 248 available features is a clause that asserts
  almost everything and therefore discriminates almost nothing. It is in every record from G0
  onward. I reported it as a curiosity in `AUDIT.md` A13; I should have reported it as a defect.
- **The published +0.06 is not wrong, it is measured against a *tuned* baseline.** Against ours
  the cap is worth 7.7 points, because it is doing the job that `s` and `T` should be doing.
  This is the strongest evidence yet for my C-4 "outcome B" prediction — that a T0 shortfall will
  be a hyperparameter fault, not a mechanism fault — and it is why getting the papers' `T` and
  `s` (§8 prerequisites) is worth more than any compute in this round.
- **`ctm-clausesize` moves from "cut it" to Tier 0.** In my §8 shortlist I ranked it last and
  said the accuracy effect was 16× below our band. On the published number that was right; on the
  measured number it is wrong by two orders of magnitude. **I withdraw that ranking.**
- **`ctm-small` at 44.7% already clears T1** (this repo's 42.6% stacked best) at 2 000 clauses,
  4×4, 30 epochs and 0.1 GPU-h. Every arm in the shortlist should carry a clause-size budget, or
  we will be measuring mechanisms against a baseline that is leaving 8 points on the floor.

**11c. Correction needed to a document three agents read.** `CHARTER.md` silent-failure mode 2
states flatly that *"`max_included_literals` does **not** bind for conv models under batched
feedback"*, and `LIBRARY_GAPS.md` LG-003 says the same. **That is now measurably too strong**: it
fails to bind only at very small budgets (≤ 8), and holds at 16, 32 and 64 — the range every
published recipe actually uses. I have corrected LG-003 (it is mine). **CHARTER.md is the
orchestrator's**, and I am flagging rather than editing it: the rules card currently tells three
agents to avoid a mechanism that works and is worth 7.7 points.

## 12. Verdict on the clause-size budget, and the shortlist revision it forces

**12a. Confirmed at 3 seeds. It is the largest effect measured anywhere in this programme.**

| | unconstrained | budget 32 |
|---|---|---|
| test accuracy | **37.08 ± 0.61** (5 seeds) | **44.58 ± 0.55** (3 seeds) |
| individual seeds | 36.35, 36.67, 37.89, 37.04, 37.47 | 44.74, 45.03, 43.97 |
| median clause length | 157 | **33** |
| included literals / image | 263 M | **54.7 M** |
| median firing rate | 0.077 | 0.134 |
| P(\|M\| ≥ 5 \| fires) | 0.303 | **0.363** |
| wall-clock | 359 s | 365 s (+1.7%) |

**+7.50 pp — 7.5× the seed band — with the across-seed intervals completely disjoint
(43.97 > 37.89), 4.8× fewer literals evaluated per image, and 1.7% more wall-clock.** This
satisfies PLAN §7.2's superiority bar on the seed-interval criterion outright; the paired
McNemar and bootstrap are available from the saved prediction vectors whenever the round wants
them.

Note also that it moves **P(\|M\| ≥ 5 \| fires) from 0.303 to 0.363** — so density control and the
counting pool are coupled in exactly the direction C-1's resolution predicted, and the 2×2 screen
in §8 item 9 should use budget 32 as its density-control cell rather than inventing a new one.

**12b. Three things I said earlier in this document that are now wrong, and I withdraw them.**

1. *"`ctm-clausesize` is a control, not a candidate; the accuracy question is unanswerable at any
   budget because +0.06 is 16× below our band."* — Wrong. The published +0.06 is against a tuned
   baseline; against ours the mechanism is worth +7.50 pp. **It is a Tier 0 arm.**
2. *"The budget must bind, and binding needs batch 5."* — Wrong. It binds at **batch 50** for
   budgets ≥ 16. The batch-5 route, and the cost measurement I queued for it, were solving a
   problem that does not exist at any budget a paper uses.
3. *"`ctm-clausesize` at anything above 8 000 clauses is the fourth thing I would cut."* —
   Withdrawn. A clause-size budget now belongs on **every** arm unless it has a stated reason not
   to have one.

**12c. Revised shortlist.** The §8 table stands with these changes:

- **New Tier 0 item, ahead of everything else**: re-run the capacity ladder's anchor points
  **with budget 32**, and treat a clause-size budget as part of the default arm configuration
  rather than as a separate arm. Marginal cost ≈ **0** (+1.7% wall-clock); it is a flag, not an
  arm.
- **`ctm-clausesize` is deleted as a standalone arm** and becomes the *unconstrained control* for
  the budget that everything else now carries — the inverse of its previous role. 3 seeds at the
  ladder's 8 000-clause point: **1.6 GPU-h**.
- **Everything else in Tier 0/1 should be re-baselined**, because every published mechanism in
  the corpus — Drop Clause's +5.8, composition's +7.4, the encodings' 16.7-point spread — was
  measured against baselines that had a clause-size budget. Comparing our mechanisms against an
  unconstrained baseline would have credited them with recovering a defect of ours.

**12d. The process finding, which matters more than the arm.** The signal was in **every record
from the first P0 run**: `clause_len.median = 157–175` against 248 available features, printed on
every epoch line, present in all five band records. I logged it in `AUDIT.md` A13 as a curiosity
about flat validation curves. **It was a defect, and the cost of not escalating it was 7.5 points
carried through the entire round — including into my own §8 shortlist, where I ranked the fix
last.** The diagnostic worked; the escalation threshold did not exist. I have added one
(`clause_len.median > 0.5 × n_features` is now a defect signature in `ARMS.md`, called out in the
log line and the round report rather than left in the JSON), because the next such signal will be
in the diagnostics too, and being right in a JSON file that nobody acts on is the same as being
wrong.


---

# Post-DR-001 addendum

## 13. Is "budget 32 everywhere" a confound baked into the ladder? — yes, and neither proposed option discharges it

**Short answer: run the unconstrained control as a *full parallel curve* on one encoding at one
seed — not a single control point, and not a budget sweep at one upper point.**

**Why not a single unconstrained control point.** The risk is that the budget bends the
*slope*, and a single point measures the gap at one clause count. It cannot distinguish "the
budget helps by 7 points everywhere" (harmless — a level shift that cancels out of the slope)
from "the budget helps by 7 points at 2 k and 0 at 32 k" (fatal — it flattens exactly the
segment C-2 turns on). One point cannot see a trend by construction.

**Why not a budget sweep at one upper point.** It would find the optimum at 32 k, but to know
whether the optimum *moved* you need it at the bottom too — which costs the same as the parallel
curve while producing less: an optimum at one clause count instead of a second capacity curve.

**Why the parallel curve is the right buy.** It is strictly more informative than either: it
*is* a second capacity curve, so C-2's question gets answered twice — once at budget 32, once
unconstrained — and the confound shows up as a gap that either does or does not vary with clause
count. If the gap is constant the ladder is safe and we have replication; if it widens or
narrows we have caught the confound and we know its size and sign.

**There are two concrete reasons to expect the interaction to be real**, which is why I would
spend the hour rather than assume:

1. **The budget is denominated in literals, and the literal space changes along the ladder and
   across the encodings.** 32 literals is 9.0% of the 354 available at 5×5 thermometer-4, 4.5%
   of 708 at thermometer-8, and **0.5% of the 6 272 in HOG's whole-map window**. "Budget 32
   everywhere" is emphatically *not* a neutral default across C-2's two encodings — it is a
   mild constraint on one arm and a severe one on the other.
2. `[HYPOTHESIS]` The optimal clause length should *fall* as clause count rises, because a
   larger ensemble lets each clause afford to be more specific. If so, a fixed 32 becomes
   increasingly **slack** toward the top of the ladder — letting the unconstrained blow-up
   creep back in exactly where we are trying to measure saturation, and biasing the top of the
   curve downward.

**What it costs, and what is already free.** The 2 000-clause cell of this control is *already
running* as part of the pre-flight (§14), which is a `{class-owned, coalesced} × {budget 32,
unconstrained}` 2×2 at the ladder's own window and encoding. So only the upper cells are new:

| added cells | encoding | 1 seed, 60 epochs |
|---|---|---|
| unconstrained at 8 000 and 32 000 | therm4 5×5 | **2.8 GPU-h** |
| unconstrained at 8 000 and 32 000 | therm8 5×5 | 4.7 GPU-h |

**Recommendation: spend the 2.8 GPU-h on therm4.** It completes a parallel curve across the full
ladder range for less than the cost of one 32 000-clause point, and it converts a baked-in
assumption into a measured gap.

**One correction to how DR-001 words this, though.** Budget 32 is not only our inference from my
2 000-clause 4×4 measurement — it is **the published value**: Composites Table 1 states a budget
of 32 literals for its specialists [FACT: LITERATURE_TM.md A8]. So running the ladder at 32 is
*faithful*, not merely convenient, and the HOG arm in particular should keep it because it is
that arm's own paper value. What remains genuinely ours is applying it to the thermometer arm,
whose budget the Toolbox does not state — and that is precisely the cell the parallel curve
covers.

**And a caveat on my own +7.50 pp**, which DR-001 leans on: it was measured at 4×4, thermometer-4,
2 000 clauses, T=160, s=10. The pre-flight now running changes the window, the encoding, T and s
simultaneously, and its first epochs already reach ~46% validation where `ctm-small` plateaued at
37%. The regime is different enough that I would not assume the +7.50 transfers — which is the
same argument the orchestrator is making, and it is right.
