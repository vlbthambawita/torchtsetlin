# ROUND-1 minutes

**Date**: 2026-09-20
**Attendees (roles)**: tm-theorist, dl-expert, research-engineer, orchestrator (compiler of
`crossfire.md`, decision author)

## The question

Which existing approaches to convolution in Tsetlin machines does the programme re-implement in
P3, in what order, and at what cost — and, from the CNN side, what is the mechanism inventory to
mine for P5 candidates? Bundled into this by the crossfire: is PLAN §2's target ladder (T0
"55–61%", T2 "beat 60.7%") correctly set at all?

This is the round the programme's contract (`CHARTER.md` C3) allows to run ahead of its own
measurements, because it is a decision about *what to measure*, not about what works — everything
after it needs `[MEASURED]` evidence.

## Each member's position, in their own terms

**tm-theorist.** Re-implement the three axes that account for essentially all published CIFAR-10
progress — Booleanization, clause count, confidence-weighted composition — plus the two
single-model mechanisms with a checkable CIFAR-10 number (vanilla CTM, Drop Clause), and drop
every arm whose paper reports no CIFAR-10 result. Central finding: 60.7% is not a CIFAR-10 result
at all, and since 2019 nobody in the corpus has changed the CTM's convolution mechanism itself —
which is both the reason the P3 list should not be a paper-enumeration and the reason the
programme has somewhere to go.

**dl-expert.** The TM's roughly 33-point CIFAR-10 gap to CNNs is not explained by binary
computation (a fully-binary net reaches 93.75%), so P3 should re-implement approaches that change
the *readout* and *receptive field* (counting pool, weighted/coalesced votes, patch geometry,
Booleanization) ahead of approaches that only add capacity. Ranked the P5 mechanism inventory as
counting pool → augmentation → dilated patches → knob annealing → online rate control → soft-`T`
distillation → skip literals for depth (later revised, see Disagreements).

**research-engineer.** Feasibility and cost, measured on this machine today, not estimated: the
harness expresses most of PLAN §9.1's arms with 0–150 lines of experiment-local code; memory is
never the constraint (peak 0.9–1.5 GB from 640–8,000 clauses, 8.4 GB at 80,000); time is. The
single fact that should reshape the shortlist is that a faithful `ctm-vanilla` (80,000 clauses) is
a 28–32 GPU-h-for-three-seeds arm — a quarter to a third of the entire P3 envelope — for a paper
that, it later turned out, reports no CIFAR-10 number to reproduce it against at all.

## Points of agreement

Recorded in `crossfire.md` §D and accepted by all three without qualification in their crossfire
responses:

- Since 2019, nobody in the local 45-paper + book-chapter corpus has changed the CTM's convolution
  mechanism (patch OR, one random matching patch per feedback event, thermometer position bits).
  The one exception, CTM-UNet, is a backprop hybrid on segmentation with no CIFAR-10 number.
- Published CIFAR-10 progress decomposes into Booleanization (16.7-point spread at fixed 2,000
  clauses), clause count (+10.9 from 2k→64k), composition (+7.4 to +11.6 over the best member) and
  Drop Clause (+5.8).
- Binary computation is not the CIFAR-10 bottleneck: best fully-binary net 93.75% vs 94.8% FP;
  binary weights cost ~nothing (BWN 56.8 vs FP 56.6 on ImageNet), binary activations cost 12.4.
- GraphTM's CIFAR-10 advantage is not a depth effect — its own run is at depth 1, and its authors
  attribute the gain to multi-view input, not depth.
- `T_ratio = 0.8` is not a safe default (measured `T/clauses` spans 0.025–1.5 across the
  bibliography); every arm must carry its own paper's `T` and `s`.
- Memory is never the constraint anywhere in PLAN §9.1; the RTX 3080 is a slower card, not a
  smaller one (1.08× at 8k clauses, 1.46× at 80k).
- The seed band — two 3-seed means must differ by ≥1.00 pp to be a difference — governs every
  comparison in the programme.
- `mctm-skip` is feasible with no subclass if the skip tensor is resampled to the clause-map grid.
- No arm uses a non-default chunk budget until the neutrality check lands (it has since landed,
  see Decisions, and is released, but with no single programme-wide setting).
- 60.7% is not a CIFAR-10 result (§A below) — this was not a live disagreement, only a shared
  finding the round had to act on.

## Points of disagreement — what was claimed, and what settled it

### A. The target ladder (not a member-vs-member contradiction, but a joint finding requiring a decision)

`[FACT: arXiv:2406.00704 §V + ref. 14]` 60.7% traces to an MSc thesis (Mathisen & Smørvik, 2020)
on a **modified** CIFAR-10, cited at third hand; its clause budget, window, T, s and epoch count
are unknown, and the thesis was unreachable from this machine. `[FACT: arXiv:1905.09688 §5]` the
2019 Convolutional TM paper reports **no CIFAR-10 number** at all. `[FACT: arXiv:2406.00704 Table
IV]` the best single-model TM is **75.4 ± 0.09%** (64,000 clauses); `[FACT: same, Tables III–IV]`
82.8 ± 0.01% is a 22-specialist composite, not a single model. Settled by direct textual/table
verification of the primary sources (orchestrator, independently, and both experts independently
in their crossfire responses). Open question O-1 (the composite's transductive `α_t`) is unresolved
and un-costed.

### C-1 — is the counting pool a standalone candidate?

- tm-theorist `[HYPOTHESIS]`: a no-op at the densities the standard feedback rule produces; pays
  off only jointly with density control. Pre-registered falsifier: median `|M_j| ≥ 5` for a firing
  clause.
- dl-expert `[HYPOTHESIS]`: OR gives the vote `C` bits, counting gives ~9.7`C` bits — predicted
  +5–12 points. Pre-committed withdrawal condition: median ≈ 1 given firing.
- **Settled by** `[MEASURED: results/ctm-small_seed{0..4}.json → diagnostics.match_count]`:
  median = 2, `P(|M|≥5 | fires) = 0.303 ± 0.002`, `P(|M|=1 | fires) = 0.356`. **Neither
  pre-registered falsifier fired.** The distribution is long-tailed, not degenerate. Both members
  agreed this makes both falsifiers invalid on methodological grounds (a falsifier stated over a
  quantity the candidate itself would change is not a falsifier — see the falsifier-discipline
  amendment under Decisions). Verdict reached jointly: **(b) — admissible only paired with density
  control, screened as a unit against a density-only control**, never standalone. A 2×2 (or
  6-cell) factorial screen (`S0`–`S5` / the theorist's naming, the "capped-count × density"
  2×2 in the dl-expert's) is the design that would resolve it at full scale; both cost ≈0.02–6
  GPU-h depending on scale. Not yet run.

### C-2 — capacity-limited or mechanism-limited?

- dl-expert `[HYPOTHESIS]`: capacity curve saturates below 70% by ~8–16k clauses; CIFAR-10 CTMs
  are mechanism-limited. Pre-registered falsifier: still climbing >1 pt/doubling at 8k clauses.
- tm-theorist `[FACT: arXiv:2406.00704 Table IV]`: 5×5 colour thermometers go 64.5→75.4 from
  2k→64k clauses, still climbing (+1.7 from 32k→64k); HOG is flat at 67.5 by 32k. `[HYPOTHESIS]`
  the ceiling is the Booleanization's, not the clause mechanism's.
- **Settled by direct table lookup** (both experts read the same primary-source table
  independently): at 8k clauses the delta is +2.6 pp/doubling, not saturating. **The dl-expert's
  own falsifier fires against their own prediction.** The disagreement dissolved into a
  reframing (Decision 1): stop asking whether it saturates and ask what the slope of accuracy
  against log₂(clauses) is. Falsifier / resolving measurement for the *programme's own* data (not
  yet run): our own capacity ladder at ≥2 encodings (thermometer + HOG), plus the `cnn-boolean` on
  HOG bits probe the theorist named as the clean falsifier for "the ceiling is the encoding's".

### C-3 — augmentation

- dl-expert: ranked it #2, predicted +1.5–4 points, zero new code, citing CNN literature
  (`[FACT: arXiv:1708.04552 Table 1]` ResNet-18 4.72% aug vs 10.63% no-aug).
- tm-theorist `[FACT: arXiv:2406.00704 Table IV]`: static flip augmentation *hurts* the best TM
  specialist by 1.3 points and raises its variance, while helping edge/threshold specialists; the
  published TM comparison is confounded by an unmatched data budget (100k vs 50k images at equal
  epochs).
- **Not fully settled — resolving measurement identified, not yet run**: a data-budget-matched
  augmentation arm (on-the-fly stochastic flip+crop at matched images-per-epoch and matched
  epochs, retuned `T`), at ≥2 clause budgets. Both members converged on this design. New joint
  finding not in either original paper: the effect **flips sign with clause budget**
  (`[FACT: arXiv:2406.00704 Table IV]` helps 10 of 11 encodings at 2,000 clauses, hurts every
  thermometer encoding at 64,000) — recorded as a screening hazard (see Decisions).

### C-4 — what is `ctm-vanilla`, and what does faithfulness cost?

- Convention question (per-class vs total clause count, a 10× budget factor): **settled**,
  `[FACT: arXiv:1905.09688 Table 1]` — the 2019 paper's clauses are per-class; `ctm-vanilla` is
  class-owned (`ConvTsetlinMachine`), not coalesced. Both members agreed from the start.
- Cost/faithfulness question: research-engineer priced the faithful 80,000-clause reading at
  `[MEASURED: results/calib_large_gpu0.json]` 28–32 GPU-h for three seeds, and recommended running
  the unqualified 8,000-clause reading first (3.9–4.1 GPU-h) to de-risk before committing the rest.
  **This was superseded, not merely settled**, once `[FACT: arXiv:1905.09688 §5]` — the 2019 paper
  reports no CIFAR-10 number at all — was pressed to its conclusion: neither the 8k nor the 80k
  reading of `ctm-vanilla` has anything to score against, so it cannot discharge T0 in the §7.5
  sense. **The research-engineer's own resolution, accepted by DR-001**: cut the 80,000-clause arm
  entirely (−31.6 GPU-h); demote the 8,000-clause run from "T0 candidate" to a cheap pre-flight
  check; re-point T0 at AAAI 2023's 69.3% (60,000-clause vanilla CTM) or the Toolbox's own
  8,000-clause thermometer column, both of which sit inside the capacity ladder the programme is
  running anyway.

### C-5 — does the protocol (validation selection) make every reproduction look worse?

- `[FACT: LITERATURE_TM.md §2.5]` no TM image paper uses a validation split; reported accuracy is
  a last-25/100-epoch mean or a peak. `[HYPOTHESIS, tm-theorist]` the systematic penalty is ≤1 pt.
- **Settled by measurement, and the hypothesis was too generous.** The harness initially could not
  even compute the comparison — `validate()` refuses `test_acc` inside any curve entry (the
  leakage guard), so the papers' own statistic was not computable from a completed run
  (research-engineer, C-5 confirmation). Fixed with `--test-curve-last N` (scores held-out
  snapshots only after the training loop ends). `[MEASURED: jobs/r1_protocol_penalty.txt]`
  val-selected 37.25%, last-25-mean 37.21% (−0.04 pp), peak 37.50% (+0.25 pp) — a real penalty of
  **≤0.25 pp**, four times below the seed band and twelve times below the §7.5 reproduction
  tolerance. Both experts and the engineer agree: report all three statistics, score reproductions
  against the paper's own estimator, and **do not widen the §7.5 tolerance** — the confirmed
  penalty is far smaller than the hypothesis that motivated the concern.

### C-6 — `ctm-clausesize`: control or candidate, and does the budget bind?

This is the one disagreement that **inverted rather than resolved cleanly**, and it produced the
round's largest number. Starting position (all three, and `CHARTER.md`'s own then-current text):
`[MEASURED: code/repro/lg003.py]` budget 8 → median clause size 10.5 at batch 50 (exceeds the
budget); `[FACT: arXiv:2301.08190 Table 3]` the published effect is a **+0.06** accuracy gain on
CIFAR-2, 16× below the seed band — so the mechanism is a control, not a candidate, and whether it
binds under batched feedback was thought to be close to moot. **The research-engineer then ran the
budget sweep the theorist asked for** (`[MEASURED: results/lg003_budget_sweep.json]`): the
overshoot is **absolute (+2.5 literals) and only at budget 8** — at 16/32/64 the cap holds (median
15.0/20.0/25.0 against caps 16/32/64). At the SOTA recipes' budget of 32, it binds at batch 50, no
sequential feedback needed. **Then, applying budget 32 to the programme's own baseline**
(`[MEASURED: results/ctm-small-budget32_seed{0,1,2}.json]`): test accuracy **44.58 ± 0.55 vs
37.08 ± 0.61** unconstrained — **+7.50 pp, 7.5× the seed band, seed intervals disjoint**
(43.97 > 37.89) — at **4.8× fewer included literals per image** (54.7 M vs 263 M) and **+1.7%
wall-clock** (365 s vs 359 s). This is the largest effect measured anywhere in the programme so
far. Cause traced by the engineer: the unconstrained baseline's median clause length was 175–175
of 248 available features — a clause asserting almost everything, present in every P0 record and
reported only as a curiosity (`AUDIT.md` A13) until this round. The published +0.06 is real, but
it is measured against a *tuned* baseline; against the programme's mis-tuned one the same
mechanism is worth two orders of magnitude more.

## Withdrawals — every member withdrew at least one of their own claims against a measurement

This is the fact about the round the note-taker was specifically asked to preserve.

- **dl-expert** withdrew the prediction that "the capacity curve saturates below 70% by ~8–16k
  clauses." Their own pre-registered falsifier ("still climbing >1 pt/doubling at 8k clauses") was
  read directly off `[FACT: arXiv:2406.00704 Table IV]` and fired against them: the measured delta
  at 8k is +2.6 pp/doubling. Quote: *"The prediction is dead and I withdraw it."*
  They also withdrew the "~9.7×" information-gain figure behind the counting-pool prediction,
  recomputing it from the measured match-count quantiles
  (`[MEASURED: results/ctm-small_seed{0..4}.json → diagnostics.match_count]`) as **1.71–1.72×**.
  Cause of the error, stated by the author: the 9.7× figure was `log₂(P+1)` — the ceiling under a
  *uniform* distribution over counts — quoted as an estimate, when the measured distribution is
  93% zeros with 35.5% of the remainder equal to exactly one.
- **tm-theorist** withdrew the sentence "the OR in `⋁_b c_j^b` is, in the trained regime, almost
  never an OR over more than one term," against
  `[MEASURED: results/ctm-small_seed{0..4}.json → diagnostics.match_count]`: 64.6% of firing
  events match ≥2 positions and 30.3% match ≥5. They also withdrew their own reading of the
  still-rising capacity curve as "the honest next move for accuracy is buy clauses and better
  encodings," replacing it with the slope reframing (Decisions, below): *"not saturated ≠ a
  route."*
- **research-engineer** withdrew three of their own claims after running the clause-size-budget
  measurement (§C-6 above): (1) that `ctm-clausesize` is a control whose "accuracy question is
  unanswerable at any budget because +0.06 is 16× below our band" — wrong once measured against
  the programme's own baseline (+7.50 pp); (2) that binding the budget requires dropping to batch
  5 — wrong, it binds at batch 50 for budgets ≥16; (3) their own ranking of `ctm-clausesize` above
  8,000 clauses as "the fourth thing I would cut" — withdrawn; a clause-size budget now belongs on
  every arm by default. They also withdrew their earlier recommendation of `max_chunk_elements =
  2**29` as a *programme-wide* standard, once the 5×5-window price list showed the dial is worth
  1.15× (not 1.2–1.5×) at the clause counts the shortlist actually uses.
- **The orchestrator** corrected `CHARTER.md`'s silent-failure mode 2, which had told all three
  agents that `max_included_literals` "does **not** bind for conv models under batched feedback" —
  a rule shown by the engineer's measurement to be true only at budgets ≤8 and false at the budgets
  every published recipe uses (16/32/64), and which had been steering the whole team away from a
  mechanism worth +7.50 pp. The orchestrator also retracted a wrong claim of their own that a
  compute queue had been "reaped at epoch 8" and lost work (`AUDIT.md` A16): the queue had never
  died: *"The retraction is recorded rather than the original claim, because an audit log with a
  fabricated failure mode in it is worse than one with a gap."*

## Evidence cited (result ids / paper references)

Primary-source `[FACT]`s: `arXiv:2406.00704` (Optimized Toolbox) §V, ref. 14, Tables III–IV;
`arXiv:1905.09688` (Granmo et al. 2019, Convolutional TM) §3, §5, Table 1; Sharma et al., AAAI
2023, Table 2 (Drop Clause); `arXiv:2309.04801` (TM Composites) Eqs 6–8, Tables 1–2;
`arXiv:2301.08190` (CSC-TM) Table 3, fn. 6; `arXiv:2507.14874` (GraphTM) §3.2, Tables 3, 16;
ISTM 2025 (CTM-UNet) Algorithm 1; BiPer (CVPR 2024) Table 1; BinaryConnect (`arXiv:1511.00363`)
Table 1; XNOR-Net (`arXiv:1603.05279`); `arXiv:1708.04552` Table 1; `arXiv:1608.06993` Table 2;
`arXiv:1312.4400`; `arXiv:1512.03385` §4.2; `arXiv:1708.06519` / `arXiv:1608.08710` Table 1;
Mathisen & Smørvik MSc thesis (cited at third hand, unreachable).

`[MEASURED]` records: `results/ctm-small_seed{0..4}.json` (seed band, match-count diagnostics);
`results/calibration_seednoise.json`; `results/calibration_throughput.json`;
`results/calib_grid_gpu0.json`, `calib_large_gpu0.json`, `calib_chunk_gpu0.json`,
`calib_gpu1_3080.json`, `calib_ladder5x5_gpu0.json`, `calib_zratio_gpu1.json`,
`calib_zratio5x5_gpu0.json`; `results/lg003_budget_sweep.json` and `code/repro/lg003.py`;
`results/ctm-small-budget32_seed{0,1,2}.json`; the `ctm-small-chunk29` and `ctm-small-protocol`
waves (`AUDIT.md` A19); `jobs/r1_protocol_penalty.txt`; `screen/cnn-ctmshape-max_seed0.json` and
`screen/cnn-ctmshape-sum_seed0.json`. Prior-programme evidence: `experiments/mctm/PLAN.md §3.1`
(firing-rate calibration 15.4%→42.6%; credit-propagation failure).

## The decision taken

`DR-001` (see `DECISIONS.md`), six parts:

1. **The target ladder is re-baselined.** T2 (primary) becomes **75.4%**; T3 (82.8%) is retained
   as a *reference*, not a target; T0 is re-pointed at the Toolbox's own 8,000-clause column
   (target left as `X`, **explicitly still blocked** pending values from the tm-theorist),
   cross-checked against AAAI 2023's 69.3%. A new primary statistic supersedes single accuracies:
   the slope of accuracy against log₂(clauses), with a promotion bar of "more than two doublings
   (≈+4 pp at matched automata budget), or it is cheaper to buy clauses."
2. A clause-size budget becomes a **default on every convolutional arm**, not a control run
   separately.
3. The P3 shortlist is fixed at ≈33 GPU-h for Tiers 0–1 plus a composite screen, of the ~120 GPU-h
   envelope (full tier breakdown in `DECISIONS.md` and `ARMS.md`).
4. Cuts, in order: `ctm-vanilla`@80,000/10×10 (−31.6 GPU-h, no CIFAR-10 number to score against);
   `ctm-graph-deep` (replaced by the far cheaper `ctm-multiview`, which tests the same claim);
   `ctm-hypervector`, `ctm-distill`, `ctm-sparse`/`ctm-absorbing`, `ctm-multitask-rgb` (no
   CIFAR-10 number, or forbidden TA-state changes).
5. Standing constraint: the faithful 82.8% (22 specialists × 64,000 clauses × 250 epochs) is
   **400–600 GPU-h for one seed** and is not reproducible on this machine; T3 may only be
   approached via the 2,000-clause composite (79.5% reported).
6. Process amendments: file ownership reassigned after a concurrent-edit collision on `ARMS.md`;
   a screening amendment exempting budget-dependent-sign mechanisms (augmentation) from
   promote-on-screen; a falsifier-discipline rule (a falsifier over a quantity the candidate itself
   changes is not a falsifier); confirmation that the §7.5 tolerance is **not** widened; `device +
   seed`, not `seed` alone, is now the reproducibility key; the chunk-budget prohibition is lifted
   (accuracy-neutral, +0.18 pp, inside the band) but with no single programme-wide default.

### Note-taker's judgement call on Decision 1, as requested

Decision 1 is a mixture of two different kinds of support, and the round's own dissent record says
so if read carefully:

- **`[FACT]`-backed, and solid**: that 60.7% is not a CIFAR-10 result; that 75.4% is the published
  single-model best; that 82.8% is a composite, not a single model; and the measured per-doubling
  deltas that produce "+2.18 pp/doubling, decaying" (+2.3, +2.3, +2.6, +2.0, +1.7 — read directly
  off `arXiv:2406.00704` Table IV) are all citations of a published table, not extrapolation.
- **`[HYPOTHESIS]`-backed, and explicitly still open**: the *promotion bar* built on that slope
  ("worth more than two doublings, ≈+4 pp at matched budget, or it is cheaper to buy clauses") is
  a policy choice about how to use the measured deltas, not itself a measured quantity — it has
  not been tested on the programme's own harness. `DR-001`'s own **Falsifier** clause admits this
  directly: *"if the 8k→32k segment of the capacity ladder gains ≈0, Decision 1's slope statistic
  is wrong … If it gains ≈+4.4 pp as the published table implies, mechanism candidates must clear
  the two-doubling bar."* That capacity ladder has not yet been run on this codebase — it is Tier
  0, item 2 of `DECISIONS.md`'s own P3 breakdown, still queued. T0's actual number (`X`) is also
  explicitly `PENDING`, not fixed, pending the tm-theorist's promised `T`/`s` values.
  In short: **the abandonment of 60.7%/82.8% as targets rests on `[FACT]`; the specific new bar
  ("two doublings") that will decide which P5 candidates get built rests on a `[HYPOTHESIS]`**
  extrapolated from someone else's clause-count/accuracy curve, applied here for the first time by
  policy, and due to be tested by our own capacity ladder before anything is promoted or killed on
  its strength.

**Dissent**: none outstanding at time of writing — the round's official dissent line records that
both experts withdrew their own C-2 positions against measurement (see Withdrawals, above).

## Action items

| Owner | Action | Output file(s) | By which gate |
|---|---|---|---|
| tm-theorist | Supply Toolbox 5×5-thermometer `T`/`s` (per-row, not just the 64k row), confirm class-owned/coalesced + "weighted" convention, the Toolbox's own 8,000-clause accuracy (target `X`), the exact HOG spec, and whether 250 epochs is required or an unexamined default | `LITERATURE_TM.md`, `ARMS.md` | Before any Tier 0 arm is admissible as a reproduction (blocks G3); requested for G1 |
| research-engineer | Implement HOG Booleanization (LG-008); run the Tier 0 capacity ladder (thermometer + HOG, {2k, 8k, 32k}, clause-size budget 32 as default) that resolves C-2 on our own harness | `results/*.json`, `ARMS.md` | G3 |
| research-engineer | Run the C-1 factorial screen (OR/count × no-density/density-control, ≥4 cells) as a unit; no standalone counting-pool arm | `screen/*.json` | G3 (feeds P5 candidate pool) |
| research-engineer | Run the C-3 data-budget-matched augmentation arm at ≥2 clause budgets (screening-exempt per Decision 6) | `results/*.json`, `ARMS.md` | G3 |
| research-engineer | Run `ctm-composite-samenc` (same-encoding vs diverse-encoding composite at equal total clauses) — separates diversity from ensembling for the 79.5%-vs-75.4%-at-0.69×-automata finding | `results/*.json` | G3 (Tier 2) |
| dl-expert | Run `cnn-boolean` on HOG bits and on thermometer bits — the theorist's own named falsifier for "the ceiling is the encoding's, not the TM's" | `results/*.json`, `LITERATURE_CNN.md` | G2/G3 |
| dl-expert | Continue P2 CNN baselines under §7 protocol (`cnn-resnet18`, `cnn-small`, sample-efficiency curves) | `results/*.json`, `LITERATURE_CNN.md` | G2 |
| orchestrator | Take Decision 1 (ladder re-baseline) to the user for confirmation, flagging the FACT/HYPOTHESIS split above | `DECISIONS.md` | G1 (digest sent) |
| orchestrator | Write `DR-002` once the capacity ladder lands, formalising or rejecting the two-doubling promotion bar as the programme's P5/P6 comparison statistic | `DECISIONS.md` | After Tier 0 capacity ladder (pre-G4) |
