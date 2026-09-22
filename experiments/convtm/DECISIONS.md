# Decision records

Append-only. One entry per decision, numbered `DR-001`, ... A record is **never edited**; it is
superseded by a later record, which names it.

A record whose **Evidence** block contains no `[MEASURED: …]` entry is rejected by the orchestrator,
per constraint C3. A decision that must be taken before its measurement exists is written as
`PROVISIONAL-PENDING-EXPERIMENT <exp-id>`, and nothing downstream may depend on it until `<exp-id>`
lands.

Template:

```markdown
## DR-00N — <one-line decision>
- **Date**: YYYY-MM-DD
- **Round / phase**:
- **Question**:
- **Decision**:
- **Evidence**:
  - [MEASURED: <result-id>] …
- **Dissent**:
- **Falsifier**: what result would overturn this
- **Supersedes** / **Superseded by**:
```

---

## DR-000 — Programme configuration accepted; P0 and P1 launched
- **Date**: 2026-09-20
- **Round / phase**: Gate 0 → P0/P1
- **Question**: Do we run the programme as specified in PLAN.md, and with what targets, budget and autonomy?
- **Decision**: Yes, with the plan's defaults, confirmed by the user:
  - target ladder as in PLAN.md §2 — **T2 (beat 60.7%, the single-model TM SOTA) is the primary
    objective**; T3 (82.8%) and T4 (within 5 points of our own small CNN) are stretch;
  - compute budget 220–300 GPU-hours across the two GPUs;
  - P3 shortlist target ~12 arms, to be fixed by DR-001;
  - **selection moves to validation** (the fixed 45k/5k split), breaking comparability with
    `experiments/mctm`'s `best_test_acc` numbers; prior arms reused as baselines are re-run;
  - autonomy: proceed G0→G7, report at every gate, stop for user approval at **G4** (limitations)
    and before **P6**'s large runs.
  - P0 and P1 run **concurrently** rather than in sequence as PLAN.md §14.3 had them — P1 is pure
    reading and does not depend on the harness. Saves roughly a day of wall-clock.
- **Evidence**: this is a process decision, not a method decision, so constraint C3 does not apply.
  No method is selected here. The one substantive input is
  [MEASURED: experiments/mctm/results/*] — the prior programme's 99 result files, which set the
  starting point recorded in PLAN.md §3 and are the reason greedy supervised stacking is not on the
  P3 list.
- **Dissent**: none recorded; the team had not yet been convened.
- **Falsifier**: if P0's measured throughput shows the ~12-arm P3 shortlist cannot fit in 300
  GPU-hours, the shortlist shrinks and DR-001 records the cut.
- **Supersedes / Superseded by**: —

### Deviation from PLAN.md worth recording
`.claude/agents/*.md` are not loaded into an already-running Claude Code session. The five agent
files are written and will register on restart; for this session the specialists run as
general-purpose agents whose first instruction is to read their own role file. Same role definition,
one indirection. No effect on the protocol.

## DR-001 — P3 shortlist fixed, ladder re-baselined, clause-size budget made a default
- **Date**: 2026-09-20
- **Round / phase**: ROUND-1 → P2/P3
- **Question**: which existing approaches do we re-implement, in what order, at what cost — and is the PLAN §2 target ladder correctly set?

### Decision 1 — the ladder is re-baselined (**user confirmation required on this item only**)
PLAN §2's T0 (55–61%) and T2 ("beat 60.7%") are withdrawn. 60.7% is not a CIFAR-10 result.
- **T0** = the Toolbox's own 8 000-clause column at matched encoding (target `X`, pending §Blocked),
  cross-checked against AAAI 2023's vanilla CTM **69.3%**.
- **T2 (primary)** = **75.4%**, the best single-model TM.
- **T3** = 82.8%, retained as a *reference*, not a target — see Decision 5.
- **New primary statistic, superseding single accuracies**: the slope of accuracy against
  log₂(clauses). Measured in the corpus at **+2.18 pp/doubling and decaying**. A candidate must be
  worth **more than two doublings (≈ +4 pp at matched automata budget)** or it is cheaper to buy
  clauses. This is the promotion bar for P5 and the headline statistic of the report.

### Decision 2 — a clause-size budget is a **default on every convolutional arm**
Not a control, not a hazard. Any arm run without one is the control, and must say so.

### Decision 3 — P3 shortlist (≈33 GPU-h for Tiers 0–1 + composite screen, of a ~120 h envelope)
**Tier 0 (~15.5 h)** — 8k pre-flight · thermometer capacity ladder {2k, 8k, 32k} × 5×5 · HOG ladder ·
`cnn-boolean` on HOG bits · conditional 64k point.
**Tier 1 (~16 h)** — Booleanization family at 2k (10 encodings) · drop-clause · data-budget-matched
augmentation (2×2×2) · counting-pool 2×2 screen · weighted · coalesced · unconstrained-baseline control.
**Tier 2** — composite screen at 2k, then the 22-specialist 2 000-clause composite (15–27 h, reported
79.5%), plus `ctm-composite-samenc` — the diversity-vs-ensembling control that has never been run.
**Also in**: `ctm-coalesced`@80k (exact published target **66.42 ± 0.19**, 4.3 h) · `ctm-multiview` ·
`ctm-cnnfeature` (M1, 0.5 h) · `mctm-calibrated` re-run under this protocol.

### Decision 4 — cuts, in order
1. `ctm-vanilla`@80 000 / 10×10 — **no published CIFAR-10 number to score against**; the capacity
   ladder subsumes it. −31.6 GPU-h. (Both other members had proposed it; the engineer's argument won.)
2. `ctm-graph-deep` — its CIFAR-10 run is depth 1 and the authors attribute the gain to multi-view
   input; replaced by `ctm-multiview` at ~1.6 h.
3. `ctm-hypervector`, `ctm-distill`, `ctm-sparse`/`ctm-absorbing`, `ctm-multitask-rgb` — no CIFAR-10
   number, and the last two need TA state-machine changes that C1 forbids.

### Decision 5 — standing constraint: the faithful 82.8% is not reproducible on this machine
22 specialists × 64 000 clauses × 250 epochs prices at **400–600 GPU-h for a single seed**. Recorded
here so it is not rediscovered. T3 may only be approached via the **2 000-clause** composite (79.5%
reported), and any composite result is compared against composite baselines, never single models.

### Decision 6 — process amendments
- **File ownership**: `ARMS.md` → research-engineer (it was clobbered by a concurrent edit).
  `LIBRARY_GAPS.md`, `AUDIT.md` → append-only, multi-author. `RISKS.md`, `DECISIONS.md` → orchestrator.
- **Screening amendment to PLAN §8 P5.3**: the plan defended against screening *false negatives*
  only. Augmentation is measured to **flip sign with clause budget** (helps 10 of 11 encodings at
  2 000 clauses; hurts every thermometer encoding at 64 000), so a screen would yield a confident
  **false positive that reverses at full scale**. Any mechanism whose effect is known to depend on
  clause budget is **exempt from promote-on-screen** and must be run at two budgets or not at all.
- **Falsifier discipline**: a falsifier stated over a quantity **the intervention itself changes** is
  not a falsifier. Both ROUND-1 pre-registrations violated this (the match-count histogram is the
  equilibrium of a model trained *under OR pooling*). Falsifiers must be stated over quantities
  invariant to the candidate, or over the candidate's own post-intervention measurement.
- **§7.5 tolerance is NOT widened.** The protocol penalty is measured at ≤ 0.25 pp, not the ≤ 1 pp
  hypothesised. Report val-selected, last-25-mean and peak; score against the paper's own estimator.
- **`device + seed`** is the reproducibility key, not `seed`. Arms may span both cards.
- **Chunk-budget prohibition lifted** (accuracy-neutral, +0.18 pp, inside the band); no non-default
  value adopted as standard, since the gain at the clause counts that matter is only 1.15×.

- **Evidence**:
  - [MEASURED: results/ctm-small-budget32_seed{0,1,2}] **44.58 ± 0.55 vs 37.08 ± 0.61** unconstrained
    — +7.50 pp, 7.5× the seed band, intervals disjoint, 4.8× fewer literals/image, +1.7% wall-clock.
    Basis for Decision 2 and for the charter correction.
  - [MEASURED: results/calibration_seednoise.json] seed band **1.00 pp** for 3-seed means; selection
    contributes only ~0.09 pp of it.
  - [MEASURED: results/calibration_throughput.json + Z-column addendum] price list
    `s/ep ≈ (C/1000)·coef(k)`, `coef = {4×4: 3.7, 8×8: 7.3, 10×10: 10.1}`; adaptive (Z=3) is
    **1.58–2.09× cheaper than therm4 at 5×5**, not the 3.6× the `F` model predicted. Basis for
    Decisions 3 and 4.
  - [MEASURED: results/ctm-small-budget-sweep] budget 8/16/32/64 → median 10.5/15.0/20.0/25.0; the
    overshoot is **absolute, not proportional**. Basis for Decision 2 and for retiring the
    "budget does not bind" rule.
  - [MEASURED: screen/cnn-ctmshape-{max,sum}_seed0] 38.86% vs 51.02% — the counting-pool proxy,
    carried as a **bound, not evidence** (Decision 3, Tier 1).
  - [MEASURED: results/ctm-small_seed{0..4} → diagnostics.match_count] median 2 given firing,
    P(|M|≥5 | fires) = 0.303. Both pre-registered falsifiers failed; basis for the falsifier
    discipline amendment.
  - [FACT: arXiv:2406.00704 §V + ref. 14] 60.7% is on a **modified** CIFAR-10, cited at third hand —
    independently verified by the orchestrator against the PDF. Basis for Decision 1.
  - [FACT: arXiv:2406.00704 Table IV] single-model best **75.4 ± 0.09%**; capacity curve
    64.5→75.4 over 2k→64k, **still climbing**; composite saturates at +0.1/doubling by 32k;
    22 specialists × 2 000 clauses (44 000 total) = **79.5%** vs one specialist at 64 000 = 75.4%.
    Basis for Decisions 1, 3 and 5.

- **Dissent**: none outstanding. Both experts withdrew their own C-2 positions against measurement —
  the DL expert's "saturates below 70% by 8–16k" was falsified by its own pre-registered criterion,
  and the theorist withdrew "almost never an OR over more than one term" (64.6% match ≥2) and the
  reading of a rising curve as a practical route.
- **Falsifier**: if the 8k→32k segment of the capacity ladder gains ≈ 0, Decision 1's slope statistic
  is wrong and the programme re-aims at mechanism work as the accuracy route. If it gains ≈ +4.4 pp
  as the published table implies, mechanism candidates must clear the two-doubling bar to be worth
  running at all.
- **Blocked on**: the Toolbox's `T`/`s`, its class-owned-vs-coalesced convention, its 8 000-clause
  accuracy (target `X`), the HOG specification, and whether 250 epochs is a requirement or a default
  (18.2 vs 4.4 GPU-h per seed at 64k). Requested from tm-theorist; HOG implementation authorised to
  proceed in parallel.
- **Supersedes**: DR-000's ladder. **Superseded by**: —

## DR-002 — Clause convention is per class (10× budget correction); DR-001 Decision 2 made provisional
- **Date**: 2026-09-20
- **Round / phase**: post-ROUND-1, before Tier 0 launch
- **Question**: what does the published clause count actually mean, and does DR-001's budget survive it?

### Decision 1 — "clauses" means **per class**; every Toolbox-scored arm runs at 10× the face value
Confirmed independently twice, so this is not an inference:
- [FACT: arXiv:2406.00704 §II-B] *"a group of n conjunctive clauses **for each class**"*, Eq. (2)
  summing j=1..n/2 positive minus n/2 negative **per class**.
- [FACT: cair/tmu `vanilla_classifier.py`] `clause_banks.populate(range(number_of_classes))` — one
  bank of `number_of_clauses` per class; `transform()` returns `(N, n_classes × number_of_clauses)`.

The paper's "2,000" is **20 000 total**; its "8,000" is **80 000 total**. **There is no 8 000-total
cell in the paper**; its cheapest measured cell is 20 000 total. The model is class-owned with fixed
polarity — `ConvTsetlinMachine`, as the engineer had already chosen for `ctm-vanilla`.

### Decision 2 — T0 comes off PENDING
Pre-flight: **20 000 total clauses, 5×5 colour thermometers, 50 epochs**, target **X = 63.3**
(paper's 64.5 at 250 epochs, minus a measured 1.2 pp epoch correction). Rule: **≥ 60.3 proceed ·
55.3–60.3 hyperparameter fault · < 55.3 implementation fault.** 2.1 GPU-h, no extrapolation.

### Decision 3 — DR-001's budget figure is withdrawn; the shortlist must be re-priced
"~33 GPU-h for Tiers 0–1" was priced at face-value clause counts and is under-funded by ≈3.5×. The
theorist's re-priced list is **117.4 GPU-h** against a ~120 h envelope, i.e. the shortlist now
consumes essentially the whole budget and arms must be cut deliberately. Engineer to return the
re-priced Tier 0 before the ladder launches. **Partly offset**: [FACT: Fig. 3 digitised, recovered
finals match Table IV to ≤ 0.3 pp] **250 epochs is an unexamined default** — 100 epochs costs
0.3–0.5 pp (inside the seed band, i.e. free) for a 60% saving; 50 epochs costs 1.1–1.2 pp. Adopted:
**100 for reproductions, 50 for ladders/screens/pre-flights with a −1.2 pp target correction.**

### Decision 4 — DR-001 Decision 2 (clause-size budget as a default) is now **PROVISIONAL**
Two reasons, both of which must be discharged before the capacity ladder is built on top of it:
- **The baseline's `T` is unreachable.** [MEASURED: ctm-small hp] 2 000 total / 10 classes = 200 per
  class = **100 positive** clauses, unweighted → maximum achievable class sum **100**, against
  **T = 160**. Feedback probability never anneals (0.478 unconstrained, 0.450 at budget 32).
  **`T` is inert in both arms, so the +7.50 pp and `T` are confounded.** Fix: compute `T_ratio`
  against the positive half (`n_clauses_per_class / 2`) unweighted, or summed positive weights when
  weighted. Re-measure the pair at a correct `T`.
- **The effect may be an artefact of our own batching.** Registered prediction P3: under
  `feedback_mode="sequential"` the +7.50 pp shrinks to ≤ +1 pp, matching CSC-TM's published +0.06
  (their unconstrained clause length 60.4 vs our 156.4). This decides whether the result is *a
  finding about Tsetlin machines* or *a caveat about our approximation*. Prioritised above the ladder.

### Decision 5 — standing rule: read the reference code before declaring a `GAP`
Three load-bearing settings in this paper family are undocumented — budget 32 appears in all 22
reference scripts but not in Table III; no validation split; transductive `α_t`. [HYPOTHESIS] the
modal cause of a T0 shortfall here is an **undocumented default**, not an implementation fault.

### Decision 6 — the composite cut hardens
At per-class counts the faithful 22-specialist composite holds **14.08 M clauses ≈ 4 000–6 000
GPU-h/seed**, not the 400–600 recorded in DR-001 Decision 5. An order of magnitude further out of
reach. The 2 000-*per-class* (20 000 total) composite remains the only reachable form.

- **Evidence**: [FACT: arXiv:2406.00704 §II-B, Eq. 2]; [FACT: cair/tmu vanilla_classifier.py];
  [FACT: arXiv:2406.00704 Fig. 3 digitised]; [MEASURED: results/ctm-small_seed* hp + feedback
  probability]; [MEASURED: results/ctm-small-budget32_seed*].
- **Dissent**: none. The engineer's own convention choice for `ctm-vanilla` is independently
  confirmed by this.
- **Falsifier**: if the re-measured budget pair at a correct `T` still shows ≥ +5 pp **and** the
  sequential control also shows ≥ +5 pp, DR-001 Decision 2 becomes final rather than provisional.
- **Supersedes**: DR-001 Decisions 2 (now provisional), 3 (budget figure) and 5 (composite cost).

## DR-003 — C-2 redesigned around a hardware wall; P3 funding restated and ranked
- **Date**: 2026-09-20
- **Round / phase**: pre-Tier-0
- **Question**: C-2 is not runnable as designed. What replaces it, and what does P3 fund?

### Decision 1 — DR-002 Decision 4 is discharged: the clause-size budget is **final, not provisional**
[MEASURED: results/ctm-small-T80-budget32_seed*] With `T` corrected from an unreachable 160 to a
reachable 80: **38.31 ± 1.05 unconstrained vs 47.77 ± 0.32 at budget 32 = +9.46 pp**, seed intervals
disjoint. Correcting `T` **added 2 pp to the effect** (+1.23 unconstrained, +3.19 at budget 32), so
budget and margin are **complements, not substitutes**, and the result was not a `T` artefact.
For scale: this is larger than **Drop Clause's +5.8**, the best published single-model CIFAR-10
mechanism. The sequential control (P3) is still running and remains the one open falsifier; seed 0
shows 47.80 batched vs 46.76 sequential, i.e. no sign yet of a batching artefact.

### Decision 2 — C-2 redesign, **accepted** (forced by measurement, not chosen)
HOG holds **11 664 literals per clause**; memory scales with *clauses × literals* through the
feedback accumulator, so it reserves 9.5 GB at 20 000 clauses and **OOMs a 24 GB card at 40 000**.
HOG's published plateau begins at 640 000 total — **32× beyond what fits**. Two capacity curves of
opposite shape are therefore not runnable on this hardware. Replacement:
1. **Thermometer capacity ladder** on the TM at {20 000, 40 000, 80 000} — the reachable part of the
   published curve, over which thermometers are predicted to gain ~5.5 points against HOG's ~2.3.
2. **`cnn-boolean` on HOG bits is promoted to the *primary* HOG evidence.** It was already the
   theorist's own named falsifier for the encoding-ceiling claim, and a CNN over 5 832 bits has none
   of the TM's `C × F` memory problem.
3. **HOG TM at its single reachable point** as a *level* comparison against the thermometer there —
   the published values are 64.2 vs 64.5, a near-equality prediction worth testing in its own right.

### Decision 3 — P3 funding restated; Tier 1 is **not** unfunded
DR-001's "~33 GPU-h for Tiers 0–1" was the erroneous face-value figure and is withdrawn (DR-002
Decision 3). **The actual P3 envelope is PLAN §10's 80–120 GPU-h.** Tier 0 at **31.8** therefore
leaves **~50–88 GPU-h**. The theorist's re-priced 117.4 h full list fits the top of the range with no
slack, so the ordering below is the funding order, not a wish list.

**Funded, in priority order:**
1. **Tier 0 core** (20.3 h) — pre-flight at 20 000 · thermometer ladder · HOG point · `cnn-boolean`
   on HOG bits. **Adaptive seeding**: run 1 seed; add the second (+16.3 h) **only if** the 20k→80k
   gain lands within 2× the seed band of the ±4 pp decision boundary. Do not buy seeds in advance.
2. **Budget control** (11.5 h) — unconstrained at 20 000 and 80 000. Two points, not a full parallel
   curve; two points reveal a *change* in the gap, which is what would bend the slope.
3. **The clause-budget prediction sweep** (P1–P4, cheap at `ctm-small` scale) — the sequential
   control, the optimum-budget sweep, and the firing-rate-mediation test. This is now the
   programme's largest own-measured effect and its mechanism is unestablished.
4. **The diversity-vs-ensembling control** — the novel finding neither the literature nor either
   expert had alone, and the cheapest route to a contribution.
5. **`ctm-dropclause`** — the only published single-model mechanism above 1 point, and the thing our
   +9.46 pp must be compared against.
6. **`ctm-weighted`** — must be measured before any readout mechanism of ours can be called novel.
7. **Booleanization family, reduced from 10 encodings to 3–4 spanning the published range** — the
   16.7-point spread is the largest published lever, but measuring all ten does not beat any of them.

**Deferred to P4/P5 or cut**: the counting-pool 2×2 screen (its own author revised it to +0–1.5 pp,
at or below the band); augmentation (sign-flips with clause budget, so it needs two budgets at 10×
the old price); `ctm-coalesced`@80k (keep only if slack remains after 1–6).

### Decision 4 — two configurations are **cite-only**, not reproducible here
- The paper's best cell, 64 000/class = **640 000 total ≈ 72 GPU-h/seed**.
- The faithful 22-specialist composite, **14.08 M clauses ≈ 4 000–6 000 GPU-h/seed** (DR-002 Dec. 6).
Both are reported as published values with our reachable points beside them, never as reproductions.

- **Evidence**: [MEASURED: results/ctm-small-T80-budget32_seed*]; [MEASURED: AUDIT A21–A23 — the
  measured 20 000-total thermometer cell at 180.7 s/epoch against a 135 s extrapolation, i.e. the
  engineer's own estimate was 34% low]; [MEASURED: HOG OOM at 40 000 on a 24 GB card].
- **Dissent**: none outstanding.
- **Falsifier**: if the sequential control returns ≤ +1 pp across 3 seeds, Decision 1 reverts and the
  clause-budget result becomes a caveat about our batching rather than a finding about TMs.
- **Supersedes**: DR-002 Decision 4 (discharged); DR-001 Decision 3 (funding).

## DR-004 — T0 discharged (above the published value); the clause-budget finding survives its falsifier
- **Date**: 2026-09-21
- **Round / phase**: Gate G3 entry
- **Question**: does the harness reproduce the literature, and is the clause-size result real?

### Decision 1 — **T0 is met.** The harness is trusted; P4–P6 are unblocked
[MEASURED: results/ctm-therm5-preflight_seed0.json] 20 000 total clauses, 5×5 colour thermometers,
T=3000, s=5.0, budget 32, **50 epochs**: **val 65.82%, test 65.10%.**
- Against DR-002's target `X = 63.3` (the paper's 64.5 minus a 1.2 pp epoch correction): **+2.52 pp**,
  well inside "proceed" (≥ 60.3).
- Against the paper's own **64.5% at 250 epochs**: **+1.32 pp on val at one fifth of the epochs.**
This is the first time this programme has matched a published TM CIFAR-10 result, and it clears it.
It also retroactively validates DR-002's epoch analysis (100/50-epoch policy) and the standing rule
to read the reference code — budget 32 came from the reference scripts, not from any paper table,
and without it this cell would almost certainly have undershot.

### Decision 2 — the clause-size budget is a **finding about Tsetlin machines**, not an artefact of our batching
The DR-003 falsifier was: sequential feedback yields ≤ +1 pp ⇒ revert. It did not.

| feedback mode | unconstrained | budget 32 | effect |
|---|---|---|---|
| **sequential** (exact algorithm) | 38.06 ± 0.38 | 46.80 ± 0.17 | **+8.75 pp** |
| **batched** (our default) | 38.31 ± 1.05 | 47.77 ± 0.32 | **+9.46 pp** |

The two modes differ by 0.71 pp — **inside the 1.00 pp seed band** — so batching is not distorting
the effect at all. [MEASURED: results/ctm-small-seq-{unc,b32}_seed*, ctm-small-T80-{unc,b32}_seed*]

**Scope of the claim, stated now so the report cannot overreach.** The *mechanism* is published
(CSC-TM, clause-size constraint). What is new is its **magnitude on CIFAR-10 convolutional TMs** —
its authors reported **+0.06 on CIFAR-2**, and the difference is explained by baseline clause bloat:
their unconstrained clauses ran to 60.4 literals, ours to **156.4 of 248 includable (63%)**. The
claim is therefore "this published mechanism is worth ~9 points where it was reported as worth
0.06, and here is why", **not** "we invented a mechanism".

- **Evidence**: as cited above; both verified by the orchestrator directly from the result records.
- **Dissent**: none. This discharges the last open falsifier from ROUND-1.
- **Falsifier (remaining)**: the effect is measured at 2 000 total clauses / 4×4. If the budget
  control at 20 000 and 80 000 (running) shows the gap closing with scale, the finding is
  scale-dependent and the report must say at which scale it holds.
- **Supersedes**: DR-003 Decision 1's provisional status (now final).

## DR-005 — P2 launched; scope, deviations and the capacity finding
- **Date**: 2026-09-21
- **Round / phase**: P2, concurrent with P3 Tier 0
- **Question**: what does P2 cost, what deviations are accepted, and what does it exist to settle?

### Decision 1 — P2 scope accepted at **47.3 GPU-h (+7.7 optional)**, repriced after the first hour
PLAN §10 budgeted 20 h for P2. The estimate is 2.4× that, but it is derived from a **single**
memory-bound calibration arm and is, by its author's own assessment, likely **2–3× pessimistic on
the dense conv arms**. The queue is ordered cheapest-first precisely so measurement replaces the
estimate early. **Standing instruction: reprice from measured records after the first hour and
report; if the measured total exceeds 30 GPU-h, cut `p2_cnn_ref` (20.9 h of reference nets nobody
disputes) before cutting anything that carries an argument.**

### Decision 2 — deviations accepted
- **Sample-efficiency curve moved ahead of the augmented reference nets** (3.0 h vs 20.9 h). The
  curve is a deliverable the TM side is scored against; the reference nets are numbers nobody
  disputes. Accepted.
- **`binaryconnect` / `xnor` / `binary-aug` cut to 1 seed and moved last.** At 3 seeds they alone
  were 36% of P2 for variants that refine rather than establish the decomposition. Reported with
  their seed count, excluded from headline comparisons. Accepted.
- **Binary arms kept at 200 epochs** despite cost: they *bound* what binary computation can reach,
  and under-training them biases the decomposition in the direction that flatters the TM. Accepted —
  this is the right instinct and the reasoning goes in the report.
- **Sample-efficiency top point is 45 000, not 50 000**, because the protocol holds 5 000 out.
  Correct; a 50 000-image CNN against a 45 000-image TM is the unlike comparison the curve exists
  to avoid.

### Decision 3 — `code/CONTRACT.md` amended by the orchestrator (my file, my error)
The contract asserted one runner for both families. It was never true: `run_arm.py` loads a Boolean
dataset unconditionally and calls TM-only methods, and `arms.py` does not call
`register_cnn_arms()`. Amended to state the real design — **two drivers, one shared per-GPU lock**
(`queue_cnn.py` imports `queue_runner._acquire`), with comparability carried by the shared
`record.py` schema and `split_hash`, not by a shared runner. The DL expert correctly refused to
edit engineer-owned files to paper over this and reported it instead.

### Decision 4 — the capacity comparison goes in the report **as measured, no extra runs**
[MEASURED: results/ctm-therm5-preflight_seed0.json `capacity`] The flagship TM holds **26.2 M
automata = 104.6 MB of state — 2.3× more storage than ResNet-18's 11.2 M fp32 parameters** — and
evaluates **436 M included literals per image**, between `cnn-small` (165 M MACs) and ResNet-18
(555 M MACs). At **65.10%** against ResNet-18's expected ~94.5%.
**The "Tsetlin machines are small and cheap" framing does not survive contact with its own capacity
record.** This is a finding the field has not stated and it costs nothing further to make.

### Decision 5 — P2's pre-registered decomposition (prediction, not result)
Recorded before the runs so it cannot be fitted afterwards. Predicted shares of the ~29.4 pp gap:
augmentation ~17%, depth/capacity ~12%, **Booleanization ~5–8%**, **binary computation ~0%**,
**single-layer + OR pooling ~62%**, **learning algorithm ~0%**. Headline: *Booleanization and binary
arithmetic together account for under 15% of the gap; the architecture accounts for roughly two
thirds; the Tsetlin learning rule itself may cost approximately nothing.*
**If this holds, P5 candidate generation targets architecture — hierarchy, pooling that preserves
count information, receptive-field growth — and not the feedback rule.**
Named falsifiers, all stated over CNN accuracies (quantities the intervention does not itself
change, per DR-001 Decision 6's falsifier discipline):
1. `cnn-ctmshape-max` at full data ≥ 75% → the architecture is not the bottleneck, the decomposition
   inverts, and P5 should target feedback instead.
2. `cnn-small-noaug` − `cnn-boolean-therm8` ≥ 8 pp → Booleanization is first-order after all.
3. `cnn-binary` < 75% → binary computation *is* a first-order cost.
4. `cnn-hog-mlp` ≥ 72% → HOG has headroom, the TM's HOG plateau is a TM limitation rather than an
   encoding ceiling, and the encoding-ceiling reading of C-2 dies.

- **Evidence**: [MEASURED: results/ctm-therm5-preflight_seed0.json]; [MEASURED:
  screen/cnn-ctmshape-{max,sum}_seed0.json]; [MEASURED: jobs/p2_cnn_cost.json, generated from
  instantiated models]. Decision 5 is explicitly `[HYPOTHESIS]` and binds nothing until measured.
- **Dissent**: none.
- **Supersedes**: PLAN §10's 20 h P2 line.

## DR-006 — A rival explanation is registered against our own headline; falsifier discipline extended
- **Date**: 2026-09-21
- **Round / phase**: P3, theory consolidation
- **Question**: is the +9.46 pp attributable to the clause-size budget specifically, or to clause length by any means?

### Decision 1 — DR-004 Decision 2 carries an **open rival explanation** until Block 2 lands
`s = 10` at therm-4 / 4×4 is **our own choice** — `ctm-small` carries `config_status="own"` and has no
paper to be faithful to — while the Toolbox uses **`s = 5.0`** for every colour-thermometer
specialist [FACT: arXiv:2406.00704 Table III]. The theory's own admission threshold is `1/(1+s)`
(0.091 at s=10, 0.333 at s=2), so **a lower `s` should shorten clauses by itself**, and part of the
measured bloat may be an untuned hyperparameter rather than a missing mechanism.

This repository's **own stored memory said "use `s` instead"** — written before any of these
measurements and never tested until now.

**Falsifier, pre-registered**: if `ctm-small-T80-s2-unc` lands within the 1.00 pp band of
`ctm-small-T80-b32` (47.77), DR-004's claim is rewritten from *"the clause-size budget is worth 9
points"* to *"clause length is worth 9 points, and the budget is one of two dials that buy it"* —
weaker, still publishable, but a **different claim**. Block 2 of `jobs/p3_budget_predictions.txt`
decides it for **1.25 GPU-h** and is launched ahead of Blocks 1, 3 and 4. The theorist's own
`[HYPOTHESIS]` is that the budget is *not* substitutable, because `s` thresholds literal *frequency*
(a low-pass filter on π) while the cap quotas literal *count* (a top-`b` selection) — so an
`s`-shortened clause should be shorter **and less accurate** at the same length.

### Decision 2 — numerical correction to DR-004
`diagnostics.clause_len` is computed on the **validation-selected snapshot**, not the final model.
DR-004's "156.4 of 248 (63%)" came from the inert-`T` arms at the selected snapshot. **The correct
equilibrium figure at a reachable `T` is 170.36 ± 0.28 = 68.7% of includable literals**, reproducing
across seeds to ±0.28 literals — a dynamical attractor, not noise. All downstream text quotes 68.7%.

### Decision 3 — falsifier discipline extended (amends DR-001 Decision 6)
DR-001 required a falsifier to be stated over a quantity the intervention does not itself change.
That is necessary and **not sufficient**: the theorist's P3 was correctly aimed by that rule and
still mis-aimed, because it named only the mechanism they had thought of ("batching explains it")
and never registered the cheaper rival ("an untuned `s` explains it").

**New rule: a pre-registration must name at least one *rival* explanation and the measurement that
separates it from the preferred one — not only the falsifier of the preferred one.** Adopted on the
proposal of the member whose own pre-registration it convicts.

### Decision 4 — the bloat-versus-headroom discriminator is funded
CSC-TM measured this mechanism at +0.06 on CIFAR-2; we measure +9.46 on CIFAR-10. Two accounts fit:
**bloat** (their unconstrained clauses 60.4 literals, ours 170.4 of 248) or **headroom** (their
unconstrained baseline already at 94.18% on two classes, ours at 38.31% on ten). They are *not* two
points on one curve. Running our own budget pair on a CIFAR-2 built from **our** pipeline separates
them for **0.62 GPU-h** and ~10 lines in `code/data.py`. Funded, ranked above sweep Blocks 3–4.

### Decision 5 — the measured ratchet, recorded as the mechanism
[MEASURED: results/ctm-small-T80-unc_seed0 curve] Epochs 1→30: clause length **86.7 → 170.6 (+97%)**
while the image firing rate moves **−1.4%** and validation accuracy **+1.1 pp**; epochs 5→30 alone
add **+34.4 literals for +0.32 pp, inside the seed band**. Per-literal selectivity `−ln q`:
**0.0440 ± 0.0002 unconstrained vs 0.2030 ± 0.0007 at budget 32 — 4.61×**.
**Clause length had not converged at 30 epochs; accuracy converged at epoch ~4.** The literals
arriving after epoch 5 are on average logically implied by those already present.
The account: inclusion is a **ratchet** — admission asks only whether a literal is frequent inside
the match set (`π > 1/(1+s)`), never whether it is useful; the only eviction path is Type Ib, i.e.
the clause failing to fire, at gain `1/s`; and **the budget is the only mechanism in the algorithm
whose activation is a function of clause length**.

- **Evidence**: [MEASURED: ctm-small-T80-{unc,b32}_seed*, ctm-small-seq-{unc,b32}_seed*];
  [FACT: functional.py:232-251, conv.py:167-176]; [FACT: arXiv:2301.08190 Table 3 + fn. 6];
  [FACT: arXiv:2406.00704 Table III].
- **Dissent**: none — Decisions 1 and 3 were both raised by the member they cost the most.
- **Supersedes**: DR-004 Decision 2's unqualified form; DR-001 Decision 6 (extended).

## DR-007 — DR-006's rival explanation is dead: `s` does not substitute for the clause-size budget
- **Date**: 2026-09-22
- **Round / phase**: P3, Block 2 complete (6 of 6 cells)
- **Question**: is the +9.46 pp attributable to the budget specifically, or to clause length by any dial?

### Decision 1 — **DR-004 stands unqualified.** DR-006 Decision 1's rival is falsified
[MEASURED: `code/summarize_budget.py --family s`] Pre-registered comparison: `s=2` unconstrained
**36.78 ± 0.99** against budget-32 **47.93 ± 0.19** — a gap of **+11.15 pp against a 1.00 pp band**.
Tuning `s` does not buy what the budget buys. DR-006's rewritten weaker claim is **not** adopted;
DR-004's original form is restored.

Stronger: **the budget pays at every `s` tested**, and its value does not depend on `s` being
mis-set —

| `s` | unconstrained | budget 32 | effect |
|---|---|---|---|
| 2 | 36.78 ± 0.99 | 48.31 ± 0.17 | **+11.53 pp** |
| 10 | 38.31 ± 1.05 | 47.77 ± 0.32 | **+9.46 pp** |
| 20 | 36.36 ± 0.58 | 45.78 ± 0.54 | **+9.42 pp** (n=2, reported not a verdict) |

All seed intervals disjoint. The best accuracy in the whole family is **`s=2` + budget 32 = 48.31**,
i.e. the two dials are **complementary**, exactly as the theorist predicted and contrary to this
repository's prior stored advice to "use `s` instead".

### Decision 2 — theory scorecard: C5.2 confirmed quantitatively, B1 confirmed in sign and **falsified in magnitude**
- **C5.2 (equilibrium firing rate `f* = 1/(1+s)`) — confirmed, and this is the theory's strongest
  quantitative success so far.** Measured `fire|eligible ÷ (1/(1+s))` for the unconstrained arms:
  **1.13, 1.12, 1.11** at s = 2, 10, 20 — constant to ~1% across a **7× movement** in the predicted
  quantity. This is the prediction D-FIRE was added to test, and it could not have been tested from
  the global firing rate (which mixes polarities).
- **B1 (clause length rises with `s`) — direction right, magnitude wrong.** `L₃₀` = **160.40 ± 0.49 /
  170.36 ± 0.28 / 172.78 ± 1.35** at s = 2/10/20: a **7.7% change for a 10× change in `s`.** The
  theorist predicted that at s=2 the admission threshold (0.333) would drive `L_unc` **below 32**,
  collapsing β to ≈1 and making the budget effect vanish. It did not: `L_unc` stayed at 160.
  **The admission threshold is therefore not the rate-limiting term for clause length**, and
  THEORY §5.5's C5.1 needs revision. The ratchet is real (the budget still pays everywhere) but its
  *governor* is not `1/(1+s)`.

### Decision 3 — the harness is verified across today's changes at **zero tolerance**
[MEASURED: ctm-small-T80-s10-{unc,b32}] The re-run under new labels returned **37.50 / 37.93 / 39.50**
and **47.80 / 48.07** — exact to the digit, against five changed files and a new diagnostics module.
This licenses the RNG-neutrality finding and means no arm run after 2026-09-21 is suspect.

### Decision 4 — the generator refused a verdict it was not entitled to
`summarize_budget.py` declined the `s=10` budget comparison on seed mismatch ([0,1,2] vs [0,1]) and
labelled `s=20` "n<3 — reported, not a verdict". Recorded because it is the mechanism working: the
verdict was produced by code holding the pre-registered rule, not by a person reading a table.

- **Evidence**: [MEASURED: results/ctm-small-T80-s{2,10,20}-{unc,b32}_seed*];
  [MEASURED: results/ctm-therm5-{preflight,20k-unc}_seed0] — the 20 000-clause budget gap, +6.43 pp.
- **Dissent**: none. DR-006 Decision 1 was raised by the theorist against their own headline and is
  now resolved in their favour; B1, also theirs, is falsified against them.
- **Falsifier (remaining)**: none outstanding for the budget claim itself. Open: what *does* govern
  `L_unc`, given it is not the admission threshold.
- **Supersedes**: DR-006 Decision 1 (resolved); DR-004 Decision 2's remaining scale falsifier
  (discharged at 20 000 clauses, +6.43 pp).

## DR-008 — C-2 verdict WITHHELD; argmax validation selection is inflating every arm
- **Date**: 2026-09-22
- **Round / phase**: P3, capacity ladder complete (1 seed)
- **Question**: does the capacity ladder settle C-2?

### Decision 1 — **the C-2 verdict is withheld.** The ladder is flat where the paper rises
[MEASURED: results/ctm-therm5-{preflight,40k,80k}_seed0]

| total clauses | plateau val (2nd half) | val-selected test | published (250 ep) |
|---|---|---|---|
| 20 000 | 64.20 | 65.10 | 64.5 |
| 40 000 | 64.55 | 65.44 | 66.8 |
| 80 000 | 63.91 | 63.61 | **69.1** |

The 20 000 point reproduces well (+0.6 above published). **The curve then does not rise**: the
published table gains **+4.6 points** over 20k→80k; we measure **−0.3 on the plateau**.
**This is not yet a finding.** Two live explanations and we cannot separate them at 1 seed:
(a) the published capacity rise does not reproduce; (b) **50 epochs is too few at the top end** — the
paper ran 250, and the epoch correction adopted in DR-002 was digitised at one clause count, so it
may not transfer to a 4× larger model.
The engineer's zero-cost control returns **ambiguous**: the 80k validation curve is flat-to-noisy
over the last ten epochs (63.0, 63.4, 63.0, 64.9, 63.9, 62.8, 64.9, 64.8, 63.3, 63.3), neither
clearly converged nor clearly still climbing.

### Decision 2 — **DR-003's adaptive-seeding rule fires: buy the second ladder seed**
The pre-registered condition was "add the second seed only if the 20k→80k gain lands within 2× the
seed band of the ±4 pp decision boundary". It landed at −0.3 against a predicted +4.6. Bought.

### Decision 3 — a protocol defect affecting **every arm in the programme**
`selected_epoch = argmax(val_acc)` over 50 epochs is a **biased estimator on a noisy curve**, and the
test accuracy at the winning epoch is an essentially uncorrelated draw from the plateau.
[MEASURED: same three records] plateau sd is **0.80–0.94 pp**; the binomial floor on 5 000 validation
images at p≈0.65 is **0.67 pp**; and argmax inflates the reported validation figure by a strikingly
consistent **+1.62 / +1.65 / +1.67 pp** across the three arms. The 80k arm's `selected_epoch = 16` is
a noise spike — epochs 44–48 sit at the same level.

This is not a leakage problem (selection is still validation-only) but it is a **precision** problem,
and it interacts badly with the 1.00 pp seed band: a decision rule with a 1 pp threshold is being fed
an estimator with ~1.65 pp of selection noise.

**Fix, and it costs no GPU time**: re-score from the **stored per-epoch curves** using a smoothed
rule — validation accuracy averaged over a centred window (3 or 5 epochs), then select. Every record
carries its full curve, so **every existing arm can be re-scored without a single re-run.** The
schema decision made in P0 is what makes this free.

### Decision 4 — `--test-curve-last` did not produce a paper-statistic field
The ladder jobs passed `--test-curve-last 25`, but the records carry no last-25 or peak test figure,
so the comparison against the papers' own estimator (DR-002 Decision 5, C-5) **cannot currently be
made**. Needs diagnosis before G3: either the flag did not fire, or the result is not being written.

- **Evidence**: as cited. All three arms carry D-TA, D-FIRE and D-ORDER, so the diagnostics did land.
- **Dissent**: none yet — this is the orchestrator's reading and the engineer and theorist have not
  responded to it.
- **Falsifier**: if seed 1 plus smoothed selection reproduces the published +4.6 rise, Decision 1(a)
  dies and the ladder simply needed better estimation. If the curve stays flat at 2 seeds **and** at
  a longer epoch budget, then the published capacity rise does not reproduce — which would be one of
  the most consequential findings available to this programme, and would need stating very carefully.
- **Supersedes**: nothing. DR-003 Decision 2's ladder design stands; only its seeding fires.
