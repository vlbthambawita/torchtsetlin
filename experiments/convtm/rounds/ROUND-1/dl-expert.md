# ROUND-1 — dl-expert position

## Answer in one sentence

The CNN side says the TM's ~33-point CIFAR-10 gap is **not** explained by binary computation — a
1-bit-weight, 1-bit-activation net reaches 93.75 % — so P3 should re-implement the approaches that
change the **readout and the receptive field** (counting pool, weighted/coalesced votes, patch
geometry, Booleanization) ahead of those that only add capacity, and the P5 mechanism inventory
should be mined in the order **counting pool → augmentation → dilated patches → knob annealing →
online rate control → soft-`T` distillation → skip literals for depth**.

---

## Claims

### On what the CNN side actually establishes

- `[FACT: BiPer, Vargas et al. CVPR 2024, Table 1]` the best known **fully binary** (1-bit weights
  *and* 1-bit activations) CIFAR-10 accuracy is **93.75 %**, on a CIFAR ResNet-18 with crop+flip,
  against a **94.8 %** full-precision baseline in the same paper. The binarisation gap is ≈ 1 point.
- `[FACT: arXiv:1511.00363 Table 1]` BinaryConnect (binary weights, real activations) scores
  **8.27 %** error against **10.64 %** for the identical full-precision architecture — binarising
  weights made it *better*, and both were trained with **no data augmentation at all**.
- `[FACT: arXiv:1603.05279]` on ImageNet AlexNet: FP 56.6 top-1, **BWN 56.8** (binary weights cost
  nothing), **XNOR-Net 44.2** (binarising activations too costs 12.4 points). The cost of binary
  computation lives in the **activations**, not the weights.
- `[HYPOTHESIS]` therefore at most ~1–5 points of the TM's gap to CNNs is attributable to 1-bit
  arithmetic. The rest is Booleanized **input**, **architecture**, and the **learning algorithm**.
  This decomposition is what P2 measures and what the report's spine should be.

### On what "the CNN baseline" even means

- `[FACT: arXiv:1708.04552 Table 1, 5 runs]` ResNet-18 on CIFAR-10: **4.72 ± 0.21 %** error with
  crop+flip, **10.63 ± 0.26 %** without. A **5.9-point** swing from augmentation alone.
  `[FACT: arXiv:1608.06993 Table 2]` ResNet-110: 6.41 with, **13.63** without.
- `[FACT: arXiv:1608.06993 Table 2 caption]` every published "no augmentation" CIFAR-10 number is
  *still regularised* — DenseNet's C10 column adds dropout 0.2 after every conv specifically for
  those runs; `[FACT: arXiv:1312.4400]` NiN without dropout scores **14.51 %** vs 10.41 % with, at
  no augmentation. **There is no widely cited CIFAR-10 CNN number with neither augmentation nor
  regularisation.**
- `[FACT: arXiv:1512.03385 §4.2]` He et al. use their 45 k/5 k split **only to fix the schedule**;
  Table 6's errors come from retraining on the full 50 k. Our protocol trains on 45 k and never
  touches the 5 k, so our ResNet-20 should land slightly *worse* than 8.75 %, and the record will
  say so rather than quietly training on 50 k to match.
- `[FACT: arXiv:1708.06519]` the widely quoted "VGG-16 CIFAR-10 = 93.66 %" is **VGG-19**, confirmed
  by reproducing its 20,040,522-parameter count. Do not cite it as VGG-16. The best peer-reviewed
  literal VGG-16 baseline is **6.75 %** error `[FACT: arXiv:1608.08710 Table 1]`.
- `[FACT: kuangliu/pytorch-cifar]` the most-quoted ResNet-18 CIFAR-10 number (93.02 %) **selects the
  checkpoint on the test set** and reports no seed variance. It is not usable as a baseline under
  PLAN §7.

### On sample efficiency, which is one of the programme's success axes

- `[FACT: arXiv:1610.02242 §4]` a supervised-only 4 000-label CIFAR-10 baseline ranges from
  **13.60 % to 35.56 %** error across papers, purely from augmentation strength and architecture —
  Laine & Aila say so explicitly about their own baseline. **A CNN sample-efficiency curve quoted
  from the literature is worthless for this programme.**
- `[FACT: survey of arXiv:1703.01780, 1610.02242, 1905.02249, 2001.07685, 1804.09170]` **no paper
  reports a supervised-only CIFAR-10 number at 5 000 or 10 000 images.** Two of our four curve
  points do not exist in the literature. We must measure all four ourselves, which P2 does.

### On the mechanism inventory (full argument in `MECHANISMS.md`)

- `[FACT: src/torchtsetlin/models/conv.py::_evaluate]` the CTM's patch aggregation is
  `matches.any(dim=1)` — **exactly a max over binary indicators**. The CTM has **max pooling and no
  average-pooling analogue at all**, while every high-accuracy CIFAR-10 architecture (NiN, All-CNN,
  ResNet, WRN, DenseNet) ends in a global **average** pool.
- `[HYPOTHESIS]` this is the largest identified structural gap: with `C` clauses and `P = 841`
  patches, the OR pool delivers **C bits** to the classifier where a counting pool would deliver up
  to **C · log₂(842) ≈ 9.7 C bits** — a ~10× information increase at **identical automata budget and
  identical interpretability**.
- **Supporting, non-admissible, and labelled as such**: a CNN with exactly the CTM's shape
  (1 conv layer, 2 000 filters, 4×4 patches, stride 1, therm4 Boolean input, global pool over the
  841 positions, linear vote) at the P5 screen scale (10 k images, 15 epochs, 1 seed,
  validation-selected) scores **38.9 %** with max-pooling and **51.0 %** with mean-pooling. Run
  today as a code-health check; the records are in the session scratchpad, **not** in `results/` or
  `screen/`, so this is **not** `[MEASURED]` and cannot justify a decision. It is a reason to run
  the TM experiment. **Request to the orchestrator**: authorise re-running these two as registered
  `screen/` records so the argument can be cited.
  **Stated confound**: on a CNN, mean-pooling also changes the optimisation (max gives gradient to
  one position). A TM has no gradient, so this supports the *information* half of the argument only.
- `[FACT: PLAN §3.1]` firing-rate calibration is structurally a BatchNorm analogue and took a
  2-layer stack from 15.4 % to **42.6 ± 0.7 %**. This is the existence proof that translating CNN
  mechanisms into TM mechanisms pays, and it is why I rank §5 (online rate control) above several
  things with larger predicted effects.
- `[FACT: PLAN §3.1]` credit propagation through clause inclusion fails for four **structural**
  reasons. I do not re-propose it. Two entries in `MECHANISMS.md` could be mistaken for it and each
  names its escape route explicitly:
  - **§6 skip literals** — layer-2 clauses draw from `concat(layer-1 bits, raw patch bits)`. Escapes
    all four because it is a **forward-path change only**: no `apply_feedback` call ever touches
    layer 1's `ta_state`. Layer 1 stays frozen, so identity cannot drift; nothing is propagated, so
    there is no signal needing a forgetting term, no Ib to blame, and no Ia:Ib ratio to set.
  - **§10 soft-`T`** — replaces the global constant `T` with a per-(example, class) target margin
    from a CNN teacher. Escapes all four trivially: it changes one scalar threshold and leaves the
    feedback rules, their ratio, the forgetting term and clause identity untouched.
- `[FACT: PLAN §3.1]` a one-layer CTM's vote is a linear threshold over OR-pooled clause bits.
  `[HYPOTHESIS]` therefore the capacity curve saturates below 70 % by ~8–16 k clauses, i.e.
  CIFAR-10 CTMs are **mechanism-limited, not capacity-limited** — adding features to a linear model
  over a fixed Boolean family has diminishing returns. **This must be tested before P5, not after.**
- `[FACT: PLAN §3.1]` random layer-1 clauses beat trained ones (37.6 % vs 15.4 %), and random +
  calibration reaches 42.6 %. `[HYPOTHESIS]` a CTM's productive regime looks less like a learned
  hierarchy and more like **a random Boolean feature expansion with a calibrated firing rate, read
  out by a boosting-like linear vote**. If so, the leverage is on feature *quality* (dilation,
  colour planes) and on the *readout* (counting pool) — which is exactly where my top ranks sit.
  I flag that this framing and my ranking are mutually reinforcing, which is a motivated-reasoning
  risk; the screens decide.

---

## What I need measured (ranked, with cost estimate)

Ordering is by expected value per GPU-hour. Items 1–3 are mine to run in P2; the rest are P3/P5
requests to the team.

| # | What | Why it is here | Owner | Cost |
|---|---|---|---|---|
| **1** | **`cnn-boolean` vs `cnn-small-noaug`** — identical architecture, identical recipe, only the input differs (12-plane therm4 vs float) | Isolates the cost of Booleanization. **This number does not exist in either literature.** Without it, "TMs are behind CNNs" cannot be decomposed and the report has no spine. | dl-expert | ~1.5 GPU-h (3 seeds each) |
| **2** | **`cnn-binary` / `cnn-binaryconnect` / `cnn-binary-boolean`** | Bounds what binary computation achieves on this task, and separates weight- from activation-binarisation (`[FACT: arXiv:1603.05279]` predicts the activations carry the cost). `cnn-binary-boolean` is the closest CNN there is to a TM's substrate. | dl-expert | ~4 GPU-h (200 ep, no AMP, 3 seeds) |
| **3** | **Sample-efficiency curves at 1 k / 5 k / 10 k / 50 k for every CNN arm** | Two of the four points do not exist in the literature; it is a stated success axis (PLAN §2); and it is the curve the TM arms are compared against. | dl-expert | ~5 GPU-h |
| **4** | **Capacity-scaling curve for `ctm-vanilla`** (≥ 4 clause counts to saturation) | **The gate.** If it still climbs > 1 point per doubling at 8 k clauses, the honest answer is "buy clauses" (PLAN §9.2.9) and most of P5 is premature. I want this **before** P5, not in P4. | research-engineer | ~2 GPU-h |
| **5** | **`ctm-count-vote` vs `ctm-vanilla` at 1 k / 4 k / 16 k clauses** | The largest structural gap I can identify. Predicted 5–12 points, and the *widening-at-low-budget* signature distinguishes an information effect from a capacity effect. | research-engineer | ~1.5 GPU-h (screens) + 6 GPU-h (full) |
| **6** | **`ctm-aug-{none,crop,flip,both}`** | Zero new code. Differentiated prediction: flip ≥ 2 × crop for a CTM, the opposite ordering for a CNN. Informative whichever way it lands. | research-engineer | ~0.5 GPU-h |
| **7** | **`ctm-dilated-d{1,2,3}`** at fixed 4×4 patch | Buys receptive field at **zero** automata cost; `unfold` takes `dilation` natively. Unexplored in the TM literature. | research-engineer | ~0.5 GPU-h |
| **8** | **`ctm-anneal-s` / `ctm-anneal-T`** | Cheapest experiment in the programme — setting an attribute between epochs. Nobody schedules anything in a TM, while no CIFAR-10 CNN result anywhere uses a constant learning rate. | research-engineer | ~0.3 GPU-h |
| **9** | **`cnn-ctmshape-binary` vs a matched `ctm-vanilla`** | The **only** clean isolation of the learning algorithm in the whole programme: same architecture, same Boolean input, same receptive field — SGD vs Type I/II feedback. | dl-expert + research-engineer | ~1 GPU-h |
| **10** | **`ctm-rate-online`** (single layer, rate control every epoch) | Best-validated translation we have, and the principled replacement for the `max_included_literals` budget that does not bind for conv models under batched feedback. | research-engineer | ~0.8 GPU-h |
| **11** | **`ctm-soft-T`** with a `cnn-small` teacher, at 50 k **and** 1 k | Injects gradient-derived knowledge with no change to the feedback rules and no interpretability cost. Predicted to help *more* at 1 k. | dl-expert (teacher) + research-engineer | ~1 GPU-h once the teacher exists |
| **12** | **`mctm-skip`** vs its own frozen layer 1 | The only proposal that structurally removes the measured 34.4 → 15.4 collapse. A clean negative here would justify dropping depth from the programme. | research-engineer | ~0.8 GPU-h; **feasibility unknown — needs a literal-concat layer 2; may be a `LIBRARY_GAPS.md` entry** |

### P3 shortlist input (for `DR-001`)

From the CNN side, ordered by what our evidence says is worth reproducing. I own none of these; this
is a vote, and the tm-theorist's paper should outrank mine on TM-internal questions.

**Promote**: `ctm-vanilla` (T0 is mandatory and gates everything), `ctm-coalesced` and
`ctm-weighted` (they change the **readout**, which is where §3 says the bottleneck is; weighted votes
are a partial step toward the counting pool and should be measured before we claim the counting pool
is novel), `ctm-boolean-*` (P2's `cnn-boolean` result is only interpretable against these),
`ctm-clausesize` (it is the control for the known `max_included_literals` gap and for §5/§12),
`mctm-calibrated` (our own best, must be re-run under the P7 protocol).

**Demote**: `ctm-hypervector` and `ctm-sparse`/`ctm-absorbing` — on my reading they target efficiency
rather than the accuracy bottleneck this programme is chasing; cheap to reinstate if P4 says
capacity is the binding constraint.

**Reprioritise**: `tm-composite` should be run **late and only at matched automata budget**
(PLAN §7.3). It is the 82.8 % headline, so it will dominate any table it appears in; reproducing it
early risks the programme's narrative becoming "ensembles work", which is already known
`[FACT: PLAN §3.3]` and is not a mechanism.

**Add**: `ctm-distill` should be reframed from "try distillation" to the specific mechanism in
`MECHANISMS.md` §10 (per-example target margins), because that version requires no change to the
feedback rules and costs nothing in interpretability.

---

## What would change my mind

1. **The capacity-scaling curve is still climbing at 8 k clauses.** Then CIFAR-10 CTMs are
   capacity-limited, my entire ranking is premature, and the correct report is "buy clauses"
   (PLAN §9.2.9). I want this measured before P5 for exactly that reason.
2. **`cnn-ctmshape-binary` lands near 60 %**, i.e. near the published CTM best. Then the CTM's
   learning algorithm is *not* the bottleneck, the architecture is, and §§9–10 of `MECHANISMS.md`
   (optimiser, soft targets) should be dropped from the pool entirely.
3. **`cnn-boolean` lands within 1 point of `cnn-small-noaug`.** Then Booleanization is nearly free,
   §11 dies, and the whole gap is architecture + algorithm — which would *raise* the counting pool
   and lower the Booleanization arms in the P3 shortlist.
4. **The counting pool gives a *uniform* gain across clause budgets** rather than a widening one at
   low budgets. Then it is a capacity effect in disguise, it should not be ranked above "more
   clauses", and I withdraw the information argument.
5. **`ctm-weighted` already captures most of the counting pool's gain.** Weighted votes are a
   partial relaxation of the same bottleneck; if P3 shows them closing most of the gap, the counting
   pool is an increment on existing work rather than a new mechanism, and must be reported that way.
6. **Flip augmentation does not beat crop augmentation for a CTM.** My account of the CTM's symmetry
   group (translation-invariant by the OR pool, not flip-invariant) would then be wrong, and the
   invariance reasoning in `MECHANISMS.md` §3, §4 and §7 all needs revisiting.
7. **Anyone shows that `mctm-skip` does touch layer 1's `ta_state`.** Then it is credit propagation
   in disguise, PLAN §3.1 has already killed it, and I withdraw it.

---

## Status of my P1 deliverables

- `LITERATURE_CNN.md` — complete. 29-row CIFAR-10 reference table with exact configurations,
  separate no-augmentation and augmentation columns, the binarised-network section, the
  sample-efficiency section, six declared deviations of our arms from the papers, and a list of the
  numbers the literature does **not** report with the P2 arm that supplies each.
- `MECHANISMS.md` — complete. 15 mechanisms, each with (a)–(e), a falsifiable prediction and its
  cheapest falsifier, plus the ranked P5 pool and the four results that would invalidate the
  document.
- `code/cnn/` — implemented and smoke-tested: `models.py`, `binary.py`, `arms_cnn.py` (24 arms),
  `train.py`. Parameter counts reproduce the literature exactly (ResNet-20 269,722; ResNet-56
  853,018; ResNet-110 1,727,962; ResNet-18 11,173,962; BNN-VGG 14,029,726). All 24 arms complete a
  2-epoch smoke test on a 2 000-image subset and produce records that pass `record.validate()`.
  **No full training has been launched; P2 starts after Gate G1.**
- **`src/` is untouched** (C1). Nothing in `code/cnn/` imports from `src/torchtsetlin`.

---

## Crossfire response

I verified the two facts that decide C-1 and C-2 against primary sources rather than accepting them
at third hand: `diagnostics.match_count` in `results/ctm-small_seed{0..4}.json`, and Table IV of
`source_documents/papers/02_image_classification/2024_Optimized_Toolbox_..._arXiv2406.00704.pdf`
read out of the local PDF. **Both go against me.** What follows concedes where the measurement says
to, and states precisely what survives.

### Accept / reject

**Accept without qualification**: section A in full (T0 → ~69 %, T2 → 75.4 %); C-4.1 and C-4.2
(clause-count convention, `ctm-vanilla` = class-owned); C-5 (log the paper's own statistic
alongside); C-6 (a control, not a candidate); all of section D. I accept the crossfire's reading of
my CNN-side proxy as weakened, and I downgrade it myself below — further than asked.

**Reject**: nothing. **Withdraw**: my C-2 prediction, entirely; my C-1 standalone prediction, in the
form I stated it; my rank-2 placement of augmentation.

**Add, on C-5**: Table IV's spread is labelled `x̄ ± s²` — that is **variance over the last 25
epochs**, not a standard deviation and not a seed spread. The 5×5 colour thermometer's 0.09 at
64 000 clauses is sd ≈ 0.30 pp *within one run*. Our 1.00 pp band is across seeds. These are
different quantities and must never be placed in the same column. The theorist's "~15× variance" is
correct as variance; as standard deviation it is 3.9×.

---

### C-2 — I was wrong. The capacity curve has not saturated.

`[FACT: arXiv:2406.00704 Table IV, read from the local PDF]` 5×5 Colour Thermometers, accuracy by
clause count:

| clauses | 2 000 | 4 000 | 8 000 | 16 000 | 32 000 | 64 000 |
|---|---|---|---|---|---|---|
| 5×5 Colour Thermometer | 64.5 | 66.8 | 69.1 | 71.7 | 73.7 | **75.4** |
| Δ per doubling | — | +2.3 | +2.3 | +2.6 | +2.0 | **+1.7** |
| HOG (32×32 window) | 64.2 | 65.7 | 66.9 | 67.3 | 67.5 | **67.5** |
| Δ per doubling | — | +1.5 | +1.2 | +0.4 | +0.2 | **+0.0** |

I predicted saturation **below 70 % by 8–16 k clauses**. At 8 k it is 69.1 and still gaining +2.6 per
doubling; at 16 k it is 71.7. My own pre-registered falsifier was "still climbing by > 1 point per
doubling at 8 k clauses". It is climbing at +2.6. **The prediction is dead and I withdraw it.** The
theorist is right on the fact, and right that HOG's flat 67.5 versus the thermometer's still-rising
75.4 is evidence that the binding ceiling is the **encoding's**, not the clause mechanism's.

**What survives, from the same table, and it changes the ranking rather than rescuing my claim.**
The crossfire quotes only the specialist rows. The composite row saturates:

| clauses per specialist | 2 000 | 4 000 | 8 000 | 16 000 | 32 000 | 64 000 |
|---|---|---|---|---|---|---|
| TM Composite (22 specialists) | 79.5 | 80.6 | 81.5 | 82.2 | 82.7 | **82.8** |
| Δ per doubling | — | +1.1 | +0.9 | +0.7 | +0.5 | **+0.1** |

`[FACT: arXiv:2406.00704 §IV, verbatim]` *"the accuracy of the TM Composite increases with the
number of clauses employed by a TM Specialist up to 32 000 clauses, with minimal increase after."*

So the **system** saturates by 32 000 even though its **parts** do not. And the matched-budget
comparison the plan demands (§7.3) is devastating for capacity:

* one specialist at **64 000** clauses → **75.4 %**
* 22 specialists at **2 000** clauses each = **44 000 total clauses** → **79.5 %**

**+4.1 points at 0.69× the automata budget**, and at roughly equal wall-clock
(`[HYPOTHESIS]`, my arithmetic on the measured price list: ≈19.5 GPU-h/seed for the 64 k specialist
at 250 epochs versus ≈18 GPU-h/seed for all 22 small ones).

Price per point at the frontier, same arithmetic: the 32 k → 64 k step buys **+1.7 points for
≈29 GPU-h at 3 seeds — about 17 GPU-h per point.** Anything delivering +2 points for 2 GPU-h is an
order of magnitude better value, which is the metric I was asked to rank on.

**Revised position.** Neither "mechanism-limited" (mine, falsified) nor purely "the encoding's
information ceiling" (the theorist's, supported but incomplete). The measurement supports:
**at fixed automata budget, encoding diversity dominates clause count; clause count still buys
accuracy, at a rapidly worsening price; and the composite — the system people actually report —
has already saturated.** My mechanism work is therefore not justified as the route to higher
accuracy. It is justified, if at all, as **efficiency and interpretability** work, and the report
should say so plainly. That is a demotion of my own contribution and I accept it.

**This also sets an exact T0 target**, which the round needed: `[FACT: Table IV]` 5×5 colour
thermometer at **8 000 clauses = 69.1 ± 0.63 (variance, last-25-epoch mean), T = 3000, s = 5.0,
weighted, 250 epochs** — same encoding, same clause count, from the paper's own table. Far better
than "~69 %" inferred.

---

### C-1 — paired only, and my information argument was wrong by a factor of five

I computed the information gain from the measured histogram instead of asserting it.
`[MEASURED: results/ctm-small_seed{0..4}.json → diagnostics.match_count]`, 5-seed means, exact
logical match, validation, 841 positions: `fire_frac` 0.0695 · median given firing **2** ·
q25 **1** · q75 **6** · p95 **24.6** · **max 672** · mean given firing 6.14 ·
P(≥5 | fires) 0.3028 · P(=1 | fires) 0.3555.

Reconstructing the conditional distribution pinned by those measured quantiles:

| | bits per clause reaching the vote |
|---|---|
| OR pool (indicator) | **0.364** |
| counting pool | **0.625** |
| **ratio** | **1.72×** (extra: **0.26 bits/clause**) |

Robust to the tail assumption (1.70–1.73× as the >24 mass is spread over 50–400 values).

**My ROUND-1 claim of ~9.7× was `log₂(P+1)` — the maximum under a *uniform* distribution over
counts.** The measured distribution is nothing like uniform: 93 % of pairs are zero and 35.5 % of
the rest are exactly one. I quoted a ceiling as if it were an estimate. That was a real error, it is
the kind of error the motivated-reasoning flag was for, and it is the single thing I most want
corrected in the record.

**Answer to the question as posed: (b) — admissible only paired with density control**, and I go
further than "paired":

* **Standalone prediction revised from +5–12 points to +0 to +1.5 points.** 0.26 extra bits per
  clause, of which a *linear* vote can use only a fraction. That is **at or below the 1.00 pp seed
  band**, so a standalone counting arm is **not screenable at 3 seeds** and should not be screened.
  By the programme's own rules it cannot be promoted on its own evidence.
* **They are complements, not two mechanisms to be ANDed.** Density control is the intervention;
  counting is the readout that lets you cash it in. Without counting, raising density drives the OR
  toward always-1 and destroys it. Without density, counting has almost nothing to count. Each is
  near-useless alone; the claim is an **interaction**, so it must be tested with a factorial.
* **The readout must be capped, not raw.** `max` given firing is **672 of 841**. A raw count lets one
  clause contribute 672 to a vote whose `T` is 3000 — the vote scale is destroyed and `T` stops
  meaning anything. From the measured distribution, `min(count, 8)` retains **96 %** of the count's
  entropy (cap 4 → 82 %, cap 2 → 74 %). **Design spec: `f(c) = min(c, 8)`**, derived from the
  measurement rather than chosen.

**Arm structure — a 2×2 factorial, because only a factorial can show an interaction:**

| | OR readout | capped-count readout `min(c,8)` |
|---|---|---|
| **standard density** | `ctm-small` (base, exists) | `ctm-count` (counting-only control) |
| **rate-controlled density** | `ctm-density` (density-only control) | `ctm-count-density` (the candidate) |

The density-only control the orchestrator asked for is the bottom-left cell. Both single-factor
cells are controls, not candidates. The quantity that promotes or kills the branch is the
**interaction term**, `(count-density − density) − (count − base)`.

**Priced on the measured model** (2 000 clauses, 4×4, `s/epoch ≈ (clauses/1000) × 3.7` on 45 k):
* screen (10 k images, 15 epochs, 3 seeds, 4 cells): ≈ 1.6 s/epoch → **≈ 0.02 GPU-h.** Free.
* full (8 000 clauses, 60 epochs, 3 seeds, 4 cells): 29.6 s/epoch → 0.49 GPU-h/seed → **5.9 GPU-h.**

**Pre-committed drop rule**: if the full-scale interaction term is below the 1.00 pp seed band, the
entire counting-pool branch is dropped and reported as a measured negative. I will not ask for a
larger budget to rescue it.

#### How much weight should the CNN proxy carry? Less than I gave it — it is a bound, not evidence.

`[MEASURED: screen/cnn-ctmshape-max_seed0.json vs -sum]` 38.86 % vs 51.02 %, +12.2 points. Side by
side with the TM:

| | fire_frac | median \|M\| given firing | P(≥5 \| fires) | mean over all pairs |
|---|---|---|---|---|
| `ctm-small` (TM) | 0.070 | **2** | 0.303 | 0.42 |
| `cnn-ctmshape-max` (CNN) | 0.0096 | **40** | 0.883 | 0.83 |

The crossfire's characterisation is right: a rare broad detector versus a common narrow one. The
correct role of the 12.2 points is therefore **the ceiling of the paired arm if density control
succeeds in broadening the distribution — not evidence that the standalone readout change is worth
anything.** It belongs in the report as "here is what the readout is worth once counts are
informative", attached to the paired arm, and nowhere else. Two further caveats I am adding myself:
the CNN reached that distribution because gradient descent could *choose* it freely, which a TM
cannot; and the optimisation confound (max routes gradient to one position) is unquantified and may
be a large share of the 12.2.

One thing genuinely worth keeping: total match mass per (image, unit) is **0.83 for the CNN versus
0.42 for the TM — within 2×**. The two substrates spend a similar budget of matches and allocate it
oppositely. `[HYPOTHESIS]` allocation, not amount, is the lever — which is exactly what density
control plus a capped count would change, and it is the cleanest statement of the paired hypothesis.

---

### C-3 — augmentation: I was wrong to rank it 2, and the published evidence is richer than either paper used

Reading the augmented and non-augmented rows of Table IV as pairs
`[FACT: arXiv:2406.00704 Table IV]`:

| specialist | Δ at 2 000 clauses | Δ at 64 000 clauses |
|---|---|---|
| Canny edge 10×10 | **+1.9** | **+1.8** |
| Adaptive Gaussian 5×5 | −0.1 | +0.1 |
| Adaptive mean 10×10 | **+1.3** | **+1.8** |
| Otsu 10×10 | **+1.6** | +0.7 |
| HOG 32×32 | +0.6 | **+1.0** |
| Colour thermometer 3×3 / 4×4 / 5×5 | +0.5 / +0.8 / +0.6 | **−1.6 / −1.5 / −1.3** |
| Adaptive colour thermometer 3×3 / 4×4 / 5×5 | +0.5 / +1.3 / +0.9 | −0.9 / −0.1 / −0.4 |

**At 2 000 clauses augmentation helps in 10 of 11 rows** (worst case −0.1). **At 64 000 it splits
cleanly by encoding family**: every threshold/edge/HOG encoding gains; every thermometer encoding
loses. The theorist's 1.3-point figure is real, and it is the *worst* case in the table, on the
single best specialist.

**Two confounds, and neither is the data budget.** First, `[FACT: arXiv:2406.00704 §IV, verbatim]`
*"we used the same hyperparameters for the non-augmented and augmented data"* — `T` and `s` were
found by Optuna at **2 000 clauses on non-augmented data** and reused at every clause count and for
every augmented arm. `T` is a vote-margin threshold; doubling the data and 32×-ing the clauses
without retuning `T` is a sufficient explanation for both the loss and the variance blow-up
(0.09 → 1.35 variance, i.e. sd 0.30 → 1.16). Second, and the one I think matters more: their
augmentation is **static dataset expansion** (50 k → 100 k, flip pre-baked), which is *not* what a
CNN does. A CNN applies a **fresh random transform per example per epoch** at an unchanged number of
steps. Static expansion changes the data budget; stochastic augmentation changes the *distribution*
at fixed budget. The TM literature tested the weaker intervention.

**Resolution**: per-epoch stochastic flip (which `data._augment_batch` already does), at **matched
images-per-epoch and matched epochs**, with `T` and `s` **retuned for the augmented arm** — and run
at **two clause budgets**, not one.

**Does it stay at rank 2? No — I move it to rank 6, and flag a methodological trap.** The measured
effect is **clause-budget-dependent, and its sign flips**. Our screens run at small clause budgets,
where Table IV says augmentation is positive in 10/11 cases. **A screen of this mechanism will
produce a confident false positive that reverses at full scale.** PLAN §8 P5.3 anticipates screens
having false negatives; this is the opposite, and it is the clearest instance in the programme of a
mechanism the screening protocol will actively mislead us about. It must be exempted from
promote-on-screen and tested at 2 k **and** 32 k, or not run.

---

### C-4, C-5, C-6 — short answers

**C-4.** **8 000 discharges T0.** T0's stated purpose is that the harness is trusted, not that a
headline is reproduced. Table IV now gives an exact same-encoding, same-clause-count reference
(69.1, 5×5 colour thermometer, T = 3000, s = 5.0, weighted, 250 epochs), which is a *stronger* test
of the harness than an 80 000-clause arm compared against a number whose configuration we are
guessing. I support the engineer's recommendation: run the 8 k reading (3.9 GPU-h), and spend the
31.6 GPU-h saved on the capacity × encoding grid, which answers a scientific question instead of a
bookkeeping one. Re-open the 80 k arm only if the envelope has slack at the end.

**C-5.** Accept. Log the last-25-epoch mean and the peak alongside the validation-selected number
for every `existing`-family arm. Note that Table IV's own statistic *is* the last-25-epoch mean, so
for the Toolbox arms it is a like-for-like comparison and the protocol penalty is directly
measurable rather than assumed. My `cnn-*` arms need no such column, and the report must never place
a validation-selected TM number and a last-25-mean TM number in the same column.

**C-6.** Confirmed: a control, not a candidate. **Recording that it does not bind is sufficient**,
and paying 10× for sequential feedback to reproduce a **+0.06** effect — 1/16 of the seed band — is
not defensible. But since clause-size budget 32 is part of two SOTA recipes, the composite
reproduction needs *something* that binds: use the density controller from C-1's factorial and state
the substitution explicitly in `ARMS.md` and the record's `notes`. One mechanism, two jobs.

---

### Final ranked P5 mechanism list, priced

Costs use the measured model `s/epoch ≈ (clauses/1000) × {3.7 @4×4, 7.3 @8×8, 10.1 @10×10}` on 45 k
images; the 5×5 figure (4.39) is `[HYPOTHESIS]`, my linear interpolation on patch cells. Seed band
1.00 pp for 3-seed means.

| Rank | Mechanism | Expected | Screen | Full (3 seeds) | Why here now |
|---|---|---|---|---|---|
| **1** | **`cnn-boolean` on the real encodings** (HOG bits, 5×5 colour-thermometer bits) | decides the programme's framing | — | **~3 GPU-h** | The theorist's own named C-2 falsifier and my arm. HOG saturates at 67.5 for a TM; if a CNN on the *same bits* reaches ~80 %, the ceiling is the TM's. If it reaches ~68 %, it is the encoding's and mechanism work is efficiency work. Nothing else in the programme buys this much framing for 3 hours. |
| **2** | **Encoding: colour-prototype planes** (§11) | 1–4 pts | 0.5 h | ~3 GPU-h | The measured decomposition gives Booleanization a **16.7-point spread at fixed 2 000 clauses** — the largest lever anyone has measured. Gated on rank 1. |
| **3** | **Capped count × density control, 2×2** (§3 + §5, merged) | interaction, or a clean negative | **0.02 h** | 5.9 GPU-h | C-1. Drop rule pre-committed. |
| **4** | **Anneal / retune `s` and `T` with clause count** (§13) | 1–3 pts + variance | 0.3 h | ~1.5 GPU-h | **Promoted.** Table III shows T and s reused from a 2 000-clause search across 2 k–64 k, and the augmented variance blow-up is consistent with mis-set `T`. This may be worth more than I thought and it is nearly free. |
| **5** | **Dilated / multi-scale patches** (§4) | 2–5 pts | 0.4 h | ~2 GPU-h | **Better supported than in ROUND-1**: the paper names the convolution window one of its three most important hyperparameters, and its best windows range 3×3 to 32×32. Dilation buys window size at zero automata cost. |
| **6** | **Augmentation, stochastic, retuned `T`, two budgets** (§7) | sign flips with budget | **exempt from screen promotion** | ~2 GPU-h | Demoted from 2. See C-3. |
| **7** | Soft-`T` distillation (§10) | 2–4 pts, more at 1 k | 0.5 h | ~1.5 GPU-h | Unchanged; teacher comes free from P2. |
| **8** | Skip literals for depth (§6) | 2–5 pts or a clean negative | 0.4 h | ~1.5 GPU-h | Feasible with no subclass per section D. |
| — | ~~Patch-draw ablation (§2)~~ | ≤1 pt | — | — | **Cut.** Below the 1.00 pp band by my own ROUND-1 estimate. |

### Final P3 shortlist vote, priced against ~120 GPU-h

| | Arm | Cost (3 seeds) | Rationale |
|---|---|---|---|
| 1 | `ctm-vanilla` at **8 000** clauses | **3.9 h** | T0, against Table IV's own 69.1 at 8 000. Run first. |
| 2 | **Capacity × encoding grid** — {2 k, 8 k, 32 k} × {5×5 colour thermometer, HOG} | **~15 h** | Settles C-2 with our own data. Highest-information block in P3. |
| 3 | `cnn-boolean` on HOG + thermometer bits | **~3 h** | The C-2 falsifier. Mine, on the 3080, in parallel. |
| 4 | `ctm-dropclause` at 8 k | **~4 h** | +5.8 in the measured decomposition; a regulariser (§12). |
| 5 | `ctm-coalesced` + `ctm-weighted` at matched 8 k | **~3 h** | Readout changes. **Weighted clauses are a partial relaxation of the same bottleneck as the counting pool — these must be measured before the counting pool can be called novel.** |
| 6 | `tm-composite` at **2 000** clauses/specialist, with `α_t` fixed on validation (O-1) | **~6 h** | 79.5 published; beats the 64 k single model by 4.1 points at 0.69× the automata. The transductive-`α_t` control is mandatory. |
| 7 | `mctm-calibrated` re-run under protocol | **~2 h** | Our own prior best. |
| 8 | `ctm-clausesize` as a non-binding control | **~1 h** | C-6; record that it does not bind. |
| | **Total** | **≈ 38 GPU-h** | leaves ~80 h for P5/P6 |

**What I cut first, in order:**
1. **`ctm-vanilla` at 80 000 clauses — 31.6 GPU-h.** A quarter of the envelope to reproduce one
   number under a clause-convention reading we are inferring. The 8 k arm discharges T0 against a
   better reference.
2. **My own patch-draw ablation (§2)** — my own ROUND-1 estimate put it under the seed band.
3. `ctm-hypervector`, `ctm-sparse` / `ctm-absorbing` — efficiency arms, and C-2 now says the accuracy
   ceiling is elsewhere. Reinstate only if P4 shows capacity is the binding constraint after all.
4. **The standalone counting-pool arm** — by C-1, it cannot clear the seed band and should never be
   screened alone.

### What would still change my mind

* **`cnn-boolean` on HOG bits reaching ~80 %.** Then the encoding is *not* the ceiling, the TM's
  extraction is, mechanism work returns to being accuracy work, and ranks 3–8 all rise.
* **The 2×2 interaction term clearing 1.00 pp at full scale.** Then density-plus-counting is a real
  mechanism and my CNN proxy was measuring something after all.
* **`ctm-weighted` capturing most of the counting pool's effect.** Then the counting pool is an
  increment on 2019 work, not a new mechanism, and must be reported as such.

### On the `--screen` flag

Noted, and the reason it is worth generalising: the protocol failures this programme is guarding
against are not failures of intent. Where a rule can be made unstateable in the harness rather than
merely written down, it should be. `record.validate()` refusing a record whose `selected_epoch`
disagrees with `argmax(val_acc)` is the same pattern. C-3 is the case that most needs it next — a
mechanism whose screen result reverses at full scale should be structurally unpromotable, not
merely footnoted.
