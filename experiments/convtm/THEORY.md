# THEORY.md — what a convolutional Tsetlin machine can and cannot represent

**Owner**: `tm-theorist` · **Phase**: P1 · **Date**: 2026-09-20

Everything here is either derived from a cited source, derived from the library source (file:line),
or tagged `[HYPOTHESIS]` with the experiment that would falsify it. Nothing here chooses a method
(C3); §6 and §7 exist to turn representational claims into things P4 can measure.

---

## 1. The CTM, formally

### 1.1 Objects

Let the Booleanized image be `x ∈ {0,1}^{Z×H×W}` — `Z` bit planes (colour channels × thermometer
levels, edge maps, …), `H×W` spatial. Fix a window `k_h × k_w` and stride `s_h × s_w`. The patch
grid is

    P_y = ⌊(H − k_h)/s_h⌋ + 1 ,  P_x = ⌊(W − k_w)/s_w⌋ + 1 ,  P = P_y · P_x

`[FACT: src/torchtsetlin/models/conv.py:66-71]`. For each patch index `b ∈ B`, `|B| = P`, the patch
feature vector is

    φ_b(x) = ( pixels in window b  ‖  therm(y_b)  ‖  therm(x_b) ) ∈ {0,1}^F ,
    F = Z·k_h·k_w + (P_y − 1) + (P_x − 1)

where `therm(y_b)` is the thermometer code `[y_b > 0], …, [y_b > P_y−2]`
`[FACT: conv.py:73-80, 113-119; Granmo et al. 2019 §3; Book Ch. 4 §4.5]`. Position bits are present
iff `position_encoding=True`.

Literals double the vector: `ℓ(φ) = (φ, ¬φ) ∈ {0,1}^{2F}` `[FACT: functional.py:44-49]`.

A clause `j` owns `2F` Tsetlin automata with integer states `a_{j,i} ∈ [0, 2N−1]`; literal `i` is
**included** iff `a_{j,i} ≥ N` `[FACT: functional.py docstring; base.py:404-407]`. Write
`I_j = { i : a_{j,i} ≥ N }`. Per-patch clause value:

    c_j^b(x) = ⋀_{i ∈ I_j} ℓ_i(φ_b(x))          (empty conjunction: 1 while learning, 0 when predicting)

`[FACT: functional.py:66-90; Granmo et al. 2019 §2.2; Book Ch. 4 §4.3]`. The library computes this
as `violations = (1 − ℓ) · includeᵀ` and tests `violations == 0`
`[FACT: functional.py:52-63]` — i.e. *a clause matches iff no included literal is false*.

### 1.2 The patch OR

    C_j(x) = ⋁_{b ∈ B} c_j^b(x)

`[FACT: Granmo et al. 2019 §3, "the output of a convolution clause is obtained simply by ORing";
Book Ch. 4 §4.6; Abeyrathna et al. 2021 Eq. 4; conv.py:146 `matches.any(dim=1)`]`.

### 1.3 The class vote

Each clause carries a fixed polarity `p_j ∈ {+1,−1}` and an integer weight `w_j ≥ 0` (class-owned
model), or a signed integer weight `w_{j,i}` per output (coalesced model):

    v_i(x) = Σ_{j ∈ class i} p_j w_j C_j(x)            [FACT: classifier.py:99-112]
    v_i(x) = Σ_{j=1..C}      w_{j,i} C_j(x)            [FACT: coalesced.py]

clamped to `[−T, T]` during learning `[FACT: classifier.py:111]`, and

    ŷ(x) = argmax_i v_i(x)                              [FACT: classifier.py:123-125]

---

## 2. Proposition 1 — a convolutional TM layer is a linear model over existential conjunctive
   patch predicates

Define the **presence map**

    Φ : {0,1}^{Z×H×W} → {0,1}^C ,   Φ(x) = ( C_1(x), …, C_C(x) ) .

Then, exactly and with no approximation,

    ŷ(x) = argmax_i ⟨ w_i , Φ(x) ⟩ .

Each coordinate of `Φ` is an **existential conjunctive patch predicate**:
`C_j(x) = ∃ b ∈ B : φ_b(x) ⊨ π_j`, where `π_j` is the conjunction over `I_j`.

So a CTM is a *linear (multiclass) classifier over a learned Boolean dictionary of "does pattern π
occur anywhere (in region R)?" questions.* The learning algorithm learns the dictionary (the `I_j`)
and the linear weights (`w`) simultaneously, but the **composition rule between them is fixed and
linear**. This one sentence generates every limitation in §3.

Two immediate consequences:

**(a) Feature map, not feature hierarchy.** `Φ` is shallow by construction. The only non-linearity
is the AND inside a clause and the OR across patches; both are *inside* a coordinate of `Φ`, never
*between* coordinates.

**(b) Everything the vote layer can do, a perceptron over `Φ` can do — and nothing more.** For
binary decisions this is literally a halfspace. For `K` classes the argmax of `K` linear forms gives
a piecewise-linear (power-diagram) partition of `{0,1}^C`, which is strictly more than one
halfspace, but is still determined by pairwise halfspaces `⟨w_i − w_{i'}, Φ(x)⟩ ≷ 0`.

---

## 3. What one layer can and cannot represent

Throughout, "presences" means the values `(C_1(x),…,C_C(x))` for a *fixed* clause set.

### 3.1 Reachable

* **"A present AND B absent."** Give the clause detecting A weight `+1` and the clause detecting B
  weight `−1` (a negative-polarity clause, or a negative coalesced weight). The class score is
  `1_A − 1_B`, which separates. ✅ `[FACT: this follows from §2; matches PLAN.md §3.1]`
* **Any threshold over presences**: "at least 3 of {A,…,E} occur" is `Σ 1_• ≥ 3`. ✅
* **Any conjunction or disjunction of presences**, including position-tagged ones: "A in the top
  half AND B in the bottom-right" is `1_{A∈R1} + 1_{B∈R2} ≥ 2`. ✅ — because each clause carries its
  own absolute rectangular region through the position thermometers
  `[FACT: conv.py:203-228 `clause_region`; Book Ch. 4 §4.5]`.
* **Translation invariance** (with `position_encoding=False`): `C_j` is invariant to any translation
  that keeps the pattern inside the image. ✅

### 3.2 Proposition 2 — the three things a clause forgets

`C_j(x)` depends on `x` **only through whether the set `M_j(x) = { b : c_j^b(x)=1 }` is empty**.
Therefore the layer discards, irrecoverably:

1. **`|M_j(x)|` — the count.** One occurrence and four hundred occurrences give the identical
   clause output. The vote is a weighted sum of indicators, never of counts.
2. **The identity of the matching positions**, beyond whatever the clause's own position literals
   already constrain. A clause reports "somewhere in `R_j`", not "at `b`".
3. **Any joint fact about two different patches.** `C_j` is evaluated one patch at a time; there is
   no clause whose truth depends on two patches simultaneously.

### 3.3 Corollary 1 — XOR of presences is unreachable

Let `f(x) = 1_A(x) ⊕ 1_B(x)` for two patterns A, B that do not fit in one window. Suppose a CTM
computes `f` with two classes. By §2(b) the decision is a halfspace in `Φ`. Restricted to the four
achievable presence vectors `(0,0),(0,1),(1,0),(1,1)` — all four are achievable, by construction of
the input — `f` is XOR, which is not linearly separable. Contradiction. **No clause budget removes
this**, because enlarging the clause set only adds coordinates that are themselves existential
conjunctions over single patches, and any such coordinate is a function of `(1_A, 1_B)` only through
monotone existential facts; the XOR of two independent existentials is not among them.
`[FACT: derivation]` `[MEASURED: experiments/mctm — an XOR-of-presences task is capped at 75% for
one layer, and reaches 87.9% with an unsupervised patch-autoencoder layer 1; PLAN.md §3.1]`

**Escape hatch, and the only one:** enlarge the window until A and B co-occur inside it, so that
"A and B" becomes a single conjunction `π_j` over one patch. That is why receptive field (§5) is the
central quantity.

### 3.4 Corollary 2 — relative spatial relations are unreachable, at any clause budget

Consider `g(x) = ∃p : (A at p) ∧ (B at p + d)` — "B sits exactly `d` below A, anywhere in the
image", the archetypal part-relation a CNN gets from a second layer.

Assume position encoding is fine enough that a clause can be tied to a single grid cell, so the
dictionary can contain `1_{A@p}` and `1_{B@q}` for every `p, q`. Then

    g = ⋁_p ( 1_{A@p} ∧ 1_{B@p+d} ) ,

a DNF of conjunctions. For `P ≥ 2` this is **not** a linear threshold over `{1_{A@p}, 1_{B@q}}`:
the linear form `Σ_p (1_{A@p} + 1_{B@p+d}) ≥ 2` also fires on `A@p₁, B@p₂+d` with `p₁ ≠ p₂`, and no
choice of weights separates the "aligned" from the "crossed" configurations, because the linear
functional sees only the two marginal sums. Adding clauses does not help: every added coordinate is
again a *single-patch* existential. **The only representable route is, again, to put A and B in one
window.** `[FACT: derivation]`

**Measured corroboration**: `[MEASURED: experiments/mctm]` a proximity task on which one layer is at
chance *by construction* reaches 99.9% with an unsupervised patch-autoencoder layer 1 (PLAN.md §3.1).

### 3.5 Corollary 3 — the layer is blind to object count

By Proposition 2(1), for any `x` and any transformation `τ` that preserves, for every clause, the
emptiness of `M_j`, we have `ŷ(τx) = ŷ(x)`. Duplicating an object, or deleting all but one copy, is
such a transformation whenever the clause set is unchanged. **Prediction** `[HYPOTHESIS]`: a trained
CTM's accuracy and class sums are invariant to object multiplicity, while a CNN's are not.
**Falsifier / measurement**: build a CIFAR-10-derived probe (tile the same 16×16 crop 1×, 2×, 4×
into a 32×32 canvas) and compare the class-sum change for the CTM against the logit change for
`cnn-small`. Cost ≈ 0.2 GPU-h. If the CTM's class sums move as much as the CNN's logits, Proposition
2(1) is not binding in practice and count-pooling is dead on arrival.

### 3.6 The empty clause is not a degenerate case — it is the initial condition

With `init="boundary"` every TA starts at `N−1`, so `I_j = ∅` for every clause, so `C_j ≡ 1` during
learning and `C_j ≡ 0` at prediction `[FACT: base.py:230-238; functional.py:85-89; Granmo et al.
2019 §2.2]`. At epoch 0 every clause therefore fires on every image and `v_i = Σ_j p_j w_j = 0` for a
balanced pool — learning starts from the all-firing state and *contracts*. This is the opposite of a
CNN's small-random-weights start, and it is why Type Ib (forgetting) is what shapes the early model.
It is also the source of CHARTER failure mode 1: forgetting `model.eval()` makes an
under-trained clause pool look like it fires on everything.

---

## 4. Capacity

### 4.1 Raw budget

The learnable state is `C × 2F` include/exclude decisions plus the weights
`[FACT: base.py docstring; ta_state is (n_clauses_total, 2·n_features)]`.

| Configuration | C | 2F | include/exclude bits | comparison |
|---|---|---|---|---|
| this repo's prior stack (therm-4, 4×4, 640 cl) | 640 | 496 | 0.32 M | `[MEASURED: experiments/mctm — n_automata 1 265 664 for the 2-layer stack]` |
| GraphTM's CoTM baseline (adaptive, 8×8, 80 000 cl) | 80 000 | 480 | 38.4 M | — |
| Toolbox SOTA member (colour-therm-8, 5×5, 64 000 cl) | 64 000 | 1 308 | 83.7 M | ResNet-18 ≈ 11.2 M fp32 = **358 M bits** |
| Toolbox member at 10×10, Z=24 | 64 000 | 4 888 | 313 M | ≈ ResNet-18 |

**Reading**: at the budgets where the TM literature gets 75%, the TM holds 10⁷–10⁸ bits of learnable
state — the same order as a small CNN. `[HYPOTHESIS]` The CIFAR-10 gap is therefore **not** a raw
storage-capacity gap; it is a gap in what those bits are allowed to express (§2–§3).
**Falsifier**: a clause-scaling curve that saturates far below the CNN while the automata budget is
already larger than the CNN's parameter budget — which is exactly what P4.1 measures. If instead the
curve is still rising steeply at our largest affordable budget, the honest reading flips to
"capacity-limited", and PLAN §9.2 item 9 applies.

### 4.2 What the published curves say

`[FACT: Grønningsæter et al. 2024, Table IV]` from 2 000 → 64 000 clauses (32×):

| specialist | 2 000 | 8 000 | 32 000 | 64 000 | Δ(32k→64k) | saturated? |
|---|---|---|---|---|---|---|
| 5×5 Colour Thermometers | 64.5 | 69.1 | 73.7 | **75.4** | **+1.7** | **no** |
| 5×5 Adaptive Colour Therm. | 62.0 | 67.2 | 71.6 | 73.3 | +1.7 | no |
| 5×5 Adaptive Gaussian | 57.9 | 63.0 | 68.2 | 70.2 | +2.0 | no |
| HOG (32×32) | 64.2 | 66.9 | 67.5 | 67.5 | **0.0** | **yes, by 32 000** |
| 10×10 Canny | 48.7 | 53.4 | 58.8 | 60.6 | +1.8 | no |
| **Composite of all 22** | 79.5 | 81.5 | 82.7 | **82.8** | **+0.1** | **yes** |

`[HYPOTHESIS]` **The clause-scaling curve saturates at the information ceiling of the
Booleanization, not at a ceiling of the clause mechanism.** HOG — a pre-pooled, already-aggregated
descriptor — saturates at 67.5%; raw colour thermometers do not saturate even at 64 000 clauses.
**Falsifier**: a Booleanization on which a CNN trained on the *same bits* substantially outperforms
the saturated TM. That is exactly the `cnn-boolean` baseline in P2, and it is why the
Booleanization-loss decomposition is the report's spine.

### 4.3 Not saturated is not the same as a route — the slope is what matters

*(Added 2026-09-20 after the ROUND-1 crossfire; this supersedes any reading of §4.2 as "buy
clauses".)*

The per-doubling deltas for the best encoding `[FACT: arXiv:2406.00704 Table IV]` are
**+2.3, +2.3, +2.6, +2.0, +1.7** — an average of **+2.18 pp per doubling**, decaying slowly.
Extrapolating at the last measured slope and pricing on `[MEASURED: results/tables_p0.md]`:

| target | clauses required | GPU-h per seed |
|---|---|---|
| 82.8% (match the composite as a **single** model) | ≈ 1.3 × 10⁶ | ≈ 800 |
| 85% (small CNN, PLAN §2 T4) | ≈ 3.2 × 10⁶ | ≈ 1 960 |
| 94% (ResNet-18, T5) | ≈ 1.3 × 10⁸ | ≈ 77 000 |

**The clause axis is logarithmically open and practically closed.** The useful consequence is a
change of statistic: a mechanism earns its place iff it **shifts** (intercept) or **steepens**
(slope) the accuracy-versus-`log₂(clauses)` curve at matched automata budget, and the quantitative
bar is **worth more than two doublings, i.e. ≈ +4 pp at matched budget** — otherwise it is cheaper
to buy clauses. Two points per arm give a slope, so this is measurable at small budgets.

### 4.4 Composition beats clause count at matched *total* budget — and the margin shrinks

Derived from `[FACT: arXiv:2406.00704 Table IV]`; the comparison is not made anywhere in the
literature. A composite of 22 members at `n` clauses each holds `22n` clauses in total:

| composite | total clauses | composite | single model at that total | Δ |
|---|---|---|---|---|
| 22 × 2 000 | 44 000 | **79.5** | ~74.5 (interpolated between the measured 32 k and 64 k points) | **+5.0** |
| 22 × 4 000 | 88 000 | **80.6** | ~76.2 | **+4.4** |
| 22 × 8 000 | 176 000 | **81.5** | ~77.9 (mild extrapolation) | **+3.6** |

Only the first row is pure interpolation. `[FACT, derived]` **At 44 000–88 000 total clauses,
composition over diverse Booleanizations is worth +4.4 to +5.0 points over a single model of the same
total clause budget, and the advantage shrinks monotonically with budget.** It is therefore the most
automata-efficient mechanism in the literature — but it is **confounded with encoding diversity**,
and the separating control (same-encoding composite vs diverse-encoding composite at equal total
clauses) has never been run. `[HYPOTHESIS]` same-encoding composition captures ≤ 2 of the 5 points.
**Falsifier**: the two composites land within the 1.00 pp seed band of each other.

---

## 5. Feedback, precisely — and what the random matching patch does

### 5.1 The rules

For example `(x, y)` the library samples one negative class `y⁻ ≠ y` `[FACT: functional.py:123-144]`
and sets per-class feedback probabilities

    p_y  = (T − v_y)/2T ,    p_{y⁻} = (T + v_{y⁻})/2T ,   0 elsewhere

`[FACT: functional.py:110-120; classifier.py:153-160; Granmo et al. 2019 §2.4; Book Ch. 4 §4.4 step 4]`.
Each clause is drawn independently at its class's probability; then

| | positive-polarity clause | negative-polarity clause |
|---|---|---|
| class = `y` (target) | **Type I** | **Type II** |
| class = `y⁻` (sampled negative) | **Type II** | **Type I** |

`[FACT: classifier.py:168-171]`. Type I splits by clause output: **Type Ia** if the clause fired,
**Type Ib** if not `[FACT: base.py:341-344]`. Then:

* **Type Ia** — for each literal of the chosen patch: if the literal is **True**, move toward
  Include with probability `p_mem` (`= 1` when `boost_true_positive`, else `(s−1)/s`); if **False**,
  move toward Exclude with probability `1/s` `[FACT: functional.py:244-251]`.
* **Type Ib** — *every* literal moves toward Exclude with probability `1/s`. **It needs no patch
  features at all** `[FACT: conv.py:158; Book Ch. 4 §4.4 step 4(a)(ii)]`.
* **Type II** — only if the clause fired: every **False** literal of the chosen patch that is
  currently excluded is pushed one step toward Include, capped at the include boundary `N`
  `[FACT: functional.py:253-257]` — deterministically, with no `s`-gate. This is the asymmetry
  behind LG-003.

### 5.2 The random matching patch: four consequences

> **Revised 2026-09-20 after `[MEASURED: results/ctm-small_seed*.json → diagnostics.match_count]`.**
> Point 2 survives, point 3 is corrected, point 4 is restated. The original point 3 claimed the OR is
> "almost never an OR over more than one term"; that is **withdrawn**. See ROUND-1 crossfire response §1.

Let `M_j(x) = { b : c_j^b(x) = 1 }` and let the update patch be `b* ~ Uniform(M_j(x))`
`[FACT: Granmo et al. 2019 §3; Book Ch. 4 §4.4 step 4(a)(i) and 4(b); conv.py:169-171]`.

1. **The update is an unbiased sample of the clause's own evidence.** Type Ia therefore refines the
   clause toward the *average* of the patches it currently matches; the fixed point is a conjunction
   that is frequent within its own match set. This is a self-referential objective, not a global one.
2. **Specificity is free.** The clause's contribution to the vote is `1` whether `|M_j| = 1` or
   `|M_j| = 841`. Type Ib punishes only `|M_j| = 0`. So the learning pressure is exactly
   *"match ≥ 1 patch of as many target images as possible and 0 patches of as many negative images
   as possible"*, and shrinking `M_j` from 841 to 1 costs nothing while strictly improving the
   second term. **There is a standing pressure toward knife-edge detectors.**
   **But it is not the only pressure — see 2b, which the first version of this section omitted.**
2b. **Type Ib is the density floor.** A clause that narrows too far stops firing on target images,
   and a non-firing clause selected for Type I feedback receives **Type Ib**, which pushes *every*
   literal toward Exclude with probability `1/s` `[FACT: functional.py:250-251; conv.py:158; Book
   Ch. 4 §4.4 step 4(a)(ii)]` — i.e. it forces the clause back open. The equilibrium density is the
   balance of Type II (narrow, to kill false positives) against Type Ib (broaden, to keep firing on
   targets). A mode at `|M| = 1` with a long tail is exactly what that balance produces.
3. **Measured, and it is long-tailed rather than degenerate.**
   `[MEASURED: results/ctm-small_seed*.json → diagnostics.match_count]` — `ctm-small`
   (2 000 clauses, 4×4, therm-4, trained, validation split, 2 000 images × 841 positions):

   | quantity | value |
   |---|---|
   | fire_frac (clause × image pairs) | **0.0698** |
   | `P(\|M\| = 1 \| fires)` | **0.354** |
   | q25 / **median** / q75 given firing | 1 / **2** / 6 |
   | p95 / max given firing | 24 / 433 |
   | mean given firing | 5.99 |
   | `P(\|M\| ≥ 5 \| fires)` | **0.303** |

   So the *mode* is 1 and a third of firing events have nothing to count — point 2 is visible — but
   **64.6% of firing events have `|M| ≥ 2` and 30.3% have `|M| ≥ 5`**, and the mean is tail-dominated.
   The earlier claim that the OR is "almost never an OR over more than one term" is **withdrawn**.
   The older `[MEASURED: experiments/mctm; PLAN.md §3.1]` 0.14% per-patch figure is not in conflict:
   it is a *per-patch* firing rate for layer-1 clauses in a greedily trained stack, a different
   quantity on a different model.
4. **Restated prediction.** `[HYPOTHESIS]` The information a count pool adds to the vote is small —
   per clause coordinate, `H(indicator) = 0.365` bits against `H(count) = 0.625` bits, a **×1.71**
   gain, not the ×10 an unconditioned reading of the count range suggests (the gain is gated by
   `fire_frac = 0.07`; `H(|M| | fires) ≈ 3.7` bits from a point-mass-plus-geometric fit matched to
   the measured `P(=1)` and mean). For comparison a CNN filter on the same task gives **×1.96**
   `[MEASURED: screen/cnn-ctmshape-max_seed0.json]` — **the ratio does not discriminate between the
   substrates, so it predicts nothing in either direction.**
   The mechanism by which a count pool could matter is therefore **not** information, it is the
   **objective**: with a threshold `κ ≥ 2`, narrowing below `κ` kills the clause, so a count pool is
   *density control applied through the forward pass*. It is confounded with an explicit
   firing-rate controller by construction and cannot be attributed without a density-only control.
   **Falsifier / screen**: the six-arm set `pool-or` / `pool-density` / `pool-thermo` /
   `pool-thermo+density` / `pool-sum` / `feedback-kpatch` in `rounds/ROUND-1/tm-theorist.md` §1.4.
   Counting is dead iff all of `pool-thermo`, `pool-thermo+density` and `pool-sum` land within the
   seed band of `pool-density`.
2c. **Correction: the dominant force adds literals, it does not remove them.** Point 2 says
   narrowing is free; it does not say what *drives* clause composition, and I had the sign of the
   dominant term wrong. The literal-adding forces are **deterministic**: Type Ia memorises every
   **True** literal of the chosen patch with probability `p_mem = 1` when `boost_true_positive`
   (the library default *and* the reference default) `[FACT: functional.py:230, 245]`, and Type II
   pushes every **False** excluded literal one step toward inclusion with no `s`-gate at all
   `[FACT: functional.py:253-257]`. The only literal-**removing** force is probabilistic at rate
   `1/s` — Type Ia's False literals and Type Ib's blanket forgetting `[FACT: functional.py:250-251]`.
   A deterministic inflow against a `1/s` outflow has a wide equilibrium, and the measurement
   confirms it: `[MEASURED: results/ctm-small_seed*.json]` an unconstrained clause settles at
   **156 of the 248 literals it could possibly include — 63%**. See §5.4.

4b. **The pool is not where the CNN's advantage lives.** A CNN's `max` → `sum` gap confounds forward
   information with *credit assignment* (max routes the gradient to one spatial position, sum to
   all). The TM has no gradient, but it has the exact analogue of the first behaviour: **the update
   already comes from one uniformly random matching patch, whatever the pool does**
   `[FACT: Granmo et al. 2019 §3; conv.py:169-171]`. Changing the pool therefore transfers only half
   the CNN mechanism; transferring the other half means changing the **draw** (κ matching patches per
   feedback event), which is a feedback candidate, not a pooling one.
   **Request to the engineer**: log the full `|M|` histogram, not only quantiles. It is free, and it
   turns the entropy figures above from an estimate into a measurement.

### 5.3 Why Type Ia and Type II draw independently

Under a class-owned CTM, Type I and Type II are mutually exclusive for a given (example, clause)
`[FACT: classifier.py:169-170 — the two masks partition `selected`]`, so patch sharing is moot.
Under a **coalesced** model a clause can have a positive weight for `y` and a positive weight for
`y⁻`, so it can receive Type I and Type II in the *same* update
`[FACT: coalesced.py:156-157]` — and the library therefore draws a separate patch for each kind
`[FACT: conv.py:166-171]`. Drawing one shared patch would force the "refine" and the "discriminate"
updates to act on the same evidence, which is not what the algorithm specifies. The repo's own
commit `5dcef4f` records the other half of this: drawing in `_evaluate` instead costs a
`(B,P,C)` rand+argmax on every forward pass (~5× slower prediction).


### 5.4 The clause-size budget: why a hard cap buys 7.50 points, and where the optimum should sit

*(Added 2026-09-20. This section is written **before** the budget sweep runs; §5.4.4 is a
pre-registration, not a fit.)*

> **STATUS, 2026-09-21 — do not edit this section; it is the record.** `[MEASURED:
> results/ctm-small-seq-{unc,b32}_seed{0,1,2}]` **P3 is falsified by its own criterion**: under
> `feedback_mode="sequential"` the gap is **+8.75 pp**, not the predicted ≤ +1 pp. `[MEASURED:
> results/ctm-small-T80-{unc,b32}_seed{0,1,2}]` §5.4.5's `T` fault is repaired and correcting it
> **added** 2 pp to the effect (+9.46 at a reachable `T = 80`), so budget and margin are
> complements. §5.4.3's reconciliation — "the +7.50 pp is a repair of what batched feedback throws
> away" — is therefore **withdrawn**. P1, P2 and P4 remain open and are the subject of
> `jobs/p3_budget_predictions.txt`. The settled account is **§5.5**.

#### 5.4.1 What was measured

`[MEASURED: results/ctm-small_seed{0,1,2}.json vs results/ctm-small-budget32_seed{0,1,2}.json]` —
identical arms (2 000 total clauses, 4×4, therm-4, `T=160`, `s=10`, batch 50, 30 epochs, unweighted)
except for `max_included_literals`:

| | clause length (mean) | as % of the 248 includable | firing rate | mean \|M\| | \|M\| given firing | per-patch match prob | per-literal selectivity `−ln q` | **test accuracy** |
|---|---|---|---|---|---|---|---|---|
| unconstrained | **156.4** | **63.1%** | 0.0695 | 0.427 | 6.14 | 5.1 × 10⁻⁴ | 0.0485 | **37.08 ± 0.61** |
| budget 32 | **32.5** | 13.1% | **0.1586** | 1.619 | 10.21 | 1.9 × 10⁻³ | **0.1923** | **44.58 ± 0.55** |
| ratio | 4.8× | | **2.28×** | 3.8× | | 3.8× | **4.0×** | **+7.50 pp** |

(`q` is the *effective* per-literal match probability, `q = (p_patch)^{1/L}`.)

#### 5.4.2 The account — §5.2 had the sign of the dominant force wrong

§5.2 point 2 said specificity is free, and it is. But "free" is not "driven": what *drives*
composition is the asymmetry recorded in point 2c. **Literals are added deterministically and removed
probabilistically**, so the unconstrained fixed point is wide, not narrow — and 63% of maximum is not
a pattern, it is **a memorised patch**. At that length the clause matches 5.1 × 10⁻⁴ of patches and
fires on 7% of images. This is over-fitting in the most literal sense available to a TM.

The measured selectivity makes the second half of the story precise. A naive `q^L` model with a
single `q` fails by two orders of magnitude, because **the cap does not merely shorten clauses — it
forces a choice of which literals survive**, and the survivors are four times more selective
(`−ln q`: 0.0485 → 0.1923). A 32-literal clause is 4.8× shorter *and* built from individually rarer
literals, and it still matches 3.8× more patches.

**Why a cap beats `s`.** `s` acts as a per-literal, per-event Bernoulli(1/s) removal; the cap acts as
a **deterministic, per-clause, per-commit** conversion of all Type Ia events into Type Ib
`[FACT: functional.py:232-242]`. Under batched feedback the deterministic inflow terms aggregate over
the whole mini-batch before a single commit, so a probabilistic outflow at rate `1/s` is out-raced,
while a deterministic gate is not. **The cap is the only deterministic forgetting force in the
algorithm.** That is the mechanism, and it predicts §5.4.4-P3.

#### 5.4.3 Reconciling +7.50 pp here with +0.06 pp in the paper

`[FACT: arXiv:2301.08190 Table 3]` CSC-TM reports the budget as worth **+0.06** on CIFAR-2 and
**+0.05** on MNIST-with-convolution, under the **sequential** algorithm, with an unconstrained mean
clause length of **60.4**. We measure **+7.50 pp** under **batched** feedback at an unconstrained
length of **156.4** — a **2.6× longer** unconstrained clause on a different dataset and encoding, so
an order-of-magnitude anchor rather than a matched comparison.

`[HYPOTHESIS]` **The +7.50 pp is a repair of what batched feedback throws away, not a new
mechanism.** Batching lets the deterministic inflow accumulate across an entire mini-batch before the
`1/s` outflow is applied once, inflating the unconstrained equilibrium; the cap restores the size
control that the sequential algorithm gets from `s` for free. This makes the budget a **correction
term for an approximation we chose**, which is exactly the kind of thing that must be disclosed in
the report rather than presented as a finding about Tsetlin machines.

#### 5.4.4 Four predictions, registered before the sweep

**P1 — the optimum is at or below 32, and the curve is flat beneath it.**
`acc(32) > acc(64) > acc(none)`, and `acc(16)` within the 1.00 pp seed band of `acc(32)`.
Reasoning: firing rate is the mediating variable (P2), and it saturates from below because shorter
clauses match in larger contiguous blobs rather than on more images — `|M|` given firing already
rises from 6.1 to 10.2 as the cap tightens, so most of the extra coverage is spent inside images the
clause already fires on. **Falsifier**: `acc(64) > acc(32)`, or `acc(16)` more than 1.00 pp below
`acc(32)` (which would mean discriminativeness binds and the optimum is sharply peaked, not flat).

**P2 — the budget's entire effect is mediated by the firing rate.**
Plot accuracy against *achieved* firing rate rather than against the budget. Points from
`{budget 16, 32, 64, none} × {s sweep} × {explicit density controller}` should **collapse onto one
curve**. **Falsifier**: at matched firing rate, the budget arm and the `s` arm differ by more than the
seed band — which would mean the cap does something beyond density control (most plausibly, that
*which* literals are kept matters independently of how many).

**P3 — the effect is largely a batched-feedback artefact.**
Under `feedback_mode="sequential"`, budget-32-versus-unconstrained should shrink to **≤ +1 pp**,
consistent with CSC-TM's +0.06. **Falsifier**: sequential also shows ≥ +5 pp. Cheap at `ctm-small`
scale, and it is the single measurement that decides whether §5.4.3 goes in the report as a caveat
or as a finding.

**P4 — the optimum budget is roughly absolute, not proportional to the encoding's width.**
Two framings disagree, which is what makes this worth running. The *proportional* framing says
`b* ∝ n_features` (13% of 248 → ≈85 at 5×5 colour thermometers, where `n_features = 654`). The
*coverage* framing says `b* ≈ [ln P − ln(1/f*)] / ln(1/q)`, which depends on the required firing rate
and the per-literal selectivity, is only **logarithmic** in the patch count, and is therefore nearly
constant across windows and encodings. **I back the coverage framing** because P2 says firing rate is
the mediator: **`b*` should vary by less than 2× while `n_features` varies by 2.6× (248 → 654) and
`P` varies by 1.6× (841 → 529).** **Falsifier**: `b*` tracks `n_features` proportionally.
This also carries a claim about the literature: if `b*` is roughly absolute, the Toolbox's use of a
single budget of 32 across all seven encodings `[FACT: all 22 reference scripts]` was right; if `b*`
is proportional, the Toolbox left accuracy on the table at its wider encodings.

#### 5.4.5 An independent second fault in the same baseline: `T` is unreachable

`[MEASURED: results/ctm-small_seed*.json hp]` `ctm-small` has 2 000 total clauses over 10 classes =
200 per class = **100 positive**, unweighted, so the **maximum achievable class sum is 100** — and
`T = 160`. The vote can never reach the margin, so the feedback probability
`(T − v)/2T` never anneals: it sits at **0.478** unconstrained and **0.450** at budget 32. The margin
is doing nothing at all, in either arm.

This is precisely the failure `ARMS.md` convention 2 warns about, and it has a one-line fix:
**`T_ratio` must be taken against the positive half, `n_clauses_per_class / 2`, not against
`n_clauses_per_class`** (for an unweighted class-owned model; with `weighted=True` the reachable
maximum is the summed positive weights instead). `[HYPOTHESIS]` a correctly set `T` will move the
optimum budget **up**, because a lower required firing rate tolerates longer clauses — **falsifier**:
re-running the budget sweep at a reachable `T` leaves the optimum unchanged. Until that is done,
**the budget sweep and the `T` setting are confounded**, and P1's ordering is a claim about
`ctm-small as configured`, not about convolutional TMs.

### 5.5 The clause-size budget, settled: the inclusion ratchet

*(Added 2026-09-21, **after** the sequential control and the corrected-`T` pair landed. §5.4 is left
exactly as it was written — it is a pre-registration and editing it would destroy the record. This
section reports what the pre-registration got right, what it got wrong, and what the account is now.
`[MEASURED: results/ctm-small-T80-{unc,b32}_seed{0,1,2}, results/ctm-small-seq-{unc,b32}_seed{0,1,2}]`)*

#### 5.5.1 The four-cell measurement

All four cells are the same arm — `ctm-small`, 2 000 total clauses, 4×4 window, therm-4 (`Z = 12`,
`n_features = 248`, `P = 841`), `s = 10`, batch 50, 30 epochs, unweighted, class-owned — with a
**reachable** `T = 80` (against the achievable class sum of 100; the earlier `T = 160` was inert,
§5.4.5). Three seeds per cell.

Two clause lengths are reported per cell and they are **not** the same quantity. `L₃₀` is the
equilibrium the process is heading for (mean over clauses at epoch 30, the last epoch); `L*` is the
length of the model that actually produced the accuracy (the validation-selected snapshot, which the
runner restores before computing the final diagnostics). They diverge in the unconstrained arms
because validation selects them **early**.

| feedback mode | budget | test acc | sel. epoch | `L₃₀` | `L₃₀` / 248 | `L*` | image firing rate | **effect** |
|---|---|---|---|---|---|---|---|---|
| batch 50 | none | 38.31 ± 1.05 | 4 / 15 / 10 | **170.36 ± 0.28** | **68.7%** | 147.5 ± 15.2 | 0.0718 ± 0.0003 | — |
| batch 50 | 32 | 47.77 ± 0.32 | 26 / 29 / 25 | **32.14 ± 0.02** | 13.0% | 32.15 ± 0.08 | 0.1452 ± 0.0011 | **+9.46** |
| sequential | none | 38.06 ± 0.38 | 15 / 10 / 26 | **164.77 ± 1.13** | 66.4% | 154.0 ± 7.7 | 0.0730 ± 0.0002 | — |
| sequential | 32 | 46.80 ± 0.17 | 29 / 24 / 18 | **31.89 ± 0.01** | 12.9% | 31.91 ± 0.07 | 0.1422 ± 0.0012 | **+8.75** |

Four facts to carry forward, in decreasing order of how much they constrain the theory:

1. **The effect does not depend on batch aggregation.** 9.46 vs 8.75 pp; the two modes differ by
   0.71 pp against a measured 1.00 pp seed band `[MEASURED: results/calibration_seednoise.json]`.
2. **Batching inflates the unconstrained equilibrium by 3.4%, not by the 2.6× that would be needed.**
   `L₃₀` = 170.36 ± 0.28 batched against 164.77 ± 1.13 sequential — the intervals *are* disjoint, so
   §5.4.3's mechanism is real and points the right way, but it accounts for **5.6 of the 138
   literals** the cap removes (170.36 → 32.14), i.e. **4%** of it. The magnitude was wrong by a factor of ~20, and that is what falsifies
   P3, not the direction.
3. **The equilibrium is a genuine fixed point.** `L₃₀` reproduces across seeds to **±0.28 literals**
   out of 170 (0.16%), and the firing rate to ±0.0003 out of 0.072 (0.4%). Nothing about this is
   noisy; it is a dynamical attractor of the feedback rule, and §5.5.2 derives it.
4. **Validation selection stops the unconstrained arm early and the budgeted arm late** (epochs
   4/15/10 versus 26/29/25). That is not a protocol artefact: the unconstrained arm's validation
   accuracy is flat from epoch 1 (§5.5.3), so selection is picking noise, while the budgeted arm
   improves monotonically for ~27 epochs. The *accuracy* of the unconstrained model converges in
   ~4 epochs while its *clause length* is still rising at epoch 30 — see §5.5.3, which is the
   central measurement of this section.

#### 5.5.2 Proposition 5 — inclusion is a ratchet: only Type Ib can remove an included literal

**Claim.** In a Tsetlin machine, convolutional or flat, an *included* literal can be moved toward
Exclude by **Type Ib feedback alone**. Type Ia's forgetting branch can never touch it, and Type II
never moves any literal downward.

**Proof.** `[FACT: functional.py:244-251; conv.py:167-176; Book Ch. 4 §4.4 step 4(a); Granmo et al.
2019 §3]` Type Ia is given only to a clause that evaluates to 1, and its forgetting branch applies
only to literals that are **False in the update patch**. In the convolutional machine the update
patch is drawn uniformly from `M_j(x)`, the set of patches the clause *matches*
(`cand = matches[b, :, j]`, `p_idx = argmax(rand · cand)`, `conv.py:167-172`), and matching requires
every included literal to be True. Hence `n_false` is identically zero on the included coordinates,
and the decrement `dec = Binomial(n_false + n_ib, 1/s)` reduces to `Binomial(n_ib, 1/s)` there
(`functional.py:250-251`). In the flat machine the same holds trivially, because a clause outputs 1
only if all its included literals are true. Type II adds and never subtracts, and is additionally
clipped so that it cannot raise a state past the include boundary
(`room = N − state`, `functional.py:256-257`). ∎

**Three corollaries, all measurable.**

* **C5.1 — the admission threshold.** For an *excluded* literal `l`, Type Ia drives it up with
  probability `p_mem·π_l` and down with probability `(1−π_l)/s`, where `π_l` is the frequency with
  which `l` is true across the clause's own matching patches. The drift is upward iff
  **`π_l > 1/(1 + s·p_mem)`**, which with the library and reference default `boost_true_positive=True`
  (`p_mem = 1`, `functional.py:230`) is **`π_l > 1/(1+s)`** — `0.091` at `s = 10`, `0.167` at
  `s = 5`, `0.333` at `s = 2`. This is an *admission* rule, not a *selection* rule: it asks whether a
  literal is frequent inside the match set, never whether it is useful.
* **C5.2 — the eviction condition.** An included literal is decremented only on Type Ib events, i.e.
  only when the clause **fails to fire** on an example selected for Type I feedback, and then only
  with probability `1/s`. Writing `f` for the clause's firing rate over its Type-I-eligible examples
  (for a positive-polarity clause of class `c`, examples with `y = c`), the ratio of admitting to
  evicting pressure is `s·f/(1−f)`, so the clause keeps growing while **`f > 1/(1+s)`** — the *same*
  threshold, now on the firing rate. `[HYPOTHESIS]` **The unconstrained equilibrium is therefore
  self-terminating only at the point where the clause is about to stop firing**, and the equilibrium
  firing rate is pinned near `1/(1+s)` independently of clause count, `T`, and dataset.
  **First check, free and encouraging but not decisive**: the measured *global* image-level firing
  rate of the unconstrained arms is **0.0718 ± 0.0003** (batched) and **0.0730 ± 0.0002**
  (sequential) against a predicted `1/(1+s) = 0.0909` at `s = 10` — within 21% of a first-order
  argument, and mode-independent as the argument requires. It is not decisive because the global
  rate mixes positive-polarity clauses (Type-I-eligible on 1/10 of images) with negative-polarity
  ones (eligible on 9/10), which is exactly what diagnostic **D-FIRE** in §5.5.9 separates. The
  sharp test is the `s` sweep: `1/(1+s)` moves by **7×** between `s = 2` and `s = 20`, so the
  unconstrained firing rate must move with it. **Falsifier**: the unconstrained firing rate is flat
  in `s` across `{2, 10, 20}` — Block 2 of `jobs/p3_budget_predictions.txt`.
* **C5.3 — the two populations of included literals.** Type Ia drives a literal all the way to
  `2N−1` (128 further increments at the default `N = 128`, `base.py:81`), because once included it is
  True in every drawn matching patch and is re-memorised on every Type Ia event. Type II can only
  ever place a literal at **exactly** the boundary state `N`, where a single Type Ib decrement evicts
  it. `[HYPOTHESIS]` The TA state histogram of a trained clause is therefore **bimodal** — a deep
  core at `2N−1` and a Type-II fringe at `N` — and a budget strips the fringe first, because eviction
  time is proportional to depth. CSC-TM describes the fringe itself
  `[FACT: arXiv:2301.08190, §"Note that although the length of the clauses are constrained…"]`:
  *"Type II Feedback is not constrained and can produce clauses with more literals than the budget.
  In this case, the TAs of the included literals will all be in the middle states… the included
  literals at boundary due to Type II Feedback do not necessarily contribute positively to the
  classification."* What is new here is the claim that **stripping that fringe is where the accuracy
  comes from**, which is a statement about the *cap's benefit*, not about the cap's overshoot.

#### 5.5.3 The ratchet, seen directly in the training curve

`[MEASURED: results/ctm-small-T80-unc_seed0.json → curve]` The unconstrained arm, per epoch:

| epoch | clause length | per-patch match prob. | image firing rate | val acc |
|---|---|---|---|---|
| 1 | 86.7 | 4.85 × 10⁻⁴ | 0.0730 | 36.52 |
| 5 | 136.3 | 5.19 × 10⁻⁴ | 0.0732 | 37.30 |
| 10 | 153.5 | 5.37 × 10⁻⁴ | 0.0720 | 36.90 |
| 20 | 165.0 | 5.49 × 10⁻⁴ | 0.0720 | 37.06 |
| **30** | **170.6** | **5.56 × 10⁻⁴** | **0.0720** | **37.62** |

> **Between epoch 5 and epoch 30 the unconstrained clause adds 34.4 literals (+25%) while its
> per-patch match probability *rises* by 7%, its image-level firing rate is unchanged to three
> decimals, and its validation accuracy moves +0.32 pp — inside the 1.00 pp seed band.**

Taking it from epoch 1 instead makes the point harder: the clause **doubles** in length, 86.7 → 170.6
(+97%), and the firing rate moves from 0.0730 to 0.0720 (−1.4%).

That is the ratchet, measured. **Clause length has not converged at 30 epochs; accuracy has.** The
literals arriving after epoch 5 are, on average, *logically implied by the ones already there*: they
do not shrink the match set, do not change the vote on any image, and do not change the loss. They
are admitted because C5.1 admits anything frequent inside an already-tiny match set, and they are
never removed because C5.2 evicts only a clause that has stopped firing — and a clause that adds
redundant literals never stops firing.

The budgeted arm is the control: `[MEASURED: results/ctm-small-T80-b32_seed0.json → curve]` length
converges within 5 epochs (27.9 → 31.4 → 32.1), the firing rate *rises* 0.113 → 0.145, and accuracy
rises 38.8 → 47.3.

**Per-literal selectivity, recomputed on the corrected-`T` arms, three seeds, at epoch 30.** Writing
`q = p_patch^{1/L}` for the geometric-mean per-literal match probability, the unconstrained clause
reaches `−ln q = 0.0440 ± 0.0002` and the budgeted one `0.2030 ± 0.0007` — **4.61× more selective per
literal** `[MEASURED: ctm-small-T80-{unc,b32}_seed{0,1,2} → curve[-1]]`, and the sequential pair gives
the same ratio (0.0452 ± 0.0003 vs 0.2054 ± 0.0004, 4.54×). The §5.4.1 figure of 4.0× was computed at
the inert `T = 160` and at the selected epoch; the corrected value is slightly larger.

#### 5.5.4 Why the cap works: it is the only closed loop from clause length to forgetting

The precise statement — §5.4.2's "the only deterministic forgetting force" was close but not right,
because the cap's forgetting is still Bernoulli(1/s):

> **The clause-size budget is the only mechanism in the algorithm whose activation is a function of
> clause length.** `[FACT: functional.py:232-242]` An oversize clause has **every** Type Ia event
> converted into a Type Ib event, with probability 1, tested per commit against `include_count`.
> Every other force is either blind to length (Type Ia admission, Type II) or coupled to it only
> indirectly, through the firing rate, with loop gain `1/s` (Type Ib, C5.2).

So the unconstrained machine has exactly one negative feedback path from length back to length, it
runs through the firing rate, and it is attenuated by `s`. §5.5.3 shows that path is effectively open
in this regime: 34 literals arrive without moving the firing rate at all, so the loop never sees
them. The cap closes a **direct, unit-gain, bang-bang** path — oversize ⇒ pure forgetting ⇒ shrink ⇒
resume — and by C5.3 what it sheds first is the shallow Type-II fringe.

`[FACT: functional.py:232-242 vs 253-257]` **The cap gates Type Ia only; Type II remains ungated**,
so an oversize clause keeps receiving inclusion pressure. This matches CSC-TM exactly
`[FACT: arXiv:2301.08190, "The difference between vanilla TM and CSC-TM lies in Type I Feedback.
Type II Feedback remains the same for both schemes"]` and is recorded as faithful in §8 D2.

#### 5.5.5 What the sequential control rules **out** — P3 falsified

§5.4.4's **P3 is falsified by its own criterion** ("sequential also shows ≥ +5 pp" — it shows
+8.75). The value of a falsified prediction is what it eliminates, and this one eliminates a whole
family of explanations at once:

| ruled out | why | evidence |
|---|---|---|
| **Batch-aggregation inflation *as the explanation*** — deterministic inflow accumulating over 50 examples before one `1/s` outflow is applied | the effect **exists** (`L₃₀` = 170.36 ± 0.28 batched vs 164.77 ± 1.13 sequential, disjoint) but is **3.4%**, against the ~2.6× needed to reconcile our 147–170 literals with CSC-TM's 60.4. It explains 5.6 of the ~138 literals the cap removes | `[MEASURED: ctm-small-T80-unc vs ctm-small-seq-unc, curve[-1]]` |
| **Boundary-clamp rectification** — per-commit rather than per-event clamping at 0 / `2N−1` preserving extreme states | a sub-case of the row above and bounded by the same 3.4%; it cannot be separated from aggregation by this experiment, and it does not need to be, because their *sum* is 3.4% | as above |
| **Event multiplicity** — "many events per commit" as the driver | batch 50 gives up to 50× the events per commit; the accuracy effect moves by 0.71 pp, inside the band | `[MEASURED: all four cells]` |
| **"The budget repairs an approximation we chose"** — i.e. the entire class of readings under which the result is about *this library* rather than about Tsetlin machines | all three above are the only routes by which `feedback_mode="batch"` differs from the published algorithm `[FACT: THEORY.md §8 D1]` | `[MEASURED: DR-004 Decision 2]` |
| **The reporting escape** — presenting +9.46 pp as a caveat rather than a result | closed deliberately, in DR-004 | — |

The last row is the one that matters for the report: because the batched path is eliminated, the
effect transfers to any faithful implementation, including `tmu`, and therefore to the published
recipes. That is what licenses §5.5.7's claim.

**What it does not settle**, stated so that nothing downstream over-reads it:

* nothing about *why* 9 pp here and 0.06 pp in CSC-TM (§5.5.6);
* nothing about scale — every cell is 2 000 total clauses, 4×4, therm-4. The budget control at
  20 000 and 80 000 is the open falsifier (DR-004);
* nothing about the *optimum* budget — only that 32 ≫ none (§5.4.4 P1, unrun);
* nothing about the mediator — P2 is unrun;
* **and, most importantly, nothing about whether `s` alone would buy the same 9 points** (§5.5.8).

#### 5.5.6 Reconciling +9.46 pp here with +0.06 pp in CSC-TM — and the bloat law

`[FACT: arXiv:2301.08190 Table 3 and fn. 6]` CSC-TM's CIFAR-2 row, **8 000 clauses, `T` = 6 000,
`s` = 10.0**, reporting accuracy with mean literals per clause in brackets:

| budget | ≤1 | ≤2 | ≤4 | ≤8 | ≤16 | ≤32 | ≤64 | **All** |
|---|---|---|---|---|---|---|---|---|
| CIFAR-2 | 69.82 (30.8) | 79.65 (30.1) | 87.01 (13.5) | 91.21 (6.8) | 93.23 (10.8) | 93.99 (20.1) | **94.24 (34.1)** | 94.18 (**60.4**) |

Note the shape: their optimum is at budget **64**, worth **+0.06** over unconstrained, and budget
**32 is worse than unconstrained** (93.99 vs 94.18, −0.19). Ours is +9.46 at budget 32.

`[HYPOTHESIS]` **The bloat law.** Define the bloat ratio `β = L_unconstrained / L_budgeted`. The
accuracy gain from a clause-size budget increases with `β` and tends to 0 as `β → 1`, because the
gain is exactly the accuracy cost of the literals the cap removes, and `β` counts them. Ours is
**`β = 5.30`** (170.36/32.14) batched and **`5.17`** (164.77/31.89) sequential at the equilibrium, or
4.60 / 4.83 if measured at the validation-selected snapshot; CSC-TM's, at its own optimum, is
**`β = 1.77`** (60.4/34.1). Quote the equilibrium figure: `β` is a statement about the fixed point of
the feedback rule, and the selected snapshot is an artefact of our protocol, which CSC-TM does not
share (it reports the worst maximum over 5 runs `[FACT: arXiv:2301.08190 Table 3 caption]`).

**Two things I will not claim.**

1. I cannot state `L_unconstrained` as a *fraction* of includable literals for CSC-TM's CIFAR-2,
   because the paper states neither its Booleanization nor its convolution window for that row
   `[FACT: arXiv:2301.08190 §"Image Processing" + fn. 6 — the window is stated for "MNIST
   w/conv." and never for CIFAR-2]`. Ours is 59.5% of 248. Their 60.4 literals could be anywhere
   from ~2% to ~45% of their includable set. Ours is 68.7% of 248 at the equilibrium and 59.5% at
   the validation-selected snapshot. This is a documented gap, not an inference.
2. I cannot treat `(1.77, +0.06)` and `(4.60, +9.46)` as two points on one curve, because a second
   explanation is equally consistent with both: **headroom**. CIFAR-2 unconstrained is already at
   **94.18%** on a two-class problem, where a memorised clause still separates vehicles from animals;
   our CIFAR-10 unconstrained is at **38.31%**, where it does not. The available gain differs by an
   order of magnitude before any mechanism is invoked.

**The discriminator, and it is cheap.** Run the identical budget pair on a **CIFAR-2 built from our
own pipeline** (the standard vehicle/animal merge of CIFAR-10), same encoding, same `s`, same clause
count, same window. `[HYPOTHESIS]` If the bloat account is right, `L_unconstrained` stays near 150
and the effect stays large (≥ +4 pp); if the headroom account is right, `L_unconstrained` stays near
150 and the effect collapses to ≲ +1 pp. **Falsifier for the bloat law**: the CIFAR-2 arm shows
`β ≥ 4` and Δ ≤ +1 pp. Cost ≈ **0.62 GPU-h** (2 arms × 3 seeds at `ctm-small` scale); it needs a
~10-line two-class label map in `code/data.py`, which is the engineer's call, so it is proposed
rather than queued.

**When the effect should be large — the prediction with the falsifier attached.**
`[HYPOTHESIS]` Δ scales with how bloated the *unconstrained* baseline is, and by C5.1 that is set by
`s` and by the encoding, not by the clause count:

* **B1 (`s`).** `L_unconstrained` increases with `s`, because the admission threshold is `1/(1+s)`.
  Hence Δ increases with `s`. Concretely: at `s = 2` the threshold is 0.333, `L_unconstrained` should
  fall **below 32**, `β ≈ 1`, and **Δ should vanish into the seed band**; at `s = 20` the threshold is
  0.048, `L_unconstrained` should exceed 170.4 and **Δ should exceed +9.46**.
  **Falsifier**: Δ(s = 2) ≥ +3 pp while `L_unconstrained`(s = 2) ≤ 40; or Δ(s = 20) < Δ(s = 10) while
  `L_unconstrained`(s = 20) > `L_unconstrained`(s = 10). Tested by `jobs/p3_budget_predictions.txt`
  Block 2, **1.25 GPU-h**.
* **B2 (scale).** The admission and eviction thresholds are both `1/(1+s)` and contain no clause
  count, so `L_unconstrained` and the unconstrained firing rate should be **approximately invariant
  to the number of clauses**, and the budget gap should therefore **not close** between 20 000 and
  80 000 clauses. **Falsifier**: `L_unconstrained` differs by more than 1.5× between the 20 000 and
  80 000 unconstrained points, or the gap at 80 000 is less than half the gap at 20 000. Tested by
  the budget control already queued (DR-003 funding item 2, 11.5 GPU-h) — **no new compute**.
* **B3 (encoding).** At fixed `s`, `L_unconstrained` grows with `n_features` only through how many
  literals clear `π > 1/(1+s)` inside the match set, which is a property of the encoding's bit
  correlations rather than of its width. **Falsifier**: `L_unconstrained` is proportional to
  `n_features` across `{166, 248, 440}`. Tested by Blocks 3–4 of the same job file, **3.33 GPU-h**;
  the same runs settle §5.4.4 P4.

A first, free data point on B3 already exists and it cuts against pure proportionality:
`[MEASURED: results/ctm-therm5-preflight_seed0.json]` at 5×5 colour thermometers
(`n_features = 654`, 2.6× wider) with `s = 5.0` and budget 32, the **achieved** mean clause length is
**27.8** — the budget does not even bind at the mean. A proportional `b*` would be ≈ 85 there.

#### 5.5.7 The novelty claim, in the exact words the report should carry

> **The mechanism is not ours; its magnitude and its explanation are.** The clause-size constraint is
> published — Abeyrathna et al., *Building Concise Logical Patterns by Constraining Clause Size*
> (IJCAI 2023, arXiv:2301.08190) — and it is set to 32 in all twenty-two reference scripts of the
> Optimized Toolbox while appearing in none of that paper's tables. Its authors measured it as worth
> **+0.06 points on CIFAR-2** and **+0.05 on MNIST-with-convolution**, and presented it as an
> interpretability and hardware mechanism. On convolutional CIFAR-10 we measure it at
> **+9.46 ± 1.1 points** (47.77 ± 0.32 against 38.31 ± 1.05, three seeds, disjoint intervals), at
> 4.6× fewer literals per image and +1.7% wall-clock — larger than Drop Clause's +5.8, the largest
> published single-model CIFAR-10 mechanism. The effect survives the exact sequential algorithm
> (+8.75 ± 0.4), so it is a property of Tsetlin machines and not of our batched approximation. Our
> contribution is therefore (i) the measurement of a two-orders-of-magnitude discrepancy against the
> mechanism's own published value, (ii) the account of it — literal *inclusion* is a ratchet whose
> admission rule `π > 1/(1+s)` never asks whether a literal is useful and whose only eviction path
> runs through the firing rate at gain `1/s`, so an unconstrained clause accumulates logically
> redundant literals indefinitely (measured: the clause doubles in length over 30 epochs, 86.7 → 170.6 of a
> possible 248, while its firing rate moves by −1.4% and its validation accuracy by +1.1 pp), and the budget is the only mechanism in the
> algorithm whose activation depends on clause length — and (iii) the prediction, falsifiable and
> under test, that the size of the effect tracks the bloat of the unconstrained baseline rather than
> the dataset. **We did not invent a mechanism. We found that a published one was mis-priced by two
> orders of magnitude on the benchmark this field cares most about, and we say why.**

Three guards on that paragraph, all of which must hold before it is printed:

* it must name CSC-TM in the **first sentence**, not in a footnote;
* the comparison to Drop Clause's +5.8 is a comparison of *reported deltas on different baselines*
  and must say so;
* §5.5.8 must be discharged. If `s`-tuning alone recovers the 9 points, the sentence changes from
  "the clause-size budget is worth 9 points" to "clause length is worth 9 points, and the budget is
  one of two dials that buy it" — which is a weaker and still publishable claim, but a different one.

#### 5.5.8 The open risk I am registering against our own headline: `s` may buy the same points

`s = 10` at therm-4 / 4×4 is **our own choice** — `ctm-small` has no paper to be faithful to
`[FACT: code/arms.py, register("ctm-small", …), config_status="own"]` — while the Toolbox uses
`s = 5.0` for every colour-thermometer specialist `[FACT: arXiv:2406.00704 Table III]`. By C5.1 the
admission threshold at `s = 10` is 0.091 and at `s = 5` it is 0.167, so **half of the measured bloat
may be an untuned `s` rather than a missing mechanism**. The repository's own prior note says as much
in the opposite direction (`MEMORY.md → tsetlin-clause-budget-batched`: *"use `s` instead"*), and it
was written before any of these measurements.

`[HYPOTHESIS]` The budget is **not** substitutable by `s`, because `s` sets a threshold on literal
*frequency* while the cap sets a quota on literal *count*: at `s = 2` the threshold `π > 1/3` also
discards the rare-but-discriminative literals that the capped clause keeps (it is a *low-pass* filter
on `π`, while the cap is a *top-b* selection), so the `s`-shortened clause should be shorter **and
less accurate** than the budget-shortened clause of the same length. **Falsifier**: the `s = 2`
unconstrained arm reaches within 1.00 pp of the `s = 10` budget-32 arm at a comparable clause length.
This is the single most decision-relevant cheap measurement I can name right now, it is Block 2 of
`jobs/p3_budget_predictions.txt`, and it costs **1.25 GPU-h**.

#### 5.5.9 Three diagnostics that would close the mechanism, all free

Proposed to the research-engineer (`code/diagnostics.py` is not mine to edit); each reads state that
is already in memory and none needs a re-run beyond the arms already queued.

| id | diagnostic | tests | cost |
|---|---|---|---|
| **D-TA** | Histogram of `ta_state` over **included** literals, summarised as `frac_at_boundary` (state `== N`) and `frac_deep` (state `≥ N + N/2`) | C5.3's bimodality and the fringe-stripping account. Predicted: unconstrained shows a large boundary mass; budget 32 shows almost none | one `bincount` on `(C, 2F)`; ~0 |
| **D-FIRE** | Per-clause firing rate **conditioned on the clause's Type-I-eligible class** — `P(fire \| y = c)` for a positive-polarity clause of class `c`, `P(fire \| y ≠ c)` for a negative one | C5.2's prediction that the unconstrained equilibrium pins this at `1/(1+s)`. The existing global rate mixes the two polarities and cannot test it | a class mask inside `firing_stats`; ~0 |
| **D-ORDER** | Marginal selectivity by inclusion depth: sort a clause's included literals by TA state and report the match probability of the top-`k` prefix for `k ∈ {8, 16, 32, all}` | the claim that late arrivals are logically redundant (§5.5.3 shows it in aggregate over epochs; this shows it within a single trained clause) | one masked `clause_outputs` per `k` on the probe set; small |

---

## 6. Receptive field

### 6.1 Definitions

* **Clause receptive field** = `k_h × k_w` pixels of `Z` planes. Exactly, with no growth mechanism.
* **Clause spatial support** = the rectangle `R_j = [y_lo,y_hi] × [x_lo,x_hi]` in patch-grid units
  implied by its position literals `[FACT: conv.py:203-228]`. This constrains *where* the k×k pattern
  may sit; it does **not** enlarge what the clause sees.
* **Model receptive field** = the union over clauses = the whole image. This is the number usually
  quoted and it is the uninformative one: what matters for §3 is the *clause* receptive field,
  because that is the only scope in which a conjunction can be formed.

### 6.2 The cost of growing `k` (H = W = 32, stride 1)

| quantity | scaling | k=3 | k=4 | k=8 | k=10 | k=16 | k=32 |
|---|---|---|---|---|---|---|---|
| patches `P = (33−k)²` | quadratic ↓ | 900 | 841 | 625 | 529 | 289 | 1 |
| features `F = Zk² + 2(32−k)` (Z=24) | quadratic ↑ | 274 | 440 | 1 584 | 2 444 | 6 176 | 24 576 |
| automata `C·2F` | **quadratic in k** | 1.0× | 1.6× | 5.8× | 8.9× | 22.5× | 89.7× |
| compute `C·P·2F` | `∝ k²(33−k)²` | 0.92× | 1.00× | 2.97× | 3.93× | 5.50× | 0.08× |

`[FACT: conv.py:66-80]` + arithmetic. Two non-obvious readings:

* **Compute peaks near `k ≈ 16`, not at `k = 32`.** The whole-image "window" (`P=1`, i.e. a flat TM)
  is the *cheapest* configuration by a factor of ~70 — which is why the HOG specialist, whose window
  is 32×32, is the one with the flat clause-scaling curve and the lowest cost.
* **Statistics degrade faster than compute.** A clause must find a conjunction that is *frequent*
  among patches. The patch value space is `2^{Z·k²}` while the supply of patches is `N·P` and
  *falls* with `k`: at Z=24, k=4 gives 3.8×10⁷ training patches over a `2^384` space; k=10 gives
  2.4×10⁷ over `2^2400`. `[HYPOTHESIS]` This, not compute, is why every published CIFAR-10 window is
  in 3–10 and why the one 32×32 window in the literature is applied to an already-pooled descriptor.
  **Falsifier / measurement**: sweep `k ∈ {3,4,5,8,10,16}` at matched **automata** budget (shrink `C`
  as `k` grows so `C·2F` is constant). If accuracy is flat in `k`, the statistical argument is wrong
  and window size is a free parameter. Nobody in the bibliography has run this
  (PLAN §9.2 item 4 / LITERATURE_TM §2.2).

### 6.3 Why depth is the cheap route, and why it is blocked here

In a CNN the receptive field grows **additively** (`+2(k−1)` per 3×3 layer) at cost **linear** in
depth, and each layer's units are graded, so the composition is a genuine hierarchy. In a CTM the
same growth costs quadratically inside one layer, and depth is measured-blocked:

* greedy supervised stacking: `[MEASURED: experiments/mctm/results/flat-auto_seed{0,1,2}]`
  **15.43 ± 1.12%**, against `[MEASURED: .../single-matched_seed*]` **35.14 ± 0.74%** for one layer
  at matched clause budget and `[MEASURED: .../single-rf_seed*]` **36.83 ± 0.91%** at matched
  receptive field;
* random layer 1 beats trained layer 1: `[MEASURED: .../mctm-random_seed*]` **36.94 ± 0.43%**;
* firing-rate calibration is the repair: `[MEASURED: .../mctm-random-calib_seed*]`
  **41.80 ± 0.98%** (`final_test_acc`; PLAN quotes 42.6 as `best_test_acc`);
* credit propagation through clause inclusion fails structurally, not incidentally (PLAN §3.1).

Note also `[MEASURED: .../mctm-nopos_seed{0,1,2}]` **10.00 ± 0.02%** — a 2-layer stack with position
encoding **off** is at chance, while `[MEASURED: .../mctm_seed*]` with it on is 12.56 ± 1.02%. Whatever
the second layer was doing, it was doing it through the position bits.

### 6.4 The four remaining ways to grow the receptive field within one layer

1. **Enlarge `k`** — quadratic in automata, and statistically self-defeating (§6.2).
2. **Stride / dilation** — stride is supported `[FACT: conv.py:58-63, 125-127]`; **dilation is not**,
   although `torch.nn.functional.unfold` accepts it (LG-007). Dilation grows the clause receptive
   field *without* growing `F` — the only free lunch in this list, and it is one call-site away.
3. **Pre-pool the input** — what HOG does, and what CTM-UNet does between blocks. Costs the
   Booleanization ceiling (§4.2).
4. **Multi-scale clause groups** — different `k` for different clause subsets, votes summed. Not
   expressible in one library model (one `patch_size` per model) but expressible as several models
   whose votes are added, which is structurally a *composite* (LITERATURE_TM §2.4). LG-007.

---

## 7. The gaps versus a CNN, stated as things that can be measured

| # | Gap | Precise statement | Measurement | Prediction (falsifier) |
|---|---|---|---|---|
| G1 | **No counting** | Prop. 2(1): the layer sees only `|M_j| > 0` | **done at small scale** — `[MEASURED: results/ctm-small_seed*.json]` fire 0.0698, median given firing **2**, q75 6, p95 24, `P(≥5\|fires)` **0.303**; repeat at ≥8 000 clauses and on a colour-thermometer encoding; plus the object-multiplicity probe (§3.5) | `[HYPOTHESIS]` the *information* gain from counting is only **×1.71** (§5.2 point 4) and the CNN's is ×1.96, so information does not discriminate; the live prediction is that a count pool **acts as density control** and does not beat a density-only control. **Falsified by** `pool-thermo` beating both `pool-or` and `pool-density` by more than the seed band in the §1.4 screen |
| G2 | **No cross-patch relations** | Cor. 2 (§3.4): relative geometry is not a linear threshold over presences | relation probe (A-above-B) at matched budget, CTM vs `cnn-small` | `[HYPOTHESIS]` CTM at chance-plus-ε where the parts do not fit in one window; CNN solves it. **Falsified if** a single-layer CTM exceeds the one-window bound |
| G3 | **No XOR of presences** | Cor. 1 (§3.3) | `make_2d_noisy_xor`-style presence-XOR at matched budget | already `[MEASURED: experiments/mctm]` 75% cap for one layer, 87.9% with a learned layer 1 |
| G4 | **Receptive field is flat and expensive** | §6.2, quadratic automata, falling patch supply | `k ∈ {3,4,5,8,10,16}` at matched automata budget | `[HYPOTHESIS]` accuracy is unimodal in `k` with the optimum moving *down* as `Z` grows. **Falsified if** flat |
| **G5** *(promoted after the crossfire)* | **All-or-nothing activation** | a clause contributes `w_j` or `0`; a CNN unit is graded. The DL expert's decomposition makes this the *expensive* half of binarisation: binary **weights** cost ≈0.2 pp on ImageNet (BWN 56.8 vs FP 56.6) while binary **activations** cost **12.4** — and a clause output is precisely a binary activation | FPTM-style graded clause (A18) or the `pool-sum` arm, screened | `[HYPOTHESIS]` graded matching helps most at *small* clause budgets (it is a clause-compression mechanism: FPTM claims ~400× on F-MNIST `[FACT: arXiv:2508.08350 abstract]`) and shrinks toward zero at 32 000 clauses. **Falsified if** the gain is budget-independent |
| G6 | **Booleanization loss** | the TM never sees the image, only the bits | `cnn-small` vs `cnn-boolean` (P2), then TM across `therm4/therm8/adaptive/hog` | `[HYPOTHESIS]` Booleanization accounts for ≥ half the CTM↔CNN gap. This is PLAN §8 P2's decomposition |
| G7 | **Cross-channel conjunctions are available but may be unused** | a window sees all `Z` planes jointly, so a clause *can* conjoin bits from different channels (this corrects PLAN §9.2 item 8, which says channels are treated independently — that is true of the *encoder*, not of the clause) | fraction of included literals spanning ≥ 2 colour channels; fraction spanning ≥ 2 thermometer levels of the same channel | `[HYPOTHESIS]` most included literals are *within* one channel's thermometer stack (they are the cheapest way to express an intensity range), so genuine colour conjunctions are rare. **Falsified if** ≥ 30% of clauses conjoin ≥ 2 channels |
| G8 | **No credit assignment** | feedback is per-clause, local, stochastic, and never travels between clauses | dead-clause fraction, fraction of clauses that ever change a decision, negation fraction | `[MEASURED: experiments/mctm]` healthy single layers sit at 57–60% negated literals; collapsed layer-2 clauses at 97.4% (PLAN §3.1). The negation fraction is the cheapest health indicator we have |
| G9 | **Batched feedback is not the algorithm** | §8 D1 | batch ∈ {5, 50, 200} at fixed everything else | `[MEASURED: PLAN §3.2]` 10–50 tracks sequential, 200 degrades. Every arm records its batch size |

---

## 8. Where this library differs from the papers — and what our results inherit

| id | Difference | Faithful? | What a result built on it means |
|---|---|---|---|
| **D1** | `feedback_mode="batch"` aggregates a whole mini-batch and commits once `[FACT: base.py:308-321]`; the algorithm in Granmo et al. 2019 §3 and Book Ch. 4 §4.4 is strictly per-example | **No** (a deliberate approximation) | Arms are comparable **only at equal batch size**. Any claim about the *algorithm* rather than *this implementation* needs one `feedback_mode="sequential"` control |
| **D2** | `max_included_literals` gates Type Ia only, leaving Type II ungated `[FACT: functional.py:232-242 vs 253-257]` | **Yes** — CSC-TM changes Type I only `[FACT: arXiv:2301.08190 §2, "Type II Feedback remains the same for both schemes"]`, and names the resulting boundary literals itself | **Corrected 2026-09-21.** This row previously read "the non-binding budget is an interaction of D1 with convolution … use `s` instead". That was derived from a budget-8, few-epoch probe and does **not** generalise: at convergence the budget binds tightly and mode-independently — `[MEASURED: ctm-small-T80-b32_seed*]` 32.15 ± 0.08 literals at batch 50, `[MEASURED: ctm-small-seq-b32_seed*]` 31.91 ± 0.07 sequential — and is worth **+9.46 / +8.75 pp** (§5.5). Type II's ungated inflow is what produces the residual fringe (§5.5.2 C5.3), not a failure to bind |
| **D3** | Integer additive clause weights clamped at ≥ 0 `[FACT: classifier.py:173-187]` | Matches the *Integer-Weighted* TM used by the CIFAR-10 SOTA; **not** the multiplicative real-valued rule of arXiv:1911.12607 §3.4 | `weighted=True` reproduces the composite recipe. A weight reaching 0 silently removes a clause from the vote; the reference clamps at 1. Worth one diagnostic: fraction of zero-weight clauses |
| **D4** | Independent random patch per feedback kind `[FACT: conv.py:166-171]` | The papers are silent; it is required to be correct for coalesced models | Only affects `ctm-coalesced*` arms |
| **D5** | `init="boundary"` (all TAs at `N−1`, all clauses empty) vs "randomly setting the states" `[FACT: arXiv:1911.12607 §2.3]` | **Partially** — `init="random"` is available | Cheap ablation; note that at epoch 0 all clauses fire (§3.6) |
| **D6** | Per-patch clause outputs computed but not exposed `[FACT: conv.py:140-146]` | n/a | Count pooling, spatial diagnostics and stacked CTM blocks need private API or a local re-implementation. **LG-005** |
| **D7** | `max_chunk_elements=2**27` default vs a conv per-example cost of `3PC + 2C·2F + P·2F`, whose `2C·2F` term is independent of batch size and of `P` `[FACT: conv.py:106-111]` | n/a | `[MEASURED: LIBRARY_GAPS LG-004, code/repro/lg004.py]` at 10×10 patches and batch 50 the library default gives chunk size **24** at 640 clauses, **9** at 2 000, **2** at 8 000 and **1** at 32 000 — i.e. a mini-batch of 50 runs as 25 sequential chunks at the `ctm-vanilla` configuration, on a card with 23 GB free. **LG-004.** A large arm that looks slow is chunk-budget-suspect before it is algorithm-suspect (CHARTER failure mode 3); arms are only compared at equal `max_chunk_elements` |
| **D8** | No dilation; one `patch_size` per model `[FACT: conv.py:125-127]` | n/a | Blocks the cheapest receptive-field mechanism (§6.4). **LG-007** |

---

## 9. The one-paragraph summary for the report

A convolutional Tsetlin machine is a linear classifier over a learned Boolean dictionary of
*existential conjunctive patch predicates* — "does this exact pattern of bits occur anywhere in this
rectangular region?". Everything it can express is a weighted sum of such presences; everything it
cannot express follows from the fact that a presence is one bit. It cannot count occurrences, it
cannot relate two occurrences to each other, and it cannot compute any non-linearly-separable
function of two presences — and no clause budget changes this, because more clauses only add more
coordinates of the same kind. Buying clauses does help, but only logarithmically: the published
curve for the best Booleanization gains **+2.18 points per doubling** and is still rising at 64 000
clauses, which puts parity with a small CNN at roughly 3 × 10⁶ clauses and 2 000 GPU-hours per seed
(§4.3). The road is open and impassable, which is why the quantity to report is the *slope* of
accuracy against `log₂(clauses)` and the bar for a new mechanism is that it beats two doublings at
matched automata budget. The only mechanism the construction offers for escaping these limits
is to enlarge the window until the interacting parts fall inside one patch, which costs automata
quadratically and starves the pattern statistics at the same time. Depth would be the cheap escape,
and in this repository it is measured-blocked. That is the representational shape of the problem
this programme is attacking. Cutting across all of it is a *learning* rather than representational
defect that we measure at nine accuracy points: literal inclusion is a **ratchet** — a literal is
admitted whenever it is true in more than `1/(1+s)` of the clause's own matching patches, which never
asks whether it is useful, and the only path that removes it runs through the clause's firing rate at
gain `1/s`. An unconstrained clause therefore accumulates logically redundant literals indefinitely
(+25% length over the last 25 epochs with no change in match probability, firing rate or accuracy),
and a hard clause-size budget — the only mechanism in the algorithm whose activation depends on
clause length — is worth **+9.46 pp** here against the **+0.06 pp** its own authors reported (§5.5).
