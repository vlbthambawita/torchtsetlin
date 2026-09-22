# MECHANISMS.md — what makes a CNN work, and what the Tsetlin analogue would have to be

**Owner**: dl-expert · **Phase**: P1 · **Status**: complete for Gate G1 · **Date**: 2026-09-20

This is the raw material for P5 candidate generation. Each entry states **(a)** what the component
does, **(b)** *why* it helps mechanistically — not by analogy, **(c)** whether a Tsetlin machine has
an analogue today, **(d)** what the Tsetlin analogue would have to be, **(e)** the experiment that
would test whether it transfers, and ends with a **falsifiable prediction** and **the cheapest
falsifier**.

**Constraint C3 governs this whole document**: a CNN mechanism does not transfer to a TM because it
is analogous. It transfers if and only if a TM experiment says so. Everything in column (d) is
`[HYPOTHESIS]` until a run says otherwise.

---

## 0. The two anchors this document is built on

**Anchor 1 — this translation exercise has already paid off once.**
[FACT: PLAN.md §3.1, from `experiments/mctm/report2/`] Firing-rate calibration — a label-free
controller walking each clause to a target per-patch firing rate — took a 2-layer stack from
**15.4 %** (greedy supervised) to **42.6 ± 0.7 %**, past both single-layer baselines at matched
clause budget and matched receptive field (35.4 % / 37.8 %). Structurally that controller is a
**BatchNorm analogue**: it holds a unit's activation statistics in a workable band. It was not
derived from BatchNorm, but it is the same mechanism, and it is the single largest measured
improvement in this repository's prior programme. §5 below takes it further.

**Anchor 2 — one whole family of translations is closed.**
[FACT: PLAN.md §3.1] **Credit propagation through clause inclusion does not work**, for four
*structural* reasons, not implementation bugs:
1. there is no forgetting term in the propagated signal;
2. Type Ib must be **blamed** rather than sampled;
3. nothing sets the Ia:Ib ratio;
4. clause identity drifts under the layer above — layer-2 clauses are rules about channel
   identities that are being rewritten underneath them.

**I do not re-propose it.** Where a mechanism below could be mistaken for a variant of it (§6
residual connections, §10 soft targets), the entry states explicitly which of the four points it
escapes and how. Any proposal that does not name its escape route should be rejected on sight.

### 0.1 The CTM, stated precisely, so the analogies are not hand-waving

From `src/torchtsetlin/models/conv.py` (READ-ONLY; quoted, not modified):

```
literals(p)   = [ patch_p bits , 1 - patch_p bits ]            (+ thermometer position bits)
clause_j(x)   = OR over patches p of  AND over included literals l  literals_l(p)
class score c = sum over j of  w_jc * clause_j(x)              (w in {+1,-1} or learned)
feedback      = Type I (Ia reinforce true literals; Ib forget with prob 1/s) and Type II,
                each event updating the clause from ONE uniformly random MATCHING patch,
                with probability (T - clip(vote, -T, T)) / 2T
```

Three consequences used repeatedly below:
* `matches.any(dim=1)` is **exactly a max over binary indicators** — the CTM's patch aggregation
  *is* max-pooling, and the CTM has **no average-pooling analogue at all**.
* The classifier sees **one bit per clause per image**. With `P = 841` patches (4×4, stride 1,
  32×32) it is discarding `log2(842) ≈ 9.7` bits of information per clause.
* `s` is a forgetting rate (weight decay), `T` is a vote-margin at which feedback switches off
  (an example-weighting / loss-temperature mechanism). **Neither is ever scheduled.**

---

## 1. Hierarchical composition / depth

**(a)** Stack layers so layer *k* computes functions of layer *k−1*'s features.

**(b) Why it helps.** Two separable reasons, and only the second is about expressivity.
*Expressivity*: depth buys representational power exponentially in depth for the same width — a
depth-*k* net needs width that a depth-1 net must pay exponentially for. *Feature reuse*: images are
compositional (edges → textures → parts → objects) and a mid-level feature is shared by many
high-level concepts, so depth **amortises** feature cost across classes. The second is the one that
actually matters on CIFAR-10 at these budgets.
The counter-evidence is important: depth is not free. [FACT: arXiv:1512.03385 Table 6] ResNet-1202
(19.4 M) scores **7.93 %** error, *worse* than ResNet-110 (1.7 M) at 6.43 %; and [FACT: same, Fig. 6
left] a plain (non-residual) 110-layer net exceeds **60 %** error. Depth without the mechanisms in
§5 and §6 makes networks worse.

**(c) TM analogue today.** Present as an architecture (`experiments/mctm`), and it **fails** in the
greedy supervised form: [FACT: PLAN §3.1] 2-layer stack **15.4 %** vs **34.4 %** for its own frozen
layer 1 alone. The measured cause is density collapse: trained layer-1 clauses fire on **0.14 %** of
patch positions, so layer-2 clauses become **97.4 % negated literals** and assert only absences.

**(d) What it would have to be.** The failure is not about depth; it is about *signal propagation*,
and the DL name for it is variance collapse. A layer-2 clause is a conjunction over layer-1 output
bits; a conjunction of *m* independent bits each firing with probability *p* fires with probability
*p^m*. At *p* = 0.0014, a 2-literal layer-2 clause fires 2×10⁻⁶ of the time — dead at
initialisation, before any learning happens. **This is precisely the problem BatchNorm and careful
initialisation solve in a CNN**, and the calibration result (Anchor 1) is the confirmation that it
is the right diagnosis. Depth in a TM therefore requires three things simultaneously: (i) layer-1
firing rate held in a workable band (§5), (ii) stable layer-1 identity, which means **freezing**
[FACT: PLAN §3.1 — "Freezing is what makes 'channel k' mean something"], and (iii) a way for depth
never to be worse than width (§6).

**(e) Experiment.** `mctm-depth-probe`: 2- and 3-layer calibrated stacks with online rate control,
at matched **automata** budget against a single wide layer; log per-layer firing rate, clause length
and negation fraction every epoch. This is P4.5.

> **Prediction [HYPOTHESIS]**: with online rate control (§5) *and* skip literals (§6), a 2-layer
> stack beats a single layer at matched automata budget by ≥ 3 points; without skip literals it does
> not reliably beat it, and the failure shows up as layer-2 negation fraction > 80 %.
> **Cheapest falsifier**: one calibrated 2-layer screen run (10 k images, 15 epochs) vs a
> matched-budget single layer. ≈ 0.3 GPU-h. If the negation fraction is healthy (55–65 %) and the
> stack *still* loses, density is not the blocker and §5 is the wrong lever.

---

## 2. Weight sharing

**(a)** The same filter is applied at every spatial position.

**(b) Why it helps.** It is usually described as parameter saving; the parameter saving is the
*less* important half. The important half is **statistical**: each filter receives training signal
from every position of every image, so its effective sample size is multiplied by the number of
positions (≈ 841 here). It also constrains the hypothesis class to translation-equivariant
functions, which is a correct and very strong prior for natural images.

**(c) TM analogue today.** **Fully present.** The CTM evaluates each clause at every patch and
updates it from a matching patch — that is weight sharing plus a stochastic positional sample of the
update. The CTM is *not* behind CNNs on this mechanism.
Two caveats. First, `position_encoding=True` **partially destroys it**: a clause may include `y>k` /
`x>k` literals and pin itself to a location. [FACT: repo `CLAUDE.md`] on translation-invariant toys
accuracy oscillates 0.5–0.9 with it on and hits 1.0 with it off. Second, and unexamined anywhere:
a CNN accumulates gradient from **all** positions; the CTM updates from **one** uniformly random
matching patch per feedback event (`conv.py::_feedback_counts`).

**(d) What it would have to be.** The patch draw is a **variance-vs-fidelity** choice that nobody in
the TM literature has priced. A CNN's per-filter update is the *sum* over positions; the CTM's is a
1-sample Monte-Carlo estimate of the same thing. Options: update from all matching patches with the
feedback probability divided by the match count; or from the top-*k*; or keep one draw but raise the
feedback probability. [FACT: repo `CLAUDE.md`] the current design is deliberate — drawing in
`_evaluate` instead costs ~5× prediction time and forces Type Ia and Type II to share a patch — so
the change must stay inside `_feedback_counts`.

**(e) Experiment.** `ctm-patchdraw-{one,all,topk}` at fixed clause budget and fixed epochs, measuring
accuracy **per epoch** and per wall-clock second (an all-patch update is more work per event).

> **Prediction [HYPOTHESIS]**: all-patch updating reaches a given accuracy in fewer epochs but not
> in less wall-clock time, i.e. it is a variance-reduction with no accuracy ceiling change (≤ 1
> point at convergence).
> **Cheapest falsifier**: two screen runs, ≈ 0.4 GPU-h. A > 2-point difference at convergence means
> the 1-sample draw is a real accuracy bottleneck and this jumps up the ranking.

---

## 3. Stride and pooling — max vs average

### This is the largest identified mechanism gap in the CTM.

**(a)** Pooling aggregates a spatial neighbourhood: max takes the strongest response, average takes
the mean; strided convolution subsamples.

**(b) Why it helps.** Three distinct jobs, and they are usually conflated.
*Invariance*: pooling makes the representation insensitive to small translations.
*Receptive-field economics*: halving resolution doubles the RF of every later layer for free (§4).
*Readout semantics*: **max is a detector** ("did this feature occur anywhere?"); **average is a
counter** ("how much of this feature is present?"). These are different functions of the same
feature map, and they carry different amounts of information. It is not an accident that every
modern CIFAR-10 architecture — ResNet, WRN, DenseNet, All-CNN, NiN — ends in a **global *average*
pool** before the classifier, not a global max pool. [FACT: arXiv:1312.4400] Network-in-Network
introduced global average pooling precisely as the classifier readout, and every high-accuracy
architecture since has kept it.

**(c) TM analogue today.** The CTM has **max pooling and nothing else**.
`conv.py::_evaluate` returns `matches.any(dim=1)`, which is exactly a max over binary indicators,
i.e. an OR. The vote is then a linear threshold over *indicators*.
**The CTM has no average-pooling analogue at all.** This is a complete absence, not a weak version.

**(d) What it would have to be.** Replace the indicator with a **count**:

```
c_j(x) = |{ p : clause_j matches patch p }|          instead of   [c_j(x) > 0]
score  = sum_j w_jc * f(c_j(x))                       f = identity, min(.,cap), or a threshold
```

Two variants worth separating:
* **`ctm-count-vote` (soft, global-average-pool analogue)**: the score is a linear function of
  counts. Needs **no new automata** — only `_votes` and the clause-output path change. Feedback is
  unchanged; only the vote magnitude, and hence the feedback probability through `T`, changes.
* **`ctm-count-pool` (hard, "fires iff ≥ k patches match")**: a per-clause threshold *k*, which is a
  new learnable quantity and a new update rule. More expressive, more expensive, more to get wrong.

**The information argument, which is why this ranks first.** With *C* clauses the OR pool hands the
classifier **C bits** per image. The counting pool hands it up to **C · log₂(P+1) ≈ 9.7 C bits** for
P = 841. That is a ~10× increase in the information reaching the vote **at identical automata
budget, identical clause semantics and identical interpretability** — the clause is still a readable
conjunction; only how often it fired is now reported instead of whether it fired.

**Why a CTM should be *more* sensitive to this than a CNN.** A CNN's max-pool sits over a small
window (2×2) and it keeps a real-valued magnitude; the CTM's OR sits over **all 841 positions** and
keeps one bit. The bottleneck is far tighter.

**(e) Experiment.** `ctm-count-vote` vs `ctm-vanilla` at **three** clause budgets (e.g. 1 k / 4 k /
16 k), 3 seeds, matched automata, same `s`, same `T`, same batch size.

> **Prediction [HYPOTHESIS]**: (i) `ctm-count-vote` beats `ctm-vanilla` at every clause budget;
> (ii) **the gap widens as the clause budget shrinks**, because the C-bit bottleneck binds harder
> when C is small; (iii) the effect is ≥ 5 points at 1 k clauses.
> (ii) is the load-bearing part — a mechanism that only helps at large budgets is a capacity effect
> in disguise, not an information effect.
> **Cheapest falsifier**: two screen runs at 1 k clauses (10 k images, 15 epochs). ≈ 0.3 GPU-h. If
> the small-budget gap is inside the screen noise band, the information argument is wrong.

**Supporting evidence from the CNN side, with its provenance stated honestly.** A single-conv-layer
CNN with exactly the CTM's shape (2 000 filters, 4×4 patches, stride 1, therm4 Boolean input, global
pool over patch positions, linear vote — `code/cnn/models.py::CtmShaped`), at the P5 screen scale
(10 000 training images, 15 epochs, 1 seed, no augmentation, validation-selected):

| pool over the 841 patch positions | test acc |
|---|---|
| `max` — the CTM's OR | 38.9 % |
| `mean` — the counting pool | **51.0 %** |

**Provenance and status**: run today on this machine as a code-health check; records are in the
session scratchpad, **not** in `results/` or `screen/`, and are therefore **not admissible** and are
**not tagged `[MEASURED]`**. Recommendation to the orchestrator: register these two as proper screen
records in P5 (`screen/cnn-ctmshape-{max,sum}.json`) so the argument can cite them.
**And note the confound**: on a CNN, mean-pooling also changes the *optimisation* (max-pooling gives
gradient to one position only). A TM has no gradient, so the CNN result supports the **information**
half of the argument and not the optimisation half. This is exactly why C3 exists: the 12-point CNN
gap is a reason to run the TM experiment, never a substitute for it.

---

## 4. Receptive-field growth

**(a)** How much of the input one unit sees.

**(b) Why it helps.** A unit cannot classify what it cannot see. The economics are the point: *L*
stacked 3×3 stride-1 layers give RF = 1 + 2L at **O(L · 9 · C²)** parameters; a single layer with
the same RF costs **O(RF² · C · C_in)** — quadratic in RF. Stride/pooling makes it multiplicative:
each 2× downsample doubles every later layer's RF for free. **Depth is the cheap way to buy
receptive field, and that is most of why CNNs are deep.**

**(c) TM analogue today.** A CTM grows RF **only** by enlarging the patch. Literals per patch =
`2 · Z · kh · kw` (therm4, 4×4 → 384; 10×10 → 2 400), so automata grow linearly in *area* while the
conjunction search space grows as 3^(Z·kh·kw) — exponentially. And it costs wall-clock:
[FACT: PLAN §3.4] 640 clauses go from **4.6 s/epoch** at 4×4 to **11.2 s/epoch** at 9×9 (2.4×).
The CTM is paying the expensive way for the thing CNNs buy cheaply.

**(d) What it would have to be.** Three options, in increasing cost:
* **Dilated patches** — sample a k×k grid of cells with spacing *d*. RF grows as `d·(k−1)+1` at
  **zero** literal-count and **zero** automata cost. A 4×4 patch at d = 2 has a 7×7 receptive field
  with the same 384 literals. `torch.nn.functional.unfold` takes a `dilation` argument natively, so
  this is a change to `_encode` and nothing else. **Nothing in the TM literature (PLAN §9.1) does
  this.**
* **Multi-scale / pyramid patch banks** — several clause pools at different (k, d), sharing one
  vote. Matches the CNN's multi-scale feature hierarchy without needing depth.
* **Depth** (§1), which is the CNN's own answer and is currently blocked.

**(e) Experiment.** `ctm-dilated-d{1,2,3}` at fixed 4×4 patch and fixed clause budget, plus
`ctm-vanilla-7x7` and `ctm-vanilla-10x10` as the dense-RF controls, all at matched **automata**
budget and matched wall-clock reported separately.

> **Prediction [HYPOTHESIS]**: 4×4 at d = 2 (RF 7×7, 384 literals) beats dense 4×4 by ≥ 2 points and
> lands within 1 point of dense 7×7 (1 176 literals) — i.e. **⅓ the automata for the same
> accuracy**. A multi-scale bank {4×4 d1, 4×4 d2, 4×4 d3} beats all single-scale variants at matched
> total automata.
> **Cheapest falsifier**: three screen runs (d = 1, 2, 3) at one clause budget. ≈ 0.4 GPU-h. If d = 2
> does not beat d = 1, receptive field is not what limits the CTM at 4×4 and §4 drops out of the
> ranking.

---

## 5. Normalisation (batch / layer norm)

**(a)** Standardise each unit's pre-activation across the batch (BN) or across features (LN), then
rescale by learned γ, β.

**(b) Why it helps.** Three mechanisms, and separating them is what makes the TM translation
possible:
1. **Signal propagation** — keeps activation variance near 1 through depth, so forward and backward
   signals neither explode nor vanish. This is the one that makes deep nets trainable at all.
2. **Landscape smoothing / effective learning rate** — BN makes the loss and its gradients more
   Lipschitz, and the scale-invariance it induces gives an automatic learning-rate decay.
3. **Regularisation** from batch noise.

**(c) TM analogue today.** **Yes, and it is the best-validated translation we have.**
[FACT: PLAN §3.1] firing-rate calibration over an *untrained* layer 1 at a 20 % target reaches
**42.6 ± 0.7 %**, versus 15.4 % for greedy supervised stacking and 35.4/37.8 % for single layers at
matched budget/RF. That controller is mechanism (1) exactly: the TM's "activation statistic" is the
per-patch **firing rate**, and its knob is **clause length** (more included literals → lower firing
rate).
[FACT: memory `mctm-density-calibration-fix`] calibrating over an *untrained* layer 1 is what works;
over a *trained* layer 1 the same controller reaches only 24.5 % (35.6 % with a clause-size cap).
There is also a hidden partial analogue of mechanism (2): the feedback probability
`(T − clip(v,−T,T)) / 2T` normalises the credit signal by the current class vote.

**(d) What it would have to be, going beyond the existing repair.** Three extensions, none tried:
1. **Online rate control** — PLAN §9.2(3). BatchNorm runs at *every step*, not once before freezing.
   The existing controller runs once, before layer 1 is frozen. Running it every epoch on a
   *trainable* layer is the faithful translation. This also **fixes a known library failure**:
   [FACT: PLAN §3.2 / CHARTER silent-failure 2] `max_included_literals` does not bind for conv
   models under batched feedback (Type II is ungated, one opportunity per patch; budget 8 → median
   clause size **75** at batch 50). A rate controller is the principled replacement for the L0
   budget that does not work.
2. **Single-layer rate control** — the cheapest possible test, and a sharp one: if rate control only
   helps stacks, it is a depth repair; if it helps a single layer too, it is a general mechanism and
   belongs in the library.
3. **Vote normalisation** — the unexplored one. In a coalesced TM all classes share a clause pool; a
   clause firing on 60 % of images dominates every class's vote regardless of how informative it is.
   Scaling each clause's vote contribution by `1/sqrt(firing_rate)` is literally what BN's rescaling
   does to a feature, and nothing in the TM literature does it.

**(e) Experiment.** `ctm-rate-online` (single layer, controller every epoch, target rates 5/10/20 %)
vs `ctm-vanilla` at matched clause budget, logging median clause length and firing rate per epoch.

> **Prediction [HYPOTHESIS]**: online rate control raises **single-layer** CTM accuracy by ≥ 2 points
> at fixed clause budget, and cuts the across-seed standard deviation by ≥ 30 %, by preventing the
> clause-length drift that `max_included_literals` fails to bind. The optimal target rate is between
> 5 % and 20 % and is *lower* than the 20 % that was optimal for the stacked layer-1 case.
> **Cheapest falsifier**: three screen runs at three target rates vs one vanilla screen.
> ≈ 0.5 GPU-h. No single-layer gain → the mechanism is depth-specific, which is still a finding, and
> it moves to §1's experiment instead of standing alone.

---

## 6. Residual connections

**(a)** `y = F(x) + x` — an identity path around a block.

**(b) Why it helps.** Four effects, in decreasing order of how well established they are.
1. **The default of an added layer becomes "do nothing".** A deeper net can represent everything a
   shallower one can, by driving F → 0. This is the direct fix for the *degradation problem*:
   [FACT: arXiv:1512.03385 §4.1, Fig. 6 left] deeper plain nets have higher **training** error —
   they are not overfitting, they are failing to optimise. Residuals remove that.
2. **Gradient highway** — the backward signal reaches early layers without passing through weight
   matrices.
3. **Implicit ensemble of shallower paths.**
4. **A much smoother loss landscape.**

**(c) TM analogue today.** **None.** There is no addition in a TM; there is no "identity conjunction"
over the previous layer that preserves information. The measured consequence is exactly the
degradation problem in TM form: [FACT: PLAN §3.1] a 2-layer stack scores **15.4 %** while **its own
frozen layer 1 alone** scores **34.4 %**. A stack that cannot fall back on its own first layer is a
plain net without shortcuts.

**(d) What it would have to be — and why it is not the thing that already failed.**
The TM's identity path is **literal-level concatenation**: give layer-2 clauses a literal vector of

```
[ layer-1 clause output bits , the raw (down-sampled) patch bits ]
```

A layer-2 clause may then include *only raw literals* and be, exactly, a layer-1 clause. Depth
becomes **structurally unable to be worse than width**, which is the entire point of a residual
connection, achieved without any arithmetic.

**Which of Anchor 2's four failure points this escapes, and how.** All four — and by the same
reason: **this is a change to the forward path only. No credit is propagated downward at any point.**
1. *No forgetting term* — nothing is propagated, so there is no signal that needs one. Layer 2's own
   Type Ib forgetting is unchanged and still applies to whichever literals it included, raw or not.
2. *Type Ib must be blamed rather than sampled* — no Type Ib is ever assigned to layer 1; layer 1
   receives no feedback at all.
3. *Nothing sets the Ia:Ib ratio* — there is no downward feedback whose ratio would need setting.
4. *Clause identity drifts under the layer above* — layer 1 stays **frozen**, so its identities are
   fixed; and the raw literals have permanently fixed identity by construction. The skip path
   strictly *strengthens* the freezing argument rather than weakening it.

If a reader thinks this is credit propagation in disguise, the test is simple: no `apply_feedback`
call ever touches layer 1's `ta_state`. If a proposed variant cannot pass that test, it is the thing
that already failed.

**(e) Experiment.** `mctm-skip`: 2-layer stack, layer 1 frozen (random or autoencoder-initialised,
rate-calibrated), layer 2 over `concat(layer-1 bits, raw patch bits)`, at matched **automata** budget
against (i) a single layer and (ii) the same stack without skip literals.

> **Prediction [HYPOTHESIS]**: the skip stack is **never worse** than its own layer 1 — the 34.4 →
> 15.4 collapse disappears entirely — and it beats a matched-automata single layer by ≥ 2 points.
> The diagnostic signature is the fraction of layer-2 included literals that are raw: predicted to
> start near 1.0 and fall as layer 1 becomes useful.
> **Cheapest falsifier**: one screen run of `mctm-skip` vs its own frozen layer 1. ≈ 0.4 GPU-h. If
> the skip stack is *still* worse than its own layer 1, depth in a TM is blocked by something
> neither density nor fallback explains, and the programme should stop buying depth and say so.
> **Feasibility flag for the research-engineer**: this needs layer 2's literal vector to be a
> concatenation of two differently shaped tensors. Whether `torchtsetlin` can express that today
> without touching `src/` is an open question for `rounds/ROUND-1/research-engineer.md`; if not, it
> is a `LIBRARY_GAPS.md` entry, not a patch (C2).

---

## 7. Data augmentation

**(a)** Random crop (pad 4), horizontal flip, and stronger policies, applied per batch.

**(b) Why it helps.** It injects the task's **known invariance group** into the training
distribution, forcing the learned function to be constant on orbits. It is a data-dependent prior,
not extra information. The size of the effect is the point: [FACT: arXiv:1708.04552 Table 1, 5 runs
each] ResNet-18 goes from **10.63 ± 0.26 %** error without augmentation to **4.72 ± 0.21 %** with
crop+flip — **5.9 points from two lines of code**. WRN-28-10: 6.97 → 3.87. [FACT: arXiv:1608.06993
Table 2] ResNet-110: 13.63 → 6.41.

**(c) TM analogue today.** Not used. The TM image literature largely does not augment (PLAN
§9.2.5). Note this is not an analogue problem — it is an omission.

**(d) What it would have to be.** **Literally the same operation.** `code/data.py::_augment_batch`
(reflect-pad 4, random 32×32 crop, horizontal flip) is channel-agnostic and applies unchanged to a
12-plane Boolean tensor. Implementation cost: a flag.
But the *expected* benefit differs from a CNN's, and differs between the two transforms, in a way
that makes a sharp prediction possible:
* **Crop** should buy a CTM **less** than it buys a CNN, because the CTM is *already* translation-
  equivariant by weight sharing and translation-*invariant* by the OR pool over all 841 positions.
  It is already in the CTM's symmetry group, except at the boundary.
* **Flip** is **not** in the CTM's symmetry group at all. A clause is a conjunction over specific
  patch cells and has no mirror-image counterpart. Flip augmentation therefore adds a genuinely new
  invariance and should buy the CTM proportionally **more**.
* **Caveat**: `position_encoding=True` breaks the crop argument, because position literals are not
  translation-invariant. Check this flag before interpreting the result (CHARTER silent-failure 5).

**(e) Experiment.** `ctm-aug-{none,crop,flip,both}` at fixed clause budget, fixed epochs, 3 seeds —
and the same four settings on `cnn-small-noaug` as the CNN-side control, which P2 produces anyway.

> **Prediction [HYPOTHESIS]**: for the CTM, **flip's gain ≥ 2 × crop's gain**; for `cnn-small`, crop's
> gain ≥ flip's gain. The orderings are opposite, and that opposition is the test.
> Absolute size: flip worth 1.5–4 points to the CTM.
> **Cheapest falsifier**: four screen runs, no new code at all. ≈ 0.4 GPU-h. If crop ≥ flip for the
> CTM, my account of the CTM's symmetry group is wrong and §3's invariance reasoning needs revisiting
> too.

---

## 8. Overparameterisation

**(a)** More parameters than training examples.

**(b) Why it helps.** *Optimisation*: sufficient width makes the loss landscape benign, so SGD
reliably finds near-global minima. *Implicit bias*: SGD selects a low-complexity interpolator among
the many that fit, so test error can keep falling past the interpolation threshold. *Lottery
tickets*: more units means more chances some random subnetwork is useful.
And the crucial counterweight: **on CIFAR-10 the returns saturate and then reverse.** [FACT:
arXiv:1512.03385 Table 6] ResNet-1202 at 19.4 M params is *worse* than ResNet-110 at 1.7 M (7.93 vs
6.43). [FACT: arXiv:1608.06993 Table 2] DenseNet-BC-100 at 0.8 M beats ResNet-110 at 1.7 M. Capacity
is not the axis that wins CIFAR-10; architecture is.

**(c) TM analogue today.** Clause count. It is the TM's most reliable knob and the one every paper
turns.

**(d) What it would have to be.** Nothing new — but the *decision* it forces is the most important
one in the programme. PLAN §9.2(9) states it: if the capacity-scaling curve is still climbing at
8 k clauses, the honest next move is more clauses, not a new mechanism.

**(e) Experiment.** P4.1's capacity-scaling curve: accuracy vs clause count over ≥ 4 points per
family, to saturation.

> **Prediction [HYPOTHESIS]**: the single-layer CTM **saturates below 70 %** by roughly 8–16 k
> clauses per class, i.e. CIFAR-10 CTMs are **mechanism-limited, not capacity-limited**.
> *Reasoning*: [FACT: PLAN §3.1] a one-layer CTM's vote is a linear threshold over OR-pooled clause
> bits. Adding clauses adds features to a **linear** model over a fixed Boolean feature family;
> once the features span what that family can express, more of them buy progressively less. The
> saturation point is where the linear-in-indicators bottleneck binds — which is exactly the
> bottleneck §3 proposes to widen.
> **Cheapest falsifier**: the P4.1 curve itself, which is mandatory anyway. **If it is still
> climbing by > 1 point per doubling at 8 k clauses, this prediction is dead, §3–§7 are premature,
> and the report says "buy clauses".** This prediction must be tested **before** the P5 candidates,
> not after.

---

## 9. The optimiser and gradient-based credit assignment

**(a)** Backprop computes ∂L/∂w exactly for every parameter; SGD+momentum follows it.

**(b) Why it helps.** Four separable properties:
1. **Exact, signed, magnitude-carrying credit per parameter** — every weight learns how much *and in
   which direction* it contributed.
2. **Global coordination** — every parameter updates from one shared loss, so units specialise to
   cover each other's errors. This is why CNN filters are diverse with no explicit diversity term.
3. **Continuous states** — infinitesimal updates, so the signal from thousands of examples can be
   averaged before anything moves.
4. **Momentum / adaptive scaling** — smooths the per-batch noise.

**(c) TM analogue today.** Type I / Type II feedback. It has **(2) in part**, and this is
under-appreciated: the feedback probability `(T − clip(v,−T,T))/2T` falls to zero once the class
vote exceeds `T`, so clauses stop updating on examples the *ensemble* already gets right. That is
genuine global coordination, and it is structurally boosting. It has **none of (1)**: credit is a
probability, not a signed magnitude, and it is assigned per *clause*, not per literal. **(3)** is
absent by construction — automata take integer steps.

**(d) What it would have to be.** **Nothing, along the obvious route.** Anchor 2 closes it: credit
propagation through clause inclusion fails for four structural reasons and I do not re-propose it.
What remains open, and does *not* require propagating anything:
* **Schedule the knobs that already exist** (§13) — `s` and `T` are the TM's learning rate and
  loss temperature, and nobody anneals either.
* **Inject gradient-derived knowledge from outside** (§10) — a CNN teacher's soft targets change
  which examples the TM is asked to fit and how confidently, and the `T` mechanism consumes that
  directly, with no change to the feedback rules.
* **Reduce the credit *sampling* variance** (§2) — a question entirely inside the existing framework.

**(e) Experiment.** **The clean isolation of the learning algorithm**, which nothing else in the
programme provides: `cnn-ctmshape-binary` vs a matched `ctm-vanilla`. Same architecture (one conv
layer, global max over patch positions, linear vote), same input (therm4), same receptive field,
comparable capacity — one trained by SGD with a straight-through estimator, one by Type I/II
feedback. **The difference is the learning algorithm and nothing else.**

> **Prediction [HYPOTHESIS]**: the SGD-trained binary CTM-shaped net beats a matched CTM by 10–20
> points, and beats it by *more* at small filter/clause counts (because gradient credit assignment
> uses each unit better).
> **Cheapest falsifier**: one `cnn-ctmshape-binary` screen vs one `ctm-vanilla` screen at matched
> filters/clauses. ≈ 0.5 GPU-h. A gap inside the seed band would mean the CTM's learning algorithm
> is **not** the bottleneck and the whole gap is architectural — which would redirect the programme
> entirely, and is the most valuable possible negative result.

---

## 10. The loss and soft targets

**(a)** Cross-entropy on one-hot labels; label smoothing; distillation on a teacher's softened
softmax.

**(b) Why it helps.** Three mechanisms:
1. **Automatic example weighting** — cross-entropy's gradient is `(p − y)`: large when the model is
   wrong, vanishing when it is confidently right. Training effort goes where the error is.
2. **Dark knowledge** — a teacher's soft target carries the *relative similarity structure* over
   classes ("this cat is 5 % dog, 0.01 % truck"). That is far more bits per example than one hard
   label, which is why distillation works with less data.
3. **Calibration** — label smoothing stops logits growing without bound and tightens class clusters.

**(c) TM analogue today.** The TM has no loss — but it already has **(1), exactly**. The feedback
probability `(T − clip(v,−T,T))/2T` is large when the class vote is wrong and zero once it exceeds
`T`. The TM implements cross-entropy's example weighting in a different algebra. What it entirely
lacks is **(2)** and **(3)**: the target is a hard class and there is no representation of "30 %
cat".

**(d) What it would have to be.** A **per-example, per-class target margin**. Replace the global
constant `T` with

```
T_c(x) = T * g( teacher_prob_c(x) )          g monotone, g(1)=1
```

so clauses for class *c* stop receiving feedback once their vote reaches the teacher's confidence,
not a constant. **This requires no change to the feedback rules at all** — only to how
`functional.feedback_probabilities` computes its threshold.
**Which Anchor-2 failure points this escapes**: all four, and trivially — it propagates nothing.
It changes a scalar threshold per (example, class). The feedback types, their ratio, the forgetting
term and clause identity are all untouched.
**Interpretability**: clause structure is unchanged; only which examples drive feedback changes. So
this is one of the few high-upside mechanisms that costs nothing on PLAN §2's interpretability axis.

**(e) Experiment.** `ctm-soft-T` with a `cnn-small` teacher (P2 produces it anyway), vs
`ctm-vanilla` at matched clause budget, **at both 50 k and 1 k/5 k training images**.

> **Prediction [HYPOTHESIS]**: +2–4 points at 50 k, and **a larger gain at 1 k–5 k**, because soft
> targets carry more bits per example and the low-data regime is bit-starved. The gain at 1 k is at
> least twice the gain at 50 k.
> **Cheapest falsifier**: two screen runs at 10 k plus two at 1 k. ≈ 0.5 GPU-h once a teacher exists.
> No gain at **either** end → the TM cannot use graded targets and the whole distillation branch
> (PLAN §9.1 `ctm-distill`) can be dropped, saving P3 budget.

---

## 11. Input representation — continuous vs Booleanized

**(a)** A CNN's first layer sees a real-valued 3-channel image; the TM sees `Z` Boolean planes.

**(b) Why the CNN's version helps.** Two things a thermometer code cannot do. **Resolution**:
thermometer-4 gives 5 distinguishable levels per channel. **Cross-channel combination**: a CNN's
first-layer filter is a *learned linear combination across R, G, B*, so "yellowish" is one feature.
A TM literal is one plane at one pixel; "yellowish" must be built as a conjunction of several
per-channel threshold literals, spending automata to express something a CNN gets in its first
layer's weights.

**(c) TM analogue today.** The Booleanization registry (`therm4`, `therm8`, `adaptive`, and HOG in
the TM Composites line). The channels are treated independently by construction.

**(d) What it would have to be.** **Colour-prototype planes**: quantise colours to *K* learned
prototypes (k-means on pixels, label-free) and emit *K* one-hot planes. A single literal then means
"this pixel is in colour cluster k", i.e. a cross-channel concept, at `Z = K` instead of `Z = 3·bits`.
Plus, per PLAN §9.2(8), explicit joint colour literals.

**(e) Experiment.** The CNN side isolates the total cost: `cnn-boolean` vs `cnn-small-noaug`,
identical architecture and recipe, only the input differing — **this number does not exist anywhere
in either literature**. Then `cnn-boolean-therm8` says whether more bits recover it, and on the TM
side `ctm-boolean-{therm4,therm8,colour16,hog}` says how much the TM recovers.

> **Prediction [HYPOTHESIS]**: Booleanization to therm4 costs a CNN **3–7 points** (`cnn-small-noaug`
> minus `cnn-boolean`); therm8 recovers **less than one third** of that, because the loss is mostly
> the *cross-channel* restriction, not the *resolution* restriction; colour-prototype planes at
> K = 16 recover more than therm8 does at Z = 24.
> **Cheapest falsifier**: `cnn-boolean` and `cnn-boolean-therm8` — both are already in P2 and cost
> ≈ 1 GPU-h together. If therm8 ≈ float, the loss is resolution, the cross-channel story is wrong,
> and the fix is simply more thermometer bits.

---

## 12. Weight decay, sparsity and explicit capacity control

**(a)** L2 decay shrinks unused weights; dropout/drop-path add noise; L0/pruning removes units.

**(b) Why it helps.** L2 keeps the effective capacity below the nominal capacity so that
overparameterisation (§8) buys optimisation without buying overfitting. In the no-augmentation
regime it is decisive: [FACT: arXiv:1312.4400] Network-in-Network without dropout scores **14.51 %**
error and with dropout **10.41 %** — **4.1 points from regularisation alone**, with no augmentation.

**(c) TM analogue today.** Two, and one of them is broken.
* **`s` (specificity)** sets the Type Ib forgetting probability `1/s` on included literals. That is
  weight decay, and it works.
* **`max_included_literals`** is an explicit L0 budget, and [FACT: PLAN §3.2, CHARTER silent-failure
  2, memory `tsetlin-clause-budget-batched`] **it does not bind for convolutional models under
  batched feedback** — Type II is ungated and a conv model gets one Type II opportunity *per patch*
  (841 for 4×4). Measured: budget 8 → median clause size **75** at batch 50, 9 at batch 5, 8
  sequential. Also: `torchtsetlin` has drop-clause (PLAN §9.1 `ctm-dropclause`), the dropout analogue.

**(d) What it would have to be.** The firing-rate controller of §5 **is** the working replacement for
the broken L0 budget, and that is the cleanest way to present it: a TM's capacity control should be
stated as a target *activation statistic*, not as a literal count, exactly as a CNN's is stated as a
penalty on the loss and not as a parameter count.

**(e) Experiment.** Folded into §5. Additionally: an `s`-sweep at fixed clause budget, which P3 needs
anyway as the reproduction control.

> **Prediction [HYPOTHESIS]**: at fixed clause budget, tuning `s` recovers most of what a *working*
> `max_included_literals` would give, so the library gap is a usability problem rather than an
> accuracy ceiling.
> **Cheapest falsifier**: an `s`-sweep vs a sequential-feedback run with a binding
> `max_included_literals` at the same clause budget. ≈ 0.4 GPU-h. A > 2-point gap in favour of the
> binding budget promotes LG-003 from usability to accuracy-critical.

---

## 13. Learning-rate schedules and annealing

**(a)** High learning rate early, low late; cosine or step decay.

**(b) Why it helps.** High lr early is *exploration* — large steps escape sharp minima and traverse
the landscape; low lr late is *exploitation* — it settles into a basin and averages out gradient
noise. Every CIFAR-10 result in `LITERATURE_CNN.md` uses a decaying schedule; none uses a constant
one. [FACT: arXiv:1512.03385 §4.2] the ResNet schedule (÷10 at 32 k and 48 k of 64 k iterations) was
itself *selected on a 45 k/5 k split* — He et al. treated the schedule as a first-class
hyperparameter.

**(c) TM analogue today.** **None. Nobody schedules anything in a TM.**
But the two knobs are sitting there: `s` sets the forgetting probability `1/s` — small `s` means
lots of forgetting, i.e. **exploration of clause structures**; large `s` means clauses hold their
shape, i.e. **exploitation**. `T` sets how large a vote margin the ensemble insists on before it
stops giving feedback — the TM's loss temperature.

**(d) What it would have to be.** Anneal `s` upward over training (`s(t) = s₀ · (1 + 2t/E)`), and/or
anneal `T` upward (demand a larger margin as training proceeds). Implementation cost: **setting an
attribute between epochs**. This is the cheapest mechanism in this entire document.

**(e) Experiment.** `ctm-anneal-s` and `ctm-anneal-T` vs `ctm-vanilla`, fixed clause budget, 3 seeds.

> **Prediction [HYPOTHESIS]**: annealing `s` from `s₀` to `3s₀` over training gives +1–3 points at
> fixed clause budget and **reduces the across-seed standard deviation**, because the late phase
> stops churning clause structure. The variance reduction is the more reliable half of the
> prediction.
> **Cheapest falsifier**: two screen runs, essentially no new code. ≈ 0.2 GPU-h — the cheapest entry
> here. Inside the seed band on both accuracy *and* variance → drop it.

---

## 14. Initialisation

**(a)** He/Glorot scaling so each layer's output variance ≈ its input's.

**(b) Why it helps.** It is §5's mechanism (1) applied once, at t = 0. Without it, deep nets start in
a regime where forward or backward signal has already collapsed and never recover.

**(c) TM analogue today.** Automata start at the include/exclude decision boundary. The
consequence is that clauses start nearly **empty**, and an empty clause evaluates **True** while
learning and **False** when predicting (CHARTER silent-failure 1). So a TM starts in the
"everything fires" regime and specialises downward — the *opposite* of a CNN, which starts at
variance 1 and must be kept there.

**(d) What it would have to be.** This is where a genuinely interesting reframing lives rather than
a candidate. [FACT: PLAN §3.1] **random layer-1 clauses beat trained ones, 37.6 % vs 15.4 %**, and
random + calibration reaches 42.6 %. Combined with §9's finding that the TM has no per-parameter
credit, the picture is that a CTM's productive regime looks less like a learned hierarchy and more
like **a random Boolean feature expansion with a calibrated firing rate, read out by a
boosting-like linear vote**. If that framing is right, then the leverage is on (i) the *quality of
the random features* — which is §4's dilation and §11's colour planes — and (ii) the *readout* —
which is §3's counting pool. Both of my top-ranked mechanisms fall out of this framing, which is
either reassuring or a warning about motivated reasoning; the experiments decide.

**(e) Experiment.** Already covered by §3, §4 and PLAN §3.1's existing measurements. No separate
initialisation arm is proposed.

> **Prediction [HYPOTHESIS]**: a *random* layer 1 with dilated multi-scale patches and rate
> calibration, read out by a counting pool, beats a *trained* single-layer CTM at matched automata
> budget.
> **Cheapest falsifier**: this is the composite of §3 + §4 + §5 and should only be run if all three
> individually pass. ≈ 1 GPU-h at screen scale afterwards.

---

## 15. Ensembling

**(a)** Average several independently trained models.

**(b) Why it helps.** Variance reduction: if errors are partly independent, averaging cancels them.
The gain is proportional to how *decorrelated* the members are, which is why ensembles of different
architectures beat ensembles of seeds.

**(c) TM analogue today.** **Already the TM state of the art.** [FACT: PLAN §2/§3.3] TM Composites
reach **82.8 %** on CIFAR-10 as a team of specialised TMs over different Booleanizations including
HOG, versus 60.7 % for the best single CTM. Ensembling is worth ~22 points in the TM world — more
than any mechanism in this document is likely to be worth.

**(d) What it would have to be.** Nothing to invent. What must be enforced is **budget honesty**: a
5-member composite is 5× the automata, and PLAN §7.3 requires any "A beats B" to state the axis. A
composite must be compared against a single model with the same total automata budget, or the claim
is empty.

**(e) Experiment.** P4.6's error-overlap measurement: do TM errors fall on the same images as CNN
errors? If yes, the gap is **representational** and ensembling TMs with TMs will saturate; if no, a
TM/CNN ensemble is on the table and is itself a candidate.

> **Prediction [HYPOTHESIS]**: CTM errors overlap CNN errors substantially more than chance but
> materially less than two CNN seeds overlap each other — i.e. there is real complementary signal,
> and a matched-budget TM composite still beats a single TM of the same total automata count.
> **Cheapest falsifier**: P4.6 needs only saved test predictions, which `record.finalize` already
> writes to `results/preds/`. ≈ 0 GPU-h beyond the runs themselves.

---

## 16. Ranked candidate pool for P5

Ranked by **expected accuracy gain per GPU-hour**, combining predicted effect size, implementation
cost and screen cost. All rankings are `[HYPOTHESIS]`; the screens decide.

| Rank | Mechanism | §  | Predicted gain | Impl. cost | Screen cost | Why it ranks here |
|---|---|---|---|---|---|---|
| **0** | **Capacity-scaling curve** | §8 | n/a — a **gate** | none (P4.1) | ~2 GPU-h | If the curve still climbs at 8 k clauses, everything below is premature. **Run first.** |
| **1** | **Counting pool** (`ctm-count-vote`) | §3 | **5–12 pts** | low (`_votes` only) | 0.3 GPU-h | Largest identified structural gap: C bits vs ~9.7 C bits at identical automata. CNN-side proxy shows 38.9 → 51.0 at screen scale. |
| **2** | **Augmentation, flip first** | §7 | 1.5–4 pts | **zero** (a flag) | 0.4 GPU-h | Worth 3–7 pts to CNNs; costs no code; differentiated flip-vs-crop prediction makes it informative either way. |
| **3** | **Dilated / multi-scale patches** | §4 | 2–5 pts | low (`unfold(dilation=)`) | 0.4 GPU-h | Buys receptive field at **zero** automata cost. Unexplored in the TM literature. |
| **4** | **Anneal `s` / `T`** | §13 | 1–3 pts + variance | **trivial** | 0.2 GPU-h | Cheapest entry in the document. Nobody schedules anything in a TM. |
| **5** | **Online rate control** | §5 | 2–4 pts + variance | medium (controller exists) | 0.5 GPU-h | Best-validated translation (Anchor 1), and the principled fix for the broken L0 budget. |
| **6** | **Soft-`T` distillation** | §10 | 2–4 pts; more at 1 k | medium | 0.5 GPU-h | Injects gradient-derived knowledge with **no** change to the feedback rules and **no** interpretability cost. |
| **7** | **Skip literals for depth** | §6 | 2–5 pts, or a clean negative | high; may need a library gap | 0.4 GPU-h | The only proposal that structurally removes the measured 34.4 → 15.4 collapse. Escapes all four Anchor-2 objections because it propagates nothing. |
| **8** | **Patch-draw (all vs one)** | §2 | ≤ 1 pt | low | 0.4 GPU-h | Cheap; most likely a variance effect only. |
| **9** | **Colour-prototype planes** | §11 | 1–4 pts | medium (new Booleanization) | 0.5 GPU-h | Gated on what `cnn-boolean-therm8` says in P2 — do not build it before that number exists. |
| — | Ensembling | §15 | ~20 pts, already known | n/a | n/a | Not a candidate. A **budget-matching obligation** for any headline claim. |

### The single experiment to run first, given 5 GPU-hours

**`ctm-count-vote` vs `ctm-vanilla` at 1 k / 4 k / 16 k clauses, 3 seeds each, matched automata.**

Because: it attacks the largest structural bottleneck I can identify (a ~10× information reduction
at the readout, at identical automata budget and identical interpretability); the implementation is
confined to the vote path; the CNN-side proxy on the *same architecture and the same Boolean input*
shows a 12-point gap; the three-budget design doubles as a capacity-scaling probe, so it partially
discharges rank-0 in the same runs; and the widening-gap-at-low-budget prediction makes the result
interpretable whichever way it comes out — a *uniform* gap would mean it is a capacity effect in
disguise and would be nearly as informative as a win.

Concurrently, on the second GPU, run rank 2 (augmentation), because it needs no new code at all and
would otherwise waste an idle card.

---

## 17. What would change my mind about this whole document

* **The capacity curve is still climbing at 8 k clauses.** Then CIFAR-10 CTMs are capacity-limited,
  every mechanism here is premature, and the report should say "buy clauses" (PLAN §9.2.9).
* **`cnn-ctmshape-binary` lands near the CTM's 60.7 %.** Then the learning algorithm is *not* the
  bottleneck, the architecture is, and §§3–6 become the whole story while §§9–10 drop out.
* **`cnn-boolean` lands within 1 point of `cnn-small-noaug`.** Then Booleanization is nearly free,
  §11 dies, and the gap is entirely architecture + algorithm.
* **The counting pool shows a *uniform* gain across clause budgets** rather than a widening one at
  low budgets. Then it is a capacity effect in disguise, and its ranking above "more clauses" is
  wrong.
