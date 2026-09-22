# MCTM on CIFAR-10 — idea and evaluation plan

## 1. The idea, stated precisely

A convolutional Tsetlin machine evaluates clause *j* on every patch of an image and
immediately collapses the result:

```
z_j = OR_p  conj_j(patch_p)          score_c = clamp( sum_j W[j,c] * z_j , -T, T )
```

The conjunction lives **inside one patch**; across patches there is only an OR followed by
a linear threshold. The MCTM keeps the spatial map instead:

```
X^(l+1)_j(r,c) = conj_j( patch_{r,c}( X^(l) ) )          X^(l) in {0,1}^{C_l x H_l x W_l}
```

so layer *l+1* forms conjunctions **over the activations of layer *l*, at different
offsets**. Layers are trained greedily and frozen, because Tsetlin automata have no
gradients to propagate.

### Why depth could help — the representational argument

A single conv layer cannot express a concept that requires a conjunction or a negation
*between spatially separated parts*, unless one patch covers both. `NOT (OR_p triangle(p))`
is an AND over patches, and no single clause sees more than one patch. A second layer
whose patch spans the (pooled) map expresses it directly. This is a representational
separation, not an analogy to CNNs, and it is what the experiments below try to detect.

### The two risks

1. **Greedy supervised saturation.** If layer 1 is trained as a classifier and frozen, it
   keeps only what made *it* accurate, and discards exactly the residual information layer 2
   would need. The ceiling of the stack may be the ceiling of layer 1.
2. **Density collapse.** A trained CTM clause includes tens of literals and therefore fires
   on a tiny fraction of patches. A conjunction of *j* such bits is satisfied with
   probability ~p^j. If the maps are too sparse, layer-2 clauses can only survive by
   including *negated* (near-constant) literals, and learn nothing. Measured on a synthetic
   pilot: a trained layer-1 clause used 42-89 literals and fired on **0.02%** of patches,
   with 28% of channels completely dead.

Risk 2 makes the naive configuration a strawman, so layer-1 density is treated as a
controlled variable rather than left at its default. The intended knob was
`max_included_literals`; E0 showed it does not bind under batched feedback in convolutional
models (Type II is ungated), so **specificity `s` is used instead** (`p_forget = 1/s`).

## 2. Implementation

`mctm.py`:

| function | role |
|---|---|
| `conv_clause_maps` | reshapes the `(B, P, C)` match tensor the library already computes into `(B, C, Py, Px)`; forces `empty_value=False` so an empty clause never emits an all-ones channel |
| `or_pool` | `k x k` OR pooling — max-pool on 0/1 bits; here a **density** control, not just downsampling |
| `randomize_clauses` | random layer-1 control: `n_include` literals per clause, random feature + random sign, never `x AND NOT x` |
| `MCTM` | frozen stack + head; `transform` streams, the driver materialises |
| `firing_stats` | per-channel firing rate, dead/saturated fractions |
| `effective_receptive_field` | RF and jump of a patch/pool stack |

Fixed configuration: layer 1 = 4x4 patches, stride 1, 128 shared clauses
(`ConvCoalescedTsetlinMachine`, so clauses are *shared features*, not class-owned);
2x2 OR pool; layer 2 = 3x3 patches over the 128x14x14 maps, 512 clauses.
Effective receptive field of the stack: **9x9 pixels**.

CIFAR-10 is Booleanized with `ColorThermometerEncoder` (4 bits/channel -> 12 planes).

## 3. Experiment plan

### E0 — density diagnostic (run first, it sizes everything else)

Sweep layer-1 specificity `s ∈ {1.5, 2, 3, 5, 10}` (plus clause-budget rows as a diagnostic
of the budget mechanism itself) and measure the firing rate
of the resulting maps plus layer 1's own accuracy. Picks the layer-1 setting used by every
MCTM arm. **If no budget puts the median firing rate in a usable band, the proposal fails
here** and the rest of the grid is reporting a negative result rather than a comparison.

### E1 — the arms

| arm | question it answers |
|---|---|
| `single-l1` | what layer 1 alone achieves (the greedy ceiling it is trained against) |
| `single-matched` | same **total** clause budget (128+512) in one layer — does depth beat width? |
| `single-rf` | one layer with patch = the stack's 9x9 effective RF — is any gain just a bigger receptive field? |
| `flat-stack` | layer-1 bits, globally OR-pooled, into a flat TM — does keeping the *spatial* map earn its cost over the trivial stack? |
| `mctm` | the proposal |
| `mctm-random` | random layer-1 clauses, same layer 2 — does the supervised local objective buy anything? |
| `mctm-nopool` | `pool=1` — is OR-pooling load-bearing? |
| `mctm-nopos` | layer-2 `position_encoding=False` |

Baselines run at the `s` that is best *for them* (E0: `s=10`), so the comparison is
conservative towards the MCTM, which is run in both regimes (`mctm` at `s1=10`,
`mctm-dense` at `s1=2`). 3 seeds per arm; report mean +- spread of best test accuracy,
wall-clock, clause count and automata count.

### E2 — tasks whose per-arm ceiling is known analytically (added after E1)

A null result on CIFAR-10 is ambiguous: its classes may simply not be concepts a second layer
helps with. Two constructed tasks remove the ambiguity (`synthetic.py`):

* `xor` — label = (square present) XOR (cross present). One layer decides by a linear
  threshold over globally OR-pooled clause bits, so it is capped at 75%; both stacks can
  express it.
* `near` — both shapes present on a ring of 16 equally spaced slots, label = are they close.
  Rotational symmetry makes each shape's marginal position identical across classes, so only
  the *joint* separation carries the label. One layer and the flat stack are at chance by
  construction; only the spatial MCTM can express it.
  (`scripts/check_leakage.py` verifies the best additive `f(pos_A)+g(pos_B)` predictor is at
  chance — the naive rejection-sampled version of this task leaks to 66%.)

Each stacked arm runs with the greedy layer 1 *and* with a hand-built **oracle** layer 1
(exact square/cross detectors), plus an `oracle64` variant padded with random distractor
channels so layer 2 faces the same channel count as the greedy case. That separates
"the architecture cannot do it" from "greedy training cannot find it".

### Falsification criteria, fixed in advance

* MCTM is **worth pursuing** only if it beats `single-matched` **and** `single-rf` **and**
  `flat-stack`, all at matched clause budget.
* If `mctm-random` matches `mctm`, the greedy supervised objective contributes nothing.
* If `mctm` <= `single-l1`, greedy saturation (risk 1) is confirmed.

---

# Follow-up: the three repairs proposed in §7.3 of the report

The first evaluation concluded that the architecture is sound and greedy supervised layer-wise
training is not, and its §7.3 named three ways out. `ideas.py` implements all three; `run_ideas.py`
(CIFAR-10) and `synthetic_ideas.py` (the two constructed tasks) run them.

## The ideas, and what each is supposed to fix

| | idea | fixes | implementation |
|---|---|---|---|
| **A** | a different objective for layer 1 | clauses too specific, information discarded | `pretrain_autoencoder_l1` — predict a patch's centre pixels from its surrounding ring, no labels; embed the flat clauses into the conv layer's literal space |
| **B** | explicit density calibration | firing rate left to chance via `s` | `calibrate_density` — a two-sided greedy walk towards a target per-patch firing rate (and, optionally, a clause-size cap) |
| **C** | credit propagation instead of freezing | greedy saturation | `CreditStack` — layer-2 Type Ia/Ib/II events resolve through an included positive channel literal to a layer-1 clause and position |

## Criteria, fixed before the runs

An idea is a repair only if it beats, at a matched clause budget:

1. the greedy stack (`mctm`, 15.4%) — necessary, not sufficient;
2. the **random-layer-1 control** (`mctm-random`, 37.6%) — a trained feature layer that does not
   beat an untrained one is not earning its objective;
3. a single layer at the same total budget (`single-matched`, 35.4%).

For **C** there is a tighter, controlled test: credit training must beat the frozen stack *with the
same layer-1 initialisation*.

## Design decisions the results turned on

* **B adds only pixel literals.** The cheapest way to hit any firing rate is a position literal
  (`y > 27` fires on 1/29 of patch rows whatever the image contains). Left free, the controller takes
  it every time and produces channels carrying no information.
* **B needs both targets.** A 55-literal clause and a 3-literal one can sit at the same firing rate
  and are not interchangeable. Under a size cap the removal rule must optimise the *joint* objective:
  dropping the most load-bearing literal sends the rate to ~1, the least to ~0.
* **C's rule as written has no forgetting term.** Type Ia and Type II both only add literals. Type Ib
  is the only one that removes them, needs no position, and must be blamed on the literal that
  actually blocked the match — crediting a random included literal erases layer 1 within an epoch.
* **C needs a warm start.** An untrained layer 2 has no included positive channel literals, so there
  is nothing to propagate through until it has some.
* **`test_ideas.py` asserts the index chain.** The credit path crosses two patch geometries and a
  pooling stage; nothing downstream looks wrong if one step is off by a row.

## What was run

* **E3** `mctm-auto`, plus `flat-auto` / `flat-random` readouts.
* **E4** `mctm-calib-r{005,01,02,05,10,20}` (firing-rate targets 0.5–20% over the *trained*
  layer 1), then `mctm-size-k{24,12,6,3}` (a size target at the best rate) with
  `mctm-calib-r20-tight` as the control that separates "cut further" from "cut shorter", then
  `mctm-rc-r{05,10,40,60}` + `mctm-random-calib` — the same sweep over an **untrained** layer 1,
  which is the configuration that clears every criterion — and `mctm-auto-calib`.
* **E5** `mctm-credit-{ia,ib,bal}` (the three rungs of the credit rule), `mctm-credit-warm`
  (warm start from the greedy layer 1 — which delivers *zero* credit events, so the step size
  is only worth sweeping with the controller running), and
  `mctm-credit-cal{-p1,,-p001}` / `mctm-auto-credit`.
* **E6** all three ideas on `xor` and `near`, where the per-arm ceiling is known analytically and the
  oracle row says how much headroom a better training scheme has.

## The result, in one line

The density diagnosis was right. A label-free controller that walks each clause towards a
target firing rate takes the stack from 15.4% to 41.8% on CIFAR-10 — past both single-layer
baselines at the same clause budget — **when it is run over a layer 1 that was never trained**.
Over the trained layer the same controller reaches 24.5%, or 35.8% with a clause-size target.
The autoencoder objective is the winner on the two constructed tasks (99.9% and 87.9%, both
past the hand-built oracle). Credit propagation does not work, for reasons `ideas.py` documents
rung by rung.

Results: `report2/`.
