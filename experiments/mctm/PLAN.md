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
