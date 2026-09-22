# Proposed torchtsetlin changes — NOT APPLIED

Constraint C1: `src/torchtsetlin/**` is read-only for the duration of this programme. Everything
this programme finds missing or wrong in the library is recorded here and applied only after the
user's explicit manual confirmation, as separate reviewed work.

`git diff --stat -- src/` is empty at every gate; see `AUDIT.md`.

Entry format:

```markdown
## LG-00N — <one-line symptom>
- **Symptom**:
- **Reproduction**: `python code/repro/lgNNN.py` -> what it prints
- **Cause**: (file:line)
- **Impact here**: [MEASURED: <id>] what it cost this programme
- **Proposed fix (NOT APPLIED)**:
- **Affected**: files
- **Test to add**:
- **Risk**: what existing behaviour/results the fix would change
```

---

## LG-003 — Clause-size budget does not bind for conv models under batched feedback **at very small budgets only** (NARROWED 2026-09-20, ROUND-1)

Carried over from PLAN §12 / repo `CLAUDE.md`; re-verified in this workspace.

> **CORRECTION, ROUND-1.** This entry was written from the repo `CLAUDE.md`'s flat claim that the
> budget "does not bind". A budget sweep (`results/lg003_budget_sweep.json`) shows the overshoot
> is **absolute (+2.5 literals) and confined to budget 8**. At 16, 32 and 64 — the range every
> published recipe uses — **the budget holds at batch 50**, and a full-scale 30-epoch run at
> budget 32 lands at median 33 against a cap of 32. Neither sequential feedback nor batch 5 is
> required. The original claim, and `CHARTER.md` silent-failure mode 2 which repeats it, are too
> strong and were steering the programme away from a mechanism that works and is worth **+7.7
> percentage points** [MEASURED: ctm-small-budget32]. `CHARTER.md` is the orchestrator's to fix.

- **Symptom** *(as originally filed, and true only at budget 8)*: `max_included_literals=8` on a
  convolutional model yields a median clause size above 8 at batch 50, at 8 at batch 5, and below
  it in sequential mode.
- **Reproduction**: `python code/repro/lg003.py` (200 clauses, 4x4, 2 epochs on 2000 CIFAR-10
  images, budget 8) → [MEASURED: lg003, 2026-09-20] median clause size **10.5 at batch 50**
  (budget exceeded), **8.0 at batch 5** (held), **7.0 sequential** (held); p95 35.1 / 13.0 /
  12.1, max 86 / 23 / 18. The direction reproduces the recorded behaviour exactly; the
  *magnitude* here is a 1.3x overshoot rather than the ~9x (budget 8 -> median 75) recorded in
  the repo `CLAUDE.md`, because this reproduction is a 2-epoch run on a small config. The
  overshoot accumulates with training, so an arm that trains for 60 epochs should be expected
  at the larger end. **Any arm relying on the budget must verify that it binds, per run, from
  `diagnostics.clause_len`** -- which the harness records every epoch.
- **Cause**: `functional.apply_feedback` gates only Type Ia against the remaining room; Type II
  runs `inc2 = min(n2, room)` per commit, and a convolutional model contributes one Type II
  opportunity *per patch* (841 for 4x4 on 32x32), so a single batched commit can walk many
  automata past the include boundary at once.
- **Impact here — REVISED**: none of the feared impact materialised. `ctm-clausesize` **can** be
  reproduced through `max_included_literals` at its published budget of 32 under ordinary batched
  feedback at batch 50. The residual defect is confined to budgets <= 8, which no published recipe
  uses. What the original over-statement *did* cost is worse than the defect: it kept the
  programme's baseline unconstrained, at a median clause length of 175 of 248 available features,
  for the whole of P0.
- **Proposed fix (NOT APPLIED)**: gate Type II by the same `room` computation as Type Ia, behind a
  flag defaulting to current behaviour; alternatively expose a post-commit size projection.
- **Affected**: `src/torchtsetlin/functional.py`, `src/torchtsetlin/models/base.py`.
- **Test to add**: conv model, batch 50, budget 8 → median size ≤ 8. (Budgets 16/32/64 already
  pass and should be pinned as a regression guard.)
- **Risk**: changes the learning dynamics of every existing conv result; needs the benchmark suite
  re-run and `docs/benchmarks.md` regenerated. **Lower priority than originally filed** — the
  practical range works today.

## LG-004 — The default chunk budget silently serialises large convolutional models

- **Symptom**: a convolutional model's throughput per clause degrades sharply with clause count,
  well before any memory pressure. Peak GPU memory stays around 1 GB across the whole measured
  grid (640→8000 clauses, 4x4→10x10 patches) while s/epoch grows faster than the clause count.
- **Cause**: `models/conv.py::_ConvMixin._chunk_elements_per_example` returns
  `3*P*C + 2*C*n_literals + P*n_literals`. The `2*C*n_literals` term (worst-case gathered literal
  rows) is **independent of the batch size and of `P`**, and it dominates. Measured chunk sizes
  for a batch of 50 (`code/repro/lg004.py`, thermometer-4, 10x10 patches):

  | clauses | `2**27` (default) | `2**29` | `2**31` | `2**33` |
  |---|---|---|---|---|
  | 640 | 24 | 50 | 50 | 50 |
  | 2000 | 9 | 37 | 50 | 50 |
  | 8000 | **2** | 9 | 39 | 50 |
  | 32000 | **1** | 2 | 10 | 40 |

  So the `ctm-vanilla` configuration in PLAN §6.2 (8000 clauses, 10x10) runs a mini-batch of 50 as
  **25 sequential chunks** at the library default, on a card with 23 GB free. At 4x4 patches the
  same collapse starts at ~2000 clauses. The model is correct throughout; only the throughput
  changes, which is exactly the
  failure mode the repo `CLAUDE.md` warns about ("wrong throughput number, right answer"), except
  that here it is produced by the *default* rather than by a wrong override.
- **Reproduction**: `python code/repro/lg004.py` → chunk size and s/epoch versus
  `max_chunk_elements` at fixed clause count. See AUDIT A5.
- **Impact here**: [MEASURED: calib_chunk] `max_chunk_elements` is a first-class throughput dial
  for this programme, so the harness exposes it (`--max-chunk-elements`), records it in `hp`, and
  **arms are only compared at equal `max_chunk_elements`** — a lower value is a pure slowdown and
  would make an otherwise identical arm look expensive.
- **Proposed fix (NOT APPLIED)**: (a) document `max_chunk_elements` as a throughput/memory dial in
  the conv models' docstrings and in `CLAUDE.md`; (b) scale the default with free device memory
  (`torch.cuda.mem_get_info`) rather than fixing it at `2**27`; (c) emit a one-time warning when
  `_chunk_size(xb) < xb.shape[0]`, so the serialisation is visible.
- **Affected**: `src/torchtsetlin/models/base.py` (`_chunk_size`, the default),
  `src/torchtsetlin/models/conv.py` (`_chunk_elements_per_example`), `docs/`, `CLAUDE.md`.
- **Test to add**: a conv model at `C=8000`, `10x10`, batch 50 reports `_chunk_size == 50` under a
  memory-scaled default on a 24 GB card.
- **Risk**: none to accuracy (chunking is a pure partition of the batch); changes memory footprint
  and every published throughput number in `docs/benchmarks.md`.

## LG-005 — No public API for patch-level clause activation maps

- **Symptom**: the `(B, P, C)` patch-match tensor is computed inside
  `_ConvMixin._evaluate` and discarded; the public surface exposes only the image-level
  `(B, C)` OR-pooled clause outputs (`evaluate_clauses`) and the vote sums (`forward`).
- **Cause**: by design — `_evaluate` returns the match tensor as an opaque feedback context.
  `src/torchtsetlin/models/conv.py::_ConvMixin._evaluate`.
- **Impact here**: every patch-level diagnostic (per-patch firing rate, the density-collapse
  probe that the previous programme's main finding rests on, spatial clause maps for the report's
  interpretability section) and any multi-layer or pooling candidate in P5 must reach through
  `model._prepare / _encode / _evaluate`. `code/diagnostics.py::patch_firing_stats` and
  `experiments/mctm/mctm.py::conv_clause_maps` are both doing this. Private-API use is recorded
  here rather than hidden, because a library change would break it silently.
- **Proposed fix (NOT APPLIED)**: a public
  `ConvTsetlinMachine.clause_maps(x, empty_value=False) -> (B, C, Py, Px)`, plus a public
  `grid(H, W)` accessor. It is a thin wrapper over code that already exists.
- **Affected**: `src/torchtsetlin/models/conv.py`, `docs/api/`.
- **Test to add**: `clause_maps(x).any(dim=(2,3)) == evaluate_clauses(x)`.
- **Risk**: none — additive.

## LG-006 — Real-valued multiplicative clause weights (the 2019 Weighted TM) are not expressible

- **Symptom**: `weighted=True` gives integer weights updated by `±1` per Type Ia / Type II event and
  clamped at `≥ 0`. The rule published as *the* Weighted Tsetlin Machine is multiplicative and
  real-valued: `w ← w·(1+γ)` on Type I, `w ← w/(1+γ)` on Type II, `w` initialised to 1.0, `γ` a
  learning rate `[FACT: Phoulady et al., arXiv:1911.12607 §3.4, Eqs 12–17]`. There is no `γ`.
- **Reproduction**: `python -c "import torchtsetlin as t; m=t.models.ConvTsetlinMachine(10,64,100,weighted=True); print(m.weights.dtype)"` → `torch.int32`; no constructor accepts a weight learning rate.
- **Cause**: `src/torchtsetlin/models/classifier.py:173-187` (`_accumulate_weights` / `_commit_weights`),
  `src/torchtsetlin/models/classifier.py:91` (`register_buffer("weights", torch.ones(C, dtype=torch.int32))`).
- **Impact here**: **low for CIFAR-10, and that is itself the finding.** Every CIFAR-10 result in the
  bibliography (CTM 2019 §2.5, TM Composites, the Optimized Toolbox, GraphTM/CoTM) uses *integer*
  weighting, which the library does implement. The gap only bites if we want a faithful
  `ctm-weighted-2019` arm — and that arm has no CIFAR-10 number to reproduce against
  (`LITERATURE_TM.md` A2). Recorded so the report does not claim to have reproduced arXiv:1911.12607.
- **Related, smaller**: the library clamps class-owned weights at `0`
  (`classifier.py:187 .clamp_(min=0)`), so a clause can be silently removed from the vote; the
  integer-weighted reference clamps at `1`. Worth one diagnostic (`fraction of zero-weight clauses`)
  before it is worth a fix.
- **Proposed fix (NOT APPLIED)**: an optional `weight_dtype`/`weight_lr` on the weighted classifier;
  when `weight_lr > 0`, store float weights and apply `w *= (1+γ)^{n_Ia} / (1+γ)^{n_II}` at commit
  (the accumulator already carries the two event counts separately). Also expose `weight_min`.
- **Affected**: `src/torchtsetlin/models/classifier.py`, `src/torchtsetlin/models/coalesced.py`.
- **Test to add**: `weight_lr=0` reproduces current integer behaviour bit-for-bit;
  `weight_lr=0.05` on Noisy XOR converges and leaves weights in `(0, ∞)`.
- **Risk**: additive if defaulted off; changing the clamp from 0 to 1 would change every existing
  weighted result.

## LG-007 — No dilation, and one `patch_size` per model: the two cheapest receptive-field mechanisms are unreachable

- **Symptom**: a convolutional TM can only grow its clause receptive field by enlarging
  `patch_size`, which grows the automata budget **quadratically** (`THEORY.md` §6.2: `F = Z·k² +
  2(32−k)`; 3×3 → 16×16 is a 22.5× automata increase at fixed clause count). Two standard CNN
  mechanisms that grow the receptive field *without* growing the feature count are unavailable:
  (a) **dilation**, (b) **multi-scale clause groups** (different `k` for different subsets of the
  clause pool, votes summed).
- **Reproduction**: `ConvTsetlinMachine(..., dilation=2)` → `TypeError: unexpected keyword`;
  `m.patch_size` is a single pair shared by all `n_clauses_total` clauses
  (`src/torchtsetlin/models/conv.py:46-63`).
- **Cause**: `_ConvMixin._encode` calls `torch.nn.functional.unfold(x, kernel_size=self.patch_size,
  stride=self.stride)` (`src/torchtsetlin/models/conv.py:125-127`). `unfold` already accepts
  `dilation`; it is simply not threaded through, and `conv_features` / `_grid`
  (`conv.py:66-80`) do not account for it.
- **Impact here**: dilation is the only receptive-field mechanism in `THEORY.md` §6.4 that is free in
  both automata and compute, and it is one call-site away. Multi-scale groups must instead be run as
  several separate models whose votes are added — which makes them structurally a *composite*
  (`LITERATURE_TM.md` §2.4) and therefore not comparable to a single model at matched budget without
  saying so. Both affect the P5 candidate space directly (PLAN §9.2 item 4).
- **Proposed fix (NOT APPLIED)**: (a) add `dilation: IntOrPair = 1` to `_ConvMixin._init_conv`,
  thread it into `_grid`, `conv_features`, `_encode` and `default_feature_names`; (b) separately,
  a `MultiScaleConvTsetlinMachine` that holds several `(patch_size, n_clauses)` groups over one
  `ta_state` is a larger change and should wait for a `[MEASURED]` result from the composite-style
  workaround first.
- **Affected**: `src/torchtsetlin/models/conv.py`.
- **Test to add**: `dilation=2, patch_size=3` on a 32×32 input gives `Py = Px = 28` and
  `conv_features` accounting that matches `unfold`'s output width.
- **Risk**: additive; `dilation=1` is the current behaviour.

## LG-008 — The Booleanization registry is missing four of the seven encodings behind the TM state of the art

- **Symptom**: the 82.8% TM Composite is built from seven image Booleanizations
  `[FACT: Grønningsæter et al. 2024, §III]`. `torchtsetlin.data.encoders` ships
  `AdaptiveThresholdEncoder` (gaussian **and** mean) and `ColorThermometerEncoder`
  (`src/torchtsetlin/data/encoders.py:268, 314`). **Otsu's thresholding, Canny edge detection,
  Histogram of Oriented Gradients, and adaptive (multilevel) colour thermometers are absent.**
- **Reproduction**: `python -c "import torchtsetlin.data as d; print(d.__all__)"` → no `Otsu`,
  `Canny`, `HOG` or `AdaptiveColorThermometer`.
- **Cause**: scope, not a defect. Three of the four need an image-processing dependency
  (OpenCV/scikit-image) or a non-trivial pure-torch implementation.
- **Impact here**: **none blocking** — these are data-side and live happily in
  `experiments/convtm/code/` under the `BOOLEANIZATIONS` registry in `code/CONTRACT.md`. Recorded
  because (i) the Booleanization axis is the largest measured lever in the literature (a **16.7
  point** spread at a fixed 2 000 clauses, `LITERATURE_TM.md` §2.3), so these encoders are the
  highest-value additions the library could make after this programme; and (ii) the report must state
  that the encoders used were ours, not the library's.
- **Proposed fix (NOT APPLIED)**: promote the P3 harness encoders into
  `torchtsetlin.data.encoders` behind the existing optional-dependency guard used by
  `data.load_*_boolean`, with a `BooleanEncoder` subclass each and thresholds kept on CPU
  (the `utils._to_device` rule).
- **Affected**: `src/torchtsetlin/data/encoders.py`, `src/torchtsetlin/data/__init__.py`, `docs/`.
- **Test to add**: each encoder is deterministic, returns `torch.bool`, and matches a stored
  reference tensor for one CIFAR-10 image.
- **Risk**: additive; new optional dependency.

## LG-009 — Whole approaches that the library cannot express at all

Grouped, because each would be a new model rather than a missing hook. Recorded so the report can
say why these are absent rather than leaving it to inference.

- **Symptom / scope**:
  | approach | what it needs | why the library cannot |
  |---|---|---|
  | **Graph TM** `[FACT: arXiv:2507.14874]` — CIFAR-10 **70.28 ± 0.17** | hypervector symbol encoding of nodes, typed edges, message passing, nested (depth>1) clauses | `HypervectorEncoder` (`data/encoders.py:341`) encodes discrete *tokens*; there is no message-passing model and no nested-clause representation |
  | **Sparse TM** `[FACT: arXiv:2405.02375]` | clauses that start empty and materialise literals on demand; a lower state bound that deletes a TA | `ta_state` is a dense `(C, 2F)` integer buffer (`models/base.py` docstring) |
  | **Contracting TM** `[FACT: arXiv:2310.11481]` | absorbing Include/Exclude states; three action lists | `functional.apply_feedback` is ergodic by construction (`functional.py:262 .clamp_(0, 2N-1)`) |
  | **Fuzzy-Pattern TM** `[FACT: arXiv:2508.08350]` | graded clause output = f(number of satisfied literals) plus a matching feedback rule | `functional.clause_violations` (`functional.py:52`) **already returns the count**, so the forward path is reachable by subclassing `_evaluate`/`_votes`; only the feedback rule is missing. **This one is partially expressible** |
- **Impact here**: none of the four has a published CIFAR-10 number except Graph TM, whose CIFAR-10
  gain its own authors attribute to multi-view Booleanization at **depth 1**
  `[FACT: arXiv:2507.14874 §3.2, Table 16]`, i.e. to something we can reproduce far more cheaply by
  feeding a `ConvCoalescedTsetlinMachine` two concatenated Booleanizations. Recorded, not
  prioritised (`rounds/ROUND-1/tm-theorist.md`).
- **Proposed fix (NOT APPLIED)**: none proposed. If a P5/P6 result makes graded clause matching
  (Fuzzy-Pattern) look promising, the smallest useful library change is an `empty_value`-style hook
  that lets a subclass supply its own `(violations → clause value)` map, plus a feedback policy that
  receives the violation counts.
- **Affected**: would be new modules.
- **Risk**: large; out of scope for this programme under C1.

## LG-010 — Paper-configuration ambiguities the library cannot resolve, recorded against ARMS.md A1–A4

Not a defect; filed here because `ARMS.md` asks P1 to settle these before any `GAP` verdict.

- **A1 — "number of clauses": per class or total?**
  `[FACT: Granmo et al. 2019, Table 1]` the CTM paper's row is literally *"#Class Clauses"*, and
  `[FACT: arXiv:2301.08190 fn. 6]` CSC-TM says "8000 clauses **per class**". `[FACT: arXiv:2309.04801
  Table 1 vs §4]` TM Composites writes an unqualified "2K weighted clauses" in Table 1 and
  "8 000 weighted clauses **per class**" in §4 of the same paper. `[FACT: arXiv:2406.00704 Tables
  II–IV]` and `[FACT: arXiv:2507.14874 Table 16]` are unqualified. **Verdict**: the harness
  convention (`hp.n_clauses` = total) is right; but `ctm-vanilla` configured "as the 2019 paper" is
  **80 000 total**, not 8 000, and the Toolbox/Composites numbers must be settled empirically
  (`rounds/ROUND-1/tm-theorist.md`, measurement M0).
- **A2 — class-owned vs coalesced for `ctm-vanilla`**: **agreed, class-owned.**
  `[FACT: arXiv:1905.09688 §2–3]` the 2019 CTM convolves the 2018 multi-class TM with fixed
  per-clause polarity; clause sharing is a separate 2021 contribution `[FACT: arXiv:2108.07594]`.
- **A3 — `T` and `s` per paper**: `T_ratio = 0.8` is **not** safe as a default. Measured across the
  bibliography, `T / (total clauses)` spans **0.025 to 1.5**: Drop Clause CIFAR-10 `T=48 000` at
  60 000 clauses = 0.80 `[FACT: AAAI 2023]`; GraphTM/CoTM `T=15 000` at 80 000 = 0.19
  `[FACT: Table 16]`; Toolbox colour thermometers `T=3 000` at 2 000 = **1.50**
  `[FACT: Table III]`; Composites HOG `T=50` at 2 000 = 0.025 `[FACT: Table 1]`. The full
  per-paper table is `LITERATURE_TM.md` §1. **Every arm must carry its paper's `T` and `s`
  explicitly; the ratio default may only be used for arms that have no paper.**
- **A4 — position encoding, thermometer or one-hot?** **Thermometer.**
  `[FACT: arXiv:1905.09688 §3]` allows "thresholding [8] or one-hot", and its worked example uses
  thresholding; `[FACT: Book Ch. 4 §4.5]` teaches thermometer only. The library's polarity is
  `y > k` (`models/conv.py:117`) whereas the 2019 text writes the complementary convention; this is
  immaterial because both literals of every feature are available to the automata.

## LG-011 — A conv `_encode` override cannot wrap the parent's, only replace it

Filed by research-engineer while answering the `mctm-skip` feasibility question. Minor; recorded
because it is a composition limit that a future library change could remove cheaply.

- **Symptom**: a subclass that widens the literal vector (extra "skip" literals appended to every
  patch) must override `conv_features` so that `n_features` is right — and then the parent
  `_ConvMixin._encode` raises `ValueError: Patch feature mismatch: got 170, expected 190`, because
  it asserts the unfolded patch width against the subclass's already-widened `self.n_features`.
  The subclass therefore cannot call `super()._encode()`; it has to reproduce the unfold (~10
  lines).
- **Reproduction**: `python code/repro/skip_literals.py` — route A (channel concatenation, no
  subclass) works; route B (global skip literals) works only with `_encode` fully reimplemented;
  route C (`_encode` overridden without `conv_features`) fails loudly inside `clause_outputs`.
- **Cause**: `src/torchtsetlin/models/conv.py:132-135` — the check is against the final
  `self.n_features` rather than against `Z*kh*kw + position bits`.
- **Impact here**: none to correctness and none to throughput — `_chunk_elements_per_example`
  reads `self.n_literals`, which is `2 * conv_features(input_shape)`, so the chunk budget stays
  correct in both routes ([MEASURED: `code/repro/skip_literals.py`] chunk 8/8 at batch 8 in both).
  Cost is ~10 duplicated lines per skip arm.
- **Proposed fix (NOT APPLIED)**: split `_encode` into `_patch_features(xb) -> (B, P, F_base)` and
  a thin literal-forming wrapper, and move the assertion into the wrapper. Subclasses then
  override `_patch_features` and compose cleanly.
- **Affected**: `src/torchtsetlin/models/conv.py`.
- **Test to add**: a subclass appending `n_skip` global literals reports
  `n_literals == 2*(base + n_skip)` and trains.
- **Risk**: none — a refactor behind an unchanged public surface.

---

## LG-005 addendum — 2026-09-21, research-engineer: D-ORDER is a third private-API consumer

No new gap; the same one, now load-bearing for a registered theory prediction.

`code/diagnostics.py::order_stats` (**D-ORDER**, THEORY.md 5.5.9) evaluates each clause's
*top-`k` prefix* — the `k` included literals with the deepest TA states — as a clause in its own
right. To do that it must:

1. reach the `(B, P, 2F)` patch literals via `model._prepare` / `model._encode` (private), and
2. re-implement the patch OR (`matches.any(dim=1)`) that `_ConvMixin._evaluate` performs
   internally, because `_evaluate` evaluates against `self.include` and there is no way to
   evaluate against a *different* include mask.

Step 2 is the new part, and it is what the LG-005 fix sketch does **not** currently cover.
`functional.clause_outputs(literals, include, include_count, empty_value)` is public and takes an
arbitrary mask, so the arithmetic is reachable — but only after the caller has re-derived `P`,
reshaped `(B*P, 2F)`, and reproduced the OR. Every one of those is a place where a library change
would break a diagnostic silently rather than loudly.

**Fix sketch, extending LG-005's (NOT APPLIED)**: give the proposed
`ConvTsetlinMachine.clause_maps(x, empty_value=False)` an optional `include=None` argument
defaulting to `self.include`, with `include_count` derived when it is passed. Three lines, purely
additive, and it turns two private-API consumers and one re-implementation into one public call.

**Affected**: `src/torchtsetlin/models/conv.py`. **Risk**: none — additive.
**Test to add**: `clause_maps(x, include=m).any(dim=(2,3))` equals a masked `evaluate_clauses`
for a mask `m` that is a strict subset of `model.include`, and equals `evaluate_clauses(x)` for
`m = model.include`.
