# Proposed torchtsetlin changes

Constraint C1: `src/torchtsetlin/**` is read-only for the duration of this programme. Everything
this programme finds missing or wrong in the library is recorded here and applied only after the
user's explicit manual confirmation, as separate reviewed work.

`git diff --stat -- src/` was empty at every gate of this programme; see `AUDIT.md`. The programme
halted on 2026-09-22. The library has changed since, **outside** this programme: entries are
re-checked against `src/` as of **2026-10-17 (0.3.1)** and each now carries a status.

## Status (re-checked 2026-10-17 against 0.3.1)

| ID | Gap | Status | Resolved by / what is left |
|---|---|---|---|
| LG-003 | Clause budget overshoots under batched conv feedback | **RESOLVED in 0.3.0** | `fb58305` hard cap (`functional._cap_new_inclusions`); conv-model regression test still missing |
| LG-004 | Default chunk budget serialises large conv models | OPEN | — |
| LG-005 | No public patch-level clause maps | **PARTIAL** | `c5deb95` added `patch_clause_outputs`; `include=` override and `grid()` accessor still missing |
| LG-006 | No real-valued multiplicative clause weights | OPEN | — |
| LG-007 | No dilation; one `patch_size` per model | OPEN | scope grew: `Conv3d*` (and possibly `Segmentation*`) need it too |
| LG-008 | Missing Otsu / Canny / HOG / adaptive colour thermometer encoders | OPEN | — |
| LG-009 | Approaches the library cannot express | OPEN (not proposed) | — |
| LG-010 | Paper-configuration ambiguities | NOT A LIBRARY GAP | kept here only because `ARMS.md` cites it by ID |
| LG-011 | Conv `_encode` override cannot wrap the parent's | OPEN | scope grew: `_Conv3dMixin._encode` duplicates the check |

**Numbering** starts at LG-003. No LG-001 or LG-002 was ever filed in this file (checked against its
git history); the first entry took its number from the `PLAN.md` §12 template. They are not missing.

**Recommended order once C1 is lifted** — the first five are small and additive:

1. LG-003: add the conv-model budget regression test.
2. LG-004 (c): one-time warning when a batch is split into chunks.
3. LG-005: `include=` override on the patch-map accessor, plus a public `grid()`.
4. LG-011: split `_encode` into `_patch_features` + literal wrapper (2D and 3D).
5. LG-007 (a): thread `dilation` through the conv mixins.

LG-006, LG-008 and LG-009 are features rather than fixes and should wait for a measured need.

**Stale cross-references (orchestrator's to fix):** `PLAN.md` §12 still uses LG-003's original
"median 75" claim as its template example, and `CHARTER.md` silent-failure mode 2 describes the
soft-cap overshoot. Both predate the 0.3.0 hard cap.

Entry format:

```markdown
## LG-00N — <one-line symptom>
- **Status**: OPEN | PARTIAL (<what is left>) | RESOLVED in <version> (<commit>)
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

## LG-003 — Clause-size budget does not bind for conv models under batched feedback **at very small budgets only** (NARROWED 2026-09-20, ROUND-1; RESOLVED in 0.3.0)

- **Status**: **RESOLVED in 0.3.0** (`fb58305`, 2026-10-01, GitHub issue #1), outside this
  programme. `functional.apply_feedback` now calls `_cap_new_inclusions`
  (`src/torchtsetlin/functional.py:272`): after each commit, a clause that would end above
  `max_included_literals` keeps a uniformly random subset of its *newly* included literals and the
  rest are reset to `N-1`. This is the "post-commit size projection" alternative below, **not** the
  Type II gating that was proposed first; Type II is still ungated (`functional.py:260-261`).
- **Still open**: the regression test. The budget is pinned for the flat classifier
  (`tests/test_models.py::test_batched_classifier_respects_max_included_literals`) and for
  segmentation (`tests/test_segmentation.py`), **not for a convolutional model** — which is where
  the per-patch Type II event count made the overshoot possible.
- **Every budget number in this programme was measured under the old soft cap** — the +9.46 pp
  budget effect, the s=2 + budget-32 cell (48.31), the sequential-feedback control (+8.75 pp) and
  the T0 cell (65.10% test). At budgets 16/32/64 the soft cap already bound, so the effects should
  survive, but they must be re-run before any new result is compared against them.

Carried over from PLAN §12 / repo `CLAUDE.md`; re-verified in this workspace.

> **CORRECTION, ROUND-1.** This entry was written from the repo `CLAUDE.md`'s flat claim that the
> budget "does not bind". A budget sweep (`results/lg003_budget_sweep.json`) shows the overshoot
> is **absolute (+2.5 literals) and confined to budget 8**. At 16, 32 and 64 — the range every
> published recipe uses — **the budget holds at batch 50**, and a full-scale 30-epoch run at
> budget 32 lands at median 33 against a cap of 32. Neither sequential feedback nor batch 5 is
> required. The original claim, and `CHARTER.md` silent-failure mode 2 which repeats it, are too
> strong and were steering the programme away from a mechanism that works and is worth **+9.46
> percentage points** (47.77 ± 0.32 vs 38.31 ± 1.05, 3 seeds, `T = 80`) [MEASURED:
> ctm-small-budget32; DECISIONS.md]. The +7.7 pp first written here came from the earlier run with
> an unreachable `T`, superseded by AUDIT A21. `CHARTER.md` is the orchestrator's to fix.
>
> The two clause-size figures come from different configurations: the **sweep** (2 epochs, small
> config, `results/lg003_budget_sweep.json`) lands at median 20.0 for budget 32; the **full-scale
> 30-epoch run** lands at median 33.

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
- **Test to add** (still missing after 0.3.0): conv model, batch 50, budget 8 → **max** size ≤ 8
  (a hard cap now, so the maximum is the right assertion, not the median); repeat for 16/32/64 as a
  regression guard.
- **Risk**: changes the learning dynamics of every existing conv result; needs the benchmark suite
  re-run and `docs/benchmarks.md` regenerated. **Lower priority than originally filed** — the
  practical range works today.

## LG-004 — The default chunk budget silently serialises large convolutional models

- **Status**: OPEN. Re-checked 2026-10-17: the default is still `max_chunk_elements=2**27`
  (`src/torchtsetlin/models/base.py:92`) and its docstring (`base.py:65-66`) still describes it only
  as a memory budget. The 3D models added since (`a3eba41`) block the patch axis when a single
  example exceeds the budget (`conv.py::_Conv3dMixin._patch_block`), which addresses the
  memory-bound volume case but not the 2D serialisation measured here.

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
- **Proposed fix (NOT APPLIED)**, in order of cost: (c) first — emit a one-time warning when
  `_chunk_size(xb) < xb.shape[0]`, so the serialisation is visible; cheap and changes nothing
  else. (a) document `max_chunk_elements` as a throughput/memory dial in the conv models'
  docstrings and in `CLAUDE.md`. (b) last — scale the default with free device memory
  (`torch.cuda.mem_get_info`) rather than fixing it at `2**27`; this is the one that changes
  memory footprint and published throughput.
- **Affected**: `src/torchtsetlin/models/base.py` (`_chunk_size`, the default),
  `src/torchtsetlin/models/conv.py` (`_chunk_elements_per_example`), `docs/`, `CLAUDE.md`.
- **Test to add**: a conv model at `C=8000`, `10x10`, batch 50 reports `_chunk_size == 50` under a
  memory-scaled default on a 24 GB card.
- **Risk**: none to accuracy (chunking is a pure partition of the batch); changes memory footprint
  and every published throughput number in `docs/benchmarks.md`.

## LG-005 — No public API for patch-level clause activation maps

- **Status**: **PARTIAL.** `c5deb95` (2026-09-23, the day after this programme halted) added
  `_ConvMixin.patch_clause_outputs(x, empty_value=None) -> (B, Py, Px, C)`
  (`src/torchtsetlin/models/conv.py:206`); `Conv3d*` has the 3D equivalent. It differs from the
  sketch below in two ways: the layout is `(B, Py, Px, C)`, not `(B, C, Py, Px)`, and `empty_value`
  defaults to `self.training`, not `False` — call it under `model.eval()` or pass
  `empty_value=False`. **Still missing**: the `include=` override (D-ORDER, below) and a public
  `grid()` accessor.
- **Action available now, within C1**: move `code/diagnostics.py` (`match_count_stats` ~225 and
  `patch_firing_stats` ~516) from `model._prepare/_encode/_evaluate` onto `patch_clause_outputs`. `order_stats`
  (~427) still needs the private path until `include=` exists.
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
- **Third consumer — D-ORDER** (research-engineer, 2026-09-21; folded in from a separate
  addendum): `code/diagnostics.py::order_stats` (THEORY.md 5.5.9) evaluates each clause's top-`k`
  prefix — the `k` included literals with the deepest TA states — as a clause in its own right. It
  must (1) reach the `(B, P, 2F)` patch literals via `model._prepare` / `model._encode`, and
  (2) re-implement the patch OR (`matches.any(dim=1)`), because `_evaluate` evaluates against
  `self.include` and there is no way to pass a different mask. `functional.clause_outputs(literals,
  include, include_count, empty_value)` is public and takes any mask, but only after the caller has
  re-derived `P`, reshaped to `(B*P, 2F)` and reproduced the OR — each a place where a library
  change would break the diagnostic silently.
- **Proposed fix (NOT APPLIED)** — originally `clause_maps(x, empty_value=False) -> (B, C, Py, Px)`
  plus `grid(H, W)`; `patch_clause_outputs` now covers the first half. What is left: an optional
  `include=None` argument on `patch_clause_outputs` (defaulting to `self.include`, with
  `include_count` derived when passed), and a public `grid(H, W)`.
- **Affected**: `src/torchtsetlin/models/conv.py` (2D and 3D mixins), `docs/api/`.
- **Test to add**: `patch_clause_outputs(x, include=m).any(dim=(1,2))` equals a masked
  `evaluate_clauses` for a mask `m` that is a strict subset of `model.include`, and equals
  `evaluate_clauses(x)` for `m = model.include`.
- **Risk**: none — additive.

## LG-006 — Real-valued multiplicative clause weights (the 2019 Weighted TM) are not expressible

- **Status**: OPEN. Re-checked 2026-10-17; the cited lines are unchanged.
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

- **Status**: OPEN. Re-checked 2026-10-17: still no `dilation` anywhere in `models/conv.py`. Line
  numbers below are from 2026-09-22; `_encode`'s `unfold` call is now at `conv.py:151-153`.
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
- **Scope grew since filing**: `_Conv3dMixin` (added `a3eba41`) builds its windows with three
  `Tensor.unfold` calls (`conv.py:557`) and needs dilation as a triple, threaded through its own
  `_grid` / `conv_features`. `models/segmentation.py::_DenseMixin` should be checked for the same.
- **Affected**: `src/torchtsetlin/models/conv.py`, possibly `src/torchtsetlin/models/segmentation.py`.
- **Test to add**: `dilation=2, patch_size=3` on a 32×32 input gives `Py = Px = 28` and
  `conv_features` accounting that matches `unfold`'s output width.
- **Risk**: additive; `dilation=1` is the current behaviour.

## LG-008 — The Booleanization registry is missing four of the seven encodings behind the TM state of the art

- **Status**: OPEN. Re-checked 2026-10-17: `data/encoders.py` now also ships `PyramidEncoder`
  (`encoders.py:385`), but none of the four encoders below. Line numbers for the two adaptive
  encoders are now `encoders.py:269, 315`.
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

- **Status**: OPEN, no fix proposed.

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

- **Status**: NOT A LIBRARY GAP. This belongs in `ARMS.md`/`DECISIONS.md`; it stays here only
  because `ARMS.md` cites it as `[FACT: LG-010 A1–A4]` in its registry and ambiguity table, and
  moving it would break those references. Do not add new paper-configuration questions here.

Not a defect; filed here because `ARMS.md` asks P1 to settle these before any `GAP` verdict.

- **A1 — "number of clauses": per class or total?**
  `[FACT: Granmo et al. 2019, Table 1]` the CTM paper's row is literally *"#Class Clauses"*, and
  `[FACT: arXiv:2301.08190 fn. 6]` CSC-TM says "8000 clauses **per class**". `[FACT: arXiv:2309.04801
  Table 1 vs §4]` TM Composites writes an unqualified "2K weighted clauses" in Table 1 and
  "8 000 weighted clauses **per class**" in §4 of the same paper. `[FACT: arXiv:2406.00704 Tables
  II–IV]` and `[FACT: arXiv:2507.14874 Table 16]` are unqualified. **Verdict**: the harness
  convention (`hp.n_clauses` = total) is right; but `ctm-vanilla` configured "as the 2019 paper" is
  **80 000 total**, not 8 000. *(Update, superseding "settle empirically":)* the Toolbox reading
  is also per class — confirmed in `[FACT: arXiv:2406.00704 §II-B]` and in the reference
  implementation (`cair/tmu`, one clause bank per class), so its "2,000" is 20 000 total and its
  "64,000" is 640 000 total. `ctm-vanilla-8k` remains useful as a measurement of the
  unqualified reading, but it no longer decides the question.
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
- **Status**: OPEN. Re-checked 2026-10-17.
- **Cause**: `src/torchtsetlin/models/conv.py:157-160` (was `:132-135` when filed) — the check is
  against the final `self.n_features` rather than against `Z*kh*kw + position bits`. The same
  check is duplicated in `_Conv3dMixin._encode` (`conv.py:563-565`), added since.
- **Impact here**: none to correctness and none to throughput — `_chunk_elements_per_example`
  reads `self.n_literals`, which is `2 * conv_features(input_shape)`, so the chunk budget stays
  correct in both routes ([MEASURED: `code/repro/skip_literals.py`] chunk 8/8 at batch 8 in both).
  Cost is ~10 duplicated lines per skip arm.
- **Proposed fix (NOT APPLIED)**: split `_encode` into `_patch_features(xb) -> (B, P, F_base)` and
  a thin literal-forming wrapper, and move the assertion into the wrapper. Subclasses then
  override `_patch_features` and compose cleanly. Apply the same split to `_Conv3dMixin._encode`.
- **Affected**: `src/torchtsetlin/models/conv.py` (2D and 3D mixins).
- **Test to add**: a subclass appending `n_skip` global literals reports
  `n_literals == 2*(base + n_skip)` and trains.
- **Risk**: none — a refactor behind an unchanged public surface.

