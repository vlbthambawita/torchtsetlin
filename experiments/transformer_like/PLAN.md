# TM query-key attention on MNIST — experiment plan

Status: **planned, not started.** Written 2026-10-16 against `torchtsetlin` 0.3.1, revision
`40f17e6` (`git diff --stat src/` empty). Specification under test:
[tm-query-key-attention-mnist-implementation-plan.md](tm-query-key-attention-mnist-implementation-plan.md)
(called *the spec* below; `§n` refers to its sections). The spec is frozen; this plan says how it is
executed in this repository and what is added to answer the user's question: **does query-key
attention improve on the ordinary (flat) TM and on the convolutional TM?**

There are no measurements in this document. Every number is taken from the spec, is derived
arithmetic (marked *derived*), is a library benchmark quoted with its source, or is a preregistered
budget or threshold. Results go to `results/**/*.json`; tables and figures are generated from them.

## 0. Ground rules

- **C1 — the library is read-only.** `src/torchtsetlin/**` is not modified during this programme
  unless the user explicitly confirms otherwise. Defects or missing hooks go to
  [LIBRARY_GAPS.md](LIBRARY_GAPS.md) with symptom, minimal reproduction, impact and a fix sketch.
  A test asserts that `git diff --stat HEAD -- src/` is empty, and every result record stores it.
- **C2 — numbers come from records.** Every number in the report is read from `results/**/*.json`
  by `code/report.py`. Nothing is hand-typed.
- **C3 — the test split is sealed.** The official 10k test set is read once per frozen
  (arm, seed) after `results/frozen_experiment.json` exists. Every test access is appended to
  `results/test_access_log.jsonl`. The orchestration layer refuses an unfrozen manifest (spec §14).
- **C4 — the spec is frozen.** Deviations and clarifications go to [DECISIONS.md](DECISIONS.md)
  (dated, with reason), never silently into code. §2.3 lists the ones already known.
- **C5 — correctness before accuracy.** No MNIST accuracy run until the Phase 1–4 gates pass.
- **C6 — negative results are findings.** The spec says so (§1) and so does this plan. A result
  is admissible only with `argv`, `seed`, `git_sha`, `src_diff` and `env` in its record.

Claim tags as in `experiments/convtm/CHARTER.md`: `[FACT: file:line]`, `[MEASURED: result-id]`,
`[HYPOTHESIS]`. Only `[MEASURED]` can justify a choice.

## 1. Objective and questions

Implement the spec's one-layer, one-head `TMQKAttention` (N0) on top of `torchtsetlin`, compare it
against the flat TM (B0) and the convolutional TM (B1) under one protocol, and run the controls
that separate *attention* from *extra features*, *extra capacity* and *token weight sharing*.

| ID | Question | Answered by |
|---|---|---|
| Q0 | Is the implementation correct (adapter fidelity, attention mathematics, isolation, save/load)? | Phases 1–4 test suite, pass/fail |
| Q1 | Does N0 beat the flat TM at the spec's equal-search budget? | N0 vs B0 |
| Q2 | Does N0 beat the convolutional TM? | N0 vs B1 (and vs B1-s7) |
| Q3 | Is a gain over B0 explained by automaton-state capacity? | N0 vs B0-M |
| Q4 | Is a gain explained by the learned Q/K class evidence alone, without attention? | N0 vs A2, A2-M |
| Q5 | Does QK *selection* matter, or does any aggregation do? | N0 vs A0, A1, A3; plus inference interventions |
| Q6 | Does weight sharing across the 16 tokens alone explain the gain? | N0 vs B1-s7 — **plan addition, not in spec** |
| Q7 | What does N0 cost (encoder + feature + head training time, full-pipeline latency, memory)? | all arms, §7 and §10 |

Every mechanism is a hypothesis. Q1–Q6 may come out negative; a negative answer is reported with
the same care (C6). The spec forbids a unit test requiring N0 > B0 (§15.5); that is the research
question.

## 2. Mapping the spec onto torchtsetlin

### 2.1 Adapter contract (spec §4) → existing API (facts from code)

| Spec operation | torchtsetlin | Notes |
|---|---|---|
| `new_tm(clauses_per_class=M, T, s)` | `tt.TsetlinMachine(n_features, 10, n_clauses=M, T, s, weighted=False, n_states=128)` | `n_clauses` **is per class** (`n_classes * n_clauses` total) [FACT: classifier.py:82]. First `ceil(M/2)` clauses positive, rest negative [FACT: classifier.py:87-88]; even `M` gives equal polarities as the spec requires. |
| `fit_epoch(X, y, shuffle)` | one pass of `model.update(xb, yb)` over shuffled mini-batches | `update` returns pre-update votes; the state lives in the `ta_state` buffer and persists across calls (spec §4 obligation 8 is a test). |
| `predict(X)` | `model.eval(); model.predict(x)` | `argmax` of clamped votes; ties → first maximal index (`torch.argmax` semantics; recorded, D11). |
| `class_scores(X)` raw signed sums | `model.eval(); out = model.evaluate_clauses(x)` then signed sum with the public `clause_class` / `clause_polarity` buffers | `forward()` returns votes **clamped to `[-T, T]`** [FACT: classifier.py:106-112]; raw sums are needed for the saturation diagnostic only (D8). |
| spec §7.1 `u = clip(s/T, -1, 1)` | `model(x) / T` | Exactly the clamp the library already applies. No private API. |
| `clause_outputs(X)` | `model.evaluate_clauses(x)` → `(B, C)` bool, prediction semantics | Empty clause = False in `eval()`, True in `train()` [FACT: base.py:232-250]. The adapter always sets `eval()` for scoring. |
| `state_manifest()` | `n_features, n_classes, n_clauses, T, s, n_states, state_dtype`, SHA-256 of `ta_state`, `include_count` histogram | Used for the frozen-encoder integrity tests (spec tests 27, 33, 35). |
| `save` / `load` | `torch.save({"state_dict", "config"})`, `load_state_dict` | Shape is reconstructed from the checkpoint for lazy models; `_refresh_include` runs in a post-hook [FACT: base.py:132]. |
| "8 automaton-state bits" | `n_states=128` → states `0..255` [FACT: functional.py:267, base.py:168] | 256 states = 8 bits per automaton, stored as `int32` (D4). |
| "disable weighted clauses, Type III, dropout" | `weighted=False`, `drop_clause_p=0`, `drop_literal_p=0`, `max_included_literals=None` | Library has no Type III. Same settings for Q, K, head and every baseline (spec §4). |
| Checkpoint rule (spec §8 Stage B) | `tt.Trainer` + `EarlyStopping(monitor="val_accuracy", patience=10, restore_best=True)` + `ModelCheckpoint(save_best_only=True)` | Both use strict `>` [FACT: callbacks.py:71-74, 108-112], so the **earliest** best epoch is kept and training stops after 10 epochs without a strictly higher value — the spec's rule verbatim. |
| B1 convolutional TM | `tt.ConvTsetlinMachine(10, M, T, s, patch_size=7, stride=1, position_encoding=True, input_shape=(1, 28, 28))` | Sliding window, 22×22 = 484 patches, 49 + 21 + 21 = 91 features per patch, clause = OR over patches [FACT: conv.py:92-106, 164-172]. |

### 2.2 Three facts that make the spec exact in this library

1. **The spec's position bits are the library's thermometer position bits.** Spec §6.2 defines
   `p(r,c) = [1[r>=1], 1[r>=2], 1[r>=3], 1[c>=1], 1[c>=2], 1[c>=3]]`. `_ConvMixin._position_bits`
   produces `1[row > k]`, `1[col > k]` for `k = 0..Py-2` [FACT: conv.py:139-145; checked 2026-10-16: `ConvTsetlinMachine(patch_size=7, stride=7)` → 55 features, `(0,0)→000000`, `(1,1)→100100`, `(3,3)→111111`], which for a 4×4 grid is the same
   six bits in the same order. So `ConvTsetlinMachine(patch_size=7, stride=7)` sees **exactly the
   spec's 55-bit tokens** `b_i`, and differs from N0 only by OR-pooling shared clauses over the
   16 tokens instead of attending between them. That is the plan-added control **B1-s7** (Q6).
2. **Clipping is already the forward pass.** `u^E = clip(s^E / T_E, -1, 1) = model(x) / T`, so the
   Q/K coordinates come straight from the public forward pass; only the diagnostic raw sums need
   the clause outputs.
3. **The pixel threshold is a strict Binarizer.** Spec §6.1 uses `u >= 128` on `uint8`. Implemented
   directly on the `uint8` tensor; it equals `Binarizer(0.5)` (strict `>`) on `u/255`
   (`127/255 = 0.498 < 0.5 < 128/255 = 0.502`). It is **not** the library's MNIST default `0.3`
   (D9); every arm gets the same bits.

### 2.3 Clarifications and deviations to record in `DECISIONS.md` on day 1

| # | Item | Decision |
|---|---|---|
| D1 | Backend. Spec §4 prefers "the project's existing original TM implementation", else TMU. | `torchtsetlin` 0.3.1 at `40f17e6`, through a thin `tm_adapter.py`. Boolean `torch` tensors, not `uint32` arrays. TMU is not installed and not used. |
| D2 | Feedback mode. The spec presumes a per-example native learner. The library default is batched feedback, whose fidelity degrades with batch size (`docs/benchmarks.md`, Noisy XOR; `CHARTER.md` item 6). | **Batch size 32 for every TM fit** (Q, K, head, B0, B1, controls), `feedback_mode="batch"`. Compare arms only at this batch size. Secondary S2 (§3.2) checks the choice with `feedback_mode="sequential"` on a subset. |
| D3 | Device. Spec §4 wants CPU as the portable reference. | Fitting and evaluation run on CUDA (the library's primary, tested path; both GPUs). The attention operator's reference is NumPy float64 on CPU. The smoke profile is run on CPU **and** CUDA and compared (bit-identical after quantization, or the boundary cases are logged). Timing: GPU with `torch.cuda.synchronize()` primary; CPU at `torch.set_num_threads(8)` secondary. |
| D4 | State bits and bytes. Spec §12 estimates 8-bit packed automata with 32-bit padding. | `n_states=128` (256 states, 8 bits logically) stored `int32` (4 B). Memory matching is done on the **automaton count** `C × 2F` (invariant to storage); persistent bytes are reported as stored (`int32`) and as logical 8-bit. No padding term. |
| D5 | Memory-matched counts (spec §12). | *Derived* with D4, defaults: N0 = 34,460,000 automata → **B0-M `M = 2198`** (34,464,640, +0.013 %), **A2-M `M = 988`** (34,461,440). Recomputed from the *selected* N0 after Phase 6 and written to the frozen manifest. |
| D6 | B1 definition. Spec §10 says "7×7 patch size", stride unspecified. | Primary **B1 = stride 1** (the standard CTM). Plan-added **B1-s7 = stride 7** (§2.2 fact 1). Both `position_encoding=True` (MNIST is fine with it on; `CLAUDE.md`). Recorded as two separate rows. |
| D7 | Clause polarity layout. | Library convention (first half positive). Even `M` only (spec §11 rejects odd counts). |
| D8 | Raw scores. | `u = model(x)/T` for the forward model; raw sums from `evaluate_clauses` + polarity buffers for the saturation diagnostic. Test 8 checks `clamp(raw, -T, T) == model(x)`. |
| D9 | Pixel threshold `u >= 128`, not `0.3`. | Spec rule. Stored in every manifest. |
| D10 | Layout and configs. Spec §14 proposes an installable `tm_qk_mnist` package. | Repo convention instead: `experiments/transformer_like/code/` run as `python -m experiments.transformer_like.code.cli ...`; module names follow the spec; configs are YAML (PyYAML 6.0.3 present). |
| D11 | Tie convention. | `torch.argmax` → lowest index. Recorded in `backend_contract.md`. |
| D12 | Plan-added arms/experiments: B1-s7, S1 (weighted clauses), S2 (sequential fidelity). | Separate, labelled rows; never merged with spec arms. |
| D13 | RNG. Spec §11 forbids `hash()`-derived seeds. | `numpy.random.SeedSequence(seed).spawn()` keyed streams for `q-patches`, `k-patches`, `q-init`, `k-init`, `head`; `torch.manual_seed(derived)` immediately before each TM's fit; Q and K fitted sequentially, never interleaved. CUDA kernel determinism is **tested** (same seed → same predictions), not assumed; the outcome is recorded. |
| D14 | Clause-size budget. `CHARTER.md` measured a large gain from `max_included_literals` on CIFAR-10. | Not used in the primary experiment (spec §4: no optional modifications, all arms alike). Noted as a possible later secondary; not run unless the user asks. |

## 3. Arms

Every arm sees the same 50k/10k split, the same `u >= 128` bits, the same batch size 32, the same
50-epoch cap and patience-10 rule, and the same selection key.

### 3.1 Spec arms (primary)

| ID | Model | Input | Purpose (spec §10) |
|---|---|---|---|
| B0 | Flat `TsetlinMachine` | 784 bits | Traditional baseline |
| B1 | `ConvTsetlinMachine`, 7×7, stride 1, position bits | `(1, 28, 28)` | Stronger spatial baseline |
| N0 | `TMQKAttention`: frozen Q/K TMs → scaled QK softmax → value mixing → 3 thresholds → head TM | 3424 bits | Proposed architecture |
| A0 | Uniform attention over active senders, same head | 3424 | Aggregation without selection |
| A1 | Self-only attention, same head | 3424 | Duplicated local features, no exchange |
| A2 | Direct Q/K bits (`u` thresholded at `[-0.5, 0, 0.5]`, zeroed for inactive tokens) + image bits | 1744 | Learned class evidence without attention |
| A2-M | A2 with the head enlarged to N0's automaton budget (D5) | 1744 | Same, capacity-controlled |
| A3 | Shuffled keys (cyclic shift over active slots), fresh head | 3424 | Key/value correspondence |
| B0-M | Flat TM at N0's automaton budget (D5) | 784 | Capacity explanation |

Exact definitions of A0–A3 are spec §10 and are not restated here; `attention.py` implements them
as `mode ∈ {qk, uniform, self_only, shuffled_keys}` and A2 as a separate transform.

### 3.2 Plan additions (D12), reported separately

| ID | Model | Why |
|---|---|---|
| B1-s7 | `ConvTsetlinMachine`, 7×7, **stride 7**, position bits | Identical tokens to N0; isolates token weight sharing + OR pooling from attention (Q6). Tuned on the B0/B1 grid (12 trials, seed 17), so the search ledger grows by 12. |
| S1 | B0, B1, N0 at their *selected* configurations with `weighted=True`, seeds `[23, 41, 59]` | `experiments/linearModel/LIBRARY_GAPS.md` LG-1 measured an unweighted `ConvTsetlinMachine` plateauing ~10 pp below the weighted one under batched feedback. If the primary B1 shows the same plateau, S1 says whether the ranking depends on it. 3 seeds (comparison arm, `CHARTER.md`). |
| S2 | B0 (selected config) and one Q/K pair with `feedback_mode="sequential"` on a 5,000-image stratified training subset, 10 epochs, vs the same with batch 32 | Checks D2: that batch-32 feedback does not change what the encoders and the baseline learn. Diagnostic; validation accuracy and Q/K score statistics compared. |

### 3.3 Interventions (no training)

The trained N0 checkpoint of every final seed is evaluated with attention replaced by the A0, A1
and A3 rules at inference (spec §10 "interventions"). Reported in their own table; they measure how
much the head *uses* the context bits, not what a retrained head would do.

## 4. Data protocol

- Source: MNIST already at `data/MNIST/raw` (loaded with `torchvision.datasets.MNIST(root="data",
  download=False)`; `torchvision` 0.25.0). Download source and raw-file SHA-256 recorded in
  `manifests/dataset_manifest.json`; identical-content duplicates across partitions counted and
  reported, protocol unchanged (spec §5).
- Split: official 60k pool → **50,000 train / 10,000 validation**, stratified, `split_seed = 314159`
  (spec §5), `stratified_take` as in `experiments/linearModel/code/data.py` (largest-remainder
  rounding, sorted indices). Saved as `manifests/split_seed314159.json` with index SHA-256 and class
  counts. Official 10k test sealed (C3). Same indices for every arm and seed; model seeds never
  touch the split.
- Smoke subset (spec §14): 1,000 train / 200 validation, stratified from the main split,
  `manifests/smoke_seed314159.json`. Never touches test.
- S2 subset: 5,000 train, stratified, `manifests/s2_seed314159.json`.
- Split at the **image** level before patch extraction; Q/K patch streams draw only from training
  images (spec §5; tests 6–7).
- Primary preprocessing `u >= 128` for every arm (D9). No augmentation, no grayscale levels.

## 5. Forward model and training, as implemented here

The mathematics is spec §7–§9 verbatim. This section fixes the library-specific choices.

**Tokens.** `tokenization.py`: `unfold`-free row-major reshape to `(B, 16, 49)`, position bits
appended → `(B, 16, 55)`; activity from the 49 pixel bits only; all 16 slots kept. Round-trip test
reconstructs the binary image.

**Stage A — encoders.** Two `TsetlinMachine(55, 10, n_clauses=100, T=50, s=5.0)` (spec §8),
separate seeds and patch streams (D13). Per epoch: 4 active patches per training image, without
replacement where possible, parent label as weak label, all-blank images skipped and counted;
≤ 200,000 patch examples per encoder-epoch (spec). 10 fixed epochs, batch 32, no checkpoint search.
Then frozen: `ta_state` SHA-256 recorded, re-checked before every use (test 27). Diagnostics on a
fixed validation patch sample: patch accuracy, per-class score spread, zero-variance rate, clipping
saturation (raw sums, D8). Diagnostics only — they select nothing (spec §8).

**Attention.** `attention.py`: float32 main path on the fitting device; `attention_ref.py`: float64
NumPy loop reference. Masked softmax with max subtraction, all-masked rows zeroed before softmax,
`alpha[i, j]` = sender `j` → receiver `i`. Spec §15.3 fixture (`context ≈ [0.802224, 0.598888]`
→ bits `[1,1,1]`, `[1,1,0]`) is a unit test.

**Features.** `g(x) = [784 image bits ; 16·55·3 context bits] ∈ {0,1}^3424`, token–feature–threshold
order; A2 `1744`. Cached as `uint8` arrays (*derived*: 50,000 × 3424 B = 171.2 MB per split per
mode) under `artifacts/features/<cache_key>.npy` with a sidecar manifest; the cache key is the
SHA-256 of (split SHA, preprocessing id, encoder checkpoint hashes, mode, temperature, value map,
thresholds, code revision) as spec §9 lists. A different encoder seed is a different key. Caches
are forbidden in timing runs (spec §16).

**Stage B — head.** `TsetlinMachine(3424, 10, n_clauses=M, T=M/2, s)`, `tt.Trainer(batch_size=32)`,
`fit(epochs=50, val_data=…)` with `EarlyStopping(patience=10, restore_best=True)`, `ModelCheckpoint`,
`CSVLogger`. The restored state is the earliest best epoch (§2.1). Q/K receive no feedback.

**Baselines.** B0 and B1 use the identical `Trainer` loop and callbacks on the 784 bits / `(1,28,28)`
planes. `_chunk_elements_per_example` is the library's; `max_chunk_elements` left at default and
recorded, so B1's throughput number is the library's real one (`CHARTER.md` item 3).

**Complete inference** (`models.py::TMQKAttention.predict_images`): raw `uint8` → bits → tokens →
Q/K forward → attention → quantize → concat → head `argmax`, from a single checkpoint file that
holds Q, K, head, all hyperparameters, class order, feature order, mask/threshold conventions and
the library revision (spec §17).

## 6. Search, selection, freeze and final seeds (spec §11)

Tuning seed `17` (not a final seed). Every classifier trial: 50-epoch cap, patience 10, batch 32.

| Family | Grid | Trials |
|---|---|---:|
| B0 | `M ∈ {500, 1000, 2000}` × `s ∈ {3, 5}` × `T/M ∈ {0.25, 0.5}` | 12 |
| B1 | same grid, patch 7, stride 1 | 12 |
| B1-s7 (addition) | same grid, patch 7, stride 7 | 12 |
| N0 | head `M ∈ {250, 500, 1000}` × head `s ∈ {3, 5}` × `temperature ∈ {0.5, 1.0}`, head `T = M/2`; **one** Q/K pair for seed 17 reused across the 12 | 12 (+1 encoder pair) |
| B0-M | `M` from the selected N0 (D5); `s ∈ {3, 5}` × `T/M ∈ {0.25, 0.5}` | 4 |
| A0, A1, A3 | N0's selected head settings and temperature | 0 (no search) |
| A2, A2-M | N0's selected head `s`, `T/M`; `M` = N0's and the matched one | 0 |

`T` from a ratio resolves as `T = max(1, floor(M·ratio + 0.5))` and the integer is saved (spec §11).

**Selection key** (spec §11): highest validation accuracy, then smaller persistent model bytes
(stored `int32`, D4), then earlier best epoch, then lexicographic config id. Selected independently
for B0, B1, B1-s7 and N0. No trials are added because an arm loses. Every trial — including
failures, OOMs and timeouts — is a record in `results/search/`.

**Freeze.** `cli freeze-experiment` writes `results/frozen_experiment.json`: selected configs,
D5 recomputation, seed list `[23, 41, 59, 83, 101]`, checkpoint rule, test procedure, code
revision, split SHA. Committed to git before any test-label read.

**Final runs.** Per seed: one fresh Q/K pair, shared only by N0 / A2 / A2-M / A3 of that seed;
fresh B0, B1, B1-s7, B0-M, A0, A1 heads; paired head-initialisation streams where shapes permit.
Then interventions (§3.3). Then the one test pass per (arm, seed), logged.

*Derived* ledger: 52 search trials + 1 tuning encoder pair; final 5 × (9 spec arms + B1-s7) =
50 head fits + 5 encoder pairs; S1 9 fits; S2 4 fits. The spec's "equal search counts, not equal
wall time" is reported as such, with validation accuracy vs cumulative training time plotted for
the selected runs (encoder fitting and feature generation charged to N0).

## 7. Resource accounting (spec §12, adapted by D4/D5)

Automaton counts `C_total × 2F` (*derived*, defaults; recomputed after selection):

| Component | `C_total` | `2F` | Automata | Stored bytes (`int32`) |
|---|---:|---:|---:|---:|
| B0, `M = 500` | 5,000 | 1,568 | 7,840,000 | 31,360,000 |
| Q encoder, `M = 100` | 1,000 | 110 | 110,000 | 440,000 |
| K encoder, `M = 100` | 1,000 | 110 | 110,000 | 440,000 |
| N0 head, `M = 500` | 5,000 | 6,848 | 34,240,000 | 136,960,000 |
| **N0 total** | | | **34,460,000** | **137,840,000** |
| B0-M, `M = 2198` | 21,980 | 1,568 | 34,464,640 | 137,858,560 |
| A2-M, `M = 988` | 9,880 | 3,488 | 34,461,440 | 137,845,760 |
| B1, `M = 500`, stride 1 | 5,000 | 182 | 910,000 | 3,640,000 |
| B1-s7, `M = 500` | 5,000 | 110 | 550,000 | 2,200,000 |

B1 is tiny on this axis but evaluates every clause on 484 patches, so the second axis required by
`CHARTER.md` — literals evaluated per image — is reported next to it (*derived*, `M = 500`:
B0 7.8 M; N0 head 34.2 M + encoders 3.5 M; B1 ≈ 440 M; B1-s7 ≈ 8.8 M). Any "A beats B" names both
axes plus wall-clock.

Memory at the defaults (*derived*): the head's feedback accumulator is four `(5,000 × 6,848)`
float32 tensors ≈ 548 MB on top of `ta_state` + `include` ≈ 274 MB (`CHARTER.md` item 9: price
memory from clauses × literals); B1 at `M = 2000`, batch 32 has a `(32, 484, 20,000)` match tensor
that the library chunks under `max_chunk_elements = 2^27`. Both fit the 10 GB card. Peak host RSS
and `torch.cuda.max_memory_allocated()` are recorded per run; spec §12's "never assert equal
process memory from equal state counts" holds.

Time: the library's own MNIST numbers [FACT: docs/benchmarks.md, torchtsetlin 0.1.0, RTX 3090,
threshold 0.3] are flat 500 clauses/class at batch 32 ≈ 12,900 images/s and conv 10×10 (361
patches), 200 clauses/class at batch 32 ≈ 10,700 images/s. Scaling by literal work (*derived*,
upper bounds, 50 epochs each): a 3424-feature head ≈ 17 s/epoch → ≤ 15 min per trial at `M = 500`,
≤ 30 min at `M = 1000`; B1 stride 1 at `M = 500–2000` ≈ 16–63 s/epoch → ≤ 13–52 min per trial.
Projected total ≤ ~35 GPU-hours. **Budget ceiling: 48 GPU-hours on the two cards, ≤ 10 days wall.**
These projections are replaced by Phase 2 cost-probe measurements; if the probe projects over
budget, drop B1 `M = 2000` first, then S1, then B1-s7's grid to the B1-selected `M` only. The order
is preregistered here so that no arm is dropped because of its result.

Hardware: RTX 3090 (24 GB) and RTX 3080 (10 GB), 32 cores, 121 GB RAM, torch 2.10.0+cu128,
Python 3.10.11. The 3090 takes B1 trials; the 3080 takes flat/head trials and encoders.

## 8. Phases and gates

Each gate records `git rev-parse HEAD`, `git diff --stat HEAD -- src/` (must be empty) and the test
count in `DECISIONS.md`.

### Phase 0 — scaffolding (½ day)
`DECISIONS.md` (D1–D14), empty `LIBRARY_GAPS.md`, layout (§14), `.gitignore` for `artifacts/` and
`logs/`, `environment.json`, `backend_contract.md` (what §2.1 states, verified by Phase 1 tests),
dataset + split + smoke + S2 manifests.
**G0:** manifests regenerate bit-identically from their seeds; raw-file SHA-256 recorded.

### Phase 1 — adapter and backend contract (1 day)
`tm_adapter.py` and spec tests 8–13 on a tiny 10-class fixture and on a small trained TM: raw sums
equal the manual signed sum; `clamp(raw) == forward`; class order; argmax/tie convention; repeated
`fit_epoch` continues from the previous state (`ta_state` hash changes, never resets); save/load
preserves votes exactly; no stale encodings. Spec test 34 (small deterministic Boolean fixture
learned) and 35 (state hash changes during fitting).
**G1:** all pass on CPU and CUDA.

### Phase 2 — data, tokenization, B0, B1 and the cost probe (1–2 days)
Spec tests 1–7 (threshold boundary 127/128, patch round trip, single-pixel placement, position bits
`(0,0)→000000`, `(3,3)→111111`, blank-but-positioned patches inactive, disjoint IDs and patch
lineage, reproducible streams). `training.py` (Trainer loop, callbacks, timing, memory, record
writer). B0 and B1 on the smoke subset, then **one full-data cost probe each** for B0 (`M = 2000`),
B1 (`M = 2000`, stride 1), B1-s7 (`M = 2000`) and the N0 head (`M = 1000`, on a provisional
feature cache from untrained-shape random bits — a cost probe, not a result), 2 epochs each,
recording seconds per epoch and peak memory. Spec milestone 2 exit: validation accuracy meaningfully
above chance on the smoke run.
**G2:** tests pass; probe projects the §6 ledger inside the §7 budget (else the preregistered
reductions are applied and written to `DECISIONS.md`).

### Phase 3 — attention without training (1 day)
`attention.py`, `attention_ref.py`, modes `qk / uniform / self_only / shuffled_keys`, A2 transform,
quantizer, feature assembly, inspectable intermediates. Spec tests 14–26 and 28–29 (rows sum to 1,
masked rows zero without NaN, shift invariance, orientation, equal keys → uniform, constant scores →
zeros, context in `[0,1]`, joint K/V/mask permutation invariance vs K-only, slot permutation
equivariance, vectorised vs loop agreement, threshold `>=` at `0, 0.25, 0.5, 0.75, 1`, feature
counts 3424/1744 and order, inactive rows zero in every mode, batch-independence, reference vs
optimised bits away from thresholds, boundary sensitivity recorded).
**G3:** NumPy reference and main path agree, including all-masked rows, on CPU and CUDA.

### Phase 4 — encoders (½ day)
`patch_training.py`: deterministic weak-label streams (D13), the two encoders, freeze, diagnostics,
checkpoints, sample manifests, a few patch-score examples. Spec tests 27, 30, 31, 33 (encoder
state unchanged by head training/inference; all-black/all-white images finite and valid; pipeline
identical after save/load; cache invalidation on encoder/temperature/feature change). Spec test 37
(label-shuffle leakage control on the smoke subset) and 36 (100-image over-fit diagnostic).
**G4:** encoders train from training images only, serialize, give finite 10-d vectors; the
diagnostics are written whatever they say (poor patch accuracy is a finding, spec §13).

### Phase 5 — N0 as a pipeline (½ day)
`models.py::TMQKAttention`, `cli smoke` (1,000/200, small even clause counts, 1 encoder epoch,
2 head epochs, same mathematics), raw-image-to-digit inference from one checkpoint, CPU vs CUDA
smoke comparison (D3), spec test 32 (test access refused before freeze).
**G5:** `pytest experiments/transformer_like/tests -q` green on CPU and CUDA; smoke runs end to end
on both devices; `git diff --stat src/` empty.

### Phase 6 — controls, search, freeze (≈ 1 day of work, ≈ 1–2 days of unattended compute)
`search.py` executes the §6 ledger from `configs/search.yaml` on both GPUs (one process per card,
one trial at a time per card), every trial a record, failures included. `resources.py` recomputes
D5 from the selected N0. `cli freeze-experiment` writes and git-commits
`results/frozen_experiment.json`.
**G6:** the frozen manifest exists and is committed; no test label has been read
(`test_access_log.jsonl` empty).

### Phase 7 — five-seed evaluation, interventions, secondary runs (≈ 2–3 days unattended)
`cli benchmark --manifest results/frozen_experiment.json`: refuses an unfrozen manifest; trains the
final arms per seed; runs the interventions; then one test pass per (arm, seed), each logged.
Timing protocol of spec §16 (100-image batch-1 warm-up, first 1,000 test IDs individually, median
and p95; five 256-batch warm-ups, five full passes over 10k, median images/s), caches disabled,
GPU synchronised, CPU secondary at 8 threads. S1 and S2 run after the primary pass, on separate
records. Resumable per completed (arm, seed); never reuses a checkpoint from another config or split.
**G7:** every planned record exists with `status ∈ {completed, failed, timed_out}`; incomplete runs
are listed by name, never silently omitted (spec §17).

### Phase 8 — report (1 day)
`report.py` → `report/REPORT.md` and figures from `results/**/*.json` only (§11).
**G8:** `report.py` runs from a clean checkout; the handoff checklist of spec §21 is filled in with
pointers to records.

## 9. Tests

`experiments/transformer_like/tests/` (run with `pytest experiments/transformer_like/tests -q`;
not part of CI). Fast, deterministic, no stochastic accuracy assertions (spec §15):

| File | Spec tests | Additions |
|---|---|---|
| `test_data.py` | 1, 6, 7 | raw-file SHA check; `load_split("test")` refuses without unseal + frozen manifest and logs when allowed |
| `test_tokenization.py` | 2–5 | B1-s7 tokens: `ConvTsetlinMachine(patch_size=7, stride=7)._encode` positive-literal half equals `tokenization.tokens` (§2.2 fact 1) |
| `test_tm_adapter.py` | 8–13, 34, 35 | `clamp(raw, -T, T) == forward`; `eval()` enforced in scoring; `n_states=128` ⇒ state range `0..255` |
| `test_attention.py` | 14–23 | shuffled-keys rule with < 2 active slots = identity; A0/A1 bypass Q/K (no encoder call) |
| `test_features.py` | 24–26, 28, 29, 33 | cache key changes on every spec §9 ingredient |
| `test_pipeline.py` | 27, 30, 31, 36 | checkpoint carries Q, K, head, conventions, revision |
| `test_isolation.py` | 32, 37 | `git diff --stat HEAD -- src/` empty (C1) |

## 10. Evaluation and preregistered decision rules (spec §16 + `CHARTER.md`)

**Per (arm, seed):** test accuracy and error count, macro-F1, confusion matrix, per-class accuracy,
prediction file (image id, label, prediction, 10 votes), timings (Q, K, feature generation, head
training; total; search cost), batch-1 latency median/p95, batch-256 images/s, persistent bytes,
checkpoint bytes, feature-cache bytes, peak RSS, peak CUDA memory, and for N0 the attention
diagnostics (valid-row entropy, diagonal mass, max weight, effective sender count, deviation from
uniform, cross-image variability, Q/K per-coordinate variance, saturation, zero-variance rate,
fraction of constant context bits; one-active-token images reported separately).

**Across seeds:** mean ± sample SD over the five seeds; paired per-seed differences
`Δ_s = acc(N0_s) − acc(X_s)` for X ∈ {B0, B1, B1-s7, B0-M, A0, A1, A2, A2-M, A3} with their
mean/SD; paired McNemar on shared test predictions per seed; paired bootstrap over test-image ids
averaged over seeds (labelled conditional on these trained models). Five seeds are too few for
strong claims from a tiny difference; the spec and the charter both say so, and the report will.

**Decision rules** (interpretation rules, not predictions). "N0 > X" is **supported** if all three
hold, else "not supported at this budget":

1. all five paired `Δ_s > 0`;
2. mean `Δ ≥ 0.25` percentage points (25 test images) **and** larger than the seed noise band
   (the larger of B0's and N0's across-seed SD);
3. paired McNemar `p < 0.05` for at least four of the five seeds.

| Reading | Requires |
|---|---|
| Q1 attention model beats the flat TM | N0 > B0 supported |
| Q2 beats the convolutional TM | N0 > B1 supported (B1-s7 reported alongside) |
| Q3 not a capacity effect | N0 > B0-M supported; if N0 > B0 but not N0 > B0-M → "capacity trade-off" (spec §16) |
| Q4 not just learned features | N0 > A2 **and** N0 > A2-M supported; otherwise "auxiliary features may explain the gain" |
| Q5 selection matters | N0 > A0, N0 > A1, N0 > A3 all supported **and** each A0/A1/A3 intervention lowers N0's accuracy by more than the seed band |
| Q6 not just token weight sharing | N0 > B1-s7 supported |
| "Attention contributes beyond extra features" (spec completion criterion) | Q3, Q4 and Q5 all supported |

If N0 loses, the result is kept and diagnosed with the spec §18 table; the frozen protocol is not
changed. If S1 reverses a ranking, the report states that the primary (unweighted) conclusion is
conditional on the clause-weighting choice, and S1 is marked as the plan-added secondary it is.

## 11. Reporting

`report/REPORT.md` generated by `code/report.py` from records:

- Main table (spec §16 template): B0, B1, B1-s7, N0, A0, A1, A2, A2-M, A3, B0-M — test accuracy
  mean ± SD, paired Δ vs B0, model bytes (stored and logical), automata, literals/image, total
  train time, batch-1 latency, batch-256 throughput. Cells read "Not run" until a record exists.
- Interventions table; S1 and S2 tables, separately headed.
- Figures 1–8 of spec §16 (architecture side-by-side, patch grid, 16×16 heat-map with receiver
  rows / sender columns, sender overlays for fixed queries, Q/K profiles, validation vs epoch and vs
  cumulative time, accuracy vs bytes and vs latency, confusion matrices and B0/N0/B0-M
  disagreements). Figure ids by a documented deterministic rule, successes and failures both.
- Search ledger (every trial, including failures), timing protocol, environment, incomplete runs.
- Observed results and interpretation in separate sections; the §10 rules applied verbatim.
- Limitations: staged (not joint) training; class-guided patch similarity, not general contextual
  learning; MNIST only; five seeds; batched feedback (D2); device (D3).

## 12. Risks

| Risk | Mitigation |
|---|---|
| Weak patch labels give near-constant Q/K → near-uniform attention (spec §18) | Diagnostics in Phase 4; temperature grid already declared; A0/A2 make the outcome interpretable rather than fatal |
| Head ignores context bits (N0 ≈ B0 whatever attention does) | Interventions (§3.3); fraction of constant context bits; literal-frequency over the context columns via `model.literal_frequency()` |
| Unweighted B1 plateaus under batched feedback (LG-1 in `experiments/linearModel`) | S1 weighted rows; B1 grid includes larger `T/M`; outcome reported either way |
| Batched feedback differs from the spec's per-example learner | D2 fixed batch 32 for all; S2 sequential check |
| Residual image bits make N0 a "B0 with more features" | B0-M and A2-M at matched automata; Pareto plot |
| CUDA nondeterminism prevents bit-reproducible reruns | D13 test; if it fails, reproducibility is claimed at the distribution level only and recorded |
| CPU/GPU disagree at quantizer boundaries | Boundary-sensitivity count per feature cache (test 29); CPU smoke comparison (D3) |
| Encoder over-fit on the training split it transforms | Stated in spec §8; training accuracy never used as evidence; cross-fitting left to a later labelled study |
| B1 runtime blows the budget | Preregistered reduction order in §7 |
| Accidental edits to `src/` | `test_isolation.py` + per-record `src_diff` + gate checks |

## 13. Out of scope

Boolean `TMQKHard` (spec §19; only after this report), multiple heads or layers, learned V, tied
QK, joint optimisation or surrogate gradients, grayscale thresholds, other datasets, text, and
library integration. Each needs its own plan after this programme reports (spec §20).

## 14. Layout and records

```
experiments/transformer_like/
  tm-query-key-attention-mnist-implementation-plan.md   # spec (frozen)
  PLAN.md  DECISIONS.md  LIBRARY_GAPS.md  backend_contract.md  environment.json
  code/      data.py tokenization.py tm_adapter.py patch_training.py attention.py
             attention_ref.py features.py models.py training.py search.py evaluation.py
             resources.py visualization.py cli.py report.py
  tests/     test_data.py test_tokenization.py test_tm_adapter.py test_attention.py
             test_features.py test_pipeline.py test_isolation.py
  configs/   smoke.yaml tm_flat.yaml tm_conv.yaml tm_conv_s7.yaml tm_qk.yaml search.yaml
  manifests/ dataset_manifest.json split_seed314159.json smoke_seed314159.json s2_seed314159.json
  artifacts/ encoders/ heads/ features/   (git-ignored; SHA-256 in every record)
  results/   search/ final/ interventions/ secondary/ frozen_experiment.json test_access_log.jsonl
  logs/
  report/    REPORT.md figures/
```

Each `results/<stage>/<run_id>.json` holds: arm, resolved config (with resolved integer `T`),
seeds (split, tuning/final, derived stream keys), split SHA, encoder checkpoint SHA-256s, feature
cache key, `git_sha`, `src_diff`, `argv`, `env` (torch, CUDA, torchtsetlin version, GPU name,
thread count), per-epoch metrics, best epoch and stop reason, validation metrics, test metrics
(Phase 7 only), timings by stage, sizes (automata, stored bytes, checkpoint bytes, cache bytes),
peak RSS and CUDA memory, attention diagnostics (N0 family), `status`.

CLI (spec §14, repo convention D10):

```bash
python -m experiments.transformer_like.code.cli prepare-data --config configs/tm_qk.yaml
pytest experiments/transformer_like/tests -q
python -m experiments.transformer_like.code.cli smoke  --config configs/smoke.yaml
python -m experiments.transformer_like.code.cli search --config configs/search.yaml
python -m experiments.transformer_like.code.cli freeze-experiment --run-dir results/search
python -m experiments.transformer_like.code.cli benchmark --manifest results/frozen_experiment.json
python -m experiments.transformer_like.code.cli report    --manifest results/frozen_experiment.json
```

## 15. Order of work (checklist)

1. Phase 0 scaffolding, `DECISIONS.md` D1–D14, manifests → G0
2. Adapter + backend-contract tests → G1
3. Data/tokenization tests, Trainer loop, B0/B1/B1-s7 smoke, cost probe, budget decision → G2
4. Attention operator + reference + feature assembly tests → G3
5. Encoders, diagnostics, leakage controls → G4
6. `TMQKAttention` pipeline, smoke on CPU and CUDA, test sealing → G5
7. Search ledger on both GPUs, D5 recomputation, freeze and commit → G6
8. Five-seed final runs, interventions, test pass, S1, S2 → G7
9. Report from records, spec §21 checklist → G8
