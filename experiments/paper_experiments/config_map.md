# config_map.md: Phase 0 configuration audit (PLAN.md section 4.1)

**Status: DRAFT 2026-10-22, written by the research engineer. It must be reviewed by the user before any measured run (PLAN section 6, Phase 0 gate).**
Items marked **[DECIDE]** need a user decision. Every other item is frozen.

Path prefixes used in citations:

| Prefix | Path (all under `experiments/paper_experiments/`, except `tt/`) |
|---|---|
| `pytm/` | `.cache/upstream/src/pyTsetlinMachine-0.6.6/pyTsetlinMachine/` (PyPI sdist; `tm.py` is byte-identical to the installed `.cache/venv-pytm/.../pyTsetlinMachine/tm.py`) |
| `par/` | `.cache/upstream/src/pyTsetlinMachineParallel-0.2.1/pyTsetlinMachineParallel/` (PyPI sdist; `tm.py` identical to installed) |
| `cuda/` | `.cache/venv-pytmcuda/lib/python3.10/site-packages/PyTsetlinMachineCUDA/` (identical to the sdist) |
| `tmu/` | `.cache/venv-tmu/lib/python3.10/site-packages/tmu/` (installed wheel 0.8.3) |
| `tmuC/` | `.cache/upstream/src/tmu-0.8.3/tmu/lib/` (GitHub tag `v0.8.3`; the wheel ships only the compiled `.so`; the Python files of the tag are identical to the installed ones) |
| `gt/` | `.cache/venv-gt/lib/python3.10/site-packages/green_tsetlin/` (identical to the sdist) |
| `gtC/` | `.cache/upstream/src/green_tsetlin-1.1.0/src/` (the C++ the installed `.so` was built from, because GT is sdist-only) |
| `tt/` | `/work/vajira/DL2026/torchtsetlin/src/torchtsetlin/` (0.4.0, HEAD `7b2e612`) |
| `demo/` | `.cache/upstream/{pyTsetlinMachine,pyTsetlinMachineParallel,PyTsetlinMachineCUDA}/` |

The sdist sha256 values are in `.cache/upstream/src/`, next to the tarballs.

Smoke tests run during this audit took a few seconds each, on Noisy XOR. Their scripts are in the session scratchpad and are not results. They are cited as `[SMOKE]`, never as measurements.

---------------------------------------------------------------------------------------------------

## (a) Semantics, library by item

### a.1 Clause count and pool structure

| Library | What `n` means | Structure | Evidence |
|---|---|---|---|
| pyTM | **per class** | one independent TM per class (`K` machines × `n` clauses); polarity alternates, even index `+`, odd index `−` | `pytm/MultiClassConvolutionalTsetlinMachine.c:61-63`; `pytm/ConvolutionalTsetlinMachine.c:235-239` |
| pyTM-Par | per class | same | `par/MultiClassConvolutionalTsetlinMachine.c:48-50` |
| pyTM-CUDA | per class | `K × n` clauses in one array; polarity `1-2*(clause&1)` | `cuda/kernels.py:188-200` |
| tmu `TMClassifier` | per class | one clause bank per class; first half `+`, second half `−` (weights ±1) | `tmu/models/classification/vanilla_classifier.py:107-129` |
| tmu `TMCoalescedClassifier` (B6 only) | **total, shared** | one clause bank; signed int32 weight per (clause, class), initialised ±1 at random | `tmu/models/classification/coalesced_classifier.py:83-92` |
| GT `TsetlinMachine` | **total, shared** | **Coalesced TM.** One pool of `n_clauses` clauses with an int16 weight per (clause, class) (`DenseState.w` has shape `(n_clauses, n_classes)`), initialised ±1 at random. A clause's polarity *per class* is the sign of its weight, and the sign can flip. | `gt/tsetlin_machine.py:22-25`; `gtC/func_tm.hpp:61-73,264-290`; `gtC/func_avx2.hpp:125-138`; `gtC/gt_common.hpp:10` |
| tt `TsetlinMachine` | per class (`n_clauses`); `n_clauses_total = K·n` | first `ceil(n/2)` of each class `+` | `tt/models/classifier.py:79-91` |
| tt `CoalescedTsetlinMachine` | total, shared | int32 weights `(C, K)`, ±1 at random | `tt/models/coalesced.py:75` |

**GT on Noisy XOR, 74%.** GT's `TsetlinMachine` is a coalesced machine, so `n_clauses=10` gives **10 clauses shared by both classes**, against 20 in pyTM. The 74% is also not a final-epoch number. `Trainer` defaults to `load_best_state=True` with the eval set used as the selection set (`gt/trainer.py:52,331-335,353-354`). With the B1 configuration, `n_clauses=10`, FNS and seed 1, the best eval epoch was **0.7440 at epoch 7** and the **final epoch (200) was 0.5042** `[SMOKE]`. In other words, 74% is the best epoch selected on the test set, and the final model is at chance level.

### a.2 T, weights and the weight update

| Library | T semantics | Weight type and init | Weight update | Evidence |
|---|---|---|---|---|
| pyTM | class sum clamped to `[−T, T]` for both learning and prediction; feedback probability `(T + (1−2y)·v)/(2T)` | `unsigned int`, init 1, **no upper cap** | only with `weighted_clauses=True`: Type Ia `w++`; Type II `w--` if `w > 1` (**min 1**) | `pytm/ConvolutionalTsetlinMachine.c:122,242-243,331,346-348,365-367` |
| pyTM-Par | same | same | same | `par/ConvolutionalTsetlinMachine.c` (diff vs pyTM touches only locks, budget and drop) |
| pyTM-CUDA | `THRESHOLD = T` exactly as passed. **The code does not multiply T by `max_weight`.** The demos pass T pre-multiplied by convention (`50*16` with `max_weight=16`, `100*100` with 255; `demo/PyTsetlinMachineCUDA/MNISTDemoWeightedClauses.py`, `FashionMNIST...py`). Targets are encoded `±T` and every class is trained (see a.10). | **`unsigned char`**, init 1; `max_weight` is capped at **255** (`min(max_weight,255)`); `max_weight=1` (default) means unweighted | `MAX_WEIGHT>1` only: Type I with clause output 1 gives `w++` if `w < MAX_WEIGHT`; Type II gives `w--` if `w > 1` | `cuda/tm.py:38-44,308-310`; `cuda/kernels.py:121-172,308` |
| tmu `TMClassifier` | class sum clamped for learning; **prediction unclamped** by default (`clip_class_sum=False`) | int32, ±1 | only with `weighted_clauses=True`. The target class increments firing clauses with an **independent** Bernoulli(`update_p`) draw (negative weights cannot cross −1); the non-target class decrements (positive weights stay ≥ 1). Magnitude is never below 1. | `vanilla_classifier.py:134-163,217-222,421-447`; `tmuC/src/WeightBank.c:35-65` |
| tmu coalesced | same clamp/no-clamp | int32 ±1 random | **always weighted.** The `weighted_clauses` argument is accepted and never read (`coalesced_classifier.py:47,68` only). Target increments, negative decrements, weights cross zero. | `coalesced_classifier.py:101-215` |
| GT | votes clamped to `[−T, T]` only inside the update probability; **prediction unclamped** | **int16**, ±1 random | per class: if `sign(w)·target > 0` and the clause fires, `w += sign` (Ia); Type II gives `w −= sign`; weights can cross 0. No cap (int16 can overflow; no saturation code). | `gtC/feedback_block.hpp:93-94,114-115,125-141`; `gtC/func_tm.hpp:264-290` |
| tt `TsetlinMachine(weighted=True)` | clamped for learning; unclamped `forward()` | int32, init 1, no cap | `dw = #TypeIa − #TypeII`, then **`clamp(min=0)`**, so a weight can reach 0 | `tt/models/classifier.py:110-116,173-187`; `tt/models/base.py:285` |
| tt coalesced | same | int32 ±1 | `+1` for selected firing target clauses, `−1` for the negative class; crosses zero | `tt/models/coalesced.py:146-183` |

### a.3 `boost_true_positive_feedback`

| Library | Default | Semantics | Evidence |
|---|---|---|---|
| pyTM / Par | **1** | 1: Type Ia increments every True literal deterministically; 0: with probability `(s−1)/s` | `pytm/tm.py:563`; `pytm/ConvolutionalTsetlinMachine.c:371-375` |
| pyTM-CUDA | 1 | same | `cuda/tm.py:38`; `cuda/kernels.py:148-152` |
| tmu | 1 | same | `tmu/models/base.py:83`; `tmuC/src/ClauseBank.c:330-334` |
| GT | **False** | same when True. **The default differs from every other library, so it must be passed explicitly** (the B6 script does pass it). | `gt/tsetlin_machine.py:82`; `gtC/func_tm.hpp:315-326` |
| tt | True | same | `tt/models/base.py:89`; `tt/functional.py` (`p_mem`) |

### a.4 Automaton states, include boundary, initial state

| Library | States | Include iff | Initial state | Evidence |
|---|---|---|---|---|
| pyTM / Par | `2^number_of_state_bits` = 256 (default 8 bits), bit-sliced | MSB set (state ≥ 128) | **127** (all low bits 1, MSB 0), just below the boundary | `pytm/tm.py:564`; `pytm/ConvolutionalTsetlinMachine.c:108-123` |
| pyTM-CUDA | 256 | MSB | **Defect:** the `prepare` kernel initialises only chunks `0..LA_CHUNKS−2`. **The last 32-literal chunk is never initialised**: on a fresh allocation it reads 0 (deepest exclude), and in a reused allocation it keeps whatever was there. For B1 (24 literals, 1 chunk) **every** automaton starts at 0, not 127 `[SMOKE: dumped ta_state_gpu, all words 0x0]`. | `cuda/kernels.py:312-317` |
| tmu | 256 | MSB | 127 | `tmu/clause_bank/clause_bank.py:129-140`; `clause_bank_cuda.py:164-170` |
| GT | int8 in `[−128, 127]` (AVX2 saturating add) or `[−127, 127]` (generic) | `≥ 0` | **random {−1, 0} with p = ½**, so about half of all literals start *included* | `gtC/func_tm.hpp:50-59,304-305`; `gtC/func_avx2.hpp:96-123` (`_mm256_adds_epi8`) |
| tt | `2N`, `n_states=N=128` by default, so 256 | `state ≥ N` | `init="boundary"`: N−1 = 127; `init="random"`: {127, 128} (the GT scheme) | `tt/models/base.py:87,167-178` |

### a.5 s semantics

All libraries apply Type Ia/Ib forgetting with probability `1/s` and memorisation (non-boost) with `(s−1)/s`. They draw the random variables differently:

| Library | Draw | s-range trick | Evidence |
|---|---|---|---|
| pyTM / Par | per clause update: **count ≈ round(Normal(n/s, n/s·(1−1/s)))**, clipped, then that many *distinct* literals uniformly at random (approximately Binomial(n, 1/s)) | `s_range`: clause `j` uses `1/(s + j(s_range−s)/C)`; default `s_range = s`, so no effect | `pytm/ConvolutionalTsetlinMachine.c:146-164`; `pytm/tm.py:205-208` |
| pyTM-CUDA | exact Bernoulli(1/s) per literal | none | `cuda/kernels.py:140-145` |
| tmu CPU | same Normal-approximated stream as pyTM; `s ≤ 1` switches randomness off | none | `tmuC/src/ClauseBank.c:42-60,305-353` |
| tmu GPU | exact Bernoulli(1/s) per literal | none | `tmu/clause_bank/cuda/clause_feedback.cu:153-159` |
| GT (AVX2, the one that runs on the 3950X) | **8-bit quantised:** P(forget) = `(⌊255/s⌋ + 2)/256`. s=3.9 gives 0.2617 (exact 0.2564); s=3 gives 0.3398 (0.3333); s=5 gives 0.2070 (0.2000); s=10 gives 0.1055 (0.1000). GT therefore behaves as if `s` were slightly smaller (2–5% relative). | `s < 1` rejected | `gtC/aligned_tsetlin_state.hpp:62-73`; `gtC/func_avx2.hpp:473-531`; `gt/backend.py:57-64` (AVX2 selected) |
| tt | exact `torch.binomial(count, 1/s)` on aggregated event counts | none | `tt/functional.py:180-190,256-262` |

### a.6 Literal budget

| Library | Argument | Rule | Evidence |
|---|---|---|---|
| pyTM | `max_included_literals` (default: all literals) | **soft:** Type Ia only if the clause has `≤ b` included literals, otherwise Type Ib. Type II and the weights are unaffected; no `w++` when Ia is replaced. | `pytm/ConvolutionalTsetlinMachine.c:362`; `pytm/tm.py:643-644` |
| pyTM-Par | **none** | — | `par/tm.py:237` (no argument) |
| pyTM-CUDA | **none** | — | `cuda/tm.py:38` |
| tmu | `max_included_literals` | soft, same `≤ b` rule (CPU and GPU). The weight update is independent of it. | `tmuC/src/ClauseBank.c:325`; `tmu/clause_bank/cuda/clause_feedback.cu:151,164` |
| GT | `literal_budget` (default None selects the no-budget kernel) | **stricter:** a firing clause with `> b` literals is treated as **not firing for the whole update**. It gets Ib under Type I, **no Type II, no weight change**. | `gtC/func_tm.hpp:239-243`; `gtC/func_avx2.hpp:407-410`; `gt/tsetlin_machine.py:350-369` |
| tt | `max_included_literals`, `budget_mode="soft"` (0.4.0 default) | soft: Ia events of oversize clauses become Ib; Type II still applies; the weight delta still counts the Ia event | `tt/functional.py:241-252`; `tt/models/classifier.py:173-183` |

### a.7 Clause drop

| Library | Support | Granularity | Evidence |
|---|---|---|---|
| pyTM | `clause_drop_p`, `literal_drop_p` | per epoch, inside `fit` | `pytm/MultiClassConvolutionalTsetlinMachine.c:172-206`; `pytm/IndexedTsetlinMachine.c:257-291` |
| pyTM-Par | none | — | — |
| pyTM-CUDA | none | — | — |
| tmu | `clause_drop_p`, `literal_drop_p` | per `fit` call, so per epoch | `vanilla_classifier.py:260-262,392-393`; `coalesced_classifier.py:228-245` |
| GT | none | — | — |
| tt | `drop_clause_p`, `drop_literal_p` | `drop_granularity="batch"` (default) or `"epoch"` | `tt/models/base.py:91-93,413-425` |

No benchmark below uses dropout (B9 is frozen with no clause drop), so these differences do not matter for the comparison.

### a.8 Convolution

| Library | Model | Patch encoding | Position encoding | Evidence |
|---|---|---|---|---|
| pyTM | `MultiClassConvolutionalTsetlinMachine2D(n, T, s, patch_dim, ...)` | stride 1, no padding | **always on, not optional:** `dim_y−patch_y` bits `[y > k]` and `dim_x−patch_x` bits `[x > k]`, plus negations | `pytm/Tools.c:31-128`; `pytm/tm.py:176-318` |
| pyTM-Par | same class | same | always on | `par/Tools.c` |
| pyTM-CUDA | `MultiClassConvolutionalTsetlinMachine2D(n, T, s, patch_dim, max_weight=...)` | same | always on | `cuda/kernels.py:360-455` |
| tmu | `TMClassifier(..., patch_dim=(h, w))` with a 3-D X | same | always on | `tmuC/src/Tools.c:124-220`; `tmu/clause_bank/base_clause_bank.py:40-58` |
| GT | **not usable through the public API.** `ConvolutionalTsetlinMachine` is not exported in `gt/__init__.py` and passes `multi_label` into the `boost_true_positives` slot (`gt/tsetlin_machine.py:383`). Its clause block never gets a patch count, so `Trainer.train()` fails with `RuntimeError: All ClauseBlocks must be init() before constructing an Executor()` `[SMOKE]`. The core has `gtc.tsetlin_im2col` with the same thermometer coordinates (`gtC/input_block.hpp:328-405`), but using it means injecting a private backend class (as `tests/test_gtc_avx2_conv_clause_block.py` does). | — | — | **GT is n/a for B2, B7 and B8** (rule P7: no use beyond the documented API) |
| tt | `ConvTsetlinMachine(n_classes, n_clauses, T, s, patch_size, stride=1, position_encoding=True, input_shape)` | same; `n_features` = `kh·kw·Z + (Py−1) + (Px−1)`, verified 8 for B2 and 136 for B7, equal to pyTM | optional, default True. **Set True to match.** CLAUDE.md warns that PE=True oscillates on tt's own 2-D XOR toy, while pyTM's 2-D Noisy XOR demo reaches 99.97% *with* coordinates. Not tuned away; a B2 result is a finding either way. | `tt/models/conv.py:60-172,285-325` |

Patch sampling: every library updates a firing clause from one uniformly random matching patch. pyTM: `ConvolutionalTsetlinMachine.c:302`. CUDA: `kernels.py:113`. tmu: `ClauseBank.c:137`. tt: `conv.py:196`, with independent draws for Ia and II, which never co-occur for a class-owned clause.

### a.9 Regression TM

| Library | Class | Output and T | Feedback probability | Weights | Evidence |
|---|---|---|---|---|---|
| pyTM / Par | `RegressionTsetlinMachine(n, T, s, weighted_clauses=...)` | `v = Σ w·c`, clamped `≤ T`; y is mapped to the integer `⌊(y−min)/(max−min)·T⌋` from the **first fit's** min/max | **`(err/T)²`** | min 1, no cap | `pytm/ConvolutionalTsetlinMachine.c:513-621`; `pytm/tm.py:990-1058` |
| pyTM-CUDA | `RegressionTsetlinMachine(n, T, s, max_weight=...)` | `class_sum` starts at `CLAUSES`; prediction clamped to `[−T, T]` | **`|err|/(2T)`** (linear, half) | uint8 ≤ 255 | `cuda/tm.py:376-399`; `cuda/kernels.py:123-129,326-328` |
| tmu | `TMRegressor(n, T, s, weighted_clauses=...)` | clamped `[0, T]` for learning, **unclamped** prediction | **`(err/T)²`** | min 1 | `tmu/models/regression/vanilla_regressor.py:120-171` |
| GT | **n/a** (no regression model) | — | — | — | `gt/__init__.py` |
| tt | `RegressionTsetlinMachine(n_features, n_clauses, T, s, weighted=..., y_range=...)` | clamped `[0, T]`; real-valued target in vote units | **`|err|/T`** (linear) | `clamp(min=0)` | `tt/models/regression.py:115-137` |

pyTM's C regression `fit` does **not shuffle** (`pytm/ConvolutionalTsetlinMachine.c:628 "Add shuffling here..."`). tmu shuffles. pyTM's `RegressionTsetlinMachine.__setstate__` calls `CreateTsetlinMachine` with 10 of 11 arguments (`pytm/tm.py:972-983`), so **unpickling a pyTM regressor is broken**; pickling works.

### a.10 Feedback granularity, batching and threads

| Library | Training loop | Shuffle per epoch | Evidence |
|---|---|---|---|
| pyTM | strictly sequential per example: target class with y=1, then **one** uniformly random other class with y=0. Flat `MultiClassTsetlinMachine` defaults to **`indexed=True`** (clause-indexed evaluation, the same algorithm with faster vote sums). The README timings use it. | yes, C `rand()` | `pytm/MultiClassConvolutionalTsetlinMachine.c:135-145,162-231`; `pytm/IndexedTsetlinMachine.c:223-385`; `pytm/tm.py:565` |
| pyTM-Par | **Asynchronous (Hogwild-style):** `#pragma omp parallel for` over examples. Clause outputs and class sums are read without locks; each clause's TA update holds a per-clause lock. With 1 thread it is sequential. | **no shuffle**: examples are processed in array order every epoch | `par/MultiClassConvolutionalTsetlinMachine.c:160-195`; `par/ConvolutionalTsetlinMachine.c` (`omp_set_lock(&tm->clause_lock[j])`) |
| pyTM-CUDA | **`fit(X, Y, epochs, incremental, batch_size=100)`.** `batch_size` is **not** a feedback-aggregation batch. Each kernel launch walks `batch_size` consecutive examples, and every GPU thread updates its clauses *immediately*, example by example. The class sum of example `e` is a **per-example cache** that a clause updates only when it re-evaluates `e`, so the sum other clauses see mixes current outputs with outputs from up to one epoch earlier (stale). `batch_size` therefore sets the synchronisation window (one `cuda.Context.synchronize()` per batch), not the update granularity. Negative classes are not sampled per example: **every** non-target class is trained, each *clause* independently with probability `1/(K−1)`. | **no shuffle** (`for e in range(0, N, batch_size)`) | `cuda/tm.py:178-242,302-340`; `cuda/kernels.py:121-129,175-222` |
| tmu CPU | sequential per example: target class plus one negative class (uniform; FNS only in coalesced); the clause output is recomputed per clause at feedback time. The constructor's `batch_size=100` only chunks incremental clause evaluation **at predict time**. | yes (`self.rng`) | `vanilla_classifier.py:367-419`; `tmu/clause_bank/clause_bank.py:186-200` |
| tmu GPU | sequential per example: one kernel launch per (example, feedback type) with a device-to-host copy of clause outputs; same algorithm | yes | `tmu/clause_bank/clause_bank_cuda.py:185-285` |
| GT | sequential per example and synchronous. `n_jobs` splits the clause pool into `n_jobs` blocks (`set_num_clause_blocks(n_jobs)`) that evaluate and update *the same example* in parallel, with a barrier between vote and update. This is model-parallel and **does not change the algorithm**, but each block has its own RNG seeded `seed+k`, so results differ between `n_jobs` values. | yes (`std::shuffle`) | `gtC/executor.hpp:99-154`; `gt/trainer.py:238-240,270-276`; `gt/tsetlin_machine.py:420-423` |
| tt | `feedback_mode="sequential"` is exactly per example; `"batch"` aggregates all events of `update(x, y)` and commits once (true mini-batch) | caller's loop (adapter) | `tt/models/base.py:300-340` |

**Consequence for D3 (tt-batch size). [DECIDE]** The only library that "exposes a batch size" in `fit` is pyTM-CUDA (default 100; every CUDA demo uses 100, NoisyXOR explicitly). tmu's `batch_size` is a predict-time chunk, and GT has none. Under the literal reading of D3, **tt-batch = 100 on every benchmark**, because CUDA runs all of B1–B10 and the 32/16 fallback never applies. The value is frozen at 100 below. **However, this matches the number, not the algorithm.** CUDA's 100 is an asynchrony/synchronisation window with immediate per-clause updates; tt's 100 aggregates 100 examples' feedback into one commit, and tt's own docs show fidelity degrading at batch 200 on Noisy XOR. The paper must not describe this as "same batch size as pyTM-CUDA" without that caveat. The alternative is 32 (flat) / 16 (conv) everywhere, which is tt's documented range. Changing it is one constant in the adapter; the user should confirm which reading applies.

### a.11 Randomness and seeding

| Library | Seedable? | How we seed run `k` (k = 0..4) | Evidence |
|---|---|---|---|
| pyTM | **No.** `fast_rand` is a static PCG state with a fixed constant and the seeding function is commented out; C `rand()` is never `srand`-ed. Two fresh processes give **bit-identical** models `[SMOKE: same state hash twice]`. | Permute the training-set order with `np.random.default_rng(k)` before `fit`. This was the only way to get run-to-run variance `[SMOKE: permuted order gives a different state]`. Applied to **all libraries** for uniformity. | `pytm/fast_rand.h:127-141` |
| pyTM-Par | No. With `OMP_NUM_THREADS=1` runs are identical; with >1 thread they are **non-deterministic** (data races on `mcg_state` and on reads) `[SMOKE: two 4-thread runs differ]` | input permutation; thread count is recorded | `par/fast_rand.h` |
| pyTM-CUDA | Indirectly. A module-level `XORWOWRandomNumberGenerator()` seeds from `np.random.randint` **at import**. | `np.random.seed(k+1)` **before** `import PyTsetlinMachineCUDA.tm` (same seed gave the same result `[SMOKE]`; atomics still make bit-reproducibility not guaranteed) | `cuda/tm.py:35`; `pycuda/curandom.py:755-760` (venv) |
| tmu | `seed=` seeds the NumPy RNG (shuffle, negative class, drops) and C `pcg32_seed(seed)`, which sets `mcg_state = seed`. **`seed=0` hangs forever**: the PCG state stays 0, `fast_rand()` always returns 0, and the distinct-literal loop never terminates `[SMOKE: timeout]`. Even seeds shorten the period. GPU curand is seeded from global `np.random` at clause-bank creation and ignores `seed`. | `seed = 2k+1` (odd, non-zero) **and** `np.random.seed(2k+1)` before the first `fit` | `tmu/models/base.py:101-104`; `tmu/clause_bank/clause_bank.py:98-103`; `tmuC/src/random/pcg32_fast.c:4-13`; `tmuC/src/ClauseBank.c:53-59`; `tmu/clause_bank/clause_bank_cuda.py:103` |
| GT | `Trainer(seed=...)`. **`seed=0` (the default) is not reproducible**: `FeedbackBlock` uses `std::random_device` and `InitializeTM` refuses seed 0. | `seed = k+1` (the B6 script uses 42) | `gt/trainer.py:50`; `gtC/feedback_block.hpp:30-32`; `gtC/func_tm.hpp:19-20`; `gtC/executor.hpp:86` |
| tt | `torch.manual_seed(k)` (+ `torch.cuda.manual_seed_all`) **before construction**, because coalesced weight init uses the global RNG | as stated | `tt/models/coalesced.py:75` |

### a.12 Data types, fit/predict contract, epoch-by-epoch control

| Library | X / Y | Epoch loop we use | Notes |
|---|---|---|---|
| pyTM / Par | any integer array, cast to `uint32`; flat `(N, F)`, conv `(N, H, W)`; classifier Y cast to `uint32`, regression Y float | `fit(X, Y, epochs=1, incremental=True)` per epoch (the README pattern), then `predict(X_test)` | Every `fit`/`predict` call **re-encodes** the whole array (`tm_encode`), and that cost is inside the README timings (`pytm/tm.py:306-316,688-701`). A first call with `incremental=True` creates the machine. |
| pyTM-CUDA | flat `(N, F)` reshaped to `(N, F, 1)`; conv `(N, H, W)` | `fit(X, Y, epochs=1, incremental=True, batch_size=100)` per epoch | Re-encoding is skipped when `np.array_equal(X, cached)`, but that comparison runs on every call (`cuda/tm.py:181,252`). **Run one process per (benchmark, seed/split).** Reused GPU memory leaks the previous model into the uninitialised last chunk (a.4), which for B3/B10 would carry a previous split's training into the next split's test. |
| tmu | **`TMClassifier` asserts `X.dtype == Y.dtype == uint32`** (`vanilla_classifier.py:380-381`); coalesced and regressor do not; conv uses a 3-D X | `fit(X, Y)` = exactly one epoch (shuffled), state persists | Train and test encodings are cached after the first call, so epoch 1 includes encoding. **Check `type(tm.clause_banks[0]).__name__ == "ImplClauseBankCUDA"` (or `tm.clause_bank`) on GPU runs:** a failed PyCUDA import falls back silently to the CPU bank with only a log warning (`tmu/models/base.py:245-247`; `clause_bank_cuda.py:40-52,431-434`). |
| GT | **X `uint8`, 2-D `(N, n_literals)`; Y `uint32`** (`gt/trainer.py:152-159`) | **one** `Trainer(tm, feedback_type=..., n_jobs=j, n_epochs=E, seed=k+1, load_best_state=False, early_exit_acc=2.0, progress_bar=False, fn_eval_score=hook)` with `set_eval_data(X_test, y_test)` and `train()` once | Do **not** call `train()` once per epoch: each call re-allocates and re-seeds the clause blocks with the same seed (`gt/tsetlin_machine.py:420-423`), so every epoch would replay the same RNG stream. `load_best_state=False` is required: the default restores the best-on-eval state (test-set selection). `early_exit_acc=2.0` is required: the default 1.0 stops at 100% test accuracy, which B1 can hit. `hook(y, y_hat)` stores `y_hat` per epoch (giving predictions for F1/precision/recall at the final epoch) and a `perf_counter` timestamp, and returns accuracy. Train time per epoch comes from `results["train_time_of_epochs"]` (pure `train_epoch`, `gt/trainer.py:312-316`); inference time = timestamp gap − train time (includes the eval-set copy `_flexible_set_data`, which is small). After `train()`, `tm._state` holds the final state. |
| tt | torch tensors resident on the device; flat `(B, F)` bool, conv `(B, 1, H, W)` bool; Y long (classifier) / float (regression) | adapter loop: `model.train()`; per shuffled mini-batch `model.update(xb, yb)`; then `model.eval()`; `model.predict(X_test)` | **`model.eval()` before predicting is mandatory** (empty-clause semantics); the adapter must assert `not model.training`. `torch.cuda.synchronize()` around timers. Never `.to("cpu", non_blocking=True)`. |

### a.13 Reading predictions, class sums and model state

| Library | Predictions / scores | Learned state (dtype, layout) | Save |
|---|---|---|---|
| pyTM / Par | `predict(X)` gives labels only; **no class-sum API** for classifiers; prediction votes clamped to `[−T,T]`, ties go to the lowest class index | `get_state()` gives per class `(clause_weights uint32[n], ta_state uint32[n·⌈2F/32⌉·8])`: bit-sliced, 8 bits per TA, padded to 32-literal chunks | `pickle` (`__getstate__`, `pytm/tm.py:594-601`); regression unpickle broken (a.9) |
| pyTM-CUDA | `score(X)` gives `(K, N)` int32 **clamped** to `[−T,T]` (`cuda/tm.py:288`); `predict` = argmax | `ta_state` uint32 bit-sliced `(K·n·chunks·8)`; `clause_weights` **uint8** `(K·n)` | `pickle` (`cuda/tm.py:60-93`) |
| tmu | `predict(X, return_class_sums=True)`, unclamped | per class `clause_bank` uint32 bit-sliced `(n·chunks·8)` **plus** an always-allocated `clause_bank_ind` (Type III, same size); weights int32 | `pickle` (`tmu/tools.py:32-43`) |
| GT | per-epoch `y_hat` via the hook; `tm.get_predictor().predict(x)` per example; `Predictor.get_votes()` for votes | `DenseState`: `c` int8 `(C, 2F)`, `w` int16 `(C, K)` (`gt/tsetlin_machine.py:22-25`) | `tm.save_state("x.npz")` (uncompressed `np.savez`) |
| tt | `model(x)` gives unclamped votes `(B, K)`; `predict` = argmax | `ta_state` **int32** `(C, 2F)`; `weights` int32. `state_dtype=torch.int16` is safe; **uint8 is not**, because negative deltas are cast to the state dtype before the clamp (`tt/functional.py:272`). | `torch.save(model.state_dict())` |

Model-size metric (a), "analytic size in the library's own format": B4 (MNIST, 10 × 2000 clauses, 1568 literals) gives pyTM/Par/CUDA/tmu ≈ 31.4 MB of TA bits (tmu × 2 with `clause_bank_ind`), GT 31.4 MB (int8), tt **125 MB** (int32). tt's number reflects a storage choice, not learned information. Report the bits needed (8 per TA) alongside.

### a.14 torchtsetlin constructor map

| Benchmark model | tt class and arguments |
|---|---|
| flat, per-class clauses | `TsetlinMachine(n_features=F, n_classes=K, n_clauses=n_per_class, T=T, s=s, weighted=W, boost_true_positive=B, feedback_mode=M)`, so `n_clauses_total = K·n` |
| conv | `ConvTsetlinMachine(n_classes=K, n_clauses=n_per_class, T=T, s=s, patch_size=p, stride=1, position_encoding=True, input_shape=(1, H, W), weighted=W, boost_true_positive=B, feedback_mode=M)` |
| coalesced (B6) | `CoalescedTsetlinMachine(n_features=784, n_outputs=10, n_clauses=C, T=C//4, s=3.0, boost_true_positive=True, max_included_literals=20, focused_negative_sampling=True, feedback_mode="batch")` |
| regression | `RegressionTsetlinMachine(n_features=F, n_clauses=n, T=T, s=s, weighted=True, y_range=(y_tr.min(), y_tr.max()), boost_true_positive=True, feedback_mode=M)` |
| common | `n_states=128`, `init="boundary"`, `budget_mode="soft"`, `state_dtype=torch.int32`, defaults otherwise |

---------------------------------------------------------------------------------------------------

## (b) Frozen per-benchmark arguments

Conventions for every row:

- Data come from the shared `.npz` files (P5). X is cast per library as in a.12 (pyTM/Par/CUDA: `uint32`; tmu: `uint32`; GT: `uint8`; tt: `bool` on the device).
- Run `k` uses the permutation `np.random.default_rng(k)` of the training set (a.11) plus each library's seed: tmu `2k+1`, GT `k+1`, CUDA `np.random.seed(k+1)` before import, tt `torch.manual_seed(k)`.
- One process per (benchmark, library, mode, seed). For B3/B10, one process per split.
- The test set is predicted after every epoch. Accuracy is read at the last epoch; the best epoch is logged only (P4).
- "per epoch" for pyTM/Par/CUDA means `fit(X, Y, epochs=1, incremental=True[, batch_size=100])`; for tmu, `fit(X, Y)`.
- tt-seq: `feedback_mode="sequential"`, adapter batch 1000 (the batch size does not affect the algorithm in sequential mode; it only bounds Python overhead). tt-batch: `feedback_mode="batch"`, **two modes (D9): `batch` = 32 flat / 16 conv, `b100` = 100**. Both shuffle with `torch.randperm` each epoch.
- tmu-GPU = tmu-CPU arguments plus `platform="GPU"`.
- Par runs at `OMP_NUM_THREADS` ∈ {4, 16}, and also 1 for a deterministic reference where it is cheap.
- GT runs at `n_jobs` ∈ {1, 4, 16}.
- **GT common configuration: `feedback_type="uniform"`** (one uniformly random negative class, the classic algorithm that pyTM, tmu and tt run). FNS (GT's default) is used only in B6. **GT clause count in common configs: `n_clauses = n_per_class × K`**, which equals the other libraries' total clauses and automata (budget matching).

### B1: Noisy XOR (12 bits, 5000/5000), 200 epochs × 5 seeds

| Lib | Constructor | Fit / epochs |
|---|---|---|
| pyTM | `MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)` (defaults: `number_of_state_bits=8, indexed=True`) | per epoch ×200 |
| Par | `MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)` | per epoch ×200 |
| CUDA | `MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)` | per epoch ×200, `batch_size=100` (demo) |
| tmu | `TMClassifier(10, 15, 3.9, boost_true_positive_feedback=0, platform="CPU", seed=2k+1)` | `fit` ×200 |
| GT | `TsetlinMachine(n_literals=12, n_clauses=20, n_classes=2, s=3.9, threshold=15, boost_true_positives=False)` | Trainer as a.12, `feedback_type="uniform"`, `n_epochs=200` |
| tt-seq / tt-batch | `TsetlinMachine(12, 2, 10, 15, 3.9, boost_true_positive=False, feedback_mode=...)` | 200 epochs |

### B2: 2D Noisy XOR (4×4, 2500/10000), 5000 epochs × 5 seeds

| Lib | Constructor | Fit |
|---|---|---|
| pyTM | `MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)`, X `(N,4,4)` | per epoch ×5000 |
| Par | same class and arguments | per epoch ×5000 |
| CUDA | `MultiClassConvolutionalTsetlinMachine2D(40, 60, 3.9, (2, 2), boost_true_positive_feedback=0)` | per epoch, `batch_size=100` |
| tmu | `TMClassifier(40, 60, 3.9, patch_dim=(2, 2), boost_true_positive_feedback=0, seed=2k+1)`, X `(N,4,4)` uint32 | `fit` ×5000 |
| GT | **n/a** (a.8) | — |
| tt | `ConvTsetlinMachine(2, 40, 60, 3.9, patch_size=2, position_encoding=True, input_shape=(1,4,4), boost_true_positive=False, feedback_mode=...)` | 5000 epochs |

Cost warning: tt-seq means 12.5 M single-example commits. Estimate the cost from the smoke phase before committing to it.

### B3: Breast cancer, 100 random 80/20 splits, 25 epochs

| Lib | Constructor | Fit |
|---|---|---|
| pyTM | `MultiClassTsetlinMachine(800, 40, 5.0)` (boost default 1); **new object per split** | `fit(epochs=1, incremental=True)` ×25 |
| Par | `MultiClassTsetlinMachine(800, 40, 5.0)` | same |
| CUDA | `MultiClassTsetlinMachine(800, 40, 5.0)`, **new process per split** (a.4) | ×25, `batch_size=100` |
| tmu | `TMClassifier(800, 40, 5.0, seed=2k+1)` | `fit` ×25 |
| GT | `TsetlinMachine(n_literals=F_bin, n_clauses=1600, n_classes=2, s=5.0, threshold=40, boost_true_positives=True)` | Trainer, uniform, 25 epochs |
| tt | `TsetlinMachine(F_bin, 2, 800, 40, 5.0, feedback_mode=...)` | 25 epochs |

`F_bin` = the width of the shared Booleanization. Split `i` uses `train_test_split(test_size=0.2, random_state=i)`; this seed is recorded.

### B4: MNIST flat unweighted (Keras, > 75), 250 epochs × 5 seeds

| Lib | Constructor |
|---|---|
| pyTM | `MultiClassTsetlinMachine(2000, 50, 10.0)` |
| Par | `MultiClassTsetlinMachine(2000, 50, 10.0)` |
| CUDA | `MultiClassTsetlinMachine(2000, 50, 10.0)` (`max_weight=1`, i.e. unweighted), `batch_size=100` |
| tmu | `TMClassifier(2000, 50, 10.0, seed=2k+1)` |
| GT | `TsetlinMachine(n_literals=784, n_clauses=20000, n_classes=10, s=10.0, threshold=50, boost_true_positives=True)`, uniform. **Coalesced, so inherently weighted; "unweighted" cannot be expressed** |
| tt | `TsetlinMachine(784, 10, 2000, 50, 10.0, weighted=False, feedback_mode=...)` |

### B5: MNIST flat weighted, 100 epochs × 5 seeds

| Lib | Constructor |
|---|---|
| pyTM | `MultiClassTsetlinMachine(2000, 5000, 10.0, weighted_clauses=True)` |
| Par | `MultiClassTsetlinMachine(2000, 5000, 10.0, weighted_clauses=True)` |
| CUDA | `MultiClassTsetlinMachine(2000, 5000, 10.0, max_weight=255)`. T is **not** rescaled (a.2); the cap of 255 is the only way to weight. `batch_size=100` |
| tmu | `TMClassifier(2000, 5000, 10.0, weighted_clauses=True, seed=2k+1)` |
| GT | `TsetlinMachine(784, n_clauses=20000, n_classes=10, s=10.0, threshold=5000, boost_true_positives=True)`, uniform |
| tt | `TsetlinMachine(784, 10, 2000, 5000, 10.0, weighted=True, feedback_mode=...)` |

### B6: MNIST clause sweep, copy of `gt_tmu_mnist_dense.py`

Data: `fetch_openml("mnist_784", version=1)`, `train_test_split(random_state=42, test_size=10000)`, `> 75`, X uint8, y uint32 (script lines 22-47). For `C ∈ {10, 100, 1000, 2000, 5000}`: `s = 3.0`, `T = C // 4` (lines 58-61). The timed region is construction, 5 epochs and a test predict after every epoch (lines 77-166). 3 repeats.

| Lib | Constructor (script line) | Loop |
|---|---|---|
| pyTM | `pyTsetlinMachine.tm.MultiClassTsetlinMachine(number_of_clauses=C, T=C//4, s=3.0, boost_true_positive_feedback=1, weighted_clauses=True, max_included_literals=20)` (139-142). **C per class, so 10·C clauses** | 5 × (`fit(epochs=1, incremental=True)`; `predict`) (144-146) |
| Par | `pyTsetlinMachineParallel.tm.MultiClassTsetlinMachine(number_of_clauses=C, T=C//4, s=3.0, boost_true_positive_feedback=1, weighted_clauses=True)` (149-150). **No budget** (unsupported) | same (152-154) |
| CUDA | **`PyTsetlinMachineCUDA.tm.MultiClassTsetlinMachine(number_of_clauses=C, T=C//4, s=3.0, boost_true_positive_feedback=1)`** (157-158). **Unweighted, no budget, per-class C**, default `batch_size=100`. GT's Table I names the *coalesced* CUDA package (`PyCoalescedTsetlinMachineCUDA`), but the script uses this class, and the coalesced package is **not installed** in `venv-pytmcuda`. We follow the script. | 5 × (`fit(epochs=1, incremental=True)`; `predict`) (160-162) |
| tmu-CPU | `TMCoalescedClassifier(number_of_clauses=C, T=C//4, s=3.0, weighted_clauses=True, focused_negative_sampling=True, platform="CPU", max_included_literals=20)` (80-88), plus `seed=2k+1` (the script passes none; we add it, and record it as a deviation). `weighted_clauses` is ignored by the class (a.2). | 5 × (`fit(X, y)`; `predict(X_test)`) (91-94) |
| tmu-GPU | same with `platform="GPU"` (99-107) | same (110-113) |
| GT | `gt.TsetlinMachine(n_literals=784, n_clauses=C, n_classes=10, s=3.0, threshold=C//4, boost_true_positives=True, literal_budget=20)` (116-118 / 127-129) | `gt.Trainer(tm, n_epochs=5, seed=42, n_jobs=j, progress_bar=False, early_exit_acc=2.0)` (120 / 132) **plus `load_best_state=False`** (the script leaves the default True; the change does not alter the timed work except for skipping state copies, and it makes the reported accuracy the final epoch). Feedback is the default FNS. `set_train_data`, `set_eval_data(X_test, y_test)`, `train()` (121-124). `j ∈ {1, 4, 16}` |
| tt-batch | `CoalescedTsetlinMachine(784, 10, C, C//4, 3.0, boost_true_positive=True, max_included_literals=20, focused_negative_sampling=True, feedback_mode="batch")`, batch 32/16 and 100 (D9). Not in the script; it mirrors the coalesced GT/tmu rows. G4 fixed: tt uses the reference FNS rule. | 5 × (shuffled mini-batch updates; `eval()`; `predict`) |
| tt-seq | n/a (PLAN 4.3: tt-seq only on B1–B5, B10) | — |

### B7: MNIST conv weighted (Keras, ≥ 75, 28×28), 30 epochs × 5 seeds

| Lib | Constructor |
|---|---|
| pyTM | `MultiClassConvolutionalTsetlinMachine2D(2000, 5000, 5.0, (10, 10), weighted_clauses=True)` |
| Par | same |
| CUDA | `MultiClassConvolutionalTsetlinMachine2D(2000, 5000, 5.0, (10, 10), max_weight=255)`, `batch_size=100` |
| tmu | `TMClassifier(2000, 5000, 5.0, patch_dim=(10, 10), weighted_clauses=True, seed=2k+1)`, X `(N,28,28)` uint32 |
| GT | n/a |
| tt-batch | `ConvTsetlinMachine(10, 2000, 5000, 5.0, patch_size=10, position_encoding=True, input_shape=(1,28,28), weighted=True)`, batch 32/16 and 100 (D9) |
| tt-seq | n/a (PLAN 4.3) |

### B8: Fashion-MNIST conv weighted (adaptive Gaussian threshold 11, 2), 30 epochs × 5 seeds

The constructors are identical to B7. Only the data change.

### B9: IMDb (5000 selected 1-2-gram features), 30 epochs × 5 seeds

| Lib | Constructor |
|---|---|
| pyTM | `MultiClassTsetlinMachine(10000, 8000, 10.0, weighted_clauses=True)` (Par README configuration; pyTM's own demo uses `s=5.0, clause_drop_p=0.75`, see M17) |
| Par | `MultiClassTsetlinMachine(10000, 8000, 10.0, weighted_clauses=True)` (`demo/pyTsetlinMachineParallel/IMDbTextCategorizationDemo.py`) |
| CUDA | `MultiClassTsetlinMachine(10000, 8000, 10.0, max_weight=255)`, `batch_size=100` (CUDA's own demo: `(10000, 80*16, 27.0, max_weight=16)`) |
| tmu | `TMClassifier(10000, 8000, 10.0, weighted_clauses=True, seed=2k+1)` |
| GT | `TsetlinMachine(n_literals=5000, n_clauses=20000, n_classes=2, s=10.0, threshold=8000, boost_true_positives=True)`, uniform |
| tt-batch | `TsetlinMachine(5000, 2, 10000, 8000, 10.0, weighted=True)`, batch 32/16 and 100 (D9) |

F1 = positive class (label 1).

### B10: California housing regression, 25 random 75/25 splits, 30 epochs

| Lib | Constructor |
|---|---|
| pyTM | `RegressionTsetlinMachine(1000, 5000, 2.75, weighted_clauses=True)`; **new object per split** (the demo reuses one, which freezes `min_y/max_y` at split 1, `pytm/tm.py:993-995`) |
| Par | same |
| CUDA | `RegressionTsetlinMachine(1000, 5000, 2.75, max_weight=255)`, new process per split, `batch_size=100` |
| tmu | `TMRegressor(1000, 5000, 2.75, weighted_clauses=True, seed=2k+1)`, new object per split |
| GT | n/a |
| tt | `RegressionTsetlinMachine(F_bin, 1000, 5000, 2.75, weighted=True, y_range=(y_tr.min(), y_tr.max()), feedback_mode=...)` |

Per epoch: pyTM/Par `fit(epochs=1, incremental=True)` (equivalent to `epochs=30` because pyTM regression does not shuffle); CUDA the same; tmu `fit`.

### Reproduction checks (seed 0 only)

| ID | Lib | Exact call |
|---|---|---|
| R1 | pyTM | `MultiClassTsetlinMachine(2000, 50*100, 10.0, weighted_clauses=True)`, Keras MNIST > 75, 60 × `fit(epochs=1, incremental=True)` + predict (`demo/pyTsetlinMachine/MNISTDemoWeightedClauses.py`) |
| R2 | Par | the same constructor, 100 epochs, `OMP_NUM_THREADS=10` |
| R3 | CUDA | `MultiClassConvolutionalTsetlinMachine2D(8000, 100*100, 10.0, (10, 10), max_weight=255)`, FMNIST adaptive threshold, 30 epochs, default `batch_size=100`. GPU memory for the clause-output cache = 10·8000·60000 B = 4.8 GB, which fits on the 3090. |
| R4 | tmu | `TMClassifier(type_iii_feedback=False, number_of_clauses=2000, T=5000, s=10.0, max_included_literals=32, platform="CPU"` and `"CUDA", weighted_clauses=True, seed=42)`, 60 epochs (`.cache/venv-tmu/.../examples/classification/MNISTDemo.py:25-74`) |
| R5 | GT | B6 rows at C ∈ {1000, 5000}, `n_jobs` ∈ {1, 4}, the script's exact Trainer arguments |
| R6 | tt-seq | the B1 row (published 99.3 ± 0.3) |

---------------------------------------------------------------------------------------------------

## (c) Mismatches that cannot be removed, and their expected effect

| # | Mismatch | Benchmarks | Expected effect |
|---|---|---|---|
| M1 | **Coalesced vs per-class pools.** GT (and tmu-coalesced and tt-coalesced in B6) share one pool with per-class weights; in common configs GT gets `n·K` total clauses (same automata count, different architecture). In B6 the script gives per-class libraries **10× the clauses** of coalesced ones at the same `C`, as GT's Table I does. | all GT cells; B6 | Different accuracy/speed trade-off, which is a model difference, not a library difference. B6 timing compares unequal models; report clauses × classes next to each row. |
| M2 | **GT has no unweighted model and no polarity-fixed clauses**; weights are int16 ±1 initialised at random and free to change sign | B1–B4 | GT's B4 ("unweighted") is really weighted; label it so. |
| M3 | **GT initial TA states are random {−1, 0}** (≈ 50% literals included at start) vs 127/boundary elsewhere | all GT | [HYPOTHESIS] Slower start, since clauses must be pruned before they fire. `[SMOKE]` B1 GT final epoch ≈ chance (0.50–0.54) with both FNS and uniform. **Do not tune; report.** |
| M4 | **GT AVX2 quantises 1/s** to `(⌊255/s⌋+2)/256` (+2–5% relative) | all GT | GT runs at an effectively smaller s (fewer literals per clause). Small accuracy effect. |
| M5 | **pyTM is unseedable** (bit-identical runs); seeds come only from the input order. pyTM-Par with >1 thread is non-deterministic and does not shuffle. CUDA does not shuffle. | all | The pyTM "5 seeds" are 5 input orders. Its variance is comparable with the others only to that extent. State this in the paper. |
| M6 | **pyTM-Par is asynchronous** (lock-free reads, per-clause locks, no shuffle) | Par cells at 4/16 threads | Accuracy can differ from pyTM; it is a different algorithm at >1 thread. |
| M7 | **pyTM-CUDA**: stale per-example class-sum cache; all negative classes trained with per-clause probability `1/(K−1)`; weights uint8 ≤ 255; **last TA chunk uninitialised** (state 0 or leftover memory); no shuffle | all CUDA | Different learning dynamics. The uninitialised chunk delays learning of the affected literals (B1: all of them) and can **leak a previous model** within one process, which is why we use one process per run. The weight cap may bind at T=5000 (B5, B7–B10). |
| M8 | **Prediction clamp**: pyTM, Par and CUDA argmax over votes clamped to `[−T,T]` (ties go to the lowest class index); tmu, GT and tt are unclamped | B4 (T=50 with 1000 positive clauses per class saturates easily) most of all | [HYPOTHESIS] pyTM/CUDA lose accuracy to ties at +T. This is the MC-1 effect in tt's history. |
| M9 | **Regression feedback probability**: pyTM/Par/tmu `(err/T)²`; CUDA `|err|/(2T)`; tt `|err|/T`; pyTM truncates targets to integers; tmu predicts unclamped | B10 | Different learning rates. tt and CUDA update much more often for small errors. The RMSE gap is partly algorithmic. |
| M10 | **Type I randomness**: pyTM/Par/tmu-CPU draw a Normal-approximated *count* of distinct literals; CUDA, tmu-GPU, tt and GT draw exact Bernoulli | all | Negligible for large literal counts; visible only for tiny `2F` (B1: 24 literals). |
| M11 | **Literal budget**: Par and CUDA have none; GT also blocks Type II and weight updates of oversize clauses; tmu does not suppress weight updates; tt counts the weight increment of an Ia that the budget turned into Ib | B6 (budget 20) | B6 rows are not budget-matched. The script defines them that way. |
| M12 | **Weight floor**: pyTM/Par/CUDA/tmu keep magnitude ≥ 1; tt clamps at 0 | B5, B7–B10 (tt weighted) | A tt clause can be silenced and then has to re-grow. Small; see G1. |
| M13 | **Negative-class sampling**: GT default FNS (common configs force `"uniform"`); tt's FNS formula ≠ tmu/GT FNS | B6 | B6 tt vs tmu/GT is not the same FNS (G4). |
| M14 | **tt-batch vs CUDA `batch_size`**: same number, different semantics (a.10) | all tt-batch | No claim of algorithmic equivalence is allowed. |
| M15 | **B6 CUDA class**: the script uses `PyTsetlinMachineCUDA.MultiClassTsetlinMachine` (unweighted, per class, no budget), not the coalesced CUDA package that GT's Table I names; that package is not installed | B6 | Our B6 CUDA row is not the Table I CUDA row. Say so in the paper. |
| M16 | **tmu coalesced ignores `weighted_clauses`**; tmu weight updates use an independent Bernoulli draw | B6 | None beyond what the class defines. Noted for the record. |
| M17 | **B3/B10 Booleanization**: the demos fit `Binarizer` on all X before splitting (test leakage into thresholds); PLAN fits it per training split | B3, B10 | Deviation from the source, in the sound direction. Small. List it under *Deviations*. |
| M18 | **B9 uses the Par configuration** (`s=10`, no clause drop); pyTM's own demo is `s=5.0, clause_drop_p=0.75`, CUDA's is `T=1280, s=27, max_weight=16` | B9 | Common configuration by design (D1). |
| M19 | **pyTM flat default `indexed=True`** (clause indexing) | pyTM flat B1, B3–B6, B9 | Same algorithm and faster, which is what the README timings measure. PLAN 3.2 excludes indexing *as an experiment*; we keep the library default. |
| M20 | **Re-encoding inside timed calls**: pyTM/Par re-encode X on every `fit`/`predict`; tmu caches after the first call; CUDA runs an `array_equal` check every call | timings | Part of each library's normal path, so it stays in. Epoch 1 is reported separately (PLAN 5.2). |
| M21 | **GT reported numbers use best-on-eval selection** (`load_best_state=True` by default) | R5, survey's 99.19% | GT's published accuracies are not final-epoch numbers; ours are. Timings are unaffected. |

---------------------------------------------------------------------------------------------------

## (d) Candidate torchtsetlin gaps (for LIBRARY_GAPS.md; `src/` untouched, `git diff --stat -- src/` empty at audit time)

| ID | Gap | Evidence | Needed for | Fix sketch |
|---|---|---|---|---|
| G1 | Weighted TM/RTM weights floor at **0**, not 1 (every reference library keeps magnitude ≥ 1) | `tt/models/classifier.py:187`; `tt/models/regression.py:137`; vs `pytm/ConvolutionalTsetlinMachine.c:346` | B5, B7–B10 faithfulness | `min_weight` kwarg (default 1 for fidelity, or keep 0 and document) |
| G2 | No `max_weight` cap | — | matching CUDA (uint8 ≤ 255) | optional `max_weight` clamp in `_commit_weights` |
| G3 | **Regression feedback probability is `|err|/T`**; pyTM, Par and tmu use `(err/T)²` (CUDA `|err|/(2T)`). tt cannot express the reference rule, and its docstring cites Abeyrathna et al. without saying it departs from both reference implementations. | `tt/models/regression.py:115-125`; `pytm/ConvolutionalTsetlinMachine.c:564`; `tmu/models/regression/vanilla_regressor.py:127` | B10 (algorithmic RMSE difference) | `feedback_power` (1 or 2) argument; decide which one the paper calls the RTM |
| G4 | Focused negative sampling weights ∝ `v − min(v) + 1`; tmu and GT use ∝ `(T + clamp(v))/(2T)` with the target excluded (and tmu skips the update when all are 0) | `tt/functional.py:125-146`; `gtC/feedback_block.hpp:84-110`; `tmu/.../coalesced_classifier.py:154-173` | B6 tt row | use the reference formula |
| G5 | Budget semantics: an Ia that the budget turns into Ib still increments the clause weight (pyTM does not); GT also suppresses Type II for oversize clauses | `tt/models/classifier.py:173-183` + `tt/functional.py:241-252` | B6 (minor) | apply the oversize mask before `_accumulate_weights` |
| G6 | Automaton storage is int32 (uint8 unsafe: negative delta cast before clamp), so 4× the model size of every other library | `tt/functional.py:272`; `tt/models/base.py:95` | model-size metric | cast delta to a signed dtype before adding, allow `uint8`, or report bits-needed |
| G7 | Prediction-vote clamp is a class attribute (`clamp_prediction_votes`); there is no constructor flag to reproduce pyTM/CUDA clamped argmax | `tt/models/base.py:285` | only for diagnosing M8; not needed for the runs | none required |

G1, G3 and G4 change measured numbers in benchmarks we run. They should go to LIBRARY_GAPS.md and to the user (D7) **before** Phase 2, because B10's tt row, the weighted rows and the B6 tt row are not like-for-like until they are fixed or disclosed.

---------------------------------------------------------------------------------------------------

## Decisions (resolved 2026-10-22)

1. tt-batch size: both, `batch` (32/16) and `b100` (D9). 2. GT clause count: `n_per_class × K`, as frozen (equal automata). 3. GT feedback: `"uniform"` in common configs, FNS in B6, as frozen. 4. G1/G3/G4: fixed in `src/` (D8).

## Open decisions for the user (original audit text)

1. **[DECIDE] tt-batch size (a.10).** The literal D3 reading gives 100 everywhere (frozen above). The alternative is 32 flat / 16 conv. CUDA's `batch_size` is not a mini-batch.
2. **[DECIDE] GT clause count in common configs**: `n_per_class × K` (frozen above; equal automata) or `n_per_class` (as in the B6 script).
3. **[DECIDE] GT feedback in common configs**: `"uniform"` (frozen above; the classic algorithm) or the library default FNS.
4. **[DECIDE] G1/G3/G4**: disclose, or fix in tt before the runs (a fix changes `src/`, so it needs explicit user confirmation under P1/C1).
