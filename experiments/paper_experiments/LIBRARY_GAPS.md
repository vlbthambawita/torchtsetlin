# torchtsetlin gaps found by the cross-library benchmark

Found during the Phase 0 configuration audit (2026-10-22, torchtsetlin 0.4.0 @ `7b2e612`), by
reading the reference implementations (pyTsetlinMachine 0.6.6, pyTsetlinMachineParallel 0.2.1,
PyTsetlinMachineCUDA 0.1.9, tmu 0.8.3, green-tsetlin 1.1.0) next to `src/torchtsetlin`. Details and
file:line evidence: `config_map.md` §(d).

**User decision (2026-10-22): fix G1, G3 and G4 in `src/` before the runs** (explicit exception to
rule P1). The fixes are in the torchtsetlin working tree (CHANGELOG "Unreleased"), and every
result record carries the `git_sha` (with `-dirty-src` while uncommitted). The other gaps stay
open and are disclosed.

| ID | Gap | Evidence | Affects | Status |
|---|---|---|---|---|
| G1 | Weighted TM / RTM clause weights are clamped at **0**; every reference library keeps them at ≥ 1. A clause driven to weight 0 stops voting and stops being useful, but still costs compute. | `models/classifier.py:187`, `models/regression.py:137` (`clamp_(min=0)`); pyTM `ConvolutionalTsetlinMachine.c:346` | B5, B7, B8, B9, B10 | **fixed 2026-10-22** (user decision): `clamp_(min=1)`; test `test_clause_weights_never_drop_below_one` |
| G2 | No `max_weight` cap (PyTsetlinMachineCUDA stores uint8 weights ≤ 255). | — | matching the CUDA demos only | **fixed 2026-10-24**: `max_weight=` on `TsetlinMachine`, `RegressionTsetlinMachine`, `CoalescedTsetlinMachine` (magnitude); test `test_max_weight_caps_clause_weights` |
| G3 | Regression feedback probability is `|err|/T`. pyTM, pyTM-Par and tmu use `(err/T)²`; CUDA uses `|err|/(2T)`. The docstring cites Abeyrathna et al. without noting the departure. | `models/regression.py:115-125`; pyTM `ConvolutionalTsetlinMachine.c:564` (`pow(1.0*prediction_error/tm->T, 2)`) | B10 | **fixed 2026-10-22**: `err.square()`; test `test_regression_feedback_probability_is_squared_error` |
| G4 | Focused negative sampling weights a class ∝ `v − min(v) + 1`; tmu and green-tsetlin use ∝ `(T + clamp(v, −T, T)) / 2T`, with the target class excluded. | `functional.py:142`; tmu `coalesced_classifier.py:154-173`; GT `feedback_block.hpp:84-110` | B6 (tt row) | **fixed 2026-10-22**: reference rule, new `T` argument; test `test_focused_negative_sampling_follows_reference_rule` |
| G5 | With a clause budget, a Type Ia event that the budget turns into Ib still raises the clause weight. pyTM does not do this. | `models/classifier.py:173-183`, `functional.py:241-252` | B6 (minor) | **closed 2026-10-24, documented**: the references disagree. tmu updates the weight regardless of the budget (`vanilla_classifier.py:151` before the budgeted `type_i_feedback` at :166); pyTM gates it (`ConvolutionalTsetlinMachine.c:362-366`). tt's soft budget already follows tmu, so this is kept and stated in the `max_included_literals` docs. |
| G6 | Automaton states are stored as int32, so the model is 4× the size of every other library's (8-bit) state. uint8 is not safe today, because a negative delta is cast to the state dtype before the clamp. | `functional.py:272`, `models/base.py:95` | model-size metric | **fixed 2026-10-24**: default `state_dtype=None` → `uint8` (8 bits per automaton, like every reference); commits form the state in a wider buffer, so narrow dtypes cannot wrap; int32 checkpoints load unchanged; tests `test_default_state_dtype_is_8_bit`, `test_apply_feedback_narrow_state_dtype_is_safe` |
| G8 | **CPU training is 7–35× slower than single-threaded pyTsetlinMachine and does not scale with threads.** Smoke runs, 1 epoch: B4 (MNIST, 20k clauses) tt-CPU×16 batch-32 1,259 s vs pyTM 36 s; B6 C=5000 tt-CPU 1,305 s (1 thread) / 1,144 s (4) / 1,102 s (16) vs pyTM 153 s. Dense float matmuls on small batches; no bit-packing. | `results/smoke/B4`, `results/smoke/B6` | every CPU efficiency row | **partly fixed 2026-10-24**, in two steps. (1) Row-sparse commit + exact sparse Type Ib sampler + reused count buffers. (2) `torch.compile` kernels (user decision 2026-10-24: compile first, no hand-written kernels): the Type Ib forgetting of whole clauses and the slab binomials are exact inverse-CDF kernels with the uniforms drawn inside the kernel (eager CPU RNG is single-threaded; `torch.binomial` was the hot spot), plus clause-major evaluation (`include @ (1-lit).T`, 2.4x faster on the CPU). Harness, B4, 1 epoch from init, seed 0: tt-CPU×16 batch-32 **1,259 → 91 s** (acc 0.9444 → 0.9441), b100 594 → 74 s (`results/smoke/B4` → `results/gapfix/`); GPU batch unchanged (13.4 → 13.5 s). Still 2.5× slower than pyTM on one core (36 s): inductor's CPU `rand` does not scale with threads and is now the floor (~1.8 ns per automaton per Type Ib row). Closing the rest needs a custom RNG or kernel; open. |
| G9 | **The exact sequential mode is slow, even on the GPU.** 406 s/epoch on B4 (one `update` per example), against 36 s for pyTM on one core; on the CPU 25,000 s/epoch. | `results/smoke/B4/*tt-*seq*` | tt-seq rows | **fixed on the GPU 2026-10-24**: flat models in sequential mode on CUDA commit each example with one fused `torch.compile` kernel (`F.commit_single_example`: every count of one example is 0/1, so the binomials are Bernoulli draws; no `(C, 2F)` count tensors, no host syncs), and the per-example clause evaluation + feedback selection is compiled too. B4, 1 epoch from init: **405.7 → 36.6 s** incl. compilation (eager today 127 s), acc 0.9447 → 0.9453 (`results/gapfix/B4_tt-cuda_seq_*`, `code/gap_timing_seq.py`; B4 has no harness seq cell since D10). Now at pyTM's 36 s. The CPU sequential mode is unchanged (~10 ms per example, bound by the per-example evaluation matvec); open. Conv models keep the row-sparse path. |
| G7 | No constructor flag for clamped prediction votes, which pyTM and CUDA use (argmax over votes clipped to [−T, T]); it is a class attribute only. | `models/base.py:285` | diagnostics only | **fixed 2026-10-24**: `clamp_prediction_votes=` constructor argument; test `test_clamp_prediction_votes_constructor_flag` |

## Round 2 (2026-10-28): gaps found in the timing runs on 0.5.0

Found in the Phase 4 timing records (B1–B7) and by profiling. **User decision (2026-10-28): stop
the run, fix the gaps in `src/`, re-run the smoke tests, then the full benchmark for the
selected GPU libraries only.** The fixes are **released as torchtsetlin 0.5.1** (commit `869c045`,
tag `v0.5.1`); the source diff from 0.5.0 is also saved as
`results/torchtsetlin_src_gapfix_2026-10-28.patch`. The GPU smoke records were made from the same
source before the commit (`git_sha` `…-dirty-src`, `lib_version` 0.5.0); the full-benchmark
records report 0.5.1. tt records measured before the fix are kept
under `results/timing_v0.5.0/` and are not mixed with the new ones. Times below are ms per
`update()` after warm-up (profiling script, RTX 3090 / 16 threads); harness numbers are in the
smoke and timing records.

| ID | Gap | Evidence | Affects | Status |
|---|---|---|---|---|
| G10 | **Batched GPU commit is ~20 full passes over the automata** (binomials, clamps, copies, include refresh), 6.5 ms of the 7.2 ms of a B4 update. | `torch.profiler`, B4 batch 32 | every tt-cuda batch / b100 row | **fixed**: one fused `torch.compile` kernel (`F.commit_dense`), binomials by inverse-CDF bisection inside the kernel. B4 batch 32 7.2 → 3.9 ms, b100 7.6 → 4.1 ms (b100: 2.4 s/epoch vs pyTM-CUDA 3.4 s). Tests `test_commit_dense_matches_apply_feedback`, `test_bisection_binomial_matches_the_pmf`, `test_fused_cuda_commits_learn_and_track_caches` |
| G11 | Clause evaluation on CUDA is an fp32 SGEMM (50% of a conv update). | profiler, B7 | GPU rows, conv most | **fixed**: include cache in fp16 on CUDA (tensor cores). Exact: only `violations == 0` is used. Test `test_clause_outputs_in_half_precision_are_exact`, `test_cuda_include_cache_is_half_precision_and_predicts_like_cpu` |
| G12 | **Conv batches split into tiny chunks**: `_chunk_elements_per_example` counted the worst-case patch draw (B·C events), so batch 16 on MNIST ran as 4 chunks, each with its own host syncs. | B7 9.6 ms/update, 1 chunk 3.8 ms | B2, B7, B8 (CPU and GPU) | **fixed**: chunk by the real `(P, C)` block; the draw is blocked by events (`_event_block`). With G10/G11: B7 9.6 → 4.1 ms/update (36 → 15.5 s/epoch; pyTM-CUDA 9.5 s). Test `test_conv_batch_is_evaluated_in_one_chunk` |
| G13 | **tt-seq on the GPU is slow for small and conv models** (B1 13.7 s/epoch vs pyTM 0.008 s; B2 7.8 s, the B2 cell took 11 h): the fused per-example kernel (G9) was skipped below `COMPILE_MIN_ELEMENTS` per step and not implemented for conv. | `results/timing_v0.5.0/B1`, `/B2` | tt-seq rows | **fixed**: compile decision on the work of the whole `update()` call; conv records per-clause matching-patch literals for `F.commit_single_example`. B1 3.1 → 0.55 ms/example (2.8 s/epoch), B2 3.6 → 0.73 ms (1.8 s/epoch). Still ~300× pyTM on B1: launch-bound, see open items |
| G14 | **Two CPU slow paths.** (a) Conv CPU training 2× slower than at smoke time: clause-major evaluation (G8) is 3× slower for conv's 5776 patch rows. (b) b100 on the CPU 6× slower than batch 32 (B6 C2000: 271 vs 44 s/epoch): counts > 63 miss the 64×32 kernel table and fall back to the eager geometric sampler. | `results/timing_v0.5.0/B7`, `/B6` | tt-cpu rows | **fixed**: (a) clause-major only for ≤ 256 rows, B7 619 → 258 ms/update; (b) CPU kernels bisect the wide table, B6 C2000 b100 2047 → 110 ms/update. Test `test_compiled_cpu_kernels_bisect_counts_beyond_the_narrow_table` |

Still open after round 2: CPU training remains slower than pyTsetlinMachine (G8; e.g. conv B7
~970 s/epoch vs 354 s); tt-seq on the GPU is launch-bound for tiny models (~0.5 ms per example
regardless of size); conv on the GPU trails pyTM-CUDA (15.5 vs 9.5 s/epoch on B7).

## Defects in the other libraries (not ours; recorded because they affect the protocol)

| Library | Defect | Handling |
|---|---|---|
| PyTsetlinMachineCUDA 0.1.9 | The last 32-literal chunk of automata is never initialised (`kernels.py:312`, loop runs to `LA_CHUNKS-1`), so it starts from whatever is in GPU memory. | One fresh process per run. Disclosed. |
| tmu 0.8.3 | Fails under numpy 2 (`np.uint32(~0)` OverflowError); `seed=0` hangs; GPU platform silently falls back to CPU if PyCUDA fails. | numpy 1.26.4 venv; seeds `2k+1`; the adapter asserts the CUDA clause bank. |
| green-tsetlin 1.1.0 | `Trainer` defaults to restoring the best eval-set state (test-set selection if eval = test); conv TM fails at `train()` via the public API. | `load_best_state=False`; GT is n/a for B2, B7, B8. |
| pyTsetlinMachine 0.6.6 | The RNG cannot be seeded (fixed constant); the regressor cannot be unpickled. | Seed k = a permutation of the training order, applied to all libraries. |
