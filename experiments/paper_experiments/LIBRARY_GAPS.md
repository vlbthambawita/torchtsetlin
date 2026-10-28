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

## Round 3 (2026-10-28): low-level kernel audit of 0.5.1 (CUDA / PyTorch)

Requested by the user after the 0.5.1 release: find what is left at kernel level. Measured on the
**RTX 3080** (GPU 1, CPU cores 24–27) so that the GPU timing run on the 3090 was not disturbed;
3080 bandwidth 760 GB/s (3090: 936), so absolute times are ~20% above the 3090's. Scripts:
`code/kernel_audit/` (`deep.py` anatomy per update, `syncs.py` sync sites, `proto_*.py` /
`kvar.py` / `graph_seq.py` prototypes, `coldstart.py`). Every prototype checks exactness
against the current path where that applies. **Status (2026-10-28, same day): all ten fixed in
the working tree** at the user's request ("correct these all GAPs"), which lifted the 2026-10-24
"compile first, no hand-written kernels" decision for these items; see the outcome table at the
end of this section.

**Where an update's time goes (0.5.1, per `update()`):**

| Shape | Wall | GPU busy | Kernels | Host syncs | Dominant cost |
|---|---|---|---|---|---|
| B4 flat, 31M automata, batch 32 | 6.8 ms | ~3.6 ms | 85 | 6 | fp32 count tensors: 3 GEMMs write 375 MB, the commit reads them, a zero pass rewrites them (~1.6 GB of traffic; floor 0.125 GB) |
| B9 flat, 200M automata, batch 32 | 20 ms | ~20 ms | 93 | 6 | same, 6.4× larger; 2.4 GB of count buffers (peak GPU 3.4 GB vs pyTM-CUDA 2.1 GB) |
| B7 conv, 5.4M automata, batch 16 | 7.2 ms | ~3.6 ms | 148 | 8 | evaluation GEMM (compute-bound, ~80% of fp16 peak), then `== 0` and `any` over the materialised (B, P, C) tensor, `nonzero` syncs for the patch draw |
| B1 flat, 480 automata, batch 32 | 1.37 ms | 0.20 ms | 95 | 4 | launches and host syncs (85% of wall) |
| B1 / B2 sequential, per example | 0.54 / 0.78 ms | ~0.03 ms | 22 / 36 | ~0 | launch-bound: ~22 kernels per example |
| B7 conv prediction, 10k images | 1.72 s | | | | (B, P, C) GEMM output + `== 0` + `any`, chunks of 18 images |

| ID | Gap | Evidence (3080) | Fix | Needs |
|---|---|---|---|---|
| G15 | **Count tensors are fp32 and three-fold.** `n_true`, `n_false`, `n2` are (C, 2F) fp32, written by `addmm_`, read by `commit_dense`, then zeroed: 12 B per automaton, ~13× the floor (state r/w + include write). | `proto_flat.py`: old path 6.0 ms (B4, b32), 4.6 ms (b100), 24.2 ms (B9 shape) | Counts in **fp16** (exact: ≤ B ≤ 2048 per chunk; GEMM on tensor cores); **no `n_false`** (= Type Ia count per clause − `n_true`, derived in-kernel); **`mm` writes, no zero pass** (accumulate only across chunks). | torch + compile only |
| G16 | **`commit_dense` fuses the `include_count` row sum**, which inductor schedules as a reduction kernel at half the speed. | `kvar.py`: 2.14 ms with the row sum, 1.00 ms without (31M automata) | Compute `include_count` from the fp16 include in a separate reduction (~0.08 ms). With G15: B4 commit+counts 6.0 → ~1.3 ms (~4.6×), B9 ~2.5–3× | torch + compile only |
| G17 | The bisection sampler costs ~0.5 ms of the 1.0 ms kernel (log2(width) dependent table gathers per automaton, also where the count is 0). | `kvar.py`: rand-only 0.53 ms vs bisect 1.0 ms | Skip the search for zero counts with masked loads (most automata get no event), or a 2-level lookup; only a hand-written Triton kernel controls this | Triton |
| G18 | **Host syncs in every update** (blocks the CPU from queueing ahead, and makes CUDA-graph capture impossible): label range checks `bool((y < 0).any())` (`classifier.py:125`, `coalesced.py:104`, 2/update), `p_class[ar, y] = …` advanced-index assignment (`classifier.py:172,174`, 2), the CDF table copied host→device on every commit (`functional.py:316`, 1), `int(events.max())` for the table width (`functional.py:891`, 1), conv `nonzero` for the patch draw (`conv.py:209`, 4). | `syncs.py` | `torch._assert_async` (or check on the host before the transfer); `scatter_`; cache the table per device; a static event bound from the batch size; dense patch draw (G20) | torch only |
| G19 | **Small models and sequential mode are launch-bound** (B1 batch: GPU busy 0.20 of 1.37 ms; sequential: ~22 launches, 0.54 ms per example). pyTM-CUDA runs B1 at 0.014 s/epoch, tt-seq at 2.8 s. | `deep.py`; `graph_seq.py`: **CUDA-graph replay of 64 sequential steps: 545 → 25 µs per example (22×)**, learns (B1 0.955 after 20 epochs) | CUDA-graph capture of K sequential steps (and of whole batched updates for small models). Prerequisites: G18 (no syncs) and in-place buffer management (`_refresh_include` / `reset_parameters` re-allocate tensors, which a captured graph would keep writing). Verify RNG advances per replay (distribution test) | torch only |
| G20 | **Conv evaluation materialises (B, P, C)** in fp16, then bool, then reduces over P; the training draw needs `nonzero` + gathers. | `proto_conv.py` / `proto_conv2.py`, one B7 prediction chunk (18 images): current 1.32 s/10k; fused fp16 0.98 (1.3×); **fused int8 0.50 (2.6×); fused bit-packed 0.43 (3.1×)**, all exact, untuned | One Triton kernel per (example, clause tile): violation GEMM over patch blocks, OR in registers, and for training the random matching patch per (example, clause) by keeping the max of `U * match` (independent keys for Type Ia / II). Removes the (B, P, C) tensors and the `nonzero` syncs | Triton |
| G21 | Include cache is 16 bits per automaton on CUDA (fp16) next to the 8-bit state; references keep 1 bit. | B9: 400 MB include cache | Bit-packed include (32 literals per int32) written by the commit kernel; used by G20's kernel | Triton (pack in the commit epilogue) |
| G22 | Even with G15/G16 the flat commit moves fp16 counts through memory; the floor (no count tensors) is 0.165 ms for B4 vs ~1.3 ms after Phase 1. | `proto_flat.py` floor line | One Triton kernel per (clause, literal) tile: the B-length dot products of the count GEMMs in registers, then sampling and the state/include update; counts never touch HBM | Triton |
| G23 | Compile start-up per process: first update 5.2 s cold, 2.1 s warm (inductor FX cache on). B3 (100 splits) and B10 (25 splits) pay it once per process. | `coldstart.py` | Keep the cache warm across processes (default dir); optionally `torch.compiler` cache artifacts shipped with the package | config |
| G24 (minor) | Conv literals built in fp32 (`unfold` of a float copy, `cat` with `1 - x`) and cast to fp16 for evaluation | `deep.py` (im2col + cat ~0.2 ms of B7) | Build fp16 / int8 literals directly on CUDA | torch only |

Not worth changing: the flat evaluation GEMM (≤ 0.1 ms even for B9, already fp16 on tensor
cores); RNG (Philox in-kernel costs ~0.1 ms per 31M draws).

### Plan

| Phase | Gaps | Approach | Expected (from the prototypes) | Effort / risk |
|---|---|---|---|---|
| 1 | G15, G16, G18, G24 | torch + `torch.compile` only (keeps the 2026-10-24 decision) | flat GPU update ~3–4× faster (B4 b32 ≈ 1.2–1.5 ms on the 3090, B9 ≈ 2.5–3×); peak GPU memory of B9 −1.6 GB; syncs 6 → 0–1 | ~1 day; distribution tests exist (`test_commit_dense_matches_apply_feedback`) and extend directly |
| 2 | G19 | CUDA graphs for sequential mode (opt-in first, then default on CUDA when the shapes are static), and for batched updates of small models | sequential ≈ 20× (B1 2.8 → ~0.13 s/epoch; B2 ≈ 15×) | ~1–2 days; must prove RNG freshness per replay and in-place buffers |
| 3 | G20, G21, G22, G17 | hand-written **Triton** kernels with the current compiled/eager paths as fall-back and as the exactness reference in tests | conv evaluation 2.6–3.1× (B7 training est. 15.5 → ~7–9 s/epoch, pyTM-CUDA 9.5; prediction 1.7 → ~0.6 s), flat commit at the bandwidth floor (B4 ≈ 0.3 ms) | ~1 week; **needs the user to lift the "compile first, no hand-written kernels" decision of 2026-10-24** |
| — | G23 | inductor cache hygiene | −3 s per process | trivial |

Any of these changes tt's speed (not what it learns), so tt rows measured with 0.5.1 would be
re-run after a release; the PyTsetlinMachineCUDA and tmu rows are unaffected.

### Outcome (implemented 2026-10-28; same GPU, RTX 3080, ms per `update()`, 0.5.1 → now)

Code: `src/torchtsetlin/_triton.py` (new), `models/base.py`, `models/conv.py`, `functional.py`,
`models/classifier.py`, `models/coalesced.py`, `utils.py`; tests `tests/test_triton.py` (+ updates
in `test_models.py`, `test_functional.py`); 253 tests pass on CPU + CUDA. Comparison script:
`code/kernel_audit/cmp_versions.sh` (0.5.1 from a worktree of tag `v0.5.1`).

| ID | Fix as implemented | Result |
|---|---|---|
| G15, G16, G22 | **Events mode + fused Triton commit** (`_triton.fused_commit`): a single-chunk CUDA update records `(literals, sel_i, sel_ii, patch_i, patch_ii)`; one kernel computes the counts per (clause, literal) tile (tensor-core dot over the batch for flat models; gather of each clause's patch for conv), samples, writes state + fp16 include + bit-packed include + include counts (atomics). No count buffer is allocated. The compiled `commit_dense` (fall-back) no longer fuses the row sum. | B4 b32 4.67 → 1.11, b100 5.05 → 1.68; B9 20.7 → 6.6 |
| G17 | Sampler: inverse CDF walked arithmetically for the first 8 steps (no memory traffic), table bisection only for draws past 8, both skipped for whole tiles that do not need them; Philox draws shared by 4 automata. | commit kernel 1.84 → 0.68 ms on the 3090 (31M automata) |
| G18 | Device-side label assert (`utils.class_targets`), `scatter_`/`gather` in feedback selection, CDF tables cached per device, table width from the batch size. | host syncs per update 4–8 → 0 (`test_updates_never_synchronise_with_the_host`) |
| G19 | CUDA graphs: `update()` shapes are captured on third use (sequential: 64-example blocks), keyed on buffer addresses + scalar hyper-parameters; graphs own their accumulator. | B1 batch 1.39 → 0.28; B1 seq 550 → 32.6 (per 1000 examples); B2 seq 788 → 65; replays bit-identical to eager (4 seeds × 4 configs) |
| G20, G21, G24 | `_triton.conv_eval`: bit-packed literals and includes, OR over patches in registers, hashed keys for the random matching patch (Type Ia / II independent, uniform: `test_conv_patch_draws_are_uniform_and_independent`); fp16 conv literals; bit-packed include kept fresh by the commit kernel. | B7 b16 5.04 → 1.53, b100 22.7 → 7.6; B7 prediction 1726 → 369 ms per 10k |
| G23 | Batched CUDA path no longer uses `torch.compile` (Triton kernels cache on disk). | first update 5.2 s → 2.2 s cold, 2.1 → 0.7 s warm |

Learning is unchanged in distribution: kernel-level distribution tests (flat and conv, boost /
budget / dropout / s < 2) and end-to-end accuracies (B2 conv batch: 0.992 Triton vs 0.993 eager,
4 seeds; Noisy XOR trainer: 0.721 vs 0.718, 8 seeds). GPU smoke (all 32 tt-cuda cells) in
`results/smoke/`, 0.5.1 tt smoke records moved to `results/smoke_v0.5.1_tt/`.

Still open: the fp16 include cache is kept next to the bit-packed one (2 bytes per automaton, used
by the flat evaluation GEMM and the fall-backs); updates that need more than one chunk (e.g.
`update()` with > ~1500 MNIST examples, or segmentation commits) use the compiled fp32-count path;
sequential mode still costs ~30 µs per example (B1: 0.15 s/epoch vs pyTM-CUDA 0.014 s, which keeps a whole epoch in one kernel);
CPU paths are unchanged (G8).

## Defects in the other libraries (not ours; recorded because they affect the protocol)

| Library | Defect | Handling |
|---|---|---|
| PyTsetlinMachineCUDA 0.1.9 | The last 32-literal chunk of automata is never initialised (`kernels.py:312`, loop runs to `LA_CHUNKS-1`), so it starts from whatever is in GPU memory. | One fresh process per run. Disclosed. |
| tmu 0.8.3 | Fails under numpy 2 (`np.uint32(~0)` OverflowError); `seed=0` hangs; GPU platform silently falls back to CPU if PyCUDA fails. | numpy 1.26.4 venv; seeds `2k+1`; the adapter asserts the CUDA clause bank. |
| green-tsetlin 1.1.0 | `Trainer` defaults to restoring the best eval-set state (test-set selection if eval = test); conv TM fails at `train()` via the public API. | `load_best_state=False`; GT is n/a for B2, B7, B8. |
| pyTsetlinMachine 0.6.6 | The RNG cannot be seeded (fixed constant); the regressor cannot be unpickled. | Seed k = a permutation of the training order, applied to all libraries. |
