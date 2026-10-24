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
| G2 | No `max_weight` cap (PyTsetlinMachineCUDA stores uint8 weights ≤ 255). | — | matching the CUDA demos only | open, low priority |
| G3 | Regression feedback probability is `|err|/T`. pyTM, pyTM-Par and tmu use `(err/T)²`; CUDA uses `|err|/(2T)`. The docstring cites Abeyrathna et al. without noting the departure. | `models/regression.py:115-125`; pyTM `ConvolutionalTsetlinMachine.c:564` (`pow(1.0*prediction_error/tm->T, 2)`) | B10 | **fixed 2026-10-22**: `err.square()`; test `test_regression_feedback_probability_is_squared_error` |
| G4 | Focused negative sampling weights a class ∝ `v − min(v) + 1`; tmu and green-tsetlin use ∝ `(T + clamp(v, −T, T)) / 2T`, with the target class excluded. | `functional.py:142`; tmu `coalesced_classifier.py:154-173`; GT `feedback_block.hpp:84-110` | B6 (tt row) | **fixed 2026-10-22**: reference rule, new `T` argument; test `test_focused_negative_sampling_follows_reference_rule` |
| G5 | With a clause budget, a Type Ia event that the budget turns into Ib still raises the clause weight. pyTM does not do this. | `models/classifier.py:173-183`, `functional.py:241-252` | B6 (minor) | open |
| G6 | Automaton states are stored as int32, so the model is 4× the size of every other library's (8-bit) state. uint8 is not safe today, because a negative delta is cast to the state dtype before the clamp. | `functional.py:272`, `models/base.py:95` | model-size metric | open |
| G8 | **CPU training is 7–35× slower than single-threaded pyTsetlinMachine and does not scale with threads.** Smoke runs, 1 epoch: B4 (MNIST, 20k clauses) tt-CPU×16 batch-32 1,259 s vs pyTM 36 s; B6 C=5000 tt-CPU 1,305 s (1 thread) / 1,144 s (4) / 1,102 s (16) vs pyTM 153 s. Dense float matmuls on small batches; no bit-packing. | `results/smoke/B4`, `results/smoke/B6` | every CPU efficiency row | found 2026-10-23; open |
| G9 | **The exact sequential mode is slow, even on the GPU.** 406 s/epoch on B4 (one `update` per example), against 36 s for pyTM on one core; on the CPU 25,000 s/epoch. | `results/smoke/B4/*tt-*seq*` | tt-seq rows | found 2026-10-23; open |
| G7 | No constructor flag for clamped prediction votes, which pyTM and CUDA use (argmax over votes clipped to [−T, T]); it is a class attribute only. | `models/base.py:285` | diagnostics only | open, low priority |

## Defects in the other libraries (not ours; recorded because they affect the protocol)

| Library | Defect | Handling |
|---|---|---|
| PyTsetlinMachineCUDA 0.1.9 | The last 32-literal chunk of automata is never initialised (`kernels.py:312`, loop runs to `LA_CHUNKS-1`), so it starts from whatever is in GPU memory. | One fresh process per run. Disclosed. |
| tmu 0.8.3 | Fails under numpy 2 (`np.uint32(~0)` OverflowError); `seed=0` hangs; GPU platform silently falls back to CPU if PyCUDA fails. | numpy 1.26.4 venv; seeds `2k+1`; the adapter asserts the CUDA clause bank. |
| green-tsetlin 1.1.0 | `Trainer` defaults to restoring the best eval-set state (test-set selection if eval = test); conv TM fails at `train()` via the public API. | `load_best_state=False`; GT is n/a for B2, B7, B8. |
| pyTsetlinMachine 0.6.6 | The RNG cannot be seeded (fixed constant); the regressor cannot be unpickled. | Seed k = a permutation of the training order, applied to all libraries. |
