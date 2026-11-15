# Rust backend for torchtsetlin: evaluation and plan

*Status: proposal, 2026-11-04. Written against torchtsetlin 0.6.1 (torch 2.10, Python 3.10; cargo 1.99 is
installed on the dev box, maturin is not).*

## Verdict

**The idea as stated — swap PyTorch for a Rust tensor implementation while every line of user code stays
the same — is not a good idea, and this document recommends against it.** The public API *is* PyTorch:
tensors in and out, `nn.Module` buffers, `state_dict`, `.to(device)`, `DataLoader`, `torch.manual_seed`.
There is no backend seam below that API which a different tensor library could fill without either keeping
torch (so nothing is replaced) or re-implementing torch's surface (unbounded work, and the GPU path would
still be torch + Triton).

**A narrower version of the idea is a good idea, and this document plans it:** Rust kernels for the *CPU
hot paths*, called on torch tensors in place, with the same structure, fall-backs and tests that the
Triton kernels in `_triton.py` have on CUDA. The measured gap is on the CPU (section 3): tt-CPU training is
7-8x slower than single-threaded pyTsetlinMachine and 20x slower than its OpenMP build, the exact
sequential mode on the CPU is ~65x slower, and the floor is now the single-threaded CPU random-number
generation inside the update — exactly what a native kernel fixes.

Recommendation: run Phase 0 (a two-day torch-only baseline plus a one-week Rust prototype) and decide on
the numbers at Gate G0. The benchmark runs it needs are subject to the standing "no timing runs without
go-ahead" rule.

## 1. What "only the backend changes" would actually require

Inventory of the torch dependency in `src/torchtsetlin` (7,600 lines):

| Dependency | Where | Replaceable by another tensor library? |
|---|---|---|
| 77 distinct `torch.*` functions, dtypes and device objects; ~15 in-place tensor methods (`index_add_`, `scatter_`, `addmm_`, `masked_fill_`, ...) | everywhere | yes, by a tensor library of the same breadth with Python bindings (none exists in Rust; see §2) |
| `nn.Module`: buffers, non-persistent buffers, `_load_from_state_dict` (lazy shapes), `register_load_state_dict_post_hook`, `_apply` override behind `.to()`, `train()`/`eval()` driving the empty-clause semantics, `parameters()` warning | `models/base.py` | no: this is the framework, not the math |
| `torch.Generator`, `torch.manual_seed`, `torch.binomial`, `multinomial`, `randperm` | `functional`, models, data | partly; the RNG streams would differ by construction |
| `torch.compile` kernels (10 sites), `_DYNAMIC` shapes, the inverse-CDF tables, the eager twins | `functional.py` | no equivalent; would need hand-written kernels |
| Triton kernels (`fused_commit`, `conv_eval`), CUDA graphs (`_graphed_update`, `_GRAPH_FLAGS`), the sync-free update contract | `_triton.py`, `base.py` | no; this is the whole 0.6.0 GPU story |
| `DataLoader`, torchvision datasets, `torch.save`/`torch.load`, `.pt` checkpoints | `train/`, `data/`, users' code | only by replacing the user's data pipeline as well |
| 153 tests, every device-marked one parametrised over `cpu` and `cuda` | `tests/` | would double |

Two rows are the real blockers. The public API is typed in torch tensors: a user calls `model(x)` with a
tensor and gets a tensor back, calls `model.to("cuda")`, `model.state_dict()`, `torch.save`, hands a
`DataLoader` to `Trainer`, and passes tensors to `metrics`, `interpret` and `viz`. "Unchanged syntax"
with a non-torch tensor needs one of:

1. torch stays the front-end tensor type and Rust only computes — then torch is not replaced; it is a
   kernel backend. That is §4.
2. a torch-compatible tensor façade in Rust — hundreds of methods, broadcasting, indexing, dtype
   promotion, device semantics, pickling; never finished, and the first thing users would hit.
3. a new API (NumPy in, NumPy out) — the honest form of a torch-free package, but then the syntax *does*
   change and the product is no longer "Tsetlin machines in PyTorch".

## 2. Why the full replacement is not a good idea

1. **It contradicts the library's identity.** The original task statement, the name, the README and the
   paper all say PyTorch-native. Users choose torchtsetlin to put Tsetlin machines next to their torch
   data pipelines and GPUs. A torch-free build gives those users nothing they asked for.
2. **On the GPU, Rust adds nothing and loses everything.** The CUDA path is Triton kernels, CUDA graphs
   and compiled per-example kernels, tuned over three rounds of kernel audits (LIBRARY_GAPS G8-G25).
   A Rust tensor library with CUDA support (candle, burn) would redo that work worse, and neither ships
   usable Python bindings. Any Rust backend is CPU-only in practice, which is fine for kernels and
   wrong for "the backend".
3. **Two implementations of every behaviour that is easy to get wrong.** CLAUDE.md lists more than a
   dozen (empty-clause semantics, batch-vs-sequential fidelity, soft budget, weight floors, two commit
   paths, cache refresh, where the patch is drawn, chunk sizing, ...). Each would become two code paths
   that must agree. They cannot agree bit-for-bit (different RNG streams), so every cross-backend test
   is statistical, like `test_fused_commit_matches_apply_feedback`.
4. **The library is still moving fast.** Five releases since 2026-10-22 and 25 numbered gap fixes; the
   research programmes keep finding library defects. Every change would land twice, and the
   experiments' rule C1 (`src/` is read-only during an experiment) would have to freeze two code bases.
5. **Packaging changes class.** Today: one pure-Python wheel, setuptools, Trusted Publishing, a
   smoke-import in a clean venv. A Rust *backend* inside the main package means binary wheels per
   platform (and per Python without abi3), a Rust toolchain in CI, and a toolchain failure that blocks
   the release of pure-Python fixes. (A separate optional native package avoids this; §4.5.)
6. **The torch-free use case that exists is deployment, not training.** Someone who wants a Tsetlin
   machine without a 2 GB torch install wants to *run* trained clauses on an edge device or in a
   browser. That is served by exporting the clauses to a small Rust `no_std` runtime (§9), not by
   re-implementing learning.
7. **"A Tsetlin machine in Rust" already exists.** `tsetlin-rs` (MIT; docs.rs and GitHub links at the
   end) has binary, multi-class, regression and convolutional models, bit-packed u64 evaluation, rayon
   training and `no_std`; it has no Python bindings and no GPU. A torch-free torchtsetlin would compete
   with it on its ground while giving up ours.

## 3. Where Rust does pay: the CPU path

All numbers are from committed records: `benchmarks/results/versions/0.6.0/tables.md`,
`experiments/paper_experiments/REPORT.md` and `LIBRARY_GAPS.md`; this box (Ryzen 9 3950X, 16 cores,
torch at 16 threads; RTX 3090).

**B4, MNIST flat, 2,000 clauses/class (20,000 clauses, 31 M automata), T = 50, s = 10; seconds per
epoch, seed-0 timing runs:**

| Library | train s/epoch | relative to pyTM, 1 thread |
|---|---|---|
| pyTsetlinMachine (C, 1 thread) | 9.3 | 1.0x |
| pyTsetlinMachineParallel (C + OpenMP, 16 threads) | 3.0 | 3.1x faster |
| PyTsetlinMachineCUDA | 3.4 | 2.7x faster |
| tmu (C, 1 thread) | 32.4 | 3.5x slower |
| **tt CPU x16, batch 32** | **69.6** | **7.5x slower** |
| **tt CPU x16, batch 100** | **64.3** | **6.9x slower** |
| tt CPU, sequential (exact algorithm) | ~600 (10 ms/example, G9) | ~65x slower |
| tt GPU, batch 100 | 0.79 | 11.8x faster |

**Where a CPU update's time goes (0.6.0 cost breakdown; 5,000 clauses, 784 features, batch 256):**

| stage | CPU ms | CUDA ms |
|---|---|---|
| evaluate | 7.5 | 0.11 |
| votes + select | 9.1 | 0.44 |
| feedback_counts (three `(C, 2F)` fp32 matmuls) | 15.0 | 0.68 |
| accumulator alloc / zero | 3.8 | 0.12 |
| commit (apply_feedback 12.6 + refresh_include 15.3) | 27.9 | 0.80 |
| **update() total** | **57.2** | **1.9** |

**CPU thread scaling (same shape):** 720 examples/s at 1 thread, 3,860 at 8, 4,136 at 16, **379 at 32**.

Why the torch formulation is stuck here, and why a native kernel is the right tool:

- **Evaluation reads a 125 MB fp32 matrix per example in sequential mode.** `include` is
  `(20,000, 1,568)` float32; the per-example matvec is memory-bound (~10 ms). Bit-packed, the same mask
  is 3.9 MB (`(C, 49)` int32 — the layout `include_bits_buffer()` already keeps for Triton), fits in L3,
  and the match test is one `AND` and a zero test per word. Part of this item is available in torch
  alone (Phase 0a); the rest of the list is not.
- **The commit is bounded by CPU random numbers.** G8: after the 0.5.0 compiled kernels the floor is
  inductor's CPU `rand`, which is single-threaded (~1.8 ns per automaton per Type Ib row); eager
  `torch.rand` and `torch.binomial` are single-threaded too. A Rust kernel draws one 64-bit word per
  eight automata (~1 ns) from a per-row generator inside a rayon loop, so it scales with cores, and
  samples Type Ib by geometric skipping: `L/s` draws per row instead of `L`.
- **The count tensors are traffic.** `n_true`, `n_false`, `n2` are three `(C, 2F)` fp32 tensors
  (94 MB at 5,000 clauses), written by matmul, read by the commit, zeroed again. The Triton path avoids
  them by computing counts per `(clause, literal)` tile from the recorded events (`fused_commit`); a
  CPU kernel can do the same.
- **Threading is uncoordinated.** The collapse at 32 threads is OpenMP on SMT siblings; a rayon pool
  sized to `torch.get_num_threads()` with row-sized work units is predictable.

Who benefits: CPU-only users (laptops, CI, teaching, Apple silicon, where there is no Triton); anyone
who needs the exact sequential algorithm at the speed of the C libraries (paper comparisons); and the
paper's open gap G8.

## 4. Recommended design: `_native`, the CPU twin of `_triton`

### 4.1 Principles

- **torch stays the tensor and API layer.** Nothing changes in the public API, in `ta_state`
  (uint8 `(C, 2F)`), in checkpoints, or in `interpret`/`viz`. Native kernels accelerate specific
  `functional` and model hooks, exactly as Triton does on CUDA.
- **In place on torch CPU tensors, zero copy.** PyO3 plus the `numpy` crate; call sites pass
  `tensor.numpy()` views of contiguous CPU tensors (shared memory; writes land in the tensor). numpy is
  already a dependency. Non-contiguous or non-CPU input: decline, eager path.
- **Every kernel has an eager twin and declines cleanly.** `available(device)` is false without the
  wheel, off the CPU, under `TORCHTSETLIN_NATIVE=0` or `F.set_compile(False)`; a kernel returns `None`
  on an unsupported shape or option (hard budget, counts beyond its table, ...) and the caller takes the
  existing path; a failure warns once and disables that kernel for the process (`_FAILED`, as in
  `_triton._failed`).
- **Reproducible and thread-count independent randomness.** The seed is drawn from torch
  (`torch.randint` on the default generator, or on `generator=` when one is given), so
  `torch.manual_seed` reproduces a run; inside the kernel, automaton `(c, l)` draws from a counter-based
  hash of `(seed, c, word)`, so the result does not depend on how rayon schedules rows. Native, compiled
  and eager paths consume random numbers differently (already the documented rule for compile):
  seeded runs reproduce under the same path.
- **Exact where the math is exact, equal in distribution where it samples**, tested as Triton is (§6).
- **Release the GIL and parallelise over clause rows**, pool sized to `torch.get_num_threads()`.
- **All three include caches stay in sync** (`include` float, `include_count`, `include_bits`), written
  by the commit kernel itself, which then sets `_bits_fresh = True` as `fused_commit` does.

### 4.2 Kernels, in build order

| # | Kernel | Replaces | Plugs into | Notes |
|---|---|---|---|---|
| K1 | `eval_bits(literals (B, 2F) 0/1, inc_bits (C, W), include_count, empty_value) -> (B, C) bool` | the `F.clause_outputs` matmul | `TsetlinMachineBase._evaluate` | packs the literals in-kernel (`O(B·2F)`, negligible next to `C·W` per row); prediction and learning |
| K2 | `commit_single(state, include, inc_bits, include_count, lit, lit_ii, fire_i, nofire_i, fire_ii, N, s, boost, budget, literal_active) -> count` | `F.commit_single_example` (compiled, CUDA-only today) | the `acc.single` branch of `_commit`; `_fuse_single_examples` admits the CPU when native is available | sequential mode is the classical algorithm; every count is 0/1, so Type Ia/II are Bernoulli draws and Type Ib is geometric skipping |
| K3 | `commit_rows(state, caches, events (lit, sel_i, sel_ii) or counts + rows, ...)` | `F.apply_feedback_rows` + `_refresh_include_at` + the three count matmuls | the sparse branch of `_commit`; `acc.record_events` on the CPU | counts per `(clause, literal)` tile from events, like Triton; declines beyond `TABLE_N` examples |
| K4 | `conv_eval_bits(neg_bits (B, P, W), inc_bits, ...) -> (B, C) bool + random matching patches` | `_evaluate_dense` + the patch draw | `_ConvMixin._evaluate` / `_evaluate_learning` | OR over patches in registers; same contract as `_triton.conv_eval` (independent patches for Type Ia and Type II) |
| K5 (optional) | fused votes + feedback selection for one example | ~50 eager ops per example in `_accumulate_literals` | `_accumulate_one` | only if K1 + K2 leave sequential mode op-bound |

Dispatch, the Triton pattern:

```python
# models/base.py
def _evaluate(self, literals, empty_value):
    if _native.available(literals.device):
        out = _native.eval_bits(literals, self.include_bits(), self.include_count, empty_value)
        if out is not None:
            return out, None
    return F.clause_outputs(literals, self.include, self.include_count, empty_value), None
```

`_commit` gains one branch before the compiled `commit_single_example` call and one before
`apply_feedback_rows`; each is `if _native.available(...) and _native.<kernel>(...) is not None:
commit weights, reset, return`.

### 4.3 Rust side

- Crate `native/` in this repository (Cargo workspace), Python module `torchtsetlin_native`, built with
  maturin, PyO3 `abi3-py39` (one wheel per platform for every Python >= 3.9).
- Dependencies: `pyo3`, `numpy`, `rayon`. RNG: a small counter-based generator (Philox-style or splitmix
  over `(seed, row, word)`); no `rand` crate needed.
- The state stays byte-per-automaton (no bit-slicing as in pyTM): AVX2 handles 32 automata per
  instruction on bytes anyway (compare-to-threshold mask, saturating add/sub), and the layout stays what
  `interpret`, `viz` and checkpoints read. Stable Rust only: auto-vectorised loops first, `std::arch`
  AVX2/NEON intrinsics behind `is_x86_feature_detected!` only where a measurement asks for them; no
  nightly `std::simd`.
- Binomials in K3 use the same inverse-CDF tables as the torch kernels, or exact per-event Bernoulli
  draws; Type Ib by geometric skipping.

### 4.4 Python side

- `src/torchtsetlin/_native.py`: `available(device)`, `_failed`, thin wrappers that check dtype,
  contiguity and shape, convert through `.numpy()`, call the extension and return `None` to decline.
  A one-to-one mirror of `_triton.py`.
- `functional.set_compile(False)` and `TORCHTSETLIN_COMPILE=0` disable native too (one switch for "all
  fast paths"); `TORCHTSETLIN_NATIVE=0` disables native alone.
- `torchtsetlin.native_available()` next to `__version__`; `extra_repr` names the path a model takes.

### 4.5 Packaging

- **A separate distribution `torchtsetlin-native`**, installed by the extra `torchtsetlin[native]`.
  `pip install torchtsetlin` stays a pure-Python wheel built by setuptools through the current publish
  workflow; the standing rule "keep the package pure-Python installable" holds.
- The native wheel is built by `PyO3/maturin-action` for manylinux2014 x86_64 and aarch64, macOS arm64
  and x86_64, Windows x64, and published on the same `v*` tag through Trusted Publishing as a second
  PyPI project.
- Version coupling: the extension exports `ABI_VERSION`; `_native.py` declines (warning once) on a
  mismatch, so an old native wheel never runs against a newer kernel contract.
- Local development: `pip install maturin && maturin develop --release -m native/Cargo.toml`.

### 4.6 Why Rust rather than a C++ torch extension

A `torch.utils.cpp_extension` module with OpenMP would share torch's thread pool and ATen's generators
and register through `torch.library`; the kernel design would be identical. Rust wins on the wheel story
(maturin builds the whole platform matrix, abi3, no compiler needed at install time), on rayon, and on
memory safety in code that writes into tensors in place; it loses direct access to `at::parallel_for`
and ATen RNG, which §4.1 replaces with its own pool and seed hand-off. Either is defensible; given the
preference for Rust, nothing in the plan depends on the choice except §4.3.

## 5. Plan and gates

Estimates are for one engineer who knows the code base; they are estimates.

**Phase 0a, torch-only baseline (2 days).** Bit-packed CPU evaluation in torch alone for one example
and small batches (`((neg_bits & inc_bits) != 0).any(-1)` over the existing `include_bits` buffer,
chunked so the `(B, C, W)` intermediate stays within a few MB). Cheap, no Rust, and it makes Phase 0b's
comparison honest: Rust has to beat torch at its best, not torch at its worst.

**Phase 0b, prototype (1 week).** K1 + K2 as a standalone maturin module, driven by a script under
`experiments/paper_experiments/code/kernel_audit/`; measure B4 (and B1 for a tiny model) on the CPU in
sequential mode at 1 and 16 threads against pyTM, pyTM-Par and Phase 0a. **Gate G0:** sequential
examples/s >= 20x today's CPU sequential path; within 2x of pyTM on one thread; >= 8x over one thread at
16 threads. If G0 fails, stop and record why here; the eager and compiled twins are untouched, so nothing
is lost.

**Phase 1, sequential mode and evaluation (2 weeks).** `_native.py`; K1/K2 wired into `_evaluate` and
`_commit`; the tests of §6; a docs page "Native CPU kernels"; CHANGELOG. Release 0.7.0 together with
`torchtsetlin-native` 0.7.0.

**Phase 2, batched CPU (2-3 weeks).** K3 with events mode on the CPU; the CPU rows of
`run_benchmarks.sh` and the regenerated `docs/cpu-vs-gpu.md`; G8 in `experiments/paper_experiments`
updated with new tt-CPU timing (its own go-ahead). **Gate G2:** >= 5x over today's batched CPU update at
the 5,000-clause / batch-256 shape (57 ms -> <= 12 ms), and thread scaling monotone up to 16 threads.

**Phase 3, convolution (2 weeks).** K4; B7 on the CPU (0.5.1: ~970 s/epoch against pyTM's 354 s).

**Phase 4, release engineering (1-2 weeks).** The wheel matrix in CI, the publish workflow, a
smoke-import of the native wheel in a clean venv, docs. Total: roughly 8-10 weeks.

## 6. Tests (mirror `tests/test_triton.py`, on the `cpu` device fixture)

- `test_eval_bits_is_exact`: K1 equals `F.clause_outputs` on random masks, both `empty_value`s, odd
  literal counts (unused high bits are zero).
- `test_commit_single_matches_eager_distribution`, `test_commit_rows_matches_apply_feedback`:
  transition frequencies over many seeds against the eager samplers, parametrised over `boost`, the
  budget, dropout (`literal_active`) and `s`, like the Triton test.
- `test_native_commits_keep_the_caches_exact`: `include`, `include_count` and `include_bits` equal
  `ta_state >= N` after every commit (the `test_include_caches_track_the_automata` contract).
- `test_native_is_reproducible_and_thread_invariant`: one seed gives one state at 1 and 8 threads.
- `test_native_declines_and_falls_back`: `TORCHTSETLIN_NATIVE=0`, non-contiguous input, hard budget,
  counts beyond the table all take the eager path with results identical to a run without the wheel.
- `test_native_learns_noisy_xor`, and the conv patch-draw uniformity test for K4.
- `tests/test_package.py`: `ABI_VERSION` of the native wheel matches; the pure wheel imports without it.

## 7. Risks

| Risk | Mitigation |
|---|---|
| K2 does not reach pyTM's per-example speed (pyTM bit-slices its state; we keep bytes) | Gate G0 exists for this; AVX2 byte lanes give the same 32 automata per op, and the prototype measures it before anything lands in `src/` |
| rayon and torch's OpenMP pool oversubscribe each other | size the pool from `torch.get_num_threads()`; release the GIL; never call torch from inside a kernel |
| Three RNG paths (eager, compiled, native) confuse users | the rule compile already has: seeded runs reproduce under the same path; `extra_repr` and a docs section say which path a model takes |
| The twins drift apart as `functional.py` changes | every twin pair has a distribution test; a change fails the test, not the user |
| A binary-wheel toolchain failure blocks a release | separate distribution; the pure wheel never waits for the native one |
| A tensor is freed while Rust still holds the view | views live only for the call; `py.allow_threads` around the compute; no stored pointers |
| Apple silicon and Windows are untested locally | the CI matrix builds every wheel and runs the tests on it |

## 8. Not to do

- No Rust tensor library, no array-API abstraction layer, no `torch.Tensor` façade.
- No Rust CUDA kernels: Triton + CUDA graphs stay the GPU path.
- No change to the public API, the checkpoint format, the `ta_state` layout or the feedback semantics.
- No native code inside the pure-Python wheel.

## 9. Later, separately: a torch-free inference runtime

The one torch-free product worth having is deployment: `torchtsetlin.export(model)` writing bit-packed
includes, weights and encoder thresholds, and a small `no_std` Rust crate (with a WASM build for the docs
site) that evaluates them. It does not touch learning, needs no backend abstraction, and can reuse the K1
code. Out of scope for this plan; noted so that "no PyTorch dependency" has a home.

## 10. Decisions needed

1. Agree with the verdict: no tensor-backend swap; native CPU kernels behind torch instead.
2. Go-ahead for Phase 0a and 0b, including the benchmark runs they need.
3. A separate `torchtsetlin-native` distribution versus native code in the main wheel (recommended:
   separate).

Sources for the prior-art claim: tsetlin-rs on docs.rs (https://docs.rs/crate/tsetlin-rs/latest) and on
GitHub (https://github.com/RAprogramm/tsetlin-rs).
