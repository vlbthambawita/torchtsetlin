# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -e ".[dev]"

pytest -q                                   # full suite (600 s per-test timeout, testpaths=tests)
pytest tests/test_models.py::test_learns_noisy_xor_batched -q        # one test
pytest -k "conv" -q                         # by keyword
ruff check src tests                        # lint (CI runs exactly this)
ruff check --fix src tests
mkdocs serve                                # docs at localhost:8000

./benchmarks/run_benchmarks.sh              # all suites -> results JSON + docs figures/page
./benchmarks/run_benchmarks.sh --quick      # ~1 min smoke test of the benchmark pipeline
./benchmarks/run_benchmarks.sh --plots-only # re-render figures/tables/page from the JSON
python -m build && python -m twine check dist/*
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest on Python 3.9 / 3.11 / 3.12, then `mkdocs gh-deploy` on
pushes to `main`. Releases go out on a `v*` tag via Trusted Publishing (`.github/workflows/publish.yml`),
which re-runs the suite and smoke-imports the built wheel in a clean venv.

Tests parametrise a `device` fixture over `cpu` plus `cuda` when available (`tests/conftest.py`), so on a GPU
box every device-marked test runs twice.

## Architecture

Four layers, from the bottom up:

1. **`functional.py`** — stateless tensor ops: `to_literals`, `clause_outputs`, `vote_sums`,
   `feedback_probabilities`, `type_i_counts` / `type_ii_counts`, `apply_feedback`. No autograd, no state,
   device-agnostic. All learning math lives here.
2. **`models/base.py::TsetlinMachineBase`** — an `nn.Module` holding the automata and running the generic
   learning loop (input coercion, lazy init, chunking, dropout masks, feedback accumulation, commit).
3. **`models/{classifier,coalesced,regression,conv,segmentation}.py`** — concrete variants that only supply
   an *output layer* and a *feedback policy* via hooks.
4. **`train/`, `metrics.py`, `interpret.py`, `viz.py`, `data/`** — Keras-style `Trainer` + callbacks,
   metrics, rule/importance extraction, matplotlib plots, and Booleanization encoders/datasets.

### The learning state

The learnable state is **buffers, not parameters** — Tsetlin machines have no gradients. `ta_state` is an
integer `(n_clauses_total, 2*n_features)` buffer; a literal is *included* when its state is `>= n_states`.
`include` (float mask) and `include_count` are **non-persistent** caches derived from it, so anything that
mutates `ta_state` must call `_refresh_include()` afterwards (a `load_state_dict` post-hook does this too).
`parameters()` deliberately warns when it returns nothing.

`n_features=None` means lazy init (`torch.nn.LazyLinear` style): the shape is allocated on the first batch,
which is why `_load_from_state_dict` reconstructs the shape from the checkpoint before loading.

### The update pipeline

`update(x, y)` is the analogue of `loss.backward(); optimizer.step()` and **returns the pre-update vote
sums** so a trainer can log batch metrics without a second forward pass. The flow:

```
update -> _prepare -> per chunk: _accumulate_chunk( _encode -> _evaluate -> _votes
                                                    -> _select_feedback -> _feedback_counts
                                                    -> _accumulate_weights )
       -> _commit( functional.apply_feedback -> _commit_weights -> _refresh_include )
```

A new model subclasses `TsetlinMachineBase` and implements `_votes`, `_select_feedback` and
`_coerce_targets`; clause weights need `_accumulate_weights` / `_commit_weights`. A different *input*
structure overrides `_encode`, `_evaluate`, `_feedback_counts` and `_chunk_elements_per_example` — that is
exactly what `models/conv.py::_ConvMixin` does, and it is mixed into each flat model to produce the `Conv*`
variants (`class ConvTsetlinMachine(_ConvMixin, TsetlinMachine)`). See `docs/guides/extending.md`.
`_Conv3dMixin(_ConvMixin)` produces the `Conv3d*` volume models by overriding only the geometry
(`_grid`, `_encode`, `_coerce_input`, ...). Its `_coerce_input` must skip `_ConvMixin`'s via
`super(_ConvMixin, self)`, which would otherwise reshape to 4D. Its `_evaluate`/`_feedback_counts` only
add patch-axis blocking and defer to the 2D code when there is a single block, which keeps the
`D = kd = 1` model bit-identical to the 2D one (`tests/test_conv3d.py` pins this; proofs are in `plans/conv3d.md`).

`models/segmentation.py::_DenseMixin` is the other input-structure mixin: it *drops* the convolutional
OR over patches and folds the patch axis into the batch axis (`B' = B * P`, one patch per pixel), so each
pixel is an ordinary flat example and the whole flat pipeline applies unchanged. Because `P` is large,
`patches_per_commit` bounds how many patches share one commit (same fidelity issue as a big batch).
Config tensors such as `class_feedback_p` are registered as **non-persistent buffers** so `.to(device)`
moves them without changing the checkpoint format — follow that pattern for any new tensor attribute.

`models/graph.py::_GraphMixin` (the graph Tsetlin machine, `GraphTsetlinMachine`) takes a
`data.graphs.GraphBatch` (padded nodes `(B, P, F)`, per-graph edge lists with local indices) instead
of a tensor. One `ta_state` row per clause covers all layers: the literal row of a node is
`[H^0 | H^1 | ... | H^{D-1}]` (own features, then one inbox of `message_size` bits per message
layer), so the deep clause is the row's AND and the feedback is the conv feedback (`_ConvMixin._feedback_counts`) on
that row (nodes = patches). Only `_evaluate` is new: it runs layer by layer, because an inbox
depends on the previous layer's matches at the in-neighbours (scatter-OR over edges, then the
shifted message positions: a product with a cached binding matrix on CUDA, a column scatter
on the CPU for big inboxes, `_dense_binding`). `_encode` returns an `_Encoded` holder, not
literals. `_feedback_counts` draws nodes exactly as the conv code does but sums the drawn rows
as a sparse `W @ H` over the positive half (the negated half is `count - W @ H`), never building
`2F`-wide rows per event. A tuple `s` (per layer, `layer_s`) commits through
`F.apply_feedback_blocks` instead of the base commit paths. On CUDA, graph updates join the conv
fast paths with nodes for patches (`_record_fast`: dense node draw, `acc.events` /
`acc.single` with the `(B, P, 2F)` literal rows): events mode for any batch, fused sequential
commits and CUDA graphs only for **padded** batches (`GraphBatch.pad`: `(B, E, 2)` edges plus a
mask, slicing never syncs and keeps the shape; `_input_padded` is set in `_coerce_input`).
`_fuse_single_examples` has no minimum model size (an eager graph step is ~5 ms of launches and
syncs). `_graph_key` adds the padded edge shape; `GraphBatch.clone`/`copy_` serve the base
capture code, so `models/base.py` needed no change. A per-layer `s` keeps the eager paths. Depth-1 batches with `GraphBatch.full_nodes` (no padding
nodes; set on the host by the builders, carried by slicing, never computed from the data) and one
node type take `_triton.conv_eval` like conv models (`_bitpacked`, `_evaluate_learning` returns
a `("nodes", lit, p_i, p_ii)` context); they learn bit for bit like `ConvCoalescedTsetlinMachine`
on CUDA.
`tests/test_graph.py` pins evaluation against a naive per-node reference, depth 1 against
`ConvCoalescedTsetlinMachine` bit for bit (eager path), padded = flat, and the fast paths
(sync-free updates, replays with fresh randomness).

`FeedbackAccumulator` carries the per-update event counts (`n_true`, `n_false`, `n_ib`, `n2`, plus `extra`
for weight deltas). `functional.apply_feedback` **uses those count tensors as scratch space** (in-place) —
do not reuse them after the call.

## Behaviours that are easy to get wrong

- **Empty-clause semantics depend on the mode.** A clause with no included literals evaluates to `True`
  while learning and `False` when predicting; `forward()`/`clause_outputs()` key this off `self.training`,
  so forgetting `model.eval()` silently changes predictions. `tests/test_models.py` pins this.
- **`feedback_mode="batch"` (the default)** aggregates all feedback of a mini-batch and applies it once.
  Fidelity to the classical algorithm degrades as batch size grows — batch 10–50 matches sequential on
  Noisy XOR, 200 degrades noticeably (see `docs/benchmarks.md`). `"sequential"` (or `update(..., sequential=True)`)
  is the exact per-example algorithm. Aggregating *all* events is deliberate: thinning Type II/Ia events per
  clause was tried and made learning much worse.
- **`max_included_literals` (clause budget) is soft by default** (`budget_mode="soft"`, tmu's rule).
  A matching clause above the budget gets its Type Ia turned into Type Ib (forgetting), so a clause
  may overshoot the budget and is then pruned. History: before 2026-10-01 the budget was soft. In
  0.3.x (issue #1) it was a hard cap that also reverted new inclusions beyond the budget
  (`_cap_new_inclusions`, still available as `budget_mode="hard"`). With the hard cap no clause ever
  exceeds the budget, so the Type Ib switch never fires and clauses freeze at exactly the budget:
  -10 pp on the CIFAR-10 thermometer CTMs (gap MC-2, `experiments/multi_convtm/LIBRARY_GAPS.md`).
  Results measured with a 0.3.x budget are not directly comparable.
- **Weights, regression feedback and focused sampling follow the reference libraries since
  2026-10-22** (G1/G3/G4 in `experiments/paper_experiments/LIBRARY_GAPS.md`): clause weights floor at
  1 (was 0), the regression feedback probability is `(err/T)**2` (was `|err|/T`), and focused
  negative sampling weights a class by `(T + clip(v, -T, T)) / 2T` (was `v - min(v) + 1`). Results
  measured before that date on weighted, regression or focused-sampling models are not directly comparable.
- **Two commit paths since 2026-10-24.** CPU updates and sequential mode use the row-sparse
  `F.apply_feedback_rows` (counts only in `acc.rows`, Type Ib drawn sparsely, include caches
  refreshed incrementally via `_refresh_include_at`); batched GPU updates keep the dense
  `F.apply_feedback`. Anything that writes Type Ia/II counts must only touch rows marked in
  `acc.rows` (`_accumulate_literals` marks every selected matching clause). The count buffers are
  reused across updates (`_new_accumulator`), so never keep a reference to them after `_commit`.
  `tests/test_models.py::test_include_caches_track_the_automata` pins the cache refresh.
- **CUDA fast path since 2026-10-28** (`_triton.py`, LIBRARY_GAPS round 3). A batched CUDA
  update that fits one chunk runs in *events mode* (`acc.record_events`): `_feedback_counts`
  stores `acc.events = (literals, sel_i, sel_ii, patch_i, patch_ii)` instead of counts and
  `_triton.fused_commit` computes, samples and applies them in one kernel. A model joins only if
  its `_feedback_counts` carries `_records_events = True`; a new output layer that overrides it
  must handle (or decline) events. Conv models evaluate with `_triton.conv_eval` on bit-packed
  literals (prediction, and learning in events mode, where it also draws the patches);
  `_evaluate_dense` is the old `(B, P, C)` path, kept for `patch_clause_outputs` and fall-backs.
  `include_bits()` is a lazily refreshed bit-packed include mask: every commit path sets
  `_bits_fresh = False` except the Triton one, which writes it. Chunk sizes differ by path
  (`_fused_chunk_size` vs `_chunk_elements_per_example`): never evaluate densely with a fused
  chunk size, it allocates `(chunk, P, C)`.
- **CUDA graphs since 2026-10-28** (`_graphed_update`). Update shapes are captured on third use
  and replayed. Anything inside `_update_block` must be sync-free (use `torch._assert_async`,
  `scatter_`, host-side bounds; `tests/test_triton.py::test_updates_never_synchronise_with_the_host`)
  and must update buffers **in place** (`include_count.copy_`, kernels writing `include`): a
  reassigned buffer changes the graph key (fine), but a tensor the graph allocated and Python
  keeps is graph-private memory. Python constants (`s`, `T`, ...) are part of the key, so
  hyper-parameter schedules recapture. A replay runs no Python: a flag an update sets in
  `__dict__` must be listed in `_GRAPH_FLAGS`, which each replay re-applies (0.6.0 missed
  `_bits_fresh`, and sequential conv models predicted from a frozen mask;
  `test_replayed_updates_leave_the_prediction_caches_fresh`). Labels on CUDA are checked device-side
  (`utils.class_targets`): a bad label surfaces as a CUDA error, not a `ValueError`.
- **`torch.compile` kernels since 2026-10-24** (`functional.py`, "Compiled kernels"). The CPU
  row-sparse commit runs Type Ib forgetting (`_forget_rows`) and its binomials
  (`_binomial_compiled`) as inverse-CDF kernels with in-kernel RNG; sequential mode on CUDA
  commits each example of a flat model with `F.commit_single_example` (recorded as factors in
  `acc.single` by the base `_feedback_counts`) and compiles `_accumulate_literals`. Every
  compiled path has an eager twin, taken below `F.COMPILE_MIN_ELEMENTS`, with an explicit
  `generator=`, under `TORCHTSETLIN_COMPILE=0` / `F.set_compile(False)`, or when compilation
  fails (`_run_compiled` returns `None`; warn once per function). Keep kernel input shapes
  stable: the inverse-CDF table is one fixed `(_TABLE_N, _TABLE_K)` shape because every new
  shape recompiles (and after 8 recompiles dynamo silently runs eager); the row-count kernels
  are `dynamic=True` (`_DYNAMIC`); with `dynamic=False` the 32-wide unrolled compare compiled
  to a 13x slower kernel. Inductor's CPU `rand` does not scale with threads; it is the floor of
  the CPU commit. Tests use the `compiled` fixture (`tests/conftest.py`) to force these paths
  on tiny inputs.
- **`ta_state` is `uint8` by default** (`state_dtype=None`, 2N-1 <= 255). Python-int arithmetic on
  it wraps silently (`state - N`); cast first. The commits form the new state in a wider buffer.
- **The convolutional random patch is drawn in `_feedback_counts`, not `_evaluate`.** `_evaluate`
  returns the whole `(B, P, C)` match tensor as its feedback context and nothing else; drawing
  there instead costs a `(B, P, C)` `rand` + `argmax` on *every* forward pass (~5x slower
  prediction and training on MNIST-sized inputs) and forces Type Ia and Type II to share one
  patch, which is wrong for coalesced/multi-label models. `tests/test_models.py::
  test_conv_prediction_draws_no_random_patches` pins the RNG-free prediction path.
- **`_chunk_elements_per_example` must scale with the real intermediates** — `P*C` for convolutional models,
  only `C + 2F` for flat ones. Getting it wrong produces many tiny chunks per batch and a 4–20× slowdown.
- **Convolutional `position_encoding=True` pins clauses to locations.** On translation-invariant toys
  (`make_shapes`, `make_2d_noisy_xor`) accuracy oscillates 0.5–0.9 with it on and hits 1.0 with it off;
  MNIST is fine with it on. Check this flag before suspecting the feedback code.
- **Version lives in two places** — `pyproject.toml` and `src/torchtsetlin/__init__.py`; `tests/test_package.py`
  asserts they agree and that every subpackage ships. The docs read it from `__init__.py` through
  `scripts/mkdocs_hooks.py` (release banner, `%%version%%` in pages), so never type it into the docs. `.gitignore` rules that could match package
  directories must stay anchored to the repo root (`/data/`, not `data/`) — an unanchored rule excluded
  `src/torchtsetlin/data` from the 0.1.0 wheel.
- **Ruff config targets py39**: `UP006/UP007/UP035/UP045` are ignored on purpose so `typing.Optional`/`List`
  keep working at runtime on 3.9. Keep using them rather than PEP 604/585 syntax in annotations.
- **Never `.to(cpu, non_blocking=True)`.** Encoders keep their thresholds on the CPU, so handing
  one a CUDA tensor triggers a device-to-host copy inside `utils.as_*_tensor`; issued
  non-blocking into pageable memory it returns *before* the data lands and the encoder
  thresholds garbage — ~40% wrong bits, no error, just a worse model. All coercion helpers go
  through `utils._to_device`, which only sets `non_blocking` for CUDA destinations. A test that
  allocates on the host and moves the tensor across will **not** catch a regression here; build
  the tensor on the device (`torch.rand(..., device=dev)`) so work stays queued on the stream
  (`tests/test_data_train.py::test_encoder_on_non_cpu_input_matches_cpu`).
- Optional dependencies are import-guarded: `viz` needs matplotlib, `data.load_*_boolean` needs torchvision,
  some examples need scikit-learn.

## Benchmarks

`benchmarks/run_benchmarks.sh` is the driver: it runs `bench_device.py` **one suite at a time**
(so a crash costs only the running suite — the JSON is merged after each), then
`plot_benchmarks.py`, which writes `docs/assets/benchmarks/*.png` (light + `_dark` pairs for the
MkDocs Material theme), the version's `tables.md`, and regenerates `docs/cpu-vs-gpu.md`
whole — prose, figures and tables are derived from the records, so **do not hand-edit that page**.
Figures skip suites that are missing from the results, so partial runs plot fine.

Results are **per library version** since 2026-10-29: `benchmarks/results/versions/<version>/`
(`cpu_vs_gpu.json` + `tables.md`; the `.log` is git-ignored), keyed by the installed
`torchtsetlin.__version__`. `bench_device.py` refuses to merge records of another version into a
file. `docs/cpu-vs-gpu.md` always shows the newest version on disk (measuring or re-plotting an
older one with `-V` only refreshes its tables), and `plot_history.py` regenerates
`docs/benchmark-history.md` + `version-history*.png` from every version. It matches
configurations across versions by their settings (`config_key`), and drops records whose
timings equal an older version's: before per-version results, a single-suite run merged into the
previous file, so 0.3.0's JSON holds 0.2.0's numbers for every suite but conv3d. Folders
0.1.2-0.5.0 were recovered from git history.

`benchmarks/literature/` is the literature catalog: `papers/*.json` (one hand-checked record per
paper in `source_documents/papers/`) is the source of truth, and `build_catalog.py` regenerates
`datasets.json`, `results.csv`, `README.md` and the SQLite database `docs/assets/literature/literature.db`
(committed; a git-ignored copy goes to `space/`). Edit the records and re-run the script; never
hand-edit the generated files. The "best value" direction per metric comes from `_DIRECTION` in
that script. Since 2026-11-05 the database also has the schema-v2 fields (task family,
canonical metric, clause count + scope, T, s, ..., tables `task_details` and `vocab`): they come
from the records when present, otherwise from the reviewable proposal CSVs in
`benchmarks/literature/mappings/` (written by `propose_mappings.py`, controlled values in
`vocab.json`), and each row's `v2_source` says which (`record` / `pdf` / `reviewed` / `proposed`).
`pdf` values come from `papers_v2/<paper_id>.json` sidecars written by re-reading the PDF per
`EXTRACTION_V2.md` (`build_catalog.py --check-v2 [ids]` validates them without writing; a full
build refuses to run while any sidecar is invalid); record errors found that way go into the
`corrections` table, not straight into the records. Plan
and status: `plans/literature-db-v2.md`. `docs/literature.md` queries the database in the browser with sql.js
(`docs/javascripts/literature.js`). Keep ranking queries on narrow columns and join the long
text only for the drawn page: dragging `notes`/`config` through the window functions made every
regroup about 3× slower. The same database also backs the Gradio app in `space/app.py`, deployed
as the HF Space `vlbthambawita/TsetlinMachines`. That Space runs on ZeroGPU and cannot be moved
to CPU on this account, so `app.py` keeps an unused `@spaces.GPU` placeholder or the Space fails
at startup.

Progress output: the driver exports `PYTHONUNBUFFERED=1` because it tees stdout into a log —
without it Python block-buffers the pipe and a suite looks frozen for minutes. `bench_device.py`
reports `[k/n] <config> (elapsed, eta)` per measured configuration through the module-level
`PROG` (`Progress`), which `main()` resets per (suite, device); a new suite should call
`PROG.plan(n)` once and `PROG.item(label)` per configuration.

## Research experiments

`experiments/{convtm,generation,mctm}/` are research programmes built *on top of* the library — not part
of the package, not covered by `pytest`/`ruff` in CI. Each has a `PLAN.md` (and `convtm` a one-page
`CHARTER.md`, read first) whose hard rule C1 is that **`src/torchtsetlin/**` is read-only while working
in an experiment** unless the user explicitly confirms otherwise; library defects found there go into
that programme's `LIBRARY_GAPS.md` instead of being fixed. Reported numbers must come from the
programme's `results/*.json` records, never hand-typed. The `.claude/agents/` personas (`tm-theorist`,
`dl-expert`, `research-engineer`, `notetaker`) belong to the `convtm` team.

## Docs

MkDocs Material with mkdocstrings reading Google-style docstrings straight from `src` — public API
docstrings are the API reference, and `docs/api/*.md` are thin `:::` stubs. Conceptual background lives in
`docs/concepts/`, runnable scripts in `examples/`, measured numbers in `docs/benchmarks.md`.

The notebooks in `examples/notebooks/` are rendered under Examples → Notebooks by `mkdocs-jupyter`
through the symlink `docs/examples/notebooks` (`execute: false`). Notebooks are committed **without outputs**, so their prose must match what
the code produces on the current library: re-run them after changes that move results.
`exclude_docs` keeps the folder's `README.md` and saved `*.pt` models out of the site. Each
notebook starts with an *Open in Colab* badge and a setup cell that sets `IN_COLAB` and installs
torchtsetlin from PyPI there; later cells use `IN_COLAB` (e.g. `DATA_ROOT`). A new notebook needs
both, plus a nav entry in `mkdocs.yml` and a row in `docs/examples/notebooks.md`;
`tests/test_notebooks.py` checks the badge, the setup cell and that no outputs are stored.

The explanatory figures on the concept pages (`docs/assets/concepts/*.svg`, light + `_dark` pairs) are
generated by `scripts/make_concept_figures.py` — edit the script and re-run it rather than hand-editing
the SVGs. Animations are CSS keyframes inside the SVG (they play in a plain `<img>`); the un-animated
state of every element must be the finished picture, because `prefers-reduced-motion` switches the
animation off. Keyframe offsets within one animation must be strictly increasing — duplicate offsets
make Chromium interpolate across them and elements fade to half opacity.
