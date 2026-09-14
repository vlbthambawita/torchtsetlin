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
3. **`models/{classifier,coalesced,regression,conv}.py`** — concrete variants that only supply an *output
   layer* and a *feedback policy* via hooks.
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
- **`_chunk_elements_per_example` must scale with the real intermediates** — `P*C` for convolutional models,
  only `C + 2F` for flat ones. Getting it wrong produces many tiny chunks per batch and a 4–20× slowdown.
- **Convolutional `position_encoding=True` pins clauses to locations.** On translation-invariant toys
  (`make_shapes`, `make_2d_noisy_xor`) accuracy oscillates 0.5–0.9 with it on and hits 1.0 with it off;
  MNIST is fine with it on. Check this flag before suspecting the feedback code.
- **Version lives in two places** — `pyproject.toml` and `src/torchtsetlin/__init__.py`; `tests/test_package.py`
  asserts they agree and that every subpackage ships. `.gitignore` rules that could match package
  directories must stay anchored to the repo root (`/data/`, not `data/`) — an unanchored rule excluded
  `src/torchtsetlin/data` from the 0.1.0 wheel.
- **Ruff config targets py39**: `UP006/UP007/UP035/UP045` are ignored on purpose so `typing.Optional`/`List`
  keep working at runtime on 3.9. Keep using them rather than PEP 604/585 syntax in annotations.
- Optional dependencies are import-guarded: `viz` needs matplotlib, `data.load_*_boolean` needs torchvision,
  some examples need scikit-learn.

## Docs

MkDocs Material with mkdocstrings reading Google-style docstrings straight from `src` — public API
docstrings are the API reference, and `docs/api/*.md` are thin `:::` stubs. Conceptual background lives in
`docs/concepts/`, runnable scripts in `examples/`, measured numbers in `docs/benchmarks.md`.
