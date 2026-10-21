# Existing Tsetlin-machine software: fact sheet

Collected on 2026-10-21 from the GitHub REST API (repo metadata, releases, tags, file trees,
workflow directories), raw source files, PyPI JSON metadata and crates.io. Nothing was installed.
"unverified" means the fact could not be confirmed from those sources.

## Summary table (for the paper's comparison table)

| Library | Language / backend | GPU | Variants | API style | Licence | Latest release | Tests / CI / docs |
|---|---|---|---|---|---|---|---|
| pyTsetlinMachine | C core, Python via `ctypes` | none (CPU, single-threaded) | multi-class, conv 2D, regression, weighted, embedding, clause indexing, literal budget, drop clause/literal | `fit(X, y, epochs)` / `predict`, sklearn-like | MIT | PyPI 0.6.6 (2024-03-05). Last GitHub release v0.6.0 (2021-05-30) | no tests, no CI, Sphinx `docs/` folder plus README |
| PyTsetlinMachineCUDA | Python with CUDA C kernels compiled at runtime through **PyCUDA** | yes, PyCUDA (`SourceModule`, `curandom`) | multi-class, conv 2D, regression, weighted clauses | `fit` / `predict`, sklearn-like | MIT | PyPI 0.1.9 (2021-06-19). Last push 2023-01 | no tests, no CI, `docs/` folder plus README |
| tmu | Python, with C (cffi) and CUDA backends | yes, optional **PyCUDA** clause bank (`clause_bank_cuda.py`, falls back to CPU if pycuda is missing). Selected by `platform='CPU'/'CUDA'` | multi-class (vanilla), coalesced, conv (`patch_dim`), regression, weighted, autoencoder, multi-task, sparse clause bank, drop clause, literal budget, Type III feedback, absorbing, focused negative sampling, composites | `TMClassifier(...).fit(X, Y)` / `predict`, sklearn-like | MIT | PyPI 0.8.3 (2024-03-25). Repo still active (push 2026-08-03) | pytest suite in `test/`, GitHub Actions (tests, wheels, docs), docs build |
| green-tsetlin | C++ core (pybind11, AVX2/NEON SIMD, thread pool), Python front end | none (CPU, multithreaded) | multi-class dense TM, sparse TM, conv clause blocks in the C++ core (conv at the Python API level: unverified), literal budget, rule-set export, C-code export, Optuna hyper-parameter search | `gt.TsetlinMachine(...)` plus a separate `gt.Trainer(tm).train()`, then `tm.get_predictor().predict(x)` | MIT | PyPI 1.1.0 (2025-08-25). Last GitHub release v1.0.1 (2024-05-08) | pytest `tests/`, GitHub Actions (pip, wheels, conda), Read the Docs (green-tsetlin.readthedocs.io) |
| GraphTsetlinMachine | Python with CUDA kernels through **PyCUDA** | yes, PyCUDA | multi-class Graph TM (graph-structured input, message passing), which includes conv-like and sequence use cases | `MultiClassGraphTsetlinMachine(...).fit(graphs, Y)` | MIT | PyPI 0.3.3 (2025-04-04) | no tests or CI found, README with examples |
| pyTsetlinMachineParallel | C (multithreaded; OpenMP unverified), Python via `ctypes` | none (CPU, multithreaded) | multi-class, conv 2D, regression, weighted | `fit` / `predict` | GitHub reports no licence (unverified). PyPI says MIT | PyPI 0.2.1 (2021-10-10) | no tests, no CI |
| Tsetlin.jl (BooBSD) | Julia, pure (SIMD, multithreaded), zero dependencies | none | Fuzzy-Pattern TM: binary and multi-class classifier, model compilation, explainability tools | Julia functions | MIT | no tagged releases. Active (push 2026-10-17) | tests/CI: unverified |
| tsetlin-rs (RAprogramm) | Rust (const generics, bitwise SIMD, `no_std`) | none (a `gpu` feature flag is only "backend traits") | binary, multi-class, regression, convolutional, sparse inference | Rust structs | MIT per README badge (GitHub API reports none) | crates.io 0.3.4 (2026-01-29) | extensive GitHub Actions (test, clippy, coverage, docs.rs) |
| FsTsetlin (fwaris) | F#, **TorchSharp/libtorch tensor ops** | yes, through libtorch tensors (CPU/GPU, "not extensively performance-tested") | binary and multi-class | F# scripts | MIT | no releases. Last push 2024-09-08 | unverified, examples only |

## Per-library details

### pyTsetlinMachine
- Repo: https://github.com/cair/pyTsetlinMachine (156 stars, not archived, last push 2025-04-01)
- PyPI: `pyTsetlinMachine` 0.6.6, uploaded 2024-03-05. The first release was 0.1.4 (2019-06-12).
  PyPI author: Ole-Christoffer Granmo.
- Backend: C sources (`ConvolutionalTsetlinMachine.c`, `MultiClassConvolutionalTsetlinMachine.c`,
  `IndexedTsetlinMachine.c`, `EmbeddingTsetlinMachine.c`) wrapped with `ctypes` in `tm.py`.
- Classes: `MultiClassTsetlinMachine`, `MultiClassConvolutionalTsetlinMachine2D`,
  `RegressionTsetlinMachine`, `EmbeddingTsetlinMachine`, `ConvolutionalEmbeddingTsetlinMachine2D`.
  Weighted clauses are a constructor flag.
- GPU: none.

### PyTsetlinMachineCUDA
- Repo: https://github.com/cair/PyTsetlinMachineCUDA (43 stars, last push 2023-01-03)
- PyPI: `PyTsetlinMachineCUDA` 0.1.9 (2021-06-19). First release 0.1.1 (2020-08-01). There is a stray
  git tag `v0.1.18` but no matching release.
- Backend: `tm.py` imports `pycuda.driver`, `pycuda.autoinit`, `pycuda.curandom` and
  `pycuda.compiler.SourceModule`. CUDA C kernels are in `kernels.py`. It implements the asynchronous
  architecture of Abeyrathna et al. (ICML 2021).
- Classes: `MultiClassTsetlinMachine`, `MultiClassConvolutionalTsetlinMachine2D`, `TsetlinMachine`,
  `RegressionTsetlinMachine`. The README demos use weighted clauses.

### tmu (Tsetlin Machine Unified)
- Repo: https://github.com/cair/tmu (174 stars, last push 2026-08-03)
- PyPI: `tmu` 0.8.3 (2024-03-25). First release 0.1.0 (2021-11-27). `tmu/__init__.py` on `main` is
  still 0.8.3. Requires Python >= 3.10. Dependencies: cffi, numpy, pandas, scikit-learn, tqdm,
  requests, xxhash. PyPI authors: Granmo and Per-Arne Andersen.
- Models (`tmu/models/`): `classification/vanilla_classifier.py` (`TMClassifier`),
  `classification/coalesced_classifier.py`, `classification/multitask_classifier.py`,
  `regression/vanilla_regressor.py`, `autoencoder/autoencoder.py`. Clause banks are `clause_bank.py`
  (C via cffi), `clause_bank_cuda.py` (PyCUDA) and `clause_bank_sparse.py`. There is also a `weight_bank/`
  and `composite/` (TMComposites).
- README feature list: continuous features, drop clause, literal budget, focused negative sampling,
  Type III feedback, absorbing actions, TMComposites. Relational TM is "in progress".
- CI: `.github/workflows/build-tests.yml` (pytest, ubuntu-20.04, Python 3.10), `build-wheels.yml`,
  `build-docs.yml`. Tests in `test/` cover classifiers, components, datasets, imports and pickle.
  `tmu.readthedocs.io` returns 404, so the hosted docs location is unverified.

### green-tsetlin
- Repo: https://github.com/ooki/green_tsetlin (24 stars, last push 2025-10-24). The `ooki-io/green-tsetlin`
  path does not exist.
- PyPI: `green-tsetlin` 1.1.0 (2025-08-25). First release 0.1.0 (2023-07-17). PyPI author is
  Sondre "Ooki" Glimsdal. Summary: "A fast Tsetlin Machine impl, based on c++". Python >= 3.8.
- Backend: a `Pybind11Extension("green_tsetlin_core")` built from `src/*.hpp` (`func_avx2.hpp`,
  `func_neon.hpp`, `func_conv_tm.hpp`, `func_sparse.hpp`, `BS_thread_pool.hpp`). It has a pure-Python
  reference core (`py_gtc/`) and exports to C (`writers/simple_c.py`, `topographical_c.py`). There is no GPU support.
- Paper: Glimsdal, Østby, Brambo, Vinje, *Green Tsetlin: Redefining Efficiency in Tsetlin Machine
  Frameworks*, arXiv:2405.04212 (2024), cited as `greentsetlin2024`.
- Tests: about 14 pytest files in `tests/`. CI: GitHub Actions (`pip.yml`, `wheels.yml`, `conda.yml`)
  plus legacy Travis/AppVeyor configuration. Docs are on Read the Docs (HTTP 200).

### GraphTsetlinMachine
- Repo: https://github.com/cair/GraphTsetlinMachine (54 stars, last push 2026-04-13). PyPI
  `GraphTsetlinMachine` 0.3.3 (2025-04-04). MIT licence.
- Backend: PyCUDA (`pycuda.driver`, `SourceModule`) with kernels in `kernels.py`. Its class is
  `MultiClassGraphTsetlinMachine`. It accompanies `granmo2025graph`.

### Other ports (lower maturity or not Python)
- cair/TsetlinMachine (Cython, 509 stars): the original research code and datasets. It is not a library release.
- cair/TsetlinMachineC, cair/fast-tsetlin-machine-with-mnist-demo,
  cair/fast-tsetlin-machine-in-cuda-with-imdb-demo (CUDA, 2019): demo code only.
- egbaquela/TsetlinMachine.jl (Julia, 1 star, last push 2022-12): inactive.
- hackjoy/tsetlin, shkmv/tsetlin, Jaghov/tsetlin-machines-rs (Rust): small or personal projects.
- REDRESS (Maheshwari et al., TPAMI 2023) and MATADOR (Rahman et al., DATE 2024) are
  **hardware and deployment toolflows**, not training libraries. They take a trained TM and produce
  compressed microcontroller code or SoC designs. No public repository was found for either
  (GitHub search returned nothing), so their availability is unverified.

## Key observations for the paper
- Every maintained GPU TM implementation (PyTsetlinMachineCUDA, tmu's CUDA clause bank,
  GraphTsetlinMachine) uses **hand-written CUDA kernels compiled at runtime with PyCUDA**. None
  uses a mainstream tensor framework. The only tensor-framework TM is FsTsetlin, which is F# on
  TorchSharp, has no releases, and is not Python.
- The CPU-focused libraries (pyTsetlinMachine, pyTsetlinMachineParallel, green-tsetlin,
  Tsetlin.jl, tsetlin-rs) have no GPU path.
- No existing library is an `nn.Module` or exposes `state_dict`, `.to(device)` or autograd-style
  integration. All use sklearn-like `fit`/`predict` (or a separate Trainer, in green-tsetlin).
- Only tmu, green-tsetlin and tsetlin-rs have tests and CI. tmu's last PyPI release
  (2024-03) predates two years of activity on `main`.
- None of the libraries above documents 3D (volumetric) convolution or dense per-pixel segmentation TMs. Their convolution is 2D patches only, although the Graph TM can encode arbitrary structure.
