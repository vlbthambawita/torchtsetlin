# paper_experiments — cross-library benchmark plan (JMLR MLOSS paper)

**Status: APPROVED 2026-10-22 with the decisions in §8. Phase 0 started.**

**Goal.** Measure how `torchtsetlin` compares with the existing Tsetlin-machine (TM) libraries on the
benchmarks those libraries already publish. Two kinds of number:

1. **ML performance**: accuracy, precision, recall and F1, or RMSE and MAE for regression.
2. **Efficiency**: training time, inference time and throughput, memory, model size, and speed-up.

**Scope rule (from the user).** No new experiment types. Every experiment below exists in at least one
other library's repository (README demo, example script, performance test) or in the paper that
introduced that library. We run every library that implements the model the experiment needs. Running a
library on an existing benchmark is not a new experiment; inventing a dataset, task or protocol would be.

The evidence for "exists in another library" is in `sources/`, two surveys made on 2026-10-22:

- `sources/survey_repos.md`: what each library's repository ships and reports, with URLs.
- `sources/survey_papers.md`: what each library's paper reports, with table and figure numbers.

---------------------------------------------------------------------------------------------------

## 1. Rules

| # | Rule |
|---|---|
| P1 | `src/torchtsetlin/**` is **read-only** in this programme. A defect found here goes into `LIBRARY_GAPS.md`. It is never fixed silently, because a fix would change the measured library. |
| P2 | Every reported number comes from a `results/*.json` record. Each record carries `argv`, `seed`, `git_sha`, library versions, `env` (CPU, GPU, driver, threads) and raw per-epoch logs. Tables and figures are generated from the records; nothing is hand-typed into the paper. |
| P3 | **No tuning.** Hyperparameters are the ones the source README or paper publishes. Where libraries' published settings differ, the **common configuration** (§4) is fixed before any run. Every departure is listed under *Deviations*. |
| P4 | **No epoch selection.** Accuracy is read at the fixed final epoch (most READMEs print the last epoch). The best epoch is logged but never used as the headline. |
| P5 | **Identical inputs.** Each dataset is Booleanized **once** into a shared `.npz` (fixed split, fixed seed). Every library reads the same bits, so encoder differences cannot pass for model differences. Booleanization time is excluded from all timings. |
| P6 | **Timing runs are exclusive.** They run alone on the machine; load average and `nvidia-smi` are logged before and after each one. Accuracy-only runs (seeds 1–4) may share the machine; their timings are recorded but never reported. |
| P7 | A library that cannot be installed on this machine is reported as such, with the error. It is not patched beyond documented build flags; every workaround goes under *Deviations*. |

---------------------------------------------------------------------------------------------------

## 2. Libraries

| Short name | Package (version to pin) | Backend | Device(s) we run | Notes |
|---|---|---|---|---|
| **pyTM** | `pyTsetlinMachine` 0.6.6 | C, single thread | CPU ×1 thread | Reference baseline for speed-ups, as in ICML 2021 and clause indexing. sdist, needs gcc. |
| **pyTM-Par** | `pyTsetlinMachineParallel` 0.2.1 | C + OpenMP | CPU, 4 and 16 threads | GT Table I uses jobs=4; we add 16 = all physical cores. |
| **pyTM-CUDA** | `PyTsetlinMachineCUDA` 0.1.9 (+ `PyCoalescedTsetlinMachineCUDA` for B6 only) | PyCUDA kernels | GPU | Stale since 2021; needs PyCUDA + nvcc. GT Table I times the *coalesced* CUDA package. |
| **tmu** | `tmu` 0.8.3 | C (CFFI) / PyCUDA | CPU (1, 16 threads where supported) and GPU | A tmu 0.8.3 + PyCUDA venv already works in `experiments/multi_convtm/.cache/venv-tmu`. |
| **GT** | `green-tsetlin` 1.1.0 | C++ / pybind11, AVX2 | CPU, jobs 1, 4 and 16 | No GPU, no regression; convolution at the Python API level is unverified (Phase 0). |
| **tt** | `torchtsetlin` @ commit to pin (0.4.0) | PyTorch | CPU (1, 4, 16 threads) and GPU | Run in two modes, fixed in advance (§4.3). |

**Hardware:** one host, `mimir`: AMD Ryzen 9 3950X (16 cores, 32 threads), 121 GB RAM, RTX 3090 (24 GB) and
RTX 3080 (10 GB). **All timing runs use the RTX 3090**, the GPU used in green-tsetlin's Table I; the 3080
is only used for accuracy seeds.

**Not run, and why** (also stated in the paper):

| Library | Reason |
|---|---|
| Tsetlin.jl | Implements the Fuzzy-Pattern TM, a different algorithm with extra hyperparameters, in Julia. Its numbers are not comparable like for like. |
| GraphTsetlinMachine | Graph-structured input; it publishes no numbers to compare against. |
| tsetlin-rs | Rust only, and its benchmarks use synthetic data only. |
| FsTsetlin | F#, no release. |

---------------------------------------------------------------------------------------------------

## 3. Benchmarks (all taken from existing libraries or their papers)

"Libraries run" lists those that implement the model. Ticks in the source column point to `sources/`.

| ID | Benchmark | Data and Booleanization | Model and published hyperparameters | Epochs × runs | Source(s) | Libraries run |
|---|---|---|---|---|---|---|
| **B1** | Noisy XOR | pyTM's `NoisyXORTrainingData.txt` / `NoisyXORTestData.txt`: 12 bits, 40% training-label noise, 5000/5000 | flat TM, 10 clauses, T=15, s=3.9, `boost_tpf=0` | 200 × 5 seeds | pyTM, pyTM-Par, pyTM-CUDA READMEs; Granmo 2018, Table 15 | all six |
| **B2** | 2D Noisy XOR | pyTM's 4×4 binary-image files, 2500/10000 | conv TM, 40 clauses, T=60, s=3.9, patch 2×2 | 5000 × 5 seeds | pyTM, pyTM-Par READMEs; conv TM paper, Table 4 | pyTM, Par, CUDA, tmu, tt (GT if conv is exposed) |
| **B3** | Breast cancer (Wisconsin diagnostic) | scikit-learn `load_breast_cancer`; pyTM `Binarizer(max_bits_per_feature=10)` fitted on each training split; 100 random 80/20 splits | flat TM, 800 clauses, T=40, s=5.0 | 25 × 100 splits | pyTM, pyTM-Par READMEs | all six |
| **B4** | MNIST, flat, unweighted | Keras MNIST 60k/10k, pixel > 75 → 1 (784 bits) | flat TM, 2000 clauses, T=50, s=10.0 | 250 × 5 seeds | pyTM, pyTM-Par READMEs | all six |
| **B5** | MNIST, flat, weighted | as B4 | weighted TM, 2000 clauses, T=5000, s=10.0 | 100 × 5 seeds | pyTM (60 ep), pyTM-Par (100 ep), pyTM-CUDA (100 ep, T=800, max_weight 16), tmu example, Weighted TM paper | all six |
| **B6** | **MNIST clause sweep, cross-library timing** | MNIST, `fetch_openml` + `train_test_split(test_size=10000, random_state=42)`, > 75, exactly as GT's `performance_tests/tmu_mnist_dense.py` | clauses ∈ {10, 100, 1000, 2000, 5000}; s=3.0; T=C/4; literal budget 20 where supported; per-library model flags exactly as in that script | 5 epochs, test set scored after every epoch and *inside* the timed region; 3 timing repeats | **green-tsetlin paper, Table I** (the only existing cross-library benchmark); ICML 2021 Figs. 4, 7 and 9 (time vs clauses) | all six, with CPU thread levels as in §2 |
| **B7** | MNIST, convolutional, weighted | Keras MNIST, pixel ≥ 75, 28×28 | conv TM, 2000 clauses, T=5000, s=5.0, patch 10×10, weighted | 30 × 5 seeds | pyTM, pyTM-Par, pyTM-CUDA READMEs; tmu example | pyTM, Par, CUDA, tmu, tt (GT if conv is exposed) |
| **B8** | Fashion-MNIST, convolutional, weighted | Keras Fashion-MNIST; `cv2.adaptiveThreshold(GAUSSIAN_C, block 11, C 2)` | conv TM, 2000 clauses, T=5000, s=5.0, patch 10×10, weighted | 30 × 5 seeds | pyTM README (common configuration); pyTM-Par / CUDA use 8000 / 10000 / 10.0 (run once as the reproduction check R3) | as B7 |
| **B9** | IMDb sentiment | Keras `imdb.npz`, `num_words=5000`, 1- and 2-grams with min. frequency 10, `SelectKBest(chi2, k=5000)` | weighted TM, 10000 clauses, T=8000, s=10.0, no clause drop | 30 × 5 seeds | pyTM-Par README (pyTM and CUDA use the same pipeline with other T/s); ICML 2021 Table 3 (accuracy **and F1**, 5 runs) | all six |
| **B10** | California housing regression | scikit-learn `fetch_california_housing`; pyTM `Binarizer(max_bits_per_feature=10)`; 25 random 75/25 splits | regression TM, 1000 clauses, T=5000, s=2.75, weighted | 30 × 25 splits | pyTM, pyTM-Par READMEs (RMSD); ICML 2021 Table 5 (MAE, 5 runs) | pyTM, Par, CUDA, tmu, tt (GT has no regression) |
| ~~B11~~ *(dropped, D2)* | Conv-TM clause scaling: time per epoch and accuracy vs clauses | MNIST, Fashion-MNIST and Kuzushiji-MNIST, adaptive Gaussian threshold (11, 2) | conv TM, patch 10×10, weighted; clauses per class ∈ {250, 500, 1000, 2000, 4000, 8000}; T and s from conv TM paper Table 1 | 10 epochs × 1 seed (time per epoch; accuracy at epoch 10 is labelled as such) | conv TM paper, Table 3; coalesced TM paper, Tables 2–4 | GPU libraries (CUDA, tmu-GPU, tt-GPU) and pyTM-Par ×16 |

### 3.1 Reproduction checks (one per library, before the comparison)

Each library is first run on its **own** published configuration, to show we installed and called it
correctly. These use seed 0 only, and the published number is printed next to ours.

| ID | Library | Check | Published value |
|---|---|---|---|
| R1 | pyTM | B5 with the pyTM README configuration (60 ep) | 98.19%, 8.80 s train / 4.15 s test per epoch |
| R2 | pyTM-Par | B5 with its README configuration (100 ep, `OMP_NUM_THREADS=10`) | 98.14%, 1.70 s / 1.08 s |
| R3 | pyTM-CUDA | B8 with its README configuration (8000 clauses, T=10000, s=10, max_weight 255, 30 ep) | 90.94%, 44.02 s / 6.12 s |
| R4 | tmu | B5 with the tmu example configuration (2000, 5000, 10, budget 32, 60 ep), CPU and GPU | none published (tmu reports no numbers) |
| R5 | GT | B6 at 1000 and 5000 clauses, jobs 1 and 4 | 18 / 90 s (jobs 1), 11 / 31 s (jobs 4), on an i9-12900K |
| R6 | tt | B1 sequential mode | 99.3 ± 0.3% (Granmo 2018, Table 15) |

**Pass criterion, fixed now.**
- Accuracy: our value must be within **max(1.0 pp, 2 × seed sd)** of the published one.
- Timing: hardware differs, so no pass/fail. The ratio to the published value is reported.
- On a failure, the comparison still runs; the failure is reported, with its likely cause, in *Deviations*.

### 3.2 Considered and excluded (stated in the paper)

| Experiment found in the survey | Why excluded |
|---|---|
| Interpretability XOR demo (pyTM) | Prints clauses; it has no metric. Our rule-extraction example covers it in the paper's usage section. |
| Amazon reviews, sparse TM (GT, Sparse TM paper) | `torchtsetlin` has no sparse TM, and the dataset (2–8M docs) is beyond this budget. |
| Clause indexing speed-ups (pyTM paper) | An algorithmic feature that `torchtsetlin` does not implement; its comparison is with itself. |
| Text sets in ICML 2021 (BBC Sports, 20NG, SemEval, JAVA/Apple, BlogFeedback, Bike Sharing) | No library ships their preprocessing, so we would have to invent it. B9 (IMDb) and B10 (regression) cover the same table. |
| CIFAR-10 / CIFAR-100 | No library repository reports a number. TMComposites on CIFAR-10 is already measured in `experiments/multi_convtm` (tt vs the authors' tmu code) and can be cited from there. |
| Literal-budget sweep (clause-size paper, Table 3) | It ablates a model hyperparameter, not a library. |
| Energy (J) | No TM library reports measured energy. |
| torchtsetlin batch-size fidelity and throughput vs batch size (current paper Fig. 1) | A `torchtsetlin`-only ablation, kept in `benchmarks/` and **not** part of this cross-library plan (see the decision in §8). |

---------------------------------------------------------------------------------------------------

## 4. Making the configurations comparable

### 4.1 Phase 0 audit (before any run)

The libraries do not mean the same thing by the same argument. Each item below is checked in each
library's source and the answer goes into `config_map.md`, which **freezes the per-library arguments**
for every benchmark before the first measured run.

| Item | Why it matters |
|---|---|
| Clauses per class vs total | pyTM and tmu take clauses **per class**; tt takes `n_clauses_total`; GT and coalesced models share one pool. |
| `T` with weighted clauses | pyTM-CUDA multiplies T by `max_weight`; tt uses integer weights. |
| `boost_true_positive_feedback` default | tt defaults to `True`; B1 sets it off explicitly. |
| Number of automaton states | pyTM uses 128 per action (to verify); tt `n_states=128`. |
| Weight scheme | Integer-weighted in all libraries? `max_weight` caps in the CUDA package. |
| Literal budget semantics | Soft (tmu, tt 0.4 default) vs hard. |
| Conv position encoding | pyTM appends thermometer-coded patch coordinates; tt `position_encoding=True` must match. |
| Seeding | Whether pyTM and pyTM-Par can be seeded at all; if not, seeds are run-indices only. |
| Model sizes reported | How each library stores its state (bit-packed, int8, int32), for the model-size metric. |

### 4.2 Configuration policy

- **Common configuration.** For each benchmark we use the hyperparameters of the earliest source listed
  in §3. When the sources disagree (B5, B8, B9), we take the configuration that every library supports
  without code changes.
- **Native configuration.** Each library's own published configuration is run only as a reproduction
  check (§3.1).
- **B6 is the exception:** it copies GT's script argument-for-argument, model class included. The
  point there is to reproduce an existing cross-library timing table.

### 4.3 torchtsetlin settings (fixed in advance; not tuned)

| Mode | Setting | Why |
|---|---|---|
| `tt-seq` | `feedback_mode="sequential"` | The exact per-example algorithm, the same one pyTM implements. Run on B1–B5 and B10 (affordable sizes). |
| `tt-batch` | `feedback_mode="batch"`, batch 32 (flat) and 16 (conv) | The library's documented recommended range (16–64 flat, 8–32 conv). Fixed now and used on every benchmark. |

Both modes use the soft clause budget (the 0.4.0 default). Data is resident on the device, which is
ordinary `torchtsetlin` use. For other libraries we use each one's own normal `fit`/`predict` path.

---------------------------------------------------------------------------------------------------

## 5. Metrics

Every metric below is either reported by an existing library or paper, or is computed from the same
predictions and timers.

### 5.1 ML performance (test set, final epoch)

| Metric | Applies to | Reported by |
|---|---|---|
| Accuracy | all classification | all libraries and papers |
| F1 | binary: positive class; multi-class: macro | ICML 2021 Table 3; coalesced TM Table 7 |
| Precision, recall | macro (binary: positive class) | computed from the same predictions, as the components of F1 |
| RMSE (the READMEs call it RMSD) | B10 | pyTM / Par / CUDA READMEs |
| MAE | B10 | ICML 2021 Table 5 |
| Mean ± sd over seeds or splits | everything | ICML (5 runs), pyTM README (100 / 25 splits) |

### 5.2 Efficiency

| Metric | Definition | Reported by |
|---|---|---|
| Training time per epoch (s) | wall clock of one `fit` epoch, GPU synchronised; median of epochs 2…E, with epoch 1 reported separately (it includes PyCUDA / JIT compilation) | all cair READMEs; ICML Figs. 6 and 8; conv TM Table 3 |
| Inference time (s) | wall clock to predict the whole test set | all cair READMEs ("test" column); clause-indexing Tables 1–3 |
| Inference throughput (examples/s) | test-set size ÷ inference time | Tsetlin.jl README (predictions/s) |
| Total time for the protocol (s) | B6: 5 epochs including per-epoch test scoring | GT Table I |
| Speed-up | pyTM single-thread time ÷ library time, same benchmark | ICML Tables 3 and 5; clause-indexing tables |
| Model size (kB) | (a) analytic size of the learned state in the library's own format, (b) size of the saved file on disk | conv TM Table 5; coalesced Tables 2–4; sparse TM Table II; Tsetlin.jl |
| Peak memory (MiB) | peak host RSS (`psutil`, sampled at 50 ms) and peak GPU memory per process (NVML, sampled the same way for every GPU library) | Measured here; in the literature memory is only analytic (sparse TM Table II, green-tsetlin text). The paper's Table 1 already has a "Peak GPU memory" row. |

Timing hygiene:
- `torch.cuda.synchronize()` around `torchtsetlin` timers. PyCUDA calls are synchronous at `fit` return.
- CPU threads are set with `OMP_NUM_THREADS` / `n_jobs` / `torch.set_num_threads`, and processes are pinned with `taskset` to physical cores.
- The CPU governor and clock are recorded.

---------------------------------------------------------------------------------------------------

## 6. Phases

| Phase | Work | Output | Gate |
|---|---|---|---|
| 0. Setup | One venv per library under `.cache/venv-<lib>` (Python 3.10, numpy 2.2.6 if it works, else a pinned 1.26 venv, recorded); install-test each; download data; write `config_map.md` (§4.1) | `results/install_*.json`, `config_map.md` | **User reviews `config_map.md`** before measured runs |
| 1. Data | Booleanize each dataset once into `.cache/data/*.npz`, with checksums | `results/data_manifest.json` | — |
| 2. Smoke | Every (benchmark × library) for 1 epoch on 1 seed: same input bits, sensible accuracy, logging works | `results/smoke/*.json` | All cells green, or a documented failure |
| 3. Reproduction | R1–R6 (§3.1) | `results/repro_*.json` | Report to the user |
| 4. Timing runs (exclusive) | Seed 0 of B1–B10 for every library, plus 3 repeats of B6 | `results/<bench>_<lib>_<mode>_seed0.json` | — |
| 5. Accuracy seeds | Seeds 1–4 (B3: splits; B10: splits), run concurrently | `results/<bench>_<lib>_<mode>_seed<k>.json` | — |
| 7. Report | `code/report.py` writes `REPORT.md`, LaTeX tables and pgfplots data into the paper repo; `reproduce.sh` reruns any stage | paper tables filled | — |

**Planned layout** (code is written after the plan is approved):

```
experiments/paper_experiments/
  PLAN.md  README.md  config_map.md  LIBRARY_GAPS.md  REPORT.md (generated)  reproduce.sh
  sources/       survey_repos.md, survey_papers.md
  code/          data.py (shared Booleanization), runners/<lib>.py (one adapter per library,
                 same CLI and JSON schema), timing.py (timers, RSS/NVML sampler), queue.py, report.py
  results/       *.json (committed)
  .cache/        venvs, data, logs (git-ignored)
```

Each library runs in its own venv through a thin adapter, so no two libraries share a process or a
dependency set. The adapters write the same JSON schema.

---------------------------------------------------------------------------------------------------

## 7. Compute estimate

These are rough figures, taken from the READMEs' per-epoch times and our own `benchmarks/` numbers.
pyTM single-thread conv is the long pole.

| Block | Dominant cost | Estimate |
|---|---|---|
| Phase 0–2 | installs, data, smoke | ~0.5 day, mostly attended |
| B7 + B8, pyTM ×1 thread | ~540 s/epoch × 30 epochs × 2 benchmarks | ~9 h exclusive (seed 0) + ~36 h of seed runs in parallel |
| B9, pyTM | ~280 s/epoch × 30 | ~2.3 h exclusive + seeds |
| B4 / B5, pyTM | 8–9 s/epoch × (250 + 100) | ~1 h exclusive + seeds |
| All other library × benchmark cells | — | ~10 h exclusive |
| B6 (3 repeats × ~11 configurations × 5 clause counts) | pyTM and tmu-CPU at 5000 clauses dominate | ~5 h exclusive |
| **Tier 1 total** | | **≈ 1 day exclusive timing + ≈ 1.5 days of concurrent seed runs, about 3 days wall clock** |
| B11 (tier 2) | 3 datasets × 6 clause counts × 10 epochs × 4 libraries | ~1–1.5 days on the 3090 |

---------------------------------------------------------------------------------------------------

## 8. Decisions (user, 2026-10-22)

| # | Question | Decision |
|---|---|---|
| D1 | Common vs native configurations | **One shared configuration per benchmark** (§4.2). The native reproduction checks R1–R6 stay. |
| D2 | Tier 2 (B11) | **Dropped.** |
| D3 | torchtsetlin modes | **Both `tt-seq` and `tt-batch`.** The batch size is **matched to the other libraries where they expose one** (PyTsetlinMachineCUDA's `fit(..., batch_size=...)` as used in its demos). Elsewhere the fixed 32 (flat) / 16 (conv) applies. The value per benchmark is frozen in `config_map.md`. |
| D4 | Paper Fig. 1 (tt-only batch-size ablation) | **Dropped from the paper.** |
| D5 | Page budget | Ignored for now. |
| D6 | Reproducibility | Keep every script needed to reproduce every result (`reproduce.sh`, adapters, data preparation, report). |
| D7 | torchtsetlin gaps | Any gap found in torchtsetlin is reported in `LIBRARY_GAPS.md` and to the user (rule P1). |
| D8 | Gaps G1, G3, G4 (audit) | **Fixed in `src/` before the runs** (user, 2026-10-22). G2, G5–G7 stay open and are disclosed. |
| D10 | Run scope after the smoke phase (the plan as written needed ~2,950 h of exclusive timing, nearly all of it tt on the CPU and tt-seq on MNIST) | **Trimmed** (user, 2026-10-23): tt-seq only on B1, B2, B3, B10; tt-CPU timing over 3 epochs (B6: full 5); everything else as published; 5 seeds. ~4.5 days timing + ~2 days seeds. |
| D9 | tt batch size, after the audit showed CUDA's `batch_size` is a sync window, not a mini-batch | **Run both**: `tt-batch` (32 flat / 16 conv) and `tt-b100` (100, CUDA's number). |
| D11 | Run scope after the B1–B7 timing runs exposed tt GPU/CPU gaps (user, 2026-10-28) | **Stop the all-library run; fix gaps G10–G14 in `src/` (released as 0.5.1); re-run smoke on every GPU cell; run the full benchmark (timing, seeds, report) only for the GPU libraries: tt, PyTsetlinMachineCUDA, tmu-GPU** (`code/run_gpu.sh`). The two baselines are the only other GPU-capable libraries and both pass smoke on all 10 benchmarks (`results/gpu_selection_smoke.md`). CPU-library records of B1–B7 are kept; tt records from before the fix are in `results/timing_v0.5.0/`. Protocol for others: `docs/guides/benchmarking.md`. |

---------------------------------------------------------------------------------------------------

## Deviations

*(Filled in during the runs.)*
