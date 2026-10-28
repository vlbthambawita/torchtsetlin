# Benchmarking against other TM libraries

This guide is the protocol behind torchtsetlin's cross-library comparison
(`experiments/paper_experiments/`). It says where the data come from, what is trained with
which arguments, what is validated and tested, and what is measured and how. Follow it to
reproduce the published numbers, or to put another library or another machine on the same
footing.

The code lives in `experiments/paper_experiments/code/`; the frozen per-library arguments, with
file-and-line evidence from each library's source, are in
`experiments/paper_experiments/config_map.md`.

## Ground rules

| Rule | What it means in practice |
|---|---|
| **Existing benchmarks only** | Every benchmark is one that another library already ships (README demo, example, performance test) or that its paper reports. Running another library on it is fair; inventing a dataset or protocol is not. |
| **No tuning** | Hyperparameters are the published ones. Where libraries publish different settings, one *common configuration* per benchmark is fixed before any run. |
| **No epoch selection** | The headline number is the **final epoch**. The best epoch is logged, never reported as the result. |
| **Identical input bits** | Each dataset is Booleanized **once** into a shared `.npz`. Every library reads the same bits, so encoder differences cannot pass for model differences. Booleanization time is not timed. |
| **Records, not hand-typed numbers** | Every number comes from a `results/*.json` record that carries the command line, seed, git SHA, library version, hardware and the per-epoch log. Tables are generated from the records. |
| **Exclusive timing** | Timing runs run one at a time on an otherwise idle machine; the load average is logged before and after each. Seed runs may share the machine, and their times are never reported. |
| **The measured library is frozen** | `src/torchtsetlin` is not changed during a run. A defect found while benchmarking is written to `LIBRARY_GAPS.md`, fixed between runs if at all, and every record after the fix is re-measured. |

## 1. Get the data

```bash
cd experiments/paper_experiments
./reproduce.sh setup     # one virtualenv per external library (.cache/venv-<lib>)
./reproduce.sh data      # download + Booleanize every dataset once -> .cache/data/<bench>.npz
```

`code/prepare_data.py` follows each upstream demo script line for line and only replaces the
loader by a direct download of the same file Keras or scikit-learn would fetch. It writes
`results/data_manifest.json` with the shape, bit density and SHA-256 of every array: compare it
before trusting a run on a new machine.

| ID | Dataset | Source file | Booleanization | Train / test |
|---|---|---|---|---|
| B1 | Noisy XOR | pyTsetlinMachine `NoisyXORTrainingData.txt` / `TestData.txt` | already Boolean (12 bits, 40% train-label noise) | 5000 / 5000 |
| B2 | 2-D Noisy XOR | pyTsetlinMachine `2DNoisyXOR*Data.txt` | already Boolean, 4×4 | 2500 / 10000 |
| B3 | Breast cancer (WDBC) | `sklearn.datasets.load_breast_cancer` | pyTM `Binarizer(max_bits_per_feature=10)` fitted on all of X | 100 random 80/20 splits, split *k* = `train_test_split(..., random_state=k)` |
| B4, B5 | MNIST | Keras `mnist.npz` | `pixel > 75`, flattened to 784 bits | 60000 / 10000 |
| B6 | MNIST (OpenML) | `fetch_openml("mnist_784")` | `pixel > 75` | `train_test_split(test_size=10000, random_state=42)` |
| B7 | MNIST | Keras `mnist.npz` | `pixel >= 75`, kept as 28×28 | 60000 / 10000 |
| B8 | Fashion-MNIST | Keras Fashion-MNIST | `cv2.adaptiveThreshold(GAUSSIAN_C, block 11, C 2)` | 60000 / 10000 |
| B9 | IMDb | Keras `imdb.npz` (`num_words=5000`) | 1- and 2-grams with frequency ≥ 10, then `SelectKBest(chi2, k=5000)` fitted on the training set | 25000 / 25000 |
| B10 | California housing | `sklearn.datasets.fetch_california_housing` | pyTM `Binarizer(max_bits_per_feature=10)` | 25 random 75/25 splits |

## 2. What to train

### Benchmarks and their published hyperparameters

`C` is clauses **per class** (pyTsetlinMachine's convention). Every value comes from the source
named in `PLAN.md` §3; nothing is tuned.

| ID | Model | C | T | s | Boost TP | Weighted | Other | Epochs | Runs |
|---|---|---|---|---|---|---|---|---|---|
| B1 | flat | 10 | 15 | 3.9 | no | no | | 200 | 5 seeds |
| B2 | conv | 40 | 60 | 3.9 | no | no | patch 2×2 | 5000 | 5 seeds |
| B3 | flat | 800 | 40 | 5.0 | yes | no | | 25 | 100 splits |
| B4 | flat | 2000 | 50 | 10.0 | yes | no | | 250 | 5 seeds |
| B5 | flat | 2000 | 5000 | 10.0 | yes | yes | | 100 | 5 seeds |
| B6 | coalesced / per-class, as in green-tsetlin's script | 10 … 5000 | C/4 | 3.0 | yes | per script | literal budget 20 where supported | 5 | 3 repeats |
| B7 | conv | 2000 | 5000 | 5.0 | yes | yes | patch 10×10 | 30 | 5 seeds |
| B8 | conv | 2000 | 5000 | 5.0 | yes | yes | patch 10×10 | 30 | 5 seeds |
| B9 | flat | 10000 | 8000 | 10.0 | yes | yes | | 30 | 5 seeds |
| B10 | regression | 1000 | 5000 | 2.75 | yes | yes | | 30 | 25 splits |

The **same model** is expressed differently by each library: per-class vs total clause counts,
`max_weight=255` as the only weighting of PyTsetlinMachineCUDA, `n_clauses = C × K` for
green-tsetlin's shared pool, position encoding for convolution, and so on. Do not translate the
arguments by hand. `code/configs.py::cell_config(bench, lib, mode)` returns the frozen
constructor arguments of every cell, and `config_map.md` §(a) explains each mapping.

### torchtsetlin modes

torchtsetlin runs in up to three modes per benchmark, fixed in advance:

| Mode | Setting | Meaning |
|---|---|---|
| `seq` | `feedback_mode="sequential"` | the exact per-example algorithm that pyTsetlinMachine implements (B1, B2, B3, B10) |
| `batch` | `feedback_mode="batch"`, batch 32 (flat) / 16 (conv) | the library's recommended mini-batch range |
| `b100` | `feedback_mode="batch"`, batch 100 | the number PyTsetlinMachineCUDA's demos pass as `batch_size` (there it is a sync window, not a mini-batch) |

The whole Boolean dataset is resident on the device, which is ordinary torchtsetlin use. The
other libraries are driven through their own normal `fit` / `predict` path.

### Training loop

The loop is the one every upstream demo uses (`code/run.py`):

1. Load the shared `.npz`; for run *k*, permute the training order with
   `np.random.default_rng(k)`. pyTsetlinMachine cannot be seeded, so this permutation is the
   common source of run-to-run variance; each library additionally gets its own seed
   (torchtsetlin `torch.manual_seed(k)`, tmu `2k+1`, green-tsetlin `k+1`).
2. Build the model (timed separately as `build_s`).
3. For each epoch: one timed training epoch (`fit(..., epochs=1, incremental=True)`, tmu
   `fit`, torchtsetlin one shuffled pass of `update()` calls), then one timed prediction of the
   **whole test set**. torchtsetlin is put in `eval()` before predicting, because empty
   clauses vote differently while learning.
4. After the last epoch: final metrics, memory peaks, model size.

One process per (benchmark, library, mode, seed) cell, each library in its own virtualenv, so
no two libraries share a process or a dependency set.

## 3. What to validate, what to test

There is **no validation split and no early stopping**. The hyperparameters are the published
ones, so there is nothing to select on a validation set, and reading the test set to pick an
epoch would be selection on the test set. Instead, validation means checking that each library
is installed and called correctly:

| Stage | Command | Checks | Pass criterion |
|---|---|---|---|
| Smoke | `./reproduce.sh smoke` | every cell runs 1 epoch (B2: 2) on seed 0: same input bits, sensible accuracy, the record is complete | the record has `status: "done"` and a non-trivial accuracy |
| Reproduction | `./reproduce.sh repro` | each library on **its own** published configuration (R1–R4) | accuracy within **max(1.0 pp, 2 × seed sd)** of the published number. Timing has no pass/fail (hardware differs); the ratio is reported |
| Test | every cell | the upstream held-out test set, predicted after **every** epoch | the **final-epoch** value is the result; `best_epoch_logged_only` is kept for diagnosis |

A failed reproduction does not stop the comparison: the failure and its likely cause are
reported under *Deviations* in `PLAN.md`.

## 4. What to measure, and how

### ML performance (test set, final epoch)

| Metric | Definition (`code/harness.py`) |
|---|---|
| Accuracy | fraction of correct test predictions |
| Precision, recall, F1 | binary: positive class 1; multi-class: macro average over classes |
| RMSE, MAE | regression (B10), on the original target scale |
| Spread | mean ± sd over seeds (B3: 100 splits, B10: 25 splits) |

### Efficiency

| Metric | How it is measured |
|---|---|
| Training time per epoch | wall clock of one training epoch with the device synchronised before the timer stops (`torch.cuda.synchronize()`; PyCUDA calls are synchronous). Reported as the **median of epochs 2…E**; epoch 1 is reported separately because it includes JIT / `torch.compile` / PyCUDA compilation |
| Inference time | wall clock to predict the whole test set (median over epochs) |
| Throughput | test-set size ÷ inference time (examples/s) |
| Protocol time (B6) | construction + all 5 epochs including the per-epoch test prediction, as in green-tsetlin's Table I |
| Speed-up | single-thread pyTsetlinMachine time ÷ library time on the same benchmark |
| Peak memory | peak host RSS (`psutil`) and peak GPU memory held by the process (NVML), both sampled every 50 ms by the same probe for every library; torchtsetlin also records `torch.cuda.max_memory_allocated` |
| Model size | (a) the learned state in the library's own storage format (bytes), (b) the size of the saved file |

### Timing hygiene

- **One timing run at a time** on an idle machine (`./reproduce.sh timing` runs strictly
  sequentially). Check `env.loadavg_start` / `env.loadavg_end` in each record and re-run cells
  that overlapped other work.
- CPU threads are set with `OMP_NUM_THREADS`, `n_jobs` and `torch.set_num_threads`, and the
  process is pinned with `taskset` to physical cores (0–15 on the reference host; 16–31 are SMT
  siblings).
- All GPU timing runs use the same card (the RTX 3090 here); the CPU governor and the driver
  version are recorded in `env`.
- Data loading and Booleanization are outside every timer.

## 5. Run it

```bash
./reproduce.sh smoke                    # Phase 2: every cell, 1 epoch
./reproduce.sh repro                    # Phase 3: R1-R4
./reproduce.sh timing                   # Phase 4: seed 0 of every cell, exclusive
./reproduce.sh seeds --jobs 6           # Phase 5: remaining seeds / splits, concurrently
./reproduce.sh report                   # Phase 7: REPORT.md + LaTeX tables from the records
```

Every stage takes filters, so a subset can be run on its own, for example only the GPU
libraries on the GPU:

```bash
./reproduce.sh smoke  --lib pytmcuda tmu tt --device cuda
./reproduce.sh timing --lib pytmcuda tmu tt --device cuda --bench B4 B7
```

Re-running a stage skips every cell whose record already has `status: "done"`, so an
interrupted run resumes where it stopped. To re-measure a cell, move its record away first.
`code/launch.py list` prints the cell matrix.

## 6. Adding a library or a machine

- **A library**: write `code/adapters/<lib>.py` with `build`, `fit_epoch`, `predict` (and
  `sync` for asynchronous devices); add its constructor mapping to `code/configs.py` and the
  semantics of every argument to `config_map.md` *before* the first measured run; give it its
  own virtualenv in `code/setup_envs.sh`; then pass smoke and a reproduction check of its own
  published number.
- **A machine**: run `./reproduce.sh data` and compare `results/data_manifest.json` checksums,
  then smoke, then the timing stage. Only compare times measured on the same machine.
