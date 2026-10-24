# Benchmarks shipped by existing Tsetlin-machine libraries (survey, 2026-10-22)

How this was collected: GitHub REST API trees, raw.githubusercontent.com files, codeload source tarballs (read only,
never installed), PyPI JSON API, crates.io API, the readthedocs site, and the arXiv LaTeX source of the
Green Tsetlin paper. "Unverified" means I could not confirm the claim from a primary source, or confirming it
would need an install or run.

Abbreviations: C = clauses (per class unless noted), T = threshold, s = specificity, WC = weighted clauses,
CTM = convolutional TM, "thr75" = `np.where(x > 75, 1, 0)` (or `>= 75` where noted), thermometer-k = k
uniform levels `x >= (z+1)*255/(k+1)`.

---------------------------------------------------------------------------------------------------------

## 1. cair/pyTsetlinMachine (CPU, single-threaded C via ctypes)

- Repo: https://github.com/cair/pyTsetlinMachine. README: https://github.com/cair/pyTsetlinMachine/blob/master/README.md
- pip: `pyTsetlinMachine` 0.6.6 (2024-03-05). **sdist only** (no wheels), so a C compiler is needed. No `requires_python` and no
  `install_requires` (https://pypi.org/pypi/pyTsetlinMachine/json). setup.py builds the `libTM` extension, and tm.py loads it with
  `np.ctypeslib.load_library`.
- README "Requirements": "Python 3.7.x", Numpy, "Ubuntu or macOS".
- Python 3.10–3.12 / numpy 2: nothing in the code blocks it. A grep found no NumPy APIs removed in 2.0, and the code uses only ctypes and
  ndpointer with no compiled numpy ABI. **Unverified** because I did not install it.
- Examples folder: NoisyXORDemo, InterpretabilityDemo, 2DNoisyXORDemo, BreastCancerDemo, MNISTDemo,
  MNISTDemoWeightedClauses, MNISTDemo2DConvolutionWeightedClauses, FashionMNISTDemo2DConvolutionWeightedClauses,
  IMDbTextCategorizationDemo, RegressionDemo (https://github.com/cair/pyTsetlinMachine/tree/master/examples).
- Hardware: **not stated** anywhere in the README.
- Metrics reported: test accuracy (classification), RMSD (regression), and wall-clock seconds per epoch for training and testing.
  The README shows the first and last epochs only, with "..." in between.

| Demo | Data / Booleanization | Model & hyperparams | Reported (README) |
|---|---|---|---|
| NoisyXORDemo | NoisyXORTrainingData.txt / TestData.txt (12 bits) | MultiClassTM(C=10, T=15, s=3.9, boost_tpf=0), 200 epochs | "Accuracy: 100.00%" |
| InterpretabilityDemo | random 20 bits, XOR(x0,x1), 10% label noise, 5000/5000 | MultiClassTM(10, 15, 3.0), 200 epochs | "Accuracy: 100.0" + printed clauses |
| 2DNoisyXORDemo | 4x4 binary images from txt | MultiClassConvTM2D(40, 60, 3.9, patch 2x2), 5000 epochs | "Accuracy: 99.97%" |
| BreastCancerDemo | sklearn breast_cancer; `Binarizer(max_bits_per_feature=10)`; 100 random 80/20 splits | MultiClassTM(800, 40, 5.0), 25 epochs | "#100 Average Accuracy: 97.54% +/- 0.29" |
| MNISTDemo | keras MNIST, thr75 (>75) flat 784 | MultiClassTM(2000, 50, 10.0), 250 epochs | #1 94.63% train 42.75s test 3.06s; #250 **98.07%**, train **7.92s**, test **3.12s** |
| MNISTDemoWeightedClauses | same | MultiClassTM(2000, 5000, 10.0, WC), 60 epochs | #1 93.65% 60.98s/3.98s; #60 **98.19%**, train **8.80s**, test **4.15s** |
| MNISTDemo2DConvolutionWC | keras MNIST, `>= 75`, 28x28 | ConvTM2D(2000, 5000, 5.0, patch 10x10, WC), 30 epochs | #1 97.53% (403.06s); #30 **99.16% (538.87s)** |
| FashionMNISTDemo2DConvolutionWC | keras F-MNIST, `cv2.adaptiveThreshold(GAUSSIAN_C, block 11, C 2)` | ConvTM2D(2000, 5000, 5.0, 10x10, WC), 30 epochs | #1 85.25% (615.68s); #30 **90.06% (560.08s)** (#29 90.39%) |
| IMDbTextCategorizationDemo | keras IMDb, num_words=5000, uni+bi-grams with min freq 10, `SelectKBest(chi2, k=5000)` | MultiClassTM(10000, 8000, 5.0, WC, clause_drop_p=0.75), 40 epochs in code | #1 85.42% train 761.58s test 9.16s; #40 **89.76%**, train **278.88s**, test **11.70s**. The header says "over 50 epochs" but the output ends at #40, an inconsistency in the README. |
| RegressionDemo | sklearn California housing, `Binarizer(max_bits_per_feature=10)`, 25 random splits (default 75/25) | RegressionTM(1000, 5000, 2.75, WC), 30 epochs per run | "#25 RMSD: 0.61 +/- 0.00 (13.73s)", where the time is for 30 epochs |

## 2. cair/PyTsetlinMachineCUDA (GPU, PyCUDA kernels)

- Repo/README: https://github.com/cair/PyTsetlinMachineCUDA/blob/master/README.md
- pip: `PyTsetlinMachineCUDA` 0.1.9 (2021-06-19), sdist only, pure Python plus CUDA source compiled at runtime by PyCUDA
  `SourceModule`. setup.py declares **no dependencies**, so PyCUDA, numpy and a CUDA toolkit with nvcc must be installed by hand.
  README Requirements: "Python 3.7.x", Numpy, PyCUDA. pycuda on PyPI is currently 2026.1 with `requires_python ~=3.8`
  (https://pypi.org/pypi/pycuda/json). Py3.10–3.12 + numpy 2: **unverified**. The library imports `pycuda.autoinit`
  at module import, so importing it requires a GPU. The last push to the repo was 2023-01-03.
- Default kernel launch `grid=(16*13,1,1), block=(128,1,1)` (tm.py).
- Hardware: **not stated** in the README. The README cites the ICML 2021 paper (Abeyrathna et al., "Massively Parallel and
  Asynchronous TM Architecture"), whose own hardware and timings I did not check (unverified).

| Demo | Data / Booleanization | Hyperparams | Reported |
|---|---|---|---|
| NoisyXORDemo | txt files | MultiClassTM(10, 15, 3.9, boost_tpf=0), `fit(..., epochs=200, batch_size=100)` | 100.00% |
| InterpretabilityDemo | random 20 bits, noisy XOR | MultiClassTM(10, 15, 3.0), 200 epochs | 100.0 |
| MNISTDemoWeightedClauses | keras MNIST thr75 (>75) flat | MultiClassTM(2000, 50*16=800, 10.0, max_weight=16), 100 epochs | #1 93.00% train 4.91s test 0.63s; #100 **98.16%**, train **3.06s**, test **0.42s** |
| MNISTDemo2DConvolutionWC | MNIST `>= 75` | ConvTM2D(2000, 750, 5.0, 10x10, max_weight=16), 50 epochs | #1 97.14% 13.83s/1.82s; #50 **99.10%**, train **9.14s**, test **1.38s** |
| FashionMNISTDemo2DConvolutionWC | adaptiveThreshold(11,2) | ConvTM2D(8000, 10000, 10.0, 10x10, max_weight=255), 30 epochs | #1 82.64% 49.47s/7.13s; #30 **90.94%**, train **44.02s**, test **6.12s** |
| IMDbTextCategorizationDemo | same IMDb pipeline (5000 words, 1–2-grams, chi2 5000) | MultiClassTM(10000, 80*16=1280, 27.0, max_weight=16), 50 epochs | #1 85.15% 18.19s/6.61s; #50 **89.57%**, train **10.41s**, test **5.84s** (#49 89.70%) |
| RegressionDemo | California housing, `Booleanizer(max_bits_per_feature=25)`, a single 75/25 split | RegressionTM(16000, 80000, 10.0, max_weight=255), 100 epochs | "#25 RMSD: 0.489 +/- 0.001 (11.97s)" |

There is no BreastCancer, MNIST-unweighted or 2D Noisy XOR demo in this README. The examples folder also has
MNISTDemoWeightedClauses, MNISTDemo2DConvolutionWeightedClauses, FashionMNISTDemo2DConvolutionWeightedClauses, IMDb, NoisyXOR,
Interpretability and Regression (https://github.com/cair/PyTsetlinMachineCUDA/tree/master/examples).

## 3. cair/pyTsetlinMachineParallel (OpenMP CPU)

- README: https://github.com/cair/pyTsetlinMachineParallel/blob/master/README.md
- pip: `pyTsetlinMachineParallel` 0.2.1 (2021-10-10), sdist only. Compiled with `-fopenmp` and linked with `-lgomp`
  (GCC/libgomp, so it will not build with Apple clang without extra work). No requires. README: Python 3.7.x, Numpy, "OpenMP 5.0",
  Ubuntu or macOS. ctypes loading is the same as pyTsetlinMachine. Py3.10–3.12/numpy 2: **unverified**. Last push 2023-01-03.
- Thread count: the install block says `export OMP_NUM_THREADS=10`. That is the only thread count the README states, and it does
  not tie it to the timings below. CPU model **not stated**.
- Same examples folder as pyTsetlinMachine.

| Demo | Hyperparams (where they differ from pyTM) | Reported |
|---|---|---|
| NoisyXOR | (10, 15, 3.9), 200 ep | 100.00% |
| Interpretability | (10, 15, 3.0) | 100.0 |
| 2DNoisyXOR | ConvTM2D(40, 60, 3.9, 2x2), 5000 ep | 99.71% |
| BreastCancer | (800, 40, 5.0), 25 ep, 100 splits | 97.54% +/- 0.29 |
| MNISTDemo (thr75) | (2000, 50, 10.0), 250 ep | #1 94.91% 4.43s/1.02s; #250 **98.21%**, train **1.70s**, test **1.06s** |
| MNIST WC | (2000, 5000, 10.0, WC), **100 ep** | #1 93.58% 6.01s/1.53s; #100 **98.14%**, train **1.70s**, test **1.08s** |
| MNIST 2D conv WC (>=75) | (2000, 5000, 5.0, 10x10, WC), 30 ep | #1 97.78% (26.13s); #30 **99.12% (29.52s)** |
| F-MNIST 2D conv WC (adaptive thr) | **(8000, 10000, 10.0, 10x10, WC)**, 20 ep | #1 84.98% (72.75s); #20 **90.77% (69.71s)** |
| IMDb | **(10000, 8000, 10.0, WC)**, no clause drop, 30 ep | #1 85.61% 86.99s/21.02s; #30 **89.08%**, train **39.81s**, test **26.38s** |
| Regression (Cal. housing, 10 bits) | (1000, 5000, 2.75, WC), 30 ep | "#25 RMSD: 0.61 +/- 0.00 (1.02s)" |

## 4. cair/tmu (Python + CFFI C + optional PyCUDA)

- Repo: https://github.com/cair/tmu (default branch `main`, last push 2026-08-03).
- pip: `tmu` 0.8.3 (PyPI release 2024-03-25). It has **wheels for cp310/cp311/cp312** (manylinux x86_64, macOS x86_64, win_amd64),
  `requires_python >=3.10`, deps `cffi>=1.15, numpy` (unpinned), pandas, scikit-learn, tqdm, requests, xxhash. Extras:
  `composite` (opencv-python-headless, pydantic, optuna, joblib) and `examples` (scikit-learn, tensorflow)
  (https://pypi.org/pypi/tmu/json, https://github.com/cair/tmu/blob/main/pyproject.toml). The README tells users to install from
  git (`pip install git+https://github.com/cair/tmu.git`), and the repo `__version__` is still "0.8.3". The C library is called
  through CFFI, so there is no numpy ABI coupling. A grep of `tmu/` found no NumPy APIs removed in 2.0. Numpy 2 compatibility is
  therefore plausible but **unverified**. GPU needs `pycuda` (not declared). There is no published wheel for macOS arm64.
- **No results are reported**. The README's "Examples - Classification" section is an empty heading, and
  examples/classification/README.md is empty. No benchmark tables, accuracies or timings are published in the repo.
- Benchmark tooling:
  - `scripts/performance_test/performance_test.py` checks out every commit of a branch, reinstalls, and runs an example
    (default `MNISTDemo.py`) with `--epochs=2`. `match.json` has the regex `"Epoch: \d+, Accuracy: ([\d.]+), Training Time: ([\d.]+)s,"`.
    This is a regression-timing harness and ships no results.
  - `scripts/example_verifier/verify_experiments.py` runs the examples.
  - Example scripts log accuracy, train_time and test_time per epoch via `tmu.tools.BenchmarkTimer`.
- `tmu/data` loaders (https://github.com/cair/tmu/tree/main/tmu/data):
  - `MNIST`: Keras mnist.npz from GCS, booleanized `> 75` and flattened to 784, uint32.
  - `FashionMNIST` and `KuzushijiMNIST`: sklearn `fetch_openml`, `> 75`.
  - `CIFAR10` and `CIFAR100`: Toronto tarballs, raw values with no booleanization.
  - `IMDB`: Keras imdb.npz, raw word ids.
  - Intrusion detection: `KDD99`, `NSLKDD`, `UNSW_NB15`, `Bot_IoT`, `CICIDS2017`, booleanized from the float bit patterns.
  - `TMUDatasetSource`: downloads from the GitHub repo `cair/tmu-datasets`, for example "XOR_biased".
  - Note: `tmu/data/__init__.py` exports only TMUDataset, Bot_IoT, CICIDS2017, KDD99, MNIST, NSLKDD, UNSW_NB15 and IMDB.
    FashionMNIST, CIFAR10 and CIFAR100 must be imported from their submodules.

### tmu non-experimental examples (examples/{classification,regression,autoencoder,composite})

| Script | Dataset / Booleanization | Model & default hyperparams |
|---|---|---|
| classification/XORDemo.py | tmu-datasets "XOR_biased" (1000/1000) | TMClassifier C=4, T=10, s=10, budget 32, WC, 60 ep |
| classification/InterpretabilityDemo.py | random 20 bits, noisy XOR (10%) | TMClassifier C=10, T=10, s=3.0 |
| classification/MNISTDemo.py | `tmu.data.MNIST` (>75, flat) | TMClassifier C=2000, T=5000, s=10, max_included_literals=32, WC, platform CPU/CPU_sparse/CUDA, 60 ep, seed 42 |
| classification/MNISTDemoCoalesced.py | MNIST >75 | TMCoalescedClassifier C=4000 (total), T=1000, s=10, WC, focused negative sampling, 60 ep |
| classification/MNISTConvolutionDemo.py | MNIST >75 reshaped to 28x28 | TMClassifier C=8000, T=10000, s=5, patch 10x10, budget 32, WC, GPU, 60 ep |
| classification/MNISTSparseAbsorbingDemo.py | MNIST >75 | TMClassifier CPU_sparse C=2000, T=5000, s=10, absorbing=100, literal_sampling 0.1, 250 ep |
| classification/MNISTDemoWithSerialization.py | MNIST | C=50, T=10, s=10, 10 ep (pickle timing) |
| classification/IMDbTextCategorizationDemo.py | Keras IMDb, 5000 words, 1–2-grams, chi2 k=5000 | TMClassifier C=10000, T=8000, s=2.0, WC, clause_drop_p 0.75, GPU, 40 ep |
| classification/IMDbSparseAbsorbingTextCategorizationDemo.py | same | CPU_sparse, C=10000, T=8000, s=2.0, absorbing 90, 60 ep |
| classification/CIFAR2Demo3x3LiteralBudget.py | CIFAR-10 to binary animal vs vehicle; thermometer-8 per RGB channel | TMClassifier C=8000, T=0.75C, s=10, patch 3x3, budget 32, WC, GPU, 250 ep, 5 ensembles |
| regression/RegressionDemo.py | California housing, `StandardBinarizer(max_bits_per_feature=10)` | TMRegressor C=1000, T=5000, s=2.75, WC, 30 ep x 25 runs (RMSD) |
| composite/TMCompositeCIFAR10Demo.py | CIFAR-10 raw; 4 components: AdaptiveThresholding (C2000, T500, s10, 10x10), ColorThermometer res 8 (C2000, T1500, s2.5, 3x3), ColorThermometer res 8 (4x4), HOG (C2000, T50, s10, unweighted); budget 32 | 30 epochs per component |
| autoencoder/DimensionalityReductionDemo.py, IMDbWordEmbeddingDemo.py | synthetic / IMDb | TMAutoEncoder |

### tmu experimental examples (examples/experimental/classification, 59 scripts; no results)

These are mostly hard-coded research scripts:
- CIFAR-10: Demo/DemoS/CoalescedDemo use `cv2.adaptiveThreshold(11,2)` per channel, C=2000, T=50, s=10, budget 32, 60 ep. The 3x3
  variants (CIFARDemo3x3: coalesced C=2500, T=0.75C, s=5; CIFARDemoVanilla3x3: C=128, s=10, CPU; CIFAR10Coalesced3x3: C=8000;
  ...LiteralBudget: C=80000, budget 32) use thermometer-8, 250 epochs and 5–10 ensembles on CUDA.
- CIFAR-100 (thermometer-8, 3x3, C=8000–80000), CIFAR-2 variants (HSV, RGC, histogram, specificity, visual tokens).
- MNIST variants: weighted (C=2000, T=5000, s=10, CUDA, 60 ep); 2D conv WC (C=2000, T=5000, s=5, 10x10, CPU, 250 ep); coalesced;
  literal budget; one-vs-one; Type III; focused negative sampling; incremental; multi-output.
- FashionMNIST: Demo3x3 (thermometer-8, coalesced), DemoS (adaptive threshold), Multioutput (fetch_openml).
- IMDb word prediction, embedding and Type III; BillionWords; Concept learning; Combinatorial composition; Relational TM.

## 5. ooki/green_tsetlin (C++/pybind11, multithreaded, AVX2/NEON, sparse TM)

- Repo: https://github.com/ooki/green_tsetlin. Docs: https://green-tsetlin.readthedocs.io/en/latest/ (Installation, User Guide, and
  tutorials for IMDB, Iris, Amazon and MNIST; the docs site has **no benchmark section**).
- pip: `green-tsetlin` 1.1.0 (2025-08-25). **All 13 releases are sdist-only**, so building needs a C++17 compiler, pybind11>=3.0.1 and
  py-cpuinfo at build time. `requires_python >=3.8`; deps `numpy>=1.24, scipy>=1.10.1, scikit-learn>=1.2, tqdm>=4.65, optuna`
  (https://pypi.org/pypi/green-tsetlin/json).
- setup.py concerns:
  - Line 11 runs `import distutils; distutils.log.set_verbosity(1)`. distutils was removed from the 3.12 stdlib, and this only works
    through setuptools' distutils shim (setuptools is imported earlier, so it probably works).
  - On Linux it always passes `-mavx2 -mfma`, so it is effectively x86-64 only.
  - Py3.10–3.12/numpy 2 is plausible because the build uses pybind11 and the numpy pin is only a lower bound, but this is
    **unverified**.
- README: API snippets only, with **no numbers**.
- Reported numbers in docs and examples:
  - MNIST tutorial (https://github.com/ooki/green_tsetlin/blob/master/docs/mnist.rst and examples/mnist.ipynb):
    - Data: `fetch_openml("mnist_784")`, then a **random 80/20 split** (`test_size=0.2, random_state=42`), so this is not the
      standard 60k/10k split. Pixels are thresholded `> 75`.
    - Hyperparameters: from an Optuna search over a 10k/1k subset (best score 0.99373): `TsetlinMachine(n_clauses=6154,
      s=21.6277, threshold=1218, literal_budget=10)`. Trained with `Trainer(n_epochs=20, k_folds=2, n_jobs=7)` in the notebook
      (`n_epochs=10, n_jobs=1` in the rst).
    - Result: `'best_eval_score': 0.9918857142857143`, with epoch training times of about **27–42 s** (first epoch 41.998 s,
      later about 27.3 s).
    - The hardware is not stated. Caveat: with k_folds=2 the eval set appears to be a fold. The progress line says
      "best eval score: 0.992 (epoch: 0)" for fold 2, which looks suspicious. I would not treat 99.19% as a held-out test accuracy.
  - Amazon reviews (docs/amazon.rst):
    - Data: 2M documents of All_Amazon_Review.json.gz, CountVectorizer 1–3-grams (min_df 3, max_df 0.8), chi2 k=1,000,000, 5
      classes (star rating).
    - Model: SparseTsetlinMachine(C=2000, s=2.0, T=5000, boost_tp, dynamic_AL; active_literals 130, clause_size 140,
      lower_ta_threshold -90).
    - Result: `train_time_of_epochs [1722.16, 1685.90, 1664.17]`, `best_test_score 0.639`.
  - IMDB:
    - Data: HF `datasets` imdb, train split only, then an 80/20 split. CountVectorizer 1–2-grams, max_features 5000, binary.
    - Training: hpsearch with C 100–1000, T 100–1500 and s 2–20.
    - No accuracy is reported. imdb_sparse_example.ipynb uses 1–3-grams with 30k features and SparseTM(C=500, s=2, T=2000,
      budget 10), also without reported numbers.
  - Iris: equal-width 4-level thermometer and hpsearch. Reports best score 1.0 with {s 2.77, C 514, T 501, budget 7}.
  - California housing notebook: 5 equal-width bins one-hot. I saw no metric output.
- `performance_tests/` (https://github.com/ooki/green_tsetlin/tree/master/performance_tests):
  - `tmu_mnist_dense.py` is the **cross-library MNIST timing benchmark**:
    - Data: fetch_openml MNIST, `train_test_split(test_size=10_000, random_state=42)`, `>75`.
    - Clause sweep: C ∈ {10, 100, 1000, 2000, 5000}, s=3.0, T=C//4, 5 epochs, the test set evaluated after each epoch, total
      wall time recorded.
    - Library configurations:
      - GT: boost_tp, literal_budget 20, n_jobs 1 or 4.
      - tmu: `TMCoalescedClassifier(weighted, focused_negative_sampling, max_included_literals=20)`, on CPU and GPU.
      - pyTsetlinMachine: (boost_tpf=1, WC, max_included_literals=20).
      - pyTsetlinMachineParallel: (boost_tpf=1, WC).
      - PyTsetlinMachineCUDA: (boost_tpf=1).
    - A comment in the file records "tmu: epoch [0] 125.900 ... [4] 82.660; gt1: 114.00 (all 5); MultiClassTsetlinMachine
      total time: 206.603".
  - `mnist_dense.py`: random-configuration scaling trials (n_literals, n_examples, n_clauses up to 20000, n_jobs up to 24) on a
    6x-enlarged MNIST. The analysis notebook shows "Intel(R) Xeon(R) CPU E5-1620 v3 @ 3.50GHz, 4 physical / 8 logical cores,
    604 trials".
  - `inference/` contains C-export inference timers for MNIST and IMDB models (no numbers committed).
- Companion paper, arXiv:2405.04212 (Glimsdal et al. 2024), Table "Time in seconds needed to train 5 epochs on MNIST, where the test set
  is evaluated after each epoch", with hardware **"i9-12900K CPU, GPU: GeForce RTX 3090"** and "All frameworks have roughly the same
  accuracy, so they are omitted". Source: https://arxiv.org/abs/2405.04212 (LaTeX e-print).

| Framework \ #Clauses | 10 | 100 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| TMU (CPU) | 29.6 | 33 | 80 | 128 | 266 |
| TMU (GPU) | 87.8 | 62 | 70 | 75 | 95 |
| pyTsetlinMachine | 4.5 | 9 | 47 | 84 | 197 |
| pyTsetlinMachineParallel (jobs=4) | 1.1 | 2 | 17 | 47 | 149 |
| PyCoalescedTsetlinMachineCUDA | 14.8 | 26 | 30 | 36 | 88 |
| GT (jobs=1) | 0.3 | 2 | 18 | 36 | 90 |
| GT (jobs=4) | 4.0 | 5 | 11 | 18 | 31 |

  The paper also claims that 8M Amazon documents with a 5M vocabulary need "a TM of 1 MiB as opposed to 18.6 GiB for a dense TM".
  That claim is not backed by a table in the repo.

## 6. BooBSD/Tsetlin.jl (Julia, Fuzzy-Pattern TM — **not a vanilla TM**)

- Repo: https://github.com/BooBSD/Tsetlin.jl (last push 2026-10-21, MIT). It is a Julia package used via `include("src/Tsetlin.jl")`,
  **not on PyPI**, and needs no external Julia deps. Examples pull MLDatasets, and the Python data-prep scripts pin keras 3.13.0,
  tensorflow 2.20.0, scikit-learn 1.8.0, pandas 2.3.3 and kagglehub 0.3.13 (examples/requirements.txt).
- Algorithm differences: it implements FPTM (arXiv:2508.08350) with extra hyperparameters `L` (max literals per clause), `LF`
  (allowed literal misses per clause), `states_num` and `include_limit`. Accuracy and speed are therefore not like-for-like with the
  vanilla or weighted TM.
- README claims (https://github.com/BooBSD/Tsetlin.jl/blob/main/README.md):
  - "MNIST training for 1,000 epochs finishes in just **5 seconds on a CPU**." Unverified: no log is shown and the configuration
    is not specified.
  - "**38 million** MNIST predictions per second at 98% accuracy, with **4.8 GB/s** throughput on a desktop CPU."
  - "Up to 15x faster training and 41x faster inference compared to the original FPTM implementation". No table is given, so
    this is unverified.
- README benchmark screenshot (https://github.com/user-attachments/assets/106c6b58-4fcd-4827-90a1-d3ff5b81c9cc), from
  `julia -O3 -t auto examples/MNIST/mnist.jl`:
  - Hardware: "CPU: AMD Ryzen 9 7950X3D 16-Core Processor", "Running in 32 threads".
  - Input and model: "Input vector size: 784 bits. Density: 16.36%" and "Average clause literal density: 11.98%".
  - Throughput: "320000000 predictions processed in 8.347 seconds. Performance: 38335896 predictions per second.
    Throughput: 4.856 GB/s. Input data size: 40.531 GB."
  - Model size: "Parameters during training: 313600. Parameters after training and compilation: 37558."
  - "Accuracy: 98.10%."
  - The built-in `benchmark()` reports predictions/s, GB/s and the parameter count. This is the only library that reports model
    size.
- Examples and the in-code accuracy comments (all use MLDatasets standard splits unless noted):

| Example | Preprocessing | Hyperparams | Reported |
|---|---|---|---|
| MNIST/mnist.jl | `booleanize(x, 0.2)`, 1 bit/pixel (a 4-level variant is commented out) | C=20, T=16, S=800, L=150, LF=75, states 256, include_limit 128, 1000 ep. The README snippet uses (0, 0.5), T=20, S=200 | screenshot above: 98.10% |
| FashionMNIST/fmnist_conv.jl | raw + oriented-gradient convolution features (3/5/7/9 kernels), booleanized | C=20, T=100, S=700, L=200, LF=200, 1000 ep | comment "Best accuracy: 93.59% after 857 epochs"; C=2: "92.53% after 713 epochs" |
| FashionMNIST/fmnist_conv_augmented.jl | + data augmentation | C=200, T=1128, S=1000, L=1000, LF=800, 200 ep | "acc: 94.49% after 40 epochs"; C=8000: "Best accuracy: 94.74% after 11 epochs" |
| FashionMNIST/fmnist_hdc.jl | HDC hypervectors 32768 bits | C=128, T=1024, S=32768, L=LF=16384, states 64000 | none |
| CIFAR10/cifar10_conv.jl | prepare_dataset.jl: oriented convolution features | C=20, T=1600, S=1000, L=LF=4000, 1000 ep | comment "(69%+ acc)" |
| CIFAR10 HDC, HDC contrastive; CIFAR100 HDC | HDC hypervectors | C=128 ... | none |
| IMDb/imdb_minimal.jl | prepare_dataset.py: Keras IMDb, 1–4-grams, chi2 12800 features, 40000 words | C=1, T=18, S=1000, L=LF=64, 1000 ep | none |
| IMDb/imdb_optimal.jl | 1–4-grams, 65535 features, 70000 words | C=200, T=250, S=2000, L=100, LF=10, 20 ep | none ("Maximum accuracy after 15-20 epochs") |
| AmazonSales/amazon.jl | Kaggle Amazon sales, noise 0.005 | C=2000, T=100, S=1000, L=50, LF=10 | none |
| NoisyParity | 12 features, 4-variable parity, 20k examples, 20% noise | 1e6 epochs | none |
| TEXT (Shakespeare char-level generation) | HDC | — | sample text only |

## 7. RAprogramm/tsetlin-rs (Rust crate)

- Repo: https://github.com/RAprogramm/tsetlin-rs (7 stars, last push 2026-06-22). crates.io `tsetlin-rs` 0.3.4, Rust edition 2024,
  rust-version 1.92. **No Python bindings and nothing on PyPI.**
- Benchmarks: Criterion `benches/benchmarks.rs` and `benches/sparse_benchmarks.rs`. All data is **synthetic**: xor, parity_8,
  "mnist_sparse" (random ~15%-density 784-bit vectors, 200 samples, not real MNIST) and nlp_5k. Examples: xor.rs, iris.rs (an
  "Iris-like" synthetic set with 8 features and 3 classes, reporting training accuracy) and benchmark_advanced.rs (synthetic
  noise/scale tests).
- README numbers (micro-benchmarks only, **no hardware stated**, no real-dataset accuracy):
  - "4.4x parallel speedup | 116x bitwise evaluation".
  - Parallel training (100 clauses, 64 features, 1 epoch): 1000 samples, Sequential 29.15 ms, V1 19.92 ms, V2 6.58 ms (4.4x).
  - Bitwise evaluation at 1024 features: scalar 1.32 µs vs bitwise 11 ns.
  - Sparse inference on "MNIST-like" 784 features: compression 117.6x, speedup 65x.
  - Clause type timings: 2.5–300 ns.
  - partial_fit: 548 µs vs fit 1.15 ms.
  - It also repeats "Original CUDA implementation reports 50x speedup on GPU".

## 8. cair/GraphTsetlinMachine (brief)

- pip `GraphTsetlinMachine` 0.3.3 on PyPI (2025-04-04, sdist, no deps declared). The repo setup.py says 0.3.4. It is CUDA-only through
  PyCUDA. The repo's pixi config pins python >=3.11,<3.12, numpy >=2.3.5, cuda 12.9 and pycuda 2025.1.x
  (https://github.com/cair/GraphTsetlinMachine/blob/master/pyproject.toml), which suggests numpy 2 works.
- Examples (https://github.com/cair/GraphTsetlinMachine/tree/master/examples):
  - MNISTVanillaDemo: thr75, C=20000, T=25000, s=10, 250 ep, hypervector 1024.
  - MNISTConvolutionDemo: thr75, 10x10 patches on a 19x19 grid, C=20000, T=25000, s=10.
  - CIFAR10Demo: C=40000, T=7500, s=20, patch 8.
  - NoisyXOR: C=10, T=100, s=1, depth 2.
  - NoisyXORMNIST, SequenceClassification, MultiOutput and NoisyOR.
- The README reports **no accuracy or timing**. It only gives grid/block settings for DGX-2/A100 `grid=(16*13,1,1)` and H100
  `grid=(16*13*4,1,1)`.

---------------------------------------------------------------------------------------------------------

## 9. Installability summary (Python 3.10–3.12, numpy 2) — none of this was install-tested

| Package (pip name) | Latest on PyPI | Artifacts | Declared python / deps | Assessment |
|---|---|---|---|---|
| `pyTsetlinMachine` | 0.6.6 (2024-03) | sdist (C ext) | none / none; README says 3.7 | likely builds with gcc; numpy-2 safe by inspection; unverified |
| `pyTsetlinMachineParallel` | 0.2.1 (2021-10) | sdist (C + OpenMP) | none | needs gcc + libgomp; unverified |
| `PyTsetlinMachineCUDA` | 0.1.9 (2021-06) | sdist (pure py) | none; needs PyCUDA + nvcc + GPU at import | unverified; stale since 2021 |
| `tmu` | 0.8.3 (2024-03) | wheels cp310–312 (x86_64 Linux/mac/win) | >=3.10; numpy unpinned; GPU needs pycuda | most likely to install cleanly; numpy 2 not verified |
| `green-tsetlin` | 1.1.0 (2025-08) | sdist only (C++17, AVX2 flags) | >=3.8; numpy>=1.24 | needs compiler; x86-64 Linux only in practice; unverified |
| `GraphTsetlinMachine` | 0.3.3 (2025-04) | sdist | none; CUDA + PyCUDA | repo pins numpy>=2.3.5, py3.11 |
| Tsetlin.jl | n/a (Julia) | git | Julia | not a Python library |
| tsetlin-rs | crates.io 0.3.4 | Rust | Rust 1.92 | not a Python library |

---------------------------------------------------------------------------------------------------------

## 10. Cross-library matrix

Cells give the reported test accuracy / time with the configuration. "code" means the script exists but reports no numbers.
"—" means there is no such experiment. Times are wall-clock per epoch unless noted, and hardware is unstated unless noted.

| Dataset · task · model | pyTsetlinMachine | pyTMParallel | PyTMCUDA | tmu | green_tsetlin | Tsetlin.jl (FPTM) | tsetlin-rs | GraphTM |
|---|---|---|---|---|---|---|---|---|
| Noisy XOR (12-bit txt) · binary · flat TM | 100% (10,15,3.9; 200 ep) | 100% | 100% | XOR_biased code | — (perf_tests/xor.py, no numbers) | NoisyParity code | synthetic XOR (Criterion µs) | NoisyXOR code |
| Noisy XOR interpretability (20 bits) | 100% | 100% | 100% | code | — | — | — | — |
| 2D Noisy XOR 4x4 · CTM 2x2 | 99.97% (5000 ep) | 99.71% | — | — | — | — | — | — |
| Breast cancer · binary · TM + 10-bit binarizer | 97.54±0.29% (100 splits) | 97.54±0.29% | — | — | — | — | — | — |
| Iris · 3-class · thermometer | — | — | — | — | 1.0 (hpsearch) | — | synthetic "Iris-like", train acc | — |
| MNIST thr75 · flat TM (unweighted) | 98.07%; 7.92s train / 3.12s test (2000,50,10; ep 250) | 98.21%; 1.70s / 1.06s | — | — | — | — | — | — |
| MNIST thr75 · flat weighted TM | 98.19%; 8.80s / 4.15s (2000,5000,10; ep 60) | 98.14%; 1.70s / 1.08s (ep 100) | 98.16%; 3.06s / 0.42s (2000,800,10, max_w 16; ep 100) | code (2000,5000,10, budget 32, 60 ep) | 99.19%* ≈27–30 s/epoch (6154 C, random 80/20 split, k-fold) | 98.10%, 38.3M pred/s, 32 thr Ryzen 7950X3D (FPTM C=20, 1-bit 0.2 thr) | synthetic "MNIST-like" only | Vanilla MNIST code |
| MNIST thr75 · cross-library 5-epoch total time, C ∈ {10..5000}, s=3, T=C/4 | 4.5–197 s | 1.1–149 s (jobs=4) | 14.8–88 s | CPU 29.6–266 s; GPU 87.8–95 s | jobs1 0.3–90 s; jobs4 4.0–31 s | — | — | — |
| MNIST ≥75 · conv TM 10x10, weighted | 99.16%; 538.87 s/ep (2000,5000,5; 30 ep) | 99.12%; 29.52 s/ep | 99.10%; 9.14s / 1.38s (2000,750,5; 50 ep) | code (8000,10000,5, GPU, 60 ep) | — | — | — | Conv MNIST code |
| Fashion-MNIST adaptiveThr(11,2) · conv TM 10x10 WC | 90.06%; 560 s/ep (2000,5000,5; 30 ep) | 90.77%; 69.71 s/ep (8000,10000,10; 20 ep) | 90.94%; 44.02s / 6.12s (8000,10000,10; 30 ep) | experimental code (DemoS adaptive; Demo3x3 thermometer) | — | 93.59% (FPTM + conv features, 857 ep); 94.74% with augmentation | — | — |
| CIFAR-10 · conv / thermometer | — | — | — | composite demo code; experimental thermometer-8 3x3 and adaptive-threshold code; no numbers | — | "69%+" comment (FPTM conv features) | — | CIFAR10Demo code |
| CIFAR-2 (animal vs vehicle) · thermometer-8 CTM 3x3 | — | — | — | code (8000 C, budget 32) | — | — | — | — |
| CIFAR-100 | — | — | — | experimental code | — | HDC code | — | — |
| IMDb (Keras, 5000 words, 1–2-grams, chi2 5000) · TM WC | 89.76%; 278.88s / 11.70s (10000,8000,5, drop 0.75; ep 40) | 89.08%; 39.81s / 26.38s (10000,8000,10; ep 30) | 89.57%; 10.41s / 5.84s (10000,1280,27; ep 50) | code (10000,8000,2.0, drop 0.75, GPU, 40 ep); sparse-absorbing code | HF imdb CountVectorizer variant, no numbers | 1–4-gram variants, no numbers | — | — |
| Amazon reviews (2M docs, 5-class) · sparse TM | — | — | — | — | 63.9%; ~1664–1722 s/ep | AmazonSales (different dataset) code | — | — |
| California housing · regression TM | RMSD 0.61; 13.73 s / 30 ep | RMSD 0.61; 1.02 s / 30 ep | RMSD 0.489; 11.97 s / 100 ep (16000 C, 25-bit) | code (same as pyTM) | notebook, no metric | — | — | — |
| Model size / memory | — | — | — | — | 1 MiB vs 18.6 GiB claim (paper) | 313,600 → 37,558 params | sparse 117.6x compression (synthetic) | — |

\* green_tsetlin's 99.19% is a "best_eval_score" from 2-fold training on an 80/20 random split of the 70k images, and fold 2 reports its
best at epoch 0. It is not comparable to the standard 60k/10k test-set numbers.

### Experiments shipped by ≥2 independent libraries (good candidates for the comparison)

1. **Noisy XOR, 12-bit txt files, flat TM (10 clauses, T=15, s=3.9, 200 ep)**: the three cair libraries. tmu has a different XOR set.
2. **MNIST, thr75, flat TM.** The weighted variant is the one most shared: pyTM, Parallel, CUDA, tmu code, GT and FPTM, plus GraphTM in
   vanilla mode.
3. **MNIST 5-epoch training-time sweep over clauses (10–5000)**: the only existing cross-library timing protocol. Script:
   green_tsetlin/performance_tests/tmu_mnist_dense.py. Results: arXiv:2405.04212, Table 1.
4. **MNIST ≥75, conv TM, 10x10 patch, weighted**: pyTM, Parallel, CUDA, tmu (code) and GraphTM (code).
5. **Fashion-MNIST, adaptive Gaussian threshold (11, 2), conv TM 10x10 weighted**: pyTM, Parallel and CUDA, plus tmu experimental code.
6. **IMDb, Keras 5000 words, 1–2-grams min-freq 10, chi2 k=5000, TM 10000 clauses weighted**: pyTM, Parallel, CUDA and tmu.
7. **California housing regression, 10-bit binarizer, (1000, 5000, 2.75, WC), 30 epochs, RMSD**: pyTM, Parallel and tmu. CUDA uses
   different settings.
8. **Breast cancer (10-bit binarizer, 800/40/5.0, 25 ep, 100 splits)**: pyTM and Parallel.
9. **2D Noisy XOR CTM**: pyTM and Parallel.

CIFAR-10 has code in tmu, Tsetlin.jl and GraphTM, but the only reported number is the "69%+" code comment in Tsetlin.jl, so
CIFAR-10 has no established reported baseline in any library repo.
