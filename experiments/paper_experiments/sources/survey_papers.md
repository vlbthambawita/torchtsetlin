# Benchmark experiments reported by TM library and algorithm papers

Source PDFs are in `/work/vajira/DL2026/PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/literature/papers/`. Text was extracted with PyMuPDF, and the key tables (green-tsetlin Tables I and II, ICML Tables 3 to 5 and Figs. 3 to 9, CoTM Table 5) were also checked against page renders. "Not stated" means I searched the paper text and found nothing.

---

## Bottom line for an MLOSS comparison (only experiment types that already exist)

| # | Experiment type | Who already reports it (library / paper) | Exact protocol to copy |
|---|---|---|---|
| E1 | **Wall-clock time to train N epochs on MNIST vs #clauses, several libraries on the same machine** | **green-tsetlin (Table I)**: GT, TMU CPU/GPU, pyTsetlinMachine, pyTsetlinMachineParallel, PyCoalescedTsetlinMachineCUDA | 5 epochs, test set evaluated after each epoch (included in the time), clauses ∈ {10, 100, 1000, 2000, 5000}, seconds, i9-12900K + RTX 3090. Accuracy is omitted ("roughly the same"). T, s and Booleanization are not stated. |
| E2 | **Training time vs #clauses, CPU vs GPU (scaling curve)** | ICML 2021 / PyTsetlinMachineCUDA (Figs. 4, 7, 9); clause indexing (Figs. 3, 5, 7); CTM (Table 3); CoTM (Tables 2 to 4) | x-axis = clauses (10 to 1280 Bike Sharing; 1000 to 15000 BBC Sports; 20 to 2560 SemEval; 1000 to 20000 MNIST/F-MNIST/IMDb), y = time [s] |
| E3 | **Per-epoch training and inference time per epoch (MNIST family, conv)** | CTM (Table 3), CoTM (Tables 2 to 4) | MNIST / F-MNIST / K-MNIST, adaptive Gaussian thresholding (window 11, C = 2), 1 bit/pixel, clauses/class 50 to 8000 |
| E4 | **Inference (test) time speed-up** | Clause indexing (Tables 1 to 3, Test columns; Figs. 4, 6, 8) | MNIST/F-MNIST with 1 to 4 threshold bits per pixel, IMDb with 5k to 20k features |
| E5 | **Speed-up ratio CUDA vs CPU together with accuracy ± std over 5 runs** | ICML 2021 (Table 3, Table 5) | 6 classification datasets plus 2 regression datasets, 5 independent runs |
| E6 | **Model size / memory** | CTM (Table 5, kB), CoTM (Tables 2 to 4, kB), Sparse TM (Table II, MiB), green-tsetlin text (1 MiB vs 18.6 GiB) | |
| E7 | **Accuracy on MNIST / F-MNIST / K-MNIST (flat and conv), IMDb, Noisy XOR, 2D Noisy XOR** | nearly every paper below | see the per-paper sections |
| E8 | **Accuracy vs literal budget** | Clause-size paper (Tables 3 and 4) | budgets 1, 2, 4, ..., 64, all |
| E9 | **Per-epoch training-time curve over epochs** | ICML (Figs. 6, 8), Sparse TM (Figs. 5, 8) | |

No paper reports **energy** (measured in Joules) for a software library. Clause-size has only an analytical hardware argument, and Drop Clause only says "inference time ... per 1000 images". No paper reports **peak RAM / GPU memory measured during training** for a software library: Sparse TM reports analytic TM-state memory, CTM/CoTM report model size. **Throughput in predictions/s** appears only in the Tsetlin.jl (FPTM) record.

---

## 1. green-tsetlin: Glimsdal, Østby, Brambo, Vinje, arXiv:2405.04212 (2024) [MOST IMPORTANT]

A 3-page library paper. Its only quantitative experiment is Table I.

1. **Datasets**: MNIST only. Train/test split is not stated, though "the test set is evaluated after each epoch" implies the standard 60k/10k. **Booleanization is not stated** (no threshold or thermometer given).
2. **Model and hyperparameters**: the number of clauses is swept over {10, 100, 1000, 2000, 5000}. The paper does not say whether this is total or per class. **T, s, weighted/unweighted, literal budget, batch and number of states are not stated.** The text says not all frameworks implement Coalesced TM (pyTsetlinMachine and pyTsetlinMachineParallel do not) or Constrained Clause Size (pyTsetlinMachineParallel and PyCoalescedTsetlinMachineCUDA do not). The compared models are therefore not algorithmically identical (vanilla vs coalesced). Epochs = 5. The code snippet in Sec. IV is a toy example (n_literals = 6, n_clauses = 8, s = 1.9, threshold = 11, n_literal_budget = 3, n_epochs = 3, seed = 42, n_jobs = -1), not the benchmark configuration.
3. **ML metrics**: none reported. Caption: "ALL FRAMEWORKS HAVE ROUGHLY THE SAME ACCURACY, SO THEY ARE OMITTED FOR BREVITY." Runs/seeds are not stated (single run implied).
4. **Efficiency, Table I**: "Time in seconds needed to train 5 epochs on MNIST, where the test set is evaluated after each epoch. With an increasing number of clauses. Hardware used: i9-12900K CPU, GPU: GeForce RTX 3090."

| Framework / #Clauses | 10 | 100 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| TMU (CPU) | 29.6 | 33 | 80 | 128 | 266 |
| TMU (GPU) | 87.8 | 62 | 70 | 75 | 95 |
| pyTsetlinMachine | 4.5 | 9 | 47 | 84 | 197 |
| pyTsetlinMachineParallel (jobs=4) | 1.1 | **2** | 17 | 47 | 149 |
| PyCoalescedTsetlinMachineCUDA | 14.8 | 26 | 30 | 36 | 88 |
| GT (jobs = 1) | **0.3** | **2** | 18 | 36 | 90 |
| GT (jobs = 4) | 4.0 | 5 | **11** | **18** | **31** |

   Unit: seconds for 5 epochs including per-epoch test evaluation. Bold marks the best value as in the paper. Library versions, thread count for TMU CPU, and warm-up/JIT handling are not stated. The authors call it "not intended as a comprehensive performance test".

   Text claim (Sec. III, no table): with Sparse TM, "training on 8M documents (Amazon Reviews) with a vocabulary of 5M only requires a TM of 1 MiB as opposed to 18.6 GiB for a dense TM". This comes from the Sparse TM paper's Table II.
5. **Compared libraries**: TMU (CPU and GPU), pyTsetlinMachine, pyTsetlinMachineParallel (4 jobs), PyCoalescedTsetlinMachineCUDA, GT with 1 and 4 jobs. Table II is a feature matrix (Dense, Coalesced, Sparse, Multithreading, CUDA, Inference, AVX2, NEON, Export). From the render: pyTsetlinMachine = Dense. pyTsetlinMachineParallel = Dense + Multithreading. PyCoalescedTsetlinMachineCUDA = Dense + Coalesced + CUDA. TMU = Dense + Coalesced + CUDA. GT = everything except CUDA.

---

## 2. Abeyrathna et al., "Massively Parallel and Asynchronous TM Architecture Supporting Almost Constant-Time Scaling", ICML 2021 (basis of PyTsetlinMachineCUDA) [MOST IMPORTANT]

**Hardware and implementations (Sec. 4)**: the proposed CUDA TM runs on a **Tesla V100 GPU (grid size 208, block size 128)**. The baselines are "fast single-core TM implementations ... both with and without clause indexing (Gorji et al., 2020)" from **github.com/cair/pyTsetlinMachine**, run on an **Intel Xeon Platinum 8168 @ 2.70 GHz**. ThunderSVM vs TM (Table 4) was run on **Google Colab** "for fair comparison" (footnote 4). The GPU type on Colab is not stated. Code: github.com/cair/ICML-Massively-Parallel-and-Asynchronous-Tsetlin-Machine-Architecture. "Each experiment is repeated five times and the average accuracy and standard deviation are reported."

### 2.1 Datasets, preprocessing and hyperparameters

| Dataset | Task | Split / preprocessing | Clauses n | T | s | Epochs |
|---|---|---|---|---|---|---|
| Bike Sharing (17,389 ex., 16 features) | regression (RTM) | 80/20. Booleanization not stated | 1280 (Figs. 5 and 6); swept 10 to 1280 (Figs. 3 and 4) | 1280 | 1.5 | 100 (Figs. 5 and 6) |
| BlogFeedback (first 20,000 of 60,021; 281 features) | regression | 80/20 | not stated | not stated | not stated | not stated |
| BBC Sports (737 docs, 4,613 terms) | novelty detection: known = Cricket and Football, novel = Rugby; TM class sums fed to MLP(100, 30), ReLU, SGD | split not stated | 5,000 | 100 | 15.0 | 100 |
| 20 Newsgroups (18,828 docs) | novelty: known = comp.graphics and talk.politics.guns, novel = rec.sport.baseball | split not stated | 10,000 | 500 | 25.0 | 100 |
| SemEval-2010 Task 8 (10,717 ex.) | binary Cause-Effect vs not | unigrams + bigrams; split not stated | 40 per class (Fig. 9 sweeps 20 to 2560) | not stated | not stated | 500 (Fig. 8 x-axis) |
| IMDb (50,000) | sentiment | unigrams + bigrams; split not stated | 7,000 per class | not stated | not stated | not stated |
| WSD "Apple" (2,984: 1,784 train / 1,200 test) | 2 senses | stop-word removal, Porter stemmer, 3,000 most frequent terms | 300 | 50 | 5.0 | not stated |
| WSD "JAVA" (5,655: 3,726 / 1,929) | 2 senses | same | 300 | 50 | 5.0 | not stated |

Weighted clauses, literal budget, batch size and number of TA states are not stated. Algorithm 1 has a "batch size b" parameter, but its value is not stated.

### 2.2 ML metrics, Table 3 (Acc and F1, mean ± std over 5 runs, %)

| Dataset | Indexed Acc | Indexed F1 | Non-indexed Acc | Non-indexed F1 | CUDA Acc | CUDA F1 | Speed-up |
|---|---|---|---|---|---|---|---|
| BBC Sports | 85.08 ± 1.75 | 85.58 ± 1.69 | 87.36 ± 1.91 | 88.02 ± 1.47 | 84.64 ± 2.20 | 86.13 ± 2.24 | 38.9× |
| 20 Newsgroup | 79.37 ± 0.25 | 80.38 ± 0.92 | 82.33 ± 0.28 | 82.89 ± 0.34 | 79.00 ± 0.46 | 78.93 ± 0.44 | 49.3× |
| SemEval | 91.9 ± 0.16 | 75.29 ± 0.25 | 92.51 ± 0.03 | 77.48 ± 1.46 | 92.02 ± 0.54 | 76.27 ± 0.53 | 1.7× |
| IMDb | 88.42 ± 2.05 | 88.39 ± 2.16 | 88.2 ± 3.14 | 88.13 ± 3.44 | 89.92 ± 0.23 | 88.90 ± 0.24 | 34.6× |
| JAVA | 97.03 ± 0.02 | 96.93 ± 0.02 | 97.50 ± 0.02 | 97.40 ± 0.02 | 97.53 ± 0.02 | 97.42 ± 0.01 | 6.0× |
| Apple | 92.65 ± 0.02 | 92.20 ± 0.02 | 92.46 ± 0.01 | 91.82 ± 0.02 | 95.01 ± 0.01 | 94.68 ± 0.01 | 9.7× |

"Speed up is calculated as how many times faster is average execution time on CUDA implementation than on Indexed implementation." Whether accuracy is the last epoch or the best epoch is not stated.

**Table 5, MAE (5 runs)**: Bike Sharing: Indexed 23.5 ± 0.04, Non-indexed 22.5 ± 0.12, CUDA 23.9 ± 0.08, speed-up **156.4×**. BlogFeedback: 3.91 ± 0.00, 3.74 ± 0.03, 3.88 ± 0.02, speed-up **211.6×**. The caption says "MAE with confidence interval".

Literature comparisons quoted in the text (not re-run): Bike Sharing TM 23.9 / ANN 27.4 / SVR 22.6. BlogFeedback TM 3.88 / ANN 7.2 / SVR 6.0. Novelty (Bhattarai et al.): BBC TM 89.47 / OC-SVM 83.53 / kNN 55.54; 20NG 82.51 / 83.70 / 81. SemEval TM 92.6, RF 88.76, NB 87.57, SVM 92.3, CNN-LSTM 90.73. IMDb TM 90.5, NBOW 83.62, CNN-BiLSTM 89.54, Tree-LSTM 90.1, Self-AT-LSTM 90.34, DistilBERT 92.82. WSD Apple TM 95.1 / ThunderSVM 80.32 / FastText-1NN 96.3 / FastTextCC-1NN 97.8 / BERT 99.0. JAVA 97.53 / 93.62 / 98.7 / 99.5 / 99.6.

### 2.3 Efficiency

- **Table 4, training time, ThunderSVM vs TM (CUDA), "average of 5 experiments", Google Colab** (s):

  | | BBC | 20NG | SemEval | IMDb | JAVA | Apple | Regr1 (Bike) | Regr2 (Blog) |
  |---|---|---|---|---|---|---|---|---|
  | ThunderSVM | 0.23 | 6.95 | 7.42 | 3.89 | 1.31 | 0.39 | 6.24 | 59.94 |
  | TM (CUDA) | 0.13 | 3.61 | 4.27 | 3.36 | 0.74 | 0.28 | 0.52 | 4.92 |

  Whether these are per epoch or total is not stated.
- **Fig. 4**: execution time [s] vs #clauses (10, 20, 40, 80, 160, 320, 640, 1280) on Bike Sharing for Non-indexed, Indexed and CUDA. Non-indexed reaches about 10,000 s at 1280 clauses and doubles with each doubling of clauses. Indexed is faster than CUDA below about 300 clauses. At 1280 clauses CUDA is "roughly 3.64 times faster than the indexed version".
- **Fig. 6**: per-epoch execution time over 100 epochs, Bike Sharing (T = 1280, s = 1.5, n = 1280). Non-indexed is about 95 to 100 s/epoch. Indexed and CUDA are about 1 to 3 s/epoch (read off the plot).
- **Fig. 7**: execution time (s) vs clauses {1000, 5000, 10000, 15000} on BBC Sports. Indexed is about 2 → 8.5 → 25 → 30 s. CUDA is about 0.2 to 0.4 s (flat).
- **Fig. 8**: per-epoch execution time over 500 epochs on SemEval. CUDA is about 280 s. Non-indexed is about 320 to 350 s. Indexed is about 380 to 560 s. The figure caption says seconds, but the values look like a different unit or a total. Cite as plotted.
- **Fig. 9**: execution time (log scale, s) vs clauses 20 to 2560 on SemEval. Indexed grows to about 5×10^5. CUDA is about 5 to 7×10^3. Text: CUDA time "increases only by 40% when the number of clauses goes from 20 to 2,560".
- Headline: "up to 50 times faster learning"; "almost constant for reasonable clause amounts (20 to 7,000 clauses on a Tesla V100 GPU)". Also extended to 15,000 clauses (Fig. 7). Memory: "our scheme adds 1 bit per example per clause for tallying". No energy, inference latency or memory measurement.

**Baselines**: pyTsetlinMachine single-core, with and without clause indexing (CPU) vs the CUDA asynchronous TM (GPU). ThunderSVM (GPU) for training time. MLP for novelty scoring.

---

## 3. Gorji et al., "Increasing the Inference and Learning Speed of TMs with Clause Indexing", IEA/AIE 2020 (arXiv:2004.03188); pyTsetlinMachine indexing

1. **Datasets**:
   - MNIST, binarized to 784 bits (**M1**), plus "2, 3 and 4 threshold-based grey tone levels" giving 1568/2352/3136 bits (**M2 to M4**). Threshold values are not stated.
   - Fashion-MNIST, the same 1 to 4 bits per pixel by thresholding (**F1 to F4**).
   - IMDb (50,000 reviews) as binary feature vectors of size 5000/10000/15000/20000 (**I1 to I4**). The feature-selection method is not stated.
   - Splits are not stated (standard ones implied).
2. **Hyperparameters**: clauses ∈ {1000, 2000, 5000, 10000, 20000}. **T, s, epochs, weighted, batch: not stated.** Hardware is not stated, and neither is the implementation language. The ICML paper says the indexed version is in cair/pyTsetlinMachine.
3. **ML metrics**: none. No accuracy is reported at all.
4. **Efficiency**: speed-up ratios (indexed vs unindexed) for training and for inference ("Test").
   - Table 1, MNIST: 784 features: 1.78/2.75 (1000 clauses) to 2.82/6.31 (10000). Max train 3.58 (1568 features, 20000 clauses). Max test 8.31 (1568 features, 10000 clauses).
   - Table 2, IMDb: train speed-up 0.76 to 1.06 (indexing slows training about 10%). Test up to **15.87** (10000 features, 20000 clauses) and 15.40 (5000 features, 10000 clauses).
   - Table 3, F-MNIST: train 0.86 to 1.70. Test 0.82 to 5.05.
   - Figs. 3 to 8: **average time per epoch [s]** (log-log) vs clauses for training (Figs. 3, 5, 7) and inference (Figs. 4, 6, 8). Absolute values are only in the plots, roughly 1 to 1000 s.
   - Abstract: "up to 15 times faster classification and three times faster learning". Conclusion: training on average 3× faster on MNIST, about 50% faster on F-MNIST, slower on IMDb; inference 3 to 8× faster on MNIST/F-MNIST and 13× on IMDb.
5. **Baselines**: the same TM without indexing. No other library.

---

## 4. Sparse Tsetlin Machine: Østby, Brambo, Glimsdal, arXiv:2405.02375 (2024); the green-tsetlin authors' sparse algorithm

**Hardware**: "Intel i5-13600KF @ 3.5GHz, running single-threaded on Ubuntu 20.04 LTS" unless otherwise noted. The Amazon experiments (Table II) used **20 CPU threads**. The implementation is not named in the text, but the green-tsetlin paper says GT is the only SparseTM implementation.

1. **Datasets and per-dataset hyperparameters** (N = clauses, a = active-literal size, t = lower-state threshold, p = max clause size):
   - CR: N = 6500, a = 200, t = 30, p = 115, dynamic AL
   - SUBJ: N = 1000, a = 80, t = 40, p = 75, dynamic
   - MPQA: N = 1000, a = 123, t = 43, p = 100, dynamic
   - SST-2: N = 3500, a = 130, t = 40, p = 75, dynamic
   - MR: N = 1500, a = 130, t = 80, p = 150, dynamic
   - PC (pros/cons): N = 3500, a = 100, t = 60, p = 130, dynamic
   - TUNADROMD (tabular malware): N = 3000, a = 100, t = 50, p = 50, static AL
   - Amazon Review Data "All" (5 labels): random subsets
   - **T and s are not stated for any dataset** even though they are listed as parameters. Splits, the bag-of-words vectorizer and vocabulary size for Table I are not stated. Focused Negative Sampling and Boosting True Positives are used.
2. **ML metrics, Table I** ("sparsity in %, the best reproducible accuracy in %, and average accuracy in % from the last 25 epochs"); runs not stated:

   | Dataset | Sparsity % | Best | Avg (last 25 ep.) |
   |---|---|---|---|
   | CR | 0.409 | 83.51 | 81.49 |
   | SUBJ | 0.189 | 90.70 | 89.40 |
   | MPQA | 0.112 | 75.80 | 74.25 |
   | SST-2 | 0.226 | 80.72 | 78.93 |
   | MR | 0.174 | 79.08 | 77.18 |
   | PC | 0.093 | 91.13 | 90.41 |
   | TUNADROMD | 6.029 | 99.66 | 99.31 |
3. **Efficiency**:
   - Fig. 5: STM training time (s) per epoch over 100 epochs on SUBJ for vocabulary V̂ = 2500 to 10000 (about 4 to 12 s/epoch). Training time grows 15.79% at epoch 1 and 1.47% at epoch 100 from V̂ = 2500 to 10000. Fig. 6 shows test accuracy per epoch.
   - Fig. 7: **cumulative training time over 100 epochs** vs V̂ (2,500 to 10,000), STM vs Contracting TM (about 600 to 1,200 s). STM +8.22%, Contracting TM +99.47%. Contracting TM settings: 1000 clauses, absorbing state 50, literal sampling 0.6.
   - Fig. 8: training time per epoch (s) over 100 epochs on SUBJ, V̂ = 5000, for **STM vs TM vs Contracting TM** (0 to about 80 s). Which implementation the dense "TM" uses is not stated.
   - **Table II, Amazon Reviews, 10 epochs, 20 CPU threads**:

     | Train docs | Test docs | Vocab | Data size (uint8 dense) | Dense TM memory | Sparse STM memory | Epoch time (s) | Test acc % |
     |---|---|---|---|---|---|---|---|
     | 80,000 | 20,000 | 250,000 | 23.3 GiB | 953.67 MiB | 0.544 MiB | 102.6 | 60.16 |
     | 800,000 | 200,000 | 1,000,000 | 931 GiB | 3814.6 MiB | 0.544 MiB | 1007.4 | 61.45 |
     | 2,000,000 | 500,000 | 2,500,000 | 4.55 TiB | 9536.74 MiB | 0.544 MiB | 2392.9 | 64.19 |
     | 4,000,000 | 1,000,000 | 2,500,000 | 11.4 TiB | 9536.74 MiB | 0.544 MiB | 4133.7 | 66.27 |
     | 8,000,000 | 2,000,000 | 5,000,000 | 45.5 TiB | 18.62 GiB | 0.544 MiB | 9916.2 | 67.31 |

     Memory is the analytic TM-state size, not a measured process RSS. Accuracy is the "best reproducible". "Traditional TM training without batching".
4. **Baselines**: Contracting TM (Bhattarai et al. 2023) and a dense "TM". No external library is named.

---

## 5. Granmo, "The Tsetlin Machine", arXiv:1804.01508 (2018); the original paper

All experiments use the Multi-Class TM. Boosting of True Positive Feedback is used for Iris, Digits and MNIST. Results are reported **after the final epoch** as mean with 95% CI, 5/95 percentiles, min and max over many replications.

| Dataset | Preprocessing / split | Clauses | s | T | TA states | Epochs | Runs | TM test acc (mean) |
|---|---|---|---|---|---|---|---|---|
| Binary Iris (150) | 4 floats → 16 bits (4 bits/feature); 80/20, 1000 random partitions | 300 | 3.0 | 10 | 100 | 500 | 1000 | **95.0 ± 0.2** (Table 6); train 96.6 ± 0.05 (Table 7) |
| Binary Digits (UCI pen-based, 10,992) | 64 features → 192 bits (3 bits/pixel); 80/20 | 1000 | 3.0 | 10 | 1000 | 300 | 100 | **95.7 ± 0.2** (Table 8); train 100.0 ± 0.01 (Table 9) |
| Axis & Allies (10,000 positions, new dataset) | 25% train / 75% test, 100 partitions; output bit 5 of 20 | 10,000 | 40.0 | 10 | 1000 | 200 | 100 | **87.7 ± 0.0** (Table 11); train 96.2 ± 0.1 (Table 12) |
| Noisy XOR with non-informative features (10,000 ex., 12 bits: 2 XOR + 10 random; 40% of training outputs flipped, test clean) | 50/50 | 20 | 3.9 (from s-sweep, Fig. 7) | 15 | 100 | 200 | 100 | **99.3 ± 0.3** (Table 15) |
| MNIST (60k/10k) | pixel > 0.3 → 1 (i.e. about 77/255) | 40,000 (4000/class) | 10.0 | 50 (text; the Fig. 10 legend says Threshold = 100 for 40,000 clauses) | 256 | 400 | 50 | **98.2 ± 0.0** (Table 16); 5%ile 98.1, 95%ile 98.3 |

- Fig. 9: Noisy XOR accuracy vs training-set size (1,000 to 20,000).
- Fig. 10: MNIST mean test and train accuracy per epoch for 10,000 clauses (T = 25), 20,000 (T = 50) and 40,000 (T = 100), all with s = 10.
- **Baselines** (same random streams, light grid search): MLP, Naive Bayes, Logistic Regression, SVM, and Random Forest (Axis & Allies only). For MNIST, numbers are quoted from LeCun's page (2-layer NN 98.6, kNN 97.2, ...) plus LR/DT/NB trained on the binarized data.
- **Efficiency**: one sentence only (Sec. 4, bit-based representation): on MNIST, "memory usage is approximately ten times smaller, learning speed 3.5 times faster, and classification speed 8 times faster" vs a 32-bit-integer-per-TA implementation. No hardware or absolute times.
- Weighted: no. Literal budget: no.

---

## 6. Granmo et al., "The Convolutional Tsetlin Machine", arXiv:1905.09688 (2019)

1. **Datasets**:
   - 2D Noisy XOR: 4×4 binary images, 2,500 train / 10,000 test, 2×2 informative patch in the upper-right corner, 40% training labels flipped.
   - MNIST, K-MNIST, F-MNIST: 60k/10k, **binarized with adaptive Gaussian thresholding, window 11, threshold value 2**, 1 bit per pixel.
2. **Table 1, CTM configurations**:

   | | Search range | 2D Noisy XOR | MNIST | K-MNIST | F-MNIST |
   |---|---|---|---|---|---|
   | Clauses per class | 1 to 8000 | 40 | 8000 | 8000 | 8000 |
   | T | 1 to 20000 | 60 | 10000 | 10000 | 10000 |
   | s | 1.0 to 20.0 | 3.9 | 5.0 | 10.0 | 10.0 |
   | W (patch) | 2 to 20 | 2 | 10 | 10 | 10 |
   | Z (bits) | | 1 | 1 | 1 | 1 |
   | Clause weighting | Yes/No | No | Yes | Yes | Yes |

   Patches are augmented with thresholded (thermometer-style) position coordinates. Epochs: 250. Hyperparameters were set by "a light manual binary search".
3. **ML metrics, Table 4** ("single-run averages, obtained from the last 100 epochs of 250"; mean ± 95% CI, 95%ile, peak):
   - CTM mean: 2D N-XOR 99.99 ± 0.0, MNIST **99.33 ± 0.0**, K-MNIST 96.08 ± 0.01, F-MNIST 91.18 ± 0.01.
   - CTM 95%ile: 100.0 / 99.38 / 96.25 / 91.39.
   - CTM peak: 100.0 / 99.40 / 96.31 / 91.50.
   - Flat TM: 99.12 / 98.57 / 92.03 / 90.09.
   - Baselines (mostly quoted from the literature): 4-NN, SVM, RF, GBC, Simple CNN, BinaryConnect, FPGA BNN, Logistic Circuits (binary/real), PreActResNet-18, ResNet18+VGG.
   - Fig. 5: single-run test accuracy per epoch. MNIST passes 99% at epoch 8 and peaks at 99.4% at epoch 234. K-MNIST peaks at 96.31% at epoch 169. F-MNIST passes 90% at epoch 9 and peaks at 91.5% at epoch 170.
4. **Efficiency**:
   - **Table 3**: "CTM mean test accuracy and execution time per epoch" (s) on an **NVIDIA DGX-2**. The GPU count and implementation are not stated in the text; the catalog record lists pyTsetlinMachineParallel as the code.

     | Clauses/class | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
     |---|---|---|---|---|---|---|
     | MNIST acc / time | 98.82 / 15.6 | 98.98 / 15.8 | 99.14 / 18.9 | 99.22 / 20.4 | 99.28 / 32.4 | 99.33 / 39.0 |
     | K-MNIST | 92.75 / 15.5 | 93.86 / 17.2 | 94.89 / 19.7 | 95.40 / 21.1 | 95.85 / 33.5 | 96.08 / 40.5 |
     | F-MNIST | 88.25 / 16.9 | 88.79 / 17.2 | 89.42 / 17.8 | 89.89 / 25.0 | 90.65 / 27.8 | 91.18 / 52.2 |
   - **Table 5, model size (kB, "assuming 32-bit real-valued weights")**: TM 4000 clauses/class 7,657. CTM 250/class 127. CTM 8000/class 2,109. Logistic Circuit binary 1,072 (MNIST) / 2,456 (F-MNIST). LC real 728 / 1,768. 3-layer CNN 8,784. ResNet 19,352.
5. **Compared to**: classical ML and DL numbers (largely quoted), plus the flat TM.

---

## 7. Glimsdal & Granmo, "Coalesced Multi-Output TMs with Clause Sharing", arXiv:2108.07594 (2021) (PyCoalescedTsetlinMachineCUDA / TMU CoTM)

1. **Datasets**:
   - 2D Noisy XOR (as in the CTM paper).
   - IMDb (50,000).
   - CIFAR-10 (60,000 32×32). Its Booleanization is not stated.
   - MNIST, K-MNIST, F-MNIST (60k/10k), adaptive Gaussian thresholding (window 11, threshold 2), 1 bit per pixel.
   - "All experiments are repeated 10 times and we report average results for the last 25 epochs, unless otherwise noted."
2. **Hyperparameters**: convolutional CoTM vs weighted convolutional TM. Clauses per class ∈ {50, 100, 250, 500, 1K, 2K, 4K, 8K}.
   - MNIST (Table 2): T = 625, 1250, 3125, 6.25K, 12.5K, 25K, 5K, 10K; s = 10 (5.0 for 4K and 8K).
   - F-MNIST (Table 3): T = 625 to 10K (as printed: 625, 1250, 3125, 6.25K, 12.5K, 2.5K, 5K, 10K); s = 15.
   - K-MNIST (Table 4): T = 625, 1250, 3125, 6.25K, 1.25K, 2.5K, 5K, 10K; s = 10.
   - The T sequences look partly mistyped in the paper; cite as printed.
   - **Patch size W, epochs, IMDb and CIFAR-10 hyperparameters: not stated.** Table 5 uses "CoTM with 64 000 clauses".
3. **ML metrics**:
   - Tables 2 to 4, accuracy % (Weighted / Coalesced):
     - MNIST 50 clauses: 97.86 / 98.33. 8K clauses: 99.33 / 99.29.
     - F-MNIST 50 clauses: 82.33 / 86.79. 8K clauses: 91.18 / 91.18.
     - K-MNIST 50 clauses: 71.99 / 89.66. 8K clauses: 96.08 / 96.33.
   - Table 5: CoTM mean 2D N-XOR 99.99 ± 0.0, MNIST 99.31 ± 0.0, K-MNIST 96.65 ± 0.01, F-MNIST 91.83 ± 0.01. 95%ile: 100.0 / 99.34 / 96.70 / 91.93. Peak: 100.0 / 99.36 / 96.71 / 92.0. Same baseline list as the CTM paper.
   - Table 6: IMDb accuracy drop vs fraction of positive training examples removed (0.0 to 0.9). ΔTM up to 30.5; ΔCoTM up to 2.8.
   - Table 7: class-wise **F1** on imbalanced CIFAR-10 (class r keeps 0.5^r of its data). "mean F1 over epochs 90 to 100, averaged over 10 independent trials".
   - Fig. 9: CoTM passes 99.3% on MNIST at epoch 19 vs epoch 57 for the weighted TM ("3 times faster").
4. **Efficiency** (NVIDIA Tesla V100, "calculating the clause outputs and updates in parallel"):
   - Tables 2 to 4 give **Training Time (s) and Testing Time (s) per epoch, and Model Size (kB)**.
   - MNIST: train 21.6 s (50 clauses) to 86.7 s (8K). Test 1.7 to 6.5 s. Size 22 to 3,516 kB.
   - F-MNIST: train 25.1 to 81.9 s. Test 1.6 to 6.4 s.
   - K-MNIST: train 24.2 to 85.1 s. Test 1.7 to 6.4 s.
   - These times are for CoTM only. The weighted-TM row has accuracy only.
5. **Baselines**: weighted convolutional TM, flat TM, and literature ML/DL numbers.

---

## 8. Phoulady et al., "The Weighted Tsetlin Machine", StarAI 2020 (arXiv:1911.12607)

1. **Datasets**:
   - MNIST 60k/10k: pixel < 77 (= ⌈0.3·255⌉) → 0, else 1. The paper writes "768 bits", which is a typo for 784.
   - IMDb 25k/25k: unigrams + bigrams, top 5,000 by χ² → 5,000 binary features.
   - Connect-4 (UCI, 67,557 positions, 3 classes): 90/10 split, 6×7×2 = 84 bits.
2. **Hyperparameters**:
   - MNIST: TM 2,000 clauses/class matched by WTM 500 clauses; also both at 4,000/class; 300 epochs.
   - IMDb: WTM 3,200 vs TM 10,000 clause-classes after 30 epochs; then 25,000 clauses/class.
   - Connect-4: WTM 200 vs TM 10,000 clauses (last 200 of 1,000 epochs); then 25,000/class.
   - **T, s, weight learning rate γ and hardware: not stated.**
3. **ML metrics** (single run; statistics over the last 50 of 300 epochs for MNIST):
   - Table 3, MNIST at 4,000/class: TM max 98.24, min 97.97, mean 98.11 ± 0.07. WTM 98.63 / 98.39 / **98.50 ± 0.05**.
   - Table 6, IMDb at 25,000/class: TM 89.86 / 89.73 / 89.80 ± 0.05. WTM 90.37 / 90.00 / **90.25 ± 0.16**.
   - Table 7, Connect-4 at 25,000/class: TM 82.93 / 81.98 / 82.49 ± 0.34. WTM 87.91 / 86.75 / **87.32 ± 0.20**.
   - Table 5: MNIST vs LeCun-page baselines.
4. **Efficiency** (text only, hardware not stated):
   - MNIST with 2,000 clauses/class, first epoch: recognition of 60k images takes 23.5 s, Bernoulli feedback generation takes 61.2 s. Binomial sampling speeds up feedback generation **7×**.
   - The WTM with 500 clauses ran about **10× faster** than the TM with 2,000 clauses over 300 epochs.
   - MNIST 4,000/class: **1.7 min (WTM) vs 2 min (TM) per epoch**.
   - IMDb 25,000/class: **14 min (WTM) vs 16 min (TM) per epoch**.
   - Connect-4 25,000/class: **76 s vs 105 s per epoch**.
   - Memory "reduced 4 times" (500 vs 2,000 clauses).
5. **Baselines**: vanilla TM (same code), literature numbers.

---

## 9. Abeyrathna et al., "Building Concise Logical Patterns by Constraining TM Clause Size", IJCAI 2023 (arXiv:2301.08190); literal budget

**Implementation and hardware**: "a CUDA implementation of CSC-TM ... runs on Intel Xeon Platinum 8168 CPU at 2.70 GHz and a Nvidia DGX-2 with Tesla V100 GPU".

1. **Datasets**:
   - NLP: BBC Sports, R8, TREC-6, SemEval-2010, IMDb.
   - Images: MNIST (flat and conv), CIFAR-2 (vehicles vs animals from CIFAR-10).
   - California Housing (5 bins, classification).
   - Energy Performance (regression, MAE).
   - Hex (board-game winner prediction).
   - LCTM self-supervised clustering on an MNIST 0/1 subset.
   - **Booleanization and splits are not stated.**
2. **Hyperparameters**:
   - BBC/TREC/R8: 8000 clauses, T = 100, s = 10.0.
   - MNIST: 8000 clauses per class, T = 10000, s = 5.0.
   - CIFAR-2: 8000 clauses, T = 6000, s = 10.0.
   - The other datasets are not stated (Hex follows Giri et al. 2022).
   - Literal budget ∈ {1, 2, 4, 8, 16, 32, 64, all}. The conv patch size for MNIST w/conv is not stated.
3. **ML metrics**: **Table 3** reports "worst maximum accuracy across 5 independent runs" (i.e., the minimum over 5 runs of the best epoch), with average literals per clause in brackets; MAE for Energy.
   - MNIST: 92.09 (budget 1) → 98.42 (budget 16) → 98.32 (all).
   - MNIST w/conv: 40.94 → 99.33 (budget 16) → 99.28.
   - CIFAR-2: 69.82 → 94.24 (budget 64) → 94.18.
   - IMDb: 81.58 → 87.73 (budget 64) → 84.23.
   - BBC: 99.1 at budget 8 vs 94.14 for all.
   - TREC: 93.2 at budget 32 vs 85.8.
   - R8: 95.8 (all).
   - SemEval: 93.65 (budget 1).
   - California Housing: 65.79 (all).
   - Energy MAE: 5.65 → 0.59.
   - Hex: 82.17 (budget 64) vs 81.43.
   - **Table 4**: MNIST with 250 to 8000 clauses × budgets (e.g. 250 clauses: 88.20 to 97.06).
   - **Table 5**: LCTM interpretability metrics ± std over 5 runs.
4. **Efficiency**:
   - Only Table 5, "Speed up" for LCTM training vs all literals: 400 literals → 1.1×, 800 → 1.3×, 1,200 → 1.3×, 1,568 (all) → 1×.
   - Sec. 5 gives an analytic hardware energy argument (P = 0.5·C·V²·f·α, savings ≈ b / l_ave). **No measured time or energy.**
5. **Baselines**: vanilla TM (budget = all), Logistic Circuits (quoted).

---

## 10. Sharma et al., "Drop Clause: Enhancing Performance, Interpretability and Robustness of the TM", AAAI 2023 (arXiv:2105.14506)

**Implementation**: "written in PyCUDA". Training ran on **16 × NVIDIA Tesla V100** for images and **1 × RTX 3070 8 GB** for NLP. NLP inference was timed on a **GTX 1080**, and CIFAR-10 inference on one V100.

1. **Datasets and hyperparameters**:
   - NLP uses the weighted TM. Image classification uses the CTM.
   - IMDb (25k/25k): 10,000 clauses, T = 8,000, s = 2.0.
   - R8 (5,485 / 2,189) and R52 (6,532 / 2,568): 3,000 clauses, T = 2,000, s = 7.0.
   - MR (10,662, split as in Tang et al. 2015): 5,000 clauses, T = 4,000, s = 6.0.
   - TREC-6 (5,500 / 500): 5,000 clauses, T = 4,000, s = 2.0.
   - MNIST and F-MNIST (60k/10k): adaptive Gaussian thresholding (as in the CTM paper), 1 bit per pixel per channel; 8,000 clauses, T = 6,400, s = 5.0.
   - CIFAR-10 and CIFAR-100 (50k/10k): 1 bit per channel; 60,000 clauses, T = 48,000, s = 10.0.
   - Drop probability p ∈ {0, 0.1, 0.25, 0.5, 0.75}. **Patch size, epochs and batch: not stated.**
2. **ML metrics**:
   - Table 1 (NLP accuracy by p): IMDb 87.2 → 91.27 (p = 0.75). R8 96.16 → 98.94. R52 89.14 → 92.75. MR 75.14 → 78.67. TREC 88.05 → 90.5 (p = 0.5).
   - Table 2 (images by p): MNIST 99.3 → 99.45 (p = 0.25). F-MNIST 91.5 → 92.5. CIFAR-10 69.3 → 75.1 (p = 0.5). CIFAR-100 35.5 → 45.2.
   - Peak values with ±: MNIST 99.45 (± 0.25), F-MNIST 92.5 (± 0.25), CIFAR-10 75.1 (± 0.4), CIFAR-100 45.2 (± 0.2), "averaged over 100 runs".
   - Tables 3 to 7: vs SVM/RF/kNN/XGB (sklearn defaults), CNN, BiLSTM, BERT, S2GC, EEV-BNN, Hybrid-ViN, AlexNet-DFA, Maxout.
   - Tables 8 and 9: robustness on MR with synonym perturbation, and MNIST-C.
3. **Efficiency** (text only):
   - CIFAR-10 training time per epoch drops from **61.83 s to 32.58 s** with p = 0.5 (16 V100).
   - **Inference 1.5 s per 1000 images** on CIFAR-10 (1 V100).
   - NLP inference "**less than 2.3 s per 1000 samples** on average on an NVIDIA GTX 1080".
   - "substantial reduction in training time proportional to the DC ratio".
4. **Baselines**: as listed above. No other TM library.

---

## 11. Tsetlin.jl / Fuzzy-Pattern TM: Hnilov, arXiv:2508.08350 (from catalog record `2025_Fuzzy_Pattern_Tsetlin_Machine_arXiv2508.08350.json`)

**Implementation**: single-author Julia, CPU. Inference throughput was measured on an **AMD Ryzen 9 7950X3D** using all cores. Code: github.com/BooBSD/FuzzyPatternTM.

1. **Datasets**:
   - IMDb: Boolean n-grams. 12,800 features (40k words, up to 4-grams) in Tables 1 and 2; 70,000 features (65,535 words) in Table 3. Split not stated.
   - Fashion-MNIST: fixed 3×3/5×5/7×7 edge kernels, feature maps thresholded on the pixel histogram. The "big" model also uses augmentation.
   - Noisy Amazon Sales (graph/recommendation): label-noise ratios 0.005 to 0.2.
2. **Hyperparameters**:
   - CoTM and Weighted CoTM: about 50 clauses/class, T = 80, s = 2, 1000 epochs.
   - Binary FPTM: 1 clause/class, T = 18, S = 1000, L = LF = 64, 1000 epochs.
   - Multi-class FPTM: 200 clauses/class, L = 100, T = 32, S = 2000, LF = 10, 200 epochs.
   - FPTM on F-MNIST: tiny 2 clauses/class; small 20; big 8000.
   - Amazon: 2000 clauses/class, T = 100, S = 1000, L = 50, LF = 10.
3. **ML metrics**:
   - Table 2 reports **median peak test accuracy** (20 FPTM runs / 4 CoTM runs): CoTM 88.47, Weighted CoTM 90.18, Binary FPTM 90.15.
   - Multi-class FPTM: 90.68 after 200 epochs; 91.8 is a test-curve peak.
   - F-MNIST (Table 4): 92.18 / 93.19 / 94.68.
   - Amazon (Table 6): FPTM 99.64 to 85.22 vs GraphTM, GCN and standard TM.
4. **Efficiency** (Table 2, IMDb, 1000 epochs):

   | Model | Time |
   |---|---|
   | CoTM, 1 thread | **27,651 s** |
   | Weighted CoTM, 1 thread | **14,362 s** |
   | Binary FPTM, 1 thread | **399 s** |
   | Binary FPTM, 32 threads | **45.5 s** (316× vs single-thread Weighted CoTM) |

   - TA state memory is 50 KB (12,800 × 2 × 2 bytes; CoTM at least 50× more).
   - **Inference throughput: 34.5 M predictions/s (51.4 GB/s)** in batch on all cores of the Ryzen 9 7950X3D.
   - The record does not say which CoTM implementation was used for the baseline timing.
5. **Baselines**: CoTM, Weighted CoTM, Composite CTM (quoted 93.0 on F-MNIST), GraphTM, GCN.

---

## 12. Other catalog records with software training/inference timing (`benchmarks/literature/papers/*.json`)

- **Contracting TM (arXiv:2310.11481)**:
  - Table I: training time (s) "from last 25 epochs" for TM vs Contracting TM (absorbing state 25 to 125) on TREC, MPQA, SUBJ, CR, R8 and R52. Examples: SUBJ TM 308.16 s vs 2.76 s; MPQA 186.07 vs 5.07.
  - Table II: R8 × literal-sampling.
  - The record notes that per-epoch vs total is unspecified, and that N, T, s and hardware are not reported.
- **Pre-Sorted TM (arXiv:2403.09680)**:
  - Tables 2 and 3: training time / inference time (s), vanilla TM (CAIR C code, 4000 clauses/class, 100 epochs, **single-threaded Apple M1**) vs Pre-sorted TM.
  - MNIST: train 3472 vs 15.58 s, inference 31.4 vs 0.32 s.
  - F-MNIST: 4484 vs 11.72 s and 36.6 vs 0.43 s.
  - K-MNIST: 4336 vs 30.89 s and 34.3 vs 0.43 s.
  - EMNIST: 2424 vs 15.58 s and 20.0 vs 0.32 s.
- **Knowledge Distillation in TMs (arXiv:2504.01798)**, pyTsetlinMachineParallel fork:
  - Table 4.7: average epoch testing time (unit unstated) for MNIST, K-MNIST, EMNIST and IMDb.
  - Example: IMDb teacher with 8000 clauses (T = 6000, s = 7) takes 17.88.
- **Verifying Properties of TMs (arXiv:2303.14464)**:
  - Table I: MNIST training time 1880.45 s (500 clauses, T = 25, s = 10, 400 iterations), 3447.27 s (1000 clauses), 6915.48 s (2000 clauses, T = 50).
- **Graph TM (arXiv:2507.14874)**:
  - Amazon Sales total training time: GraphTM 133.75 s (CUDA) vs standard TM 1068.99 s.
  - Table 8, NCBI viral genome, 10 epochs: GraphTM 62.47 / 84.37 s vs LSTM 26.02 s, GCN 226.36 s, and other baselines.
- **Uncertainty Quantification in the TM (arXiv:2507.04175)**: CIFAR-10 CTM (2000 clauses/class, T = 20,000, s = 20, 3×3 patch, literal budget 64) takes about 5 min/epoch.
- **HAR Convolutional TM (Sensors 2026)**:
  - Training time for 1000 epochs, Intel i7-12700H: 153 s (2 bins) and 344 s (5 bins).
  - The same runs on a Cortex-A53 (Arduino UNO Q): 575 s and 1477 s.
  - Inference throughput: 106,625 ops/s on the i7 vs 7,704 ops/s on the A53.
- **IoMT IDS papers (arXiv:2604.03205, 2605.16707)**: per-sample inference latency (µs) on an Apple M4 vs DT/RF/XGBoost/LightGBM/kNN. Examples: TM 0.743 µs; Sparse TM 16.89 µs vs TM 66.24 µs.
- **FastOmniTMAE (arXiv:2605.06982)**: training time CPU vs FPGA (ZCU104), 848.7 s vs 119.9 s (7.08×).
- Hardware-accelerator records (ASIC/FPGA/in-memory: 2020 asynchronous logic, IMPACT, 65 nm accelerators, TsetlinKWS, RISC-V, flexible substrate, TsetlinWiSARD) report energy per frame, latency and power. Those are chip metrics, not software-library metrics.

---

## Cross-paper notes relevant to designing a "no new experiment types" comparison

1. **The only multi-library head-to-head in the literature is green-tsetlin Table I.** It covers MNIST, 5 epochs, time-to-train including per-epoch test evaluation, 5 clause counts, one machine, a single run, and no accuracy shown. It does not state T, s or Booleanization. Copying it exactly therefore needs a choice of these, ideally following tmu/pyTsetlinMachine examples. Two candidate conventions:
   - pyTsetlinMachine's MNIST demo: threshold 75, 2000 clauses, T = 50, s = 10.0. This is from memory of the repo README and is not in any of these PDFs, so check it before citing.
   - The original 2018 paper: > 0.3·255, T = 50, s = 10.
2. **CPU vs GPU scaling with #clauses** (ICML Figs. 4, 7, 9) and **indexed vs unindexed time per epoch vs #clauses** (Gorji Figs. 3 to 8) are established experiment types. Their datasets: MNIST/F-MNIST with 1 to 4 bit thresholds, IMDb with 5k to 20k features, Bike Sharing, BBC Sports, SemEval.
3. **Per-epoch training time, per-epoch test time and model size vs clauses/class for convolutional MNIST/F-MNIST/K-MNIST** come from CTM Table 3 and CoTM Tables 2 to 4. Both use adaptive Gaussian thresholding (window 11, C = 2) and clauses/class 50 to 8000. CTM uses a 10×10 patch; the CoTM patch size is not stated.
4. Accuracy reporting conventions differ:
   - mean ± 95% CI after the final epoch over 50 to 1000 runs (2018);
   - single-run mean of the last 100 of 250 epochs (CTM);
   - mean of the last 25 epochs over 10 runs (CoTM);
   - last 50 of 300 epochs, single run (WTM);
   - mean ± std over 5 runs (ICML);
   - worst-of-5 maximum accuracy (clause size);
   - best and last-25-epoch mean (Sparse TM);
   - average over 100 runs (Drop Clause);
   - median peak (FPTM).
5. Hardware reported:
   - i9-12900K + RTX 3090 (GT)
   - Xeon Platinum 8168 + V100 (ICML, clause size)
   - DGX-2 (CTM)
   - V100 (CoTM)
   - i5-13600KF single thread, and 20 threads for Amazon (Sparse TM)
   - 16×V100 / RTX 3070 / GTX 1080 (Drop Clause)
   - Ryzen 9 7950X3D (FPTM)
   - not stated (clause indexing, WTM, 2018)
