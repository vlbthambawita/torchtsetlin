# Plan: bit-plane Booleanization on Fashion-MNIST, flat vs 2D conv vs 3D conv TM

Status: **v4 (2026-11-13): staged; stage A = conv3 with a fixed window depth kd = 8 (all bit planes).
Stage B (the comparison) waits for the user's check of stage A (§4, §9). Smoke test passed; pricing running.**
Request: `source_documents/TODO/quesry_FMNIST_testing_plane_binary.md` (idea: `query_01.md`).
Programme folder: `experiments/fmnist_bitplanes/` (to be built). W&B: `simulamet_mlc/TM_fMNIST`.

## 1. Question

A grey pixel `v ∈ [0, 255]` is 8 bits: `v = B0 + 2·B1 + … + 128·B7`. Is this **bit-plane** encoding a
useful Booleanization for a Tsetlin machine, compared with the usual ones, and does a **3D
convolution over the plane axis** help?

- **Q1** On Fashion-MNIST, how do a flat TM and a 2D convolutional TM with the usual Booleanization
  (the paper's B8 adaptive Gaussian threshold) compare with a 3D convolutional TM fed thermometer
  planes, and with a 3D convolutional TM fed bit planes?
- **Q2** At an equal bit budget (8 Boolean planes per pixel), is bit-plane encoding better or worse
  than thermometer encoding, for each architecture?
- **Q3** Which planes do the learned clauses actually use? The low bits are expected to be close to noise.
- **Q4** What does each choice cost in time per epoch, time to best, memory and literal count?

Why Fashion-MNIST and not MNIST: 48.5 % of Fashion-MNIST pixels are grey (1–249, median of the
non-zero pixels 164), against 11.8 % on MNIST, where the grey is only anti-aliasing along stroke
edges (measured 2026-11-13). Intensity carries real information only on Fashion-MNIST.

Literature (the 70 PDFs in `source_documents/`, searched 2026-11-13): no TM paper uses 8-bit bit
planes on images. Granmo 2018 (arXiv:1804.01508 §5.2) uses a 3-bit code on 8×8 digits as a
*harder* benchmark. Granmo's TM book (Ch. 4) argues for thermometer over binary codes, because
a range needs a single pair of literals with a thermometer code. So this is new ground.

## 2. Design

### 2.1 Encodings (all computed in the harness, library read-only)

| Id | Encoding | Planes per pixel | How |
|---|---|---|---|
| **AGT** | the paper's B8: OpenCV adaptive Gaussian threshold (block 11, C = 2) | 1 | `paper_experiments/.cache/data/B8.npz`, sha256 checked against its manifest |
| **THERM8** | thermometer, 8 global uniform thresholds `t_k = 255·k/9` (`v ≥ t_k`) | 8 | `ThermometerEncoder(8, "uniform", value_range=(0, 255))` |
| **BIT8** | plain bit planes, MSB first | 8 | `BitPlaneEncoder(8)` |
| **GRAY8** | Gray-coded bit planes (`g = v ^ (v >> 1)`), MSB first | 8 | harness, then `BitPlaneEncoder(8)` |
| BIT4 / THERM4 | the top 4 bits / 4 uniform thresholds (ablation, §2.5) | 4 | as above |

- **Global thresholds.** The thresholds are the same for every pixel. Per-pixel quantiles would give each image position its own
  thresholds and break the weight sharing of the convolution.
- **"Normal Booleanization" in the 3D cell (decision D1, to confirm).** The user asked for a 3D conv with normal
  Booleanization. A single AGT plane has no depth (with D = 1, the 3D model is the 2D model bit for bit,
  `tests/test_conv3d.py`). So the 3D "normal" cell uses THERM8, the ordinal multi-level form of the usual
  thresholding. Its planes stack along depth the same way as the bit planes, so the two 3D cells differ only in the encoding.
- **Checks in the smoke test.** The bit planes must reconstruct `v` exactly (`Σ 2^k B_k == v`). Thermometer planes must be monotone
  (plane k+1 ⊆ plane k). Gray decoding must round-trip.

### 2.2 Architectures

| Id | Model | Input | Literals per clause (2·features) |
|---|---|---|---|
| **FLAT** | `TsetlinMachine` | 784 × planes, flattened | 1 568 (AGT), 12 544 (8 planes) |
| **CONV2** | `ConvTsetlinMachine`, patch 10, stride 1, position encoding | `(planes, 28, 28)`: planes as channels | 272 (AGT), 1 672 (8 planes) |
| **CONV3** | `Conv3dTsetlinMachine`, patch `(kd, 10, 10)`, stride 1, position encoding | `(1, D = planes, 28, 28)`: planes along depth | `2·(kd·100 + (8 − kd) + 36)` |

- **CONV3 with kd = 8** has a single depth position, so it sees the same literals per patch as CONV2.
  CONV3 is only different when **kd < 8**: then a clause slides along the plane axis too.
  - For thermometer planes this sliding means **contrast-shift invariance**: one pattern ("the edge where intensity crosses level k")
    can match at any level k.
  - For bit planes, a bit's meaning depends on its position, so sliding has no obvious meaning. Depth position
    encoding lets a clause pin itself to specific planes.
- **kd = 8, fixed (user, v4).** The window covers all 8 bit planes, so every clause sees whole 8-bit pixels and
  can combine bits of any significance. With D = 8 there is a single depth position: this CONV3 has exactly
  the literals of CONV2 with 8 bit-plane channels (checked: identical literals, 361 patches × 1 672).
  Stage A therefore tests **a convolutional TM on bit planes**; sliding along the bit axis (kd < 8) is the
  optional check A3b. (v3 picked kd from {1, 2, 4} in the pilot.)

### 2.3 Cells

| | AGT | THERM8 | BIT8 | GRAY8 |
|---|---|---|---|---|
| FLAT | **core** (normal TM) | control | control | control |
| CONV2 | **core** (conv TM) | control | control | control |
| CONV3 | — (equals CONV2) | — (user: skip) | **core** (3D, bit planes) | control |

v2: 3 core cells and 7 controls. The user dropped the 3D cell with the usual Booleanization (D1). The controls separate the effect of the
encoding (read along a row) from the effect of the architecture (read down a column). Without them, "3D
bit planes beat 2D AGT" could not be attributed to either. That gives **10 cells**.

### 2.4 Fixed settings (no tuning beyond the pilot)

| # | Decision | Value | Why |
|---|---|---|---|
| D2 | Base hyper-parameters | **Conv:** the paper's B8 (weighted, boost, s = 5, T = 5000 at 2 000 clauses/class). **Flat:** B4 (unweighted, boost, s = 10, T = 50 at 2 000). | Same bases as the MNIST batch × clauses grid, so results connect to it |
| D3 | Clauses per class | **C ∈ {2 000, 8 000}** | B8's own value plus a larger model. The MNIST grid: conv flat from 2k up (99.1–99.2 %), flat gains up to about 16k |
| D4 | T | `T = T_ref · C / 2000` (scaled with C) | MNIST grid: fixed T cost flat 1–2.5 pp at 16k–64k, conv nothing |
| D5 | Batch size | **B = 32**, `feedback_mode="batch"` | MNIST grid: B = 32 equals sequential exactly at 2k and 16k for both models (conv 99.20 = 99.20, flat 97.88 = 97.88). Conv is robust up to 128; flat collapses at 1024. B = 128 only if pricing (Phase 1) shows CONV3 is too slow, and then for **every** cell |
| D6 | Early stopping | As MNIST v4: **validation** split 50 000 / 10 000 (stratified, seed 0); patience **25**, `min_delta = 0` (any new best counts, a tie does not), min 30 epochs, max 500, 24 h cap; checkpointed, resume bit-identical | Request: "similar to the MNIST experiments" |
| D7 | Statistic | Test accuracy at the best-validation epoch (test is evaluated every epoch, never used for a decision) | As MNIST D4 |
| D8 | Seeds | **Seed 0 for all cells first**; seeds 1–2 for the 4 core cells + THERM8/BIT8 controls only after the user approves | Matches the current MNIST policy (seed 0 only for now) |
| D9 | Data order | Depends on the seed only, identical across cells | As MNIST D7 |
| D10 | Timing | Every run on one GPU, alone (see §6) | Q4 compares times |

### 2.5 Ablation (Phase 5, small)

- **Do the low bits matter?** BIT4 (top 4 bits only) vs BIT8, and THERM4 vs THERM8, on the best architecture from
  Phase 4, at C = 2 000 only. If BIT4 ≈ BIT8, the low bits add only cost. That answers Q3 causally, alongside
  the clause-usage statistics.

## 3. Measurements

Each per-epoch row in the JSON record (and W&B) holds the MNIST metrics: val_acc, best_val_acc,
epochs_since_improvement, test_acc, train_acc_running, train_s, val_s, test_s, examples_per_s, peak_gpu_mib,
mean_clause_len, frac_empty_clauses, vote_saturation, elapsed_s. Added for this programme:

- **plane_usage**: the fraction of included literals that fall on each plane, split into positive and negated
  literals. This is computed from `ta_state` per epoch (cheap, one reduction) and is the main Q3 evidence.
  For CONV3 it is reported per depth offset inside the window, plus the depth-position literals.
- **per-class test accuracy** at the best epoch (Fashion-MNIST's errors concentrate on shirt / T-shirt / pullover / coat).
- **n_literals**, the input size, so cost can be compared with the bit budget.
- At stop: test accuracy at the best epoch, best epoch, time to best, s/epoch, peak memory, stop reason.

## 4. Phases (staged, v3)

**Stage A: 3D convolution only (now).** Only the two CONV3 cells run: CONV3 × BIT8 (core) and
CONV3 × GRAY8. Both are 3D conv experiments; GRAY8 shows whether the code of the bit planes matters.
The flat / 2D conv cells are **not** run.

| Phase | What | Runs | Gate |
|---|---|---|---|
| 0 Build | Harness `experiments/fmnist_bitplanes/`, encodings + checks, smoke test (builds every model of both stages at full size, 1 epoch on 2 000 images; resume check on CPU + cuda:1), W&B smoke runs in group `smoke` | smoke only | none (smoke tests are allowed) |
| A1 Pricing | 300 timed updates per CONV3 configuration: 2 encodings × 2 C × kd ∈ {8, 4} (4 = the optional real-3D check) → s/epoch, memory → cost table | 8 short | user go-ahead (given) |
| A2 Pilot | Validation only, C = 2 000, seed 0, max 60 epochs, kd = 8: s ∈ {2.5, 5, 10} for both CONV3 cells. Pick s by best validation accuracy; ties go to s = 5 | 6 | user sees the choice |
| A3 Main | 2 CONV3 cells × C ∈ {2 000, 8 000}, seed 0, kd = 8, chosen s, full early stopping | 4 | **user go-ahead** |
| A3b (optional) | Real 3D check: kd = 4 (5 depth positions, the window slides along the bits) at the chosen s, C = 2 000, both encodings. Shows whether sliding along the bit axis adds anything over kd = 8 | 2 | user asks |
| A4 Report | `REPORT.md`: test accuracy at the best-validation epoch, time per epoch / to best, plane usage per depth offset, per-class accuracy, curves | — | **the user checks the results** |

How to judge stage A without running the comparison cells: there is an existing external
reference, the paper's own B8 result (2D conv TM, AGT, 2 000 clauses/class, 30 epochs):
**≈ 90.0 %** test accuracy (tt, 5 seeds, `paper_experiments/results/seeds/B8`). It was measured with
30 fixed epochs, not early stopping, so it is a reference point, not a matched control.

**Stage B: the comparison (only if the user decides so after stage A).** The 8 flat / 2D conv cells
of §2.3 (AGT, THERM8, BIT8, GRAY8), the same phases (pricing, pilot, main) with
`queue_runner.py --stage B`, then:

| Phase | What | Runs |
|---|---|---|
| B0 Sanity | CONV2 × AGT, C = 2 000, seed 0: the test accuracy at epoch 30 must fall within 0.5 pp of B8's ≈ 90.0 % | 1 |
| B1–B3 | pricing (16), pilot (8 × 3 = 24), main (8 × 2 = 16) | 56 |
| B4 Ablation | §2.5 (BIT4 / THERM4) | 4 |
| B5 Seeds | seeds 1–2 for the cells in D8 | ~24 |
| B6 Report | the full comparison table and figures; paper outputs only on request | — |

The pilot is a fairness device, not tuning for a headline number. A fixed s would favour whichever
setting happens to suit it. Every cell gets the same small search, on validation.

## 5. Hypotheses (stated before running)

- **H1** THERM8 > BIT8 for FLAT and CONV2. A range of intensities needs one pair of literals in a
  thermometer code, but a conjunction over several bits in a binary code.
- **H2** Clauses rarely include the low bit planes (B0–B3); BIT4 ≈ BIT8 (Phase 5).
- **H3** CONV3 × BIT8 vs CONV2 × BIT8: sliding along the bit axis gains little or nothing, because a bit's meaning depends on its position (the THERM8 version of H3 was dropped with CONV3 × THERM8).
- **H4** GRAY8 lies between BIT8 and THERM8: neighbouring values differ in one bit, but ranges still need several literals.
- **H5** Any 8-plane encoding beats single-plane AGT on Fashion-MNIST, at a cost of several times the s/epoch.

## 6. Compute, GPUs, W&B

- **Cost (rough, replaced by Phase 1).** Scaled from MNIST pricing (conv 2k, B = 32: about 3 s/epoch; flat 8k: about 5 s/epoch) by
  literal count and patch count:
  - CONV2 with 8 planes: ~4–6× the AGT cost.
  - CONV3 with kd = 1: ~8× more patches.
  - Stage A (CONV3 only): set by pricing (A1); CONV3 has 5–8× more patches than CONV2, so it is the most expensive architecture.
  - Stages A + B together, seed 0: roughly **30–80 GPU-h**; pilots 10–20 GPU-h; seeds 1–2 about 2× the main phase.
  - Memory: the flat 8-plane model at 8k has 80 000 × 25 088 automata = 2·10⁹ (2 GB uint8 state + 4 GB fp16 include cache). That's tight on the 3080's 10 GB; the smoke test checks it.
- **GPU (decision D11, to confirm).** The RTX 3090 runs the MNIST seed-0 grid until about 2026-11-17. The RTX 3080 is
  reserved for the user's own tests. Options:
  - (a) start after the MNIST grid on the 3090;
  - (b) use the 3080, if this is the test it was kept free for.
  
  All timed runs of this programme (Phases 1 and 4) must run on one GPU type so that Q4's times are comparable.
- **W&B.** Entity `simulamet_mlc`, project **`TM_fMNIST`**, on the configured host `api.forge.coreweave.com`.
  The project exists (checked 2026-11-13).
  - Groups: `smoke`, `pricing`, `pilot`, `main`, `ablation`, `seeds`.
  - Run id `fm-<tag>`, `resume="allow"`, tag `<arch>_<enc>_c<C>_s<seed>[_kd<k>][_sX]`.
  - W&B is a mirror. The JSON records are the source, and a tracking failure never stops a run (same `tracking.py` as MNIST).

## 7. Rules

- `src/torchtsetlin/**` is read-only. Every encoding needed exists (`ThermometerEncoder`, `BitPlaneEncoder`) or is a
  few lines in the harness (Gray code). Library defects go to `experiments/fmnist_bitplanes/LIBRARY_GAPS.md`.
- Data: the raw Fashion-MNIST files cached by `paper_experiments` (`.cache/data/raw/fashion-*.gz`) and its B8.npz,
  sha256 recorded in each record.
- Every reported number comes from a `results/<phase>/<tag>.json` record with argv, seed, git sha and env. Nothing is hand-typed.
- GPU runs beyond the smoke test need the user's go-ahead at the gates above.

## 8. Risks

| Risk | Mitigation |
|---|---|
| CONV3 at kd = 1 has 2 888 patches per image, so it may be slow or memory-bound | Phase 1 measures it; the pilot may drop kd = 1; `patches_per_chunk` bounds memory |
| The library's default chunk budget splits every CONV3 model into patch blocks and takes it off the fused CUDA path; at 8k and kd = 4, into blocks of one patch (found in the smoke test, `LIBRARY_GAPS.md` FB-1) | The harness passes `max_chunk_elements=2**30` to CONV3: one block for every model, ~150 ms per update, < 2 GB |
| An 8× wider flat input dilutes the feedback (more literals for s to cover) | The pilot searches s for each cell |
| Small differences, one seed | Differences under ~0.3 pp are not claimed until seeds 1–2 |
| The 3080 has 10 GB: the flat 8-plane model at 8k (2·10⁹ automata, uint8 state + fp16 include cache ≈ 6 GB) may not fit | The smoke test builds every model at full size; a cell that does not fit is reported, not silently shrunk |

## 9. Decisions (user, 2026-11-13)

1. **D1:** no 3D cell with the usual Booleanization. CONV3 runs only with BIT8 (core) and GRAY8 (control).
2. **D11:** the **RTX 3080 only**, until the user asks for the 3090. Every timed run is on the 3080.
   `queue_runner.py` refuses the 3090 without `--allow-3090`.
3. The pilot's s search (and kd for CONV3) stays in. Build the harness and run the smoke test.
5. **kd = 8 (v4):** the 3D window covers all 8 bit planes; the pilot searches s only.
4. **Staging (v3):** run the 3D conv experiments first (stage A). The comparison with the flat / 2D
   conv cells (stage B) runs only if the user decides so after checking the stage-A results.
   `queue_runner.py` defaults to `--stage A`.
