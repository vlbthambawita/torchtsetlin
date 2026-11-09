# Plan: torchtsetlin 0.7.0 on CIFAR-10 with the *Optimized Toolbox*'s best hyper-parameters, against the published numbers

Status: **plan v3, nothing run** (2026-11-09). GPU runs need the user's go-ahead.

Question (one): with the paper's best-reported configuration taken as is, what accuracy does the
newest torchtsetlin get, and how does it compare with the values the paper publishes? Nothing is
tuned, no ladders, no sweeps. **The paper's numbers are cited from Table IV, not re-measured: the
authors' tmu code is not run.** Only torchtsetlin results are produced.

Target paper: Grønningsæter, Smørvik, Granmo, *An Optimized Toolbox for Advanced Image
Processing with Tsetlin Machine Composites*, arXiv:2406.00704 (ISTM 2024),
`source_documents/papers/02_image_classification/2024_Optimized_Toolbox_...pdf`.
Authors' code: `github.com/cair/An-Optimized-Toolbox-for-Advanced-Image-Processing-with-Tsetlin-Machine-Composites`
@ `a4f6093`, tmu `TMClassifier`, GPU.

## 1. The paper's best configuration (used verbatim)

- **Hyper-parameters: Table III**, per specialist (found by Optuna at 2 000 clauses and *reused
  unchanged at every clause count*, §IV-B). `max_included_literals=32` (from the scripts),
  weighted integer clauses, 250 epochs, no validation split, no epoch selection.
- **Clauses: 64 000 per class** (the paper's best column; tmu convention, 640 000 clauses per
  specialist in total, `experiments/convtm/DECISIONS.md` DR-002).
- **22 specialists** (7 Booleanizations × windows, each plain and with horizontal-flip
  augmentation = 100 000 training images).
- **Targets (Table IV, 64 000 column):** composite **82.8 ± 0.01**; best single specialist
  **5×5 colour thermometers 75.4 ± 0.09**; every other specialist's 64 000 value.
- **Statistic:** mean ± sample variance over the **last 25 of 250 epochs**, test set, one run.
  Composite (Eqs. 4–5, `CIFAR10lastepochs.py`): `α_t = max − min` of specialist *t*'s test class
  sums, `argmax_i Σ_t c_t^i / α_t`, computed per epoch over the same 25 epochs.

| Specialist (×2: plain, Augmented) | Window | Weighted | T | s | Paper @ 64k (plain / aug) |
|---|---|---|---|---|---|
| Canny edge detection | 10×10 | yes | 1 500 | 10.0 | 60.6 / 62.4 |
| Adaptive Gaussian thresholding | 5×5 | yes | 2 500 | 3.5 | 70.2 / 70.3 |
| Adaptive mean thresholding | 10×10 | yes | 500 | 10.0 | 68.4 / 70.2 |
| Otsu's thresholding | 10×10 | yes | 3 000 | 10.0 | 65.0 / 65.7 |
| Colour thermometers 3×3 / 4×4 / 5×5 | 3, 4, 5 | yes | 3 000 | 5.0 | 74.3, 75.1, **75.4** / 72.7, 73.6, 74.1 |
| Adaptive colour thermometers 3×3 / 4×4 / 5×5 | 3, 4, 5 | yes | 3 000 | 5.0 | 71.7, 73.0, 73.3 / 70.8, 72.9, 72.9 |
| HOG (flat, 5 832 bits) | 32×32 | no | 50 | 10.0 | 67.5 / 68.5 |

## 2. Mapping onto torchtsetlin (fixed before any run)

| # | Item | Decision |
|---|---|---|
| M1 | Model | `ConvTsetlinMachine(n_clauses=64000/class, weighted=True, max_included_literals=32, position_encoding=True)` (tmu appends thermometer-coded patch coordinates); HOG = flat `TsetlinMachine`, unweighted. Soft budget = library default (tmu's rule). |
| M2 | Script vs paper conflicts | The scripts differ from Table III for **adaptive Gaussian** (script 10×10, T=1 500, s=10; table 5×5, T=2 500, s=3.5) and the **5×5 augmented thermometer** script uses `patch_size=4`. **Table III wins** (it is what the paper reports as the found hyper-parameters). Logged as a known ambiguity. |
| M3 | T at 64 000 clauses | §IV text says "scale T thereafter"; §IV-B and the scripts keep the 2 000-clause T. **T fixed, as the code does.** |
| M4 | Encoders | The scripts' cv2 calls ported verbatim: Canny (`GaussianBlur 3×3`, `Canny(100, 200)`), adaptive mean/Gaussian (`blockSize 11, C 2`), Otsu, HOG (`≥ 0.1`), thermometer (`x ≥ (z+1)·255/9`, 8 levels), `mlt_temp` (adaptive thermometers). Each checked **bit for bit** against the script on 100 images before training. |
| M5 | Feedback | Batched, batch 50 (the paper's library updates per example). Recorded as D1. |
| M6 | HOG at 64k | 640 000 clauses × 11 664 literals = 22 GB of state: does not fit 24 GB. HOG runs at the largest size that fits (32k or 16k, Phase 0 decides). The paper's HOG row is flat from 32k (67.5/67.5; aug 68.4/68.5), so the published effect is ≤ 0.1 pp. Deviation D2. |
| M7 | Seeds | One run per specialist, as in the paper. (A second seed of the 5×5 thermometer only if the first is a GAP, to tell noise from a real shortfall.) |
| M8 | Pass criterion | A cell is **REPRODUCED** if within **1.5 pp** of the paper's value (the paper's own variances are ≤ 0.3 at 64k except the augmented thermometers, ≤ 1.4); otherwise **GAP**, with a cause. Fixed now. |
| M9 | GPUs | RTX 3090 (24 GB) for the thermometer specialists (2.6 GB of state each) and HOG; RTX 3080 (10 GB) for the 3-plane specialists (1.3 GB of state) in parallel, if Phase 0 shows the working set fits. |

## 3. Cost (estimates; Phase 0 replaces them)

Old-library figures (`experiments/convtm/ARMS.md`, RTX 3090, batch 50): 5×5 thermometer 180.7 s/epoch at 2k/class, ~5 200 est. at 64k; other windows 3–10 s/epoch per 1 000 clauses. The current library is **7.2× faster** than 0.5.0 on the comparable conv benchmark (B7, `experiments/paper_experiments/results/timing{_v0.5.0,}/B7/B7_tt-cuda_batch_s0.json`: 1 083 s → 150 s), and 0.3.x/0.5.0 share the old conv path. Applying ÷6:

| Run | Est. s/epoch | 250 epochs |
|---|---|---|
| 5×5 colour thermometer, 64k/class (best single TM) | ~870 | **~60 GPU-h** |
| each thermometer-type specialist (12 of them; augmented ×2 data) | ~500–900 | 35–120 h each |
| each 3-plane 10×10 specialist (6) / 5×5 adaptive Gaussian (2) | ~250–500 | 20–70 h each |
| HOG at 16–32k (2) | ~150–300 | 10–20 h each |
| **all 22 + composite** | | **~1 100 GPU-h** ≈ 45 days on the 3090 alone, ~30 with the 3080 helping |

Memory at 64k, 5×5 thermometer (2F = 1 308): `ta_state` uint8 0.84 GB + `include` fp16 1.67 GB + bits ≈ 2.6 GB of state; the CUDA events path allocates no `(C, 2F)` count tensors. Working set at 640 000 clauses untested → Phase 0.

## 4. Steps

### Phase 0: fit-and-price check (≈ 1–2 GPU-h, needs go-ahead)
On 0.7.0: 20 batches + one test pass for each specialist *type* at 64k/class (HOG at 16k, 32k, 64k to find its ceiling), on both GPUs. Records s/epoch, test s, peak GPU MiB. Output: the measured version of §3 and the run order.

### Phase 1: harness (no GPU time)
`experiments/toolbox_cifar10/` (rule R1: `src/torchtsetlin/**` read-only; gaps to `LIBRARY_GAPS.md`; every number from `results/*.json` with argv, seed, git sha, library version, env). Reuse from `experiments/multi_convtm/`: trainer (`train_specialist.py`, per-epoch unclamped test class sums), composite rule (`composite.py`, verified to 0.04 pp on the authors' released sums), three encoders. Nothing from the tmu side is used. Add: the four missing encoders (M4), the 22-row Table III registry, **checkpoint + resume every 10 epochs** (a 60-h run must survive a restart), a queue runner over both GPUs, `report.py` → `REPORT.md` + `paper/generated/toolbox_*.tex`, `reproduce.sh`. 1-epoch smoke of all 22 on 1 000 images.

### Phase 2: the runs, in this order (useful answers first)
1. **5×5 colour thermometers at 64k** — the best single specialist (75.4). First answer in ~2.5 days.
2. The other 5 colour/adaptive-thermometer plain specialists, then the 3-plane plain specialists (3080 in parallel), then HOG → the **11-specialist non-augmented composite** (no published row, but reported).
3. The 11 augmented specialists → the **22-specialist composite at 64k vs 82.8**.
The composite is recomputed from the saved class sums as each specialist finishes, so the report is always current.

### What comes free from the same runs (no extra GPU time)
- **Speed at the published setting:** train s/epoch, test s, inference ex/s, peak GPU MiB per specialist, from the run records (torchtsetlin only; the paper reports no timings, and the authors' code is not run).
- **Batching:** the whole reproduction runs at batch 50; whether that matches the per-example reference is answered by the pass criterion itself. A batch-size sweep is *not* in this plan (add later if a specialist is a GAP and batching is a suspect: at 2k it costs ~2 h per batch size).

## 5. Deliverables for the paper

- `toolbox_best.tex`: Table IV's 64 000 column, 22 specialists + composite: published value (cited) beside torchtsetlin 0.7.0, ✓/GAP per cell, with the deviations D1–D2 in the caption.
- `toolbox_speed.tex`: torchtsetlin train s/epoch, inference ex/s and peak GPU MiB per specialist at the published setting.
- One line of prose per result, generated from the records.

## 6. Risks

- **Estimates in §3 are extrapolated** (÷6 from old measurements, linear in clauses); Phase 0 is cheap and comes first.
- **HOG** was 3.5 pp below the TM Composites paper at 2k in `multi_convtm` (soft budget, 0.3.2). If it is still a GAP on 0.7.0 it costs the composite ~0.3–0.5 pp and goes to `LIBRARY_GAPS.md`.
- **Wall time** (~30–45 days for the full composite): the run order above delivers the best-specialist answer first and a partial composite continuously; the user decides after Phase 0 whether to run all 22 or stop after the non-augmented 11.
- **Composite is transductive** (α over the test sums), as published; reported as such.
