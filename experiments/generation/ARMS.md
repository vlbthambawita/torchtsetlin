# Arm registry — generative TM programme

Nothing runs that is not registered here first. `family`: `baseline` (no TM, defines floor/ceiling),
`primary` (the idea under test), `ablation`, `control`. `status`: `planned` | `implemented` |
`running` | `done` | `GAP` | `dropped`. **Every arm below is `planned`; none has been run.**

Costs are [ESTIMATE] from the `docs/benchmarks.md` anchor (39 000 ex/s at 5 000 clauses × 1 568
literals, RTX 3090), scaling as `n_clauses × 2F`. Fixes referenced as E*/P* are defined in
`VALIDITY.md`.

## Phase 0 — calibration (gate G0)

| arm | family | what it is | config | cost |
|---|---|---|---|---|
| `p0-fixedpoint` | primary | Synthetic contexts with known `P(y=1)`; measure `v/T` against `2p−1`. Tests **H0**. | F=12, M=512 contexts, T sweep {10,25,50,100,200}, 200 clauses, s=10 | 20 min |
| `p0-batch` | control | Same, sweeping batch {1,10,50,100,500} × `feedback_mode` {batch, sequential}. | best T from `p0-fixedpoint` | 30 min |
| `p0-repetition` | primary | Contexts repeated {1,10,100,1000}× — the calibration-vs-repetition curve. Tests **H1**. | as above | 20 min |
| `p0-multilabel` | control | K ∈ {16,64,256}, multi-class vs `CoalescedTsetlinMachine(multi_label=True)`; ECE over **all** K outputs, not argmax. Settles **P7**. | `negative_scale=1.0` | 30 min |
| `p0-mode` | control | Vote shift between `train()` and `eval()` scoring; empty-clause census. Confirms **E4**. | — | 5 min |
| `p0-sampling` | control | 10⁶ draws from `p̂`; KL(empirical ‖ true). | — | 5 min |
| `p0-capacity` | control | s ∈ {2,5,10,20} × clauses ∈ {50,200,800}. Sets Phase 1–2 defaults. | — | 30 min |

## Phase 1 — MNIST (gate G1)

| arm | family | what it is | config | cost (×3 seeds) |
|---|---|---|---|---|
| `mnist-marginal` | baseline | Lookup table `P(ink \| position, digit)`. No model. The NLL floor every TM arm must beat. | — | 2 min (CPU) |
| `mnist-logreg-window` | baseline | Logistic regression on the *identical* window features. Isolates "TM" from "these features". | same features as `mnist-tm-window` | 20 min |
| `mnist-pixelcnn-small` | baseline | Small PixelCNN for the NLL ceiling and the literature anchor. | ~5 layers | 1 h |
| `mnist-tm-window` | **primary** | Causal window (prev 2 rows + 4 left ≈ 60 px) + row/col thermometers + digit one-hot ≈ 126 feat. `(image,position)`-level shuffle (**E2**), 100 sampled positions/image/epoch. Applies **E1, E3**. | `TsetlinMachine(n_classes=2)`, 400 clauses/class, T,s from G0, 20 epochs | ~1 h |
| `mnist-tm-canvas` | ablation | The original full-canvas design with **E1** applied (position thermometer, no redundant one-hot) ≈ 850 feat. Breadth vs repetition — tests **H1** on real data. | 500 clauses/class | ~4 h |
| `mnist-tm-winsweep` | ablation | Window ∈ {1 row+2, 2 rows+4, 4 rows+8}. The H1 curve. | — | ~3 h |
| `mnist-tm-nocal` | control | Sampling from raw thresholded votes, no calibration. Shows what calibration buys. | reuses `mnist-tm-window` checkpoint | 10 min |
| `mnist-tm-gray16` | ablation | 16 intensity bins instead of binary. | `CoalescedTsetlinMachine(multi_label=True)`, 2 000 shared clauses | ~3 h |
| `mnist-cnn-judge` | control | CNN digit classifier trained on the same split, used only to score generated samples. Never used for tuning. | — | 10 min |

## Phase 2A — ECG codebook study, no TM (gate G2)

| arm | family | what it is | config | cost |
|---|---|---|---|---|
| `ecg-cb-grid` | baseline | Fixed-grid nonoverlapping blocks — the document's literal design. | B ∈ {25,50,100} × K ∈ {64,256,512,1024} | ~3 h CPU |
| `ecg-cb-raligned` | primary | R-peak-aligned tokenisation + explicit RR stream. The **P1** fix. | same grid | ~3 h CPU |
| `ecg-cb-8lead` | ablation | Quantise only the 8 independent leads (I, II, V1–V6); derive III/aVR/aVL/aVF. The **P3** fix. | best B, K | 1 h CPU |
| `ecg-cb-pq` | ablation | Product quantiser instead of k-means — gives the compositional sub-codes `ecg-tm-compositional` needs. | 4 sub-codebooks | 1 h CPU |

Gate G2 metric: per-lead held-out SNR ≥ 20 dB, seam discontinuity ratio ≤ 2.

## Phase 2B — ECG token models (gate G3)

| arm | family | what it is | config | cost (×3 seeds) |
|---|---|---|---|---|
| `ecg-marginal` | baseline | i.i.d. sampling from the token marginal. Floor. | — | 5 min |
| `ecg-markov-L` | baseline | **Count-based n-gram with backoff over the same tokens — the arm that can falsify H4.** If the TM does not beat this, the TM adds only interpretability. | L ∈ {4,8,16} | 20 min (CPU) |
| `ecg-roundtrip` | baseline | Codebook encode→decode of *real* held-out ECGs. Ceiling: nothing generated can beat it. | best G2 config | 10 min |
| `ecg-tm-onehot` | control | The document's literal design: `ONE_HOT(token, K+1)` × L context. Control for **P2**. | `Coalesced(multi_label=True)`, 4 000 clauses, L=16 | ~2 h |
| `ecg-tm-compositional` | **primary** | Context tokens encoded by product-quantiser sub-codes / centroid thermometers so clauses generalise across similar morphologies. The **P2** fix. | as above | ~2 h |
| `ecg-tm-phase` | **primary** | + cardiac phase (samples since last R, thermometer) and RR conditioning, L ≥ 20. The **P1b/P4** fix. | as above | ~3 h |
| `ecg-tm-8lead` | ablation | On the 8-independent-lead codebook; 4 leads derived. Lead-identity residual becomes exact. **P3**. | as above | ~2 h |
| `ecg-tstr-judge` | control | Rhythm classifier trained on synthetic, tested on real (TSTR). Scoring only. | — | 30 min |

## Registration rules

1. An arm enters this table **before** code is written for it; `code/arms.py` is the machine-readable
   source of truth and this file is the human index.
2. Every `primary` arm names the `baseline` or `control` it is scored against (C3).
3. Cost is re-tagged **[MEASURED]** after seed 0 completes; the estimate is kept alongside so the
   anchor's accuracy is itself a recorded result.
4. An arm that fails to reproduce is marked `GAP`, not silently dropped, and gets a
   `LIBRARY_GAPS.md` entry if the cause is in `src/torchtsetlin`.
