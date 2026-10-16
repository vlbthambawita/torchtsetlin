# Decisions and deviations (rule C4)

Spec: `tm-query-key-attention-mnist-implementation-plan.md` (frozen). Plan: `PLAN.md`.
Library revision at start: `40f17e68fbbf594851effe7bb93c885cdc51f363` (`torchtsetlin` 0.3.1),
`git diff --stat HEAD -- src/` empty. Environment: `environment.json`.

## 2026-10-16 — day 1 (PLAN §2.3)

- **D1 Backend.** `torchtsetlin` 0.3.1 through `code/tm_adapter.py`; Boolean torch tensors. TMU unused.
- **D2 Feedback.** `feedback_mode="batch"`, **batch 32 for every TM fit** (Q, K, heads, baselines). S2 checks it.
- **D3 Device.** Fitting on CUDA; attention reference in NumPy float64 (tests). Tests run on CPU and CUDA.
- **D4 State.** `n_states=128` (states 0..255, 8 logical bits) stored `int32`. Memory matched on automaton count.
- **D5 Matched counts.** Recomputed from the *selected* N0 at freeze time (`code/resources.py`).
- **D6 B1.** Primary stride 1; B1-s7 (stride 7) plan addition. `position_encoding=True` for both.
- **D7 Polarity.** Library convention, first `M/2` clauses positive. Even `M` only (adapter rejects odd).
- **D8 Scores.** `u = forward(x)/T` (clamped votes); raw sums only for diagnostics. Encoder diagnostics
  assert `clamp(raw, -T, T)/T == u` on every run.
- **D9 Pixel rule** `u >= 128` on uint8 for every arm.
- **D10 Layout.** `experiments/transformer_like/code/`, run as `python -m experiments.transformer_like.code.<module>`.
- **D11 Ties.** `torch.argmax` → lowest index (tested).
- **D12 Plan additions.** B1-s7, S1 (weighted), S2 (sequential) are separate labelled rows.
- **D13 RNG.** `SeedSequence([seed, sha256(name)])` streams; `torch.manual_seed` per TM fit.
  Tested 2026-10-16: same seed → identical `ta_state` after 2 epochs on CPU **and** CUDA
  (`test_same_seed_same_state`), so runs are bit-reproducible on a given device.
- **D14 Clause budget** not used (spec §4).

## 2026-10-16 — implementation

- **D15 Own classifier loop instead of `tt.Trainer`.** `code/training.py` implements the same rule
  as `EarlyStopping(patience=10, restore_best=True)` (strict `>`, earliest best epoch restored, stop
  after 10 non-improving epochs) but times training and validation separately, so validation
  evaluation is not charged to training time. Restoration is tested
  (`test_earliest_best_checkpoint_restored`).
- **D16 Constant-row guard in score normalisation.** The spec requires a constant score vector to
  map to zeros (§7.1, test 19). In float32 the mean of ten equal values leaves a ~1e-8 residual,
  which `1/sqrt(1e-6)` amplifies to ~3e-5, so the test failed. `normalize_votes` now zeroes rows
  whose max equals their min before scaling. Non-constant rows are unchanged (votes are integers,
  so distinct coordinates differ by at least `1/T`).
- **D17 Cache format.** Feature caches are `np.packbits` arrays in `artifacts/features/<key>.npz`;
  the key hashes the image ids, preprocessing, mode, temperature, thresholds, encoder state hashes
  and the source of `attention.py`/`features.py`/`tokenization.py` (spec §9).
- **D18 Prediction files** are gzip CSV (`results/<stage>/predictions/*.csv.gz`), columns as spec §17.
- **G0 passed:** manifests regenerate bit-identically (`test_manifests_reproduce`); raw-file
  SHA-256 in `manifests/dataset_manifest.json`; no identical-content duplicates across
  train/valid/test.
- **G1 passed:** adapter contract tests (spec 8–13, 34, 35) pass on CPU and CUDA.
- **D19 Exact uniform context (A0).** With the same encoders, CPU and CUDA feature bits were
  identical for every mode except `uniform`, where 3,288 of 34.2 M validation bits differed:
  `k/m` averages land exactly on 0.25/0.5/0.75 and summing `(1/m)·bit` rounds differently by
  device. A0's context is now `(integer sender counts) / m`, exact at the thresholds; after the
  change 0 bits differ (`results/cpu_cuda_feature_check.json`). QK-mode contexts within 1e-5 of a
  threshold (192–257 values per 10k images) are counted there; they did not flip between devices.
- **D20 A2-M matching.** A2-M's head is sized so that A2-M's *total* automata (head + the same Q/K
  encoders) matches N0's total; PLAN §2.3 D5 matched the head alone (M = 988 → 982 at defaults).
- **G2 passed (2026-10-16).** Data/tokenization tests pass; smoke run end to end on CPU and CUDA,
  all arms above chance (`results/smoke_{cpu,cuda}/`). Cost probe (`results/probe/`, 2 epochs,
  RTX 3090): B0 M=2000 11.5 s/epoch, B1-s7 M=2000 5.4 s, N0 head M=1000 23.7 s, B1 stride-1
  M=2000 32.0 s; peak ≤ 2.4 GB. Worst case 50 epochs: B1 M=2000 ≈ 27 min/trial; the search
  projects to < 10 GPU-hours. **No preregistered reduction needed.**
- **G3 passed.** Attention tests (spec 14–24, fixture) and feature tests pass on CPU and CUDA.
- **G4 passed.** Tuning encoders (seed 17, `results/encoders/seed17.json`): 200,000 patches/epoch,
  no blank images, ~78 s per encoder; validation patch accuracy Q 33.3 %, K 33.0 % (chance 10 %);
  mean per-token SD of `u` 0.17, no clipping saturation, no zero-variance tokens.
- **G5 passed.** 65 tests green on CPU and CUDA; test split refused before freeze; `src/` untouched.
- **D21 Scheduling.** Search: 2 jobs per GPU on both cards (selection uses validation accuracy only;
  timings in search records are co-scheduled). Final: 1 job per GPU, so per-record training times
  are uncontended but come from two different GPUs (GPU name in every record). Inference timing,
  the cost comparison of record, runs serially on the RTX 3090 afterwards.

## 2026-10-16 — after the search, before freezing (no test label read)

- **Search result that motivates D22** (`results/search/`): for B0 and B1 every T/M = 0.25 trial
  beats every T/M = 0.5 trial (B0 0.912–0.924 vs 0.860–0.874; B1 0.906–0.930 vs 0.873–0.890), and
  T/M = 0.5 trials of every family peak at epoch 1. `results/diagnostics/overfit_seed17.json`:
  at T/M = 0.5 train and validation accuracy fall together while mean clause size grows every
  epoch (N0 head 209 → 417 literals in 6 epochs, B0 160 → 264), i.e. it is not over-fitting but
  unbounded clause growth when the vote margin is half the clause count. The spec fixes N0's head at
  `T = M/2` (§11) while B0/B1 may choose 0.25, so the primary grid gives the baselines a
  hyper-parameter the proposed model lacks — and that hyper-parameter dominates.
- **D22 S3 "equal-grid" secondary (plan addition, decided before freezing).** The primary
  comparison stays exactly as specified. S3 adds the 12 N0 trials with head `T/M = 0.25`
  (same M, s, temperature grid, tuning seed 17, stage `search_s3`); the S3 N0 configuration is
  selected over all 24 N0 trials with the same key; then N0, A0, A1, A2, A2-M, A3 at that head
  setting and B0-M re-matched to its automaton count (B0-M re-tuned on s × T/M as in the primary)
  run on the 5 final seeds (stage `secondary_s3`), reusing each seed's encoders. B0, B1 and B1-s7
  are shared with the primary (their grids already contained 0.25). Reported in its own section
  with the same decision rules; never merged with the primary verdicts.
