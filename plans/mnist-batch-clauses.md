# Plan: batch size × clause count on MNIST, flat vs convolutional TM (JMLR paper)

Status: **plan v3, harness built + smoke test** (2026-11-10), `experiments/mnist_batch_clauses/`. Timing / multi-seed runs need the user's go-ahead;
smoke runs are fine. v2: clause counts confirmed by the user as **per class** (D0). v3 (user):
large batches may need more epochs than B = 1, so the fixed 30-epoch budget is replaced by
**early stopping with patience 25 epochs** (D3); the equal-update control (old D5) is dropped,
because early stopping already lets every batch size train until it stops improving.

Target: the JMLR draft (`PAPER---torchtsetlin--Journal-of-Machine-Learning-Research/paper/`,
section *Performance and Fidelity*), extending the existing *Batched versus sequential feedback*
subsection (`sec:batching`), whose only evidence today is the 40-clause B2 sweep
(`generated/batch_sweep_table.tex`: 99.6 % at batch 16–50, 92.5 / 75.8 / 60.5 % at 100 / 200 / 500).

## 0. Questions

- **Q1.** How does final test accuracy depend on the mini-batch size `B` of `feedback_mode="batch"`,
  from `B = 1` (the online algorithm) up to `B = 1024`?
- **Q2.** Does the answer change with model size (clauses per class 1 000 … 64 000)? I.e. is there a
  largest "safe" batch `B*(C)`, and does it move with `C`?
- **Q3.** Is it the same for a flat `TsetlinMachine` and a `ConvTsetlinMachine`?
- **Q4.** What does batching buy: s/epoch, examples/s, accuracy per GPU-second (Pareto front)?

Prior evidence that makes Q2 non-trivial: the CIFAR-10 diagnostic
(`experiments/toolbox_cifar10/results/diagnostics/therm5_c*_b{1,50}.json`, 10 epochs, T fixed at 3 000)
gives b1 vs b50 = 63.4 / 62.9 at 2k, 65.5 / 62.4 at 8k, 65.8 / 56.6 at 16k: **the batch penalty
grows with clause count when T is held fixed.** Whether that is batching itself or T saturating as
votes grow is exactly what the T control (§3, D2) separates.

## 1. Hypotheses (fixed before any run)

- **H1** Accuracy is flat in `B` up to some `B*`, then falls (as on B2).
- **H2** With T fixed, `B*` shrinks as `C` grows; with T scaled ∝ C the dependence is weaker or gone.
- **H3** Accuracy rises with `C` and saturates; conv > flat at every `C` (B4 98.0 vs B7 99.1 at 2k).
- **H4** s/epoch falls with `B` until the GPU saturates (~B ≥ 256), so the Pareto-optimal batch is
  near `B*`, not at the largest batch.
- **H5** Epochs to the best validation accuracy grow with `B` (fewer commits per epoch), but
  GPU-seconds to the best are smallest at a moderate `B`.

## 2. Grid

| Factor | Levels |
|---|---|
| Model | flat `TsetlinMachine`; conv `ConvTsetlinMachine` |
| Clauses **per class** (D0; tt/tmu convention, ×10 classes = 10 000 … 640 000 in total) | 1 000, 2 000, 4 000, 8 000, 16 000, 32 000, 64 000 |
| Batch size `B` | 1, 16, 32, 64, 128, 256, 512, 1024 |
| Seeds | 0, 1, 2 |

2 × 7 × 8 × 3 = **336 runs** in the main grid.

Base configurations = the paper's own MNIST benchmarks
(`experiments/paper_experiments/code/configs.py`), so the grid's `(2k, b32)` / `(2k, b16)` cells
reproduce numbers already in the paper:

| | flat (= B4) | conv (= B7) |
|---|---|---|
| Data | MNIST, `x > 75` → 784 bits (`prepare_data.py::b4`) | MNIST, `x >= 75`, 28×28 (`::b7`) |
| Model | `TsetlinMachine(T, s=10, boost_true_positive=True, weighted=False)` | `ConvTsetlinMachine(T, s=5, patch_size=10, position_encoding=True, weighted=True, boost_true_positive=True)` |
| T at 2 000/class | 50 | 5 000 |
| Reference (5 seeds, paper) | 98.0 % @ 250 ep, b32 | 99.1 % @ 30 ep, b16 |

State at 64k/class: flat 640 000 × 1 568 literals ≈ 1.0 GB `uint8` + 2.0 GB fp16 include ≈ 3.2 GB;
conv ≈ 0.6 GB. Both fit either GPU.

## 3. Decisions (fixed now)

| # | Item | Decision |
|---|---|---|
| D0 | Clause count | **Per class** (user, 2026-11-10): `n_clauses=C` in torchtsetlin, as in tmu and the Optimized Toolbox paper; 64k/class = 640 000 clauses. All tables, figures, W&B tags (`c<C>`) and macros use the per-class number and say so. |
| D1 | Hyper-parameters | B4 / B7 as above; nothing tuned per cell. Only `C`, `B` (and `T`, D2) vary. |
| D2 | **T vs clause count** | **Main grid: T ∝ C** (`T = T_ref · C / 2000`: flat 25 … 1 600; conv 2 500 … 160 000), so the vote margin per clause is constant. **Control arm: T fixed** at `T_ref` for C ∈ {2k, 16k, 64k} × B ∈ {1, 32, 256} × both models, seed 0 (18 runs). Answers H2 and explains the CIFAR-10 diagnostic. |
| D3 | **Epochs: early stopping** (user, v3) | Large batches make far fewer commits per epoch (b1024: 49/epoch vs b1: 50 000), so a fixed epoch budget would penalise them unfairly. Every run trains until **no improvement for 25 consecutive epochs** (patience 25). "Improvement" = monitored accuracy exceeds the best so far by **≥ 0.05 pp** (5 of 10 000 examples), so noise-level wiggles do not reset the patience. Limits: **min 30 epochs**, **max 500 epochs**, and a **24 GPU-h wall-clock cap** per run; a run ending on a cap is flagged `stop_reason="max_epochs"` / `"time_cap"` and reported as such (not dropped). The reported test accuracy is the one at the best-validation epoch, read from the per-epoch curve (test is evaluated every epoch, so no best-model checkpoint is kept). |
| D3a | **What is monitored** | Early stopping looks at a **validation set, not the test set**: 10 000 images held out of the 60 000 MNIST training images (stratified, fixed split seed 0, the same for every run), training on the other 50 000. Stopping on test accuracy and then reporting test accuracy would select on the test set and inflate every number by the max-over-noise (on 10 000 examples ≈ +0.1–0.2 pp, larger for the noisier large-batch curves), biasing exactly the B comparison we want. Test accuracy is still computed and logged every epoch (curves, W&B) but never used for a decision. |
| D4 | Statistic | Per run: **test accuracy at the best-validation epoch** (primary), plus best-validation accuracy, last-epoch test accuracy, `best_epoch`, `commits_to_best`, `gpu_s_to_best`, `stop_reason`. Cell: mean ± sd over 3 seeds. A fixed-budget view is also reported for free from the curves: test accuracy at epoch 30 (every run has ≥ 30 epochs). |
| D6 | B = 1 | `feedback_mode="batch", batch_size=1` (exact online semantics, binomial draws). **Fidelity check:** `feedback_mode="sequential"` with B = 32 at C ∈ {2k, 16k}, both models, seed 0 (4 runs); must match the b1 cells within seed noise. |
| D7 | Data order | Per-seed, per-epoch permutation from a seeded generator, **identical across batch sizes and clause counts** for that seed, so `B` and `C` are the only things that change. Model seed = data seed. |
| D8 | Timing | Wall-clock is only comparable on one GPU with nothing else on it. **Seed 0 of every cell runs on the RTX 3090, one run at a time** → s/epoch, examples/s, peak MiB for the paper. Seeds 1–2 may run on the 3080 (accuracy only; timings recorded but not reported). Must not overlap the `toolbox_cifar10` queue on the same GPU. |
| D9 | Claims | `B*(C)` = largest `B` whose cell mean is within **0.3 pp** of the `B = 1` cell *and* within 2 sd. Fixed now; all 336 cells are reported, none dropped. |
| D10 | Library | torchtsetlin as installed (0.7.0), version + git sha in every record. `src/torchtsetlin/**` read-only; defects go to `LIBRARY_GAPS.md`. |

## 4. Cost (estimates; Phase 0 replaces them)

From existing 0.7.0 records on the 3090: flat 2k b32 = 1.6 s/epoch (b100 0.8); conv 2k b16 = 4.7 s/epoch;
CIFAR-10 conv b1 = 81 / 236 / 427 / 1 233 s/epoch at 2k / 8k / 16k / 64k (larger input than MNIST).
Assuming time ∝ C above ~4k, a validation pass per epoch (≈ test pass, small), and run lengths
under early stopping of **~60 epochs for B ≤ 64** (best near epoch 30–40 + 25 patience) and
**~100–250 epochs for B ≥ 128** (unknown: the main uncertainty):

| Block | per seed |
|---|---|
| flat, B ≥ 16, all C | ~15–25 GPU-h |
| conv, B ≥ 16, all C | ~50–80 GPU-h |
| flat, B = 1, all C | ~30–40 GPU-h |
| conv, B = 1, all C | ~40 GPU-h |
| **one seed of the grid** | **~140–190 GPU-h** |
| 3 seeds | ~420–560 GPU-h |
| controls D2 + D6 | ~40 GPU-h |
| **total** | **~450–600 GPU-h ≈ 10–13 days on both GPUs** (uncertainty ±2×) |

Early stopping roughly triples the 30-epoch cost (~200 GPU-h in v2): the patience alone adds 25
epochs to every run. Seed 0 therefore runs first and its measured run lengths decide seeds 1–2.
If too slow, in this order: (a) seeds 1–2 only for C ≤ 16k plus 64k (skip 32k); (b) B = 1 seeds
0–1 only for C ≥ 32k (the sequential check D6 anchors it); (c) patience 15 for seeds 1–2 —
each only with the user's OK and stated in the paper.

## 5. Weights & Biases (`simulamet_mlc/torchtsetlin`)

Same pattern as `experiments/toolbox_cifar10/code/tracking.py` (copied, not imported):

- **One W&B run per training run**, deterministic id `mbc-<model>-c<C>-b<B>-<tmode>-s<seed>`,
  `resume="allow"`, so a restarted run continues its W&B run.
- `group="mnist_batch_clauses"`; `job_type` ∈ `smoke`, `phase0`, `grid`, `control_T`, `control_seq`. Tags: `flat|conv`, `c<C>`, `b<B>`, `T-scaled|T-fixed`, `s<seed>`, GPU name.
- `config`: all hyper-parameters, `feedback_mode`, `batch_size`, `n_clauses`, `T`, seed, the
  early-stopping rule (`patience=25`, `min_delta=0.05`, `min_epochs=30`, `max_epochs=500`,
  time cap, `monitor="val_acc"`), split sizes, git sha, library version, env (torch, CUDA, GPU, host).
- **Per epoch** (step metric `epoch`, second axis `commits`): `val_acc`, `best_val_acc`,
  `epochs_since_improvement` (the patience counter, so a stalled run is visible live), `test_acc`, `train_acc_running`,
  `train_s`, `test_s`, `examples_per_s`, `commits` (cumulative `update` calls), `peak_gpu_mib`,
  `mean_clause_len` and `frac_empty_clauses` (read from `include_count`, no state change),
  `vote_saturation` (fraction of test examples whose winning vote sum reaches T — the D2 signal).
- **Summary**: `test_acc_at_best` (primary), `best_val_acc`, `best_epoch`, `epochs_run`,
  `stop_reason` (`patience` / `max_epochs` / `time_cap`), `commits_to_best`, `gpu_s_to_best`,
  `test_acc_epoch30`, last-epoch test, total train s, peak MiB, `status`; interrupted runs exit 143.
- **Aggregate runs** (`job_type="report"`, one per completed seed and at the end): a `wandb.Table`
  of every cell plus the report figures (heatmaps, curves, Pareto) as images, and a W&B Report
  "MNIST: batch size × clauses" built from the group.
- W&B is a **mirror, not the record**: `results/*.json` are the source of every reported number;
  any W&B error is printed once and ignored; `--no-wandb` / `MBC_WANDB=0` disable it;
  `code/wandb_backfill.py` uploads records made while offline.

## 6. Steps

### Phase 1 — harness (no GPU time)
`experiments/mnist_batch_clauses/` with `README.md` (rules R1 read-only library, R2 numbers only from
`results/*.json`), `LIBRARY_GAPS.md`, and `code/`:
- `common.py` — data (reuse `paper_experiments` data cache / `prepare_data.py` b4, b7), the fixed
  stratified 50 000 / 10 000 train/validation split (indices saved to `results/split.json` with a
  hash, checked on every load), grid registry, run tag ↔ config, env capture.
- `train.py` — one run: `--model --clauses --batch-size --t-mode --seed --device
  [--sequential] [--patience 25 --min-delta 0.05 --min-epochs 30 --max-epochs 500 --time-cap-h 24]`;
  per epoch: train, validate, test, update the patience counter. Resume checkpoint every 5 epochs **includes the early-stopping
  state** (best value, best epoch, counter, RNG states), so a restarted run stops exactly where an
  uninterrupted one would. JSON record to `results/<arm>/<tag>.json` with the full curve; the resume checkpoint is deleted when
  the run stops.
  The library's `tt.EarlyStopping` callback is not used because the harness drives `update()`
  directly and must checkpoint the stopping state; the rule is the same (patience on a monitored metric).
- `tracking.py` — §5. `queue.py` — file-based queue over GPUs, one run per GPU, skips done
  records, honours D8 (seed-0 cells pinned to the 3090).
- `report.py` → `REPORT.md`, `results/summary.json`, figures, and the paper's
  `generated/batch_clauses_{flat,conv}_table.tex`, `generated/batch_clauses_numbers.tex`
  (`\R{bc/<model>/c<C>/b<B>}` macros), `figures/batch_clauses_*.pdf`.
- `reproduce.sh`.
- **Smoke** (allowed now): every one of the 112 configurations for 1 epoch on 2 000 training images;
  8 of them (both models, C ∈ {1k, 8k, 64k}, B ∈ {1, 32, 1024}) mirrored to W&B in the separate group
  `mnist_batch_clauses_smoke` so the real group stays clean; checks configs, records, resume, W&B wiring. Plus an
  early-stopping unit check: with `--patience 2 --min-epochs 1` on a tiny run, kill and resume
  mid-patience and confirm the same `best_epoch` / stop epoch as an uninterrupted run.

### Phase 2 — Phase 0 pricing (≈ 2–3 GPU-h, needs go-ahead)
Both models, C ∈ {1k, 8k, 64k} × B ∈ {1, 32, 1024}: one epoch each (b1 at 64k: 2 000 updates,
extrapolated). Records s/epoch, peak MiB. Output: measured §4 and the queue order.
**Sanity gate:** flat (2k, b32) and conv (2k, b16), run with the new protocol, must land within
≈ 0.3 pp of the B4 / B7 5-seed results (B4 98.0 %, B7 99.07–99.19 %; those trained on all 60 000
images, so a small shortfall from the 50 000-image split is expected and is noted, not chased);
otherwise stop and find out why. The gate also gives the first measured run lengths under early
stopping.

### Phase 3 — the runs, most useful answer first
1. **Seed 0, full grid, on the 3090** (also yields all timings) + D6 sequential check.
2. **D2 T-fixed control** (18 runs) — answers H2 early.
3. **Checkpoint with the user:** seed 0's measured run lengths and GPU-hours → confirm or trim
   seeds 1–2 (§4).
4. Seeds 1 and 2 (3080 + 3090 when free).
All runs (grid and controls) use the same early-stopping rule (D3). `report.py` after each step;
W&B aggregate run after each.

### Phase 4 — write-up
- `REPORT.md` in the experiment folder: all 336 cells, `B*(C)` per model, H1–H4 verdicts.
- Paper (in the paper repo, commit only when asked):
  1. **Figure:** two heatmaps (flat | conv), rows C, columns B, colour = mean accuracy, `B*(C)` marked.
  2. **Figure:** accuracy vs B, one line per C (±sd band), both models; inset/second panel with
     T-fixed control.
  3. **Figure:** accuracy vs training GPU-seconds to the best epoch (Pareto), coloured by B.
  4. **Figure:** epochs to best (and commits to best) vs B, one line per C — the "large batches
     need more epochs" result (H5).
  5. **Table** (appendix): full grid mean ± sd of test-at-best, best epoch, s/epoch, plus the
     epoch-30 fixed-budget accuracies; count of runs that hit a cap.
  6. Text in `sec:batching`: replaces "the documentation recommends 16–64" with the measured `B*(C)`;
     the B2 sweep stays as the small-model case.
- If the measured safe range differs from the docs' "16–64", update `docs/` guidance and
  `CLAUDE.md` ("batch 10–50 matches sequential…") in a separate library commit.

## 7. Risks

- **Run length under early stopping is unknown** (mostly the B ≥ 128 runs); seed 0 measures it and
  the user re-confirms the budget before seeds 1–2 (Phase 3 step 3).
- **b1 cost at 64k** — each b1 64k run is ≥ 55 epochs (30 minimum + 25 patience); mitigations in §4.
- **TM validation curves are noisy and can creep up slowly**, so patience could keep resetting;
  `min_delta = 0.05 pp` and the 500-epoch / 24 GPU-h caps bound this, and capped runs are reported.
- **Smaller training set** (50 000 instead of 60 000) lowers every accuracy slightly versus B4/B7;
  it is the same for every cell, so the B and C comparisons are unaffected.
- **Timing contamination** from the `toolbox_cifar10` queue sharing the GPUs (D8).
- **T scaling choice** is the main design lever; the control arm reports the other choice, so the
  paper can show both.
- conv `position_encoding=True` is right for MNIST (B7 uses it; see CLAUDE.md caveat).
