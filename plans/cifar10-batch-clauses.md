# Plan: batch size × clause count on CIFAR-10, flat vs 2D conv vs 3D conv TM (4 × V100)

Status: **plan v1 + harness** (2026-11-14), `experiments/cifar10_batch_clauses/`. Tested on the CPU on
mimir (encodings, job lists, pilot choice, ordering, every model path); the first GPU run is the
smoke test in `setup.sh` on the V100 server. Companion of `plans/mnist-batch-clauses.md` (same
protocol: early stopping on a validation split, T ∝ C, seed 0). Runs on a separate server with
four 32 GB V100s, one runner script per GPU, results mirrored to W&B
[`simulamet_mlc/TM_CIFAR`](https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).

User decisions (2026-11-14): 32 GB V100s; models **conv + flat + conv3d**; **reduced grid**
(C 1k–32k × all 8 batch sizes, seed 0, plus controls); the server gets the code by **git clone**
and has internet (CIFAR-10 download, pip).

## 0. Questions

- **Q1.** How does final CIFAR-10 test accuracy depend on the batch size `B` of `feedback_mode="batch"`,
  from `B = 1` (online) to `B = 1024`?
- **Q2.** Does the largest safe batch `B*(C)` move with model size (1k … 32k clauses per class)?
- **Q3.** Is it the same for a flat TM, a 2D ConvTM and a 3D ConvTM, and does CIFAR-10 (harder, more
  clauses needed, noisier) behave like MNIST?
- **Q4.** What does batching buy on a V100: s/epoch, GPU-hours to the best epoch (Pareto)?

Prior evidence: the CIFAR-10 diagnostic (`experiments/toolbox_cifar10/results/diagnostics/`, 10
epochs, T fixed at 3 000) gave b1 vs b50 = 63.4 / 62.9 at 2k, 65.5 / 62.4 at 8k, 65.8 / 56.6 at 16k:
the batch penalty grows with C when T is fixed. MNIST seed 0 (T ∝ C): conv flat in B up to 128,
−0.3 to −0.5 pp at B ≥ 256; flat collapses at B = 1024.

## 1. Hypotheses (fixed before any run)

- **H1** Accuracy is flat in `B` up to some `B*`, then falls.
- **H2** With T fixed, `B*` shrinks as `C` grows; with T ∝ C the dependence is weaker (control arm).
- **H3** Accuracy rises with `C` without saturating by 32k (Toolbox: 63.6 → 72.3 % from 2k to 32k).
- **H4** s/epoch falls with `B` only up to a moderate batch; the Pareto-optimal batch is near `B*`.
- **H5** Epochs to the best grow with `B`; GPU-hours to the best are smallest at a moderate `B`.
- **H6** conv (thermometer) ≥ conv3 (bit planes) at every C: a thermometer is ordinal, a bit plane
  is not, and with window depth 8 the two models have identical literal counts (§2), so the input
  code is the only difference.

## 2. Grid

| Factor | Levels |
|---|---|
| Model | `conv` `ConvTsetlinMachine`; `conv3` `Conv3dTsetlinMachine`; `flat` `TsetlinMachine` |
| Clauses **per class** (D0) | conv, conv3: 1 000, 2 000, 4 000, 8 000, 16 000, 32 000; flat: 1 000 … 8 000 (D2) |
| Batch size `B` | 1, 16, 32, 64, 128, 256, 512, 1024 |
| Seed | 0 (D13) |

Main grid: 48 + 48 + 32 = **128 runs**; controls 12 runs; pilot 6; pricing 24 timings.

| | conv | conv3 | flat |
|---|---|---|---|
| Input | colour thermometer, 8 levels per channel (`x ≥ k·255/9`), 24 × 32 × 32 (the Toolbox's `color_thermometer`, checked bit for bit) | 8 bit planes per channel, MSB first, volume 3 × 8 × 32 × 32 (planes reconstruct the pixels exactly) | colour thermometer, 3 levels per channel, 9 216 bits |
| Window | 3 × 3, stride 1, position encoding | (8, 3, 3): all 8 planes, 30 × 30 positions | – |
| Literals | 548 | 548 | 18 432 |
| T at 2 000/class, s | 3 000, 5 | 3 000, 5 | pilot (D3) |
| Other | weighted, boost TP, `max_included_literals = 32` (soft) | same | unweighted, boost TP, budget 32 |
| Reference | Toolbox Table IV, 3 × 3 thermometers: 63.6 % at 2k … 72.3 % at 32k (tmu, 250 epochs) | – | – |

conv is the Optimized Toolbox's 3 × 3 colour-thermometer specialist (arXiv:2406.00704, Table III;
`experiments/toolbox_cifar10`), the cheapest of its thermometer windows (3 × 3: 280 s/epoch vs 5 × 5:
438 s at 64k on the 3090) at −1.1 pp. conv3 follows the MNIST addendum / `fmnist_bitplanes`
(conv3 × bit8): with window depth = 8 there is one depth position, so it is a 2D conv over 24
bit-plane channels. That is stated in the paper; the 3D model is what the library user would write.

## 3. Decisions (fixed now)

| # | Item | Decision |
|---|---|---|
| D0 | Clause count | Per class, as MNIST (`n_clauses=C`; 32k/class = 320 000 clauses). |
| D1 | Hyper-parameters | §2; nothing tuned per cell except the flat pilot (D3). Only C, B (and T, D7) vary. |
| D2 | Memory | V100 32 GB. Flat state ≈ 15 GB per 10⁹ automata (MNIST flat 64k: 15.3 GB at 1.0·10⁹), so flat 8k (1.5·10⁹) ≈ 22 GB fits, 16k (2.9·10⁹) does not: flat stops at 8k. A cell that still runs out of memory is recorded `status="oom"` and reported as OOM, never dropped. |
| D3 | Flat pilot | No published CIFAR-10 flat configuration with this input. Pilot at C = 2 000, B = 32, ≤ 60 epochs over (T_ref, s) ∈ {50, 200, 800} × {10, 5} (T ≤ C/2, so unweighted votes can reach T). Chosen on **validation** accuracy; ties go to the earlier entry (B4's MNIST (50, 10) first). Written automatically to `results/pilot/choice.json`; flat grid runs wait for it. |
| D4 | Validation | 5 000 images held out of the 50 000 training images (stratified, split seed 0, indices + hash in `results/split.json`, re-checked on every load); training on 45 000. Test set evaluated every epoch, never used for a decision. |
| D5 | Data | Official CIFAR-10 python version, downloaded by `setup.sh` (md5 of the archive, sha256 of all six pickles, the same files as `toolbox_cifar10`). No augmentation. |
| D6 | Early stopping | As MNIST v4: patience 25 epochs, any new best validation accuracy counts (min_delta 0, ≥ 1 more correct validation image; a tie does not), min 30 epochs, max 500, **24 h cap per run**. Capped runs are flagged (`stop_reason`) and reported. Statistic: test accuracy at the best-validation epoch. Pilot: max 60 epochs. |
| D7 | T vs C | **Grid: T ∝ C** (`T = T_ref·C/2000`). **Control: T fixed** at T_ref: conv C ∈ {8k, 32k} × B ∈ {1, 32, 256}, flat C = 8k × same B (9 runs). |
| D8 | Fidelity | `feedback_mode="sequential"` with B = 32 at C = 2k for all three models (3 runs), against the B = 1 cells. |
| D9 | Chunk budget | `max_chunk_elements` for conv **and** conv3 = the smallest power of two ≥ 1.05·(3PC + 2CL + PL) (never below the default 2²⁷): 2²⁷ … 2³¹ from 1k to 32k. It keeps conv3's patch axis in one block, on the fused CUDA path (library gap FB-1, GitHub #17), and conv gets the same value so the two differ only in the input. A speed/memory knob, recorded per run; flat keeps the default. |
| D10 | Data order | Per-epoch permutation from a generator seeded by the seed only: identical across B, C and models. |
| D11 | Sanity gate | conv 2k B = 32 runs first among the grid cells (its T equals the Toolbox's). If its test accuracy at best < **60 %** (Toolbox 63.6 % after 250 epochs on 50 000 images), `results/STOP.json` halts all four runners for inspection. |
| D12 | B\*(C) | Largest B whose test accuracy is within **1.0 pp** of the B = 1 cell. One seed on 10 000 test images at ~65 % has a binomial SE of ≈ 0.5 pp, so MNIST's 0.3 pp would be noise. All cells are reported. |
| D13 | Seeds | Seed 0 only (as MNIST, user 2026-11-12); seeds 1–2 only with the user's OK after seed 0. |
| D14 | Timing | Four identical V100s, one run per GPU, nothing else on them; the four runners run concurrently (shared host CPU; recorded). Each record carries GPU name, slot and memory. s/epoch is comparable across the V100s, **not** with the MNIST 3090 timings. |
| D16 | Models kept | The model at the **best-validation epoch** of every grid/pilot/control run (user, 2026-11-14): `models/<arm>/<tag>.pt` = state_dict + hp + epoch + val/test accuracy + git sha, overwritten at each new best, written after the epoch's timing (not counted in s/epoch). **Git-ignored, never committed**; ~35 GB in total (flat ≈ 22 GB). Resume checkpoints (`.cache/ckpt/`, every 5 epochs) are separate and deleted when a run ends. |
| D15 | Library | torchtsetlin from the cloned repo (`pip install -e`), so every record has the version and git sha. `src/torchtsetlin/**` read-only; gaps go to `LIBRARY_GAPS.md`. |

## 4. Cost (estimates; the pricing phase measures them on the V100)

From 3090 records (toolbox_cifar10): conv 3 × 3 thermometer 64k, B = 50: 280 s/epoch on 50 000
images; B = 1 is 1.9–3× slower than B = 50; MNIST flat B = 1 is ~20× slower than B = 32. Assuming
time ∝ C, 45 000 training images, V100 ≈ 3090 (±50 %), ~150 epochs per run (MNIST conv: best epoch
~90 + 25 patience), and the 24 h cap:

| Block | s/epoch (B ≥ 16 / B = 1), 1k → top C | GPU-h |
|---|---|---|
| conv, 48 runs | 5 → 140 / 15 → 340 | ~125 |
| conv3, 48 runs | same as conv | ~125 |
| flat, 32 runs | 7 → 55 / 170 → 1 350 (B = 1 at 4k, 8k hit the 24 h cap) | ~95 |
| controls (12) | | ~85 |
| pricing + pilot | | ~5 |
| **total** | | **~435 GPU-h ≈ 4–5 days on 4 V100s** (range 3–10 days) |

The longest single runs are capped at 24 h (32k B = 1, flat 8k B = 1). Longest-first ordering keeps
all four GPUs busy until the end.

## 5. Phases (all automatic once the four scripts run)

**Phase 0, setup (`./setup.sh`, once, ~20 min):** venv with PyTorch (cu126 wheels: the V100 is sm_70)
and the repo's torchtsetlin; GPU check (each GPU runs a tiny conv/conv3/flat update; Triton fall-back
warnings are printed); CIFAR-10 download and checks; W&B login check; smoke test on GPU 0 (18 runs:
each model at its smallest and largest C × B ∈ {1, 32, 1024}, 1 epoch on 2 000 images; three
mirrored to W&B group `smoke`) and a resume check (interrupted + resumed run = uninterrupted run,
curve for curve).

**Phases 1–4 (`./run_gpu0.sh … ./run_gpu3.sh`).** One shared job list; each runner takes the next
free job in this order:

1. **Pricing** (24 × 200 timed updates): s/epoch and peak memory per model × C ∈ {1k, 8k, top} ×
   B ∈ {1, 32, 1024}; feeds the ordering.
2. **Flat pilot** (6 runs, D3) → `results/pilot/choice.json`.
3. **Sanity cell** conv 2k B = 32 (D11), then the **grid**, longest first (flat cells once the pilot
   has chosen).
4. **Controls**: sequential (D8), then T fixed (D7).

A runner exits when nothing is left. Restarting a script resumes: finished runs are skipped,
interrupted ones continue from their last checkpoint (every 5 epochs, including the early-stopping
state and RNG states).

**Phase 5, report (mimir):** copy `results/` back (`rsync` or commit), `python code/report.py` →
`REPORT.md` (per-model grids of test accuracy, best epoch, s/epoch, GPU-h to best; B\*(C); pilot;
controls; pricing) and `results/summary.json`. Paper figures/tables as in the MNIST plan §6 Phase 4,
with a CIFAR-10 column; the paper text compares B\*(C) between MNIST and CIFAR-10.

## 6. Weights & Biases

Project `simulamet_mlc/TM_CIFAR` on `https://api.forge.coreweave.com` (set by `env.sh`; log in once
with `wandb login --host https://api.forge.coreweave.com`). One W&B run per training run, id
`cbc-<arm>-<tag>`, `resume="allow"` (a resumed run continues its W&B run), group = arm (`grid`,
`pilot`, `control_T`, `control_seq`, `smoke`). Config = all hyper-parameters + early-stopping rule +
env + git sha. Per epoch: `val_acc`, `best_val_acc`, `epochs_since_improvement`, `test_acc`,
`train_acc_running`, `train_s`, `val_s`, `test_s`, `examples_per_s`, `commits`, `peak_gpu_mib`,
`mean_clause_len`, `frac_empty_clauses`, `vote_saturation`. Summary: `test_acc_at_best`, `best_epoch`,
`stop_reason`, `gpu_s_to_best`, …. W&B is a mirror: `results/*.json` are the record; W&B errors are
printed and ignored (`--no-wandb` / `CBC_WANDB=0` turn it off).

## 7. How to run (on the V100 server)

```bash
git clone https://github.com/vlbthambawita/torchtsetlin.git && cd torchtsetlin/experiments/cifar10_batch_clauses
wandb login --host https://api.forge.coreweave.com     # once (or let setup.sh tell you)
./setup.sh                                             # env, data, GPU check, smoke test (~20 min)
./run_gpu0.sh; ./run_gpu1.sh; ./run_gpu2.sh; ./run_gpu3.sh   # one runner per V100, in the background
./status.sh                                            # progress of all GPUs
./stop.sh            # graceful (after current runs);  ./stop.sh --now  immediate, resumable
rm STOP && ./run_gpu0.sh ...                           # continue after a stop
python code/report.py                                  # REPORT.md
```

## 8. Risks

- **V100 support.** PyTorch's CUDA 12.8+ wheels may lack sm_70 kernels; `setup.sh` installs the
  cu126 wheels and checks the arch list. Triton kernels that fail on the V100 fall back to the torch
  path with a warning (slower, same results); setup prints any such warning, and the pricing
  timings show the cost.
- **Run length** under early stopping on CIFAR-10 is unknown (noisier validation, slower learning);
  the 24 h cap bounds it, and capped runs are reported.
- **Flat memory** estimate (D2) comes from MNIST; pricing measures the peak at 1k and 8k first.
- **Concurrent runners** share host CPU and PCIe; B = 1 runs are launch-bound. Recorded, and the
  same for every cell.
- **conv3 = 2D conv on bit planes** when kd = 8 (H6 tests the code, not the operator).
- **Single seed**: differences under ~1 pp are within noise (D12).
