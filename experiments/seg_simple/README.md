# Simple Tsetlin segmentation on Kvasir-SEG: clauses and patch size against quality

The tutorial model (`tt.SegmentationTsetlinMachine`, `docs/examples/segmentation.md`) applied to Kvasir-SEG at
64 px. Every trained model is a Tsetlin machine. The programme measures segmentation quality against

- the number of clauses per class (50 to 3 200, patch 3),
- the patch size (1 to 13, 400 clauses per class),
- both together (a 3 × 4 grid),

and, as extensions: augmentation by flips and rotations, bit-plane and 8-level thermometer encodings, wide
teams of machines with different patch sizes, and 2D / 3D convolutional machines on a 9×9 window per pixel.
Deviations from the plan are in `DECISIONS.md`, library gaps in `LIBRARY_GAPS.md`.

## Run it on the server (three A100s, one tmux session each)

```bash
cd experiments/seg_simple
./setup.sh                 # once: environment, data caches, gate G0, tests, smoke test
./run_gpu0.sh              # phase 'price'; start the scripts of the other two GPUs in their sessions
./status.sh                # progress, running jobs, gate G0, the cost table once pricing is complete
```

Kvasir-SEG is read from `/work/vajira/data/Kvasir-SEG` (`images/` and `masks/`); `SS_DATA=/other/path ./setup.sh`
overrides it. The runners share one job list and resume after a stop or a crash.

`run_gpuK.sh` runs on GPU index K. On the A100 server of the earlier programmes the three GPUs are 0, 3 and 1,
so the scripts are `run_gpu0.sh`, `run_gpu3.sh` and `run_gpu1.sh`. In a session that sees only one GPU
(`CUDA_VISIBLE_DEVICES`), any script uses that GPU and K only names the runner. With other GPU counts set
`SS_GPUS` so that the wall-time estimate of the cost table is right.

Phases, to be opened one after the other with the same argument in every session:

| Phase | Command | What runs |
|---|---|---|
| `price` | `./run_gpuK.sh` | 3 epochs of every distinct dense shape (47 jobs): seconds per epoch on this GPU |
| `main` | `./run_gpuK.sh main` | clause line, patch line, grid; 18 cells × 3 seeds = 54 runs |
| `side` | `./run_gpuK.sh side` | arms `T`, `nopos`, `bal`, `shuffle`, `sym`, `bits`, `t8`, `px128`; 113 runs |
| `wide` | `./run_gpuK.sh wide` | 18 extra member runs, then 33 team evaluations |
| `price3d` | `./run_gpuK.sh price3d` | 3 epochs of every window-model shape (23 jobs) |
| `conv3d` | `./run_gpuK.sh conv3d` | dense references, 2D and 3D window models; 69 runs and 3 team evaluations |
| `improve` | `./run_gpuK.sh improve` | pixels sampled across images, wide OR pyramid, adaptive-threshold bits; 5 arms × 3 seeds = 15 runs (added 2026-11-27, `DECISIONS.md` D8) |

Read `./status.sh` after `price` and after `price3d`: it prints the projected GPU-hours against the caps of the
plan (160 without the 3D phase, 50 for it). Over a cap, apply the trimming rule of the plan before starting.

Options: `./run_gpuK.sh main 3` starts three runners on the GPU (1 to 4, see below); `SS_WANDB=offline` or `0`; `SS_PATIENCE`
(default 30) and `SS_MAX_EPOCHS` (300); `./stop.sh` stops after the current epoch, `./stop.sh --now` at once.

## Phase `improve`

The main phase gave a test mIoU of 0.50 to 0.53 for every clause count and patch size, the level of the
position prior. This phase tests the two changes that helped in `experiments/segmentation` (October 2026), and
adaptive-threshold bits, at 400 clauses per class:

| Arm (tag) | Bits of a pixel | Features |
|---|---|---|
| `pool.T4.s1.n400` | T4, 3×3 neighbourhood: the bits of `n400.k3`. Only the training order changes | 124 |
| `pool.T4.s32.n400` | T4, 3×3 neighbourhood at OR-pooled scales 1, 2, 4, 8, 16, 32 | 664 |
| `pool.AG.s32.n400` | adaptive Gaussian threshold (one plane per colour), same pyramid | 178 |
| `pool.T4+AG.s32.n400` | both, same pyramid | 826 |
| `pool.T4+AG.s1.n400` | both, 3×3 neighbourhood only | 151 |

All arms keep the 8 position bits per axis, T = 0.2 n, s = 10 and commits of 128 pixels. Training pixels: 512
per image and epoch, drawn with replacement (half uniform, a quarter spread evenly over the classes of the
image, a quarter from the 2-pixel boundary band), pooled over all 700 images and mixed before they are
committed. The running training mIoU of these runs is measured on the sampled pixels, so it is not comparable
with the dense runs; the mIoU of the full training split at the best epoch is. An epoch took about 4 s on an
RTX 3090 in a local check, so the phase is small; no pricing phase belongs to it.

## Several runners per GPU

A dense run needs about 1 GiB of GPU memory and its time is set by the number of small commits, not by the
size of the model, so one run leaves an A100 mostly idle. Several runners on one GPU can therefore finish
more runs per hour. Whether they do is measured, not assumed: price with one runner per GPU first, then
start the main phase with `./run_gpuK.sh main 3` and compare the seconds per epoch in `./status.sh` or W&B with
the priced ones. If an epoch is more than three times slower, go back to fewer runners (`./stop.sh`, then
restart; runs resume). Records of runs that shared a GPU hold seconds per epoch that are not comparable with
runs that had it alone; the report's cost columns say nothing about that, so keep the number of runners the
same within a phase.

## What is written where

| | |
|---|---|
| `results/runs/<tag>.json` | one record per run: settings, per-epoch history, validation / test / train metrics at the best validation epoch, seconds per epoch |
| `results/floors.json`, `results/gate_G0.json` | floors without image content; the tutorial reproduction |
| `models/<tag>/best.pt`, `last.pt` | automata of the best validation epoch; checkpoint for resuming (not in git) |
| `figures/panels/<tag>/` | image, true mask and prediction of 8 fixed validation images every 15 epochs |
| W&B project `TM_SEG_SIMPLE` | the same curves and panels, run id `ss-<tag>`, grouped by set or arm |
| `report/REPORT.md` | `python code/report.py`: tables, figures and the verdicts on H1 to H10 |

Tags: `n400.k3__s0` is 400 clauses per class, patch 3, seed 0. Suffixes name an arm: `.sym` augmentation,
`.B8` bit planes, `.T8` 8-level thermometer, `.nopos` no position bits, `.T0.1` margin 0.1 n, `.bal`, `.shuffle`,
`.sampled`, `.px128`. `pool.T4+AG.s32.n400__s0` is a model of phase `improve`: encoding T4+AG, OR pyramid up to
scale 32 (`s1` is the 3×3 neighbourhood alone), 400 clauses per class. `win3d.T8.k5.d2.n400__s0` is a 3D window model with spatial patch 5 and depth 2.

## Code

| File | |
|---|---|
| `code/data.py` | manifest check, caches, encodings (T4, T8, B8, AG, T4+AG), pyramid planes and pixel sampling of phase `improve`, the 8 symmetries, floors |
| `code/models.py` | the four kinds of model behind one interface, the confusion-matrix metrics |
| `code/train.py` | one run: early stopping, checkpoints, record, panels |
| `code/evals.py` | team evaluations |
| `code/jobs.py` | the job list of every phase |
| `code/queue_runner.py` | one runner per GPU, shared claims |
| `code/report.py` | status, cost table, report |
| `code/g0.py` | gate G0 |
| `tests/` | T1 to T6 of the plan, encodings, windows, report on fabricated records |
