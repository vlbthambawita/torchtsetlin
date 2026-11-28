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
| `coarse` | `./run_gpuK.sh coarse` | 256 px images: convolutional machines that classify 16×16-pixel cells from a window; 10 arms × 3 seeds = 30 runs (added 2026-11-27, `DECISIONS.md` D9) |
| `multi` | `./run_gpuK.sh multi` | 256 px images: clause groups of several neighbourhood sizes per pixel with one joint vote sum; 9 arms × 3 seeds = 27 runs (added 2026-11-28, `DECISIONS.md` D10) |

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

## Phase `coarse`

No per-pixel model of the earlier phases passed a test mIoU of 0.55 (position prior 0.529). This phase asks
whether a Tsetlin machine can find the polyp at all when it sees a region instead of a pixel. It runs on
**256 px** images (the user, 2026-11-27); the earlier phases used 64 px.

The image is cut into cells of 16×16 pixels (a 16×16 grid). A cell is polyp when at least half of its pixels
are. A `tt.ConvTsetlinMachine` classifies each cell from a window centred on it (zero border outside the
image): its clauses slide a k×k patch over the window and fire when the pattern is anywhere in it. The cell
map is enlarged to the image and scored on the pixels. The record of every run also holds the score of the
true cell map (`cell_oracle`: what this cell size allows) and the floors at its resolution (`floors`).

| Arm (tag) | Change from the reference |
|---|---|
| `cell.c16.w64.k5.T4.n400.px256` | reference: 256 px, cell 16, window 64, patch 5, T4, 400 clauses per class |
| `cell.c16.w32...`, `cell.c16.w96...` | window 32 and 96 (128 is too slow, `DECISIONS.md`) |
| `...k3...`, `...k9...` | patch 3 and 9 |
| `...T4+AG...` | adaptive-threshold bits added |
| `...n1600...` | 1 600 clauses per class |
| `...n400.pos.px256` | the cell's position in the image added (7 bits per axis, the same in every pixel of the window) |
| `cell.c32.w64...` | cells of 32 pixels (8×8 grid) |
| `cell.c4.w16.k5.T4.n400` | 64 px image, cell 4, window 16: the same field of view at the old resolution |

T = 0.2 n, s = 10, weighted clauses and batches of 64 windows, as for the window models. Each epoch trains on
as many cells per image as the grid has, at random offsets (any block, not only the grid), drawn from all
images. Validation and test use the grid. The running training mIoU is over those sampled cells. Without the
`.pos` arm the model does not know where the cell is, so beating the position prior means it reads the image.

Before the first run on a machine: `. ./env.sh && python code/data.py` builds `.cache/data/kvasir_256.pt`
(`./setup.sh` does it too). No pricing phase belongs to this phase; seconds per epoch from a local check are in
`DECISIONS.md`.

## Phase `multi`

The user's idea (2026-11-28), tested here before any library change: one machine in which every pixel is
judged by several groups of clauses, each reading a neighbourhood of another size around that pixel.

```
pixel ─┬─ group  3 × 3   (80 clauses per class)  ─┐
       ├─ group  5 × 5   (80)                     │
       ├─ group  9 × 9   (80)                     ├─► one vote sum ─► polyp / background
       ├─ group 17 × 17  (80)                     │
       └─ group 25 × 25  (80)                    ─┘
```

Every group sees all T4 planes of its neighbourhood at full resolution (zero border outside the image) and the
8 + 8 position bits: 124, 316, 988, 3 484 and 7 516 bits. The groups learn **jointly**: feedback is drawn from
the sum of all votes against the sum of the margins (T = 0.2 × clauses of the group, so 80 in total), exactly as
if they were one machine whose clauses differ in what they can read. Once one group handles a kind of pixel,
the others get no feedback for it. Images are 256 px. Training pixels are sampled across images as in phase
`improve` (512 per image and epoch, commits of 128); s = 10, weighted clauses.

| Arm (tag) | Question |
|---|---|
| `multi.k3-5-9-17-25.n80x5.T4.px256` | reference: five sizes, 80 clauses per class each (400 in total) |
| `multi.k3.n400...`, `multi.k9.n400...`, `multi.k25.n400...` | one size with all 400 clauses, same training: does the mix beat a single size? |
| `...n160-100-80-40-20...` | more clauses for the small sizes |
| `...n20-40-80-100-160...` | more clauses for the large sizes |
| `...n320x5...` | 320 per size (1 600 in total) |
| `...n80x5.T4+AG...` | adaptive-threshold bits added |
| `...n80x5.T4.indep...` | the same groups trained apart, votes added only to predict: does joint feedback matter? |

The record of a run holds the included literals per clause of every group (`literals_per_clause_k<size>`) and
the floors at 256 px. Implementation: every group is a library machine fed the pixel's neighbourhood as a 1 × 1
image; the joint feedback is the team code of `experiments/mask_rcnn_tm/code/joint_wide.py`, loaded from there.
Needs the 256 px cache (`python code/data.py`).

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
scale 32 (`s1` is the 3×3 neighbourhood alone), 400 clauses per class. `cell.c16.w64.k5.T4.n400.px256__s0` is a model of phase
`coarse`: cell of 16 pixels, window of 64, patch 5, 256 px images (`.pos`: with the cell's position in the image). `win3d.T8.k5.d2.n400__s0` is a 3D window model with spatial patch 5 and depth 2.

## Code

| File | |
|---|---|
| `code/data.py` | manifest check, caches, encodings (T4, T8, B8, AG, T4+AG), pyramid planes and pixel sampling of phase `improve`, the 8 symmetries, floors |
| `code/models.py` | the six kinds of model behind one interface, the confusion-matrix metrics |
| `code/train.py` | one run: early stopping, checkpoints, record, panels |
| `code/evals.py` | team evaluations |
| `code/jobs.py` | the job list of every phase |
| `code/queue_runner.py` | one runner per GPU, shared claims |
| `code/report.py` | status, cost table, report |
| `code/g0.py` | gate G0 |
| `tests/` | T1 to T6 of the plan, encodings, windows, report on fabricated records |
