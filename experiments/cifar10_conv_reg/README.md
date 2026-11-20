# cifar10_conv_reg: drop clause, augmentation and 64 000 clauses for the single ConvTM on CIFAR-10

**Question.** The best single model of `../cifar10_batch_clauses` memorises the training set
(`conv_c32000_b32_Tscaled_s0`: 99.1 % running training accuracy, 72.2 % validation, no
augmentation, no dropout). Do drop clause (`drop_clause_p`) and training-time augmentation raise
its test accuracy, and does 64 000 clauses per class continue the trend? Plan (hypotheses, phases,
cost): `torchtsetlin_plans/experiments/cifar10_conv_reg/PLAN.md`. W&B mirror:
https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR (runs `ccr-…`, groups `ccr-screen`,
`ccr-factorial`).

Builds on `../cifar10_batch_clauses` (CIFAR-10 files, the 45 000 / 5 000 split, the model, the
runner scripts) and `../cifar10_joint_teams` (the augmentation policy, the evidence maps).

## Quick start (server with three A100s: GPUs 0, 3 and 1)

```bash
git clone -b joint-wide-tm https://github.com/vlbthambawita/torchtsetlin.git   # or git pull
cd torchtsetlin/experiments/cifar10_conv_reg
wandb login --host https://api.forge.coreweave.com   # once per server
./setup.sh          # once: environment, GPU check, data, smoke test (about 5 min)
./run_gpu.sh 0
./run_gpu.sh 3
./run_gpu.sh 1
./status.sh         # progress, cost table, chosen p*
```

`setup.sh` reuses the Python environment of `../cifar10_joint_teams/.venv` when that programme
was set up on the server, and creates `.venv` here otherwise. Its smoke test runs on GPU 0;
`CCR_SMOKE_GPU=3 ./setup.sh` picks another one. If a shell sees exactly one GPU (one reserved GPU
per tmux session), `run_gpu.sh K` uses that GPU and K only names the runner.

All runners must run in this folder: they share one job list through `.cache/claims/`. Each
starts a background process (log `logs/gpuK.log`, one log per run in `logs/runs/`);
`CCR_FG=1 ./run_gpu.sh 0` keeps it in the foreground. Re-running a script resumes: finished runs
are skipped, interrupted runs continue from their checkpoint (every 5 epochs). `./stop.sh` stops
after the current runs, `./stop.sh --now` immediately; `rm STOP` to continue.

## What runs

| Order | Phase | Runs | What |
|---|---|---|---|
| 1 | `pricing` | 8 timings | seconds per epoch and peak memory at 8k / 16k / 32k / 64k clauses, with and without drop clause |
| 2 | `factorial` | 2 | `base` and `aug` at 64 000 clauses (they do not need the screen) |
| 3 | `screen` | 8 | 8 000 clauses: augmentation {none, geo+col} × `drop_clause_p` {0, 0.1, 0.25, 0.5}; chooses p\* |
| 4 | `factorial` | 10 | the rest of {16k, 32k, 64k} × {`base`, `aug`, `drop`, `aug_drop`}, longest first |

With three runners, two start the 64 000-clause runs and the third works through the screen.
p\* is the smallest `p > 0` within 0.5 pp of the best `p > 0` with augmentation, on validation
(`results/screen/choice.json`); if no `p > 0` comes within 0.5 pp of `p = 0`, drop clause is
reported as not helping at 8 000 clauses and the factorial uses p\* = 0.1.

Every run: 3 × 3 colour-thermometer ConvTM (8 levels per channel), weighted, soft literal budget
32, position encoding, s = 5, **T = 1.5 × clauses per class**, batch 32, seed 0; 45 000 training /
5 000 validation images; early stopping on validation accuracy with **patience 50** (minimum 30,
maximum 500 epochs, 72 h per run). Statistic: test accuracy at the best-validation epoch.

- **Augmentation `geo+col`**: reflect-pad to 40 + random 32 × 32 crop, horizontal flip (p 0.5),
  colour jitter (p 0.5; brightness, contrast, saturation ± 20 %, hue ± 0.02), on the RGB image
  before the thermometer. One augmented image per training image and epoch; the random stream
  depends only on (seed, epoch, image index). Validation and test images are never augmented.
- **Drop clause**: each clause is switched off with probability `p` for one update (a new mask per
  update); all clauses vote at prediction. T is not rescaled.
- **`train_eval_acc`**: accuracy in prediction mode on 5 000 fixed un-augmented training images
  (`results/train_eval.json`), every epoch. The running training accuracy is measured under
  augmentation and dropout and cannot be compared between arms; this one can. `gap` = it minus the
  validation accuracy.

## What is kept

| Path | What |
|---|---|
| `results/<phase>/<tag>.json` | the record of a run: settings, environment, git sha, the whole curve, the summary. Source of every reported number |
| `models/<phase>/<tag>.pt` | **not in git**: the model at the best-validation epoch (state_dict, settings, epoch, validation and test accuracy), overwritten at every new best. Load with `train.load_model(path, device)` |
| `figures/<phase>/<tag>/eNNN_{panel,health,curves}.png` | the figures of every 20th epoch and of the last one; `best_panel.png`, `best_health.png` for the best-validation model |
| `REPORT.md`, `results/summary.json`, `figures/accuracy_vs_clauses.png` | written by `python code/report.py` |

Figures (also sent to W&B as `viz/*`):

- **panel** — ten fixed validation images, one per class (`results/panel.json`): the image, the
  evidence map of the predicted class (each firing clause's weight spread over its matching
  windows; the map sums to the class sum, asserted), the map of the true class when the
  prediction is wrong, and the three strongest matching clauses.
- **health** — histograms of clause length, clause weight, clause firing rate on 1 000 validation
  images, and the validation vote margin relative to T.
- **curves** — training (running and clean) and validation accuracy, and the gap, against epoch.

`python code/viz.py models/factorial/<tag>.pt` redraws panel and health for a saved model.

## Rules

| # | Rule |
|---|---|
| R1 | `src/torchtsetlin/**` is **read-only** here. Library defects go to `LIBRARY_GAPS.md`. |
| R2 | Every number comes from a `results/<phase>/<tag>.json` record. W&B is a mirror. `REPORT.md` is generated; do not hand-edit. |
| R3 | Nothing is tuned beyond the screen, and that choice uses validation only. |
| R4 | Early stopping watches the validation split, never the test set. |
| R5 | Seed 0 only until the user approves more; differences below 0.5 pp are ties. |

## Knobs

`CCR_VENV=<path>|none`, `CCR_WANDB=0` (no W&B), `CCR_PHASES=pricing,screen` (restrict a runner),
`CCR_DROP_P=0.25` (fix p\* without a screen), `CCR_WORKERS=<n>` (augmentation processes per run),
`CCR_DEVICE=cpu` (harness test without a GPU), `CCR_FG=1` (foreground runner).

A single run by hand:

```bash
. ./env.sh && cd code
python train.py --clauses 64000 --aug geo+col --drop-clause-p 0.25 --phase extra --device cuda:0
```

## Layout

| Path | What |
|---|---|
| `setup.sh`, `env.sh` | one-time setup; shared environment |
| `run_gpu.sh`, `gpu_device.sh` | one runner per GPU |
| `status.sh`, `stop.sh` | progress; graceful / immediate stop |
| `code/common.py` | factors, job lists, the choice rule, data, records |
| `code/train.py` | one run (or `--calibrate N` for pricing) |
| `code/augment.py` | augmentation policies and the worker pool (copy of the joint-team module without the OpenCV binarizations) |
| `code/viz.py` | the three figures |
| `code/queue_runner.py` | the shared job list, claims in `.cache/claims/`, the choice of p\* |
| `code/tracking.py` | W&B mirror, id `ccr-<phase>-<tag>` |
| `code/resume_check.py` | smoke: interrupted + resumed run = uninterrupted run (with augmentation and dropout) |
| `code/report.py` | `REPORT.md`, `results/summary.json`; `--status` for progress |
