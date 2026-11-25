# mask_rcnn_tm: detect-then-segment with Tsetlin machines on Kvasir-SEG

Research programme on top of the library (`src/torchtsetlin/**` is read-only here, rule C1).
Plan: `torchtsetlin_plans/experiments/mask_rcnn_tm/PLAN_KVASIR.md`. Deviations from it:
`DECISIONS.md`. Status: **harness built and smoke-tested on 2026-11-25; no experiment has been
run.** Numbers in any report come from `results/runs/*.json` (`python code/report.py`).

A region proposal model finds polyps, a RoI classifier re-scores the proposals, a box head
refines them and a mask head segments a 32×32 crop of each region. Every head is a Tsetlin
machine (or a wide team of them) on Boolean encodings of the RGB image; baselines are Mask R-CNN
variants, a U-Net and the dense Tsetlin machine of the segmentation programme.

## Run it on the server (four V100s, one tmux session per GPU)

```bash
git clone <this repository> && cd torchtsetlin/experiments/mask_rcnn_tm
./setup.sh                                  # once, any session: environment, data check, tests, smoke test
./run_gpu0.sh                               # tmux session 0  (default phase: price)
./run_gpu1.sh                               # tmux session 1
./run_gpu2.sh                               # tmux session 2
./run_gpu3.sh                               # tmux session 3
./status.sh                                 # progress, from anywhere
./status.sh --price                         # after the pricing phase: s/epoch and the projected cost
```

The data is read from `/work/vajira/DATA/Kvasir-SEG` (the folder that holds `images/` and
`masks/`, 1 000 JPEG files each); `KV_DATA=/other/path ./setup.sh` overrides it and stores the path
in `.kv_data`. `setup.sh` checks every file against the SHA-256 of `manifests/kvasir_split.json`. W&B: `wandb login --host https://api.forge.coreweave.com` before `setup.sh`;
runs go to `simulamet_mlc/TM_KVASIR`. Without a connection: `KV_WANDB=offline` (then
`wandb sync .cache/wandb/offline-run-*`) or `KV_WANDB=0`.

The runners share one job list (`code/jobs.py`). Phases are opened by hand, with the same value
in all four sessions, and a gate is read between them:

| `KV_PHASES` | What runs | Read before the next phase |
|---|---|---|
| `price` (default) | 3 epochs of every head and baseline, oracle ceilings | `./status.sh --price`; gate K0b in `results/runs/oracle-K0b.json` |
| `base,p1` | baselines, dense TM, stage A (single members), controls, baseline (iv) heads | K0c, K1, K2 (development): `python code/report.py` |
| `p2` | wide heads (stage B), capacity (stage C), seeds, K3 | K3, K5 |
| `alb` | the AlbumentationsX arm (optional; run it before `p3` if it should count) | `select-<head>-aug` records |
| `p3` | out-of-fold proposals, stage 2 retrained on them, end-to-end on validation | K2 (final), K4 |
| `final` | cross-validation, oracle swaps, **the sealed split (once per arm)** | the report |

Later stages depend on selection jobs (`select-*`): they read finished records, apply the
pre-registered rule and write the chosen configuration; the jobs that build on it appear on the
runners' next pass. Nothing is chosen by hand. `python code/queue_runner.py --gpu 0 --dry-run`
(with `KV_PHASES` set) lists what would run next and what waits for what.

Other knobs: `KV_SLOTS=2` (two runners per GPU; the Tsetlin heads are small), `KV_FG=1`
(foreground), `KV_SEEDS=0`, `KV_PATIENCE` (default 75), `KV_PANEL_EVERY` (default 15),
`KV_MAX_EPOCHS` (safety cap, 1 500). `./stop.sh` stops every run after its current epoch;
everything resumes from `models/<tag>/last.pt` when the runners are started again.

## What a run writes

| Where | What |
|---|---|
| `results/runs/<tag>.json` | the record: job, environment (commit, `src_tree_sha256`, GPU), history per epoch, best epoch, validation metrics |
| `models/<tag>/best.pt`, `last.pt` | best state (early stopping) and the resume checkpoint (not in git) |
| `figures/panels/<tag>/{test,val}_epoch_NNNN.png` | image, real mask, prediction, error map for the 16 + 16 panel images, every 15 epochs and at the end |
| W&B `TM_KVASIR`, run `kv-<tag>` | the same curves and panels (masks and boxes as overlays), grouped by head, filterable by encoding, convolution, clauses, T, s, augmentation |
| `results/stats/<tag>.json` | per-image scores and IoU matrices of the end-to-end evaluations (for the bootstrap) |

Early stopping: patience 75 epochs on the head's validation metric, earliest epoch on ties.
The test panel is visual only (rule C8): no metric is computed on test images before the
`sealed-*` jobs of the final phase, which refuse to run twice.

## Files

| File | Content |
|---|---|
| `code/data.py` | manifest check, instances, 128 px cache, symmetries, folds, panel manifest |
| `code/augment.py` | training views: `none`, `sym` (default), `alb`, `alb-photo`, `alb-geo` |
| `code/boolenc.py` | encodings T4, T8, G8, AG, T4+AG; OR pyramid; position bits |
| `code/detection/` | anchors, assignment, deltas and bins, proposals, crops, paste-back, AP evaluator (no Tsetlin imports) |
| `code/members.py`, `code/joint_wide.py` | members, teams (single / joint / composite), the gradient member of baseline (iv) |
| `code/heads.py` | the six heads: inputs, one epoch, fixed validation set, monitor, panel |
| `code/pipeline.py` | end-to-end inference, oracle swaps, dense + connected components, scoring |
| `code/train.py`, `code/baselines.py` | training with early stopping, resume, W&B, panels |
| `code/evals.py` | combine, select, props, e2e, k3, oracle, sealed |
| `code/jobs.py`, `code/queue_runner.py` | the job graph and the per-GPU runner |
| `code/report.py` | status, pricing table, `report/REPORT.md` |
| `tests/` | detection machinery and evaluator parity (K0a), harness tests T3, T7–T12, job graph |

Tests: `python -m pytest tests -q` (needs the data cache; `tests/test_after_smoke.py` needs the
smoke test). The smoke test alone: `KV_WANDB=0 python code/queue_runner.py --gpu 0 --phases smoke`
through `CUDA_VISIBLE_DEVICES=<one GPU>`.
