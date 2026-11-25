# Decisions and deviations from PLAN_KVASIR.md

Dated entries. The plan is `torchtsetlin_plans/experiments/mask_rcnn_tm/PLAN_KVASIR.md`.

## 2026-11-25, while building the harness

| # | Plan | Harness | Reason |
|---|---|---|---|
| D1 | anchors at stride 8 (§5) | **stride 4**: 32×32 cells, 24 shapes, 24 576 anchors | gate K0b. On the 128 px grid the harness trains on, stride 8 reaches best-anchor recall 0.988 on train (O1 needs ≥ 0.99); stride 4 reaches 0.999 (val 0.970). The audit in the plan used native boxes scaled to 128 px and found 0.992; the boxes of the resized masks differ by up to a pixel. Measured: `results/runs/oracle-K0b.json`, `manifests/anchors.json` |
| D2 | `setup.sh` caches the eight symmetric copies | symmetries are applied on the device per epoch (`augment.make_view`) | exact tensor operations; one code path for `sym` and `alb`; test T8 |
| D3 | batches of 128 rows from 16 images | rows are gathered from chunks of 64 images, shuffled across the chunk, committed in mini-batches of the planned size | more mixing across images, which the segmentation programme found to matter; memory stays bounded |
| D4 | mask head predicts with `SegmentationTsetlinMachine` after a state load if T1 passes | every head predicts with the flat model it was trained as, on all rows | no dense class is used anywhere, so T1, T2, T4, T5 of `PLAN.md` have nothing to test; T3 and T7–T12 are in `tests/test_harness.py` |
| D5 | composite members "trained independently" | a composite is a `combine` job: it loads the members' own best checkpoints (each early-stopped alone) and validates the sum | the definition of a composite; costs no training |
| D6 | "three best different (encoding, convolution) families" | a family is one (encoding, convolution type, patch, scales or depth, crop) setting, so the three best distinct stage-A settings | the stage-A grid has one run per setting |
| D7 | AlbumentationsX pipeline copied from `cifar10_joint_teams/code/augment.py` | own implementation in `code/augment.py` | that file transforms images only; here masks, boxes and proposals go through the transform. No worker pool: 700 images take about 2 s |
| D8 | baselines use `alb` when the Tsetlin heads adopt it | baselines (i), (iii) and the dense TM are trained at both levels in phase `alb`; the report lists both | heads adopt `alb` one by one, so there is no single level to copy |
| D9 | dense TM validation mIoU | monitored on every second pixel; the end-to-end jobs score the full grid | validation each epoch would otherwise cost several times a training epoch |
| D10 | stage-2 sampling of §5 (64 RoIs per image, 25 % foreground) | used for the classifier; the box head takes up to 8 foreground RoIs per image, the mask head 2 | only foreground RoIs train these heads |
| D11 | AlbumentationsX | tested here only with `albumentations==2.0.8` (the build machine has Python 3.10); `setup.sh` installs AlbumentationsX on Python ≥ 3.11 and test T12 runs there | |

## Not built yet

| Item | Plan | State |
|---|---|---|
| RPN-C control arm | §6 | not built (needs a coalesced subclass with an ignore mask); RPN-A, the per-shape control, is built |
| Bootstrap intervals for gates K4 and K5 in the report | §10 | the per-image statistics are saved (`results/stats/`) and `detection.evaluate.bootstrap_ap` exists; `report.py` prints mean ± s.d. over seeds only. K3 has its interval |
| Explanation cards, gate K6 | §10 | not built |
| Ablations P1–P13 of TM B.7 | PLAN.md §9 | only the ones the search covers (P3 bins in stage A of the delta head; P9 as the recall gap in the `props` record) |
| Decision D-5's skip rule | PLAN.md §8 | proposals are always out of fold; the in-sample gap is recorded |
