# Decisions and deviations from the plan

The plan is `torchtsetlin_plans/experiments/seg_simple/PLAN.md` (private repository). Rule C6: a change
after the plan was fixed is written here with its reason.

| | What the plan says | What the harness does | Why |
|---|---|---|---|
| D1 | The dense references of the 3D phase are "the model of section 5" (with 8 position bits per axis). | They are trained **without** the image position bits (`n400.k9.<enc>.nopos`, with and without pixel sampling). | A window model only sees the 9×9 window, not where the pixel is in the image. A reference that knows the position would mix the effect of the position with the effect of the convolution. T8 with position bits is still measured: arm `t8` now has (400, 3) and (400, 9). |
| D2 | Window models are updated with batches of 256 windows. | Batches of **64** windows. | With 256 the clauses of `win3d.T8.k5.d4` collapsed in a ten-epoch local check (hundreds of literals per clause, a quarter of the clauses empty, every pixel predicted as one class). With 64 the same model learns. This agrees with the library's note that batched feedback departs from the classical algorithm above a batch of about 50. Class-balanced sampling and a clause budget were tried in the same check and did not help; they are not in the harness. |
| D3 | One pricing phase; "the conv3d phase has its own pricing jobs". | Two phases, `price` and `price3d`. `price` holds one job per distinct shape of the main, side and wide runs (47), not only the 18 main cells. | The side arms change the cost (128 px, bit planes, T8, augmentation), so they are priced too. |
| D4 | 77 side runs, later 110 after the arms `sym`, `bits` and `t8`. | 113 side runs. | Arm `t8` has two cells (D1). |
| D5 | 66 window-phase runs. | 69 training runs: per encoding two dense references without position bits (all pixels, sampled pixels) instead of one new dense cell. | D1. |
| D6 | `Wgreedy` is offered "the best" window model. | A separate team `Wgreedy3d` (phase `conv3d`) chooses among the 18 main cells and all window models of the same seed. | `Wgreedy` belongs to phase `wide`, which finishes before the window models exist. |
| D7 | Four V100s, one runner each. | Three A100s (the user, 2026-11-26). `run_gpuK.sh` takes the GPU index and up to four runners per GPU; the wall-time estimate of the cost table uses `SS_GPUS` (default 3). | The hardware changed. The caps in GPU-hours are unchanged; an A100 hour is not a V100 hour, so the pricing phase on the A100s is what counts. Results are bit-reproducible only on the same GPU model. |
| D8 | The plan has no phase after `conv3d`. | Phase `improve` (2026-11-27, asked for by the user after the main results): 5 arms × 3 seeds at 400 clauses per class, model kind `pool` (README, "Phase `improve`"). | Every main run scored at the level of the position prior. The October programme measured +0.05 mIoU from pixels sampled across images plus a wide OR pyramid; the adaptive-threshold bits did best in the heads of `mask_rcnn_tm`. Rule C5 (the tutorial's fixed model) holds for the earlier phases; this phase changes the training order and the input bits on purpose, and nothing else (T, s, commit size, position bits, weighted clauses stay). The sampling recipe (512 pixels, shares 0.25 / 0.25) is the October finalist's, not tuned here. |
| D9 | — | Phase `coarse` (2026-11-27, asked for by the user after the `improve` results): model kind `cell` on **256 px** images (the user, same day), 10 arms × 3 seeds (README, "Phase `coarse`"). | The `improve` arms gained at most +0.02 test mIoU and their polyp IoU equals the position prior's. Before any two-step model (coarse map, then outline) is built, the coarse step is tested alone. Cell 16 and window 64 keep the image-to-cell and cell-to-window ratios small enough to price (below); one arm repeats the same field of view at 64 px. The largest window is 96, not 128: a 128 px window took about 10 minutes per epoch locally (LIBRARY_GAPS SS-5). Rule set here: if an arm without `.pos` beats the position prior on test mIoU by more than twice the paired standard error and its polyp IoU passes 0.35, the refinement step is worth building; otherwise the Booleanized input, not the per-pixel design, is the limit for these models. |
| D10 | — | Phase `multi` (2026-11-28, the user's idea, to be tested before any library change): model kind `multi` on 256 px images, 9 arms × 3 seeds (README, "Phase `multi`"). | Sizes 3, 5, 9, 17, 25 are the user's. Choices made here: the groups learn from one joint vote sum (rule `sum` of `joint_wide.py`; one arm trains them apart as the control), equal clause counts in the reference, 400 clauses per class in total so that the single-size controls and the earlier phases compare, T4 planes at full resolution in every size (no pooling), the position bits in every group, the pixel sampling of phase `improve`. The joint code is loaded from `experiments/mask_rcnn_tm`; a change there changes this phase. |

## Observations before any run on the server (local RTX 3090, not results)

- A dense epoch took about 27 s whatever the clause count and patch size tried (400 clauses, patch 3 and 9).
  `patches_per_commit=128` means about 22 000 commits per epoch on 700 images of 64×64, and the time per
  commit dominates. The plan's projection assumed the time grows with clauses × features, from a model that
  commits far less often. Expect the cost of the dense phases to be set mostly by the number of runs and
  epochs. The pricing phase on the A100s decides; the trimming rule of the plan applies if it is over the cap.
- Training on sampled pixels through `ignore_index` does not make a dense epoch faster: every pixel is still
  evaluated.
- Validation mIoU moves by several points from epoch to epoch. The best epoch is therefore partly a lucky
  one; test is scored once from it, so the test number stays a fair estimate.

## Observations before the `improve` runs (local RTX 3090, four epochs, not results)

- About 4.3 s per epoch for both the smallest and the largest arm; peak GPU memory 0.3 and 1.9 GiB.
- Validation mIoU still moved between 0.29 and 0.53 from epoch to epoch in these four epochs.
- The largest arm (826 features) had 241 included literals per clause after four epochs with s = 10. If its
  clauses keep growing, a clause budget or a larger s is the next thing to try; neither is in this phase.

## Observations before the `coarse` runs (local RTX 3090, two epochs, not results)

| Shape (256 px) | Train s / epoch | Validation s / epoch | Peak GPU memory |
|---|---|---|---|
| reference (cell 16, window 64, patch 5, T4, 400 clauses) | 36 | 15 | 4.0 GiB |
| window 32 | 11 | 2 | 3.8 GiB |
| cell 32 | 9 | 4 | 3.9 GiB |
| patch 9 | 73 | 42 | 4.0 GiB |
| 1 600 clauses | 94 | 15 | 4.0 GiB |
| T4+AG | 44 | 17 | 4.2 GiB |
| window 128 (not in the phase) | 181 | 433 | 4.5 GiB |

- Window 96 was timed per window only: 0.45 ms per training window and 1.61 ms per predicted window, which
  gives about 80 + 60 s per epoch. Patch 3, the `.pos` arm and the 64 px arm were not timed.
- At about 100 epochs per run the phase is roughly 50 GPU-hours on a 3090. A run needs about 4 GiB, so several
  runners fit on one A100.
- The true cell map scores 0.921 validation mIoU with cells of 16 pixels and 0.854 with cells of 32. The position
  prior at 256 px scores 0.529 test mIoU, as at 64 px.
- Validation mIoU after two epochs was between 0.43 and 0.55 and still moved from epoch to epoch. Clauses stayed
  small (14 to 48 literals).

## Observations before the `multi` runs (local RTX 3090, two epochs, not results)

| Shape (256 px) | Train s / epoch | Validation s / epoch | Peak GPU memory |
|---|---|---|---|
| reference (five sizes, 80 clauses per class each) | 29 | 6 | 3.8 GiB |
| one size, 25 × 25, 400 clauses | 22 | 8 | 3.9 GiB |
| five sizes, 320 each | 47 | 12 | 3.9 GiB |

- At about 100 epochs per run the phase is roughly 30 GPU-hours on a 3090.
- Prediction does not go through the members' `forward`: a call costs about 35 ms whatever the size of the
  member, which was minutes per split at 256 px. `MultiSize.votes` computes the same votes with one matrix
  product per group; `tests/test_harness.py::test_multi_groups_read_centred_neighbourhoods` checks that the
  two agree exactly.
- After two epochs the large groups held many literals per clause (3 × 3: 22, 9 × 9: 126, 25 × 25: 627 in the
  reference; 1 777 for 25 × 25 with 320 clauses per size), and validation mIoU was 0.39 to 0.53. If the clauses
  of the large groups stay this long, a clause budget for them is the next thing to try; it is not in this phase.

## Not built

- The `px128` arm trains at 128 px, where the same patch covers half the area; the report lists it in the
  side-arm table only (no separate patch line at 128 px).
- `report.py` computes the verdicts of H1 to H10 as the plan defines them. It does not test the monotone
  growth of the overfitting gap along the whole line (H4), only the two ends.
