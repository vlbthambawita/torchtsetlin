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

## Observations before any run on the server (local RTX 3090, not results)

- A dense epoch took about 27 s whatever the clause count and patch size tried (400 clauses, patch 3 and 9).
  `patches_per_commit=128` means about 22 000 commits per epoch on 700 images of 64×64, and the time per
  commit dominates. The plan's projection assumed the time grows with clauses × features, from a model that
  commits far less often. Expect the cost of the dense phases to be set mostly by the number of runs and
  epochs. The pricing phase on the V100s decides; the trimming rule of the plan applies if it is over the cap.
- Training on sampled pixels through `ignore_index` does not make a dense epoch faster: every pixel is still
  evaluated.
- Validation mIoU moves by several points from epoch to epoch. The best epoch is therefore partly a lucky
  one; test is scored once from it, so the test number stays a fair estimate.

## Not built

- The `px128` arm trains at 128 px, where the same patch covers half the area; the report lists it in the
  side-arm table only (no separate patch line at 128 px).
- `report.py` computes the verdicts of H1 to H10 as the plan defines them. It does not test the monotone
  growth of the overfitting gap along the whole line (H4), only the two ends.
