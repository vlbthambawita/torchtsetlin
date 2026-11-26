# Library gaps found in this programme

`src/torchtsetlin/**` is read-only here (rule C1). Defects and missing pieces are listed, not fixed.

| | Gap | Effect here | Workaround in the harness |
|---|---|---|---|
| SS-1 | There is no dense 3D model: `Conv3dTsetlinMachine` classifies a whole volume, and `SegmentationTsetlinMachine` has no depth axis. | The 3D question of the plan cannot use a per-pixel model with a 3D patch. | Each pixel's 9×9 window is classified as a small volume (`code/models.py`, `win3d`). This costs one model evaluation per pixel and forces pixel sampling in training. |
| SS-2 | `ignore_index` skips the feedback of a pixel but not its evaluation. | Training a dense model on 256 sampled pixels per image is as slow as training on all 4 096. | None. The sampled dense reference is only a control, 6 runs. |
| SS-3 | The time of a dense update is dominated by the number of commits (`patches_per_commit=128`), not by the model size, for the sizes of this plan. | Small models cost as much as large ones; the grid is more expensive than a clauses × features estimate says. | None (the commit size is the tutorial's and is part of the fixed model, rule C5). |
