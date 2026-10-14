# Improving Tsetlin-machine segmentation

The complete experiment plan is [experiments/segmentation/PLAN.md](../experiments/segmentation/PLAN.md).
It answers [segimporvements_task.md](segimporvements_task.md) and builds on the
[segmentation implementation plan](segmentation_plan.md), which was completed in v0.2.0.

The current library already predicts a label for every pixel. The research question is
whether learned context, better spatial reasoning, and better use of feedback can close
its quality and efficiency gap to a trained CNN. That gap must first be measured on
shared datasets and splits; synthetic scene accuracy alone does not establish it.

The real-data benchmarks are:

- **Kvasir-SEG:** `/work/vajira/DATA/Kvasir_seg_original/Kvasir-SEG`, binary polyp
  segmentation with a fixed 70/15/15 train/validation/final split, keeping duplicate
  groups together.
- **ACDC:** `/work/vajira/DATA/ACDC`, using the extracted `ACDC/database/` tree.
  Split its 100 training patients into 80/20 train/validation and reserve the 50 testing
  patients for final evaluation. Train on 2D ED/ES slices; evaluate reconstructed volumes
  with per-structure Dice and HD95, keeping every patient's slices/phases together.

Synthetic datasets remain mechanism checks. Dataset files are read-only; the experiment
will create its own manifests and caches. Models are trained separately for each dataset.

The proposed order is:

1. Audit evaluation, memory, feedback order, and tuning; establish dense-TM and U-Net baselines.
2. Test shuffled commits, targeted pixel sampling, and occupancy/contrast features.
3. Test a coarse-to-fine TM with learned vote features and fine-resolution skip inputs.
4. Compare Potts-model refinement and edge-aware diffusion using the same frozen TM votes.
5. Explore learned graph messages and topology-aware feedback only if earlier diagnostics
   identify a remaining need.
6. Confirm finalists on Kvasir-SEG and ACDC, independent seeds, distribution shifts, and measured
   resource budgets before proposing any public library changes.

The full plan specifies mathematics, controls, datasets, ablations, metrics, run records,
compute limits, stopping rules, and integration criteria. Its numerical thresholds are
proposed experimental criteria, not measured results. This delivery contains planning
documents only; no experiments have been run and no library behavior has changed.
