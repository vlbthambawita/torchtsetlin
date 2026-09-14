# Changelog

## 0.1.0 (2026-09-14)

First public release.

- Models: `TsetlinMachine` (multi-class, optional integer clause weights),
  `CoalescedTsetlinMachine` (shared clauses, multi-class / multi-label),
  `RegressionTsetlinMachine`, convolutional variants (`ConvTsetlinMachine`,
  `ConvCoalescedTsetlinMachine`, `ConvRegressionTsetlinMachine`, `Conv1dTsetlinMachine`).
- Learning options: specificity `s`, vote margin `T`, memory depth `n_states`, boosted
  true-positive feedback, clause size constraint (`max_included_literals`), drop-clause and
  drop-literal regularisation, focused negative sampling, batched or exact sequential feedback.
- Data preparation: thermometer / one-hot / bit-plane / adaptive-threshold / colour thermometer
  / hypervector encoders, synthetic datasets, torchvision helpers.
- Training: `Trainer` with `DataLoader` support, callbacks (early stopping, checkpoints, CSV
  logging, hyper-parameter schedules), metrics.
- Interpretation: rule extraction, clause activity/precision, closed-form global and local
  feature importance, per-example explanations.
- Visualisation: memory plots, automata heat-maps, convolutional clause patches, confusion
  matrices, trustworthiness curves.
- Documentation site (MkDocs) and examples.
