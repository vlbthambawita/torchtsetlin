# Library gaps (rule C1)

`src/torchtsetlin/**` is read-only in this programme. Missing hooks or defects found while
working here are recorded below with a minimal reproduction, impact and a fix sketch.

## LG-1 (observation, not confirmed as a defect) — unweighted `ConvTsetlinMachine` plateaus on MNIST

- **Repro:** `ConvTsetlinMachine(10, 200, T=100, s=10, patch_size=10, position_encoding=True,
  input_shape=(1,28,28))`, MNIST `x > 0.5`, 50k train / 10k val (`manifests/split_seed0.json`),
  batch 32, `feedback_mode="batch"`, seed 0. Val accuracy 0.872, 0.848, 0.861 (epochs 1–3) and
  0.85–0.87 for 30 epochs (`results/source/conv_M200_T100_s10_*`). Same with `weighted=True`:
  0.952, 0.963, 0.965.
- **Why it might be a gap:** the original CTM (Granmo et al., 2019) is unweighted and reports
  high MNIST accuracy, so a 10-point gap from weighting alone is surprising. Could be batch-mode
  fidelity for conv feedback, `T` scaling, or expected behaviour at this clause count.
- **Impact here:** none after D10 (conv sources are weighted).
- **Next step (outside this programme):** compare `feedback_mode="sequential"` on a 5k subset.
