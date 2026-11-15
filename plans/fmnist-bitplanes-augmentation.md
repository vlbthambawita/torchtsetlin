# Plan: stage-A Fashion-MNIST runs again with data augmentation, plus best-validation models

Programme: `experiments/fmnist_bitplanes` (main plan `plans/fmnist-bitplanes.md`). Status
(2026-11-14): **decided and running.** User decisions: (1) "apply both" = `shift2flip` for all
aug runs, **no policy pilot** (Phase 1 dropped); (2) replay of the best cell only, after the aug
runs; (3) go-ahead. Phase 0 passed (`results/smoke/augment_checks.json`; resume with augmentation
gives an identical curve; a reloaded model reproduces its saved test accuracy). Chain
`code/aug_chain.sh` (logs `logs/q_aug.log`, `logs/q_replay.log`) started 2026-11-14 22:30 on the 3080. GPU: the RTX 3080 (cuda:1) only. W&B:
`simulamet_mlc/TM_fMNIST`.

## 1. What and why

Stage A (finished 2026-11-14, seed 0, B = 32, s = 5, kd = 8, no augmentation):

| Cell | C/class | Test acc. at best val. | Best epoch / run |
|---|---|---|---|
| conv3 bit8 | 2 000 | 86.77 | 63 / 88 |
| conv3 gray8 | 2 000 | 87.15 | 69 / 94 |
| conv3 bit8 | 8 000 | 88.02 | 90 / 115 |
| conv3 gray8 | 8 000 | 88.59 | 118 / 143 |

The user wants:
1. the same four runs again **with data augmentation**;
2. the **model at the best-validation epoch** of every run saved, so it can be reloaded for
   inference and its clauses inspected;
3. those models kept in a folder that is **not committed** to GitHub.

Questions:
- **QA1.** Does augmentation raise test accuracy at the best-validation epoch?
- **QA2.** Does it change the bit8 vs gray8 gap (+0.4 / +0.6 pp without augmentation)?
- **QA3.** Is the gain larger at 8 000 clauses/class than at 2 000? More clauses can absorb more
  variation.
- **QA4.** Are the augmented models more robust to shifted test images? With
  `position_encoding=True` a clause is pinned to a location, so the unaugmented models may be
  fragile to shifts.
- **QA5.** What do the clauses learn? Which planes and locations do they use, and how do
  augmented and unaugmented models differ?

## 2. Augmentation design

**Where it is applied: on the uint8 pixels, before Booleanization, on the GPU, freshly for every
batch.** Booleanizing first and augmenting the bits would be wrong for bit planes: a pixel's
8 bits must stay together, and any interpolation would produce invalid codes. So each training
batch goes through three steps: raw pixels, then augmentation, then encoding (bit8 or gray8, a
few integer ops on the GPU), then `model.update`.

Validation and test images are **never** augmented. Early stopping still watches the unaugmented
validation split, which keeps the statistic comparable with stage A.

**Policies.** Only geometric ones are used, with no interpolation, so pixel values stay exact
uint8 values:

| Id | Transform | Rationale |
|---|---|---|
| `shift2` | random translation of ±2 px in x and y (zero padding = background), independent per image | standard for 28 × 28 images; tests QA4 directly |
| `shift2flip` | `shift2` plus a horizontal flip with p = 0.5 | common for Fashion-MNIST, but every shoe in the dataset faces left, so a flip can create images the test set never has |

Rotation, scaling and elastic deformations are out of scope: they need interpolation and change
pixel values, which then changes the bit codes in complicated ways. Noise and erasing are out of
scope too.

**Choice between the two policies.** A short pilot (§4, Phase 1), decided on validation
accuracy, in the same way as the s pilot. Nothing else is retuned: s = 5, T = 5000·C/2000,
kd = 8, B = 32, patch 10 and position encoding all stay as in stage A. If augmentation needs a
different s, that is a separate follow-up and is not done silently.

**Implementation** (in the harness, `src/torchtsetlin/**` stays read-only):
- `code/augment.py` (new):
  - `augment(x_u8, gen, policy)` pads by 2, picks a per-image offset in [0, 4]², crops with a
    gather and optionally flips. It works on a `(B, 28, 28)` uint8 CUDA tensor.
  - Offsets and flips are drawn from a dedicated CPU `torch.Generator` seeded with
    `2000 + seed`, then moved to the GPU. The augmentation stream is therefore reproducible and
    separate from the batch-order generator (`1000 + seed`) and from the model's RNG.
- `encode_gpu(enc, x_u8)` produces bit8/gray8 on the GPU. A unit check asserts that it equals
  `common.encode` bit for bit on the unaugmented training set.
- The batch shape is unchanged (B = 32, with a last batch of 16), so the CUDA-graph capture of
  `update` is unaffected. Re-encoding 50 000 images per epoch costs milliseconds against
  26–96 s of training per epoch. Pricing confirms this (Phase 0).
- The record's `hp.augment = {policy, shift, flip_p, aug_seed}`. Tag suffix `_aug-<policy>`.
  New arms `aug_pilot` and `aug`, so stage-A records are never touched.

## 3. Best-validation models (not in git)

**Folder.** `experiments/fmnist_bitplanes/models/<arm>/<tag>.pt`.
- A new `experiments/fmnist_bitplanes/.gitignore` with `models/` keeps it out of git, as in
  cifar10_batch_clauses. `git check-ignore` is run before the first commit to prove it.
- `FMB_MODELS=<path>` can point the folder elsewhere.

**When saved.** At every new best validation accuracy. The file is written to `.tmp` and then
renamed, so a crash never leaves a half-written model. When the run stops, the file holds exactly
the best-validation-epoch model.

**Contents.**
- The model: `state_dict` (automata states, clause weights).
- Its scores: `epoch`, `val_acc`, `test_acc`, `test_acc_per_class`.
- What is needed to rebuild it with no other file: `tag`, `hp` (arch, enc, clauses, T, s, kd,
  patch, position encoding, input shape), the `augment` settings, `git_sha`, `env` and
  torchtsetlin `__version__`.

The run record gets `best_model: <path>`. These are the same rules as the CIFAR harness. Smoke
runs never save a model.

**Size.**
- conv3 kd8 has 8·100 + 2·(19 + 19) = 876 features, so 1 752 literals.
- With uint8 automata, 2 000/class gives 20 000 × 1 752 ≈ 35 MB and 8 000/class ≈ 140 MB.
- Eight models in total come to about 0.7 GB. The pilot models are kept too (about 0.14 GB).

**Unaugmented models.** Stage A saved no models, so clause comparisons (QA5) and the shift test
(QA4) have nothing to compare against. Arm `replay` reruns **only the best stage-A cell**, conv3
gray8 at 8 000 clauses/class (88.59 %), unchanged, with model saving: about 3.8 GPU-hours (143
epochs × 96 s). The user decided this on 2026-11-14: "only rerun without augmentation the best
parameters, not all". QA4/QA5 compare aug vs no-aug on this cell; the other three aug models get
the same analysis without an unaugmented partner. This doubles as a **reproducibility check**: the replay's
epoch-by-epoch validation and test curve is compared with the stage-A record.
- If it matches exactly, the saved model *is* the stage-A model.
- If not, both curves are reported and the replay record is what QA4/QA5 use. Stage A's reported
  numbers stay as they are.

## 4. Phases (all on the 3080; each phase needs the previous one to pass)

| Phase | What | Runs | Estimate |
|---|---|---|---|
| 0 | **Implementation + checks.** augment.py, encode_gpu, model saving; unit checks: GPU encoding = CPU encoding; augmentation with offset 0 and no flip is the identity; shifts land where expected; reloading the saved model reproduces the recorded val/test accuracy exactly; models/ is git-ignored. Smoke: 1 epoch on 2 000 images per cell and policy, plus a resume-from-checkpoint check with augmentation. Pricing: s/epoch with vs without augmentation. | smoke + 4 pricing | ~30 min |
| 1 | **Policy pilot.** conv3 {bit8, gray8} × {shift2, shift2flip}, C = 2 000, ≤ 60 epochs. Per encoding the policy with the higher best validation accuracy wins; ties go to `shift2` (the simpler one). Written to `results/aug_pilot/choice.json`. | 4 | ~2 h |
| 2 | **Main.** conv3 {bit8, gray8} × C {2 000, 8 000} with the chosen policy, full early stopping (patience 25, min 30, max 500 epochs, 24 h cap), best models saved. | 4 | ~12–20 h (augmentation usually needs more epochs) |
| 3 | **Replay** without augmentation, with model saving (§3): only the best stage-A cell, conv3 gray8 8 000/class. | 1 | ~4 h |
| 4 | **Analysis** of the saved models (§5): CPU/GPU, minutes. | — | < 1 h |

Total is about 19–27 GPU-hours on the 3080. Phases 2 and 3 are queued back to back by one script
(`code/aug_chain.sh`, in the style of `pilot_then_main.sh`), so the GPU never idles between them.
Phase 3 can run first if the user wants the unaugmented models sooner.

## 5. Inference and clause inspection (Phase 4)

`code/infer.py --model models/<arm>/<tag>.pt [--device cuda:1]`:
- rebuilds the model from `hp`, loads it in `eval()` mode and predicts the test set;
- **asserts** that the accuracy equals the stored `test_acc`, a load/rebuild check;
- writes `results/inference/<tag>.json` with the confusion matrix, per-class accuracy and the
  class sums;
- **shift test (QA4):** accuracy on the test set shifted by every (dx, dy) in [−3, 3]², a 7 × 7
  grid per model, plus horizontal flip. This is the same zero-padded shift as training, applied
  deterministically.

`code/clauses.py --model ...` writes `results/clauses/<tag>.json` and small figures to
`figures/clauses/`. The numbers and figures are committed; the models are not.
- **Statistics:**
  - distribution of clause length (included literals);
  - share of empty clauses;
  - weight distribution per class;
  - literals per bit plane, positive vs negated (`plane_usage`, already logged per epoch).
- **Location:** where clauses are pinned. The position bits in each clause are decoded to the
  range of (y, x) patch positions it allows, giving a heat map per class.
- **Top clauses:** for each class, the top-k clauses by weight, rendered as an 8-plane × 10 × 10
  include mask (+ literal / − literal).
  - For gray8/bit8 a pixel-level summary is also shown: the set of grey values a clause's
    literals on one pixel accept, since a conjunction over the 8 bits of one pixel is an interval
    or set of grey values.
  - This answers what a bit-plane clause "means" in pixel space.
- **Activity:** with `tt.interpret.clause_activity` / `clause_precision` on the test set (where
  they support conv3), how often each top clause fires and how precise it is. Where the library
  helpers do not cover conv3, the harness computes the same quantity with the model's
  `patch_clause_outputs`. Any missing library support goes to `LIBRARY_GAPS.md`, not into the
  library.
- **Comparison (QA5):** the same tables for the replay (unaugmented, conv3 gray8 8k) and the matching
  aug model side by side.
  Expectation: augmented models have looser position ranges and shorter clauses.

`code/report.py --arm aug` adds an `REPORT_aug.md` section that pairs each aug cell with its
stage-A and replay cell, and includes the shift-robustness grid and the clause summary.

## 6. Hypotheses (stated before running)

- **HA1.** Augmentation raises test accuracy at 8 000/class by ≥ 0.5 pp; at 2 000/class the gain
  is smaller or zero (capacity).
- **HA2.** Augmented runs stop later: best epoch ≥ 1.5× stage A.
- **HA3.** gray8 stays ahead of bit8.
- **HA4.** Unaugmented models lose ≥ 5 pp at a 2 px shift; augmented models lose ≤ 1 pp.
- **HA5.** With `shift2flip`, the flip hurts or does nothing, because test shoes all face left. The
  pilot therefore picks `shift2`.

All results are single seed (seed 0), so differences under about 0.5 pp are reported as
indicative, not significant. More seeds are possible later, only with the user's OK.

## 7. Rules (unchanged from the main plan)

- `src/torchtsetlin/**` is read-only.
- Every number comes from `results/<arm>/<tag>.json`, `results/inference/` or `results/clauses/`.
- The 3080 only; the 3090 stays with MNIST.
- Validation decides everything; the test set never decides anything.
- Commits only when the user asks; `models/` never.

## 8. Risks

- **Slower convergence.** Augmented 8k runs may approach the 24 h cap. 143 epochs took 3.8 h in
  stage A, so even 3× that fits. A capped run is reported as `time_cap`, not hidden.
- **CPU→GPU traffic.** The augmentation draws are tiny (offsets and flip bits, 3 bytes per image).
  The raw training set (39 MB uint8) stays on the GPU.
- **Replay is not bit-identical** (nondeterministic atomics in a kernel). This is handled in §3:
  reported, not blocking.
- **Clause decoding for conv3 kd8** depends on the literal layout, which is channel-major
  (Z·kh·kw, then position bits; `feature_groups` in train.py). A unit test builds a model,
  includes one known literal by hand in a copy, and checks that the decoder names the right
  plane, row and column.

## 9. Decisions for the user

1. **Policy:** run the `shift2` vs `shift2flip` pilot (recommended), or pick one directly.
2. **Replay:** decided. Only the best cell (conv3 gray8 8k, about 4 h). Open: before or after
   the aug runs (default: after).
3. **Go-ahead** to implement Phase 0 and then start the chain on the 3080.

## 10. Addendum (user, 2026-11-14): activation maps from the saved models

For every saved best-validation model (`models/aug/`, `models/replay/`), `code/activation_maps.py`
draws a 2 × 10 grid: row 1 one correctly classified test image per class (the first in test
order), row 2 one misclassified test image per true class (the first), each with the evidence for
the **predicted** class on top. Attribution is exact: each firing clause's signed vote is split
equally over the 10 × 10 windows where it matches and spread over their pixels, so a map sums to
the unclamped class vote (checked per image). Output: `figures/activation/<arm>/<tag>.png`,
`results/activation/<arm>/<tag>.json`, and the image logged to the run's W&B run
(`activation_maps`). `code/post_chain.sh` runs `infer.py` + `activation_maps.py` on every model
once `aug_chain.sh` has finished (log `logs/post_chain.log`).
