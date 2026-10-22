# Library gaps found while reproducing TM Composites

Rule R1: `src/torchtsetlin/**` is read-only here. Each entry records the symptom, a minimal
reproduction, the impact on this programme and a fix sketch. Nothing here was applied.

## MC-1: Prediction vote sums are clamped to [−T, T], so `forward()` cannot return tmu-style class sums

- **Status**: FIXED 2026-10-20 in `src/` at the user's request, overriding rule R1 for this entry. `forward()` returns unclamped sums via `clamp_prediction_votes = False`. Regression keeps the clamp. `predict_proba`/`confidence` clamp explicitly. Tests are in `tests/test_models.py` (`test_prediction_votes_are_not_clamped` and three more), and the change is noted in CHANGELOG "Unreleased". The workaround in `code/train_specialist.py` is now equivalent to `forward()` and stays for the existing records.
- **Symptom**: `TsetlinMachine.forward()` returns `_votes(clause_out, clamp=True)`
  (`src/torchtsetlin/models/classifier.py:106-112`, called from `models/base.py:280`). The clamp
  belongs to learning (the feedback probability uses the clamped sum), but it is also applied at
  prediction. There is no public way to get the unclamped class sums.
- **Evidence**: the authors' released tmu class sums (`data/tmcomposites_reference/`, epoch 99)
  lie outside [−T, T] for 77.9% (T10), 72.1% (C3), 73.1% (C4) and 40.2% (HoG) of the entries.
  They go down to −5 628 against T = 500 for T10. tmu's `predict(return_class_sums=True)` does not clamp.
- **Impact here**:
  1. The composite normalises by each specialist's class-sum range. Clamped sums make `α_t = 2T`
     and flatten every confident score to ±T, which changes the published method.
  2. Plain `predict()` accuracy suffers too. Every class that saturates at +T ties for the argmax,
     and ties go to the lowest class index. The smoke run measured T10 at epoch 1 as 38.90% clamped
     against 39.77% unclamped. **Measured over all 12 runs at epoch 100**, the clamp costs 0.6-3.9 pp
     of single-model accuracy (largest for the 3x3 thermometer model; `REPORT.md` section 4, generated
     from `test_acc_clamped_votes` in `results/*.json`). Any `torchtsetlin` user who calls
     `predict()` on a model whose class sums exceed T pays this today.
- **Workaround used**: `code/train_specialist.py::class_sums` mirrors `forward()` with
  `clamp=False`. `check_against_forward` asserts at every epoch that clamping it reproduces
  `forward()` exactly.
- **Fix sketch (as applied)**: `forward(x, clamp=False)` by default at prediction, keeping the
  clamp inside `update()` only; or a public `class_sums(x)` / `decision_function(x)`.
  `predict_proba(method="linear")` already rescales by T and would need checking.
- **Risk**: changing `forward()`'s default changes every `predict()` result in the library (for
  the better, by the tie argument), so it needs a CHANGELOG entry.

## MC-2: The hard clause budget disables the budget's own Type Ia -> Type Ib switch

- **Status**: OPEN (found 2026-10-21). An opt-in fix, `budget_mode="soft"`, has been added to the
  library (default unchanged); changing the default is a decision for the maintainer.
- **Symptom**: with `max_included_literals=32`, the 3x3 / 4x4 colour-thermometer and HoG
  specialists plateau 7-12 pp below the paper and below the authors' own code (tmu), which
  reproduces Table 2 on this machine (`results/tmu_*_run0.json`).
- **Cause**: `functional.apply_feedback` turns a firing clause's Type Ia into Type Ib only when
  `include_count > budget` (`functional.py:239`). Since 0.3.0, `_cap_new_inclusions` also
  reverts every inclusion beyond the budget (the hard cap), so no clause ever exceeds it and the
  switch never fires. Clauses fill to exactly the budget and their contents stop changing,
  except through slow Type Ia forgetting of false literals.
  tmu gates Type Ia on `included_literals <= max_included_literals`
  (`tmu/clause_bank/cuda/clause_feedback.cu:164`, v0.8.3) and never reverts. A clause therefore
  overshoots the budget, takes Type Ib (forget every literal with probability 1/s), shrinks, and
  regrows with different literals.
- **Evidence** (3x3 thermometer specialist, seed 0; `results/tmu_C3_clausestats.json`,
  `results/ttl_clausestats.json`):
  - **tmu:** median clause size 29-31; about 20% of clauses above 32 (max 41).
  - **torchtsetlin, hard budget:** every clause at exactly 32.
  - **torchtsetlin, soft budget:** median 31; 22% above 32.
  - **Accuracy at epoch 20, batch 50:** hard budget about 51%, soft budget 59.1% (60.5% at
    epoch 10), against tmu's about 60-61% at the same epochs. No budget gives 54.5%: the pruning
    is what helps, not merely the absence of a cap.
- **Fix**: `budget_mode="soft"` (TsetlinMachineBase and `functional.apply_feedback`) skips the
  hard revert, so a clause can overshoot and is then pruned, as in tmu. Tests:
  `tests/test_functional.py::test_soft_budget_allows_overshoot_then_prunes`.
- **Open question for the default**: 0.3.0 made the cap hard because a batched commit can
  overshoot it by far more than one sequential step can (issue #1). At batch 50 the soft rule
  measured here overshoots no more than tmu does (median 31, 22% above 32). A middle ground
  would be soft gating plus a much looser hard ceiling, for example 2x the budget.
