# Backend contract (spec §4, PLAN §2.1)

Backend: `torchtsetlin` 0.3.1 at `40f17e6`, used through `code/tm_adapter.py`. Every row is
enforced by a test in `tests/test_tm_adapter.py` (CPU and CUDA) unless noted.

| Obligation | How it holds | Test |
|---|---|---|
| Score columns in class order 0..9 | `clause_class = arange(C) // M`; votes column `c` sums class-`c` clauses | `test_raw_scores_are_manual_signed_sums`, `test_learns_boolean_fixture_and_class_order` |
| Raw vs clipped votes | `class_scores` = raw signed sums from `evaluate_clauses`; `votes` = library forward, clamped to `[-T, T]`; `unit_scores = votes / T` = spec `clip(s/T, -1, 1)` | same + encoder diagnostics assert it per run |
| Clause count meaning | `n_clauses` = clauses **per class**, both polarities, first `M/2` positive | manual-sum test |
| Literal order, negation, empty clauses | library appends complements; empty clause False in `eval()` (prediction), True in `train()`; adapter scores in `eval()` and restores the mode | `test_scoring_uses_eval_semantics_and_restores_mode`, `test_tie_goes_to_lowest_index` |
| Input dtype | Boolean torch tensors on the model device | — |
| Ordinary Type I/II feedback only | `weighted=False`, no dropout, no clause budget, `boost_true_positive=True` (library default) | config in every record |
| All 10 classes in each fit stream | `fit_epoch` raises otherwise | — |
| Repeated `fit_epoch` continues learning | state persists in `ta_state`; hash changes, never reset | `test_fit_continues_and_changes_state` |
| Save/load gives identical predictions | `state_dict` + config; `_refresh_include` in the load hook | `test_save_load_exact`, `test_pipeline_save_load` |
| Tie convention | `torch.argmax`, lowest index | `test_tie_goes_to_lowest_index` |
| Determinism | same `torch.manual_seed` → same state on CPU and CUDA | `test_same_seed_same_state` |
| State bits | `n_states=128` → states in `[0, 255]` | `test_fit_continues_and_changes_state` |
