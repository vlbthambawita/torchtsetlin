# Models

All models are `torch.nn.Module`s with the same learning interface:

* `model.update(x, y) -> votes` — one learning step on a batch,
* `model(x)` / `model.forward(x) -> votes` — vote sums `(B, n_outputs)`,
* `model.predict(x)`, `model.predict_proba(x)`, `model.confidence(x)`,
* `model.rules()`, `model.clause_expression(j)`, `model.included_mask()`, `model.state_summary()`.

Dense (segmentation) models add `model.vote_map(x) -> (B, K, H, W)`,
`model.predict(x) -> (B, H, W)` and `model.clause_map(x)`; their `forward()` returns the
folded votes `(B·H·W, K)`.

::: torchtsetlin.models.TsetlinMachine

::: torchtsetlin.models.CoalescedTsetlinMachine

::: torchtsetlin.models.RegressionTsetlinMachine

::: torchtsetlin.models.ConvTsetlinMachine

::: torchtsetlin.models.ConvCoalescedTsetlinMachine

::: torchtsetlin.models.ConvRegressionTsetlinMachine

::: torchtsetlin.models.Conv1dTsetlinMachine

::: torchtsetlin.models.SegmentationTsetlinMachine

::: torchtsetlin.models.CoalescedSegmentationTsetlinMachine

::: torchtsetlin.models.TsetlinMachineBase
