# Regression and multi-label

## Regression

```python
model = tt.RegressionTsetlinMachine(n_features=F, n_clauses=500, T=200, s=3.0, y_range=(0.0, 1.0))
trainer = tt.Trainer(model, batch_size=16)
trainer.fit((xb_train, y_train), epochs=30, val_data=(xb_val, y_val))   # logs val_mae, val_rmse, val_r2
model.predict(xb_test)                     # floats in [y_min, y_max]
model.rules()                              # 'IF x3 AND NOT x7 THEN +0.005'
```

All clauses are positive; the number of matching clauses (times weights) divided by `T`
gives the position in `[y_min, y_max]`. `T` is therefore also the output resolution — choose
it ≥ the number of distinct output levels you need. Pass `y_range` explicitly (otherwise it
is inferred from the first batch with a warning). Convolutional regression:
`ConvRegressionTsetlinMachine`.

## Multi-label

```python
model = tt.CoalescedTsetlinMachine(n_features=F, n_outputs=7, n_clauses=2000, T=100, s=10, multi_label=True)
model.update(xb, y_multi)                  # y_multi: (B, 7) Boolean
model.predict(xb)                          # (B, 7) Boolean, v_i > 0
model.predict_proba(xb)                    # (B, 7) in [0, 1]
```

Every output is updated per example: positive labels drive Type I feedback towards their
clauses, negative labels Type II (scaled by `negative_scale`, the `q` parameter of the
multi-task convolutional TM literature). The `Trainer` reports subset accuracy, Hamming
accuracy and micro-F1 for multi-label models.

## Multi-class with shared clauses

The same `CoalescedTsetlinMachine` with `multi_label=False` is a multi-class classifier whose
clauses are shared by all classes — useful when classes have common sub-patterns or when you
want a smaller model than `n_classes × n_clauses`.
