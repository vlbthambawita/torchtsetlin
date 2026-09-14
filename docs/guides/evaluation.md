# Evaluation and metrics

Vote sums behave like logits, so any PyTorch/scikit-learn metric works. `torchtsetlin.metrics`
keeps the common ones on the device:

```python
votes = model(x_test)                                     # (B, K), eval() mode
tt.metrics.accuracy(votes, y_test)
cm = tt.metrics.confusion_matrix(votes, y_test)           # rows = true, cols = predicted
tt.metrics.precision_recall_f1(cm)                        # per-class tensors
print(tt.metrics.classification_report(votes, y_test, class_names))
```

Regression: `regression_metrics(pred, y)` (MAE, RMSE, R²). Multi-label:
`multilabel_metrics(pred_bool, y_bool)` (subset accuracy, Hamming accuracy, micro-F1).

## Confidence and trustworthiness

Chapter 7 of the book defines the confidence of a prediction as the winning vote sum and
calls a model *trustworthy* when accuracy increases with confidence:

```python
model.confidence(x)                                       # (v_max + T) / 2T in [0, 1]
curve = tt.metrics.trustworthiness_curve(votes, y_test)   # accuracy at each confidence level
tt.viz.plot_trustworthiness(curve)
```

## Probabilities and calibration

```python
model.predict_proba(x)                    # linear score (1 + v_k/T)/2, normalised
model.predict_proba(x, method="softmax")  # softmax(v / T)
tt.metrics.expected_calibration_error(model.predict_proba(x), y)
```

The linear score follows the uncertainty quantification literature for Tsetlin machines
(`P(y|x) = (1 + v/T)/2`); low normalised scores flag uncertain or out-of-distribution
inputs.

## Composites of several machines

Machines trained on different Booleanizations can be combined by normalising each member's
vote sums by its own range and summing (TM composites):

```python
def composite_votes(models, xs):
    total = 0
    for m, x in zip(models, xs):
        v = m(x)
        total = total + v / (v.max() - v.min()).clamp_min(1e-6)
    return total
```
