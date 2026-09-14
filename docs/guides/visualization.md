# Visualisation

`torchtsetlin.viz` needs matplotlib (`pip install "torchtsetlin[viz]"`). Every function
returns the `Axes` it drew on.

| Function | What it shows |
|---|---|
| `plot_history(history)` | metric curves per epoch |
| `plot_confusion_matrix(cm, class_names)` | confusion matrix heat-map |
| `plot_clause_memory(model, clause)` | book-style memory of one clause: bar per literal, dashed include boundary |
| `plot_ta_states(model)` | heat-map of all automata (clauses × literals) |
| `plot_literal_frequency(model)` | how many clauses include each literal |
| `plot_feature_importance(importance, names)` | horizontal bars |
| `plot_conv_clause(model, clause)` / `plot_conv_clauses(model, clauses)` | convolutional clause patches (black = on, white = off, gray = free) with the allowed region |
| `plot_votes(votes, y)` | distribution of winning vote sums, correct vs incorrect |
| `plot_trustworthiness(curve)` | accuracy and coverage vs confidence |
| `plot_clause_weights(model)` | weights of weighted / coalesced models |

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tt.viz.plot_clause_memory(model, clause=0, ax=axes[0])
tt.viz.plot_votes(model(x_test), y_test, ax=axes[1])
plt.show()
```
