# Interpreting a model

A trained Tsetlin machine is a list of rules. `torchtsetlin.interpret` helps you read them,
rank them and explain single predictions.

## Rules

```python
model.feature_names = ["four_wheels", "transports_people", "wings", "yellow", "blue"]
model.class_names = ["plane", "car"]
for rule in model.rules(polarity=1):           # only clauses voting *for* their class
    print(rule)
# IF four_wheels AND transports_people AND NOT wings THEN car
model.clause_info(3)                            # dict: class, polarity, weight, n_literals, expression
model.clause_literals(3)                        # {'positive': [0, 1], 'negated': [2]}
```

Coalesced models list every output a clause votes on with its weight; regression models show
the contribution of each clause.

## Which clauses matter?

```python
act = tt.interpret.clause_activity(model, x, y)          # (K, C): firing rate per class
cp = tt.interpret.clause_precision(model, x, y)          # support and precision per clause
top = cp["precision"].argsort(descending=True)[:10]
```

## Feature importance (closed form)

Blakely & Granmo (2020) define literal importance directly from the clauses, without
perturbation:

* **global**: the weighted fraction of positive clauses of a class that include a literal,
  `tt.interpret.global_feature_importance(model)` → `(K, 2F)`;
* **local**: the weight of the matching clauses of the predicted class that rely on each
  literal for a specific input, `tt.interpret.local_feature_importance(model, x)` → `(B, 2F)`.

For thermometer-encoded data, sum the bits of each original feature:

```python
imp = tt.interpret.global_feature_importance(model, class_index=1)
groups = [i // 8 for i in range(n_columns * 8)]           # 8 bits per column
per_feature = tt.interpret.aggregate_literal_importance(imp, groups)
tt.viz.plot_feature_importance(per_feature, column_names)
```

## Explaining one prediction

```python
print(tt.interpret.explain(model, x_test[0], feature_names=model.feature_names))
# prediction: 1  votes: [-10.0, 7.0]
#   [+1 -> output 1] clause 50: IF a>=0.12 AND NOT b>=0.41
#   ...
```

## Inspecting the memory itself

`model.ta_state` is the raw `(C, 2F)` state matrix; `model.included_mask()` the Boolean
include matrix; `model.literal_frequency()` counts how many clauses use each literal;
`model.state_summary()` gives quick statistics. See [Visualisation](visualization.md) for
plots of all of these.
