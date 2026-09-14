# Quick start

This page walks through the complete workflow on the *Noisy XOR* benchmark from the original
Tsetlin machine paper: 12 Boolean features of which only the first two matter
(`y = x0 XOR x1`), with 40 % of the training labels flipped.

## 1. Data

Inputs must be **Boolean** (`torch.bool`, or anything that casts to it: 0/1 integers, floats,
NumPy arrays). Continuous or categorical data is *Booleanized* first — see
[Booleanization](../concepts/booleanization.md). Labels are integers `0..n_classes-1`.

```python
import torch, torchtsetlin as tt

x_train, y_train = tt.data.make_noisy_xor(5000, noise=0.4, seed=0)
x_test, y_test = tt.data.make_noisy_xor(2000, noise=0.0, seed=1)
x_train.shape, x_train.dtype, y_train[:5]
# (torch.Size([5000, 12]), torch.bool, tensor([0, 1, 1, 0, 1]))
```

## 2. Model

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = tt.TsetlinMachine(
    n_features=12,   # Boolean inputs (None = infer from the first batch)
    n_classes=2,
    n_clauses=20,    # clauses per class (half vote for, half against)
    T=15,            # vote margin
    s=3.9,           # specificity
    n_states=50,     # memory depth per action (default 128)
).to(device)
```

A `TsetlinMachine` is a `torch.nn.Module`. Its state is the integer buffer `model.ta_state`
of shape `(n_classes * n_clauses, 2 * n_features)`; `model.to(device)`, `model.state_dict()`
and `torch.save` work as usual. There are no gradient parameters — learning happens through
feedback, not back-propagation.

## 3. A manual training loop

The Tsetlin machine analogue of `loss.backward(); optimizer.step()` is one call:

```python
for epoch in range(60):
    model.train()
    perm = torch.randperm(len(x_train))
    for i in range(0, len(x_train), 10):
        idx = perm[i:i + 10]
        votes = model.update(x_train[idx].to(device), y_train[idx].to(device))
        # `votes` are the class sums *before* the update -> free batch metrics
    model.eval()
    with torch.no_grad():
        acc = (model(x_test.to(device)).argmax(1) == y_test.to(device)).float().mean()
    print(f"epoch {epoch + 1}: test accuracy {acc:.3f}")
```

`model(x)` returns the vote sums `(B, n_classes)` clamped to `[-T, T]` — treat them like
logits (`argmax` for the class, `model.predict_proba(x)` for probabilities). `model.train()`
/ `model.eval()` matter: while learning, a clause without literals is considered *True*; when
predicting it is *False*.

## 4. ... or use the Trainer

```python
trainer = tt.Trainer(model, batch_size=10, callbacks=[tt.EarlyStopping(patience=20)])
history = trainer.fit((x_train, y_train), epochs=100, val_data=(x_test, y_test))
trainer.evaluate((x_test, y_test))        # {'accuracy': 0.99...}
trainer.predict(x_test[:5])               # tensor([1, 0, 0, 1, 1])
history.plot()                            # matplotlib curve of val_accuracy
```

The `Trainer` accepts `(x, y)` tensors, a `torch.utils.data.Dataset` or a `DataLoader`, moves
batches to the model's device, and logs one line per epoch.

## 5. Read the model

```python
for rule in model.rules()[:5]:
    print(rule)
# IF x0 AND NOT x1 THEN 1
# IF x1 AND NOT x0 THEN 1
# IF x0 AND x1 THEN NOT 1
# ...
print(tt.interpret.explain(model, x_test[0]))
```

Continue with [choosing a device](devices.md), the [training guide](../guides/training.md)
or the [interpretability guide](../guides/interpretability.md).
