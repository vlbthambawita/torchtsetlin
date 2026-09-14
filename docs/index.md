# torchtsetlin

**GPU-enabled, PyTorch-native Tsetlin machines.**

`torchtsetlin` implements the Tsetlin machine family — the propositional-logic learning
algorithm introduced by Ole-Christoffer Granmo — as ordinary `torch.nn.Module` objects. The
automata states live in tensors, so a model moves between CPU and GPU with `.to(device)`, is
saved with `state_dict()`, and learns from mini-batches with a single `model.update(x, y)`
call. Everything a Tsetlin machine workflow needs is included: Booleanization of data,
training and evaluation loops, metrics, rule extraction, feature importance and plots.

```python
import torchtsetlin as tt

x_train, y_train = tt.data.make_noisy_xor(5000, noise=0.4)          # Boolean features, int labels
x_test, y_test = tt.data.make_noisy_xor(2000, noise=0.0, seed=1)

model = tt.TsetlinMachine(n_features=12, n_classes=2, n_clauses=20, T=15, s=3.9).to("cuda")

trainer = tt.Trainer(model, batch_size=10, callbacks=[tt.EarlyStopping(patience=20)])
trainer.fit((x_train, y_train), epochs=100, val_data=(x_test, y_test))
print(trainer.evaluate((x_test, y_test)))       # {'accuracy': 0.99}
print(model.rules()[:3])                        # ['IF x0 AND NOT x1 THEN 1', ...]
```

## Why a Tsetlin machine?

* **Interpretable by construction.** Every clause is a conjunction of literals
  (`x0 AND NOT x1`) that votes for or against a class. The whole model *is* the explanation.
* **Bit-level efficient.** Inputs are Booleans and the learned state is a matrix of small
  integers; inference is an AND over included literals plus a vote count.
* **Universal approximator** with summation-based decisions, like a neural network, but
  learned without gradients through reward/penalty feedback to learning automata.

## What is in the box

| Area | Highlights |
|---|---|
| Models | `TsetlinMachine`, `CoalescedTsetlinMachine`, `RegressionTsetlinMachine`, `ConvTsetlinMachine` (2D), `Conv1dTsetlinMachine`, convolutional coalesced and regression variants |
| Learning options | vote margin `T`, specificity `s`, memory depth, boosted true-positive feedback, clause weights, clause size constraint, drop-clause / drop-literal, focused negative sampling, batched or exact sequential feedback |
| Data | `ThermometerEncoder`, `OneHotEncoder`, `BitPlaneEncoder`, `AdaptiveThresholdEncoder`, `ColorThermometerEncoder`, `HypervectorEncoder`, synthetic datasets, torchvision helpers |
| Training | `Trainer` (tensors, `Dataset` or `DataLoader`), callbacks (early stopping, checkpoints, CSV logs, schedules), `History` |
| Evaluation | accuracy, confusion matrix, precision/recall/F1, regression metrics, multi-label metrics, calibration, trustworthiness curves |
| Interpretation | rules, clause activity and precision, closed-form global/local feature importance, per-example explanations |
| Visualisation | memory plots, automata heat-maps, convolutional clause patches, confusion matrices, vote distributions |

## Where to go next

* [Installation](getting-started/installation.md) and the [quick start](getting-started/quickstart.md).
* [How a Tsetlin machine learns](concepts/tsetlin-machine.md) — the algorithm, in the vocabulary of the book *An Introduction to Tsetlin Machines*.
* [Batched vs sequential feedback](concepts/batching.md) — what `model.update` does with a mini-batch and how to choose a batch size.
* The [API reference](api/models.md).

## Acknowledgements

The algorithms implemented here follow the Tsetlin machine literature by Granmo and
colleagues (the original TM, the convolutional TM, the weighted TM, the coalesced TM,
drop-clause, clause-size constraints, closed-form interpretability, TM composites and
confidence) and the textbook *An Introduction to Tsetlin Machines* (tsetlinmachine.org).
See the concept pages for references.
