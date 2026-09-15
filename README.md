# torchtsetlin

**GPU-enabled, PyTorch-native Tsetlin machines** — models, data preparation, training,
evaluation, interpretation and visualisation in one package.

[![CI](https://github.com/vlbthambawita/torchtsetlin/actions/workflows/ci.yml/badge.svg)](https://github.com/vlbthambawita/torchtsetlin/actions)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://vlbthambawita.github.io/torchtsetlin/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A [Tsetlin machine](https://tsetlinmachine.org) learns propositional rules
(`IF x0 AND NOT x1 THEN class 1`) with teams of learning automata instead of gradients.
`torchtsetlin` implements the family as ordinary `torch.nn.Module`s: the automata live in
tensors, so a model moves between CPU and GPU with `.to(device)`, is saved with
`state_dict()`, and learns from mini-batches with a single `model.update(x, y)` call.

```python
import torchtsetlin as tt

x_train, y_train = tt.data.make_noisy_xor(5000, noise=0.4)          # Boolean features, int labels
x_test, y_test = tt.data.make_noisy_xor(2000, noise=0.0, seed=1)

model = tt.TsetlinMachine(n_features=12, n_classes=2, n_clauses=20, T=15, s=3.9).to("cuda")

# PyTorch-style loop: update() is the analogue of loss.backward(); optimizer.step()
for epoch in range(50):
    model.train()
    for i in range(0, 5000, 10):
        model.update(x_train[i:i+10].cuda(), y_train[i:i+10].cuda())
    model.eval()
    acc = (model(x_test.cuda()).argmax(1) == y_test.cuda()).float().mean()

print(model.rules()[:2])   # ['IF x0 AND NOT x1 THEN 1', 'IF x1 AND NOT x0 THEN 1']
```

Or with the built-in trainer:

```python
trainer = tt.Trainer(model, batch_size=10, callbacks=[tt.EarlyStopping(patience=20)])
trainer.fit((x_train, y_train), epochs=100, val_data=(x_test, y_test))
trainer.evaluate((x_test, y_test))          # {'accuracy': 0.99}
```

## Features

| | |
|---|---|
| **Models** | `TsetlinMachine` (multi-class, optional integer clause weights), `CoalescedTsetlinMachine` (shared clauses; multi-class or multi-label), `RegressionTsetlinMachine`, `ConvTsetlinMachine` / `Conv1dTsetlinMachine` and convolutional coalesced / regression variants |
| **Learning** | vote margin `T`, specificity `s`, memory depth, boosted true-positive feedback, clause-size constraint, drop-clause / drop-literal, focused negative sampling, **batched** (GPU-friendly) or **exact sequential** feedback |
| **Data** | thermometer, one-hot, bit-plane, adaptive-threshold, colour-thermometer and hypervector encoders; synthetic datasets; torchvision helpers |
| **Training** | `Trainer` for tensors / `Dataset` / `DataLoader`, callbacks (early stopping, checkpoints, CSV logs, hyper-parameter schedules), `History` |
| **Evaluation** | accuracy, confusion matrix, precision/recall/F1, regression and multi-label metrics, calibration, trustworthiness curves |
| **Interpretation** | rule extraction, clause activity/precision, closed-form global & local feature importance, per-example explanations |
| **Visualisation** | memory plots, automata heat-maps, convolutional clause patches, confusion matrices, vote distributions |

## Installation

```bash
pip install torch            # pick the CUDA/CPU build from pytorch.org
pip install torchtsetlin     # + optional extras: [viz] [vision] [sklearn] [docs] [all]
```

From source: `pip install -e ".[dev]"`, then `pytest` and `mkdocs serve`.

## How learning works (in one paragraph)

Each clause is an AND of literals (`x_k` or `NOT x_k`); every (clause, literal) pair has a
Tsetlin automaton whose integer state decides whether the literal is *included*. For a
training example, clauses of the true class receive **Type I** feedback (memorise the
example's True literals, forget the rest) and clauses of a random other class receive
**Type II** feedback (add a False literal so the clause stops matching), each with a
probability controlled by the vote margin `T`. `torchtsetlin` evaluates a mini-batch with one
matrix product, counts the feedback events per (clause, literal) with another, and turns the
counts into binomial state transitions — the whole update is a handful of tensor ops. See the
[concept pages](https://vlbthambawita.github.io/torchtsetlin/concepts/tsetlin-machine/) for
details and references.

## Documentation

* Getting started, concepts, guides, examples and the full API reference:
  <https://vlbthambawita.github.io/torchtsetlin/> (or `mkdocs serve` locally).
* Runnable scripts in [`examples/`](examples): Noisy XOR, MNIST (flat and convolutional),
  tabular data with thermometer encoding, regression.
* Worked notebooks in [`examples/notebooks/`](examples/notebooks): Iris (rules you can read),
  MNIST (convolutional clauses and GPU throughput) and CIFAR-10 (booleanizing colour images),
  plus two that work through the Tsetlin-machine segmentation literature — the Convolutional
  Regression TM (ICMLT 2021) and CTM-UNet (ISTM 2025), the latter building a dense per-pixel
  Tsetlin segmentation head.

## Citation

If you use torchtsetlin in research, please cite the Tsetlin machine papers whose algorithms
you rely on (see the documentation) and this package:

```bibtex
@software{torchtsetlin,
  author = {Thambawita, Vajira},
  title  = {torchtsetlin: GPU-enabled, PyTorch-native Tsetlin machines},
  year   = {2026},
  url    = {https://github.com/vlbthambawita/torchtsetlin}
}
```

## License

MIT — see [LICENSE](LICENSE).
