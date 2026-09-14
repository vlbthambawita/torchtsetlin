# Example notebooks

Three end-to-end walkthroughs, in increasing order of difficulty. Each one covers the same
arc — booleanize, train, evaluate, interpret — on a dataset that stresses a different part of
it. All were executed on an RTX 3090; the outputs in the committed notebooks are from those
runs.

| notebook | dataset | model | what it is really about | result | runtime |
|---|---|---|---|---|---|
| [`01_iris_tabular_tsetlin.ipynb`](01_iris_tabular_tsetlin.ipynb) | Iris (scikit-learn) | `TsetlinMachine` | Thermometer encoding, and reading the learned rules back as intervals in centimetres | 1.000 test accuracy | ~1 min (CPU) |
| [`02_mnist_tsetlin.ipynb`](02_mnist_tsetlin.ipynb) | MNIST (torchvision) | `TsetlinMachine`, `ConvTsetlinMachine` | Convolutional clauses, GPU throughput, and the batch-size / feedback-fidelity trade-off | 0.941 flat, 0.978 convolutional | ~4 min |
| [`03_cifar10_tsetlin.ipynb`](03_cifar10_tsetlin.ipynb) | CIFAR-10 (torchvision) | `ConvTsetlinMachine` | Booleanizing colour photographs — where the accuracy ceiling is actually set | 0.556 (10-class), 0.891 (vehicle/animal) | ~15 min |

Each notebook ends with a short takeaways section. The three experiments worth skipping to:

* **MNIST, section 7** — batched vs sequential feedback: identical accuracy (0.8761 vs 0.8756)
  at 65x the throughput, which is the empirical case for `torchtsetlin`'s batched default.
* **CIFAR-10, section 2** — the booleanization alone moves accuracy by ~12 points, more than any
  single Tsetlin hyper-parameter.
* **CIFAR-10, section 4** — the classes the model fails on are exactly the classes whose clauses
  need the most literals. The model diagnoses its own failure.

## Running them

```bash
pip install -e ".[dev]"          # torch, torchvision, scikit-learn, matplotlib, pandas
pip install jupyterlab
jupyter lab examples/notebooks
```

MNIST and CIFAR-10 download to `../../data/` (i.e. `data/` at the repo root, which is
gitignored) on first run. Everything runs on CPU as well — the Iris notebook is CPU-only by
design — but the two image notebooks are written for a GPU and will be roughly 10-30x slower
without one.

Datasets come from torchvision where they exist there (MNIST, CIFAR-10) and from scikit-learn
otherwise (Iris).
