# Example notebooks

Seven end-to-end walkthroughs. The first three cover the same arc — booleanize, train, evaluate,
interpret — on a dataset that stresses a different part of it. Notebooks 4 and 5 work through the
Tsetlin-machine **segmentation** literature paper by paper, reproducing what reproduces and
saying so when it doesn't. The sixth introduces the graph Tsetlin machine, and the seventh is a
walkthrough of `SegmentationTsetlinMachine` on generated street scenes. The notebooks are
stored without outputs; the results and runtimes below are from running them on the current
release (RTX 3090 / RTX 3080), and yours will differ a little from run to run.

| notebook | dataset | model | what it is really about | result | runtime |
|---|---|---|---|---|---|
| [`01_iris_tabular_tsetlin.ipynb`](01_iris_tabular_tsetlin.ipynb) | Iris (scikit-learn) | `TsetlinMachine` | Thermometer encoding, and reading the learned rules back as intervals in centimetres | ~0.98 test accuracy | ~1.5 min (CPU) |
| [`02_mnist_tsetlin.ipynb`](02_mnist_tsetlin.ipynb) | MNIST (torchvision) | `TsetlinMachine`, `ConvTsetlinMachine` | Convolutional clauses, GPU throughput, and the batch-size / feedback-fidelity trade-off | ~0.95 flat, ~0.98 convolutional | ~1 min |
| [`03_cifar10_tsetlin.ipynb`](03_cifar10_tsetlin.ipynb) | CIFAR-10 (torchvision) | `ConvTsetlinMachine` | Booleanizing colour photographs — where the accuracy ceiling is actually set | ~0.56 (10-class), ~0.89 (vehicle/animal) | ~2 min |
| [`04_convolutional_regression_tsetlin.ipynb`](04_convolutional_regression_tsetlin.ipynb) | synthetic (paper's Fig. 3) | `ConvRegressionTsetlinMachine` | **C-RTM (ICMLT 2021)** — how a TM emits a *number* instead of a class, and what its clauses say afterwards | 0.00 MAE noise-free with 8 weighted clauses (exact on 8 of 12 restarts); mask recovered exactly | ~6 min |
| [`05_ctm_unet_segmentation.ipynb`](05_ctm_unet_segmentation.ipynb) | synthetic CamVid-like scenes | `TsetlinMachine` (dense per-pixel head) | **CTM-UNet (ISTM 2025)** — the only TM segmentation paper, and which of its ingredients actually carry the result | ~0.996 pixel acc, ~0.97 mIoU | ~1 min |
| [`06_graph_tsetlin.ipynb`](06_graph_tsetlin.ipynb) | synthetic sequences, MNIST (torchvision) | `GraphTsetlinMachine` | **Graph TM (Granmo et al., 2025)**: deep clauses that read a node and its neighbours, traced with `explain_node`; at depth 1 it learns bit for bit like a convolutional machine | 1.000 on three A's and on run lengths (depth 1: 0.73 / 0.35) | ~1 min |
| [`07_segmentation_tsetlin.ipynb`](07_segmentation_tsetlin.ipynb) | synthetic street scenes (`make_scenes`) | `SegmentationTsetlinMachine` | Dense per-pixel labels: the row-prior floor, context from position literals and an OR-pooled pyramid, explaining one pixel, `patches_per_commit` and class-balanced feedback | ~0.95 mIoU with pyramid + position (row prior: 0.41) | ~3 min |

## The papers

Notebooks 4 and 5 correspond to the two papers in
[`source_documents/papers/03_segmentation/`](../../source_documents/papers/03_segmentation):

* **[C-RTM](04_convolutional_regression_tsetlin.ipynb)** — Abeyrathna, Granmo & Goodwin,
  *Convolutional Regression Tsetlin Machine*, ICMLT 2021
  ([doi](https://doi.org/10.1145/3468891.3468901)). Not a segmentation paper, but the closest
  published mechanism for continuous, patch-derived TM output.
* **[CTM-UNet](05_ctm_unet_segmentation.ipynb)** — Liu & Abeyrathna, *CTM-UNet*, ISTM 2025
  ([doi](https://doi.org/10.1109/ISTM67926.2025.00013)). The only paper that puts a Tsetlin
  machine into a segmentation architecture.

Both notebooks are written to be *usable as reviews*: they state each paper's claim, reproduce
it where it reproduces, and quantify the gap where it doesn't.

## The experiments worth skipping to

* **MNIST, section 7** — batched vs sequential feedback: the batched model is no worse (about
  0.87 vs 0.84 after 3 epochs on 10k images) and roughly 30x faster on an RTX 3090, which is the
  empirical case for `torchtsetlin`'s batched default.
* **CIFAR-10, section 2** — the booleanization alone moves accuracy by ~14 points, more than any
  single Tsetlin hyper-parameter.
* **CIFAR-10, section 4** — the classes the model fails on are exactly the classes whose clauses
  need the most literals. The model diagnoses its own failure.
* **C-RTM, section 5** — the clauses decode into statements about *named pixels*, and probing the
  model recovers the data-generating weight-mask exactly (when training reached zero MAE). The
  decomposition it finds is not the one the paper describes, which is the more interesting result.
* **C-RTM, section 7** — exact recovery is a basin you hit or miss (8 of 12 runs with 8 weighted
  clauses; none in our tests with the paper's 15 under the 0.5 weight and feedback rules), not a
  limit you converge to. Selecting restarts by *training* MAE finds it without touching test
  labels.
* **CTM-UNet, section 5** — the paper's headline 96.35% / 86.43% is binary sky-vs-rest; a
  baseline that knows only a pixel's *row* scores 0.895 / 0.724, and the U-Net it is compared
  against scores 99.99%.
* **CTM-UNet, section 6** — no gradient can cross a CTM block, so the paper's `Conv2D` front-end
  is a frozen random projection. Measured, it *costs* more than half of the pixel accuracy
  (about 0.95 -> 0.39).
* **CTM-UNet, section 7** — multi-scale context lifts building IoU from about 0.47 to about 0.95,
  entirely TM-natively (OR-pooled Boolean planes, thermometer-cast votes). That is what U-Net is
  for.

## Running them

**In Google Colab:** click the *Open in Colab* badge at the top of a notebook (or see the
[notebooks page](https://vlbthambawita.github.io/torchtsetlin/examples/notebooks/) of the
documentation). The first code cell installs torchtsetlin from PyPI when it runs in Colab; for
notebooks 2–7 pick *Runtime → Change runtime type → T4 GPU* first.

**Locally:**

```bash
pip install -e ".[dev]"          # torch, torchvision, scikit-learn, matplotlib, pandas
pip install jupyterlab
jupyter lab examples/notebooks
```

MNIST and CIFAR-10 download to `../../data/` (i.e. `data/` at the repo root, which is
gitignored) on first run. Everything runs on CPU as well — the Iris notebook is CPU-only by
design — but notebooks 2–5 are written for a GPU and will be roughly 10-30x slower without one.

Datasets come from torchvision where they exist there (MNIST, CIFAR-10), from scikit-learn
otherwise (Iris), and are generated in-notebook for 4 and 5 (the paper's own synthetic
construction, and CamVid-like scenes — CamVid itself is not redistributable here).
