# Example notebooks

Five end-to-end walkthroughs. The first three cover the same arc — booleanize, train, evaluate,
interpret — on a dataset that stresses a different part of it. The last two work through the
Tsetlin-machine **segmentation** literature paper by paper, reproducing what reproduces and
saying so when it doesn't. All were executed on an RTX 3090; the outputs in the committed
notebooks are from those runs.

| notebook | dataset | model | what it is really about | result | runtime |
|---|---|---|---|---|---|
| [`01_iris_tabular_tsetlin.ipynb`](01_iris_tabular_tsetlin.ipynb) | Iris (scikit-learn) | `TsetlinMachine` | Thermometer encoding, and reading the learned rules back as intervals in centimetres | 1.000 test accuracy | ~1 min (CPU) |
| [`02_mnist_tsetlin.ipynb`](02_mnist_tsetlin.ipynb) | MNIST (torchvision) | `TsetlinMachine`, `ConvTsetlinMachine` | Convolutional clauses, GPU throughput, and the batch-size / feedback-fidelity trade-off | 0.941 flat, 0.978 convolutional | ~4 min |
| [`03_cifar10_tsetlin.ipynb`](03_cifar10_tsetlin.ipynb) | CIFAR-10 (torchvision) | `ConvTsetlinMachine` | Booleanizing colour photographs — where the accuracy ceiling is actually set | 0.556 (10-class), 0.891 (vehicle/animal) | ~15 min |
| [`04_convolutional_regression_tsetlin.ipynb`](04_convolutional_regression_tsetlin.ipynb) | synthetic (paper's Fig. 3) | `ConvRegressionTsetlinMachine` | **C-RTM (ICMLT 2021)** — how a TM emits a *number* instead of a class, and what its clauses say afterwards | 0.00 MAE noise-free; mask recovered exactly | ~25 min |
| [`05_ctm_unet_segmentation.ipynb`](05_ctm_unet_segmentation.ipynb) | synthetic CamVid-like scenes | `TsetlinMachine` (dense per-pixel head) | **CTM-UNet (ISTM 2025)** — the only TM segmentation paper, and which of its ingredients actually carry the result | 0.996 pixel acc, 0.968 mIoU | ~8 min |

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

* **MNIST, section 7** — batched vs sequential feedback: identical accuracy (0.8761 vs 0.8756)
  at 65x the throughput, which is the empirical case for `torchtsetlin`'s batched default.
* **CIFAR-10, section 2** — the booleanization alone moves accuracy by ~12 points, more than any
  single Tsetlin hyper-parameter.
* **CIFAR-10, section 4** — the classes the model fails on are exactly the classes whose clauses
  need the most literals. The model diagnoses its own failure.
* **C-RTM, section 5** — the clauses decode into statements about *named pixels*, and probing the
  model recovers the data-generating weight-mask exactly. The decomposition it finds is not the
  one the paper describes, which is the more interesting result.
* **C-RTM, section 7** — exact recovery is a basin you hit or miss (9 of 12 runs), not a limit you
  converge to. Selecting restarts by *training* MAE finds it without touching test labels.
* **CTM-UNet, section 5** — the paper's headline 96.35% / 86.43% is binary sky-vs-rest; a
  baseline that knows only a pixel's *row* scores 0.895 / 0.724, and the U-Net it is compared
  against scores 99.99%.
* **CTM-UNet, section 6** — no gradient can cross a CTM block, so the paper's `Conv2D` front-end
  is a frozen random projection. Measured, it *costs* 5.5 points of pixel accuracy.
* **CTM-UNet, section 7** — multi-scale context lifts building IoU from 0.41 to 0.95, entirely
  TM-natively (OR-pooled Boolean planes, thermometer-cast votes). That is what U-Net is for.

## Running them

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
