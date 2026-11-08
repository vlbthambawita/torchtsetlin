# Notebooks

End-to-end walkthroughs as Jupyter notebooks. The pages below show the code and the explanations;
the notebooks are stored without outputs, so run one to see its results. The easiest way is
**Google Colab**: open a notebook with its badge, and the first cell installs torchtsetlin from
PyPI. The datasets download on first use.

| Notebook | Open | Data | Model | What it is about | Runtime |
|---|---|---|---|---|---|
| [Iris, a readable model](notebooks/01_iris_tabular_tsetlin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/01_iris_tabular_tsetlin.ipynb) | Iris (scikit-learn) | `TsetlinMachine` | Thermometer encoding, and reading the learned rules back as intervals in centimetres | ~1.5 min (CPU) |
| [MNIST on the GPU](notebooks/02_mnist_tsetlin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/02_mnist_tsetlin.ipynb) | MNIST (torchvision) | `TsetlinMachine`, `ConvTsetlinMachine` | Convolutional clauses, GPU throughput, and the batch-size / feedback-fidelity trade-off | ~1 min |
| [CIFAR-10](notebooks/03_cifar10_tsetlin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/03_cifar10_tsetlin.ipynb) | CIFAR-10 (torchvision) | `ConvTsetlinMachine` | Booleanizing colour photographs, and where the accuracy ceiling is actually set | ~2 min |
| [Convolutional regression (C-RTM)](notebooks/04_convolutional_regression_tsetlin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/04_convolutional_regression_tsetlin.ipynb) | synthetic (the paper's Fig. 3) | `ConvRegressionTsetlinMachine` | C-RTM (ICMLT 2021): how a TM outputs a number, and what its clauses say afterwards | ~6 min |
| [CTM-UNet segmentation](notebooks/05_ctm_unet_segmentation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/05_ctm_unet_segmentation.ipynb) | synthetic CamVid-like scenes | `TsetlinMachine` (dense per-pixel head) | CTM-UNet (ISTM 2025): which of its ingredients carry the result | ~1 min |
| [Graph Tsetlin machine](notebooks/06_graph_tsetlin.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vlbthambawita/torchtsetlin/blob/main/examples/notebooks/06_graph_tsetlin.ipynb) | synthetic sequences, MNIST | `GraphTsetlinMachine` | Granmo et al. (2025): deep clauses over graphs, reading them back, and depth 1 = convolution | ~1 min |

## Running in Colab

* For notebooks 2–6 choose *Runtime → Change runtime type → T4 GPU* before running. A T4 is
  slower than the RTX 3090 and 3080 the runtimes above were measured on, so expect them to take longer.
  Notebook 1 runs on the CPU.
* The Colab links open the notebooks from the `main` branch on GitHub.
* Colab already has torch, torchvision, scikit-learn, matplotlib and pandas; the setup cell only
  adds torchtsetlin.

## Running locally

```bash
pip install torchtsetlin jupyterlab        # or pip install -e ".[dev]" in a checkout
jupyter lab examples/notebooks
```

MNIST and CIFAR-10 download to `data/` at the repository root on first run. Every notebook also
runs on the CPU, but notebooks 2–6 are written for a GPU and are much slower without one.
The source files are in
[`examples/notebooks/`](https://github.com/vlbthambawita/torchtsetlin/tree/main/examples/notebooks).
