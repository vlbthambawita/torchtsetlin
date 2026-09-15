# Example scripts

Every script is self-contained, takes `--device` (CUDA when available, otherwise CPU) and
prints what the model *learned*, not just its score. The worked notebooks live in
[`notebooks/`](notebooks).

## Flat models

| script | model | data | what it shows |
|---|---|---|---|
| [`noisy_xor.py`](noisy_xor.py) | `TsetlinMachine` | `make_noisy_xor` | the benchmark of the original paper; `--sequential` for the exact per-example algorithm |
| [`tabular_breast_cancer.py`](tabular_breast_cancer.py) | `TsetlinMachine` | scikit-learn breast cancer | thermometer encoding of continuous columns, rules in the original units |
| [`regression_sine.py`](regression_sine.py) | `RegressionTsetlinMachine` | synthetic 1-D | a TM that emits a number |
| [`mnist_flat.py`](mnist_flat.py) | `TsetlinMachine` | MNIST (torchvision) | 784 pixels as flat features, checkpoints and a classification report |

## Convolutional models

A convolutional Tsetlin machine slides its clauses over the input and fires if **any** window
matches, so one clause covers every position. These four cover the whole convolutional API —
2-D and 1-D, classification, shared clause pools and regression.

| script | model | data | result | runtime |
|---|---|---|---|---|
| [`shapes_conv.py`](shapes_conv.py) | `ConvTsetlinMachine` | `make_shapes` (12×12, generated) | 1.000 test accuracy on every seed tried, against 0.55–0.91 for a flat machine with the same clause budget | 6 s |
| [`conv1d_ramps.py`](conv1d_ramps.py) | `Conv1dTsetlinMachine` | noisy 1-D signals (generated) | 0.996 test accuracy; the clauses print as the ramp they detect | 5 s |
| [`conv_regression_blobs.py`](conv_regression_blobs.py) | `ConvRegressionTsetlinMachine` | squares of varying size (generated) | MAE 0.20 side-length units, R² 0.96 | 24 s |
| [`mnist_conv.py`](mnist_conv.py) | `ConvTsetlinMachine`, `ConvCoalescedTsetlinMachine` | MNIST (torchvision) | 0.979 with the defaults (200 clauses/class); 0.985 with `--coalesced --clauses 1000 --T 500` | 80 s / 50 s |

Timings are for the default arguments on an RTX 3090; the three generated-data scripts are
comfortable on a CPU as well (`--device cpu`).

### The flags that matter

* `--patch` / `--stride` — the window and how far it moves. A 28×28 image with 10×10 windows
  and stride 1 has 361 windows per image; stride 2 leaves 100, which is about 1.5× faster per
  epoch and costs roughly a point of accuracy.
* `--position-encoding` — appends thermometer-encoded window coordinates so a clause can
  demand *where* it matches. It **hurts** when the pattern really can appear anywhere:
  `shapes_conv.py` scores 1.000 without it and about 0.5 with it. Digits are not like that —
  on MNIST the coordinates are worth about a point (0.972 against 0.959 after 3 epochs) —
  which is why `mnist_conv.py` has them on and offers `--no-position-encoding` instead.
* `--coalesced` (`mnist_conv.py`) — one clause pool shared by all classes with an integer
  weight per (clause, class), instead of `n_clauses` clauses *per class*. On MNIST that is
  0.985 from 1000 shared clauses against 0.979 from 2000 per-class ones, in less time.

### Reading a clause

All four scripts print their clauses as a picture of the window, `#` = pixel must be on,
`.` = must be off, blank = don't care. `conv1d_ramps.py` prints the thermometer bit planes of
one window, so a rising ramp shows up as a literal staircase:

```
clause 51 of 'rising': fires on 12% of signals, 100% of them class 'rising'
  ........
  .......#
   .....##
  #....###
  #...####
  #..#####
  #.######
```

`model.clause_patch(i)` returns that map as a tensor, `model.clause_region(i)` the window
coordinates the clause allows, and `tt.viz.plot_conv_clauses(model, range(32))` draws a grid
of them (`--plot` in `shapes_conv.py` and `mnist_conv.py`).

## Running them

```bash
pip install -e ".[dev]"     # torch, torchvision, scikit-learn, matplotlib
python examples/shapes_conv.py --flat
python examples/mnist_conv.py --clauses 2000 --T 2500 --epochs 30
```

MNIST downloads to `--root` (default `./data`, gitignored) on first use.

The same material in the documentation site: [Convolution on generated
data](../docs/examples/convolution.md) and [MNIST](../docs/examples/mnist.md) walk through
these scripts, [`docs/concepts/convolution.md`](../docs/concepts/convolution.md) has the
algorithm, and [`docs/benchmarks.md`](../docs/benchmarks.md) the throughput numbers.
