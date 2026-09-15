# Benchmarks

All numbers below were measured with torchtsetlin 0.1.0, PyTorch 2.10 / CUDA 12.8 on a single
NVIDIA RTX 3090, with the whole Boolean dataset resident on the GPU. They are meant to show
the *throughput* of the batched update and that accuracy is on par with the literature after
a few epochs — not to be state-of-the-art results (those need hundreds of epochs and larger
clause budgets).

The systematic CPU-vs-GPU sweep — throughput against batch size, clause budget and feature
count, the model zoo, and where the time inside one `update()` goes — lives on its own page:
**[CPU vs GPU](cpu-vs-gpu.md)**. That page is generated end to end by
`benchmarks/run_benchmarks.sh`, so you can replace its numbers with your own hardware's.

## MNIST (`pixel > 0.3`, 60 000 training images)

| Model | batch | examples / s | s / epoch | test accuracy after 3 epochs | peak memory |
|---|---|---|---|---|---|
| `TsetlinMachine`, 500 clauses/class, T=25, s=10 | 32 | 12 900 | 4.6 | 95.3 % | 0.32 GB |
| `TsetlinMachine`, 500 clauses/class, T=25, s=10 | 100 | 39 000 | 1.5 | 95.5 % | 0.32 GB |
| `TsetlinMachine`, 2000 clauses/class, weighted, T=50, s=10 | 100 | 13 300 | 4.5 | 94.8 % | 1.07 GB |
| `ConvTsetlinMachine`, 10×10, 200 clauses/class, weighted, T=100, s=10 | 32 | 10 700 | 5.6 | 97.2 % | 0.35 GB |
| `ConvTsetlinMachine`, 10×10, 200 clauses/class, weighted, T=100, s=10 | 100 | 12 500 | 4.8 | 96.8 % | 0.35 GB |
| `TsetlinMachine`, 500 clauses/class, `feedback_mode="sequential"` | 1 | 500 | 120 | — | — |

The 500-clause flat machine reaches 95.8 % after 5 epochs; the literature reports ≈ 98 % for
2000 clauses per class after 400 epochs and ≈ 99 % for convolutional machines with 8000
clauses per class. The cost of a batched update is dominated by per-clause work
(`n_clauses × 2F` element-wise passes), so larger batches are almost free: going from batch
32 to 100 triples the throughput of the flat model.

## Noisy XOR: batch size vs accuracy

20 clauses, T=15, s=3.9, 100 states, 2000 training examples with 40 % label noise, 3 seeds
(CPU):

| batch size | epochs | updates | test accuracy (3 seeds) |
|---|---|---|---|
| 1 (sequential) | 40 | 80 000 | 0.954 / 0.955 / 0.993 |
| 10 | 100 | 20 000 | 0.942 / 0.993 / 0.931 |
| 50 | 200 | 8 000 | 0.993 / 0.965 / 0.877 |
| 200 | 400 | 4 000 | 0.788 / 0.771 / 0.889 |

See [Batched vs sequential feedback](concepts/batching.md) for the interpretation.

## Reproducing

The accuracy numbers above come from the example scripts:

```bash
python examples/mnist_flat.py --clauses 500 --T 25 --epochs 5 --batch-size 100
python examples/mnist_conv.py --clauses 200 --T 100 --epochs 3 --batch-size 32
python examples/noisy_xor.py --batch-size 10
```

The throughput numbers on [CPU vs GPU](cpu-vs-gpu.md) come from the benchmark driver, which
runs every suite, writes `benchmarks/results/cpu_vs_gpu.json`, and regenerates the figures,
tables and that page:

```bash
./benchmarks/run_benchmarks.sh                  # every suite, cpu + cuda:0 (~20-25 min)
./benchmarks/run_benchmarks.sh --quick          # ~1 min smoke test, docs untouched
./benchmarks/run_benchmarks.sh -s batch models  # a subset
./benchmarks/run_benchmarks.sh --plots-only     # re-render from the results already on disk
./benchmarks/run_benchmarks.sh --help           # every option
```
