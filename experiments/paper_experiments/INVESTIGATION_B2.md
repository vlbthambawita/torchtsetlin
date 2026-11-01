# Investigation: B2 (2-D Noisy XOR, conv) results for torchtsetlin 0.6.0

Opened 2026-11-01 after the full GPU run (`859e229`, `REPORT.md`). Library changes need the
user's go-ahead, and so does any re-run of timing/seed cells (see `PLAN.md`).

## What the run shows

| tt mode | 0.5.0 (seed 0) | 0.5.1 (seed 0, 1005 epochs) | 0.6.0 (5 seeds) | other libraries |
|---|---|---|---|---|
| seq | 100.0 | 100.0 | **78.52 ± 13.40** | 98.8-99.7 |
| batch (16) | 100.0 | – | 99.61 ± 0.27 | |
| b100 | 93.6 | – | **92.47 ± 3.83** | |

Two separate problems:

1. **seq is a 0.6.0 regression.** Seed 0 on 0.5.1 and 0.6.0 gives the same accuracy for epochs
   1-4 (53.7, 75.3, 65.7, 57.2). After that 0.5.1 keeps learning to 100%, while every 0.6.0 seed
   holds one value for thousands of epochs (seed 0: 57.2 from epoch 4 to 5000).
2. **b100 is old.** 0.5.0 already gave 93.6, so this is not a regression. It is a question of how
   faithful batched feedback is at batch 100 on this tiny conv task.

## Problem 1: stale bit-packed include mask after CUDA-graph replays (gap G25)

### Mechanism (confirmed 2026-11-01)

1. Sequential mode on CUDA runs in 64-example blocks that are captured as CUDA graphs (G19) and
   replayed from their third use on.
2. The conv model predicts with `_triton.conv_eval`, which reads `include_bits()` (G20/G24).
3. `include_bits()` repacks the mask only when the Python flag `_bits_fresh` is False. The
   sequential commit clears that flag (`base.py`, the fused `F.commit_single_example` path).
4. A flag set in Python is not part of the graph: `graph.replay()` updates `ta_state` and
   `include` on the device, but never clears `_bits_fresh`.
5. Once every update block replays (from epoch 4 or 5), `predict()` keeps using the bits packed
   at the last eager update. **The model keeps learning, but its predictions stay frozen.**

The check is `code/diag_b2_stale_bits.py` (B2, seed 0, the benchmark's batch-1000 calls; run it
from this folder with the tt Python). Up to epoch 4 the bits were fresh. From epoch 5 on,
`include_bits()` differed from `pack_bits(include)` after every epoch, and predict and
forced refresh disagreed: for example 61.9 against 88.6, and 80.6 against 93.6.

### Why only conv with sequential mode on CUDA

- Batched conv on CUDA commits with the Triton kernel. That kernel writes the bits inside the
  graph, so the flag stays correct.
- Flat models never read `include_bits()` when they predict.
- On the CPU there are no graphs.
- G19's check ("replays bit-identical to eager") compared the learning state, not predictions
  read through the cache.

### Steps

1. **Regression test, written first so it fails on 0.6.0.** Add it to `tests/test_triton.py`:
   - train a small `ConvTsetlinMachine(feedback_mode="sequential")` on CUDA for enough calls that
     the 64-blocks replay;
   - assert `include_bits()` equals `pack_bits(include)`;
   - assert `predict()` equals the result with `cuda_graphs = False` under the same seed.

   Repeat the test for a coalesced and a weighted conv model, since they share the path.
2. **Fix.** After every `graph.replay()`, restore the Python-side state the captured block left
   behind. The minimum is to record `_bits_fresh` at the end of capture and set it after each
   replay. That gives False for sequential blocks and True where the Triton commit wrote the
   bits.
3. **Audit the rest of `_update_block` for Python-side state.** Collect every `self.__dict__` or
   attribute write reachable from `update()`: `_bits_fresh`, `_acc_dirty`, update counters,
   schedules and caches. Each one either becomes device state or gets replayed as above. Add the
   rule to CLAUDE.md under "CUDA graphs".
4. **Side issue: graph cache churn.** One key stays in "watch" for the whole run; the
   diagnostic's cache always holds one `watch` entry, so a block shape keeps running eagerly or
   is being evicted. Check `_GRAPH_CACHE = 4` against the block shapes that sequential calls
   produce: 64, the tail of 1000 (40), and the tail of 2500 split into 1000s (52). Fixing this is
   speed only, not correctness.
5. **Confirm on B2 without touching `results/`.**
   - Run the 0.6.0 diagnostic and the fixed build for seed 0, about 50 epochs each.
   - The fixed build should match `TORCHTSETLIN_CUDA_GRAPHS=0`.
   - It should also follow 0.5.1's curve (about 99% by epoch 50).
6. **Release 0.6.1** with the fix, the test and a CHANGELOG line. 0.6.0 predictions from
   sequential conv models trained on CUDA are affected. The workaround on 0.6.0 is
   `model.cuda_graphs = False` or `TORCHTSETLIN_CUDA_GRAPHS=0`.
7. **Re-run the affected cells on 0.6.1, with the user's go-ahead.**
   - These are the B2 `tt-cuda_seq` cells only: timing s0 plus seeds s1-s4. That is 5 runs of about
     10-20 min each.
   - No other cell trains a conv model in sequential mode: B7/B8 have no seq mode, and B1/B3/B10
     are flat.
   - Move the 0.6.0 B2 seq records aside to `results/timing_v0.6.0_tt/` and
     `results/seeds_v0.6.0_tt/`.
   - Regenerate `REPORT.md`.
   - Timing should barely change: the fix adds one repack of 40×1 words per predict.
8. **Check the library benchmark.** `benchmarks/bench_device.py`'s `feedback` suite is flat, so the
   0.6.0 docs figures are unaffected. Confirm this, and say so in the 0.6.1 notes.

## Problem 2: b100 on B2 (fidelity of batched feedback, not a regression)

The CLAUDE.md note "batch 10-50 matches sequential on Noisy XOR, 200 degrades" was measured on
flat Noisy XOR. B2 is a 4×4 conv task with 40 clauses and T = 60. At batch 100 each clause takes
about 100 examples' worth of feedback per commit, so the same over-shooting is to be expected.

1. **Characterise it.** Run 5 seeds × 500 epochs at batches 16, 32, 50, 100 and 200 with 0.6.1.
   This is a diagnostic: write it to `results/diag_b2_batch/`, not `results/seeds/`.
   - If accuracy falls with batch size the way it does on flat Noisy XOR, it is the known
     fidelity cost.
   - If b100 sits apart from that trend, look at conv-specific feedback: one random patch per
     (example, clause) at T = 60.
2. **For the paper,** report b100 on B2 as a finding about batch fidelity on a tiny conv task,
   next to batch 16 (99.6) and seq. Do not tune it away (rule P7 / D10).

## Not ours

pyTM-CUDA on B9 (81.27 ± 4.45, recall 72.8 ± 11.5) and on B8 (88.37 ± 2.08) varies across
seeds, while tmu and tt do not. Look at the per-seed records before the paper says anything
about it. One bad seed is likely, but this has not been checked yet.
