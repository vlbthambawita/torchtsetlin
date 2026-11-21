# Library gaps found by cifar10_graph_tm

`src/torchtsetlin/**` is read-only in this programme. Defects and missing features of the
library found here are recorded in this file, not fixed.

## GT-1: the bit-packed clause kernel scans every literal word (cost, not correctness)

Found 2026-11-21 (torchtsetlin 0.7.3, RTX 3090) while checking the paper-setting GraphTM at full
size: 80 000 clauses, 625 nodes, 1 778 node features (3 556 literals), batch 16.

- One update takes about 1.36 s; 98 % of it is `_triton.conv_eval` (1.34 s), the commit 0.02 s.
- A `ConvCoalescedTsetlinMachine` with the same windows on the same 27 planes (1 776 features)
  takes 1.30 s per update, so this is the convolutional kernel, not the graph code.
- At 5 000 clauses the same update takes 47 ms: 29 × the time for 16 × the clauses.
- The time per update stayed at 1.36 s over the first 120 updates while the mean clause length
  changed, which fits a kernel whose work grows with nodes × clauses × literal words whatever
  the clause length (clauses hold about 32 literals under the soft budget). The kernel source
  was not read here; this is an inference from the timings.

Effect here: the paper-setting GraphTM needs about 2.4 h per epoch (70 h for 30 epochs on a
3090). If the inference is right, a kernel that visits only the included literals of a clause
would make wide node features affordable. Numbers are from functional checks of 5 to 120
updates, not from a benchmark run.

## Notes (not defects)

- There is no public way to build a padded `GraphBatch` on the GPU without the validating
  constructor, so `code/graphs.py` keeps one padded template per batch size and replaces its
  `node_features` (public attributes only).
