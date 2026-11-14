# Library gaps found by cifar10_batch_clauses

`src/torchtsetlin/**` is read-only in this programme (README R1). Defects or missing features found
while running it are recorded here, not fixed in place.

Known before the first run (worked around in the harness, D9):

| # | Gap | Workaround | Status |
|---|---|---|---|
| — | Conv3d patch blocking falls to single-patch blocks and off the fused CUDA path for large C (fmnist_bitplanes FB-1, GitHub #17) | `max_chunk_elements` sized for one patch block (conv and conv3 alike) | open |
| — | Flat models need ~15 GB per 10⁹ automata in batched GPU training (mnist_batch_clauses MBC-1 follow-up, GitHub #18) | flat limited to 8k clauses/class | open |
