# Library gaps found in fmnist_bitplanes

`src/torchtsetlin/**` is read-only in this programme (README R1); gaps are recorded here, not fixed.

## FB-1: Conv3d patch blocking falls off the fused CUDA path (open, worked around; GitHub #17)

**Found** 2026-11-13 in the smoke test (torchtsetlin 0.7.1, RTX 3080).
`_Conv3dMixin._patch_block` splits the patch axis whenever
`3·P·C + 2·C·L + P·L > max_chunk_elements` (default 2**27). For the Fashion-MNIST bit-plane volumes
`(1, 8, 28, 28)` with a `(kd, 10, 10)` window this is true for every model in the plan:

| kd | C/class | P | literals | patch block (default budget) |
|---|---|---|---|---|
| 2 | 2 000 | 2 527 | 484 | 1 893 (2 blocks) |
| 4 | 2 000 | 1 805 | 880 | 1 623 (2 blocks) |
| 2 | 8 000 | 2 527 | 484 | 231 (11 blocks) |
| 4 | 8 000 | 1 805 | 880 | **1** (1 805 blocks) |

At 8 000 clauses/class and kd = 4 the term `2·C·L` alone (1.4·10⁸) exceeds the budget, so the
formula's `max(1, …)` gives blocks of a single patch: one smoke epoch on 2 000 images had not finished
after 13 minutes, against ~25 s at kd = 2. More than one block also means the update leaves the fused
(events-mode) CUDA path, which `CLAUDE.md` says needs a single block.

**Workaround (harness):** `Conv3dTsetlinMachine(..., max_chunk_elements=2**30)`. Every conv3 model of the
plan then has one block; measured 140–175 ms per update of 32 at C ∈ {2 000, 8 000}, kd ∈ {2, 4}, peak
< 1.7 GB (shared GPU, so an upper bound). The records carry `max_chunk_elements` and `patch_blocks`.

**Suggested library fix:** the budget mixes the dense-evaluation intermediates (`2·C·L`, `3·P·C`) with
the fused path, which does not allocate them; and the fallback should never shrink to a single patch
(e.g. fall back to the fused chunk size, or warn). Not fixed here (R1).
