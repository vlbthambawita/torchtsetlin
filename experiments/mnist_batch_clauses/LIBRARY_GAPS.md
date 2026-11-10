# Library gaps found by mnist_batch_clauses

`src/torchtsetlin/**` is read-only in this programme (README R1). Defects or missing features
found while running it are recorded here, not fixed in place.

| # | Date | Gap | Impact | Status |
|---|---|---|---|---|
| MBC-1 | 2026-11-10 | Counting included literals per clause (`torch.sum(state >= N, dim=1)` in `F.commit_dense`, `F.apply_feedback*` and `_refresh_include`) materialised a bool mask plus an int64 copy: 9 bytes per automaton, a 7.5 GiB spike at flat 64k/class (640 000 × 1 568). Same root cause as `toolbox_cifar10` TB-1. Smoke: flat 32k b128 OOM on the RTX 3080, flat 64k B ≥ 64 OOM on the RTX 3090 (24 GB). | Would have removed 5 grid cells (flat 64k, B ≥ 64). | **fixed 2026-11-10 with the user's OK** (exception to R1): `F.count_included` counts in row blocks of 2^26 automata (~0.6 GiB scratch, same speed, shape-only loop so CUDA-graph safe), used at every site; `tests/test_functional.py::test_count_included_matches_the_one_shot_sum_in_row_blocks`; full suite 336 passed. Re-run smoke: all five flat-64k cells run on the 3090, peak 15.3 GB. Remaining: flat 64k still peaks at 15.3 GB vs ~3.2 GB of state (dense commit count buffers), so flat ≥ 32k stays 3090-only. |
