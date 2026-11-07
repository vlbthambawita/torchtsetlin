# Graph Tsetlin machine: CUDA fast paths (graph-tm.md, step 9)

*Status: **phases 0–5 implemented**, 2026-11-07; phase 6 (optional kernels, ragged storage)
open, not needed by the measurements so far. Parent plan: `plans/graph-tm.md` (steps 1–8 done,
commits `a416264`, `01305f0`). Tracks GitHub issue #8. See "Implementation notes" at the end.*

## TL;DR

`GraphTsetlinMachine` opts out of every CUDA fast path the other models use: events mode
(`_triton.fused_commit`), fused sequential commits (`F.commit_single_example`) and CUDA graphs
(`_events_eligible`, `_fuse_single_examples`, `_graphs_ok` all return `False`). The cost shows
most where launches dominate: on an RTX 3090, sequential updates take ~5.6 ms per graph, against
~0.2 ms per graph batched (smoke numbers, `plans/graph-tm.md` implementation notes).

The existing kernels need **no change to join**:

* `fused_commit` in its `CONV` mode already takes, per (example, clause), the index of the
  random matching patch for Type Ia and for Type II, plus a `(B, P, 2F)` literal tensor. A graph
  is the same thing with nodes for patches and `[H⁰ | … | H^{D−1} | 1 − …]` as the literal row
  (Lemma 2 of the parent plan).
* `commit_single_example` already takes `(C, 2F)` per-clause literal rows (the conv fused
  sequential path).

The work is mostly on the graph side. In order:

1. Make the learning update **sync-free**.
2. Give `GraphBatch` a **fixed-shape layout**, so a slice has a static shape.
3. Then switch the three opt-outs back on.

A Triton node-evaluation kernel and ragged storage come last, and only if the profile asks for
them.

## 1. Where the time goes and what blocks each path

| Path | What it needs | What the graph code does now |
|---|---|---|
| Events mode (batched, CUDA) | `_feedback_counts` records `acc.events = (lit, sel_i, sel_ii, p_i, p_ii)` with no host sync; `_records_events = True` | Draws nodes inside a loop over `torch.nonzero(sel)` blocks (host sync, data-dependent shapes); sums with a sparse COO product (coalesce syncs) |
| Fused sequential (CUDA) | `acc.single = (rows_i, rows_ii, fire_i, nofire_i, fire_ii)`, `(C, 2F)` rows; `_records_single = True` | Not recorded; sequential steps take the row-sparse commit (`F.apply_feedback_rows`, host syncs) |
| CUDA graphs | A sync-free `_update_block`; `xb[start:start+step]` with a static shape; `gx.copy_(xs)`; a `_graph_key` from shapes | `GraphBatch.__getitem__` trims padding (`int(n_nodes.max())`), checks bounds (`int(idx.min())`) and rebuilds edge offsets (`cumsum`, `int(counts.sum())`): syncs, and the shape depends on the data |

Other host syncs to find and remove (audited in Phase 0):

* `chunk_indices(nz[0].numel(), …)`;
* `_coerce_input` checks;
* `if enc.src.numel() == 0` (a shape, so no sync; fine);
* `include_count > 0` masks (fine).

## 2. Design

### 2.1 Fixed-shape `GraphBatch` (Phase 1)

Add an opt-in padded layout:

```python
g = GraphBatch.from_sequences(seqs, sym).pad(max_nodes=P, max_edges=E)   # or pad() = batch max
```

* Nodes: already padded to `P` (`node_features (B, P, F)`, `node_mask`). `pad` fixes `P`
  instead of the batch maximum.
* Edges: stored as `(B, E, 2)` local indices plus `edge_type (B, E)` and `edge_mask (B, E)`, in
  place of the flat `(ΣE, 2)` list with `edges_per_graph`. `flat_edges()` for a padded batch is
  pure arithmetic, with no `cumsum` to host:

  ```python
  src = (arange(B)[:, None] * P + edges[..., 0]).reshape(-1)
  ```

  Messages from padding edges are masked: `sent[src] * edge_mask`.
* `__getitem__` on a padded batch keeps `P` and `E` (no trimming, no host bounds check), so
  `g[i:i+1]` and `g[perm[i:i+25]]` have static shapes and are sync-free.
* `GraphBatch.copy_(other)` (in place, same layout) is for the CUDA-graph input buffer.

The unpadded layout stays the default and keeps today's behaviour. Padding is what a dataset with
a fixed graph shape (images: every graph has the same `P` and `E`; fixed-length sequences) gets
for free. For uneven graphs it costs `Σ (P − n_i)` wasted node rows, which is why Phase 6 exists.

### 2.2 Sync-free node draw (Phase 2)

Replace the `nonzero` loop by dense draws over `(B, P, C)`:

```python
key = torch.rand(B, P, C) * matches          # 0 where the clause does not match
p_i  = key.argmax(dim=1)                      # (B, C) random matching node, Type Ia
p_ii = (torch.rand(B, P, C) * matches).argmax(dim=1)   # independent one, Type II
```

* With `shared_node_draw=True`, `p_ii = p_i`.
* This is the draw `_ConvMixin` used before `conv_eval` took it over, which CLAUDE.md warns
  against on the *prediction* path. It is only on the learning path here, where `matches`
  already exists as a `(B, P, C)` tensor. A hashed-key draw (as `conv_eval` does) saves the two
  `rand` passes if the profile shows them (Phase 5).
* The random stream differs from the current loop, so this path is not bit-identical to the
  eager one: it is equal in distribution. The eager `W @ H` path stays for the CPU and for
  fall-backs. The depth-1 bit-exactness against `ConvCoalescedTsetlinMachine` holds on the eager
  path, which the CPU test pins.

### 2.3 Events mode (Phase 3)

* `_feedback_counts` in events mode: `acc.n_ib += sel_nofire_i.sum(0)`,
  `acc.n_ia += sel_fire_i.sum(0)`, and `acc.events = (lit, sel_fire_i, sel_fire_ii, p_i, p_ii)`.
* `lit` is the `(B, P, 2F)` literal tensor in float16. The first version materialises it from
  `xs` (`cat(xs + [1 − x])`).
  * Memory is `B·P·2F·2` bytes: about 100 MB for 100 graphs × 30 nodes × depth 3 × `M = 4096`.
  * `_events_chunk_size` must bound it: events mode needs the whole update in one chunk.
* `_events_eligible`: the base conditions (CUDA, Triton, soft budget, `B < TABLE_N`), plus
  `layer_s is None`. `fused_commit` takes one `s`. Per-layer `s` in the kernel means per-literal
  `q` tables, so it stays on the block commit for now.
* `_evaluate_learning` needs no override: the layered evaluation and its `(B, P, C)` matches are
  computed as now.
* **Optional `HALF` kernel flag** (only if the profile shows the `cat` or its memory): pass the
  positive half `(B, P, F)` and have `_commit_kernel` read `1 − lit[l − F]` for `l ≥ F`. This is
  one extra `constexpr` branch in the `CONV` gather, and changes nothing for the existing models
  (the flag is off).

### 2.4 Fused sequential commits (Phase 4)

* In `acc.fuse` mode with `B == 1`:

  ```python
  acc.single = (full[0, p_i], full[0, p_ii], sel_fire_i[0], sel_nofire_i[0], sel_fire_ii[0])
  ```

  This is the `_ConvMixin` code with nodes for patches. `full` is the `(P, 2F)` literal tensor
  of the one graph.
* Set `_records_single = True`. `_fuse_single_examples` reverts to the base rule (CUDA, soft
  budget, enough work) plus `layer_s is None`.
* The compiled `_accumulate_literals` is traced through the layered evaluation. That is a new
  graph for dynamo, so watch recompiles: shapes are static only with the padded layout. The
  unpadded layout is shape-polymorphic per graph and would recompile per node count.
  * Rule: fuse only padded batches.
  * Otherwise fall back to the eager sequential path. It is still faster after Phase 2, because
    there are no `nonzero` syncs.

### 2.5 CUDA graphs (Phase 5)

* `_graphs_ok`: the base rule, applied to padded batches only. Both batched (events) and
  sequential (fused) updates qualify.
* `_graph_key`: `xb.shape` → `(B, P, F, E, n_edge_types, n_node_types)` of the padded batch.
  The buffers and scalar hyper-parameters are as in the base.
* `_graphed_update` and `_capture_block` call `xs.clone()` / `gx.copy_(xs)` and slice
  `xb[start:start + step]`. With `GraphBatch.copy_` and static-shape slicing these work as
  written. Make the base use a small `_clone_input` / `_copy_input` hook rather than calling
  tensor methods directly, so other structured inputs can join later.
* Caches created during an update must exist before capture, or the graph owns their memory
  (CLAUDE.md: "a tensor the graph allocated and Python keeps is graph-private memory"):
  * `_bind_cache` (binding matrix);
  * `_layer_cols`.

  The two eager warm-up runs build them. Their keys depend on `msg_bits._version`, which never
  changes during training. A test pins that a replay does not rebuild them.
* `_GRAPH_FLAGS`: add any `__dict__` flag the graph update sets (none today besides the base
  ones; re-check after Phases 3–4).

### 2.6 Later, only if the profile asks (Phase 6)

* **Bit-packed node evaluation.** A Triton kernel computes one layer's per-node matches from
  bit-packed literals: `conv_eval` without the OR over patches, writing `(N, C)` bool and
  ANDing the running match. Today each layer is a float16 tensor-core matmul, which was not the
  bottleneck in the smoke profiles (`aten::mm` ~20% of CUDA time). Do it only if measured.
* **Fused inbox kernel.** Scatter the matches along the edges and set the bound message bits
  in one kernel, without the `(N·nt, C)` count tensor.
* **Ragged storage** for very uneven graphs:
  * Evaluate on the `N_real` real nodes, with `graph_of_node`.
  * The OR over nodes becomes `scatter_reduce(amax)` into `(B, C)`.
  * The draw becomes a segment argmax of packed `(key << 32 | node)` values, as `conv_eval`
    packs them.

  This is shape-polymorphic, so eager-only (no CUDA graphs). It is a memory and speed win when
  `max n_i ≫ mean n_i`. It also helps the CPU.

## 3. Phases, tests and acceptance

Each phase ends with:

* `ruff check src tests`;
* `pytest tests/test_graph.py tests/test_triton.py` on CPU and CUDA;
* a smoke profile (no timing runs without the user's go-ahead; memory note).

| Phase | Work | Tests |
|---|---|---|
| 0 | Sync audit: run a batched and a sequential graph update under `torch.cuda.set_sync_debug_mode("warn")`, list every sync with its line | — (findings into this plan) |
| 1 | Padded `GraphBatch` layout: `pad`, `(B, E)` edges + mask, static slicing, `copy_`; evaluation uses it | Padded and unpadded batches give identical `node_clause_outputs`, `evaluate_clauses` and eager updates (same seed); slicing a padded batch is sync-free; naive-reference test on padded batches |
| 2 | Dense node draw on CUDA learning path | Draws are uniform over matching nodes and independent between Ia and II (as `test_conv_patch_draws_are_uniform_and_independent`); `shared_node_draw` gives equal indices |
| 3 | Events mode | Given the same events and seed, a graph model's `fused_commit` equals `ConvCoalescedTsetlinMachine`'s, bit for bit (depth 1, `from_images`); events-mode counts equal the eager `W @ H` counts for fixed draws; `test_fused_paths_are_used…` extended to graphs; per-layer `s` and hard budget stay off the path |
| 4 | Fused sequential | `acc.single` rows equal `full[0, p]`; padded sequential learns the AAA task; unpadded sequential still uses the eager path |
| 5 | CUDA graphs | Graph update is sync-free (`test_updates_never_synchronise_with_the_host` for graphs, batched and sequential); replays learn and draw fresh randomness; caches stay fresh after replays (`_bits_fresh`); recapture on hyper-parameter change; `cuda_graphs = False` and `TORCHTSETLIN_TRITON=0` fall back |
| 6 | Optional kernels / ragged | Exact equality with the dense path on evaluation; distribution tests for draws |

**Acceptance.** All of these on the smoke configurations of `plans/graph-tm.md` (RTX 3090):

* Learning results are unchanged: the AAA and noisy-XOR tests, and `examples/graph_sequence.py`
  at 1.000 on its 4 seeds.
* Sequential updates of padded batches are at least 5× faster per graph than today's 5.6 ms.
* Batched updates are no slower.

The actual speed-ups go into the benchmarks only after the user okays a benchmark run. A
`graph` suite in `benchmarks/bench_device.py` (sequence task, `from_images` MNIST at depth 1)
is part of that request, not of this plan.

## 4. Risks and open questions

* **Fidelity of padded vs unpadded.** Padding nodes must never match (`node_ok`) or send
  (`edge_mask`). Phase 1's equality tests pin this before anything else builds on it.
* **Memory of the events literal tensor** at large `M` and depth. The `HALF` flag halves it.
  If that is still too big, events mode declines (chunked eager path), the same as conv models
  that exceed `_events_chunk_size`.
* **Dynamo recompiles in Phase 4.** Every distinct `(P, E)` is a new trace, and dynamo gives up
  after 8 (CLAUDE.md, compiled kernels). Datasets should pad to one or a few shapes; `pad()`
  without arguments pads to the batch maximum, which varies. Document "pad to the dataset
  maximum" in the guide.
* **CUDA-graph cache size** (`_GRAPH_CACHE = 4`). One shape per padded dataset is fine; mixing
  several pad sizes thrashes it.
* **Per-layer `s`** stays on the eager block commit. Extending `fused_commit` with per-literal
  probabilities touches the shared kernel and its tables, so it is a separate decision.
* **The base CUDA-graph code** calls tensor methods on the input. The `_clone_input` /
  `_copy_input` hooks are a small change to `models/base.py`. Every other model keeps its
  behaviour (tested by the existing `test_triton.py`).

## 5. Files

* `src/torchtsetlin/data/graphs.py`: `pad`, the padded edge layout, static slicing, `copy_`.
* `src/torchtsetlin/models/graph.py`: the dense draw, events and single-example recording, the
  eligibility hooks, and the CUDA-graph key and input hooks.
* `src/torchtsetlin/models/base.py`: the `_clone_input` / `_copy_input` hooks (Phase 5).
* `src/torchtsetlin/_triton.py`: only the optional `HALF` flag (Phase 3) and the Phase 6
  kernels.
* `tests/test_graph.py`, `tests/test_triton.py`.
* Docs:
  * `docs/guides/graphs.md`: padding for speed;
  * `docs/concepts/graph-tm.md`: drop "no fast path";
  * `CLAUDE.md` graph note;
  * `CHANGELOG.md`.

## Implementation notes (2026-11-07)

What the implementation did, relative to the plan above:

- **Phase 0, sync audit** (`torch.cuda.set_sync_debug_mode("warn")`, batched and sequential,
  CUDA):
  - Every slice of a `GraphBatch` synchronised, including the update's own `xb[0:B]` chunk,
    in the constructor's validation, the padding trim, `edge_graph` (`repeat_interleave`) and
    the edge offsets.
  - The graph `_feedback_counts` synchronised in its `nonzero` loop.
  - The commits synchronised in `apply_feedback` (batched, budget check) and in the row-sparse
    `apply_feedback_rows` (sequential).
- **Phase 1.** `GraphBatch.pad`, `unpad`, `clone`, `copy_`, `is_padded`, `edge_slots` and
  `flat_edge_mask`. `edges`, `edge_type` and `edges_per_graph` are now properties, derived for
  padded batches.
  - Padded slicing is plain tensor indexing with no checks (`_getitem_padded`); `cat` keeps the
    padded layout.
  - The model masks unused edge slots in `_inbox`.
  - Tests:
    - padded = flat for outputs, including extra padding nodes;
    - eager updates are bit-identical at the same node count;
    - padded slicing passes `set_sync_debug_mode("error")`.
- **Phases 2–4.** One method, `_record_fast`, writes `acc.events` (events mode) or
  `acc.single` (fused sequential) from a dense `_draw_nodes` (`rand · matches`, argmax over
  nodes; `shared_node_draw` reuses the Ia draw). The literal tensor is materialised in float16
  (no `HALF` flag: not needed so far).
  - Events mode needs no padding: `fused_commit` takes `B` and `P` at run time, so flat batches
    use it too.
  - **Change from the plan:** `_fuse_single_examples` has no minimum model size. The base rule
    (`ta_state.numel() · steps ≥ 2^18`) kept the sequence example (6,480 automata, Trainer
    batches of 25) on the eager path at ~4.8 ms per graph. An eager graph step costs that in
    launches and syncs whatever the model size, so a padded graph model fuses whenever it can.
- **Phase 5.** `_graphs_ok` accepts only padded batches. `_graph_key` is the base key on the
  node features plus the padded edge shape. **No change to `models/base.py`** was needed:
  `GraphBatch.clone()` and `copy_()` are exactly what the base capture code calls.
- **Measured (smoke, RTX 3090, sequence example, 40 clauses, depth 2):**

  | Update | Before | Now |
  |---|---|---|
  | Sequential | 4.8 ms per graph | 0.065 ms per graph replayed (~74×); an epoch of 2,000 graphs takes 0.1 s instead of 9.7 s |
  | Batched (25) | 0.23 ms per graph | 0.013 ms per graph (~17×) |

  The example reaches 1.000 on seeds 0–3. The first sequential call compiles (~6 s once).
- **Tests** (`tests/test_graph.py`, CUDA ones need Triton):
  - the dense draw is uniform and independent;
  - events carry the literal rows and matching nodes;
  - padded updates take the fused paths and learn the AAA task (flat sequential, per-layer `s`
    stay eager);
  - padded updates never synchronise (batched and sequential);
  - CUDA-graph replays learn and draw fresh randomness (restoring caches in place, so the
    replay itself is tested).
- **Not done (phase 6):** bit-packed node evaluation, a fused inbox kernel and ragged storage.
  The profiles do not call for them yet.
