# Graph Tsetlin machine (GraphTM): implementation plan

*Status: **plan**, 2026-11-07. Nothing implemented yet. Tracks GitHub issue #8 ("Implement GRAPH TM").*

## TL;DR

The Graph Tsetlin Machine (GraphTM, Granmo et al., arXiv:2507.14874 [R1]) is a **convolutional
coalesced TM whose "patches" are graph nodes**, plus **message passing**. A deep clause
`C_j = C_j^0 ∧ C_j^1 ∧ … ∧ C_j^{D-1}` has one component per layer:

- `C^0` reads the node's own features;
- `C^i` (i ≥ 1) reads the node's **inbox**, the messages its neighbours sent in layer i−1.

Each message says "clause k matched at a neighbour reached over an edge of type t". A clause is
True for a graph if its deep clause holds at **any** node, the OR over nodes that the
convolutional TM [R3] takes over patches. Voting and learning are the coalesced TM's [R4].

That maps well onto torchtsetlin:

1. **Depth 1 is the existing `ConvCoalescedTsetlinMachine` on a different set of patches.** The
   paper says so (Sec. 3.2: "GraphTM effectively reduces to the convolutional variant of CoTM").
   `plans/conv3d.md` already showed that `_ConvMixin`'s evaluation and feedback never use the
   grid, only a set of patches `(B, P, 2F)`. Nodes are that set, padded to the largest graph
   in the batch and masked.
2. **Depth > 1 fits in one `ta_state`.** Concatenate the per-layer inputs into one literal
   vector per node, `[H^0 | H^1 | … | H^{D-1}]`, with `H^0` of size `F` and each `H^i` of size
   `M`. A clause is then a single row of `2(F + (D−1)M)` automata. The deep clause is that
   row's AND, because AND is associative (Lemma 1). The reference implementation [R2] gives
   feedback to every layer at the **same randomly chosen node where the full clause is True**.
   That is exactly what `_ConvMixin._feedback_counts` already does with one concatenated
   literal row (Lemma 2). Only evaluation changes, because `H^i` depends on the layer-(i−1)
   matches of the neighbours. `_evaluate` must therefore run the layers in order and hand the
   built literal tensor to `_feedback_counts`.

So the work is:

- a graph input container;
- a `_GraphMixin` that overrides `_coerce_input`, `_encode`, `_evaluate` and a thin
  `_feedback_counts` wrapper;
- message passing as sparse scatter plus a small matmul;
- clause decoding for interpretability;
- the validation experiments the paper and reference code define.

No change to `functional.py` is needed for the first version. Per-layer `s` (Step 7) is the only
item that touches it.

Estimate: about 4–6 working days to a tested CPU/GPU eager model with docs. The Triton/CUDA-graph
fast paths are deliberately out of scope for v1 (Step 9). `experiments/convtm/LIBRARY_GAPS.md`
LG-009 estimated 8–15 d. The difference is that this plan reuses the convolutional machinery
instead of porting the CUDA kernels.

---

## 1. What the GraphTM computes (the specification we implement)

Sources: paper [R1] Sec. 2 and Appendix A.3, Algorithm 2; reference implementation [R2]
`GraphTsetlinMachine/kernels.py` and `tm.py` at commit `c6129e1` (2026-04-13). Where they
differ, the code is what produced the paper's numbers, so we treat it as normative and the
paper as explanation.

**Input.** A directed multigraph per example:

- nodes `v_q`, each with a set of **symbols** (properties);
- typed edges `(v_q, v_r, t)`;
- optionally a node type.

Symbols become a sparse binary hypervector `H^0_q` of size `F` (`Graphs` in [R2] `graphs.py`):

- **random**: each symbol owns `hypervector_bits` random positions;
- **double hashing**: positions `i mod F` and `p − (i mod p)`, with `p` the largest prime below `F`;
- **one-hot**: `F = #symbols`.

A node's vector is the OR (bundle) of its symbols' vectors. The second half of the literal
vector is the negation, as in every TM. Encoding symbols as hypervectors is the Hypervector TM
of Halenka et al. [R5]; the underlying sparse binary codes are Rachkovskij's [R9] and
Kanerva's [R10].

**Layer 0.** `M^0_{j,q} = C^0_j(H^0_q)` for every clause `j` and node `q`
([R2] `calculate_messages`). Clause `j` is evaluated only on nodes whose type equals
`j mod n_node_types`; on other nodes it is False.

**Message passing, layers i = 1 … D−1** ([R2] `prepare_messages`, `exchange_messages`,
`encode_messages`):

- The inbox `H^i_r` of size `M` (`message_size`) starts empty.
- Every clause `j` owns `message_bits` positions `h_j` in `[0, M)`: random by default, double
  hashing optional, or one-hot.
- For every edge `(q → r, type t)` with `M^{i-1}_{j,q}` True, set the bits
  `(h_j + t) mod M` of `H^i_r`.

  The edge type **binds** the message by a cyclic shift (paper Algorithm 2: `I_new ← I_cl + e`).
  In one-hot mode `M = C · n_edge_types` and `h_j = j · n_edge_types`, so every (clause, edge
  type) pair has its own bit and nothing collides.

**Deep matches.** `M^i_{j,r} = M^{i-1}_{j,r} ∧ C^i_j(H^i_r)` ([R2]
`calculate_messages_conditional`; paper A.3). This is a running conjunction at the **same**
node: a message is sent at layer i only if the clause's components `0…i` all hold there.

**Clause output and vote.**

- `c_j = OR_r M^{D-1}_{j,r}` ([R2] `evaluate`).
- Class sums are `v_k = clip(Σ_j w_{jk} c_j, −T, T)`, with signed integer weights per
  (clause, class), initialised to ±1 at random (`negative_clauses = 1` in [R2] `prepare`).

**Learning** ([R2] `select_clause_node`, `select_clause_updates`, `update`, `update_message`):

- **Node draw.** For each clause, one node is drawn uniformly at random from the nodes where the
  full deep clause is True (`clause_node`), or none (`-1`).
- **Clause selection.** For each class `k`, the clause is selected with probability
  `|y_k − v_k| / 2T`, where `y_k = ±T`. For the negative classes this probability is further
  multiplied by `q/(K−1)`. Type I goes to clauses whose weight sign agrees with the target,
  Type II to the others.
- **Weights.** A clause that has a drawn node (matches somewhere) moves its weight by ±1.
- **Automata.** Every layer's automata get standard Type I / Type II feedback against **that
  layer's vector at the drawn node**: `H^0` for layer 0, `H^i` for layer i.
- **Clause not matching anywhere.** Type I acts as Type Ib (forgetting) on all of the clause's
  automata, and Type II does nothing.
- **Literal budget.** `max_included_literals` counts the includes of **all** layers together
  (`number_of_include_actions` is summed over layers). Type Ia is allowed while
  `count ≤ budget`, otherwise only Type Ib applies. This is the tmu-style soft budget.
- **Boost.** `boost_true_positive_feedback = 1` by default.
- **Per-layer specificity.** `s` may differ per layer (`s` given as a tuple).
- **Order.** The reference trains **one graph at a time**, which is sequential feedback.

**Interpretation.** A message literal is written `l1:0` or `r1:0`:

- edge type (here l or r);
- message layer;
- index of the clause whose earlier component matched at the neighbour.

Decoding traces it back to node symbols (paper Sec. 3.1 Tables 1, 13; [R2] `get_messages`,
`get_clause_literals`).

## 2. How it maps onto torchtsetlin

### Lemma 1: one concatenated automaton row computes the deep clause

Let a clause's include mask be split by layer, `I = [I^0 | I^1 | … | I^{D-1}]` (positive and
negated halves likewise), and let a node's literal vector be `x_r = [H^0_r | H^1_r | … ]`. The
flat clause evaluates `AND_{l ∈ I} x_r[l]`, which equals
`AND_i AND_{l ∈ I^i} H^i_r[l] = AND_i C^i(H^i_r) = M^{D-1}_r`.

An empty component contributes `True`, as in the paper ("an empty clause matches any input",
A.3). ∎

The same split computes every intermediate `M^i` from the prefix of layers `0…i`. So
`_evaluate` runs one masked match per layer and ANDs it into a running `(B, P, C)` tensor. It
never needs a second automaton tensor.

### Lemma 2: the feedback is the conv feedback on the concatenated row

`_ConvMixin._feedback_counts` (`models/conv.py`) draws, per selected (example, clause) event, a
uniformly random patch among those where the clause matches. It then adds that patch's whole
literal row to `n_true / n_false / n2`.

With patches = nodes, rows = `[H^0 | H^1 | …]` and matches = `M^{D-1}`, that row restricted to
layer `i` is exactly `H^i` at the drawn node: the vector [R2] `update_message` uses for layer
`i`. Type Ib, the boost flag and the soft budget act per automaton, and the budget compares the
row's total include count. That is the reference's sum over layers. ∎

**Fidelity differences we keep, documented and tested:**

| Point | Reference [R2] | torchtsetlin | Exact-match setting |
|---|---|---|---|
| Node draw | one node per clause per example, shared by all classes and by Type I/II | an independent draw per feedback event (conv path, so coalesced Ia and II can hit different patches) | option `shared_node_draw=True` draws once per (example, clause) |
| Negative classes | every class updated, negatives scaled by `q/(K−1)` | multi-class mode: target plus **one** sampled negative | `multi_label=True` with one-hot targets and `negative_scale = q/(K−1)` is the same rule (`CoalescedTsetlinMachine._select_feedback`, multi-label branch) |
| Order | sequential, one graph at a time | `feedback_mode="batch"` by default | `sequential=True` / `feedback_mode="sequential"` |
| Empty clause at prediction | votes (no special case) | whole clause empty → False when predicting (`clause_outputs`, `self.training`) | keep torchtsetlin's rule; an empty clause also sends no messages at prediction. Document it. A component that is empty while the clause is not is still True. |
| `s` | per layer allowed | one `s` | Step 7 |
| State | bit-sliced `STATE_BITS` counters | `uint8` `ta_state`, `n_states` | same automaton, different storage |

## 3. Design

### 3.1 Input: `GraphBatch` (`src/torchtsetlin/data/graphs.py`)

`GraphBatch` is a padded container that keeps the base learning loop's assumption that an input
batch is indexable along dim 0:

- `node_features`: `(B, P, F)` bool. `P` is the largest node count in the batch; padding rows
  are zero.
- `node_mask`: `(B, P)` bool.
- `node_type`: `(B, P)` int8/16, optional.
- `edge_index`: `(2, E)` long, indices into the flattened `B·P` node axis, sorted by source.
- `edge_type`: `(E,)` int16.
- `edge_graph`: `(E,)` long, the graph id of each edge, for slicing.
- `n_edge_types`, `n_node_types`: ints, fixed per dataset (stored so subsets keep them).
- `__len__`, `__getitem__(int | slice | LongTensor)` re-index nodes and edges. `.to(device)`,
  `.shape` gives `(B, P, F)` so `xb.shape[0]` works in the base class.

Builders:

- `GraphBatch.from_lists(features, edges, edge_types, node_types=None)`, the general case;
- `from_sequences(tokens, symbols, edge_types=("left", "right"))`, as in [R2]
  `SequenceClassificationDemo`;
- `from_image_patches(x, patch_size, stride, position_encoding=True, edges=None)`, the
  disconnected-nodes image encoding of paper Sec. 3.2 / Fig. 2, which reuses the
  `_ConvMixin` unfold;
- `from_networkx` / `from_pyg`, import-guarded and optional.

`SymbolTable(symbols, dim, bits, encoding="random" | "double_hashing" | "one_hot", seed)` maps
symbol names to hypervector positions, as in [R2] `Graphs.__init__`. It reuses the table logic
of `data/encoders.py::HypervectorEncoder` (Halenka et al. [R5]) and keeps the inverse for
decoding.

Node features may also be **plain Boolean vectors** (patch pixels and thermometer bits). A
one-hot symbol table is exactly that. This is what makes the depth-1 equivalence test bit-exact.

### 3.2 Model: `GraphTsetlinMachine` (`src/torchtsetlin/models/graph.py`)

```python
class _GraphMixin(_ConvMixin): ...            # overrides geometry + evaluation only
class GraphTsetlinMachine(_GraphMixin, CoalescedTsetlinMachine):
    def __init__(self, n_features, n_outputs, n_clauses, T, s=10.0, *,
                 depth=1, message_size=256, message_bits=2,
                 message_encoding="random",   # | "double_hashing" | "one_hot"
                 n_edge_types=1, n_node_types=1,
                 shared_node_draw=False, multi_label=False, **kwargs): ...
```

- Total features: `n_features + (depth−1)·message_size`. `n_features=None` stays lazy, the size
  being taken from the first `GraphBatch`.
- The clause message hypervectors `msg_bits (C, message_bits)` are a **persistent buffer**, so a
  checkpoint reloads the same bindings. Per-type shifted bit matrices are built lazily as a
  **non-persistent** cache (`(n_edge_types, C, M)` bool, or index form), following the CLAUDE.md
  rule for config tensors.
- `_evaluate(literals0, empty_value)`:
  1. Run the layer-0 match `(B, P, C)` with `include[:, layer-0 slice]` via
     `F.clause_outputs` on the `(B·P, 2F)` rows. AND it with `node_mask` and the node-type
     mask.
  2. For each later layer: build `H^i` by message passing (3.3), take the layer-`i` match of
     `include[:, layer-i slice]`, and AND it into the running match.
  3. Return `matches.any(1)` and the context `(matches, full_literals)`, with the full
     literals assembled as `(B, P, 2F_total)`.

  "Whole clause empty" uses `include_count == 0`. Component emptiness needs no special case
  (Lemma 1).
- `_feedback_counts` unpacks the context and calls `_ConvMixin._feedback_counts` with the full
  literals and matches. With `shared_node_draw`, it draws one node per (example, clause) and
  reuses it.

  The class sets `_records_events = False` and `_records_single = False`, so the Triton events
  path and the fused single-example path never see graphs in v1. `_graphs_ok` returns False:
  graph shapes vary, so the update is not captured as a CUDA graph.
- `_chunk_elements_per_example` must count the real intermediates (CLAUDE.md: a wrong value
  costs 4–20×):
  - `P·C·D` for the matches;
  - `P·2F_total` for the literals;
  - `P·M` per inbox;
  - `E/B · C` for the message scatter.
- Prediction-time API:
  - `forward(g)`;
  - `predict(g)`;
  - `node_clause_outputs(g, layer=None) -> (B, P, C)` per layer, the analogue of [R2]
    `transform_nodewise`;
  - `transform(g) -> (B, C)` clause outputs as features, as in [R2] `transform`.

### 3.3 Message passing on tensors

Given the running layer-(i−1) match `Mprev (B·P, C)` and the edges `(src, dst, t)`:

- **one-hot messages:** `inbox[dst, j·n_t + t] |= Mprev[src, j]`. This is one `index_add_` of
  `Mprev[src]` into a `(B·P, C, n_t)` view, followed by `> 0`. Its cost is `E·C` with no matmul.
- **hashed messages:** first `S_t = scatter_OR(Mprev[src_e] for edges of type t → dst)`, of
  shape `(B·P, C)` per type, using `index_add_` on a float or int buffer and then `> 0`. Then
  `inbox |= (S_t @ Bind_t) > 0`, where `Bind_t (C, M)` holds clause `j`'s bits shifted by `t`.
  That is `n_t` matmuls of `(B·P, C) × (C, M)`, the dominant cost.

  Chunk over destination rows if `B·P·C` exceeds the chunk budget. The alternative is a
  `torch.sparse` CSR adjacency per type, `A_t @ Mprev`, which is better when `E ≫ B·P`; choose
  by benchmark (Step 8).

Messages are Boolean ORs, so the order of edges is irrelevant and the batched and per-graph
computations are equal. A test pins this against a naïve per-edge Python loop.

## 4. Steps

Each step ends with `ruff check src tests` and the new tests green on CPU and CUDA (the `device`
fixture).

1. **`GraphBatch` + `SymbolTable`.** Builders, slicing, `.to()`, `n_*_types` bookkeeping.
   Tests:
   - slicing a batch and concatenating it back is the identity;
   - edges stay within their graph after any re-indexing;
   - `SymbolTable` positions equal [R2] `Graphs` for one-hot and double hashing, which are
     deterministic (`sympy.prevprime` is replaced by a small prime sieve, avoiding a dependency).
2. **Depth-1 `GraphTsetlinMachine` = `ConvCoalescedTsetlinMachine`.** Build graphs with
   `from_image_patches(..., edges=None)` from the same Boolean images. Tests:
   - copying one `ta_state` and weight buffer into both models gives identical `forward`
     votes;
   - with fixed draws (monkeypatched `torch.rand`, or a shared `generator`), one `update` gives
     identical `ta_state`/weights;
   - variable node counts plus padding give the same result as each graph alone.

   This is the paper's own sanity claim (Sec. 3.2) made exact.
3. **Message passing, depth ≥ 2.** Implement 3.3. Tests:
   - inboxes equal the naïve loop on random graphs, for each `message_encoding`;
   - the paper's worked example, Appendix A.3, Algorithm 2 / Table 15: the five-node sequence
     "BAAAE", `HV=8`, `Icl=[4,5]`, giving the stated `H^1_2 = [00001110 11110001]`;
   - Lemma 1: the concatenated-row match equals the layer-by-layer conjunction.
4. **Learning at depth ≥ 2.** Wire the context into `_feedback_counts`, including
   `shared_node_draw`. Tests:
   - the feedback counts for a hand-built two-node graph equal a direct evaluation of the
     reference update rules in Sec. 1;
   - sequential mode on one graph equals batch mode with batch size 1;
   - a learning test: the three-consecutive-"A" sequence task of paper Sec. 3.1 (5-letter
     sequences, 2 layers, 4 clauses), required to reach 100% noise-free test accuracy as the
     paper reports, within the 600 s test budget at a reduced size;
   - [R2] `NoisyXORDemo` settings (depth 2, 10 clauses, T=100, s=1.0): it must learn noisy
     XOR, which needs depth 2 because the two inputs sit on different nodes.
5. **Interpretation.**
   - `clause_expression(j, symbols=table, edge_type_names=...)` prints `A ∧ ¬B ∧ r1:0 ∧ l2:3`.
     For hashed messages it decodes only bits that identify a unique (clause, type) pair and
     marks collisions.
   - `rules()` prints with weights, like `CoalescedTsetlinMachine.rules`.
   - `explain_node(g, b, node)` traces a deep clause back to node symbols, as in paper Sec. 3.1
     Table 1.

   Test: on the Sec. 3.1 task, the learned clauses decode to statements about "A" at offsets
   −1/0/+1.
6. **Docs.**
   - `docs/concepts/graph-tm.md`: deep clauses, messages, edge binding, the depth-1 = CoTM
     reduction and Lemmas 1–2, with figures from `scripts/make_concept_figures.py` (light and
     `_dark` pairs, animated per CLAUDE.md rules).
   - `docs/guides/graphs.md`: building `GraphBatch`es, sequence and image recipes, decoding
     clauses.
   - `docs/api/` stubs; the `docs/concepts/tsetlin-machine.md` "Variants implemented" table;
     a `CLAUDE.md` architecture note (the `_GraphMixin`, the concatenated row, no events or
     CUDA-graph path).
   - An `examples/graph_sequence.py` script.
7. **Per-layer specificity (optional).** [R2] accepts `s=(s0, s1, …)`. This needs a per-literal
   `1/s` vector in `F.apply_feedback` / `apply_feedback_rows` (`functional.py`), plus the
   compiled twins and the Triton commit, because Ib and the non-boosted Ia draw with
   probability `1/s`. Gate it behind `s` being a tuple, and keep scalar `s` bit-identical
   (test).
8. **Performance.** Profile the CPU and GPU eager paths on the depth-1 MNIST configuration and
   on a depth-2 sequence task.
   - Fix `_chunk_elements_per_example` against measured peak memory.
   - Choose scatter or sparse-matmul message passing.
   - Measure only `--quick`-style smoke runs until the user okays benchmark runs (memory note:
     no timing runs without a go-ahead).
9. **Fast paths (later, separate plan).**
   - Events-mode Triton evaluation, because `_triton.conv_eval` already handles a padded patch
     axis; the node mask has to be added.
   - CUDA-graph capture for fixed-shape graph datasets (images: every graph has the same `P`
     and `E`).
   - Ragged (unpadded) node storage for very uneven graphs.
10. **Reproduce paper numbers (needs the user's go-ahead; benchmark-class runs).**
    - **Disconnected nodes (Sec. 3.2, Table 3; hyper-parameters Table 16):**
      - MNIST: 2,500 clauses, 10×10 window, HV 128, depth 1, 30 epochs, T=3,125, s=10; reported
        **98.42 ± 0.05**, CoTM 98.93 ± 0.02;
      - F-MNIST: 40,000 clauses, 3×3, T=15,000, s=10; reported **89.49 ± 0.09**;
      - CIFAR-10: 80,000 clauses, 8×8, budget 32, T=15,000, s=20, two views (adaptive Gaussian
        threshold plus 8-bin colour thermometer); reported **70.28 ± 0.17** vs CoTM 66.42.

      The reported number is the mean of the last five epochs' test accuracy, so selection is
      on the test set; this is noted in the literature DB sidecar.
    - **Multivalue noisy XOR (Sec. 3.8, Fig. 8, Table 22):** depth 2, s=2.2, T=10×clauses,
      500 symbols, HV 2048, message HV ∈ {2048, 4096, 8192}, 1% noise.

    These go into the `experiments/paper_experiments` programme, as a new row next to the
    existing cross-library comparisons. Its `PLAN.md` currently lists GraphTsetlinMachine as
    "publishes no numbers to compare against"; update that, because the paper now does.

## 5. Risks and open questions

- **Memory at paper scale.** CIFAR-10 at depth 1 is the existing 80k-clause CTM, so nothing new.
  Depth ≥ 2 with hashed messages adds `(D−1)·2M` automata per clause and a
  `(B·P, C) × (C, M)` matmul per edge type per layer. The batch must chunk by graphs.
- **Batched feedback fidelity.** As with conv models, batch feedback departs from the sequential
  reference as the batch grows (`docs/benchmarks.md`). Message passing adds a second coupling
  inside an update, because a commit changes the messages other clauses see. Measure batch
  10/50/200 against sequential on the sequence task before recommending a default.
- **Hash collisions.** With random message bits, `C·n_t·message_bits` set positions out of `M`
  collide. The paper reports accuracy falling at 2,000 clauses vs 1,000 at the largest message
  size, likely from collisions (Sec. 3.8). Expose a `message_collision_rate()` diagnostic.
- **Empty-clause rule.** We keep torchtsetlin's "empty is False at prediction" rule, which
  differs from [R2]. If the reproduction in Step 10 falls short, check this first, before
  suspecting the feedback code.
- **Node types.** [R2] assigns clause `j` to node type `j mod n_node_types`, which is a hard
  partition. We copy it; a learned assignment is out of scope.
- **Regression / autoencoder heads.** [R2]'s roadmap lists them as future work. The mixin design
  allows a `GraphRegressionTsetlinMachine(_GraphMixin, RegressionTsetlinMachine)` later at
  almost no cost, the same way `Conv*` variants are produced.

## 6. Files

New:

- `src/torchtsetlin/data/graphs.py` (`GraphBatch`, `SymbolTable`);
- `src/torchtsetlin/models/graph.py`;
- `tests/test_graph.py`;
- `docs/concepts/graph-tm.md`, `docs/guides/graphs.md`, `docs/api/graph.md`;
- `examples/graph_sequence.py`.

Edited:

- `models/__init__.py`, `data/__init__.py` (exports);
- `tests/test_package.py` (if the subpackage list changes);
- `mkdocs.yml` (navigation);
- `CLAUDE.md`;
- `docs/concepts/tsetlin-machine.md`;
- `experiments/convtm/LIBRARY_GAPS.md` (close LG-009 when done);
- `functional.py`, only in Step 7.

## References

Local PDFs are under `source_documents/`. Each paper also has a record in the literature catalog
(`benchmarks/literature/papers/`).

- **[R1]** O.-C. Granmo, Y. Abdelwahab, P.-A. Andersen, K. A. K. Borgersen, P. F. A. Clarke,
  K. Dumbre, Y. Grønningsæter, V. Halenka, R. Helin, L. Jiao, A. Khalid, R. Omslandseter, R. Saha,
  M. Shende, X. Zhang. *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With
  Graphs.* arXiv:2507.14874v2, 6 May 2026. https://arxiv.org/abs/2507.14874.
  Local: `source_documents/papers/01_foundations/2025_The_Tsetlin_Machine_Goes_Deep_Graphs_arXiv2507.14874.pdf`.
  Parts used:
  - Sec. 2: the algorithm;
  - Sec. 3.1: the sequence task;
  - Sec. 3.2 / Table 3 / Table 16: disconnected nodes;
  - Sec. 3.8 / Fig. 8 / Table 22: multivalue XOR;
  - App. A.2 Alg. 1: the coalesced update;
  - App. A.3 Alg. 2 / Tables 13–15: message hypervectors and the worked example.
- **[R2]** CAIR, *GraphTsetlinMachine* (reference CUDA/PyCUDA implementation, MIT licence).
  https://github.com/cair/GraphTsetlinMachine, commit `c6129e1466d87ebafe411c8a334276ad53c10763`
  (2026-04-13).
  - `GraphTsetlinMachine/graphs.py`: `Graphs`, the symbol hypervectors.
  - `GraphTsetlinMachine/kernels.py`:
    - `calculate_messages`, `calculate_messages_conditional`;
    - `prepare_messages`, `exchange_messages`, `encode_messages`;
    - `evaluate`, `select_clause_node`, `select_clause_updates`;
    - `update`, `update_message`.
  - `GraphTsetlinMachine/tm.py`: `CommonTsetlinMachine._evaluate`, `_fit`,
    `MultiClassGraphTsetlinMachine`.
  - `examples/`: `SequenceClassificationDemo.py`, `NoisyXORDemo.py`, `MNISTConvolutionDemo.py`.

  PyPI `GraphTsetlinMachine` 0.3.3 is CUDA-only; see
  `experiments/paper_experiments/sources/survey_repos.md` §8.
- **[R3]** O.-C. Granmo, S. Glimsdal, L. Jiao, M. Goodwin, C. W. Omlin, G. T. Berge.
  *The Convolutional Tsetlin Machine.* arXiv:1905.09688, 2019. https://arxiv.org/abs/1905.09688.
  Local: `source_documents/papers/01_foundations/2019_The_Convolutional_Tsetlin_Machine_arXiv1905.09688.pdf`;
  also Book Chapter 4, `source_documents/book_chapters/Tsetlin_Machine_Book_Chapter_4_Convolution.pdf`.
  Covers the OR over patches and the random matching patch in feedback.
- **[R4]** S. Glimsdal, O.-C. Granmo. *Coalesced Multi-Output Tsetlin Machines with Clause
  Sharing.* arXiv:2108.07594, 2021. https://arxiv.org/abs/2108.07594.
  Local: `source_documents/papers/01_foundations/2021_Coalesced_Multi_Output_Tsetlin_Machines_with_Clause_Sharing_arXiv2108.07594.pdf`.
  Covers the shared clause pool and signed weights. Implemented here as
  `CoalescedTsetlinMachine`.
- **[R5]** V. Halenka, A. K. Kadhim, P. F. A. Clarke, B. Bhattarai, R. Saha, O.-C. Granmo, L. Jiao,
  P.-A. Andersen. *Exploring Effects of Hyperdimensional Vectors for Tsetlin Machines.*
  arXiv:2406.02648, 2024. https://arxiv.org/abs/2406.02648.
  Local: `source_documents/papers/02_image_classification/2024_Effects_of_Hyperdimensional_Vectors_for_Tsetlin_Machines_arXiv2406.02648.pdf`.
  Covers the symbol hypervectors; already behind `data/encoders.py::HypervectorEncoder`.
- **[R6]** O.-C. Granmo. *The Tsetlin Machine: A Game Theoretic Bandit Driven Approach to Optimal
  Pattern Recognition with Propositional Logic.* arXiv:1804.01508, 2018.
  https://arxiv.org/abs/1804.01508.
  Local: `source_documents/papers/01_foundations/2018_The_Tsetlin_Machine_Game_Theoretic_Bandit_Driven_Approach_arXiv1804.01508.pdf`.
  Covers Type I/II feedback and the noisy XOR task.
- **[R7]** Book, *An Introduction to Tsetlin Machines*, Chapter 4 (Convolution), as [R3]. The
  library's concept pages already cite it.
- **[R8]** J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl. *Neural Message
  Passing for Quantum Chemistry.* ICML 2017, arXiv:1704.01212. The message-passing framing
  that GNNs share with the GraphTM; context for the docs page only.
- **[R9]** D. A. Rachkovskij. *Representation and processing of structures with binary sparse
  distributed codes.* IEEE Trans. Knowledge and Data Engineering 13(2):261–276, 2001. Cited by
  [R1] for its sparse hypervectors.
- **[R10]** P. Kanerva. *Hyperdimensional Computing: An Introduction to Computing in Distributed
  Representation with High-Dimensional Random Vectors.* Cognitive Computation 1(2):139–159,
  2009. Background for bundling (OR) and binding (here: cyclic shift).

Related, in this repo:

- `plans/conv3d.md`: the "patches are an abstract set" argument this plan reuses.
- `experiments/convtm/LIBRARY_GAPS.md` LG-009 and `experiments/convtm/LITERATURE_TM.md` row A5:
  the GraphTM gap and its CIFAR-10 configuration.
- `experiments/paper_experiments/sources/survey_papers.md` (Graph TM timings) and
  `survey_repos.md` §8 (installability).
