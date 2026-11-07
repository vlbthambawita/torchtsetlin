# Graph Tsetlin machines

A graph Tsetlin machine (GraphTM; Granmo et al., 2025, arXiv:2507.14874) learns clauses over
**graph-structured input**: sequences, grids, relations, or several views of one example. It
generalises the [convolutional machine](convolution.md). Where a convolutional clause looks at
one window at a time, a graph clause looks at one node *and at what the node's neighbours
reported*, and through several layers of such reports it can describe a whole sub-graph.

## The input: nodes, symbols and typed edges

A graph is a set of nodes with directed, typed edges between them. Each node carries a Boolean
feature vector, usually the **bundle** (bitwise OR) of the sparse hypervectors of its
*symbols* (properties), as in the hypervector TM (Halenka et al., 2024): a letter, a pixel
value, a coordinate. Edges have a type, such as "left" and "right" in a sequence or "parent" and
"measurement" in a medical record. A node hears from the nodes that have an edge *into* it.

In torchtsetlin a batch of graphs is a [`GraphBatch`](../guides/graphs.md), and
`SymbolTable` turns symbol names into hypervector positions.

## Deep clauses

A GraphTM clause has one **component per layer**:

    C = C⁰ ∧ C¹ ∧ … ∧ C^(D-1)

* `C⁰` is an ordinary clause over the node's own features.
* `C^i` (`i ≥ 1`) is a clause over the node's **inbox** of layer `i`. The inbox holds a message
  `(clause k, edge type t)` for every in-neighbour, reached over an edge of type `t`, at which
  clause `k`'s components `0 … i-1` all match.

So a node first checks its own properties. A clause that matches there tells the neighbours,
and the next layer can then require "and my left neighbour matched clause 3, and my right
neighbour matched clause 0". Each layer reaches one hop further. A clause is True for the
graph when its deep clause holds at some node: the same OR over positions that makes a
convolutional clause translation tolerant.

<figure markdown="span">
  ![A sequence BAAAE: layer 0 matches A at X1 to X3, messages go to both neighbours, and the layer-1 component that needs a match on both sides only holds at X2](../assets/concepts/graph-messages.svg#only-light){ loading=lazy } ![A sequence BAAAE: layer 0 matches A at X1 to X3, messages go to both neighbours, and the layer-1 component that needs a match on both sides only holds at X2](../assets/concepts/graph-messages_dark.svg#only-dark){ loading=lazy }
  <figcaption>Three consecutive A's with a depth-2 clause, the paper's worked example. Clause 0's first component <code>C⁰ = A</code> matches at X1–X3, and each match is sent to both neighbours, labelled with the edge it travels along. The second component asks for a match on both sides, so the deep clause holds only at X2.</figcaption>
</figure>

### Messages are symbols too

An inbox is a Boolean vector of `M` bits (`message_size`). Clause `k` owns `message_bits`
positions in it, and a message arriving over an edge of type `t` sets those positions
**shifted cyclically by `t`**. This binds the message to the edge type: "clause 3 via left" and
"clause 3 via right" set different bits. The inbox is the OR of all messages received, so a
clause component over it is an ordinary clause over Boolean literals, and the whole machinery
of literals, automata and feedback applies unchanged.

Three encodings are available (`message_encoding`):

| Encoding | Bits per message | Collisions | Readable as |
|---|---|---|---|
| `"random"` (default, as the reference code) | `message_bits` random positions out of `M` | possible | decoded when unambiguous |
| `"double_hashing"` | 2 (`k mod M`, `p − (k mod p)`) | possible | decoded when unambiguous |
| `"one_hot"` | 1, `M = n_clauses · n_edge_types` | none | `right1:0` exactly |

Hashed messages keep the inbox small however many clauses there are, at the price of
collisions: when many clauses share few bits, a message literal can no longer tell them apart.
The paper sees accuracy *fall* from 1,000 to 2,000 clauses at a fixed message size on its
multivalue XOR task, and attributes it to such collisions. `model.message_collision_rate()`
reports the fraction of messages whose bits are all shared with others. One-hot messages never
collide but grow with the clause count, which suits small and medium models and makes every
clause readable.

## Learning

Voting is the [coalesced](tsetlin-machine.md#variants-implemented) machine's: one pool of
clauses with a signed integer weight per class. Feedback is the convolutional machine's, applied
to the deep clause. For each feedback event, one node where the **whole** clause matches is
drawn at random, and every layer's component is updated against that node's input of that layer:
its own features for `C⁰`, its inbox of layer `i` for `C^i`. A clause that matches nowhere
receives only forgetting (Type Ib). The literal budget `max_included_literals` counts the
included literals of all layers together. As in the reference implementation, `s` may be one
value per layer, `s=(s0, s1, …)`: `s0` for the node features, `s_i` for the inbox of layer `i`.

The budget matters more than in a flat machine. At the start every clause is empty, matches at
every node and so sends messages from everywhere; a boosted Type Ia event then includes every
message literal of the drawn node at once. Without a budget, clauses can settle on "nothing
special here, and everything matched around me", which is True in every graph. On the sequence
example (`examples/graph_sequence.py`) depth 2 without a budget stays at chance, and with
`max_included_literals=8` it reaches 100% on every seed tried.

An empty component matches (it constrains nothing), in learning and in prediction: messages are
the next layer's input, and a clause can learn to rely on those of an empty clause, which say
no more than "a neighbour exists on that side". Only an entirely empty clause's own vote is
dropped when predicting, as in every torchtsetlin model.

## How torchtsetlin implements it

Concatenate a node's per-layer inputs into one row, `[H⁰ | H¹ | … | H^(D-1)]`. A clause is then
a single row of automata over that row, and:

1. **The deep clause is the row's AND.** AND is associative: the AND over all included literals
   of the row is the AND, over layers, of each component on its own input. An empty component
   contributes True.
2. **The feedback is the convolutional feedback on that row.** Drawing one matching node and
   updating every layer from it is exactly what the convolutional machine does with one
   matching patch and its literal row. (The code sums the drawn rows as one sparse product of
   "events per clause and node" with the node inputs, which gives the same counts without
   copying a row per event.)

So a `GraphTsetlinMachine` stores one `ta_state` of shape `(C, 2·(F + (D−1)·M))`, like every
other model, and reuses the convolutional feedback code. Only evaluation is new, because an inbox
depends on the previous layer's matches at the neighbours. It runs layer by layer: match
component `i` on the inputs of layer `i`, AND it into the running match, and send the result
along the edges (a scatter-OR, then a matrix product with the shifted message positions). The
derivation is in `plans/graph-tm.md`.

Two consequences:

* **Depth 1 *is* the convolutional coalesced machine.** With no message layers and the graphs of
  `GraphBatch.from_images` (one node per patch, no edges), a `GraphTsetlinMachine` given the same
  automata, weights and random stream learns bit for bit like `ConvCoalescedTsetlinMachine`.
  The test suite checks this.
* **Batched feedback works as for other models** ([batching](batching.md)). A batch of graphs is
  padded to its largest node count and evaluated in one pass; `feedback_mode="sequential"` is the
  reference implementation's one graph at a time.

### Where torchtsetlin differs from the reference code

| Point | Reference GraphTsetlinMachine | torchtsetlin default | To match the reference |
|---|---|---|---|
| Random node per clause | one per (graph, clause), shared by all classes and by Type I and II | one per feedback event | `shared_node_draw=True` |
| Negative classes | every class, scaled by `q / (K−1)` | the target and one sampled negative class | `multi_label=True`, one-hot targets, `negative_scale = q / (K−1)` |
| Order | one graph at a time | batched | `feedback_mode="sequential"` |
| Empty clause when predicting | votes | does not vote; its messages still flow | — |

## When to use it

* **Relations matter more than positions.** In a sequence, "an A whose left and right
  neighbours are A" is one depth-2 clause, wherever it occurs. A flat machine needs one clause per
  position, and a convolutional one needs a window wide enough to hold the whole pattern.
* **Several views of one example.** Different encodings of the same data can be different
  nodes with their own symbols, so one graph can carry, say, a thresholded image and a colour
  thermometer side by side. The paper attributes its CIFAR-10 result (70.28% vs 66.42% for the
  convolutional machine, both at depth 1) to this.
* **Variable-size input.** Graphs in a batch may have different numbers of nodes and edges.

Depth costs time. Every extra layer adds a pass over the edges and `M` automata per clause; keep
`depth` at the number of hops a pattern really needs.

## References

* O.-C. Granmo et al. *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With
  Graphs.* arXiv:2507.14874, 2025 (v2 2026). Reference implementation:
  [cair/GraphTsetlinMachine](https://github.com/cair/GraphTsetlinMachine).
* O.-C. Granmo, S. Glimsdal, L. Jiao, M. Goodwin, C. W. Omlin, G. T. Berge. *The Convolutional
  Tsetlin Machine.* arXiv:1905.09688, 2019.
* S. Glimsdal, O.-C. Granmo. *Coalesced Multi-Output Tsetlin Machines with Clause Sharing.*
  arXiv:2108.07594, 2021.
* V. Halenka et al. *Exploring Effects of Hyperdimensional Vectors for Tsetlin Machines.*
  arXiv:2406.02648, 2024.
