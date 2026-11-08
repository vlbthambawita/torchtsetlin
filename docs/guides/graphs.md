# Graph input

`GraphTsetlinMachine` learns from batches of graphs. This guide shows how to build them, train,
and read the learned clauses. The ideas behind it (deep clauses, messages, edge types) are in
[Graph Tsetlin machines](../concepts/graph-tm.md).

## Symbols and node features

Every node carries a Boolean feature vector of the same length `F`. The usual way to build one
is a `SymbolTable`: each symbol (a letter, a word, a quantised value, a coordinate) owns a few
positions of a sparse hypervector, and a node's vector is the OR of its symbols' vectors.

```python
from torchtsetlin.data import SymbolTable

symbols = SymbolTable(["A", "B", "C"], encoding="one_hot")           # F = 3, exact
words = SymbolTable(vocabulary, dim=1024, bits=4, encoding="random")  # F = 1024, hashed
words.vector(["good", "movie"])     # (1024,) bool: the bundle of two symbols
words.decode(v)                     # symbols whose positions are all set in v
```

One-hot tables keep every symbol its own feature, which makes clauses exactly readable; hashed
tables (`"random"`, `"double_hashing"`) keep `F` fixed for large vocabularies. Node features
need not come from a symbol table: any Boolean vector works, such as an image patch with its
position bits.

## Building a `GraphBatch`

=== "Sequences"

    ```python
    from torchtsetlin.data import GraphBatch

    g = GraphBatch.from_sequences([["A", "B", "A"], ["C", ["A", "B"]]], symbols)
    ```

    One node per element (an element may be a set of symbols). Node `k` sends to `k − 1`
    over edge type 0 (`"left"`) and to `k + 1` over type 1 (`"right"`).

=== "Images"

    ```python
    g = GraphBatch.from_images(x, patch_size=10, stride=1)   # x: (B, Z, H, W) Boolean
    ```

    One node per patch and no edges: the patch pixels plus thermometer-coded coordinates,
    exactly the features of `ConvTsetlinMachine(patch_size=10)`. At depth 1 this graph machine is
    the convolutional coalesced machine.

=== "Any graph"

    ```python
    g = GraphBatch.from_graphs(
        node_features=[f0, f1],                  # per graph, (n_i, F) Boolean
        edges=[[(0, 1), (1, 0)], [(0, 2)]],      # per graph, (source, destination) pairs
        edge_types=[[0, 1], [0]],                # per graph, one type per edge
        node_types=None,                         # optional, per graph (n_i,)
        n_edge_types=2,
        edge_type_names=["parent", "child"],
    )
    ```

    Edges are directed: a node hears from the nodes with an edge *into* it, so an undirected
    relation needs an edge each way, usually with two types.

A `GraphBatch` behaves like a tensor along the graph axis: `len(g)`, `g[3]`, `g[2:10]`,
`g[torch.tensor([4, 1])]`, `g.to("cuda")` and `GraphBatch.cat([...])`. Nodes are stored padded to
the largest graph of the batch, with each graph's node count in `g.n_nodes`.

## Training

```python
import torchtsetlin as tt

model = tt.GraphTsetlinMachine(
    n_features=symbols.dim,      # F, the node feature count
    n_outputs=3,                 # classes
    n_clauses=40, T=200, s=3.0,
    depth=2,                     # layers: each one reaches one hop further
    message_encoding="one_hot",  # or "random" / "double_hashing" with message_size=...
    n_edge_types=2, edge_type_names=["left", "right"],
    max_included_literals=8,     # important with message layers, see below
)
trainer = tt.Trainer(model, device="cuda", batch_size=25)
trainer.fit((g_train, y_train), epochs=10, val_data=(g_test, y_test))
```

`update(g, y)`, `forward(g)`, `predict(g)` and the `Trainer` work as for every other model. The
other `CoalescedTsetlinMachine` options (`multi_label`, `max_included_literals`,
`feedback_mode`, ...) carry over.

Choosing the settings:

* **`depth`** is the number of hops a pattern spans, plus one: "an A with an A on each side"
  needs 2. Each extra layer costs a pass over the edges and `message_size` more automata per
  clause.
* **`max_included_literals`.** Use a budget with message layers. At the start every clause
  matches everywhere and sends messages from every node, and one boosted Type Ia event can
  include all of them; without a budget, clauses can collapse into patterns that hold in every
  graph (see [learning](../concepts/graph-tm.md#learning)).
* **`message_encoding`.** `"one_hot"` gives each (clause, edge type) its own inbox bit:
  collision free and readable, but the inbox has `n_clauses · n_edge_types` bits per layer. For
  thousands of clauses use `"random"` with `message_size` a few times
  `n_clauses · n_edge_types · message_bits` and check `model.message_collision_rate()`.
* **`s`** may be one value per layer, `s=(s0, s1, …)`. With `s` close to 1 use
  `feedback_mode="sequential"`: forgetting then has probability close to 1, and aggregated
  over a batch it wipes the clauses (the reference noisy-XOR demo's `s=1.0` learns only one
  graph at a time; `s=2` learns it batched).
* **`n_node_types`**: clause `j` only matches nodes of type `j mod n_node_types` (as in the
  reference implementation), which splits the clauses between kinds of node.
* **To reproduce the reference code**, use `feedback_mode="sequential"`,
  `shared_node_draw=True`, and for several classes `multi_label=True` with one-hot targets and
  `negative_scale=q / (K − 1)` (see [the differences](../concepts/graph-tm.md#where-torchtsetlin-differs-from-the-reference-code)).

### Speed on the GPU: pad the dataset

On CUDA, pad the whole dataset to one shape before training:

```python
P = max(g_train.shape[1], g_test.shape[1])                                   # nodes
E = int(max(g_train.edges_per_graph.max(), g_test.edges_per_graph.max()))   # edges
g_train, g_test = g_train.pad(P, E), g_test.pad(P, E)
```

A padded batch keeps that shape when it is sliced, without looking at the data, so updates can
use the fused kernels and be replayed from CUDA graphs. On the sequence example (RTX 3090):

| Update | Flat batches | Padded batches |
|---|---|---|
| batched (25 graphs) | ~0.23 ms per graph | ~0.013 ms per graph |
| sequential (one graph at a time) | ~4.8 ms per graph | ~0.065 ms per graph |

(smoke measurements, not benchmarks). The results are the same in distribution; padding nodes
and unused edge slots never match or send. What each layout gets:

* **Batched updates** run in events mode (one Triton commit kernel) for flat and padded
  batches. Padded batches are also replayed from CUDA graphs.
* **Sequential updates** use the compiled single-example commit and CUDA graphs only for padded
  batches, because both are built per input shape.
* **Depth 1 with full graphs** (every graph has all `P` nodes, as with `from_images`) runs on the
  convolutional machine's bit-packed kernel: the model is then exactly a convolutional
  coalesced machine over the node features, and as fast.
* Pad to **one** shape: every distinct `(P, E)` is compiled and captured on its own. A per-layer
  `s` or `budget_mode="hard"` keeps the eager paths. On the CPU nothing changes.

## Reading the clauses

```python
for j in range(model.n_clauses):
    print(model.clause_expression(j, symbols=symbols), model.weights[j].tolist())
```

prints deep clauses in the paper's notation, for example

```text
A AND left1:0 AND right1:0                [-58, 63]
```

"this node is an A, and clause 0's first component matched at the node a *left*-going edge came
from (the right neighbour) and at the node a *right*-going edge came from (the left
neighbour)". `left1:0` names the edge type, the message layer and the clause the message reports.
With a symbol table, node literals are decoded to symbols; hashed messages are decoded when all
of their bits are included and printed raw (`m1[17]`) otherwise.

`model.rules(symbols=symbols)` gives the same as `IF … THEN` rules with the votes.

`model.explain_node(g, clause, graph=i, symbols=symbols)` explains one clause at one node of
graph `i` (by default the first node where it holds), tracing every message back to the
neighbours that sent it:

```text
clause 1 holds at node 2 [A] of graph 0
  layer 0: A
  layer 1: left1:0 AND right1:0
    left1:0 from node 3 [A] over left: clause 0 up to layer 0
      layer 0: A
    right1:0 from node 1 [A] over right: clause 0 up to layer 0
      layer 0: A
```

A component that fails is marked `<- fails here`.

Two accessors expose what happens inside one graph:

* `model.node_clause_outputs(g, layer=i)`: `(B, P, C)`, where clause components `0 … i` all
  match. With the default last layer this says *which nodes* make a clause True.
* `model.node_messages(g, layer=i)`: `(B, P, M)`, the inbox of every node at layer `i ≥ 1`.

`model.evaluate_clauses(g)` gives the `(B, C)` clause outputs, usable as Boolean features for
another model.

## Example

[`examples/graph_sequence.py`](https://github.com/vlbthambawita/torchtsetlin/blob/main/examples/graph_sequence.py)
counts runs of consecutive A's in sequences of varying length, the reference implementation's
sequence demo, and prints the learned deep clauses.

The notebook [Graph Tsetlin machine](../examples/notebooks/06_graph_tsetlin.ipynb) walks through
the same ideas: the paper's three-A's task with four readable clauses, `explain_node`,
run lengths, the padded fast path, and MNIST as graphs learning bit for bit like
`ConvCoalescedTsetlinMachine`. It opens in Google Colab.
