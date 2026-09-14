# Batched vs sequential feedback

The classical Tsetlin machine processes **one example at a time**: evaluate the clauses,
give feedback, update the automata, move on. On a GPU that leaves most of the hardware idle,
so `torchtsetlin` offers two ways to consume a mini-batch.

## `feedback_mode="batch"` (default)

All `B` examples of the batch are evaluated against the *current* automata. For every
(example, clause) pair the algorithm decides whether the clause gets Type I or Type II
feedback exactly as in the sequential algorithm. The resulting feedback *events* are then
**counted** per (clause, literal):

* `n_true[j, k]` — Type Ia events where literal `k` was True (memorise candidates),
* `n_false[j, k]` — Type Ia events where literal `k` was False (forget candidates),
* `n_ib[j]` — Type Ib events (forget everything),
* `n2[j, k]` — Type II events where literal `k` was False.

These counts are two small matrix products, and the stochastic transitions become
**binomial draws**: `Binomial(n_true, (s-1)/s)` increments, `Binomial(n_false + n_ib, 1/s)`
decrements, and `min(n2, N - state)` deterministic Type II increments (an automaton is never
pushed past the include boundary by Type II, exactly as in the sequential algorithm where the
clause stops matching once a literal is included). The automata are updated once per batch.

With `B = 1` this is the classical algorithm. For `B > 1` it is an approximation in the same
sense as mini-batch SGD: the examples of a batch do not see each other's updates. Empirically
(Noisy XOR, 20 clauses, `T=15`, `s=3.9`, `N=50`, 2000 training examples, 3 seeds):

| batch size | epochs | updates | test accuracy |
|---|---|---|---|
| 1 (sequential) | 40 | 80 000 | 0.95 – 0.99 |
| 10 | 100 | 20 000 | 0.93 – 0.99 |
| 50 | 200 | 8 000 | 0.88 – 0.99 |
| 200 | 400 | 4 000 | 0.77 – 0.89 |

Small and medium batches (≈ 8–64) keep the accuracy of the sequential algorithm while
running one to two orders of magnitude faster on a GPU; very large batches need more epochs.
Clause weights (weighted and coalesced models) change by at most the number of events in the
batch, so also prefer moderate batch sizes there.

## `feedback_mode="sequential"`

```python
model = tt.TsetlinMachine(..., feedback_mode="sequential")
# or per call
model.update(x, y, sequential=True)
```

Examples are processed one by one (still vectorised over all clauses and literals, so it is
fast for large models on a GPU, just not as fast as batching). Use it when you want the
exact reference dynamics, for small data, or to validate a batched configuration.

## Chunking

Independently of the feedback mode, a batch is split into chunks so that the intermediate
tensors (`(chunk, patches, clauses)` for convolutional models) respect
`max_chunk_elements`. Chunking never changes the result of a batched update; it only bounds
memory.

## Practical guidance

* Start with a batch size of 16–64 for flat models and 8–32 for convolutional models.
* If accuracy lags the sequential algorithm, halve the batch size or increase the number of
  epochs before touching `T` and `s`.
* The `update()` return value (vote sums before the update) gives you a free running training
  accuracy; the `Trainer` logs it as `batch_accuracy`.
