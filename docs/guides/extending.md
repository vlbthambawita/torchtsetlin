# Extending torchtsetlin

The models are thin layers over `torchtsetlin.functional`. To build a new variant, subclass
`TsetlinMachineBase` and implement the *output layer* and *feedback policy*:

```python
class MyTM(tt.TsetlinMachineBase):
    def __init__(self, n_features, n_clauses, T, **kw):
        self.T = T
        super().__init__(n_features, n_clauses, **kw)

    def _votes(self, clause_out, clamp=True):            # (B, C) bool -> (B, n_outputs)
        ...

    def _coerce_targets(self, y, n, device):             # validate / convert targets
        ...

    def _select_feedback(self, votes, y, clause_out):    # -> (type_i, type_ii, aux) masks (B, C)
        ...
```

Optional hooks: `_accumulate_weights(acc, aux, clause_out, sel_fire_i, sel_fire_ii)` and
`_commit_weights(acc)` for clause weights; `_encode`, `_evaluate`, `_feedback_counts` for a
different input structure (this is how the convolutional mix-in works); `rules()` for
interpretation.

The generic `update()` handles input coercion, lazy initialisation, chunking, dropout masks,
counting feedback events and applying them with `functional.apply_feedback`.

## Using the functional API directly

```python
from torchtsetlin import functional as F
lit = F.to_literals(x)                                   # (B, 2F)
include = (state >= N).float()
c = F.clause_outputs(lit, include, empty_value=True)     # (B, C)
v = F.vote_sums(c, weights, T)
n_true, n_false, n_ib = F.type_i_counts(lit, sel_fire_i, sel_nofire_i)
n2 = F.type_ii_counts(lit, sel_fire_ii)
F.apply_feedback(state, n_true, n_false, n_ib, n2, n_states=N, s=s)
```

Everything is plain tensor code, so `torch.compile`, custom CUDA kernels or alternative
sampling schemes can be dropped in.
