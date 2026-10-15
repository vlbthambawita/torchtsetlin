# Library gaps found while building on torchtsetlin

`src/torchtsetlin/**` is read-only in this programme (C1). Gaps are recorded here with a
minimal reproduction, impact and a fix sketch; they are worked around in `code/`.

| ID | Gap | Impact | Work-around | Fix sketch |
|---|---|---|---|---|
| IMN-LG-001 | `ThermometerEncoder` uses the same number of thresholds for every column; on tabular data binary/low-cardinality columns get duplicate thresholds (constant or repeated bits). | Wasted literals; duplicated bits in clause explanations. | `code/booleanize.py::TabularBooleanizer` (unique quantile cuts per column, cut midpoints for ≤ `n_bits + 1` distinct values, bit→column map). | A `ragged=True` mode returning a flat bit vector plus a `column_of_bit` index. |
| IMN-LG-002 | No public accessor for the signed per-clause vote weight or for unclamped votes. | Building a clause basis needs the private `_signed_weights()` / `_votes(clamp=False)`. | Read-only use of the private helpers (D1), pinned by `tests/test_imn.py::test_clause_basis_reproduces_tm_votes`. | Public `clause_vote_weights()` and `votes(x, clamp=False)`. |
