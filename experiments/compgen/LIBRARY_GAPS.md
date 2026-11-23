# compgen — library gaps

`src/torchtsetlin/**` is read-only in this programme (README R1). Defects or missing features found
while working here are recorded here, not fixed in place.

| # | What | Workaround | Status |
|---|---|---|---|
| CG-1 | No way to restrict a clause to one kind of literal. The benchmark's reference setting (negated input, negated literals not appended) could only be emulated through the private literal-dropout mask, which also switched off CUDA-graph replay (sequential: 16.5 s per epoch instead of 1.0 s on an RTX 3080). | — | **fixed in 0.7.4** (GitHub #23, `literals="negated"`), on the user's decision |
| CG-2 | No one-output (binary) classifier with unweighted clauses. The reference model has one output with 200 positive / negative clauses; `TsetlinMachine` needs two classes (2 × 100 clauses, two class sums), and `CoalescedTsetlinMachine(n_outputs=1, multi_label=True)` always learns clause weights. | arm `neg` uses two classes × 100 clauses | open (note) |
