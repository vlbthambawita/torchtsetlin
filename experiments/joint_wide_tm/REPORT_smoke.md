# joint_wide_tm — smoke

Test accuracy (%) at the best-validation epoch; `val` = that validation accuracy. Composite = members trained independently, scale from the validation set (`valalpha`); `fixed` = v/T; `paper` = test-set scale (transductive, reference).

## Team S: H3 + H5 + H10

| Seed | H3 | H5 | H10 | Composite (val) | fixed | paper | J-sum (val) | J-one (val) | Jown | W-sum | W-one |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 73.80 | 72.40 | 74.00 | 76.00 (73.80) | 76.00 | 75.60 | 65.00 (70.00) | 56.00 (62.20) | 72.60 | 59.40 (H10) | 65.80 (H10) |

J-sum − composite: Δ = -11.00 pp over 1 seed(s) (per seed -11.00); mean J-sum 65.00 vs W-sum 59.40.

J-one − composite: Δ = -20.00 pp over 1 seed(s) (per seed -20.00); mean J-one 56.00 vs W-one 65.80.

## Team P: A10 + H3 + H4

| Seed | A10 | H3 | H4 | Composite (val) | fixed | paper | J-sum (val) | J-one (val) | Jown | W-sum | W-one |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 76.60 | 73.80 | 76.20 | 76.60 (73.80) | 76.20 | 76.20 | 65.00 (69.60) | 61.80 (55.80) | — | — | — |

J-sum − composite: Δ = -11.60 pp over 1 seed(s) (per seed -11.60).

J-one − composite: Δ = -14.80 pp over 1 seed(s) (per seed -14.80).

