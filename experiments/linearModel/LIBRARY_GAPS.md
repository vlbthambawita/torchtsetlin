# Library gaps (rule C1)

`src/torchtsetlin/**` is read-only in this programme. Missing hooks or defects found while
working here are recorded below with a minimal reproduction, impact and a fix sketch.

## LG-1 (CLOSED 2026-10-17 — not a library defect) — unweighted `ConvTsetlinMachine` plateaus on MNIST

- **Repro:** `ConvTsetlinMachine(10, 200, T=100, s=10, patch_size=10, position_encoding=True,
  input_shape=(1,28,28))`, MNIST `x > 0.5`, 50k train / 10k val (`manifests/split_seed0.json`),
  batch 32, `feedback_mode="batch"`, seed 0. Val accuracy 0.872, 0.848, 0.861 (epochs 1–3) and
  0.85–0.87 for 30 epochs (`results/source/conv_M200_T100_s10_*`). Same with `weighted=True`:
  0.952, 0.963, 0.965.
- **Why it might be a gap:** the original CTM (Granmo et al., 2019) is unweighted and reports
  high MNIST accuracy, so a 10-point gap from weighting alone is surprising. Could be batch-mode
  fidelity for conv feedback, `T` scaling, or expected behaviour at this clause count.
- **Impact here:** none after D10 (conv sources are weighted).
- **Next step (outside this programme):** compare `feedback_mode="sequential"` on a 5k subset.

### Resolution (2026-10-17)

**Not a defect; no library change.** The premise was wrong: the original CTM (arXiv:1905.09688)
reports its MNIST numbers **with integer clause weighting on** (8 000 clauses/class, T=10 000, s=5;
`experiments/convtm/LITERATURE_TM.md` row A1). There is no published unweighted CTM result to
match, and a weighting gap of this size agrees with what the transformer_like programme measured
(conv 93.7 → 98.9 with weights).

Diagnostics (`code/lg1_diag.py`, records in `results/lg1/*.json`, seed 0, `split_seed0`, 3 epochs
full / 5 epochs on the first 5k training examples, val accuracy per epoch):

| run | change vs repro config | val acc | mean literals/clause |
|---|---|---|---|
| `full_base` | — (repro) | 0.877 0.877 0.854 | 89.0 |
| `sub_batch` / `sub_seq` / `sub_b1` | 5k subset; batch 32 / sequential / batch 1 | 0.847 / 0.840 / 0.840 (ep 5) | 69.4 / 68.7 / 68.7 |
| `full_w` | `weighted=True` | 0.952 0.963 0.965 | 19.9 |
| `full_T25` | T=25 | 0.922 0.921 0.926 | 72.0 |
| `full_s5` | s=5 | 0.900 0.897 0.896 | 89.2 |
| `full_M1000_T500(_s5)` | M=1000, T=500 (s=5) | ≈0.90 / ≈0.91 | ≈89 |
| `full_M2000_T1000_s5` | M=2000, T=1000, s=5 | 0.916 0.914 0.912 | 89.2 |
| `full_noboost` | `boost_true_positive=False` | 0.853 0.874 0.872 | 86.5 |
| `full_budget32` | `max_included_literals=32` | 0.803 0.807 0.807 | 32.0 |
| `full_nopos` | `position_encoding=False` | 0.819 0.814 0.815 | 61.5 |

Findings:

1. **Batched feedback is not the cause.** Sequential and batch-32 end within 0.01 of each other on
   the 5k subset and grow clauses identically; batch 1 reproduces sequential bit-for-bit (same
   RNG stream), which also checks the two code paths against each other.
2. **The plateau is clause bloat, which is inherent to the unweighted algorithm here.** Unweighted
   clauses grow to ~89 of at most 136 includable literals (a near-memorised 10x10 patch plus its
   position) whatever M or s is, and accuracy stays at 0.85–0.92. Clause count helps only a little
   (M=2000: 0.916). Type II on a conv clause only excludes *one* matching patch per event, so
   a clause that fires in many positions can only be silenced by growing. Weighting provides a
   second, much faster channel (Type II takes away vote weight directly), and with it clauses stay
   at ~20 literals.
3. **Lower T/M is the only unweighted lever that helps.** T=25 (T/M = 0.125) gives +5 points,
   which matches the transformer_like finding that lower T/M helps flat models. s, boost,
   the clause budget and dropping position encoding do not help.

**Impact:** none. D10 (weighted conv sources) stands. Use `weighted=True` for any convolutional
TM, or tune T well below M/2 if an unweighted machine is required.

