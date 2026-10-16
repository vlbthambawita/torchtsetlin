# Decisions and deviations (rule C4)

Spec: `boolean-gated-linear-tsetlin-implementation.md` (frozen). Plan: `PLAN.md`.
Library revision at start: `3cede48c9e7b9761e30976ea073c6c5aecd2be15` (`torchtsetlin` 0.3.1),
2026-10-15: HEAD moved to `e805000` mid-session (commit by the user; `src/` unchanged, `git diff --stat src/` empty).
`git diff --stat src/` empty. (PLAN.md was written against `06a2ce5`; the commits since touch
only `experiments/IMN`, not `src/`.)

## 2026-10-15 — day 1 (PLAN §2.2)

- **D1 Encoder threshold.** Primary `Binarizer(0.5)` (spec §3.1, `x > 0.5`). Secondary source TM
  at `0.3` (`M = 20`, A0/A1/A3 only) to check threshold-specificity.
- **D2 Clause polarity.** Dropped from the hybrid output as in spec §4.2; kept as metadata and
  reported next to the learned intercept sign as a diagnostic only.
- **D3 Sweep budget.** Integration run: 10 sweeps (cost probe). Main runs: `max_sweeps = 2000`,
  stop on `NO_ACCEPTED_MOVES`; budget stops reported as such.
- **D4 Gram-form optimizer** used only after the Phase 4 equivalence gate; spec-literal
  reference optimizer is the ground truth.
- **D5 Plan-added arms** A1c/A2c (continuous ridge), A5 (random gates), A3-L (integer line
  search) are reported in separate, labelled rows.
- **D6 Convolutional TM (user request, 2026-10-15).** The user asked to test the traditional TM,
  the convolutional TM and the new idea on MNIST. PLAN §11 kept convolutional *linear outputs*
  (spec §17.1 pooled form) out of scope; that stays out. Added instead:
  - **A0c** — `tt.ConvTsetlinMachine` source model (10x10 patches, stride 1,
    `position_encoding=True`, same `Binarizer(0.5)`, same split and selection rule), at the same
    clause counts `M` as the flat source TMs plus a larger reference size.
  - **A3c / A1c-conv** — the hybrid of spec §4.1 *unchanged*, with gates `C_kj(x)` = the ConvTM
    clause's own output (OR over patch positions, prediction semantics) taken from
    `evaluate_clauses`. A gate is still a fixed Boolean function of `x`, so the fixed-design
    algebra of spec §9 holds exactly; no pooling of linear outputs over patches is introduced.
    Adapter deployment mode: *library evaluator* (spec §15.2) — gates come from the pinned
    library, not from a literal exporter, because a conjunction exporter cannot reproduce the OR
    over patches (spec §6.2 last bullet). Reported as separate rows.
- **D7 Gate encoding for the hybrid.** The frozen-rule evaluator (`adapter.evaluate_frozen`)
  re-implements spec §6.3 from exported literals for flat TMs and is checked element-wise
  against `evaluate_clauses` (G1). Prediction and fitting then use the frozen-rule evaluator.

## 2026-10-15 — gates G0–G5

- **G0–G4 passed** (`pytest experiments/linearModel/tests -q`: 33 passed, CPU + CUDA). Manifests
  regenerate bit-identically. On random small problems the spec-literal transcription
  (`code/spec_literal.py`), the numba reference engine and the Gram engine accept *identical*
  move sequences for both the one-state rule and A3-L. The MNIST-integration part of G4 runs in
  Phase 6 (`results/integration_*.json`).
- **G5 (regression check) — violation found and investigated** (`results/phase5_regression.json`).
  The continuous fit respects `M h^2/8` for every `h`, and the continuous fit *rounded to the
  state grid* respects `M h^2/8 + delta` (representability is fine). The spec's one-state
  optimizer — and A3-L — stop at a coordinate-stationary state that violates the bound at
  `h = 0.25` (sup error ~3.4x the bound, training objective ~40x the continuous one). Cause:
  coordinate trap (spec §13.2, §16 "coordinate-local limitation"). For a gate far from
  `x = 0`, its intercept column and slope column are nearly collinear, and the integer grid
  has no improving single-coordinate move. Checked directly: from the terminal state a joint
  `(b_j, w_j)` move lowers the objective. This is an optimizer limitation, not an
  implementation bug, and it is relevant to MNIST, where pixels are near-binary (`C_j` and
  `C_j x_f` are nearly collinear wherever pixel `f` is saturated under the clause).
- **D8 Plan addition: A4q (rounded ridge).** The A4 continuous solution rounded to the state
  grid and clipped to `[-Q, Q]`, no optimization. It is not an optimizer result (spec §17.2
  lists ridge initialization as a separate experiment). It is reported as a diagnostic row so
  that Q4 can be split into a *quantization* part (A4 vs A4q) and an *optimizer* part (A4q vs
  A3).

## 2026-10-15 — Phase 6 integration run (`results/integration/`, `results/integration_summary.json`)

- **G4 (MNIST part) passed.** On the 2k/1k integration set, ref and Gram engines made identical
  accepted-move sequences for 10 sweeps in all 8 paired runs (A1, A2, A3, A3-L; flat, conv and
  random gates), identical final objectives, max per-sweep drift ~1e-16. From here on the Gram
  engine is used (D4).
- **Coverage observation (not a result, M = 2 debug models).** With 2 clauses per class, on
  average 0.32 of a class's gates fire per training example (random gates: 0.24), so most
  examples see only `beta_k` for most classes. Even the continuous optimum on these gates (A4) is
  far below the global linear model A2. The hybrid's ceiling is bounded by gate coverage; this is
  what the main grid over `M` measures.
- **A5 calibration** is imperfect at M = 2 for clauses with many literals (rate 0.003 vs 0.027
  target in class 0). The per-run `gate_stats.random_match` report is carried into every A5 record
  and must be shown next to A5 numbers.
- **D9 Source-TM grid for reference sizes.** The PLAN grid `T in {M/2, M, 2M}` is used for
  `M in {2, 10, 20}` (flat and conv). For the reference curve (flat `M in {50, 100, 500}`) it would
  put `T` far above usual MNIST settings, so those use `T in {25, 50, 100}`, `s in {5, 10}`. The
  conv reference `M = 200` uses `T in {50, 100, 200}`, `s = 10`, unweighted. A0 only, selection on
  validation as everywhere else.

## 2026-10-15 — D10 conv sources are weighted

The first conv source grid (unweighted, as the flat models) plateaued far below the library's
documented conv MNIST numbers: `M = 200, T = 100, s = 10, Binarizer(0.5)`, batch 32, seed 0
gave val 0.872 / 0.848 / 0.861 over epochs 1–3 unweighted, against 0.952 / 0.963 / 0.965 with
`weighted=True` (diagnostic, not a recorded run). The library's conv example and benchmark use
`weighted=True`. From now on **all conv source TMs (A0c and the gates of A1/A3/... on conv) use
`weighted=True`**. Clause weights do not enter the hybrid (spec §4.2; metadata only), so the
gate semantics are unchanged. The unweighted conv records (`results/source/conv_M*`, no `_w`)
are kept and labelled superseded; they are not used. Flat sources stay unweighted (PLAN).
The unweighted plateau is logged in `LIBRARY_GAPS.md` as an open observation.

## 2026-10-15 — Gate G6: cost measurements and the main grid (`results/cost_probe/`)

Measured at `N = 50,000`, `M = 20`, `D_total = 157,010`, on M = 20 conv gates (an unweighted
probe source, used for cost only). Hardware: 32 cores shared with the source-TM jobs.

| item | measured |
|---|---|
| Gram build, dense row chunks | 509–601 s, 27 GB peak RSS |
| Gram build, block-sparse (co-active rows; D11) | 66 s, max abs diff to dense 1.9e-14 |
| Gram-engine sweep | ~1.15 s |
| Reference-engine sweep | ~5 s |
| A3 (`1/64, 256, 1e-3`) to `NO_ACCEPTED_MOVES` | 31 sweeps, 35 s; 0 % at `±Q`, 78 % zero |
| A3-L same config | 46 sweeps, 57 s |
| A4 ridge (Cholesky, 10 outputs) | 56 s; `max |theta| = 0.33` |

- **D3 revised by measurement:** runs end by `NO_ACCEPTED_MOVES` long before 2000 sweeps, so the
  sweep budget is not a cost driver. The cap stays at 2000.
- **Kill criterion** (stop < 50 sweeps with > 30 % at `±Q`) **not triggered** (0 % at the bound).
  Instead the coefficient *resolution* is binding (most optimal coefficients are below
  `delta / 2`); the preregistered `(delta, Q)` grid already covers `delta = 1/256`.
- **D11 block-sparse Gram.** `build_gram(..., blocked=True)` (default) is tested equal to the
  dense build (`tests/test_opt.py::test_gram_matches_dense`). The G4 integration equivalence was
  re-run with it: identical move sequences in all 8 pairs (`results/integration/`; the dense-Gram
  run is kept in `results/integration_densegram/`).
- **Main grid decision.** `M in {2, 10, 20}` × {flat, conv}, full preregistered grid for every
  integer-search arm (A1, A3, A3-L, A5; 9 configs each) and `lambda` grids for A4/A1c/A2c.
  A4q is computed for every `(delta, Q, lambda)`. Each arm selects its own hyperparameters on
  validation (seed 0). A5 is *reported at the A3-selected configuration* (PLAN); its full grid is
  run too because it is cheap. A5 exists for flat gates only: random gates are drawn as
  conjunctions over pixels, and there is no matching random construction for conv clauses (OR over
  patches) in this plan. Projected cost: < 10 CPU-core-hours plus ~6 GPU-hours of source-TM
  training, well inside §8.
- **`M = 50`**: all 10 Gram matrices in lockstep need 123 GB (> 121 GB RAM). Deferred. It is
  feasible later with a per-output fit plus stored per-sweep validation scores, run only if the
  `M <= 20` results make it worth doing.

- **D12 A5 random-gate construction fixed (2026-10-15, before selection).** The first seed-0 A5
  runs drew each clause's `L_c` literals *uniformly* from one anchor example's true literals. With
  `L_c` ~ 100-600 that gives conjunctions only the anchor satisfies: achieved mean training
  activation 0.0014 vs the source's 0.208 (flat M = 20), `frac_within_tol = 0` at every M, and A5
  validation accuracy ~0.12-0.16 (class prior only). That measured "gates off", not "matched random
  gates", so it failed the control's own preregistered matching requirement. Fix: sample the anchor's
  true literals without replacement with weight `freq(literal)**alpha` (Gumbel top-k), where `freq`
  is the label-free training frequency, and bisect `alpha` per clause towards the target rate (16
  rounds x 64 candidates). Achieved: `frac_within_tol` 1.00 / 0.96 / 0.985 and mean rate
  0.2013 / 0.2282 / 0.2090 vs target 0.2014 / 0.2280 / 0.2076 at M = 2 / 10 / 20 (draw 0). The
  construction still uses no labels. The invalid records, artifacts and logs are kept in
  `results/invalid/a5_uniform_seed0/` and excluded from selection and reporting; all A5 groups were
  rerun with the fixed construction before `select`.

- **Freeze (2026-10-16).** `configs/final.json` (57 entries, sha256 `da7bc116…0256`) committed as
  `a564197` before any test access (user-approved). The reference A0 rows (`*_seed0_ref`) were first
  missed by `select` (suffix filter) and added before the commit.
