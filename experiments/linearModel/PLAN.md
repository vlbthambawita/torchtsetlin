# Boolean-gated linear outputs for Tsetlin machines — experiment plan

Status: **planned, not started.** Written 2026-10-14 against `torchtsetlin` v0.3.1, revision
`06a2ce5`. Specification under test:
[boolean-gated-linear-tsetlin-implementation.md](boolean-gated-linear-tsetlin-implementation.md)
(called *the spec* below; section numbers like §9.2 refer to it).

There are no measurements in this document. Every number below is either taken from the spec,
derived arithmetic (marked *derived*), or a preregistered budget/threshold — never a result.

## 0. Ground rules

- **C1 — library is read-only.** `src/torchtsetlin/**` is not modified during this programme
  unless the user explicitly confirms otherwise. Missing hooks or defects go into
  `LIBRARY_GAPS.md` with a minimal reproduction, impact and a fix sketch. Record the
  `git rev-parse HEAD` and `git diff --stat src/` at the start and at each gate.
- **C2 — numbers come from records.** Every number in the report is read from
  `results/*.json` by a script. No hand-typed results.
- **C3 — test split is sealed.** The official MNIST test set is touched once per
  preregistered configuration, after selection is frozen (Phase 7). Every test evaluation is
  logged in `results/test_access_log.jsonl`.
- **C4 — spec is frozen.** Deviations from or clarifications of the spec go into
  `DECISIONS.md` (dated, with reason), not silently into code. §2 below lists the ones already
  known.
- **C5 — correctness before accuracy.** No MNIST accuracy run until the Phase 1–4 gates pass.

## 1. Objective and questions

Test the spec's idea: keep a trained TM's Boolean clauses frozen, replace each clause's fixed
vote by an affine function of the raw numerical input, and learn its coefficients as bounded
integer states with the exact full-loss coordinate search of §9–10.

| ID | Question | Answered by |
|---|---|---|
| Q0 | Is the implementation algebraically correct (adapter fidelity, loss-change identity, descent, save/load)? | Phases 1–4 test suite; pass/fail |
| Q1 | What do local numerical slopes add over fitted per-clause constants on the *same* gates? | A3 vs A1 |
| Q2 | Does the hybrid beat the source TM it was built from? | A3 vs A0 |
| Q3 | Are the conditional gates useful, or does a single global linear model do as well? | A3 vs A2 (and A2c) |
| Q4 | How much is lost through finite-state optimization/quantization? | A3 vs A4 (continuous ridge, same gates) |
| Q5 | Do *learned* TM clauses matter, or does any partition of similar coverage work? | A3 vs A5 (random gates) — **plan addition, not in spec** |
| Q6 | What does it cost (fit time, memory, serialized size, inference time)? | All arms, §8 |

All mechanisms are hypotheses. Q1–Q5 can come out negative; a negative answer is a valid
outcome and is reported with the same care.

## 2. Mapping the spec onto torchtsetlin (facts from code) and known deviations

### 2.1 Adapter contract (spec §6.1) → existing API

| Spec operation | torchtsetlin implementation | Notes |
|---|---|---|
| `TRAIN_OR_LOAD` | `tt.TsetlinMachine(784, 10, n_clauses=M, T, s)` + `tt.Trainer`, `ModelCheckpoint`; `torch.load` + `load_state_dict` | State is buffers (`ta_state`, `weights`); `_refresh_include` runs on load. |
| `LIST_OUTPUTS_AND_CLAUSES` | `clause_class = arange(C) // n_clauses` ([classifier.py](../../src/torchtsetlin/models/classifier.py)) | Bank `k` = clauses `[k*M, (k+1)*M)`; stable. |
| `GET_INCLUDED_LITERALS` | `model.included_mask()` → `(C, 2F)` bool | Columns `0..F-1` = positive literal `x_u`, `F..2F-1` = `NOT x_u` ([base.py](../../src/torchtsetlin/models/base.py)). Library appends complements itself — do not append twice (§3.2). |
| `GET_CLAUSE_METADATA` | `clause_polarity` (first `ceil(M/2)` positive), `weights` (int, used only if `weighted=True`) | Kept as metadata only (§4.2). |
| Empty-clause semantics | Prediction mode: empty clause = **False** (`evaluate_clauses` uses `empty_value=False`) | Recorded per rule as `empty_clause_output = 0`. Pinned by `tests/test_models.py`. |
| `EVALUATE_RAW_CLAUSES` | `model.eval(); model.evaluate_clauses(x_bool)` → `(B, C)` bool | Raw outputs before polarity/weights/clamp. Clause/literal dropout only affects feedback, not evaluation (checked: `_clause_active` used in `_accumulate_literals` only). |
| `PREDICT_ORIGINAL` | `model.eval(); model.predict(x_bool)` | Votes clamped to `[-T, T]`, argmax. |
| `SNAPSHOT_IDENTITY` | SHA-256 of `ta_state`, `weights`, `n_states`, shape, library version + git rev | Computed before/after freeze and after every coefficient fit. |

Every operation exists; no library change is expected. If one turns out to be missing, it goes
to `LIBRARY_GAPS.md` and is worked around inside `experiments/linearModel/code/`.

### 2.2 Clarifications and deviations to record in `DECISIONS.md` on day 1

| # | Item | Decision |
|---|---|---|
| D1 | Encoder threshold. Spec §3.1 proposes `x > 0.5`; the library's MNIST examples use `0.3`. | Primary: `Binarizer(0.5)` as in the spec. Run one declared secondary source TM at `0.3` (A0/A3 only) to check the conclusion is not threshold-specific. |
| D2 | Clause polarity. Spec drops it from the hybrid (§4.2). Each class bank therefore contains `ceil(M/2)` "for" and `floor(M/2)` "against" clauses, all now with signed free coefficients. | Follow the spec. Report polarity of each clause next to its learned intercept sign as a diagnostic only. |
| D3 | One-state moves vs. sweep budget. With `delta = 1/64`, reaching `|theta| = 1` needs ≥ 64 sweeps (*derived*: one move per coordinate per sweep). The spec's 10-sweep integration budget cannot approach stationarity. | Integration run keeps 10 sweeps (it is a cost probe). Main runs use `max_sweeps = 2000` with the spec's `NO_ACCEPTED_MOVES` stop; budget stops are reported as such. |
| D4 | Fast exact optimizer. A Gram-form implementation (§5.2 below) is algebraically identical to the spec's sequential search. | Spec-literal reference optimizer is the ground truth; the Gram form is used only after the equivalence gate (Phase 4) passes. |
| D5 | Plan-added arms not in the spec: A2c (continuous global ridge), A5 (random gates), A3-L (integer line search variant). | Reported in separate, labelled rows; never merged with spec arms. |

## 3. Arms

Every arm uses the same split, numerical scaling `x = pixel/255` (row-major, `d = 784`), the
same Boolean encoder (where it has gates), the same squared-error/one-hot objective with
`1/(2N)` normalisation (§9.1), and the same validation-selection key (§10.1).

| Arm | Gates | Output per class `k` | Optimizer | Source |
|---|---|---|---|---|
| A0 | Source TM clauses | Original clamped votes, argmax | TM feedback | spec §14.3 |
| A1 | Same frozen clauses | `beta_k + sum_j C_kj b_kj` (`P_kj = ∅`) | Integer-state search | spec §14.3 |
| A2 | None | `beta_k + sum_f w_kf x_f` | Integer-state search | spec §14.3 |
| A3 | Same frozen clauses | `beta_k + sum_j C_kj (b_kj + w_kj^T x)` | Integer-state search | **proposed** |
| A4 | Same frozen clauses | Same as A3, continuous | Closed-form ridge (§14.3) | spec §14.3 |
| A1c / A2c | as A1 / A2 | continuous | Closed-form ridge | plan addition |
| A5 | Random conjunctions, matched per bank to the source's literal-count distribution and calibrated to the same training activation rate (±10% per clause) | Same as A3 | Integer-state search | plan addition |
| A3-L | Same frozen clauses | Same as A3 | Integer **line** search: per visit, move to `clip(round(-g/(delta h)))` | plan addition (spec §17.2-style variant) |

A3-L is still exact full-loss descent (the coordinate objective is a 1-D quadratic, and the
integer minimiser is the rounded continuous one, then checked with the exact `DeltaJ`), but it is
*a different algorithm* from the spec's one-state rule and is reported separately. It exists to
separate "the finite-state idea is weak" from "one-state moves are slow".

## 4. Data protocol

- MNIST from `./data/MNIST` (already present; loaded via torchvision as in
  `tt.data.load_torchvision_boolean`, but the experiment keeps the **float** images too).
- Split: official 60k train pool → 50,000 train / 10,000 validation, stratified by digit,
  `split_seed = 0`. Indices + SHA-256 saved in `manifests/split_seed0.json`. Official 10k test
  sealed (C3).
- Integration subset (§14.2): 2,000 train / 1,000 validation, stratified, saved in
  `manifests/integration_seed0.json`.
- Encoder: `tt.data.Binarizer(0.5)` (D1); it has no fitted state, so there is no leakage risk,
  but its threshold and `inclusive=False` are stored in every artifact.
- Inputs validated finite; no augmentation (§3.3).

## 5. Implementation

All code lives in `experiments/linearModel/code/`; tests in `experiments/linearModel/tests/`
(run with `pytest experiments/linearModel/tests -q`; not part of CI).

### 5.1 Modules

| Module | Contents | Spec |
|---|---|---|
| `data.py` | Splits, manifests, float + Boolean views, one-hot targets | §3, §14.1 |
| `source_tm.py` | Train/select/save source TMs; seeds; provenance record | §7 |
| `adapter.py` | Spec §6.1 contract over `TsetlinMachine`; `FrozenRule`, fingerprints, `FREEZE_RULES` with probe-equivalence check | §6 |
| `hybrid.py` | `HybridModel` record, `BUILD_LAYOUT`, state decoding, `PREDICT_SCORES` (direct path), `DESIGN_COLUMN`, `SCORES_FROM_STATES` (column path), `SAVE_HYBRID` / `LOAD_HYBRID` | §5, §8, §15.2 |
| `opt_ref.py` | Spec-literal `FIT_COEFFICIENTS`, `BEST_ONE_STATE_MOVE`, `AUDIT_NEIGHBORS`, float64, lazy columns over active rows only | §9–10 |
| `opt_gram.py` | Gram-form exact optimizer (§5.2), same API and trace format | D4 |
| `ridge.py` | Continuous ridge solve for A1c/A2c/A4 via Cholesky of the same Gram; `mu_k = lambda_min(H_k)` for the §13.3 bound | §14.3, §13.3 |
| `gates_random.py` | A5 random-gate generator | D5 |
| `run.py` | One config → one `results/<run_id>.json` + artifact | — |
| `report.py` | Tables/figures from `results/*.json` only | C2 |

Storage: states `int16` (bound checks in `int32`); residuals, dot products, objective in
`float64` (§10.4). Gates cached as `bool` on the device; numerical inputs `float64` on CPU,
`float32` copies allowed only for prediction-time speed checks.

### 5.2 Fast exact optimizer (Gram form)

For output `k` let `H_k = A_k^T A_k / N + lambda I` and maintain `g_k = A_k^T R_k / N + lambda theta_k`.
A move `m` on coordinate `r` changes `R_k` by `delta m a_r`, hence

```
g_k  <-  g_k + delta * m * H_k[:, r]          # exact; includes the ridge diagonal
```

so each coordinate decision is O(1) (`g_r`, `H_rr` are already known) and each accepted move
is one O(D_k) axpy. This makes the same decisions, in the same random coordinate order, as the
spec's residual form in exact arithmetic. Once per sweep, `R` is rebuilt from the states and `g`
recomputed from scratch (the spec's drift check, §10.2), and the drift between the maintained
and rebuilt `g` is logged.

`H_k` has block structure `H[(j,f),(j',f')] = (1/N) sum_i C_ij C_ij' x_if x_if'`, so each block
is a Gram of `X` restricted to the rows where both clauses fire. Build it block-wise in float64
over co-active rows (GPU, chunked). Sizes (*derived*, `D_k = 1 + M(785)`):

| Clauses per class `M` | `D_k` | `D_total` | `H_k` float64 | all 10 `H_k` |
|---|---|---|---|---|
| 2 | 1,571 | 15,710 | 19.7 MB | 0.20 GB |
| 10 | 7,851 | 78,510 | 0.49 GB | 4.9 GB |
| 20 | 15,701 | 157,010 | 1.97 GB | 19.7 GB |
| 50 | 39,251 | 392,510 | 12.3 GB | 123 GB — process one output at a time, outputs are separable (§10.2) |

The coordinate loop runs in `numba` (0.62.1 is installed) on CPU. Outputs are independent and
run in parallel processes. `M = 50` is attempted only if Phase 6 cost measurements allow it.

## 6. Phases and gates

### Phase 0 — scaffolding (½ day)

`DECISIONS.md` (D1–D5), `LIBRARY_GAPS.md` (empty), `README` pointer, directory layout (§9),
source-tree hash recorded, manifests written.
**Gate G0:** manifests reproducible from seed (re-generate → identical SHA).

### Phase 1 — adapter and frozen rules (1 day)

Implement `adapter.py`. Tests (spec §16 rows "Encoder boundaries", "Complement mapping",
"Adapter fidelity", "Empty/contradictory clauses"):

1. Values `0.5 - eps`, `0.5`, `0.5 + eps` encode as `0, 0, 1`.
2. `literals[:, u]` and `literals[:, F+u]` always disagree.
3. For a small trained TM (M = 4) and for the main source TMs: frozen-rule evaluation equals
   `evaluate_clauses` element-wise on all train/val probes + threshold-boundary probes.
4. Hand-built fixtures by writing `ta_state` on a *copy* of a model: an empty clause (→ 0 in
   prediction mode), a contradictory clause `x_u AND NOT x_u` (never fires).
5. `SNAPSHOT_IDENTITY` unchanged before/after freezing.

**Gate G1:** all pass on CPU and CUDA.

### Phase 2 — hybrid model, prediction, persistence (1 day)

Tests ("State decoding", "State bounds", "Active/Inactive contribution", "Overlapping clauses",
"No active clauses", "No clauses in a bank", "Direct vs column prediction", "Multiclass
decoding", "Regression transform", "Save/load fidelity"):

- Spec §12.1: contribution `0.66` before and `0.74` after the move, with `delta = 0.1`.
- Spec §12.2 regression fixture: states `(2,4)`, `(2,-1)`, `delta = 0.5` reproduce the table
  `(-0.5 → 0, 0 → 1, 0.5 → 0.75)`.
- Direct and column paths agree (≤ 1e-12 abs) for 100 random valid state arrays.
- Argmax ties go to the lowest index.
- Save → load → identical states, gates, fingerprints and bit-identical scores.

**Gate G2:** all pass.

### Phase 3 — reference optimizer (1–2 days)

Tests ("Exact loss-change formula", "Residual maintenance", "Full-loss descent", "Optimizer
accuracy on a simple fixture", "Coordinate-local limitation", "Stop reporting", "Frozen-state
integrity", "Split isolation"):

- For random small `A` (incl. intercept columns, `lambda > 0`), the formula `s g + s² h / 2`
  equals recomputed trial-loss differences for every feasible ±1 neighbour (≤ 1e-12).
- Incremental `R` equals full recomputation after each sweep.
- Every accepted move strictly lowers `J`.
- `a = [1,2]`, `y = [2,4]`, `delta = 0.5`, `Q ≥ 4`, `lambda = 0` reaches coefficient 2.
- Trap fixture `A = [[1,1],[0,0.1]]`, `Y = [0,0.1]`, `delta = 1`, `Q = 2`, `lambda = 0`:
  stops at `(0,0)`, audit reports stationary, `(-1,1)` has lower loss — documented limitation.
- Outward moves at `±Q` are never proposed; no overflow.
- Budget stop yields `stop_reason = SWEEP_BUDGET` and non-stationary audit; selected and
  terminal checkpoints carry separate audits.
- A fingerprint mismatch (deliberately mutated gate cache) aborts with an error.
- Instrumented data loader asserts test labels are never read during fit/selection.

**Gate G3:** all pass.

### Phase 4 — Gram-form optimizer equivalence (1 day)

- On the Phase 3 fixtures and on 20 random small problems (N ≤ 500, D ≤ 200, mixed
  `lambda`), `opt_gram` and `opt_ref` with the same seed produce **identical accepted-move
  sequences**. Any divergence must be explained by a documented near-tie
  (`|DeltaJ + accept_tol| < 1e-10`) — otherwise it is a bug.
- On the 2k/1k MNIST integration set with M = 2: identical move sequence for the first 10
  sweeps, or first divergence at a logged near-tie; final objectives agree to 1e-9 relative.
- Same for A3-L against a spec-style reference implementation of A3-L.
- Ridge (`ridge.py`) checked against `numpy.linalg.lstsq` on the augmented system for small sizes.

**Gate G4:** equivalence holds; only then may the Gram form be used in Phases 5–7.

### Phase 5 — regression verification task (½ day)

Spec §12.2 and §13.4 as an *optimization* check with known fixed gates:

- 1-D target `f(x) = sin(3x)` on `[-1, 1]`, interval gates of width `h ∈ {1, 0.5, 0.25}` from
  thermometer thresholds, N = 2,000, `delta = 1/64`, `Q = 256`, `lambda = 0`.
- Record sup-error against the existence bound `M h²/8 + delta` (with `M = max|f''| = 9`), and
  the gap to the continuous least-squares fit with the same gates.
- Optional: gates from a trained `tt.RegressionTsetlinMachine` on the same data (spec §3.3);
  its own normalisation is *not* applied to the hybrid output (§4.1).

Not an accuracy claim; a sanity check that the optimizer reaches the representable region.
**Gate G5:** error bound respected by the continuous fit; finite-state fit within `delta` +
reported optimization gap; any violation investigated before Phase 6.

### Phase 6 — MNIST integration run and cost probe (1 day)

Exactly the spec §14.2 configuration: 2,000/1,000 stratified, `Binarizer(0.5)`, source TM
`M = 2` per class (one positive, one negative clause), `delta = 1/64`, `Q = 256`,
`lambda = 1e-3`, 10 sweeps, `accept_tol = 1e-12`, float64, lazy columns. Run A0, A1, A2, A3,
A4 with both optimizers.

Measure and record: Gram build time and peak memory, seconds per sweep (ref vs Gram), accepted
moves per sweep, state-bound occupancy, clause activation rates, number of all-zero slope columns
(pixels that are always 0 in training), audit counts.

Then extrapolate cost to `N = 50,000` and `M ∈ {10, 20, 50}`, and **choose the main grid
from these measurements** (write the decision and the measurements it rests on into
`DECISIONS.md`).
**Gate G6:** projected main grid fits the compute budget in §8.

### Phase 7 — main MNIST experiments (≈ 3–5 days wall, mostly unattended)

**Source TMs.** `M ∈ {2, 10, 20}` (plus 50 if G6 allows), `seed ∈ {0, 1, 2}`, `T` and `s`
chosen per `M` by a small validation grid (`T ∈ {M/2, M, 2M}` clipped to ≥ 2, `s ∈ {3, 5, 10}`),
batch size 32, up to 30 epochs, best epoch by validation accuracy (ties → earlier epoch).
Secondary encoder check (D1): `Binarizer(0.3)`, `M = 20`, seeds 0–2, arms A0/A1/A3 only.
Also train reference source TMs at `M ∈ {50, 100, 500}` (A0 only) to draw the TM accuracy–size
curve the hybrids are placed against.

**Coefficient hyperparameters** (validation only, per `M`, seed 0, then frozen for seeds 1–2):

| Parameter | Grid |
|---|---|
| `lambda` | `{1e-4, 1e-3, 1e-2}` |
| `(delta, Q)` at constant range `Q·delta = 4` | `(1/16, 64)`, `(1/64, 256)`, `(1/256, 1024)` |
| `max_sweeps` | 2000, stop on `NO_ACCEPTED_MOVES` |
| `accept_tol` | `1e-12` |

All grid points fit in `int16` (largest `Q = 1024`); moves are still bound-checked in `int32`.
For A4/A1c/A2c only `lambda` is tuned. A5 uses the A3-selected hyperparameters with 3 random
gate draws per source seed.

**Selection:** spec §10.1 key `(-accuracy, MSE)` on validation; earliest checkpoint on ties.
Then freeze `configs/final.yaml` (git-committed) **before** any test evaluation.

**Test evaluation:** one pass per frozen (arm, M, seed), logged in `test_access_log.jsonl`.

### Phase 8 — analysis and report (1–2 days)

`report/REPORT.md` generated from `results/*.json`, with:

- Main table: test accuracy mean ± SD over seeds for A0–A5, A3-L per `M`; paired per-seed
  differences for A3−A1, A3−A0, A3−A2, A4−A3, A3−A5.
- Accuracy vs. model size and vs. inference time (Pareto plot) including the A0 reference curve
  at large `M`.
- Optimization diagnostics: objective vs. sweep, accepted moves per sweep, terminal vs. selected
  audit, fraction of coefficients at `±Q`, empirical `J(q) - J(Theta*)` vs. the §13.3 bound
  (with `mu_k` from `ridge.py`; noted where the bound does not apply because of saturated
  coordinates).
- Per-class accuracy / confusion matrices; per-clause activation rate vs. size of its learned
  slopes; example slope maps (`w_kj` reshaped to 28×28) for the most active clauses.
- Honest limitations section (spec §13.5).

## 7. Preregistered decision rules

A difference is called **supported** if the paired per-seed difference has the same sign for
all 3 seeds **and** the mean exceeds 0.5 percentage points of test accuracy. Otherwise
"not supported at this budget". These are interpretation rules, not predictions.

| Question | Supported if |
|---|---|
| Q1 slopes help | A3 − A1 supported positive |
| Q2 beats source | A3 − A0 supported positive (also report vs. A0 at matched serialized size) |
| Q3 gates help | A3 − A2 **and** A3 − A2c supported positive |
| Q4 quantization cost small | A4 − A3 ≤ 0.5 pp mean (report the gap either way) |
| Q5 learned gates matter | A3 − A5 supported positive |

If Q3 fails, the hybrid's gain (if any) is attributed to linear modelling, not to the TM. If Q5
fails, the TM's clauses are acting as a generic partition.

Kill / pivot criteria:

- Phase 4 equivalence cannot be established → run main experiments with `opt_ref` only, at the
  `M` that budget allows; record in `DECISIONS.md`.
- A3 hits `NO_ACCEPTED_MOVES` within 50 sweeps with > 30% coefficients at `±Q` → range too
  small; re-run grid with `Q·delta = 8` before drawing conclusions.
- A3 training loss far below A4's while validation is worse → over-fitting; widen `lambda` grid
  upward once.

## 8. Compute budget

Hardware: RTX 3090 (24 GB) + RTX 3080 (10 GB), 32 CPU cores, 121 GB RAM.

*Derived* sizes for `M = 20`, `N = 50,000`: a dense design matrix would be 62.8 GB per spec §15.1
(never allocated); the Gram matrices are 19.7 GB total (§5.2); gates `N × 200` bool = 10 MB;
`X` float64 = 314 MB. Reference-path work per sweep is `O(N · D_total) ≈ 7.9e9` multiply-adds
(*derived*) plus a Python-level loop over 157k coordinates; whether that is seconds or minutes is
measured in Phase 6, not assumed.

Budget ceiling for Phase 7: **≤ 72 GPU-hours and ≤ 150 CPU-core-hours**. If Phase 6
extrapolation exceeds it, drop `M = 50` first, then `delta = 1/256`, then A3-L.

## 9. Layout and records

```
experiments/linearModel/
  boolean-gated-linear-tsetlin-implementation.md   # spec (frozen)
  PLAN.md  DECISIONS.md  LIBRARY_GAPS.md
  code/        data.py source_tm.py adapter.py hybrid.py opt_ref.py opt_gram.py
               ridge.py gates_random.py run.py report.py
  tests/       test_adapter.py test_hybrid.py test_opt_ref.py test_opt_gram.py test_ridge.py
  configs/     integration.yaml grid_M{2,10,20}.yaml final.yaml
  manifests/   split_seed0.json integration_seed0.json
  artifacts/   source TM checkpoints, hybrid artifacts (git-ignored; SHA in results)
  results/     <run_id>.json, test_access_log.jsonl
  logs/
  report/      REPORT.md, figures
```

Each `results/<run_id>.json` holds: arm, config, seeds, split SHA, encoder, source-TM identity
and hyperparameters, `freeze_id`, coefficient hyperparameters, trace (objective, accepted moves,
val metrics, bound occupancy, time per sweep), stop reason, selected/terminal sweep and audits,
val metrics, test metrics (Phase 7 only), sizes (coefficients, literals, serialized bytes),
timings (source-TM fit, coefficient fit, total, inference), peak memory, hardware, library rev.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Few-clause source TMs give clauses that rarely fire → slopes fit on tiny supports | Log per-clause support; `lambda` grid; A5 control at matched activation |
| One-state moves converge too slowly for useful runs | `max_sweeps = 2000`; A3-L variant isolates the effect (D3, D5) |
| Float near-ties make ref/Gram trajectories diverge | Logged near-tie analysis in G4; objective-level agreement as the fallback criterion |
| Hybrid gain is just "a linear model with more parameters" | A2/A2c and A5 controls; Pareto plot vs. size |
| Squared error on one-hot is a weak classification surrogate | Stated limitation (spec §9.1); continuous ridge A4 shares it, so Q4 is unaffected |
| Gram memory at large `M` | Per-output processing; `M = 50` only if G6 allows |
| Accidental edits to `src/` | Hash check at every gate (C1) |

## 11. Out of scope

Joint clause/coefficient learning, convolutional linear outputs (spec §17.1), mini-batch
coefficient updates, shared clause banks, feature subsets, linear *gates* (spec §Purpose),
energy claims, and library integration. Each needs its own plan after this programme reports.

## 12. Order of work (checklist)

1. Phase 0 scaffolding, `DECISIONS.md` D1–D5, manifests → G0
2. Adapter + fidelity tests → G1
3. Hybrid model, prediction paths, save/load → G2
4. Reference optimizer + audit + spec §16 tests → G3
5. Gram-form optimizer + ridge + equivalence → G4
6. Regression verification → G5
7. Integration run, cost measurements, grid decision → G6
8. Source TMs, validation grids, freeze `final.yaml`, test pass → Phase 7
9. Report from records → Phase 8
