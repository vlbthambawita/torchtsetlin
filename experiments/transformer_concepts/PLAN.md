# Positional encoding and attention for Tsetlin machines: experiment plan

Theory: [THEORY.md](THEORY.md) (propositions referred to as A1…C4). Deviations:
[DECISIONS.md](DECISIONS.md). Library defects: [LIBRARY_GAPS.md](LIBRARY_GAPS.md).
Predecessor: `experiments/transformer_like/` (QK attention on MNIST, negative; read its
`report/REPORT.md`).

## 0. Ground rules

- **C1** `src/torchtsetlin/**` is read-only. Every arm runs on the public API. The pair TM is a
  `Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)` over pair columns built by
  `code/pairs.py` (THEORY §B.1). The graph arms use `GraphTsetlinMachine` with experiment-built
  `GraphBatch`es.
- **C2** Every reported number comes from a `results/**/*.json` record. The report is generated.
- **C3** Selection uses validation data only. Test sets are generated once from a held-out seed.
  Every test read is appended to `results/test_access_log.jsonl`, and finals refuse to run without
  `results/frozen_experiment.json`.
- **C4** Deviations go into `DECISIONS.md` with a date.
- **C5** No timing, search or seed runs without the user's go-ahead (a smoke of seconds is fine).
- Defaults learned from `transformer_like`: weighted clauses, $T/M = 0.25$, batch 32,
  `feedback_mode="batch"`. Unweighted and sequential runs are separate, labelled rows.

## 1. Questions and hypotheses

| id | hypothesis | primary evidence |
|---|---|---|
| **H1** | Positional codes matter in the way A1–A8 predict. Thermometer ≥ one-hot > binary/Gray for interval rules on absolute-position tasks (T6). Relative codes beat any absolute code on order and offset tasks (T1, T4) at equal clause count. | P-arms on T6, T1, T4 |
| **H2** | Pair attention (A2) beats the flat, conv and graph-chain TMs on tasks that need content-dependent routing (T1–T4), at $L \in \{16, 32\}$. | A2 vs best baseline |
| **H2a** | The gain comes from cross-token reads, not from extra features: A2 beats its self-only twin (A2-self) and its no-relative-code twin (A2-norel). | ablations; **kill criterion K1** |
| **H3** | Pair attention does not hurt where routing is useless (T5 bag, T6 absolute) by more than 1 pp. | A2 vs best baseline on T5, T6 |
| **H4** | Learned pair clauses are readable as attention rules: on T1/T2/T4 most high-weight clauses have literals in both the query and the key block, and their offset literals match the planted offset. | interpretation diagnostics (§7) |
| **H5** | Sharing across positions improves sample efficiency on T2 (THEORY B7): A2 needs fewer training examples than the flat TM for the same accuracy. | learning curves, $N_\text{train} \in \{1\text{k}, 4\text{k}, 16\text{k}\}$ |

## 2. Tasks

Generators in `code/tasks.py` (all with an explicit `torch.Generator`). Default $V = 16$ symbols;
lengths $L \in \{16, 32\}$ for the main grid and $L = 64$ for the scaling rows (§3.3).

| id | task | needs attention? | label | theory |
|---|---|---|---|---|
| T1 | ordered pair: one A, one B; is A before B | yes (order of two content-defined tokens) | 2 classes | B5, A5, A6 |
| T2 | associative recall `k1 v1 … kn vn q` → value of q | yes (offset varies per example; three tokens) | $\lvert\Sigma_V\rvert = 4$ | B7 |
| T3 | repeat detection: does any symbol occur twice | yes (equality across arbitrary distance) | 2 | B8 |
| T4 | needle: symbol at fixed distance $d$ after the marker ($d = L/4$) | yes, but a kernel of $d+1$ also solves it | 4 | B6 |
| T5 | bag control: are symbols 0 and 1 both present | no (order irrelevant) | 2 | A4 |
| T6 | absolute position: symbol at position 3 | no routing, absolute position only | 4 | A1–A3 |

Splits per (task, $L$): train 20,000, validation 5,000, test 10,000, generated from seeds
`SeedSequence([base, sha256(task, L, split)])`. The test seed is written to the frozen manifest
before any test set is generated. A generated set is cached with its SHA-256.

**Real dataset (after gate G4, one only; the user chooses).** Candidates with TM results in the
literature catalog (`benchmarks/literature/results.csv`):

- **IMDB sentiment** (82 TM accuracy rows; weighted TM with drop clause 91.3 %). Natural tokens,
  and word order matters a little ("not good"). This tests H3 and whether pair clauses find
  negation-scope rules. Needs a vocabulary cut (e.g. top 5k words) and truncation to $L \le 64$.
- **MIT-BIH arrhythmia** (1 TM row: 94.2 % for PVC identification). Closest to the user's cardiac
  work, and relative timing (RR intervals) is exactly what the relative code expresses.
  Booleanisation: thermometer-quantised samples, downsampled to $L \le 64$ tokens per beat window.

Recommendation: MIT-BIH if H2 holds on T2/T3 (variable offsets), IMDB if only T1 holds (order
without long-range lookup).

## 3. Arms

### 3.1 Baselines

| arm | model | input |
|---|---|---|
| B0 | `TsetlinMachine` (flat) | one-hot sequence, $LV$ bits |
| B1 | `Conv1dTsetlinMachine`, $k = 1$, no PE | per-token one-hot |
| B2 | `Conv1dTsetlinMachine`, $k = 3$, thermometer PE (library) | per-token one-hot |
| B3 | `Conv1dTsetlinMachine`, $k = d + 1$, no PE (T4 only: the "oracle window", B6.4) | per-token one-hot |
| B4 | `GraphTsetlinMachine`, depth 2, chain graph (`GraphBatch.from_sequences`) | the reference sequence model |

### 3.2 Positional-encoding arms (H1)

`Conv1dTsetlinMachine`, $k = 1$, `position_encoding=False`, with the code appended as channels by
`code/pairs.py::token_columns`: **P-1hot**, **P-therm**, **P-bin**, **P-gray**. They run on T1, T4,
T6 (and T5 as a control where position should not help).

### 3.3 Attention arms

| arm | construction | pairs | relative code |
|---|---|---|---|
| A1 | pair TM, $w = 1$ | all $L^2$ | thermometer ($2L-2$ bits) |
| **A2** (primary) | pair TM, $w = 2$, last bit | all $L^2$ | thermometer |
| A3 | pair TM, $w = 2$, last bit | all $L^2$ | log-bucketed (THEORY A.4) |
| A4 | graph TM depth 2, complete graph, edge type = offset bucket | $L(L-1)$ edges | via edge types |

Scaling rows at $L = 64$: A2, A3, A4, B0, B2 only.

### 3.4 Ablations of A2 (trained, same search budget)

| arm | change | tests |
|---|---|---|
| A2-self | pairs restricted to $i = j$ (= conv TM, B4.1) | H2a, kill K1 |
| A2-norel | relative code removed | H2a, A7 |
| A2-abs | relative code replaced by thermometer of $i$ and of $j$ | A7 (absolute cannot replace relative) |
| A2-shuf | relative code permuted across pairs, one fixed permutation per example | is the gain the code or the extra bits |

### 3.5 Inference-time interventions (no training)

On the trained A2 of each final seed: (a) restrict pairs to $i = j$; (b) restrict to $|j-i| \le 1$;
(c) replace the relative block by that of a random other pair. These are the analogues of the
`transformer_like` self-only/uniform swaps, which showed that the QK model there did not use the
cross-token reads.

### 3.6 Secondary rows

- **S1** matched automata: B0 and B2 widened to A2's automaton count ($K \cdot 2M \cdot 2F$).
- **S2** sequential feedback on T1 and T2 for A2 and B0 (batched-feedback fidelity).
- **S3** unweighted clauses for A2 and B0.

## 4. Search, freeze and final runs

- Fixed: $M$ (clauses per class) = 100 and $T = 25$ for every arm, so $T/M = 0.25$. Batch 32,
  weighted, `n_states` default.
- Searched per (task, arm) at $L = 32$, on validation: $s \in \{5, 10, 15, 25\}$ ×
  `max_included_literals` $\in \{\text{none}, 8\}$ = 8 configurations, 1 seed. The smoke showed
  that $s$ is the critical setting for pair arms (DECISIONS D4). The selected config is reused at
  $L = 16$ and $L = 64$.
- Early stopping: up to 50 epochs, patience 10, earliest best validation epoch restored (the
  `transformer_like` D15 rule).
- Freeze: `results/frozen_experiment.json` (configs, seeds 23, 29, 31, 37, 41, test seeds, code
  hashes) before any test read.
- Finals: 5 seeds per (task, $L$, arm).

## 5. Decision rules (preregistered)

For each comparison X vs Y on a (task, $L$): **supported** needs all five paired seed differences
positive, a mean gap ≥ 0.5 pp that exceeds the seed noise band (2 × the larger seed SD), and
McNemar $p < 0.05$ on ≥ 4 of 5 seeds (paired on the same test examples). **Reverse supported** is
the mirror image. Anything else is **inconclusive**. A bootstrap 95 % CI over test examples is
reported for each mean gap.

- **H2** is supported on a task if A2 beats every baseline (B0–B4, and B3 on T4) under the rule.
- **H2a** needs A2 > A2-self and A2 > A2-norel under the rule.
- **H3** holds if A2 is not reverse-supported against the best baseline with a gap > 1 pp.
- **H5** compares accuracies at each $N_\text{train}$ under the same rule.

**Kill criteria.**

- **K1:** after the search (Phase 4), if A2-self is within 0.5 pp of A2 on T1, T2 *and* T4
  (validation), stop. Attention is then not being used, which is the `transformer_like` failure
  mode. Report and close.
- **K2:** if the Phase 2 probe projects more than 40 GPU-hours, apply the reductions of §8 before
  asking for the go-ahead.
- **K3:** if A2 is at chance on T1 after the search, suspect the harness first (the hand-clause
  tests must pass), then $s$.

## 6. Phases and gates

| phase | work | gate (pass criterion) |
|---|---|---|
| P0 theory | `THEORY.md`, this plan; literature search for TM attention outside the local collection (P0.2) | **G0** user review of THEORY/PLAN |
| P1 harness | runner with JSON records (resumable, one record per (task, L, arm, config, seed)), manifests, graph-batch builders for A4/B4, tests | **G1** tests pass on CPU and CUDA: pair columns = naive loop (done); hand clauses solve T1, T2 (done) and T4; A2-self clause outputs = conv TM on tokens for random include masks; complete-graph builder = naive; same seed → identical `ta_state` |
| P2 probe | 2 epochs per arm at $L \in \{16, 32, 64\}$ | **G2** cost table written; projection ≤ 40 GPU-h or reductions applied; **user go-ahead** |
| P3 search | §4 on validation | **G3** every arm above chance on T5/T6; K1, K3 checked |
| P4 freeze | write the frozen manifest | **G4** frozen file exists; user chooses the real dataset |
| P5 finals | 5 seeds; ablations; interventions; S1–S3; H5 curves | **G5** all records present or listed as failed |
| P6 real data | one dataset, same protocol (own search, freeze, 5 seeds) | **G6** user go-ahead after G5 |
| P7 report | `report/REPORT.md` generated from records | — |

## 7. Diagnostics and interpretation (H4)

`code/interpret.py` (to write in P1), on each final A2:

- For each clause with weight in the top quartile: does it include literals in the query block,
  the key block(s), and the relative block? What offset interval do its relative literals admit
  (via `_thermometer_range`-style decoding in experiment code)?
- **Attention map:** for each test example, the set of matching pairs of the clauses that voted
  for the predicted class; histogram of $j - i$ over matching pairs, and compared against the
  planted offset (T4: $d$; T2: key slot of the query).
- **Rule printout:** the top clauses per class in the `q=… ∧ k0=… ∧ rel∈[a,b]` form, using
  `pairs.pair_feature_names`.

## 8. Cost and resources

GPUs: RTX 3090 (24 GB) and RTX 3080 (10 GB). Smoke (`results/smoke.json`; 4,000 examples,
$L = 12$, $M = 40$): after warm-up, an epoch takes about 0.04 s for every arm. The first arm of
each task also pays capture/compile time (0.2–0.25 s per epoch). Pair cost scales with
$L^2 \cdot F$ (THEORY §B.5), so a rough projection for 20,000 training examples and up to 50
epochs is:

| $L$ | pair arms, per run | other arms, per run |
|---|---|---|
| 16 | ≈ 0.5 min | < 0.5 min |
| 32 | ≈ 3 min | < 1 min |
| 64 | ≈ 25 min | ≈ 1–2 min |

Memory: pair columns are built **per batch** from the `(N, L)` token tensor (D8). A pre-built
pair dataset at $L = 64$ would be 14.5 GB (THEORY B.5); a batch of 32 is ≈ 70 MB in flight.

Runs: search $6 \times 13 \times 8 \approx 620$ (most under a minute); finals
$6 \times 2 \times 17 \times 5 \approx 1{,}000$; scaling and secondary rows ≈ 200. Projection:
**≈ 15–25 GPU-hours**, about one day of wall time on the two GPUs. The P2 probe replaces this
estimate. If it exceeds 40 GPU-h, apply these reductions in order: drop P-bin/P-gray from T4;
drop $L = 64$ for A4; reduce the search to $s \in \{10, 25\}$.

## 9. Risks

| risk | mitigation |
|---|---|
| Attention unused (as in `transformer_like`) | K1, interventions §3.5, ablations §3.4 |
| Dilution of Type Ia with $L^2$ pairs (THEORY C2) slows learning at $L = 64$ | causal/windowed pair sets; higher $s$; report learning curves, not only final accuracy |
| Synthetic tasks too easy (everything at 100 %) | report epochs-to-99 % and $N_\text{train}$ curves (H5) alongside accuracy |
| Synthetic wins do not transfer | the P6 real dataset is mandatory before any general claim |
| Pair memory at large $L$ | bucketed code (A3), graph arm (A4) never materialises pairs |
| Graph TM complete-graph cost | depth 2 only; `GraphBatch.pad` for the CUDA fast paths |

## 10. Out of scope (this round)

Next-token generation and causal language modelling; multi-layer stacks beyond the depth-2 graph
TM; counting aggregation (THEORY B.4; would need a custom `_evaluate`, possible later as experiment
code); a learned separate value map (B2 shows it adds nothing under OR aggregation); differentiable
surrogates; any change to `src/`.

## 11. Layout

```
experiments/transformer_concepts/
  idea.md, idea_v2.md       the brief
  THEORY.md                 propositions and proofs
  PLAN.md                   this file
  DECISIONS.md              dated deviations
  LIBRARY_GAPS.md           library defects (rule C1)
  code/tasks.py             task generators (T1–T6)
  code/pairs.py             pair columns, token columns with position codes
  code/verify_theory.py     brute-force checks of A1–A3, A7 -> results/theory_checks.json
  code/smoke.py             implementability smoke -> results/smoke.json
  code/smoke_t4_s.py        T4 / specificity diagnostic -> results/smoke_t4_s.json
  tests/test_pairs.py       pair columns vs naive; hand clauses on T1, T2
  results/                  JSON records only
```

Run from the repository root:

```bash
python -m experiments.transformer_concepts.code.verify_theory
pytest experiments/transformer_concepts/tests -q
python -m experiments.transformer_concepts.code.smoke --device cuda
python -m experiments.transformer_concepts.code.smoke_t4_s
```
