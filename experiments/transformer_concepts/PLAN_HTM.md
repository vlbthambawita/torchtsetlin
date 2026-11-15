# Plan: HTM ideas in Tsetlin machines (H1–H3)

Theory: `THEORY_HTM.md` (claims H1.1–H3.9, summary table §S). Status: **plan only, nothing run.
Needs the user's go-ahead.** Written 2026-11-09.

## Rules (as for the rest of the programme)

- **C1** `src/torchtsetlin/**` is read-only. What the library cannot do is implemented from
  scratch in `code/htm/`; library defects go into `LIBRARY_GAPS.md`.
- **C2** Every reported number comes from a JSON record in `results/htm/`; the report is
  generated (`code/htm/report.py` → `REPORT_HTM.md`), not hand-typed.
- **C3** Settings are chosen on validation data only. One freeze file per experiment
  (`results/htm/frozen_<exp>.json`) is written before any test-set access. Test access is
  logged in `results/test_access_log.jsonl`.
- **C4** Deviations are logged in `DECISIONS.md` (from D18 on).
- Everything lives in `experiments/transformer_concepts/`.

## Gates

| gate | condition | if it fails |
|---|---|---|
| G0 | exact checks X13–X16 pass | fix the theory first; no experiment uses a failed claim |
| G1 | from-scratch TM at $\delta = 0$ matches the library (E-H1a) | stop H1 experiments, report the mismatch |
| G2 | the SP reaches its target density and is stable (E-H2b, stage 1) | report, skip stage 2 |
| G3 | fixed registers solve T3/T2 (E-H3b) | the H3 theory is wrong; do not run E-H3c |

Order: X13–X16 → D-H2a (free, uses the running `qkv` model) → E-H3a, E-H3b (cheap,
synthetic) → E-H1a, E-H1b → E-H3c → E-H2b → E-H1c (CIFAR-10, the most expensive) → report.

---

## Part 0. Exact checks (CPU, minutes)

`code/htm/verify_htm.py` → `results/htm/exact/X*.json`. Each check either matches an exact
count or fails loudly.

- **X13 (H3.1, and the equality remark).** Enumerate every pair clause as a product of per-block
  truth sets: receiver and sender sets are $\{a\}$ or $\Sigma \setminus N$ (the sets a
  conjunction of one-hot literals can define), the offset set is an interval (thermometer),
  receiver extras likewise. Compute the minimum positive DNF for T3 by exact set cover over the
  whole domain for $(L, V) \in \{(3,3), (3,4), (4,4), (4,5)\}$, window $w = 1$ and $w = 2$.
  Pass: minimum $= V$ for every instance (this also settles the $w = 2$ sketch on these sizes).
  Equality remark: for $V \le 6$ a linear program finds no threshold of $[q \| k]$ literals that
  computes $[x_i = x_j]$ (infeasibility certificate saved).
- **X14 (H1.1, H1.2).** For $k \le 10$ and all $\delta < k$: the minimum strict DNF of
  $\mathrm{Th}_{k-\delta}$ (by prime implicants and exact cover) has $\binom{k}{\delta}$ terms. For
  the thermometer code of an offset in $[-(L-1), L-1]$, $L \le 8$: every complement of an
  interval is one $(k-1)$-tolerant clause and no strict clause.
- **X15 (H2.1, H2.2).** Monte Carlo of the OR inbox density and its empirical entropy for
  $n \in \{8, 63\}$, $p$ on a grid: matches $1 - (1-p)^n$ within 3 SE and the entropy peaks at
  $p^\star \pm$ one grid step. Same for the hashed graph inbox: simulate the library's message
  hashing (`GraphTsetlinMachine.msg_bits`, read-only use) with random clause matches at rate $f$;
  compare the density with $1 - \exp(-bCfn/M)$.
- **X16 (H2.3).** One-winner literal SP on small data ($F = 8$, 3 prototypes, 200 points):
  run the expected (mean-increment) update to convergence from 20 random starts; every limit is
  a Lloyd fixed point of Hamming $k$-medians (each prototype = coordinate-wise majority of its
  cell, every point assigned to a nearest prototype).

---

## Part H1. Tolerant clauses

**Code.** `code/htm/tolerant.py`: a from-scratch vectorised TM (flat, and conv over columns, so
it also runs pair columns from `code/pairs.py`) with a `delta` argument. Mismatch counts are one
matrix product: $\text{miss} = |I| - \ell(u) \cdot \mathbb 1_I$. The feedback rule is §H1.4.
Same semantics as the library: class-wise clauses, weighted or not, `boost_true_positive=True`,
batch feedback, uniform random matching column for conv Type Ia/II.

**E-H1a: equivalence at $\delta = 0$ (gate G1).** The from-scratch TM and the library on
Noisy XOR (flat) and T5 / T4 at $L = 16$ (pair columns), 10 seeds each, same settings.
Pass: the mean accuracy difference is inside the 95 % bootstrap CI of the seed spread, and the
per-epoch mean number of included literals agrees within 10 %. (Bit-exact agreement is not
expected: the random streams differ.)

**E-H1b: noise (tests H1.3, H1.5).** Tasks: T5 and T4 (pair columns) and T7 (flat), with i.i.d.
bit flips on the Boolean input columns at rate $\eta \in \{0, 0.02, 0.05, 0.1\}$ (train and
test noisy). Arms: $\delta \in \{0, 1, 2, 4\}$, same clause budget. Search $s \in \{5, 15, 40\}$
per $(\delta, \eta)$ on validation, seed 7; finals 5 seeds. Predictions:
- at $\eta = 0$, $\delta \in \{0, 1, 2\}$ tie (difference inside CI);
- accuracy of $\delta = 0$ falls with $\eta$ faster than that of $\delta = 1, 2$; the gap at
  $\eta = 0.1$ is positive on at least 2 of the 3 tasks;
- $\delta = 4$ is worse than $\delta = 2$ at $\eta \le 0.05$ (too permissive).
- Diagnostic: the measured clause firing rate on noisy positives against $R_\delta(k, \eta)$ for
  each learned clause's $k$ (H1.3).

**E-H1c: CIFAR-10 (optional, after E-H1b is positive).** Flat TM on therm4, with
$\delta \in \{0, 2, 4\}$ and the D15 settings (2,000 clauses, $T = 80$, budget 32, 30 epochs).
The pair version only if the flat result is positive, since a from-scratch pair TM on
4,096 columns is far slower than the library's Triton path. Cost estimate: flat about 10 min
per run on one GPU (from-scratch, no Triton), pair several hours.

---

## Part H2. Sparse value codes

**D-H2a: is the `qkv` inbox saturated? (tests H2.2; free).** On the saved final `qkv` model
(`results/cifar/qkv_*_seed1_e30.pt`, D16/D17) and 1,000 validation images: per patch, the share
$f$ of layer-0 parts that match, and the density of the inbox bits. Compare with H2.2.
Prediction: $f > 0.02$ and inbox density $> 0.95$. A second number for the record: the share of
layer-2 clause literals that are inbox literals (if inboxes are saturated, the learner should
have little use for them).

**E-H2b: SP values for attention (tests H2.1, P-H2b).** `code/htm/spatial_pooler.py` (from
scratch).
- *Stage 1 (unsupervised).* Literal SP on the 64 non-overlapping 4×4 therm4 patch tokens of
  CIFAR-10 (192 bits + 14 position bits), $m = 1024$ columns, $k$ winners per token, trained on
  the 45,000 training images. Densities $k/m \in \{0.5, 1, 2, 4, 8\}\%$; $p^\star = 1.1\%$ for 63
  senders. Record: density, per-column usage (dead columns), code stability across the last two
  epochs. Gate G2.
- *Stage 2 (supervised, library).* Each token's features: its patch bits, its position bits,
  and for each of the 16 edge types of `TokenGraphs` the OR of the SP codes of the senders of
  that type. That is $206 + 16m$ bits per token, so use $m = 256$ here (stage 1 is rerun at
  that size if needed). A `Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)` reads
  the 64 token columns (Proposition H2.4: an ordinary conv TM). D15 settings, search
  $s \in \{2, 10\}$ and the five densities on validation (seed 0), finals at the best setting,
  seeds 1–3, test set.
- Comparators already measured: `pair_self`, `pair`, `qkv`, `qkv_d1`, `ctm` (`results/cifar/`).
- Predictions: (i) SP-value attention beats `qkv` at the same clause budget; (ii) validation
  accuracy against density is inverted-U with the peak within a factor 3 of $p^\star$
  computed for the actual aggregation set size (senders per edge type, not 63: the near types
  have 1–3 senders, the far ones up to ~20). Whether it beats `ctm` is open.
- Cost: stage 1 is cheap (one matrix product per batch). Stage 2 runs at about the `pair_self`
  speed (64 columns), minutes per epoch.

---

## Part H3. Repeat detection, recurrent state, local targets

Tasks and data as in the main programme: T3 at $L \in \{16, 32\}$ ($V = L + 16$), T2 at
$L \in \{17, 33\}$; 20,000 / 5,000 / 10,000 examples; final seeds 23, 29, 31, 37, 41.

**E-H3a: is the pair TM's failure on T3 a settings problem? (tests P-H3a).** A2 pair TM, T3 at
$L = 16$, $s \in \{80, 160, 320\}$, $M \in \{100, 300\}$ per class, 40 epochs, 3 seeds,
validation only. Prediction: all at chance ($\le 52\%$). This closes the "maybe higher $s$"
question left by `REPORT.md`. If any cell learns, H3.2's reading is wrong and the report says so.

**E-H3b: fixed registers (tests H3.3–H3.5; gate G3).** `code/htm/registers.py` computes, per
sequence, the seen register $s_t$, the comparator bit $r_t = \langle e(x_t), s_{t-1} \rangle$
and the transition memory $b$ (all deterministic). Arms (library models):
- *T3-seen*: conv TM over time steps reading $[e(x_t) \| s_{t-1}]$ (no comparator);
- *T3-comp*: flat TM reading $\bigvee_t r_t$ plus $e$-features (the comparator);
- *T2-mem*: flat TM reading $[e(x_{L-1}) \| b_{L-2}]$;
- controls: the same arms with the register **shuffled across examples** (must be at chance),
  and the pair TM numbers from `REPORT.md`.
Predictions: T3-comp $\ge 99\%$ at every $s$ in the grid; T3-seen near chance (H3.5c: same
share as the pair TM); T2-mem $\ge 95\%$ at $L = 33$, where the pair TM's share halves (H3.5a),
with fewer clauses than the pair TM needed (exploratory T2 needed $M = 300$). The point of this
experiment is the **contrast** between T3-seen and T3-comp: it isolates the comparator
(HTM's predicted-vs-actual signal) as the mechanism, and checks the witness-share theory.

**E-H3c: learning the state with local targets (tests H3.7–H3.9).** `code/htm/clause_rnn.py`:
state bits are the outputs of a `CoalescedTsetlinMachine(multi_label=True)` (library) reading
$[e(x_t) \| h_{t-1}]$; the comparator and the readout as in E-H3b; no gradient anywhere. Arms:
- *target = seen set*, teacher forcing, then free running at test time;
- *target = next token*, read as a **score** (vote sum > 0 as the predicted set, HTM's
  "predictive cells"), teacher forcing, then free running;
- *target = next token*, read as a decision (argmax), to show H3.8's failure mode.
Measurements: per-bit state accuracy (teacher-forced and free-running) against the true
register; T3 and T2 accuracy with the learned state; error growth with $L$ ($L = 8, 16, 32$)
against Conjecture H3.9's linear bound. Predictions: the seen-set target learns the register
near-perfectly (H3.7) and T3 accuracy $\ge 95\%$ in free running at $L = 16$; the next-token
decision arm learns nothing useful (H3.8); the score arm is in between, which is the honest
test of HTM's own objective.

Statistics for E-H1b, E-H3a–c: per comparison, mean difference with a paired bootstrap 95 % CI
over seeds and the exact McNemar test per seed, as in `REPORT.md`. A claim is "supported" when
the CI excludes 0 and McNemar is significant (p < 0.01) in at least 4 of 5 seeds.

---

## Compute

| experiment | runs | device | estimate |
|---|---|---|---|
| X13–X16 | 4 checks | CPU | < 30 min |
| D-H2a | 1 analysis | GPU | < 10 min |
| E-H3a | 18 | GPU | ~2 h (two GPUs) |
| E-H3b | ~60 (search + finals + controls) | GPU | ~2 h |
| E-H1a | 60 | GPU | ~1 h |
| E-H1b | ~400 (search + 5-seed finals) | GPU | ~6–8 h |
| E-H3c | ~90 | GPU | ~4 h |
| E-H2b | stage 1: 5 SPs; stage 2: ~15 runs | GPU | ~4 h |
| E-H1c | 3–9 | GPU | ~1 h flat; pair only if approved separately |

About one night on the two GPUs, after the CIFAR-10 runs now in progress have finished.

## Deliverables

- `code/htm/` (verify_htm, tolerant, spatial_pooler, registers, clause_rnn, report) and tests
  for the from-scratch parts in `tests/test_htm.py`.
- `results/htm/` (JSON records, freeze files, predictions).
- `REPORT_HTM.md`, generated, with a claim-by-claim table that updates `THEORY_HTM.md` §S
  (as `REPORT.md` did for `THEORY.md`), and the theory file's changelog updated.

## What would change the plan

- X13 fails: H3.1 is wrong. Rework §H3 before running E-H3a/b.
- D-H2a shows no saturation: drop prediction P-H2b's motivation, keep E-H2b as a plain
  SP-feature test.
- E-H3b T3-seen learns (against H3.5c): the witness-share theory is incomplete; stop and
  analyse before E-H3c.
- E-H1a fails: no tolerant-clause experiment is reported; the gap is described instead.
