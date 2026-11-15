# Decisions and deviations (rule C4)

Brief: `torchtsetlin_plans/experiments/transformer_concepts/idea_v2.md`. Theory: `THEORY.md`. Plan: `torchtsetlin_plans/experiments/transformer_concepts/PLAN.md`.
Library: `torchtsetlin` 0.7.0 at commit `de03558f`, `src/` unchanged.

## 2026-11-08 — design round

- **D1 Attention = clause over pairs.** The primary construction is the pair clause (THEORY §B.1),
  not separate Q/K/V clause banks. Reason: B2 (keys and values collapse under OR aggregation) and
  B3 (two banks are a tied pair clause) show that the separate banks add no expressivity. They
  would also bring back the credit-assignment problem that sank `transformer_like` (Q/K trained on
  proxy labels).
- **D2 No library change.** Pair TM = `Conv1dTsetlinMachine(kernel_size=1,
  position_encoding=False)` over `(B, F_pair, L²)` columns from `code/pairs.py`. It inherits the
  conv feedback (C1) and the CUDA fast paths unchanged.
- **D3 Relative code = thermometer of $j - i$** ($2L-2$ bits), with a log-bucketed variant as arm A3,
  because of Theorem A7 and Corollary A8.
- **D4 Specificity must be searched for pair arms.** Smoke diagnostic (`results/smoke_t4_s.json`,
  T4, $L = 12$): the hand clause on the pair columns is exact (accuracy 1.0). After 20 epochs the
  pair TM is at chance with $s = 5$, $M = 40$ and reaches ≥ 0.96 with $s = 15$ or $M = 100$. So the
  5-epoch smoke failures on T2/T4 (`results/smoke.json`) are settings, not features. The search
  grid includes $s \in \{5, 10, 15, 25\}$.
- **D5 CUDA run-to-run variation.** The same seeded T4 diagnostic gave 0.993 and 0.979 at
  $s = 15$, $M = 40$ in two runs. CUDA updates are not bit-reproducible here, so the P1 test for
  "same seed → same state" must run on CPU, or the variation must be documented as seed noise.
  To check in P1.
- **D6 Smoke settings are not final settings.** The smoke used $M = 40$, $T = 10$, $s = 5$ and
  4,000 training examples. The plan fixes $M = 100$, $T = 25$.

## 2026-11-08 — theory review

- **D7 THEORY.md reviewed proposition by proposition** (`THEORY_REVIEW.md`). One proof was wrong
  (B6.3, the conv-TM impossibility: the separating pair of sequences did not have equal $k$-gram
  sets for $k \ge 2$; corrected and now brute-force checked), one claim was false as stated
  (B6.4: one clause only for kernel exactly $d+1$), the pair-tensor memory figures were wrong
  (B.5: $F = 177$ at $L = 64$, not 300; $L = 256$ is 37 MB per example as Boolean, not 10 MB),
  and the flat-TM clause counts in D.1 were mislabelled. B7.2 was upgraded from sketch to proof;
  the "value after the last key" impossibility was downgraded from proved to sketch.
- **D8 Pair columns are built per batch**, never pre-built for a dataset: 20,000 examples at
  $L = 64$ would be 14.5 GB as Boolean columns (THEORY B.5). The harness (P1) gathers
  `pairs.pair_tokens` on each mini-batch of `(B, L)` tokens.
- **D9 The $s$ grid may need to grow with $L$.** THEORY C2 (corrected) predicts the $s$ needed for
  a pair clause to pick up its key literal grows about linearly with $L$ (the $1/s$ threshold on
  the drawn pair, diluted by $L - 1$ matching pairs). The search grid $\{5, 10, 15, 25\}$ is kept
  for $L = 32$; if the best $s$ is the largest value, extend the grid to $\{40, 60\}$ before
  freezing (to log at G3).

## 2026-11-08 — overnight proof run (torchtsetlin_plans/experiments/transformer_concepts/PROOF_PLAN.md)

- **D10 GPU pinning (LG-1).** Triton kernels fail for a model on `cuda:1` while the current device
  is 0, and the library falls back to slower paths. Every worker pins its GPU with
  `CUDA_VISIBLE_DEVICES` and uses `device="cuda"`. Library untouched; the gap is in `LIBRARY_GAPS.md`.
- **D11 Search grid.** $s \in \{5, 15, 40\}$ (D9: the grid reaches high $s$ from the start), no
  clause-budget dimension, $M = 100$ per class, $T = 25$, weighted, batch 32, ≤ 40 epochs with
  patience 8 (stop early at validation 1.0). Graph arms: `GraphTsetlinMachine`, depth 2,
  $M \cdot K / 2$ shared clauses, hashed messages (256 bits, 2 per message). These replace
  PLAN §4's budget search.
- **D12 T5 is a co-occurrence task, not a control; T7 added.** "Symbols 0 and 1 both present" is an
  AND of two existentials over different positions: one pair clause $[q=0] \land [k_0=1]$, but no
  single-token clause, and every baseline in the search sat at 0.50–0.55. T5 is kept and
  reported as an attention task. The no-routing control for H3 is the new **T7** (presence of
  symbol 0: one token-level clause). It was added before the freeze, so it goes through the same
  search.
- **D13 D2a was degenerate.** In the no-distractor needle (T4n) the label symbol occurs only at
  the needle, so the one-literal clause $[k_0 = b]$ solves it and there is no dilution to
  measure ($s^\star = 2$ at every $L$). The dilution prediction is tested on T4 with distractors,
  extended to $V \in \{6, 48\}$ (distractor rate $1/(V-1)$) at $L \in \{16, 32\}$ (D2c,
  `jobs_threshold2.json`). The T4n rows stay in the report as a negative design note, and their
  conv control is uninformative for the same reason.
- **D14 Exact checks extended.** X1b (new Proposition A3b) and X12 (exact flat cost of T1,
  Proposition D1) were added after X10 showed the minima.

## 2026-11-09 — CIFAR-10 (user request: normal TM vs transformer-concept TM)

- **D15 CIFAR-10 protocol** (`code/cifar.py`). Data: the convtm therm4 cache (12×32×32 bits), the
  convtm stratified 45,000 / 5,000 split (seed 1234), official test set read once per final run.
  All arms use the convtm `ctm-small` configuration: 200 clauses per class (2,000 total), T = 80,
  unweighted, soft literal budget 32, batch 32, 30 epochs, best-validation epoch kept. Attention
  arm: tokens are the 64 non-overlapping 4×4 patches; one column per token pair with
  [token_i | token_j | therm(dy) | therm(dx) | therm(row_i) | therm(col_i)] (F = 426, 4,096 pairs).
  Ablations: `pair_local` (|dy|, |dx| ≤ 1) and `pair_self` (i = j, no attention).
  Search: s ∈ {2, 10, 40}, seed 0, validation only. Finals: best s, seeds 1–3, test set.
  The s = 40 search run for the full `pair` arm was still queued when every other arm had
  picked s = 2 by a wide margin; finals for the other arms started before it finished.
- **D16 Q/K/V arm (user request).** `qkv` = library `GraphTsetlinMachine`, depth 2, one node per
  4×4 patch (192 patch bits + 14 position bits), edges j → i for all j ≠ i, 16 edge types
  (8 compass directions × near/far). Layer-1 clause matches on node j are the values, sent as
  hashed messages (1,024 bits, 2 per message) bound to the edge type; the inbox is the OR over
  senders; layer 2 ANDs the query literals on node i with inbox literals, so a clause can
  read several senders. Unlike the class-wise arms it is coalesced (2,000 shared clauses,
  per-class weights), because that is the graph TM's output layer; T = 80, budget 32.
  Control `qkv_d1`: the same machine at depth 1 (no message passing). Search s ∈ {2, 10},
  seed 0, then finals at the best s. About 6× slower per epoch than `pair`.
- **D17 One test seed for `qkv` (user decision, 2026-11-09).** Because a 30-epoch `qkv` run takes
  hours, its final is seed 1 only; every other arm (including `qkv_d1`) keeps seeds 1–3. The
  report compares `qkv` seed 1 against each arm's seed-1 run and against the 3-seed spread.

## 2026-11-09 — HTM programme (torchtsetlin_plans/experiments/transformer_concepts/PLAN_HTM.md, user go-ahead: "run the HTM plan after CIFAR-10")

- **D18 Exact checks: two test bugs, one theory error.** X15 first run failed only because a
  Monte Carlo density of exactly 1.0 has standard error 0 (tolerance now `3 * max(se, 1/T)`);
  X16 first run stopped the SP simulation before the permanences had settled (stop rule now: no
  connected literal can flip). Both first runs kept as `*_first_run.json`. X13 found a real
  error: the $w \ge 2$ extension of Proposition H3.1 was false (one clause can cover two repeated
  symbols); the exact minimum is still $V$ on 4 instances, now Conjecture H3.1b in THEORY_HTM.md.
- **D19 E-H3a falsified prediction P-H3a; E-H3b shuffle control leaks through presence.**
  P-H3a said the pair TM stays at chance on T3 ($L = 16$) for $s \le 320$, $M \le 300$. With
  $M = 300$ and $s \in \{160, 320\}$ it reaches 0.71–0.77 validation accuracy and is still rising
  at 40 epochs (`results/htm/eh3a/`). The witness-share threshold of H3.2 ($s \gtrsim 1/\rho - 1
  = 59$) is consistent with this; the prediction overreached by treating the junk-literal
  competition as fatal. Reported as a failed prediction. The `H_T3_comp_shuf` control reaches
  0.60 at $s = 15$: its presence bits (kept unshuffled) carry the count of distinct symbols
  ($L - 1$ in positives, $L$ in negatives), which a vote can partly threshold. It is still far
  from the comparator arm; reported as a leaky control, not removed.
- **D20 E-H1b tasks changed before any run.** PLAN_HTM listed T5 and T4 (pair columns) and T7
  for the noise test. The planted pair clauses of T5/T4 have 2–3 literals, where H1.3 predicts
  no benefit and $\delta \ge k$ makes a clause constant. E-H1b uses instead a **noisy-prototype**
  task (`exp_h1.prototypes`: 4 classes, each a mixture of 3 random 64-bit prototypes, i.i.d. bit
  flips $\eta$ on every bit; Bayes accuracy recorded), the regime H1.3 is about, and keeps T7
  (flat, bit flips on the one-hot input) as the short-pattern case where no gain is predicted.
  Pair columns stay in E-H1a (equivalence). The library-vs-from-scratch equivalence metric
  "included literals within 10 %" is noisy: a 2-epoch probe gave the library 15.9 (CPU) and 12.2
  (CUDA) on the same T5 data and seed, and the from-scratch TM 26.0 and 18.2 for two seeds; it is
  reported, but the accuracy criterion is the gate.
- **D21 E-H2b density grid and search length.** For the 16 edge types of `TokenGraphs` the
  number of senders per (type, receiver) slot ranges from 1 to 48 (mean 5.0), so the density
  that maximises the mean inbox entropy (H2.1 applied slot by slot, `effective_pstar`) is about
  0.32, not the 1.1 % of 63 pooled senders. The planned grid (0.5–8 %) would not contain the
  predicted peak: it is extended to $k \in \{1, 3, 5, 10, 20, 40, 80\}$ of $m = 256$
  (0.4–31 %). To fit the compute, the search runs 10 epochs (finals 30, as D15). One SP per
  density (seed 0, 2 epochs on the 45,000 training images); gate G2 passed on the k = 5 smoke
  (density exactly k/m, 3 of 256 columns unused, 78 % code overlap between the last two epochs).
- **D22 E-H3c search widened; T2 memory fix.** The first search (`results/htm/h3c_search_v1/`)
  varied only $s$, with a hand-picked state-TM size of $4V$ clauses and $T = 15$; at $L = 32$ the
  free-running seen state was only 53–74 % bit-accurate, and the T2 runs ran out of GPU memory
  (an int32 cumulative sum over the $V^2$ transition bits; now a Boolean running OR). A
  validation-only diagnostic showed the state-TM size was the limit: $16V$ clauses, $T = 50$,
  $s = 5$ give 99.9 % teacher-forced and 99.2 % free-running bit accuracy after one epoch. The
  search now covers clauses $\in \{4V, 16V\}$ (T2: $\{2.5V^2, 10V^2\}$) × $T \in \{15, 50\}$ ×
  $s \in \{2, 5, 15\}$, still selected by the self-supervised state-bit accuracy (no labels),
  ties → fewer clauses, smaller $s$.
- **D23 No library gap in CPU vs CUDA batch feedback.** The D20 probe (one seed) suggested the
  library learns shorter clauses on CUDA than on CPU. With 5 seeds (`code/htm/lib_device_check.py`,
  `results/htm/lib_device_check.json`, T5 pair columns, 4,000 examples, 2 epochs) the mean
  included literals per clause are 15.5 (CPU, SD 1.9) and 15.8 (CUDA, SD 5.5): seed noise, not a
  device difference. Nothing goes into LIBRARY_GAPS.md.
- **D24 E-H1b prototype noise grid extended.** With $\eta \le 0.1$ the 12 random 64-bit
  prototypes are about 32 bits apart and every arm reached 100 % validation accuracy (Bayes
  100 %): no information about tolerance. Before any test access on the new cells, the grid gets
  $\eta \in \{0.2, 0.3, 0.35\}$, chosen by their Bayes validation accuracy (99.8, 94.5, 82.2 %;
  computed from the generator, no TM involved), with its own search, freeze
  (`frozen_h1b_hard.json`) and 5-seed finals. The $\eta \le 0.1$ finals are kept and reported as
  the ceiling they are.
- **D25 E-H1c run (condition met).** E-H1b's hard-noise search showed tolerance helping
  monotonically in $\delta$ at $\eta \ge 0.3$ (validation), so the CIFAR-10 flat test runs: the
  from-scratch TM with $\delta \in \{0, 2, 4\}$, seeds 1–3, D15 budget (200 clauses per class,
  $T = 80$, batch 32, 30 epochs, best validation epoch) and $s = 2$ (the flat TM's searched value,
  D15). No literal budget: the from-scratch TM does not implement it, so $\delta = 0$ here is the
  baseline, not the library's 42.5 % flat number. All three $\delta$ are reported; none is
  selected.
