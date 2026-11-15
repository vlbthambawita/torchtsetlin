# Plan for proving and testing THEORY.md

Started 2026-11-08 (overnight run). Everything lives in `experiments/transformer_concepts/`;
`src/torchtsetlin/**` is read-only (C1). The library is used as the *object under test*: hand-built
clauses are loaded into real `torchtsetlin` models (by writing `ta_state`), so a check passes only if
the library itself computes what the theory says. Results are JSON records under `results/`;
`REPORT.md` is generated from them by `code/report.py` (C2).

The claims in `THEORY.md` §E fall into three kinds, and each kind gets a different kind of evidence:

| kind | what counts as evidence | part |
|---|---|---|
| exact statements (constructions, equalities, lower bounds) | the written proof **plus** an exhaustive or exact computation on every instance small enough to enumerate, and the construction run through the library | I |
| learning-dynamics predictions (C1, C2) | a controlled measurement on the library's feedback, compared with the predicted formula | II |
| empirical hypotheses (H1–H5 of `PLAN.md`) | preregistered experiments with seeds and decision rules | III |

## Part I — exact verification (`code/verify_exact.py` → `results/exact/*.json`)

| id | claim | method | pass criterion |
|---|---|---|---|
| X1 | A1, A2, A3 (interval costs per code) | exact minimum literal counts and minimum clause covers for all intervals, $n \le 12$ | A1: max literals ≤ 2; A2: literals $= n - \lvert S\rvert$; A3: table reproduced |
| X1b | **new A3b**: in binary with $n = 2^m$, the interval $[1, n-2]$ needs exactly $m$ clauses (proof: covers ↔ strongly connected digraphs) | exact cover via prime implicants, $m = 2..4$ | min cover $= m$ |
| X2 | A5 (T1 at chance for single-token sets), and its scoping (bigrams beat chance) | enumerate all T1 inputs for small $L$, group by symbol set / bigram set, compute the Bayes-optimal accuracy of any function of that set | symbol set: exactly 0.5; bigram set: $> 0.5$ |
| X3 | A6 (conv + thermometer PE solves T1 with $2(L-1)$ clauses per class) | load the $f_k, g_k, h_k, e_k$ clauses into the library's `Conv1dTsetlinMachine` with its **own** thermometer position bits; predict on every T1 input | accuracy 1.0 |
| X4 | A7 (no absolute code gives cheap relative position) | for each code and $n \le 32$: the tightest conjunction containing any two fooling points leaves the relation (this is the proof, computed); plus an explicit cover of size $n - d$ / $n - 1$ | all fooling pairs separated; cover exact |
| X5 | B1 (pair clause = $\exists (i,j)\, Q \land K \land R \land A$) | random include masks loaded into the library pair TM vs. a direct NumPy evaluation of the factorised formula | identical on all tested inputs |
| X6 | B3 scope remark (popcount ≠ vote) | explicit counterexample | vote and popcount disagree |
| X7 | B4.1 (self-only pair TM = conv TM with kernel $w$ on the END-padded sequence) | random clauses mapped both ways, both models evaluated by the library | identical clause outputs |
| X8 | B5, B6.1, B7.1, B8 (one-clause / few-clause pair constructions) | hand clauses loaded into the library pair TM; prediction on exhaustive (small) and 10,000 random (larger) inputs, several $L$ | accuracy 1.0 |
| X9 | B6.3 (conv kernel $k \le d$ cannot solve T4) | the two separating sequences evaluated by the library's `Conv1dTsetlinMachine` under random clauses, plus the $k$-gram set check | identical outputs, different labels |
| X10 | B6.2, B7.2 (flat lower bounds), D.1 (flat T1 counts) | **exact minimum positive DNF** over the one-hot domain: implicants are products of per-position symbol sets; enumerate all products on small domains, keep prime implicants, exact set cover | min ≥ the proved bound (and report the exact value) |
| X11 | B9 (positive + dominating negative clause) | library weighted classifier, all four input combinations | truth table matches |

## Part II — learning dynamics (`code/exp_dynamics.py`, `code/exp_threshold.py`)

**D1 (C1 + the premise of C2): the drawn pair is uniform and the mean drift is
$\rho - (1-\rho)/(s-1)$.** One clause that includes only an always-true feature $f_0$, on an
example with $m$ matching columns of which $r$ carry a second feature $f_1$ ($\rho = r/m$). Run many
library updates, resetting `ta_state` after each, and measure the state drift of $\lnot f_1$
relative to $f_0$. Under uniform pair draws and the library's probabilities ($(s-1)/s$, $1/s$) the
ratio $E[\Delta f_1] / E[\Delta f_0]$ equals $\rho - (1-\rho)/(s-1)$. Grid: $m \in \{4, 16\}$, $r$
over a range, $s \in \{3, 10\}$, CPU (row-sparse commit) and CUDA (events / fused path). Pass:
measured ratio within 3 standard errors of the formula in every cell.

**D2 (C2): the specificity needed by a pair clause grows with the number of pairs.** Refined
prediction (before running): for T4 the key literal $[k_0 = b]$ is true on the drawn pair of an
under-specified clause $[q = M]$ with probability $\rho(L, V) = 1/L + (1 - 1/L)\,\pi$, where $\pi$ is
the background rate of $b$ among fillers. The literal drifts towards inclusion iff $s > 1/\rho$. So:

- **D2a, no distractors** ($\pi = 0$, label symbols never appear as fillers): the smallest $s$ that
  learns, $s^\star(L)$, should grow roughly linearly in $L$;
- **D2b, with distractors** ($\pi = 1/(V-1)$): $s^\star$ should grow with $L$ but saturate near $V - 1$;
- **control**: the conv TM with kernel $d+1$ (no dilution: one matching window) should have a
  $s^\star$ that does not grow with $L$.

Grid: $L \in \{8, 12, 16, 24, 32\}$, $s \in \{2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64\}$, 2 seeds,
fixed epochs. $s^\star$ = smallest $s$ with mean validation accuracy ≥ 0.9. Pass for D2a: $s^\star$
non-decreasing in $L$ and Spearman correlation between $s^\star$ and $1/\rho$ ≥ 0.8; control:
$s^\star$ range across $L$ no wider than one grid step… reported, not forced.

## Part III — empirical hypotheses (`code/exp_main.py`)

The `PLAN.md` comparison, reduced to fit one night on two GPUs (deviations logged in
`DECISIONS.md`):

- tasks T1–T6 at $L = 16$ and $L = 32$ (T2 uses odd lengths 17 and 33);
- arms: B0 flat, B1 conv $k=1$, B2 conv $k=3$ + library PE, B3 conv $k=d+1$ (T4), P-therm /
  P-1hot / P-bin / P-gray (T1, T4, T6), B4 graph chain depth 2, A1 pair $w=1$, A2 pair $w=2$,
  A3 pair bucketed, A4 graph complete depth 2, A2-self, A2-norel, A2-abs, A2-shuf;
- $M = 100$ per class, $T = 25$, weighted, batch 32; search $s \in \{5, 15, 40\}$ at $L = 32$ on
  validation, 1 seed; selected $s$ reused at $L = 16$;
- freeze → 5 seeds (fewer if the probe says so, logged) → test, with the access log;
- decision rules, kill criterion K1 and the H2/H2a/H3 verdicts exactly as `PLAN.md` §5;
- H5: T2 learning curves, $N_\text{train} \in \{1\text{k}, 4\text{k}, 16\text{k}\}$, A2 vs B0;
- H4 / interventions on the trained A2 models: self-only at inference, $|j - i| \le 1$ window,
  shuffled relative block; clause-block composition; offset histogram of matching pairs.

## Order and stopping

1. Part I (CPU/GPU, minutes). Any failure is investigated before Part III: either the theory or
   the code is wrong, and the report says which.
2. D1, then a cost probe, then D2 and Part III in parallel on the two GPUs.
3. Theory corrections that the evidence forces go into `THEORY.md` with a dated changelog and
   into `REPORT.md`.
4. `REPORT.md` is generated last, from the records only.
