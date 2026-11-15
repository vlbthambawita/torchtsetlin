# Hierarchical Temporal Memory ideas in Tsetlin machines: theory for H1–H3

Companion to `THEORY.md` (notation §0, rectangle property, pair clause §B.1, drift formula §C2).
Plan: `PLAN_HTM.md`. Status labels as in `THEORY.md`: **[proved]**, **[checked]** (brute force,
to be run as X13–X16 of `PLAN_HTM.md`), **[sketch]**, **[conjecture]**, **[empirical]**.

**Changelog.** 2026-11-09: first version (design only). Same day, after X13–X16: the $w \ge 2$
sketch of H3.1 was false (counterexample from X13), now Conjecture H3.1b; X13–X16 otherwise pass.
After all runs (same day): status column of §S updated from `REPORT_HTM.md`; H1.5 refuted as
stated, P-H3a and P-H2b (ii) fail, H3.3–H3.8 supported.

Contents: §0 HTM in one page and the dictionary to TMs · §H1 tolerant clauses · §H2 sparse
value codes and OR aggregation · §H3 repeat detection, recurrent state and local targets ·
§S summary of claims · references.

---

## 0. HTM in one page, and the dictionary

HTM (Hawkins & Ahmad 2016; Cui, Ahmad & Hawkins 2016, 2017; *not in `source_documents/papers/`,
verify before citing*) has three parts that matter here.

- **Synapses with permanences.** A synapse has a scalar permanence $\pi \in [0,1]$ and is
  *connected* iff $\pi \ge \pi_0$. Learning moves permanences up for active inputs and down for
  inactive ones, on segments that were active and correct.
- **Dendritic segments with a threshold.** A segment is active iff at least $\theta$ of its
  connected synapses see an active input.
- **Spatial pooler (SP).** $m$ columns compute their overlap with the input. Inhibition keeps
  the $k$ columns with the largest overlap (k-winner-take-all, typically $k/m \approx 2\%$).
  The winners' synapses learn. The SP is unsupervised and its output density is fixed by
  construction.
- **Temporal memory (TM\_HTM).** Each column has several cells. A cell is *predictive* if one of
  its distal segments is active on the cells that were active at $t-1$. On input $x_t$, predicted
  cells of the active column fire. If no cell was predicted the whole column fires ("bursts").
  Burst vs predicted is HTM's surprise signal. Learning is local and one-shot-fast: segments
  grow synapses to the previously active cells. No error is propagated anywhere.

| HTM | Tsetlin machine (`THEORY.md` §0) |
|---|---|
| permanence $\pi$, connected iff $\pi \ge \pi_0$ | TA state, literal included iff state $\ge N$ |
| segment active iff $\ge \theta$ connected synapses active | clause true iff **all** included literals true |
| synapses only on active inputs (positive literals) | literals $u$ and $\lnot u$ |
| increment active, decrement inactive (on a correct, active segment) | Type Ia: true literals up ($p_\text{mem}$), false down ($1/s$) |
| k-WTA inhibition, fixed output density | none: clause firing density is whatever learning produces |
| local targets (prediction of the next input) | one global target (the class) for every clause |

The dictionary suggests three changes, one per hypothesis: give clauses a threshold (H1), give the
value layer a fixed density and a local learning rule (H2), and give sequence models a recurrent
state with local targets (H3).

---

## H1. Tolerant clauses

### H1.1 Definition

**Definition.** A **$\delta$-tolerant clause** with include set $I$ ($|I| = k$, no complementary
pair in $I$) is
$$C_\delta(u) = \Big[\textstyle\sum_{l \in I} (1 - \ell_l(u)) \le \delta\Big],$$
true iff at most $\delta$ included literals are false. $\delta = 0$ is the ordinary clause. With
only positive literals and $\theta = k - \delta$ it is an HTM segment. Empty-clause convention
as in the library ($k = 0$: true while learning, false when predicting). For $\delta \ge k$ a
non-empty clause is constant true; implementations cap $\delta \le k - 1$.

**Geometry.** On the included coordinates, $C_\delta$ is the Hamming ball of radius $\delta$
around the pattern that makes every included literal true. A strict clause is a subcube; a
tolerant clause is a ball inside the same subcube's coordinates.

### H1.2 Expressivity

**Proposition H1.1 (strict-clause cost of one tolerant clause) [proved; checked X14].** Let $I$
have $k$ literals over $k$ distinct variables and $0 \le \delta < k$. Every DNF of strict
clauses over the same literals that computes $C_\delta$ exactly on $\{0,1\}^k$ has at least
$\binom{k}{\delta}$ clauses, and $\binom{k}{\delta}$ suffice.

*Proof.* Rename so that every included literal is a positive variable $z_1..z_k$ (flip the
variables of negated literals; this is a bijection of the cube that maps clauses to clauses).
Then $C_\delta = \mathrm{Th}_{k-\delta}(z)$, the monotone threshold "at least $k - \delta$ ones".
*Upper bound:* the OR over all $(k-\delta)$-subsets $S$ of $\bigwedge_{i \in S} z_i$, i.e.
$\binom{k}{k-\delta} = \binom{k}{\delta}$ clauses. *Lower bound:* let $D$ be any DNF of (possibly negated) literals computing
$\mathrm{Th}_{k-\delta}$. For a $(k-\delta)$-subset $S$ let $z^S$ be the point with ones exactly on
$S$; it is positive, so some term $t$ of $D$ is true on $z^S$. Then $t$ contains $z_i$ as a
positive literal for every $i \in S$: otherwise, since $t$ is true on $z^S$ it does not contain
$\lnot z_i$, so it does not mention $z_i$ at all and is also true on $z^S - e_i$, which has
$k - \delta - 1$ ones and is negative. So $t \supseteq \{z_i : i \in S\}$, and $t$ is false
on $z^{S'}$ for every other $(k-\delta)$-subset $S'$ (some $i \in S \setminus S'$ is 0 there).
Distinct subsets therefore need distinct terms. ∎

So one tolerant clause with $k = 12$, $\delta = 3$ replaces 220 strict clauses. The TM becomes a
depth-2 threshold circuit (threshold of thresholds), strictly more expressive per clause.

**Proposition H1.2 (OR as a single clause) [proved].** With $\delta = k - 1$, $C_\delta$ is the
disjunction of its $k$ literals. In particular one tolerant pair clause can express "receiver
is one of $\{a_1, .., a_r\}$" with $r$ positive one-hot literals (a strict clause needs the
$V - r$ negated literals of the complement), and "offset **outside** an interval",
$[j - i > t_1] \lor \lnot[j - i > t_2]$ with $t_2 < t_1$, which no strict thermometer clause can
express (a strict thermometer clause is one interval, `THEORY.md` A1). ∎

**What tolerance does not buy: equality.** $[x_i = x_j]$ on one-hot tokens is not a single
tolerant clause, and no threshold of literals computes it (§H3.1, Proposition H3.1 and the
remark after it). Tolerance changes the shape of a clause's truth set (balls instead of cubes)
but each clause still thresholds a *linear* count of literals.

### H1.3 Noise

**Proposition H1.3 (recall and false alarm under bit noise) [proved].** Let a pattern make all
$k$ included literals true, and let each input bit flip independently with probability $\eta$.
Then the clause fires on the noisy pattern with probability
$$R_\delta(k, \eta) = \Pr[\mathrm{Bin}(k, \eta) \le \delta].$$
On a uniformly random input it fires with probability
$F_\delta(k) = 2^{-k} \sum_{i \le \delta} \binom{k}{i}$. ∎

Consequences, in numbers. For $k = 12$, $\eta = 0.1$: $R_0 = 0.28$, $R_1 = 0.66$, $R_2 = 0.89$;
the false-alarm rate rises from $2.4 \cdot 10^{-4}$ to $3.2 \cdot 10^{-3}$ and $1.9 \cdot 10^{-2}$.
A strict clause trades recall for precision by **shortening** (fewer literals); a tolerant
clause can keep $k$ literals (high specificity on random inputs) and still recall noisy patterns.
For $k$ literals and noise $\eta$, the tolerance that maximises $R_\delta - F_\delta$ is
$\delta \approx \eta k$ (Chernoff: $R_\delta \to 1$ once $\delta \ge \eta k + O(\sqrt{k})$, and
$F_\delta$ stays exponentially small while $\delta < k/2$).

### H1.4 Feedback for tolerant clauses

**Definition (feedback rule).** Exactly the TM rules (Granmo 2018, as implemented in
`functional.py`), with "clause output" replaced by $C_\delta$:

- Type Ia (clause should fire and does): every literal true on the example (or the drawn
  patch/pair) moves towards inclusion with probability $p_\text{mem}$; every false literal moves
  towards exclusion with probability $1/s$. This includes the up to $\delta$ *included false*
  literals: tolerance gives them a way out.
- Type Ib (should fire, does not): every literal moves towards exclusion with probability $1/s$.
- Type II (fires, should not): every *excluded false* literal moves towards inclusion. Each
  inclusion adds one to the mismatch count, so $\delta + 1$ of them must land before the clause
  stops firing on that example.
- Clause selection probabilities from the vote sum and $T$ unchanged.

**Proposition H1.4 (reduction and drift) [proved].** (i) At $\delta = 0$ the rule is the library's
rule. (ii) The drift ratio of `THEORY.md` §C2 holds unchanged with $\rho$ the probability that the
literal is true conditioned on "$C_\delta$ fires" (and the drawn pair): the derivation uses only
the per-event probabilities, not the firing condition. ∎

**Remark H1.5 (what changes in the dynamics) [sketch].** Two effects pull in different
directions. (a) More Type I events per positive example on noisy data, because the clause fires
on more of them (Proposition H1.3), and the Type Ia events now come from noisy copies, so
$\rho$ for a planted literal is $1 - \eta$ (as for a strict clause), but measured on far more events.
(b) Type II needs $\delta + 1$ inclusions to silence a false alarm, so negatives are removed
more slowly. Proposed rule of thumb, to test: $\delta$ small (1–2) helps on noisy inputs and is
neutral on clean ones; large $\delta$ makes clauses fire on too much and Type II cannot keep up.
**[empirical]**

---

## H2. Sparse value codes and OR aggregation

### H2.1 Why density matters: the OR inbox

In the Q/K/V graph arm (`code/cifar.py`, D16) node $i$ receives the OR over its $n$ senders of
their message bits. The same holds for any "pre-aggregated" attention (§H3.4).

**Proposition H2.1 (information in an OR inbox) [proved].** Let each sender set an inbox bit
independently with probability $p$. The bit's density is $q = 1 - (1-p)^n$, and its entropy
$h(q)$ (binary entropy) is maximal, 1 bit, at $q = 1/2$, i.e.
$$p^\star = 1 - 2^{-1/n} \approx \frac{\ln 2}{n}.$$
Any information the bit carries about the senders is at most $h(q)$. ∎

For $n = 63$ senders (64 patch tokens): $p^\star = 1.09\%$. At $p = 5\%$, $q = 0.960$ and
$h(q) = 0.24$ bits; at $p = 10\%$, $q = 0.9987$ and $h(q) = 0.014$ bits. A value code must be
about **as sparse as HTM's 2%** for OR aggregation over 64 tokens to carry information.

**Corollary H2.2 (the hashed graph inbox) [proved, given the independence approximation].** In
`GraphTsetlinMachine` with $C$ clauses, $b$ message bits per clause and inbox size $M$, if a
fraction $f$ of the clauses' layer-0 parts match at a sender, each sender sets about $bCf$ of the
$M$ positions (with cyclic shifts per edge type, positions are spread over the inbox). An inbox
position is then 0 with probability about $\exp(-bCfn/M)$. For the CIFAR-10 `qkv` arm ($C = 2000$,
$b = 2$, $M = 1024$, $n = 63$), $q \ge 0.99$ as soon as $f \ge 0.019$, i.e. as soon as 38 of the
2,000 layer-0 parts match at a typical patch. ∎

**Prediction P-H2a [empirical].** In the trained `qkv` model the layer-0 firing fraction $f$ is
well above $0.02$, the inbox density is near 1, and the depth-2 model gains little over its
depth-1 control. If instead $f$ is small and the inbox density is moderate, a `qkv` loss to
`ctm` must have another cause.

### H2.2 The spatial pooler with literals is Hamming vector quantisation

**Definition (literal SP).** $m$ columns, column $c$ with a connected set $I_c$ of literals.
Overlap $o_c(u) = \sum_{l \in I_c} \ell_l(u)$. Winners $W(u)$: the $k$ columns with the
largest overlap (ties broken by index). Learning: for each winner, connected-set literals that
are true gain permanence and false ones lose it; the permanence crosses $\pi_0$ to
connect/disconnect.

**Proposition H2.3 (fixed points of the one-winner literal SP) [proved].** Let $k = 1$ and let
every column connect, for each variable, exactly one of its two literals (a "full" column, the
limit in which a column has a synapse on every input). Then $o_c(u) = F - d_H(u, w_c)$ for the
prototype $w_c \in \{0,1\}^F$ that makes $I_c$ true, so the winner is the Hamming-nearest
prototype. With the permanence rule in expectation (mean increments), a configuration is a
fixed point iff every prototype is a coordinate-wise **majority** (Hamming median) of the inputs
it wins, and every input is won by a nearest prototype: the fixed points of Lloyd's algorithm
for $k$-medians in Hamming space. ∎

*Proof.* The overlap identity is direct. For the expected update, coordinate $v$ of column $c$
moves towards the literal that is true on the majority of $c$'s won inputs (increments for true,
decrements for false), so it is stationary iff it already is that literal (or there is a tie). ∎

So a literal SP is an online binary $k$-medians quantiser, and with $k$ winners of $m$ it is a
soft (multi-assignment) version. It needs no labels, and its output density is exactly $k/m$,
which Proposition H2.1 can set: $k/m \approx p^\star$.

**Proposition H2.4 (a frozen value layer restores the conv-TM learning problem) [proved].** If
the value codes $v_j$ are computed by a fixed map (an SP trained first and frozen), then any
attention TM whose clauses read $[x_i \,\|\, \mathrm{OR}_{j \in N_t(i)} v_j \,\|\, \text{pos}]$
is a conv TM over fixed per-token vectors. Its feedback is the conv feedback (`THEORY.md` C1),
and the non-stationarity of C3 (inbox literals of other clauses changing while those clauses
learn) disappears. ∎

**Prediction P-H2b [empirical].** On CIFAR-10, the SP-value attention arm beats the `qkv` arm at
the same clause budget, and its accuracy as a function of the SP density $k/m$ is
inverted-U-shaped with the peak within a factor of 3 of $p^\star$ for the aggregation set size
used. Whether it beats `ctm` is open.

---

## H3. Repeat detection, recurrent state and local targets

### H3.1 The exact cost of repeat detection for pair clauses

T3 (`tasks.repeat_detect`): $L$ tokens, label 1 iff some symbol occurs twice; negatives have all
tokens distinct, positives have exactly one symbol twice. $V \ge L$ (the programme uses
$V = L + 16$).

**Proposition H3.1 (T3 needs exactly $V$ pair clauses) [proved for window $w = 1$; checked X13].**
For the pair TM with sender window $w = 1$, any position code, with or without receiver extras,
a positive-clause DNF that computes T3 exactly needs at least $V$ clauses, and $V$ suffice.

*Proof.* Upper bound: $C_a = [q = a] \land [k_0 = a] \land [j - i > 0]$ fires iff some pair
$i < j$ has $x_i = x_j = a$. The OR over $a \in \Sigma$ is T3.

Lower bound. Let $C$ be a clause of the DNF, so $C$ fires on no negative. A pair column $(i, j)$
reads only $x_i$, $x_j$ and bits that depend only on $(i, j)$.

(1) *Witnesses are the repeat.* Let $C$ fire on a positive $X$ through the column $(i, j)$. If
$i = j$, or $i \ne j$ and $x_i \ne x_j$, let $Z$ agree with $X$ at $i$ and $j$ and carry
distinct fresh symbols elsewhere (possible since $V \ge L$). $Z$ is a negative, and $C$ fires on
$Z$ through the same column: contradiction. So $i \ne j$ and $x_i = x_j$ is the repeated symbol
of $X$.

(2) *One symbol per clause.* Let $C$ fire on positives $X$ (repeat $a$, witness $(i,j)$) and $Y$
(repeat $b$, witness $(i', j')$). By the rectangle property on the column's blocks, $C$ is true
on the column with receiver token $a$ (from $X$), sender token $b$ (from $Y$, $y_{j'} = b$) and
position bits of $(i, j)$ (from $X$). Let $Z$ have $a$ at $i$, $b$ at $j$ and distinct fresh
symbols elsewhere. If $a \ne b$, $Z$ is a negative on which $C$ fires: contradiction. So $a = b$.

Every symbol is the repeated symbol of some positive, so at least $V$ clauses are needed. ∎

**Correction (X13, 2026-11-09).** The first version of this file claimed the same proof for
$w \ge 2$ as a sketch ("the mixing step goes through block by block"). That is false: with
$w = 2$ a single clause can cover positives of **two** repeated symbols. The counterexample found
by X13 is $[q = 0] \land [k_0 = 1] \land [k_1 \in \{0, 1\}]$: a column reading
$(0, 1, 0)$ or $(0, 1, 1)$ always contains a repeat, of 0 or of 1. Step (1) still holds (the
column must contain the repeat), but step (2)'s mix can land on a column that contains a
*different* repeat. `results/htm/exact/X13_first_run.json` holds the failed run.

**Conjecture H3.1b ($w \ge 2$) [conjecture; checked on 4 instances].** The minimum is still
$V$: the exact minimum cover over all maximal valid pair clauses equals $V$ for
$(L, V, w) \in \{(3,3,2), (3,4,2), (4,4,2), (4,5,2)\}$ (X13, integer program). The trivial
lower bound from what X13 measured (a clause covers at most 2 repeated symbols) is $\lceil V/2
\rceil$.

**Remark (no threshold computes equality).** $[x_i = x_j]$ on one-hot $x_i, x_j$ is not a linear
threshold of literals: if $w \cdot e_a + u \cdot e_a \ge \theta > w \cdot e_a + u \cdot e_b$
for all $a \ne b$, adding the inequalities for $(a, b)$ and $(b, a)$ gives
$w_a + u_a + w_b + u_b \ge 2\theta > w_a + u_b + w_b + u_a$, the same sum. So neither tolerant
clauses (H1) nor an HTM segment computes token equality. A dot product (a product of the two
tokens) does. That is the transformer's $q_i \cdot k_j$, and it is what TMs lack. **[proved]**

### H3.2 Why T3 is hard to learn, not only to express

The X8 hand construction showed that the clauses exist in the library, and every arm stayed at
chance (`REPORT.md`). §C2's drift heuristic, applied to the one-literal-short clause, says why.
Call the **witness share** of a literal $\ell$ needed by a planted clause $C^\star$ the
probability that $\ell$ is true on the drawn column, given that $C^\star \setminus \{\ell\}$
fires on a positive example. It is the $\rho$ of the drift formula at the last step.

**Proposition H3.2 (witness shares) [proved under the uniform-draw property C1, $w = 1$, self
pairs included as in `pairs.py`].**

| task, planted clause | one-short clause | witness share of the missing literal |
|---|---|---|
| T5 $[q=0] \land [k_0=1]$ | $[q = 0]$ | $\approx 1/L$ |
| T4 $[q=M] \land [k_0=b] \land [\text{offset} = d]$ | $[q=M] \land [\text{offset}=d]$ | $1$ |
| T1 $[q=A] \land [k_0=B] \land [j-i>0]$ | $[q=A] \land [k_0=B]$ | $1$ on positives |
| T3 $[q=a] \land [k_0=a] \land [j-i>0]$ | $[q=a] \land [k_0=a]$ | $\dfrac{1}{4(L-1)}$ (best of the three orders) |

*Proof of the T3 row.* Given that $a \in X$ and $X$ is positive, $a$ is the repeated symbol with
probability $1/(L-1)$ ($L - 1$ distinct symbols, one repeated). If $a$ occurs once,
$[q=a] \land [k_0=a]$ matches only the self pair $(i,i)$, where $[j - i > 0]$ is false. If $a$ is
repeated, it matches 4 columns (two self, two cross) and $[j-i>0]$ is true on one. Uniform draw:
$\frac{1}{L-1} \cdot \frac14$. The other orders give $\approx 1/(L(L-1))$
($[q=a] \land [j-i>0]$, missing $[k_0 = a]$) and smaller. ∎

At $L = 16$ the T3 share is $1/60$, against $1/16$ for T5 and 1 for T1 and T4. There are also
$V = 32$ such clauses to find per class. Each gets this share only on the $1/(L-1)$ of the
positives that are "about" its symbol, and the junk literals it competes with have the same
Type II push. The factor $1/(L-1)$ is the core: the label is symbol-agnostic ("*some* symbol
repeats"), but every clause that can express it is symbol-specific (Proposition H3.1).
**Prediction P-H3a [empirical]:** raising $s$ and $M$ alone does not make the pair TM learn T3 at
$L = 16$ ($s \le 320$, $M = 300$).

### H3.3 Recurrent Boolean state

**Definition (clause RNN).** A state $h_t \in \{0,1\}^m$, $h_{-1}$ fixed, each bit an OR of
clauses over $[e(x_t) \,\|\, h_{t-1}]$ (one small clause bank per bit, or a multi-output TM).
A readout TM reads $h_{L-1}$, or the OR over $t$ of clauses over $[e(x_t) \| h_{t-1}]$ (a conv TM
over time steps).

**Proposition H3.3 (what a clause RNN can hold cheaply) [proved].**
(a) *DFA.* An $n$-state DFA over $\Sigma$ is simulated with $m = n$ one-hot state bits and one
two-literal clause $[h_{t-1}[q]] \land [x_t = a]$ per transition, $nV$ clauses in all.
(b) *Seen set.* The "not yet seen" register $\bar s_t[a] = \bar s_{t-1}[a] \land [x_t \ne a]$
($\bar s_{-1} = \mathbf 1$) is **one clause per bit** (an AND recurrence), $V$ clauses for all
of $\Sigma$. Its complement is the seen set $s_t$.
(c) *Transition memory.* $b_t[k, v] = b_{t-1}[k,v] \lor ([x_{t-1} = k] \land [x_t = v])$ holds
the bigrams seen so far; with a one-hot "previous token" register ($p_t[a] = [x_t = a]$, one
clause per bit) it is 2 clauses per bit. ∎

**Proposition H3.4 (T3 and T2 become easy readouts) [proved].**
(a) With the seen register, T3 $= \bigvee_t \bigvee_a [x_t = a] \land s_{t-1}[a]$: still $V$
clauses (H3.1 is about the representation, and this is the same equality), **but** with a fixed
comparator bit $r_t = \bigvee_a [x_t = a] \land s_{t-1}[a]$, i.e. the Boolean dot product
$\langle e(x_t), s_{t-1} \rangle$, T3 is the single literal $\bigvee_t r_t$, one clause.
(b) With transition memory, T2 (query $q$ in the last slot, label the value after $q$) is
class $v$ $= \bigvee_{k \in \Sigma_K} [x_{L-1} = k] \land b_{L-2}[k, v]$: $|\Sigma_K|$ clauses
per class, the same count as the pair TM with $w = 2$ (`THEORY.md` B7.1). ∎

**Proposition H3.5 (witness shares with pre-aggregated memory) [proved].** For a flat TM reading
the pre-aggregated bits (no drawn pair: the bits are features of the example):
(a) T2 with $b$: for the planted clause of class $v$ and key $k$, the one-short clause
$[x_{L-1} = k]$ fires on all examples with query $k$, and $b[k, v]$ is true iff the label is
$v$, so the witness share on class-$v$ positives is **1**. For the pair TM ($w = 2$), the
one-short clause $[q = k] \land [k_1 = v]$ matches every column whose receiver holds $k$ (query
or key slot) and whose sender is a key with value $v$, about $2n/|\Sigma_V|$ columns ($n$ key
slots), of which 2 have $[k_0 = k]$: share $\approx |\Sigma_V|/n$, i.e. 0.5 at $L = 17$ and
0.25 at $L = 33$ for the programme's $|\Sigma_V| = 4$. The gain from pre-aggregation grows with
the number of slots.
(b) T3 with the comparator bit: one literal, witness share 1.
(c) T3 with the seen register but no comparator: the one-short clause $[x_t = a]$ (in a conv TM
over time steps) matches 1 or 2 steps; $s_{t-1}[a]$ is true on one of two when $a$ is repeated:
share $\frac{1}{L-1} \cdot \frac12$, the same order as the pair TM. ∎

So the state alone does not help repeat detection. **The comparator** (HTM's
predicted-vs-actual signal) does, because it is a fixed dot product that turns a symbol-specific
condition into one symbol-agnostic bit.

### H3.4 Pre-aggregation vs the pair clause

The registers of H3.3 OR over time *before* any clause reads them; the pair clause ORs *inside*
the clause. These are different operators.

**Proposition H3.6 (incomparable expressivity) [proved].** (a) A flat clause over pre-aggregated
bigram bits can AND several existentials ("$ab$ occurs **and** $cd$ occurs"): one clause, where
a pair clause needs a sender window covering both bigrams. (b) A pair clause binds the parts of
one witness ("$a$ immediately followed by $b$, at relative offset in $[3, 5]$ from an earlier $c$")
through one column. Pre-aggregated bigram bits cannot express binding to a shared position
without position-indexed bits ($V^2 L$ of them). (c) Pre-aggregated bits have witness share 1
for any target that is a single existential (H3.5a), against $1/(\#\text{matching columns})$ for
the pair clause. ∎

### H3.5 Learning the state locally (the HTM part)

H3.3–H3.4 use **fixed** registers. HTM's claim is that such a state can be learned with local
targets and no backpropagation. In TM terms:

**Definition (local targets).** Each state bit $c$ gets its own target sequence $y^c_t$ computed
from the input stream alone (self-supervised), and is trained as an ordinary TM output on
inputs $[e(x_t) \| h_{t-1}]$. *Teacher forcing* feeds the target state $y_{t-1}$ in place of
$h_{t-1}$ during training. *Free running* feeds the model's own $h_{t-1}$.

Candidate targets, in increasing order of how much they presuppose:
- next token, $y^a_t = [x_{t+1} = a]$ (generic; HTM's own objective);
- seen set, $y^a_t = s_t[a]$ (union pooling: "has $a$ occurred");
- previous token, $y^a_t = [x_t = a]$ (trivial, a delay line).

**Proposition H3.7 (teacher forcing makes each bit a flat problem with share 1) [proved].**
Under teacher forcing with target $s_t$, bit $a$'s inputs at step $t$ are $e(x_t)$ and $s_{t-1}$.
Its target is the OR of the two single literals $s_{t-1}[a]$ and $[x_t = a]$, each with witness
share 1 on the positives it covers. It is learned by two one-literal clauses: the easiest flat-TM
problem there is. ∎

**Remark H3.8 (next-token targets on T3 are weak) [proved for the Bayes rule; learning
empirical].** On T3 negatives the next token is uniform over the unseen symbols, so
$\Pr[x_{t+1} = a \mid \text{history}] = \bar s_t[a] / (V - t - 1)$. The not-seen register is the
sufficient statistic, but every target bit has base rate $\le 1/(V - L)$. A TM output that
thresholds a vote predicts 0 everywhere, the trivial optimum for accuracy. So the generic
next-token target **does not** carry the seen-set state into a TM unless the output is read as a
score or set (HTM's "predicted set") rather than a decision. This is a real obstacle, and the
plan tests both target kinds.

**Conjecture H3.9 (free running) [conjecture].** If teacher-forced bits have per-step error
$\varepsilon$, the free-running seen register is wrong at step $t$ with probability at most
$t\varepsilon$ (errors are absorbing in an OR/AND recurrence), so the readout's error on T3 grows
at most linearly in $L$.

---

## S. Summary of claims

| id | statement | status | test |
|---|---|---|---|
| H1.1 | one $\delta$-tolerant clause = exactly $\binom{k}{\delta}$ strict clauses | proved | X14 |
| H1.2 | $\delta = k-1$ gives a disjunction; unions of offset intervals in one clause | proved | X14 |
| H1.3 | recall $\Pr[\mathrm{Bin}(k,\eta)\le\delta]$, false alarm $2^{-k}\sum_{i\le\delta}\binom ki$ | proved; recall direction seen on proto at high $\eta$ | E-H1b |
| H1.4 | tolerant feedback reduces to the library at $\delta=0$; drift formula unchanged | proved; G1 passes on xor, T4, T5 | E-H1a |
| H1.5 | small $\delta$ helps on noisy inputs, neutral on clean | **refuted as stated**: helps only on long patterns at $\eta \ge 0.2$; hurts on T7 (even clean) and CIFAR-10 | E-H1b, E-H1c |
| H2.1 | OR over $n$ senders informative only for density $\approx \ln 2/n$ | proved | X15 |
| H2.2 | the `qkv` hashed inbox saturates once 2% of the clauses match at a patch | proved (approx.); P-H2a holds | D-H2a |
| H2.3 | literal SP with one winner = Hamming $k$-medians (fixed points) | proved | X16 |
| H2.4 | frozen value layer → conv-TM learning problem | proved | – |
| P-H2b | SP-value attention beats `qkv`; peak density near $p^\star$ | (i) holds (best token model, below `ctm`); (ii) inverted U holds, peak ~4× below $p^\star$: narrowly fails | E-H2b |
| H3.1 | T3 needs exactly $V$ pair clauses ($w = 1$) | proved, checked | X13 |
| H3.1b | the same for $w \ge 2$ (the $w=1$ proof does not extend: counterexample) | conjecture, 4 instances | X13 |
| – | no threshold of literals computes token equality | proved | X13 |
| H3.2 | T3 witness share $1/(4(L-1))$ vs $1/L$ (T5), 1 (T1, T4) | proved; but P-H3a fails (pair TM learns T3 at $M=300$, high $s$) | E-H3a |
| H3.3–H3.4 | seen register one clause/bit; comparator makes T3 one clause; T2 $|\Sigma_K|$/class | proved; supported | E-H3b |
| H3.5 | pre-aggregated memory: share 1 for T2, and for T3 with comparator only | proved; supported (comp vs seen contrast; comp_shuf control leaks, D19) | E-H3b |
| H3.6 | pre-aggregation and pair clause incomparable | proved | – |
| H3.7 | teacher-forced state bits are one-literal flat problems | proved; supported (free-running T3 ≥ 95 % at $L=16$) | E-H3c |
| H3.8 | next-token targets have base rate $\le 1/(V-L)$; TM decision outputs ignore them | proved (Bayes); supported, and the score read-out fails too | E-H3c |
| H3.9 | free-running error at most linear in $L$ | conjecture; not contradicted (growth convex in $t$, test not sharp) | E-H3c |

## References

- Granmo, O.-C. (2018). The Tsetlin machine. arXiv:1804.01508.
- Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence
  memory in neocortex. *Frontiers in Neural Circuits* 10:23. *(verify before citing)*
- Cui, Y., Ahmad, S., & Hawkins, J. (2016). Continuous online sequence learning with an
  unsupervised neural network model. *Neural Computation* 28(11). *(verify before citing)*
- Cui, Y., Ahmad, S., & Hawkins, J. (2017). The HTM spatial pooler. *Frontiers in Computational
  Neuroscience* 11:111. *(verify before citing)*
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear transformers are secretly fast weight
  programmers. ICML. *(context for §H3.4; verify before citing)*
