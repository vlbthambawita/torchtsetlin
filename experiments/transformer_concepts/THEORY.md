# Positional encoding and key–query–value attention in Tsetlin machines: theory

Status labels: **[proved]** a complete proof is given here; **[checked]** additionally verified by
brute force (`code/verify_theory.py` → `results/theory_checks.json`, or a unit test in
`tests/test_pairs.py`); **[sketch]** the argument is given but not every case is written out;
**[conjecture]** believed, not proved; **[empirical]** only an experiment can answer it.

**Changelog.** 2026-11-08: first version; review (`THEORY_REVIEW.md`). 2026-11-08/09 overnight
(`PROOF_PLAN.md`, evidence in `REPORT.md`): new Proposition A3b and Proposition D1, new Conjecture
B7.3, B6.2 shown tight, the C2 drift formula corrected to the library's `boost_true_positive`
default, and the C2 dilution prediction tested (direction confirmed, magnitude partly).

Contents: §0 notation · §A positional encoding · §B attention as a clause over pairs ·
§C learning under TM feedback · §D worked examples · §E summary of claims.

---

## 0. Notation and conventions

- Input sequence $X = (x_0, \dots, x_{L-1})$ of symbols from an alphabet $\Sigma$, $|\Sigma| = V$,
  one-hot encoded: $e(x) \in \{0,1\}^V$. Positions are $0$-indexed.
- Literals of a Boolean vector $u \in \{0,1\}^F$: $\ell(u) = [u, \lnot u] \in \{0,1\}^{2F}$.
- A clause with include set $I \subseteq \{1..2F\}$: $C(u) = \bigwedge_{k \in I} \ell_k(u)$.
  Empty-clause convention (library): $C \equiv 1$ while learning, $C \equiv 0$ when predicting.
  All expressivity statements below concern **non-empty** clauses in prediction mode.
- Clause $j$ has polarity $p_j \in \{+1, -1\}$ and weight $w_j \ge 1$ (weighted TM) or $w_j = 1$.
  Class score $v_c(X) = \mathrm{clip}\big(\sum_{j \in c} p_j w_j C_j(X), -T, T\big)$, prediction
  $\arg\max_c v_c$.
- Specificity $s$, Type I / Type II feedback as in Granmo (2018), batched over a mini-batch by
  default (`feedback_mode="batch"`).
- **Convolutional TM** (Granmo et al., 2019): patches $u^{(1)}, \dots, u^{(P)}$ of $X$; clause output
  $C(X) = \bigvee_{p=1}^{P} \bigwedge_{k \in I} \ell_k(u^{(p)})$; Type Ia and Type II feedback for a
  clause use one uniformly random *matching* patch, drawn independently per event
  (`_ConvMixin._feedback_counts`).
- **Exact DNF view.** When a statement says "a positive-clause TM needs $\ge N$ clauses", it means:
  if a class score is the OR of its positive clauses (no negative clauses, threshold at one vote),
  then computing the class exactly on the task domain needs $\ge N$ clauses. TM scores are
  thresholds of clause sums with negative clauses; lower bounds for that richer form are marked
  **[conjecture]**.

**Rectangle property** (used repeatedly). A conjunction over literals of a vector that splits into
blocks with disjoint variables, $u = (u_1, \dots, u_r)$, factorises:
$C(u) = \bigwedge_b C_b(u_b)$. Hence the truth set of $C$ is a product $S_1 \times \dots \times S_r$
with $S_b = \{u_b : C_b(u_b) = 1\}$, and if $C$ is true on $u$ and $u'$ it is true on every
**mix** $z$ with $z_b \in \{u_b, u'_b\}$ for each block $b$. For one-hot tokens the blocks are the
positions: a flat clause constrains position $q$ to a symbol set $S_q \subseteq \Sigma$.

---

## A. Positional encoding

### A.1 Codes

A position code is a map $c : \{0..n-1\} \to \{0,1\}^m$. Compared here:

| code | $m$ | $c(p)_k$ |
|---|---|---|
| one-hot | $n$ | $[p = k]$ |
| thermometer (as `conv.py`) | $n - 1$ | $[p > k]$, $k = 0..n-2$ |
| binary | $\lceil \log_2 n \rceil$ | bit $k$ of $p$ |
| Gray (reflected) | $\lceil \log_2 n \rceil$ | bit $k$ of $p \oplus (p \gg 1)$ |

A clause over $\ell(c(p))$ selects a set of positions. Rules that care about position usually
need an **interval** $[a, b] = \{a, \dots, b\}$ ("in the first half", "after position 5").

### A.2 Cost of an interval

**Proposition A1 (thermometer) [proved, checked].** Every interval $[a, b] \subseteq \{0..n-1\}$ is
the truth set of one clause with at most 2 literals: $[p > a-1] \land \lnot[p > b]$ (drop the first
literal if $a = 0$, the second if $b = n-1$).

*Proof.* $[p > a-1] \iff p \ge a$ and $\lnot[p > b] \iff p \le b$. ∎

**Proposition A2 (one-hot) [proved, checked].** A single position $\{a\}$ costs one literal
$[p = a]$. An interval $S$ with $|S| \ge 2$ is the truth set of one clause, and the minimum number
of literals is exactly $n - |S|$.

*Proof.* A positive literal $[p = k]$ restricts the truth set to $\{k\}$, so a clause whose truth
set has $\ge 2$ elements contains only negative literals $\lnot[p = q]$, each removing exactly one
position $q$. Removing the $n - |S|$ positions outside $S$ needs exactly that many. ∎

*Edge case for A1 and A2.* The full set $S = \{0..n-1\}$ costs zero position literals under both
codes. That is not an empty clause in practice: position literals are always conjoined with token
literals, which keep the clause non-empty.

**Proposition A3 (binary, Gray) [proved for the characterisation; values checked].** A clause over a
binary or Gray code selects $\{p : c(p) \in Q\}$ for a subcube $Q$ of $\{0,1\}^m$. An interval
is one clause iff it equals such a set, which for the binary code with $n = 2^m$ means an aligned
dyadic block $[t 2^r, (t+1) 2^r - 1]$ (a subcube with free bits $U$ and $\ge 2$ elements is an
interval iff $U = \{0, \dots, r-1\}$: a free bit $u$ above a fixed bit $v$ puts $x$ and $x + 2^u$ in
the set but not $x + 2^v$ between them). A general interval therefore needs several clauses. The
dyadic decomposition gives the upper bound $2\lceil \log_2 n \rceil - 2$; the exact minimum is
smaller, because a clause whose truth set is *not* an interval may still lie inside $S$ (for
$n = 8$, $[1, 6] = \{1,3\} \cup \{2,6\} \cup \{4,5\}$: three subcubes, against four dyadic blocks).
No lower bound growing with $\log n$ is proved here. Brute force (exact minimum covers, all
intervals):

| $n$ | binary: intervals that are one clause | binary: max clauses | Gray: one clause | Gray: max clauses | thermometer / one-hot: max clauses |
|---|---|---|---|---|---|
| 4 | 7 / 10 | 2 | 8 / 10 | 2 | 1 |
| 6 | 11 / 21 | 3 | 14 / 21 | 2 | 1 |
| 8 | 15 / 36 | 3 | 19 / 36 | 3 | 1 |
| 10 | 19 / 55 | 4 | 26 / 55 | 3 | 1 |

(from `results/theory_checks.json`; extended to $n = 12$ in `results/exact/X1.json`). Gray codes make every adjacent pair $\{p, p+1\}$ a subcube
(consecutive codes differ in one bit), which is why more short intervals are single clauses.

**Proposition A3b (binary codes need $\log_2 n$ clauses for some interval) [proved, checked].**
With the binary code and $n = 2^m$, $m \ge 2$, the interval $[1, n-2]$ needs exactly $m$ clauses.

*Proof.* $[1, n-2]$ is every string except $0^m$ and $1^m$. A subcube avoiding $0^m$ fixes some
bit to 1, and one avoiding $1^m$ fixes some bit to 0, so every implicant lies inside some
$Q_{uv} = \{x : x_u = 1, x_v = 0\}$, $u \ne v$, which is itself an implicant. A minimum cover may
therefore use only sets $Q_{uv}$, i.e. the arcs $u \to v$ of a digraph $G$ on the $m$ bit
positions. A string with support $U$ ($\emptyset \ne U \ne [m]$) is covered iff some arc leaves
$U$. Every non-empty proper $U$ has a leaving arc iff $G$ is strongly connected, and a strongly
connected digraph on $m \ge 2$ vertices needs at least $m$ arcs (each vertex needs an out-arc),
which a directed cycle attains. ∎ (Exact covers for $m = 2, 3, 4$: `results/exact/X1b.json`.)
So the binary code has a lower bound growing with $\log n$ after all; the maximum over all
intervals computed for $n \le 12$ never exceeds $\lceil \log_2 n \rceil$.

**Consequence.** For clause-based rules, **thermometer** is the right absolute code: every
interval costs ≤ 2 literals, so a small `max_included_literals` budget and a moderate $s$ suffice.
One-hot is as good for single positions but costs $n - |S|$ literals for wide intervals. Binary and
Gray save input bits ($\log n$ instead of $n$) but break intervals into several clauses.

### A.3 What the convolutional OR does with and without positions

**Proposition A4 (set invariance) [proved].** A convolutional clause without position literals
depends only on the **set** of patch vectors $\{u^{(1)}, \dots, u^{(P)}\}$. For tokens with
`kernel_size = 1` it depends only on the set of symbols present.

*Proof.* $\bigvee_p C(u^{(p)})$ is invariant under any permutation of the patches and under
duplicating a patch. ∎

**Corollary A5 [proved; consistent with smoke].** On task T1 (one A and one B at random distinct
positions, label $[\mathrm{pos}(A) < \mathrm{pos}(B)]$), every model whose output is a function of
the **set of single tokens** (`kernel_size = 1`, no position literals) has accuracy exactly $1/2$
on the task distribution: swapping the positions of A and B is a measure-preserving bijection of
the examples that leaves the symbol set unchanged and flips the label, so of each such pair of
examples the model gets exactly one right. (Smoke: `conv_k1_nopos` 0.477 after 5 epochs,
`results/smoke.json`.) The corollary does **not** cover kernels $k \ge 2$: a set of $k$-grams
reveals the order of A and B when they are within $k - 1$ positions of each other, so a conv TM
without position literals can be slightly above $1/2$ (`pair_self_only` with $w = 2$ sees
bigrams; its 0.471 is consistent but not implied).

**Proposition A6 (position literals turn OR-pooling into located detectors) [proved].** With
thermometer position literals and `kernel_size = 1`, T1 is solved exactly by $2(L-1)$ clauses per
class and a vote threshold. Let $a = \mathrm{pos}(A)$, $b = \mathrm{pos}(B)$ and, for
$k = 0..L-2$,

$$f_k = \exists p: [x_p = A] \land \lnot[p > k],\qquad g_k = \exists p: [x_p = B] \land [p > k],$$
$$h_k = \exists p: [x_p = B] \land \lnot[p > k],\qquad e_k = \exists p: [x_p = A] \land [p > k].$$

Then $\sum_k (f_k + g_k) = (L-1-a) + b$ and $\sum_k (h_k + e_k) = (L-1-b) + a$. With $f, g$ voting
for class 1 and $h, e$ for class 0 (unit weights, $T \ge 2(L-1)$), $v_1 - v_0 = 2(b - a)$, positive
iff $a < b$. ∎

So absolute position makes order *learnable* by a convolutional TM, but only as a vote count
over $\Theta(L)$ located clauses that must all cooperate. The pair clause of §B does it with one
clause. (Smoke: `conv_k1_therm` reaches 0.85, `pair_w2` 1.00 after 5 epochs.)

### A.4 Relative position cannot come from absolute codes

**Theorem A7 (no absolute code gives cheap relative position) [proved, checked].** Let $c$ be
**any** position code and consider clauses over $[\ell(c(i)), \ell(c(j))]$ for a pair of
positions. Then

1. any DNF whose truth set on $\{0..n-1\}^2$ is $R_d = \{(i, j) : j - i = d\}$ has
   $\ge n - d$ clauses;
2. any DNF whose truth set is $R_< = \{(i, j) : i < j\}$ has $\ge n - 1$ clauses.

*Proof.* By the rectangle property each clause's truth set is a product $A \times B$. (1) If
$A \times B \subseteq R_d$ contains $(i, i+d)$ and $(i', i'+d)$ with $i \ne i'$, it contains
$(i, i'+d) \notin R_d$. So each clause covers at most one of the $n - d$ points of $R_d$. (2) The
points $(k, k+1)$, $k = 0..n-2$, are a fooling set: a rectangle containing $(k, k+1)$ and
$(k', k'+1)$ with $k < k'$ contains $(k', k+1)$, and $k' \ge k+1$, so it is not in $R_<$. ∎

Brute force confirms both bounds are tight for one-hot, thermometer, binary and Gray codes at
$n = 4, 5, 6$ (`results/theory_checks.json`; the thermometer upper bound for $R_<$ is
$\bigcup_k \{i \le k\} \times \{j > k\}$).

**Corollary A8 (relative literals) [proved].** Add the **relative code** $r(i, j) =$
thermometer of $j - i$ over $\{-(L-1), \dots, L-1\}$ ($2L - 2$ bits $[j - i > t]$). Then
$R_<$ is one literal $[j - i > 0]$, $R_d$ is two literals $[j - i > d-1] \land \lnot[j - i > d]$,
and every offset interval costs ≤ 2 literals (Proposition A1). The code is exactly
**shift-equivariant**: $r(i + t, j + t) = r(i, j)$.

This is the Boolean reason relative encodings exist. Sinusoidal and rotary encodings
(Vaswani et al., 2017; Su et al., 2021) make $\langle q_i, k_j \rangle$ depend on $j - i$ through a
*product* of position factors; a conjunction has no such product across the two positions
(rectangle property), so in a TM the offset has to be **given as a pair feature**. Theorem A7
says no choice of absolute code can avoid this.

**Bucketed relative code.** $2L-2$ bits per pair makes the pair input cubic in $L$ (§B.5). A
log-bucketed code (exact offsets up to $\pm 8$, then thermometer over $\lceil \log_2 |j-i| \rceil$
with sign; as in T5, Raffel et al., 2020) keeps $O(\log L)$ bits; intervals of offsets that align
with bucket boundaries still cost ≤ 2 literals, other intervals become unreachable. It is an arm
in the plan.

---

## B. Key–query–value attention as a clause over pairs

### B.1 The pair clause

For a set of admissible pairs $\mathcal{P} \subseteq \{0..L-1\}^2$ (all pairs, a causal set
$j \le i$, or a window) and a sender window $w \ge 1$, define the **pair vector**

$$\pi(i, j) = \big[\, e(x_i) \;\big\|\; \tilde e(x_j), \dots, \tilde e(x_{j+w-1}) \;\big\|\; r(i, j) \;\big\|\; a(i) \,\big]$$

where $\tilde e$ is the one-hot over $\Sigma \cup \{\mathrm{END}\}$ (past the end), $r$ the relative
code and $a(i)$ optional receiver features (absolute position, an "is last" bit). A **pair clause**
is

$$C(X) = \bigvee_{(i, j) \in \mathcal{P}} \; \bigwedge_{k \in I} \ell_k(\pi(i, j)).$$

This is exactly a convolutional TM whose "patches" are the pair vectors. `code/pairs.py` builds
them as columns so that `Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)` runs it
**on the unchanged library**; `tests/test_pairs.py` checks the columns against a naive loop.

**Proposition B1 (a pair clause is Boolean hard attention) [proved].** Split $I$ by block,
$I = I_Q \cup I_K \cup I_R \cup I_A$. By the rectangle property

$$C(X) = \bigvee_{(i,j) \in \mathcal{P}} \underbrace{Q(x_i) \land A(i)}_{\text{query}} \land \underbrace{K(x_j, \dots, x_{j+w-1})}_{\text{key}} \land \underbrace{R(j - i)}_{\text{relative position}}$$

with $Q, K, R, A$ conjunctions. Reading $\alpha_{ij} = Q(x_i) \land A(i) \land K(x_{j..}) \land R(j-i)$
as a Boolean attention matrix, $C(X) = \bigvee_{i,j} \alpha_{ij}$: hard attention with **existential
aggregation**. ∎

The analogy to softmax attention $\sum_j \mathrm{softmax}_j(\langle q_i, k_j \rangle) v_j$:
the score becomes a conjunction, the softmax becomes "any match", and the sum over receivers
becomes the OR over $i$.

**Proposition B2 (keys and values collapse under OR) [proved].** For any Boolean "value test"
$V$ on the sender, $\bigvee_{i,j} [\alpha_{ij} \land V(x_j)] = \bigvee_{i,j} [Q(x_i) \land (K \land V)(x_j) \land R(j-i)]$,
again a pair clause. So under existential aggregation a separate value map adds nothing: the
literals on the sender block play the key and value role at once. ∎

A separate $V$ matters only when aggregation is not a monotone OR: *unique / argmax* reads ("the
value at the best-matching key") or counting reads. §B.4 treats both.

**Proposition B3 (separate Q and K clause banks are a tied pair clause) [proved].** Any
one-layer model whose attention is $a_{ij} = Q_r(x_i) \land K_r(x_j) \land R(j-i)$ for clauses
$Q_r, K_r$ from two banks, aggregated by OR, computes a pair clause whose include set is the union
of the include sets. A two-bank design is therefore the pair TM with **parameter tying** (one $Q_r$
shared by several heads), not a more expressive model. ∎

*Scope.* B3 covers the AND-match compatibility $a_{ij} = \bigvee_r Q_r(x_i) \land K_r(x_j)$ of the
brief (option a). The popcount compatibility $\langle Q(x_i), K(x_j) \rangle \ge \theta$ (option b)
is a threshold of $m$ pair conjunctions *at one pair*; as a DNF it is $\bigvee_{|S| = \theta}
\bigwedge_{r \in S} Q_r(x_i) \land K_r(x_j)$, i.e. $\binom{m}{\theta}$ pair clauses. It is **not** a
TM vote over the $m$ pair clauses $Q_r \land K_r$: that vote is $\sum_r \exists (i,j)\, [Q_r(x_i)
\land K_r(x_j)]$, which counts clauses matched by *some* pair, not the best pair's score.

This is why this programme drops the two-stage Q/K/V pipeline of `experiments/transformer_like/`
(frozen Q/K TMs trained on proxy labels). The pair clause learns all blocks jointly from the class
signal (§C).

### B.2 Relation to the existing models

**Proposition B4 [proved].**

1. *Self-only attention is the conv TM.* With $\mathcal{P} = \{(i, i)\}$ the relative code is a constant block, and the receiver block duplicates the first sender token. So the pair vectors are the kernel-$w$ token windows plus constant and duplicated literals, and the pair TM computes exactly the functions of a conv TM with kernel $w$ and no position literals over the sequence **right-padded with $w - 1$ END symbols** (there are $L$ self pairs but only $L - w + 1$ unpadded windows; the END literals let a clause say "the last token is $a$"), plus the receiver extras $a(i)$, which play the role of position literals. Training is not bit-identical: the constant literals still receive feedback (an always-false literal that Type II includes switches the clause off). The arm `pair_self_only` is this control.
2. *The pair TM sees more than any conv TM with kernel $< \max |j - i| + w$ on the tasks of §D,*
   and less than a conv TM whose kernel covers the whole sequence (the flat TM), which can
   represent any Boolean function but must learn a separate clause per absolute placement
   (Proposition B6).
3. *Graph TM of depth 2 on the complete graph.* Take `GraphTsetlinMachine(depth=2)` with one node
   per token and an edge $j \to i$ for every $j \ne i$, typed by a relative-offset bucket $t(j - i)$.
   Layer 1's inbox at node $i$ holds message $(k, t)$ iff $\exists j: t(j - i) = t \land$ clause $k$'s
   layer-0 component matches $x_j$ (Granmo et al., 2025; `models/graph.py`). A depth-2 clause is
   therefore
   $$Q(x_i) \land \bigwedge_{(k, t) \in I_1} \exists j\, [t(j-i) = t \land K_k(x_j)] \land \bigwedge_{(k,t) \in I_1^-} \lnot \exists j\, [\dots]$$
   at some node $i$: **several existential attention reads at one receiver**, plus negated reads.
   Each read's key $K_k$ is *another clause's* layer-0 part, shared across clauses (tied keys).
   The pair TM has one read per clause but any key and offset; the graph TM has many reads per
   clause and negation, but keys borrowed from layer 0 and offsets limited to the edge types. ∎

### B.3 Expressivity: what attention buys

All tasks are defined in `code/tasks.py`; §D works them by hand.

**Proposition B5 (order, T1) [proved, checked].** One pair clause decides T1 exactly:
$[q = A] \land [k_0 = B] \land [j - i > 0]$. A conv TM without position literals cannot do better
than chance (Corollary A5). A conv TM with thermometer positions needs a vote over $2(L-1)$
clauses per class in the construction of Proposition A6 (no matching lower bound proved:
**[conjecture]** $\Omega(L)$).

**Proposition B6 (needle at distance $d$, T4) [proved; 3 checked].** Domain: exactly one marker
$M$ at a position $p \le L-1-d$; class $b$ iff $x_{p+d} = b$; other positions hold arbitrary
non-marker symbols; $V \ge 4$. Then

1. one pair clause per class suffices: $[q = M] \land [k_0 = b] \land [j - i > d-1] \land \lnot[j-i > d]$
   (checked: accuracy 1.0 on 5,000 generated examples, `results/smoke_t4_s.json`);
2. a positive-clause flat TM needs $\ge L - d$ clauses per class;
3. a conv TM without position literals and kernel $k \le d$ cannot compute it when $L \ge d + 2k$;
4. a conv TM without position literals and kernel exactly $d+1$ needs one clause per class; with
   kernel $k > d+1$ two clauses per class suffice when $L \ge 2k - d - 2$.

*Proof of 2.* Fix a filler $\varphi \notin \{M, b\}$ and the canonical positives $e_p$: marker at
$p$, $b$ at $p+d$, $\varphi$ elsewhere, $p = 0..L-1-d$. Suppose one flat clause $t$ (an implicant of
class $b$ on the domain) is true on $e_p$ and $e_{p'}$ with $p < p'$. Build the mix $z$: take
positions $p$ and $p'$ from $e_{p'}$ (so $z_p = \varphi$, $z_{p'} = M$) and every other position
from $e_p$. Then $z_{p'+d} = (e_p)_{p'+d} = \varphi$, because $p' + d \notin \{p, p+d\}$. So $z$ is in
the domain (one marker, at $p'$) and has label $\varphi \ne b$. By the rectangle property $t(z) = 1$,
contradicting that $t$ is an implicant. Each clause covers at most one $e_p$, and there are $L - d$
of them. ∎

*Proof of 3.* Take distinct symbols $M, b, c, \varphi$ and $X_1$: $M$ at $0$, $b$ at $d$, $c$ at
$d + k$, $\varphi$ elsewhere, $L \ge d + 2k$; $X_2$: the same with $b$ and $c$ swapped. Both are in
the domain ($p = 0 \le L - 1 - d$) with labels $b$ and $c$. Their $k$-gram sets are equal: the
only window containing position $0$ starts at $0$ and ends at $k - 1 < d$, so it is
$M \varphi^{k-1}$ in both; the windows containing $d$ start at $d-k+1, \dots, d$, all $\ge 1$ and
all ending before $d + k$, so they are $\varphi^{a} b \varphi^{k-1-a}$ for every $a = 0..k-1$;
the windows containing $d + k$ start at $d+1, \dots, d+k \le L - k$, all after $d$, so they are
$\varphi^{a} c \varphi^{k-1-a}$ for every $a$; every other window is $\varphi^k$. Swapping $b$ and
$c$ maps this set to itself. By Proposition A4 every clause, hence every vote, agrees on $X_1$ and
$X_2$. ∎ (Brute force over $1 \le k \le d \le 6$: `results/theory_checks.json`, key
`needle_kgram`. The first version of this proof put $c$ at $2d + k + 1$ with $L = 2d + k + 2$;
there $c$ sits in the last window only, so the $k$-gram sets differ for $k \ge 2$. Caught by the
same check.)

*Proof of 4.* Kernel $d + 1$: the clause $[\text{first} = M] \land [\text{last} = b]$ on
$(d+1)$-grams; the window starting at the marker exists because $p \le L - 1 - d$. Kernel
$k > d + 1$: a clause must fix the marker's offset $o$ inside the window and then covers the
markers $p \in [o, o + L - k]$. The offsets $o = 0$ and $o = k - 1 - d$ (the largest with
$o + d \le k - 1$) cover $[0, L-k] \cup [k-1-d, L-1-d]$, which is all of $[0, L-1-d]$ iff
$k - 1 - d \le L - k + 1$. ∎

So for a **fixed** offset a wide enough conv kernel is as good as attention; the conv TM with
kernel $d + 1$ is the "oracle window" baseline in the plan. Attention should win when the
relevant offset **varies per example** (T2, T3) or exceeds any affordable kernel.

**Proposition B7 (associative recall, T2) [proved; 1 checked].** Sequence
$k_1 v_1 \dots k_n v_n\, q$ with distinct keys from $\Sigma_K$, values from $\Sigma_V$, query
$q \in \{k_1..k_n\}$, label the value after $q$'s occurrence. $L = 2n + 1$.

1. With sender window $w = 2$ and an "is last" receiver bit, $|\Sigma_K|$ pair clauses per class
   suffice: $c_{a,b} = [i = L-1] \land [q = a] \land [k_0 = a] \land [k_1 = b] \land \lnot[j - i > -1]$.
   (`tests/test_pairs.py::test_hand_clauses_solve_associative_recall`: exact, and exactly one class
   fires per example.) The "is last" literal is redundant on T2 as generated (keys are distinct
   and value symbols never equal key symbols, so $[q = a]$ with a $j < i$ carrying $x_j = a$
   already forces $i$ to be the query slot); it is kept for readability and for variants where a
   value may equal a key.
2. A positive-clause flat TM needs $\ge n$ clauses per class. *Proof.* Fix the key arrangement
   $k_1, \dots, k_n$ and a value $c \ne b$. The canonical positives $e_s$, $s = 1..n$, have value
   $b$ at slot $s$, $c$ at every other slot and query $k_s$. Let a flat clause $t$ be an implicant
   of class $b$ true on $e_s$ and $e_{s'}$, $s \ne s'$. The mix $z$ that takes the query position
   from $e_{s'}$ and everything else from $e_s$ has values $b$ at $s$, $c$ elsewhere (in particular
   at $s'$) and query $k_{s'}$, so it is in the domain with label $c$; by the rectangle property
   $t(z) = 1$, a contradiction. Each clause covers at most one canonical positive. ∎

**Conjecture B7.3 (exact flat cost on recall) [conjecture; 6 exact instances].** The minimum
positive flat DNF for one class of T2 has $n \cdot \min(|\Sigma_K|, 2^n)$ clauses. Exact values
(`results/exact/X10.json`): $(n, |\Sigma_K|) = (2,3), (2,4), (2,5), (2,6), (3,3), (3,4)$ give
$6, 8, 8, 8, 9, 12$, matching the formula in every case. The saturation at $2^n$ comes from
clauses that constrain the query and the key slots to *sets* of keys ("query $\in A$, key at $s$
$\in A$, other keys $\notin A$"), which a pair clause cannot state.

The proved bound alone does **not** separate the pair TM from the flat TM in clause count
($|\Sigma_K|$ vs. $\ge n$ per class). If B7.3 holds, the flat TM needs $n$ times the pair TM's
clauses whenever $|\Sigma_K| \le 2^n$, and the pair TM needs more when $|\Sigma_K| > n \cdot
2^n$. What
the pair clause gains is **sharing**: $c_{a,b}$ is trained by every example where key $a$ maps to
$b$, at any slot, while a flat clause is tied to one slot. The expected effect is on **sample
complexity** **[empirical]**. With $w = 1$, whether a vote over pair clauses can solve T2 exactly
is open; the label depends on three tokens (query, key, value) and a pair clause sees two
**[conjecture: no, with $o(n)$ clauses per class]**.

**Proposition B8 (repeat detection, T3) [proved for the pair TM].** "Some symbol occurs twice"
is $\bigvee_{a \in \Sigma} \exists (i, j): [q = a] \land [k_0 = a] \land [j - i > 0]$: $V$ pair
clauses, two of them needed per symbol only if the class is split. Equality of two tokens is the
pairwise relation a pair clause expresses directly; a flat TM needs a clause per (symbol, position
pair) in the obvious construction. Lower bounds for the flat and conv TMs on T3 are
**[conjecture]** (a conv TM with kernel $k$ sees repeats only within distance $< k$).

### B.4 Beyond existential reads

**Proposition B9 (exclusion through polarity) [proved].** With one positive clause $C^+$ of
weight $w^+$ and one negative clause $C^-$ of weight $w^- \ge w^+$ in a class (and nothing else),
$v_c > 0 \iff C^+ \land \lnot C^-$. So a TM class score can express "a matching pair exists and no
pair of another kind exists": the Boolean version of a softmax that is *suppressed* by a
competing key. ∎

What one pair clause should **not** be able to express is a read conditioned on *another*
sender, such as "the value after the **last** occurrence of key $a$", which is
$\exists j\, [\dots] \land \lnot \exists j' > j\, [x_{j'} = a]$: an existential over two positions
inside which a universal over a third position sits. A pair clause is a single existential over
two positions with a window. Candidates for the remedy are a triple clause ($O(L^3)$ inputs),
the depth-2 graph TM (negated inbox literal, Proposition B4.3) or stacking. **[sketch: the
impossibility for a single pair clause of window $w$ needs a shifting argument over sequences
with two $a$-blocks more than $w$ apart; not written out. The depth-2 construction is also a
sketch.]**

*Counting reads* (softmax with equal scores averages the values; a TM analogue is
$[\#\{(i,j) : \alpha_{ij}\} \ge \theta]$) are not available in the library: the convolutional OR
saturates at one match. They are left as an optional arm that needs its own `_evaluate`
(experiment code only), with Type I feedback still drawn from a random matching pair.

### B.5 Cost

Per example, with $K$ classes and $M$ clauses per class (bit-packed evaluation does $2F/64$ word
operations per clause and patch):

| model | patches $P$ | features per patch $F$ | clause-evaluation work |
|---|---|---|---|
| flat TM | 1 | $LV$ | $KM \cdot 2LV$ |
| conv, kernel $k$, thermometer PE | $L - k + 1$ | $kV + L - k$ | $KM \cdot L \cdot 2(kV + L)$ |
| pair, all pairs, full relative code | $L^2$ | $V + w(V+1) + 2L - 2\ (+1)$ | $KM \cdot L^2 \cdot 2((w+1)V + 2L)$, **cubic in $L$** |
| pair, bucketed relative code | $L^2$ | $(w+1)V + O(\log L)$ | $KM \cdot L^2 \cdot 2((w+1)V + \log L)$ |
| pair, causal or window $W$ | $L(L+1)/2$ or $LW$ | as above | halves, or $L W$ instead of $L^2$ |
| graph TM, depth 2, complete graph | $L$ nodes, $L(L-1)$ edges | $V + $ inbox | inbox build $O(L^2 M)$ per layer, scatter-OR over edges |

The pair input is materialised as a `(B, F, L²)` Boolean tensor (one byte per bit), which the
model expands to `(B, L², 2F)` literals before `conv_eval` bit-packs them. With $V = 16$, $w = 2$,
the full relative code and the last bit, $F = 3V + 2 + 2L - 2 + 1 = 2L + 49$
(`pairs.pair_feature_names`):

| $L$ | $F$ | $P = L^2$ | packed | Boolean input per example | literals per example |
|---|---|---|---|---|---|
| 32 | 113 | 1,024 | 14 KB | 0.12 MB | 0.23 MB |
| 64 | 177 | 4,096 | 89 KB | 0.72 MB | 1.45 MB |
| 256 | 561 | 65,536 | 4.4 MB | 37 MB | 74 MB |

A batch of 32 at $L = 64$ is ≈ 70 MB in flight: fine. But 20,000 training examples at $L = 64$
are 14.5 GB as Boolean pair columns, so the harness must build pair columns **per batch** from
the `(N, L)` token tensor (which is what `pairs.pair_tokens` does; it is a gather, not a
computation) rather than pre-building the dataset. At $L = 256$ even one example is large, so the
bucketed code or a window is needed. The graph TM never builds the pair tensor, but its edge
list is $O(L^2)$.

---

## C. Learning under TM feedback

**Proposition C1 (the pair TM learns with the conv TM's feedback, unchanged) [proved].** Because the
pair TM *is* a conv TM over pair vectors (§B.1), its Type Ia and Type II events each update the
clause from one uniformly random matching **pair**. Every literal block (query, key, relative
position, receiver extras) receives feedback from the class signal through the same rule.
Nothing is trained on proxy labels, and no credit has to pass between modules. ∎

This removes the two failure modes of `experiments/transformer_like/`: the Q/K encoders there were
fitted separately on per-patch digit labels (so Q and K lived in the same 10-d class space and
cross-patch similarity carried no new information), and nothing pushed the head's error back into
them.

**Remark C2 (what is and is not known about convergence) [conjecture / empirical].** Convergence
proofs exist for the vanilla TM on small operators (identity/NOT, AND, XOR; Zhang et al., 2022;
Jiao et al., 2023; *not in `source_documents/papers/`, verify before citing*). We know of none for
the convolutional TM, so none transfers to the pair TM. The heuristic that does transfer:

- **The planted rule is not a fixed point by itself; the usual $s$ trade-off decides.** Let
  $C^\star$ be the planted clause of T1. On a positive example it matches exactly one pair,
  $(\mathrm{pos}A, \mathrm{pos}B)$, so every Type Ia event uses that pair. Type Ia pushes *every*
  literal true on that pair towards inclusion with probability $p_\text{mem}$ (1 by default, see
  the next bullet), including the offset
  literals specific to this example (e.g. $\lnot[j - i > 3]$ when the offset is 3). The clause then
  over-specialises to one offset, stops matching positives with other offsets, and Type Ib pushes
  its literals towards exclusion with probability $1/s$. Literals true on the drawn pair of
  *most* positive examples ($[q=A]$, $[k_0=B]$, $[j-i>0]$) win this tug of war. Offset-specific
  literals win it only if $s$ is large relative to the spread of offsets. This is the flat TM's
  frequency/specificity balance with "true on the drawn pair" in place of "true on the example".
- **The inclusion threshold, on the drawn pair.** [drift formula: verified, D1; its use as a
  learning criterion: heuristic] Under Type Ia a literal is pushed towards inclusion with
  probability $p_\text{mem}$ when it is true on the drawn pair and towards exclusion with
  probability $1/s$ when it is false. In torchtsetlin $p_\text{mem} = 1$ by default
  (`boost_true_positive=True`) and $(s-1)/s$ otherwise (`functional.apply_feedback`). With
  $\rho = \Pr[\ell = 1 \mid \text{drawn pair}]$, the drift relative to an always-true included
  literal is
  $$\frac{E[\Delta \ell]}{E[\Delta \ell_\top]} = \rho - \frac{1 - \rho}{s \, p_\text{mem}},$$
  towards inclusion iff $\rho > 1/(s+1)$ by default ($\rho > 1/s$ without boosting). For a flat
  clause the conditioning event is "the clause fires on the example"; for a pair clause it is
  "the clause fires on the example **and** this pair is the one drawn among its matching pairs".
  *Correction (2026-11-08, night):* the review version of this bullet used $p_\text{mem} =
  (s-1)/s$ for the default model; experiment D1 (`results/dynamics/D1_first_run.json`) failed in
  29/48 cells because of it. With the library's actual $p_\text{mem}$ the formula holds in 96/96
  cells (both modes, CPU and CUDA, batched and sequential; `results/dynamics/D1.json`). The same
  measurement confirms that the drawn pair is uniform among the matching pairs (C1): the ratio
  is linear in $\rho = r/m$ with the predicted slope for $m = 4$ and $m = 16$.
- **Dilution grows with the number of pairs.** An under-specified clause (say only $[q = M]$ on
  T4) matches $L - 1$ pairs of every positive example and the drawn pair is uniform among them, so
  the literal the clause still needs, $[k_0 = b]$, is true on the drawn pair with probability
  about $1/(L-1)$ plus the distractor rate. By the previous bullet it is included only if that
  exceeds $1/s$: at $L = 12$ this needs $s \gtrsim 11$. A flat clause has no such factor, and a
  conv clause has $1/(\#\text{matching patches})$. Prediction: the $s$ needed to get off the
  ground grows about linearly with $L$ (for all-pairs attention), and learning slows as $L$ grows.
  The smoke agrees with the $s$ part: on T4 at $L = 12$, $s = 5$ stays at chance after 20 epochs
  with $M = 40$, and $s = 15$ reaches 0.98 (`results/smoke_t4_s.json`). The same threshold
  explains the first bullet: an offset-specific literal such as $\lnot[j - i > 3]$ is true on the
  drawn pair of roughly a $1/(L-1)$ share of T1 positives, so it is kept iff $s \gtrsim L - 1$.
  Large $s$ is thus needed to *find* the key literal and at the same time lets the clause
  over-specialise; the search in `PLAN.md` §4 is designed around this trade-off.

  *Test (D2, 2026-11-08 night; `REPORT.md` Part II).* With the default boost the prediction is
  $s > s_\text{pred} = 1/\rho - 1$, $\rho = 1/L + (1 - 1/L)\pi$. **Direction confirmed; magnitude
  partly.** (i) The no-dilution control (conv TM with kernel $d+1$, one matching window) works at
  every $s \in [2, 64]$ for $L = 16$ and $32$. (ii) For the pair TM the lowest working $s$ rises
  with $L$ ($V = 16$: any $s$ at $L \le 12$; 8 at $L = 16$–24; 24 at $L = 32$) and with $V$ at
  $L = 32$ (12, 24, none for $V = 6, 16, 48$), the order the formula predicts. (iii) The values
  match $s_\text{pred}$ at $L = 16$–24 (7–8.5) but are about $2.5\times$ larger at $L = 32$. At
  small $L$, $s$ well below $s_\text{pred}$ also works, which the single-literal argument does
  not explain: other routes to the planted clause exist. (iv) The upper edge (accuracy falls
  for large $s$ at $L = 32$) is the over-specialisation of the first bullet. (v) The flat TM
  shows the same effect at least as strongly (no working $s$ at $L = 32$ with this budget), so
  needing high $s$ is not specific to attention: the flat TM is diluted over the positions of
  the needle in the same way. A drift argument on one literal is a useful guide to the
  direction, not a quantitative law.

**Proposition C3 (graph TM, depth 2) [proved for the feedback rule; learning dynamics empirical].**
A deep graph clause is a single conjunction over the node literal row $[H^0 \| H^1]$. Its feedback
is the conv feedback on that row with nodes for patches (`_GraphMixin._feedback_counts`), so
layer-0 and layer-1 literals of a clause learn together. However, the inbox literals $(k, t)$
refer to *other* clauses' layer-0 parts, which keep changing while those clauses learn. The
receiver therefore learns on non-stationary features. This is the obstacle of the MCTM programme
(credit cannot be pushed down through clause inclusion), in a milder form: all clauses learn
against the same class signal at the same time, whereas MCTM stacked layers greedily.

**C4 (settings that interact with this) [empirical, with priors].**

- *Specificity.* Pair vectors are wide ($F \approx (w+1)V + 2L$) and the planted rules are short
  (2–5 literals). The prior from the smoke and from the flat TM theory is that $s$ must be higher
  than for images. Search $s \in \{5, 10, 15, 25\}$.
- *Clause budget.* `max_included_literals` (soft, tmu rule) fits naturally: the target rules
  have ≤ 5 literals. Search {none, 8}.
- *T/M.* $T/M \approx 0.25$ and weighted clauses (lessons of `transformer_like`). Unweighted is a
  secondary arm only.
- *Batched feedback.* A batch of 32 means 32 random matching pairs per selected clause per update.
  The batch-vs-sequential gap is expected to be no larger than for the conv TM, since the update
  rule is the same. A sequential-mode check on one task is in the plan.

---

## D. Worked examples

Alphabet sizes as in `code/tasks.py`. "Clauses" means clauses per class, prediction mode.

### D.1 T1, ordered pair, $L = 9$, $V = 6$

| model | construction | clauses | literals each |
|---|---|---|---|
| pair TM, $w = 1$ | $[q=A] \land [k_0=B] \land [j-i>0]$ (class 1) | 1 | 3 |
| conv, $k=1$, no PE | impossible (Corollary A5) | – | – |
| conv, $k=1$, thermometer PE | $f_k, g_k$ / $h_k, e_k$ of Proposition A6 | $2(L-1) = 16$ | 2 |
| flat TM, positive, naive | $\bigvee_{a<b}$ $[x_a=A] \land [x_b=B]$ | $\binom{L}{2} = 36$ | 2 |
| flat TM, with negations | $[x_a = A] \land \bigwedge_{q<a} \lnot[x_q = B]$ for $a = 0..L-2$ | $L - 1 = 8$ | $\le L - 1$ |
| **flat TM, optimal** (Prop. D1) | free position $f = 2t-1$: $\bigwedge_{q<f} \lnot[x_q = B] \land \bigwedge_{q>f} \lnot[x_q = A]$ | $\lceil (L-1)/2 \rceil = 4$ | $L - 1$ |

**Proposition D1 (exact flat cost of T1) [proved, checked].** The minimum positive flat DNF for
"A before B" has exactly $\lceil (L-1)/2 \rceil$ clauses.

*Upper bound.* For odd $f \le L - 1$ let $t_f$ say "no B before $f$, no A after $f$". On the domain
(one A, one B) $t_f$ forces $\mathrm{pos}A \le f \le \mathrm{pos}B$ with the two distinct, so A is
before B: $t_f$ is an implicant. A positive $(a, b)$, $a < b$, is covered by any odd
$f \in [a, b]$, which exists ($a$ or $a+1$). The odd $f \le L-1$ number $\lceil (L-1)/2 \rceil$.
*Lower bound.* The canonical positives $e_a$ (A at $a$, B at $a+1$, filler elsewhere) for even
$a \le L-2$ form a fooling set: for $a < a'$ (so $a' \ge a + 2$) the mix taking positions
$a, a'$ from $e_{a'}$ and the rest from $e_a$ has B at $a+1$ and A at $a' > a+1$: a negative. ∎
Exact minima for $L = 4..7$ (`X10`) and the construction loaded into the library flat TM for
$L$ up to 33 (`X12`) agree. So on T1 the flat TM needs about $L/2$ clauses, the pair TM one, and
the conv TM with positions $2(L-1)$ in the construction of A6.

The pair clause is checked exactly on 500 examples (`tests/test_pairs.py::test_hand_clause_solves_ordered_pair`).
Class 0 is the same clause with $\lnot[j - i > 0]$; a single positive clause for class 1 plus the
$\arg\max$ also works.

### D.2 T4, needle at distance 4, $L = 12$

Pair TM: 1 clause per class, 4 literals ($q = M$, $k_0 = b$, $[j-i>3]$, $\lnot[j-i>4]$). Flat
(positive DNF): $\ge L - d = 8$ per class (Proposition B6.2), matched by
$\bigvee_p [x_p = M] \land [x_{p+4} = b]$. Conv without PE: impossible for $k \le 4$; one clause for
$k = 5$.

### D.3 T2, associative recall, $n = 4$ slots, $|\Sigma_K| = 6$, $|\Sigma_V| = 3$, $L = 9$

Pair TM ($w = 2$, last bit): $|\Sigma_K| = 6$ clauses per class, 5 literals each (Proposition
B7.1, tested). Flat: $\ge 4$ per class; the natural DNF has $4 \times 6 = 24$. The pair TM's
advantage here is sharing across slots, not clause count.

---

## E. Summary of claims

| # | claim | status |
|---|---|---|
| A1 | thermometer: every position interval ≤ 2 literals | proved, checked |
| A2 | one-hot: interval $S$ costs $n - \lvert S\rvert$ literals | proved, checked |
| A3 | binary/Gray: intervals split into $O(\log n)$ clauses (values tabulated) | characterisation proved; values checked |
| A4 | conv OR without PE depends only on the set of patches | proved |
| A5 | T1 at chance for single-token sets | proved |
| A6 | conv + thermometer PE solves T1 with $2(L-1)$ voting clauses | proved |
| A7 | no absolute code makes $j - i = d$ or $i < j$ cheaper than $n - d$ / $n - 1$ clauses | proved, checked |
| A8 | relative thermometer code: offset relations ≤ 2 literals, shift-equivariant | proved |
| B1 | a pair clause = Boolean hard attention with existential aggregation | proved |
| B2 | key and value collapse under OR aggregation | proved |
| B3 | separate Q/K clause banks = tied pair clause | proved |
| B4 | self-only pair TM = conv TM on the END-padded sequence; depth-2 graph TM = multi-read attention with tied keys | proved |
| B5 | T1: 1 pair clause | proved, checked |
| B6 | T4: 1 pair clause/class vs ≥ $L-d$ flat; conv needs kernel > $d$ (1 clause at $d+1$, 2 above) | proved; 1 and 3 checked |
| B7 | T2: $\lvert\Sigma_K\rvert$ pair clauses/class ($w=2$); flat ≥ $n$; advantage is sharing | proved; 1 checked |
| B7' | T2 not solvable by $w=1$ pair votes with $o(n)$ clauses | conjecture |
| B7.3 | flat recall needs $n \cdot \min(\lvert\Sigma_K\rvert, 2^n)$ clauses | conjecture (6/6 exact instances) |
| A3b | binary $[1, n-2]$ needs exactly $\log_2 n$ clauses | proved, checked |
| D1 | flat T1 needs exactly $\lceil (L-1)/2 \rceil$ clauses | proved, checked |
| B8 | T3: $V$ pair clauses | proved (flat/conv lower bounds: conjecture) |
| B9 | positive + dominating negative clause = "exists and not exists" | proved |
| B9' | "value after the last $a$" is beyond one pair clause | sketch |
| C1 | pair TM uses the conv feedback unchanged; all blocks learn from the class signal | proved |
| C2 | convergence of conv/pair TM; $1/s$ threshold on the drawn pair; required $s$ grows with $L$ | open; mean-drift heuristic |
| C3 | depth-2 graph TM: joint feedback, non-stationary inbox features | proved (rule) / empirical |
| — | general TM (threshold + negative clauses) lower bounds | conjecture |

## References

- Granmo, O.-C. (2018). *The Tsetlin Machine.* arXiv:1804.01508. (`source_documents/papers/01_foundations/`)
- Granmo, O.-C. et al. (2019). *The Convolutional Tsetlin Machine.* arXiv:1905.09688. (local)
- Phoulady, A. et al. (2019). *The Weighted Tsetlin Machine.* arXiv:1911.12607. (local)
- Glimsdal, S., Granmo, O.-C. (2021). *Coalesced Multi-Output Tsetlin Machines with Clause Sharing.* arXiv:2108.07594. (local)
- Abeyrathna, K. D. et al. (2023). *Building Concise Logical Patterns by Constraining Tsetlin Machine Clause Size.* arXiv:2301.08190. (local)
- Granmo, O.-C. et al. (2025). *The Tsetlin Machine Goes Deep: Logical Learning and Reasoning with Graphs.* arXiv:2507.14874. (local)
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Shaw, P., Uszkoreit, J., Vaswani, A. (2018). *Self-Attention with Relative Position Representations.* NAACL.
- Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* JMLR (bucketed relative positions).
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
- Zhang, X., Jiao, L., Granmo, O.-C., Goodwin, M. (2022). *On the Convergence of Tsetlin Machines for the IDENTITY- and NOT Operators.* IEEE TPAMI. *(not local; verify)*
- Jiao, L. et al. (2023). *On the Convergence of Tsetlin Machines for the XOR Operator.* IEEE TPAMI. *(not local; verify)*

**Novelty.** No paper in `source_documents/papers/` or `benchmarks/literature/` defines attention
for TMs. The closest prior work is the graph TM (message passing over given edges) and
`experiments/transformer_like/` (this repository, negative on MNIST). That does not rule out work
outside the local collection; a literature search is task P0.2 of `PLAN.md`. The pair-token
construction is a direct consequence of the convolutional TM and should be presented as such,
not as a new learning algorithm.
