# Review of THEORY.md (2026-11-08)

A proposition-by-proposition check of `THEORY.md` as written on 2026-11-08, done before any
experiment. Every proof was re-derived by hand; the combinatorial claims were additionally checked
by brute force (`code/verify_theory.py` → `results/theory_checks.json`) and the two cost/memory
tables were recomputed from `code/pairs.py`. The file was corrected in place; this note records
what was wrong, what was weak, and what was confirmed.

## Verdict

The main line holds: a pair clause is Boolean hard attention (B1), the relative offset must be a
pair feature because no absolute code can supply it (A7, the one genuine theorem in the file),
and the pair TM learns with the convolutional feedback unchanged (C1). Of the 19 claims in the
summary table, one proof was wrong, one statement was false as stated, one memory table was
wrong by 2–4×, and two status labels were too strong. Nothing that the plan depends on changed
direction, but two consequences for the plan came out of the review (D8, D9 in `DECISIONS.md`).

## Errors found and corrected

### 1. B6.3 — the conv-TM impossibility proof was wrong (serious)

**Claim:** a conv TM without position literals and kernel $k \le d$ cannot solve the needle task.

**What was wrong:** the two sequences meant to have identical $k$-gram sets but different
labels placed the second labelled symbol $c$ at position $2d + k + 1$ in a sequence of length
$2d + k + 2$. That puts $c$ in the **last** window only, so $c$ appears in exactly one $k$-gram
(at its last slot) while $b$ appears in $k$ of them (at every slot). The sets differ for every
$k \ge 2$. Brute force: 15 of the 21 cases with $1 \le k \le d \le 6$ fail
(`theory_checks.json`, `needle_kgram.original_failures`).

**Fix:** put $c$ at $d + k$ with $L \ge d + 2k$. Then the windows containing $b$ are
$\varphi^{a} b \varphi^{k-1-a}$ for all $a$, the windows containing $c$ are
$\varphi^{a} c \varphi^{k-1-a}$ for all $a$, the marker's window is $M\varphi^{k-1}$, and
everything else is $\varphi^k$; swapping $b$ and $c$ fixes the set. The written proof now lists
each window family. Brute force over $1 \le k \le d \le 6$: all equal
(`needle_kgram.fixed_all_equal`). The proposition also now states its real hypotheses
($L \ge d + 2k$, $V \ge 4$, non-marker fillers).

### 2. B6.4 — "kernel $\ge d+1$ needs one clause" is false for $k > d+1$

For kernel exactly $d + 1$ the clause $[\text{first} = M] \land [\text{last} = b]$ works, because
the window starting at the marker always exists. For a wider kernel a clause has to fix the
marker's offset $o$ inside the window, and the window with that offset does not exist for
markers near the end. Corrected statement: one clause at $k = d+1$; two clauses (offsets $0$ and
$k - 1 - d$) for $k > d+1$ when $L \ge 2k - d - 2$. The plan's "oracle window" arm B3 already
used $k = d + 1$, so no change there.

### 3. B.5 — pair-tensor memory figures were wrong

The text said $4096 \times 300$ bits ≈ 150 KB per example at $L = 64$ and ≈ 10 MB at $L = 256$.
From `pairs.pair_feature_names` with $V = 16$, $w = 2$, last bit: $F = 2L + 49$, so $F = 177$ at
$L = 64$ (89 KB packed, but **0.72 MB** as the Boolean input tensor the library receives and
1.45 MB as literals) and $F = 561$ at $L = 256$ (**37 MB** as Boolean, not 10 MB). The relevant
number for the harness is the dataset: 20,000 examples at $L = 64$ are 14.5 GB as Boolean pair
columns. Consequence: pair columns must be gathered per mini-batch, never pre-built (D8 in
`DECISIONS.md`, §8 of `PLAN.md`). A table with the recomputed values replaces the sentence.

### 4. D.1 — flat-TM clause counts were mislabelled

The row gave "$\le L - 1$ clauses" for the positive construction $\bigvee_{a<b} [x_a = A] \land
[x_b = B]$, which has $\binom{L}{2}$ clauses. The $L - 1$ count belongs to the construction with
negations, $[x_a = A] \land \bigwedge_{q < a} \lnot [x_q = B]$. Split into two rows with the right
numbers (36 and 8 at $L = 9$).

### 5. A3 — "the minimum number grows with $\log n$" was unsupported

The dyadic decomposition of an interval gives the **upper** bound $2\lceil \log_2 n \rceil - 2$
clauses. The brute-force minima are smaller than that bound (3 instead of 4 at $n = 8$), because
a clause whose truth set is not an interval can still lie inside the target interval
($[1, 6] = \{1, 3\} \cup \{2, 6\} \cup \{4, 5\}$ in 3-bit binary). No lower bound growing with
$\log n$ was proved, and the text now says so. The characterisation (a subcube with $\ge 2$
elements is an interval iff its free bits are the lowest $r$ bits) was added with its one-line
proof; it matches the brute-force counts exactly (7, 15 single-clause intervals at $n = 4, 8$).

## Status labels corrected

- **B7.2 upgraded, sketch → proved.** The mixing argument for the flat-TM lower bound on
  associative recall was complete; only the write-up was missing. Written out.
- **B9′ downgraded, proved → sketch.** "The value after the *last* occurrence of key $a$" was
  claimed to be beyond a single pair clause "by the rectangle property". The rectangle property
  alone does not give that: a pair clause with a sender window $w$ sees $w + 1$ tokens, and ruling
  it out needs a shifting argument with two $a$-blocks more than $w$ apart. Not written out, so
  marked as a sketch.
- **B4.1 made precise.** The self-only pair TM equals a conv TM with kernel $w$ on the sequence
  **right-padded with END**, not on the raw sequence: there are $L$ self pairs but only
  $L - w + 1$ unpadded windows, and the END literals can express "the last token is $a$".
- **A5 scoped.** The chance-level corollary holds for models that see the *set of single
  tokens*. It does not cover kernels $\ge 2$: a set of bigrams reveals order when A and B are
  adjacent. The `pair_self_only` smoke arm ($w = 2$) was wrongly cited as an instance; its 0.471
  is consistent but not implied.
- **B3 scoped.** It covers the AND-match compatibility. The popcount form
  $\langle Q(x_i), K(x_j) \rangle \ge \theta$ is $\binom{m}{\theta}$ pair clauses as a DNF and is
  *not* a TM vote over the $m$ pair clauses (that vote counts clauses matched by some pair, not
  the best pair's score). Added as a remark.

## Weak argument tightened

**C2 (learning heuristics).** The original bullets said "large $s$ helps" with a hand-waving
dilution argument. Checked against `functional.apply_feedback` (`p_mem = (s-1)/s`,
`p_forget = 1/s`): under Type Ia the mean drift of a literal is towards inclusion iff
$\Pr[\ell = 1 \mid \text{drawn pair}] > 1/s$. For a pair clause the conditioning event includes
"this pair was the one drawn among the clause's matching pairs", which divides the frequency of
the needed key literal by the number of matching pairs ($L - 1$ for a one-sided clause on T4).
Two things follow that the plan should expect:

1. the $s$ needed to pick up the key literal grows about linearly with $L$ for all-pairs
   attention ($s \gtrsim 11$ at $L = 12$, which matches the smoke: $s = 5$ fails, $s = 15$ works);
2. the same threshold lets offset-specific literals survive once $s \gtrsim L - 1$, so the clause
   can over-specialise. Both effects pull on $s$ in the same direction at small $L$ and apart at
   large $L$.

This is a mean-drift heuristic, not a convergence result, and is labelled as such. Consequence
for the plan: D9 (extend the $s$ grid at $L = 32$ if the best value is the largest).

## Confirmed as written

- **A1, A2** (thermometer ≤ 2 literals; one-hot exactly $n - |S|$ literals): proofs correct;
  brute force agrees for $n \in \{4, 6, 8, 10\}$. Added the zero-literal edge case (the full
  set) and why it is harmless.
- **A4, A6**: correct. A6's vote identity $v_1 - v_0 = 2(b - a)$ checked exhaustively for
  $L \in \{5, 9, 12\}$ in the review (not recorded; it is a two-line sum).
- **A7**: the fooling-set proofs are correct for *any* position code, since the rectangle
  property only uses that the two position blocks have disjoint variables. The brute-force
  tightness check uses closures of ≤ 2 points, which can overestimate a minimum cover; the found
  covers equal the proved lower bound, so the minimum is pinned regardless.
- **A8, B1, B2, B5, B6.1, B6.2, B7.1, B8, B9, C1, C3**: correct. B6.2's mixing argument was
  re-checked for the case $p' = p + d$ (the marker of $e_{p'}$ lands on the needle slot of
  $e_p$): the mix still has one marker and a filler at $p' + d$, so the proof covers it.
- **B4.3** (depth-2 graph TM as multi-read attention with tied keys): matches the inbox
  semantics in `models/graph.py` (message $(k, t)$ present iff some in-neighbour over an edge
  of type $t$ matches clause $k$'s layer-0 part). With hashed messages the inbox bits collide;
  the statement is exact for `message_encoding="one_hot"`.

## Still open (unchanged)

Lower bounds for the general TM (threshold of weighted clauses with negative polarity) on every
task; the $\Omega(L)$ lower bound for the conv TM with absolute positions on T1; whether $w = 1$
pair votes can solve associative recall; any convergence statement for the convolutional or pair
TM. These are marked as conjectures or open in the summary table and none of them is needed for
the experiments to be informative.
