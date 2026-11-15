## What the evidence does to THEORY.md

| claim | before tonight | after |
|---|---|---|
| A1, A2 (interval costs: thermometer ≤ 2 literals, one-hot $n - \lvert S\rvert$) | proved, checked $n \le 10$ | proved, checked $n \le 12$ (X1) |
| A3 (binary / Gray) | values only, no lower bound | **new Proposition A3b**: binary $[1, n-2]$ needs exactly $\log_2 n$ clauses (proof via strongly connected digraphs; X1b) |
| A5 (T1 at chance for token sets) | proved | exact Bayes accuracy 0.5 confirmed; bigram sets beat chance (X2) |
| A6 (conv + thermometer PE solves T1 with $2(L-1)$ voting clauses) | proved | confirmed through the library's own position bits (X3) |
| A7 (no absolute code gives cheap relative position) | proved, checked $n \le 6$ | separation computed for all four codes up to $n = 32$ (X4). **Empirically**: relative codes beat absolute ones on T1 (L = 32) and T4, tie on T5/T6, and lose on T2 (see Part III) |
| B1 (pair clause = $\exists(i,j)\,Q \land K \land R \land A$) | proved | the library's pair TM matches a symbolic evaluator on {{x5_clauses}} random clauses per code, CPU and CUDA (X5) |
| B3 scope (popcount ≠ vote) | remark | explicit counterexample (X6) |
| B4.1 (self-only pair TM = conv TM on END-padded sequence) | proved | identical clause outputs both directions, $w = 1..3$ (X7) |
| B5, B6.1, B7.1, B8 (hand pair constructions) | proved (B5, B7.1 tested) | accuracy 1.0 in the library on exhaustive and random inputs, CPU and CUDA (X8) |
| B6.2 (flat needle ≥ $L - d$) | proved | **tight**: exact minimum $= L - d$ on 5 instances (X10) |
| B6.3 (conv $k \le d$ cannot solve T4) | proof corrected in the review | the separating pair gives identical library outputs (X9) |
| B7.2 (flat recall ≥ $n$) | proved | exact minima are larger: **new Conjecture B7.3**, $n \cdot \min(\lvert\Sigma_K\rvert, 2^n)$, fits {{b73_fit}} instances (X10) |
| D.1 (flat cost of T1) | construction with $L - 1$ clauses | **new Proposition D1**: exactly $\lceil (L-1)/2 \rceil$ (fooling-set lower bound + construction; X10, X12) |
| B9 (polarity exclusion) | proved | truth tables match in the library (X11) |
| C1 (drawn pair uniform; conv feedback unchanged) | proved by construction | measured: the Type Ia drift is linear in the share of matching pairs with the predicted slope (D1) |
| C2 drift formula | $p_\text{mem} = (s-1)/s$ (**wrong for the library default**) | corrected to $p_\text{mem} = 1$ (`boost_true_positive=True`); {{d1_pass}} cells within 3 SE (D1) |
| C2 dilution ($s$ must grow with the number of pairs) | prediction | **direction confirmed** in $L$ and in the distractor rate, with a no-dilution control; **magnitude off** by up to ≈ {{d2_ratio}}× at $L = 32$; the flat TM shows the same effect (D2) |

## Answers to the research questions

**Does the theory hold?** Every formal statement that was marked proved survived an exact or
library-level check ({{exact_pass}}), and three new exact results came out of the checks (A3b, D1, and the
tightness of B6.2). One learning-theory formula was wrong because it assumed the textbook feedback
probability rather than the library's default; the corrected formula fits every measurement. The
dilution heuristic predicts the *direction* of the specificity requirement correctly and its
*size* only roughly.

**Does attention help a Tsetlin machine?** Where the label is a relation between two tokens at
content-dependent positions, yes, decisively: co-occurrence (T5) is solved only by pair clauses,
and order (T1) is solved more accurately by pair clauses than by every baseline at $L = 32$. The
gain comes from the cross-token pairs (A2-self fails), and for order/offset tasks also from the
relative code (A2-norel and A2-shuf fail; A2-abs is worse on T1 and fails on T4), exactly as
Theorem A7 predicts. Where a fixed window or fixed positions suffice (T4 needle, T6), the flat TM
and the right conv kernel tie with attention, also as predicted (B6.4). The controls (T6, T7) show
no cost for attention.

**Where it fails, and why.** Expressivity is not the bottleneck on recall (T2) and repeat
detection (T3): exact pair-clause solutions exist and run in the library (X8). In the
preregistered finals no arm learned T2 well. The exploratory runs (validation only, after the
freeze) show that this was the **settings**: with $M = 300$ clauses per class and $s = 40$ the
pair TM reaches {{t2_a2_best}} % and A2-norel {{t2_norel_best}} % on T2 at $L = 17$, while the flat TM
stays at {{t2_flat_best}} % under every setting tried. So recall is the clearest case for attention
in this programme, but it rests on exploratory runs and needs a preregistered confirmation. T3
stays at chance for every arm and setting tried ({{t3_best}} % at best). Each repeat-detection clause
$[q = a] \land [k_0 = a] \land [j - i > 0]$ fires on only a $1/V$ share of the positives, which by
the C2 argument is a very weak signal. The graph-TM arms (depth 2, chain and complete) were weak
everywhere except T7; they used one untuned configuration (hashed messages, shared clauses), so
this is not evidence against the graph TM in general.

**Positional encodings (H1).** The clause-cost ranking of A1–A3 (thermometer best) does *not*
predict learned accuracy: binary and Gray codes ($\lceil \log_2 L \rceil$ bits) beat thermometer
on T1 at $L = 32$ and on T4, and all codes tie on T6. Cheap clauses are not the same as
learnable clauses; fewer input bits seem to matter more under TM feedback. This is an empirical
finding without a theory here.

## Limitations

- Synthetic tasks only; no real dataset was run (PLAN P6 needs the user's choice).
- One search seed and a three-value $s$ grid; for T2 the selection at $L = 33$ (where nothing
  learned) made the chosen $s$ close to arbitrary for $L = 17$.
- Batch feedback only (no sequential-mode rows), weighted clauses only, $M = 100$ per class in the
  finals. The exploratory T2 runs show that $M$ matters a great deal for recall.
- The graph arms used one configuration.
- D2 used 8,000 examples and ≤ 15 epochs; $s^\star$ depends on that budget.
- Lower bounds are for positive-clause DNFs; nothing is proved for thresholds with negative clauses.

## Library gaps

LG-1 (Triton kernels on a non-current CUDA device) — see `LIBRARY_GAPS.md`. The library was not
modified.

## Deviations

D10–D14 in `DECISIONS.md` (GPU pinning, search grid, T5 reclassified and T7 added before the freeze,
the degenerate no-distractor D2 variant, extended exact checks).

## Next steps

1. Confirm the recall result with a preregistered run ($M = 300$, $s \in \{40, 80\}$, A2 / A2-norel /
   flat, 5 seeds, test set). Study repeat detection separately: the clauses exist, but the
   learner does not find them.
2. A theory for why fewer input bits beat cheaper clauses (H1), probably through the same
   drawn-literal dilution as C2 applied to position bits.
3. One real dataset (IMDB or MIT-BIH, PLAN §2) with A2 vs the flat and conv TMs.
4. Fix LG-1 in the library when allowed.
