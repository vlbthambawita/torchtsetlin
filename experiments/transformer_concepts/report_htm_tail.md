## Reading of the results

Numbers are in the tables above; this section only says what they mean. Claim status is also
written back into `THEORY_HTM.md` §S.

### H3: what HTM's comparator buys

- **The pair TM's failure on T3 is partly a settings problem (P-H3a fails).** At M = 300 and very
  high s the pair TM learns T3 well above chance. H3.2's witness-share argument still explains
  why ordinary s fails (the share is tiny), but "no setting works" was too strong: with enough
  clauses and a very low forgetting rate the rare witnesses accumulate. See D19.
- **The comparator is the mechanism.** T3-comp is perfect at both lengths, while T3-seen (the
  same register without the predicted-vs-actual bit) stays near chance, as H3.5 predicts. The
  contrast is the point of E-H3b, and it holds. T2-mem is near perfect at L = 33, with the
  shuffled control at chance.
- **Caveat on the T3-comp control.** The shuffled comparator is not at chance, because the
  presence bits in the T3-comp features leak part of the label (D19). The unshuffled arm's
  perfect score does not depend on that leak, but the control is weaker than planned.
- **Local targets learn the register (H3.7 supported).** With teacher forcing on the seen-set
  target, the learned state is almost bit-exact in free running and T3 is solved at L = 16 above
  the 95 % bar. T2's memory state is learned just as well.
- **The next-token targets learn nothing useful, read as a decision or as a score.** H3.8
  predicted this for the argmax read-out. The "score" arm, HTM's own predictive-cell objective,
  was expected to land in between, and it does not: it is also at chance. With the base rate
  below 1/(V − L), a vote-sum threshold has nothing to separate. HTM's sequence memory relies on
  sparse, many-cell predictions that a class-wise clause bank does not reproduce.
- **Error growth (H3.9).** Free-running error rises along the sequence and is worst at L = 32,
  which is where the readout drops. Within a sequence the growth is convex rather than linear,
  because later steps carry larger sets. Bounded by a constant per-step rate, the conjecture is
  not contradicted, but the test is not sharp.

### H1: tolerant clauses

- **The from-scratch TM matches the library at δ = 0 (gate G1 passes)** on all three tasks.
- **Tolerance helps only on long, noisy patterns.** On the noisy-prototype task the gain grows
  with δ once noise is high, as H1.3's recall formula predicts. It never closes the gap to the
  Bayes rate, and at low noise the task is at the ceiling for every δ.
- **On short patterns tolerance is harmful, even without noise.** On T7 every δ > 0 is far below
  δ = 0. A clause with k ≤ δ literals is constant (H1.1), so the useful one- and two-literal
  clauses of a presence task cannot exist. The prediction "δ ∈ {0, 1, 2} tie at η = 0" fails on
  T7.
- **On CIFAR-10 tolerance hurts monotonically in δ.** Clauses grow longer with δ (see
  `included_per_clause` in `results/htm/h1c/`): the learner adds about δ extra literals to make up
  for the slack, and generalisation gets worse. This δ = 0 baseline has no literal budget (D25), so
  it sits below the library's flat number in REPORT_CIFAR10.md.
- **Net.** H1.5 is refuted as a general statement. It holds only in the regime H1.3 describes:
  long patterns with i.i.d. bit noise of at least about 20 %. Natural images are not in that
  regime.

### H2: sparse value codes

- **The Q/K/V inbox is saturated (P-H2a holds).** Most layer-1 inbox bits are on, as H2.2
  predicts, yet inbox literals make up most of the layer-2 clauses' included literals. The learner
  spends its literals on bits that carry almost nothing.
- **Fixed sparse SP codes fix it (P-H2b (i) holds).** SP-value attention beats `qkv` and `qkv_d1`
  by a wide margin, and it is the best token model in the programme.
- **It still does not beat the convolutional TM, and its gain over `pair_self`/`pair_local` is
  small.** Attention over sparse value codes is a working TM mechanism, but at this budget it is
  not a better image model than convolution.
- **The density curve is an inverted U, as predicted.** The peak sits about a factor 4 below the
  effective p*, just outside the predicted factor 3, so P-H2b (ii) narrowly fails.
- **What the miss suggests.** The mean-entropy p* treats the 16 edge types as equally useful,
  but the near types (1–3 senders) carry most of the signal and favour sparser codes. The miss
  points the same way.

## Failed or partly failed predictions

| prediction | outcome |
|---|---|
| P-H3a (pair TM at chance on T3 at every high-s cell) | fails at M = 300 (D19) |
| E-H3c score arm "in between" | fails: at chance like the decision arm |
| E-H1b "δ ∈ {0,1,2} tie at η = 0" | holds on proto (ceiling), fails on T7 |
| E-H1b "δ = 4 worse than δ = 2 at η ≤ 0.05" | not testable on proto (ceiling); on T7 all δ > 0 are close |
| H1.5 (tolerance helps on noisy inputs) | only at high noise on long patterns; hurts on CIFAR-10 |
| P-H2b (ii) (peak within factor 3 of p*) | narrowly fails (about factor 4) |

## Limits

- One CIFAR-10 setting (D15 budget, 30 epochs). The SP stage-2 search was 10 epochs, and qkv
  had one seed (D17).
- The from-scratch tolerant TM has no literal budget, so E-H1c compares δ only against its own
  δ = 0 baseline.
- E-H3 tasks are synthetic, with V = L + 16. HTM's own benchmarks (streaming anomaly detection,
  sequence prediction) were not run.
- The registers in E-H3b are hand-built. E-H3c learns them, but only with a teacher-forced,
  hand-chosen target.

## Next steps (not run)

1. **SP values with a per-edge-type density.** Make the near types sparser, the far types
   denser, following H2.1 per slot rather than on the mean. This is the direct follow-up to the
   P-H2b (ii) miss.
2. **SP values with `ctm`-style local patches.** Combine convolutional features with the
   SP-value inbox to see whether attention adds to convolution rather than replacing it.
3. **Learned comparators.** Train the comparator bit with a local target instead of computing it,
   to close the loop from E-H3c to E-H3b.
4. **Tolerance only where it fits.** A per-clause δ that is allowed only above a minimum clause
   length, e.g. δ ≤ k/8. That avoids the constant-clause failure on short patterns.
