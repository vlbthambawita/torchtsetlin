## Reading

- **The convolutional TM is the best TM here, and attention does not close the gap.** All
  pair-attention arms sit about where the patch TM without attention sits, a little above the
  flat TM and well below the convolutional TM. The likely reason for the convolutional TM's lead
  is not attention but **overlap**: it evaluates every clause on all 841 overlapping 4×4 patches
  (stride 1), while every token-based arm sees only the 64 non-overlapping patches. A pair
  clause can relate two patches, but it cannot see the 3/4 of the 4×4 windows that straddle
  token borders.
- **Attention itself contributes nothing measurable on CIFAR-10 in this configuration.** Global
  pair attention ties its no-attention ablation, and neighbour attention is less than one point
  above it. In the synthetic tasks the pair clause won exactly where the label is a relation
  between two tokens at content-dependent positions (co-occurrence T5, order T1). With these
  settings, CIFAR-10 class evidence does not seem to come from such pairwise relations of
  4×4 patches. Global pairs are also 64× more columns to search per clause (the dilution of
  `THEORY.md` C2): s = 2 won the search for every arm, the opposite of what the synthetic
  attention tasks needed.
- **The explicit Q/K/V arm is the weakest.** Two causes, both measured. (i) The graph TM uses
  coalesced clauses with per-class weights; its depth-1 control already loses to the class-wise
  arms, so part of the loss is the output layer, not attention. (ii) Message passing on top of it
  loses again, and D-H2a shows why: the OR inbox is almost always full, so the received
  "values" carry almost no information about the senders (`THEORY_HTM.md` H2.1–H2.2). The
  per-class table shows the effect: the Q/K/V model collapses on several classes. This is the
  failure that the HTM plan's sparse-value arm (H2, E-H2b) targets.
- **Cost.** Global pair attention is about 40× slower per run than the convolutional TM for no
  gain; neighbour attention costs little and gives at most a small gain over no attention.

## Limits

- One clause budget (2,000), one T, one Booleanization, 30 epochs; the convtm programme reached
  higher accuracy with far larger CTMs, so the absolute numbers are those of a small budget.
- s searched with one seed; three final seeds per arm, one for Q/K/V (D17).
- Tokens are non-overlapping 4×4 patches. Overlapping tokens (stride 2) would test the overlap
  explanation directly, at 4× the tokens and 16× the pairs.
