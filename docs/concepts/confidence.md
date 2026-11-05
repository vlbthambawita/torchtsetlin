# Confidence and composites

A Tsetlin machine's vote sum is more than a decision: it says how strongly the clauses agree.
This page follows Chapter 7 of Granmo's *An Introduction to Tsetlin Machines*. It covers how
to read confidence from the votes, how to tell whether that confidence can be trusted, and
how several machines with different skills can be combined into a **composite** by trusting
whichever one is confident.

## Confidence is the size of the vote sum

Take the book's example: predicting whether a student passes a course from three facts, with
three clauses.

* `Lectures` votes for Pass.
* `Textbook` votes for Pass.
* `NOT Assignments` votes for Fail.

A student who followed the lectures and read the book gets two votes for Pass and none
against, so `v = +2`. Every clause that fired agrees, and the machine is **confident**. A
student who only read the book but skipped the assignments gets one vote each way, so
`v = 0`. The clauses disagree, and the machine is **uncertain**. A large negative sum is
confident too, in the other class.

| | book definition | library |
|---|---|---|
| one binary machine | `abs(v)` | — |
| multi-class machine | `max_k v_k`, the winning class's vote sum | `model.confidence(x)` = `(clip(max_k v_k, -T, T) + T) / 2T` |

The library rescales the winning vote sum to `[0, 1]` so that machines with different `T`
can be compared. Below `T` the order is the book's. Every prediction whose winning vote
reaches `T` gets confidence 1, so to rank the most confident predictions use the raw votes
from `model(x)`, which are not clipped.

## Trustworthy confidence

High confidence is only useful if it comes with high accuracy. The book's 2 × 2 table:

| | low accuracy | high accuracy |
|---|---|---|
| **high confidence** | overconfident | justifiably confident |
| **low confidence** | justifiably uncertain | underconfident |

A model's confidence is **trustworthy** when accuracy rises with it. To check, sort the test
predictions by confidence and measure the accuracy of those above each threshold. On the
six students of the book's example:

| confidence at least | +0 | +1 | +2 |
|---|---|---|---|
| correct | 5 of 6 | 5 of 5 | 2 of 2 |
| accuracy | 83 % | 100 % | 100 % |

Acting only on confident predictions removes the one mistake. The book's CIFAR-100 machine
shows the same shape at scale: accuracy climbs from about 34 % for the least confident
images to 100 % for the most confident.

```python
votes = model(x_test)                                     # raw vote sums, eval() mode
curve = tt.metrics.trustworthiness_curve(votes, y_test)   # accuracy and coverage per level
tt.viz.plot_trustworthiness(curve)
```

`trustworthiness_curve` uses the winning raw vote sum as confidence and places its
thresholds at quantiles of it. Read the `coverage` entry next to `accuracy`: it is the
fraction of examples a threshold keeps.

## Composites: let the confident machine decide

Different machines can have different skills. Two hospitals may train on different patient
populations. One machine may read blood samples and another the images. On CIFAR-10, one may
see adaptive-threshold edges, another colour thermometers, and another histograms of
gradients. A **Tsetlin machine composite** combines them without retraining:

1. Each member runs on its own input as usual, but outputs its **vote sums** instead of a
   class. Any kind of Tsetlin machine can take part.
2. **Normalise.** Members differ in clause count, `T` and data, so their raw vote sums are on
   different scales. Each member computes its own factor once, the spread of the vote sums it
   produces on a reference set: `alpha = max(v) - min(v)`. It divides its votes by that factor.
3. **Add** the normalised vote sums of all members and take the arg-max, as for a single
   machine.

A confident member has a large normalised sum and dominates. An uncertain member sums to
about zero and barely affects the result. In the book's breast-cancer example, the
Recurrence machine votes `+2` (normalised `+1`) and the Non-recurrence machine does not fire
(`0`), so the composite predicts Recurrence.

!!! warning "Fix the factor once per member"
    `alpha` belongs to the member, not to the batch. Computing `max - min` over each batch you
    happen to predict makes one example's prediction depend on the other examples in its
    batch, and changes results with the batch size. Fit the factors once on a reference set,
    such as a validation split, and reuse them. The TM Composites paper uses the test set's
    10 000 images, as does our reproduction.

The combination is plug-and-play: no member is retrained and members can be added later.
In this repository's reproduction of the CIFAR-10 composite (`experiments/multi_convtm/`,
three seeds on library version 0.3.2), four specialists reached 68.6 % against 56.8 % for the
best single specialist. That gain of about 12 points matches the paper's. The absolute numbers are lower than the paper's,
which the report attributes mainly to the 0.3.x hard clause budget; see
`experiments/multi_convtm/REPORT.md`.

How to build one in code is in [Evaluation and metrics](../guides/evaluation.md#composites-of-several-machines).

## References

* O.-C. Granmo, *An Introduction to Tsetlin Machines*, Chapter 7, tsetlinmachine.org.
* O.-C. Granmo, *TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin
  Machines*, arXiv:2309.04801 (2023).
