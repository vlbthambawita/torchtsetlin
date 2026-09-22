# ROUND-1 digest — 2026-09-20

**Question**: which existing Tsetlin-machine convolution approaches do we re-implement in P3, in
what order, at what cost — and was the programme's target ladder set correctly?

## What happened

- The team checked the number PLAN.md's targets were built on (60.7%) against the primary sources
  and found it does not describe a CIFAR-10 result.
- Both specialists (theorist, DL expert) each pre-registered a falsifiable prediction on the
  round's two hardest questions (does a counting pool help; is the TM capacity-limited or
  mechanism-limited) and then measured them. Both predictions failed by their own stated criteria,
  and both authors withdrew them in writing.
- The research engineer re-checked a rule in the team's own rulebook (`CHARTER.md`) that said a
  clause-size control "does not bind," measured it directly, and found the rule was wrong for
  every budget a published recipe actually uses — and that fixing it is the single biggest gain
  found in the programme so far.
- The team fixed a ~33 GPU-hour first-phase (P3) shortlist out of a ~120 GPU-hour budget, and cut
  the single most expensive planned experiment (a 31.6 GPU-hour reproduction) because the paper it
  would have reproduced turns out to report no CIFAR-10 number at all.
- The orchestrator corrected the rulebook and retracted an earlier wrong claim that a compute
  queue had crashed and lost work — it had not.

## What we now know that we did not know before

- **60.7% is not a CIFAR-10 result.** It traces to an MSc thesis on a modified CIFAR-10, cited at
  third hand. The real published bar for a single TM model is **75.4%** — the target moved by
  **14.7 points** overnight, and the widely-repeated 82.8% is a **22-model ensemble**, not a
  single model.
- **A clause-size limit (cap each learned rule at 32 conditions) is worth +7.50 percentage points**
  on our own baseline (44.58% vs 37.08%, averaged over 3 runs each), while the model looks at
  **4.8 times fewer conditions per image** and runs only 1.7% slower. This is the largest effect
  measured in the programme so far — bigger than any accuracy mechanism reported in the
  literature we reviewed — and it was sitting undetected in every run since day one because the
  team's own rulebook said (wrongly) that this control doesn't work at these settings.
- **Two runs must differ by at least 1.00 percentage point** before we treat the difference as
  real, given how noisy this training method is run-to-run. This number now gates every comparison
  in the programme.
- **The capacity question is a slope, not a ceiling.** Accuracy rises by about **+2.18 percentage
  points every time the number of learned rules doubles**, and that rate is falling. Reaching CNN-
  level accuracy (~94%) by adding more rules alone would need roughly **100 million rules and an
  estimated 77,000 GPU-hours per run** — the literature's own numbers say this route is open in
  principle and closed in practice.
- **A 22-model ensemble of small models (2,000 rules each) beats one big model (64,000 rules) by
  4.1 points, using 31% fewer total rules** (79.5% vs 75.4%). Nobody has yet tested whether that
  gain comes from the models being *different from each other* or just from there being *more of
  them* — a cheap (3 GPU-hour) experiment now exists to find out.
- Reproducing the field's best published result faithfully (the 82.8% ensemble) would cost an
  estimated **400-600 GPU-hours for a single run** — 2-3x the entire programme's compute budget.
  It is now a documented constraint rather than an implicit assumption.

## What we decided, and what it rests on

The team's decision record (`DR-001`) re-sets the programme's targets: the primary goal becomes
beating **75.4%** (a real, published, single-model number), the 82.8% ensemble is kept only as a
reference point, and a clause-size limit is now applied by default to every experiment. A
~33 GPU-hour list of experiments is locked in for the next phase.

**This rests on a mix of solid and provisional evidence, and the user is being asked to confirm
it.** That 60.7%/82.8% are the wrong targets, and that 75.4% is the right one, is settled — it
comes directly from published tables, independently checked by two people. The *new pass/fail bar*
for future ideas ("an idea must be worth more than doubling the rule count twice, roughly +4
points, or it's cheaper to just add more rules") is a reasonable extrapolation from someone else's
published curve, applied here as a working policy for the first time — it has **not yet been
tested on our own system**. That test is the first thing queued next.

## What is still open, or could still sink this

- The team's own target number for the cheapest reproduction check (T0) is not yet fixed — it is
  waiting on exact settings from the literature review that one team member still owes.
- Whether a "counting" readout mechanism helps at all is unresolved; both original predictions
  about it failed, and the team agreed it can only be tested paired with a second mechanism
  (density control), not on its own. That paired test has not run yet.
- Whether augmentation (mirroring/cropping training images) helps or hurts genuinely depends on
  how many rules the model has — it helps at small scale and hurts at large scale in the published
  data — so a screening shortcut the team planned to use would give a confidently wrong answer
  here and has been specifically disabled for this one mechanism.
- The programme's own capacity curve (the one that will confirm or break the new pass/fail bar
  above) has not been measured yet on this codebase.

## What runs next, and roughly how long

- Fill in the missing literature settings (no compute; blocks everything else) — needed within
  the next working session.
- Run the ~33 GPU-hour first-phase shortlist: capacity ladder at two encodings, a clause-budget
  reproduction check, a same-vs-different-encoding ensemble control, an augmentation check at two
  model sizes, and a paired counting/density screen. Estimated at roughly 1.5-2 days of GPU time
  across the two available cards, run cheapest-first.
- Report back once the capacity ladder lands — that single result either confirms the new
  pass/fail bar for future ideas or forces the team to rethink it.
