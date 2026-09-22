# Programme log — convolution for Tsetlin machines

Maintained by the note taker. One dated entry per round, 10-20 lines, newest at the bottom. This is
the running history: it must read as a coherent story from top to bottom.

---

## 2026-09-20 — ROUND-1: the target ladder was wrong, and the P3 shortlist is fixed

The programme's first research round. Convened to decide which existing Tsetlin-machine
convolution approaches to re-implement in P3, in what order, at what cost, and to check whether
PLAN.md's target ladder (T0 "55-61%", T2 "beat 60.7%") was set correctly. It was not: 60.7% traces
to an MSc thesis on a *modified* CIFAR-10, cited at third hand, not to a CIFAR-10 result at all.
The real single-model bar is 75.4% (64,000 clauses, Optimized Toolbox 2024) — the target moved by
14.7 points — and 82.8% is a 22-specialist composite, not a single model, priced at 400-600 GPU-h
per seed and not reproducible on this machine at any budget.

All three specialists withdrew a pre-registered claim of their own against measurement during the
round (`rounds/ROUND-1/minutes.md` records each in full): the DL expert's capacity-saturation
prediction and its "9.7x" information-gain figure (recomputed as 1.71x from the measured data);
the theorist's "almost never an OR over more than one term" and their reading of a still-rising
capacity curve as a route to accuracy; and the research engineer's claims about the clause-size
budget, reversed after measuring it directly worth **+7.50 pp** (44.58 vs 37.08 pp, 3 seeds, 4.8x
fewer literals per image) — the largest effect measured in the programme so far, found by checking
a `CHARTER.md` rule that had been steering the team away from it. The orchestrator corrected that
rule and retracted an earlier wrong claim that a compute queue had crashed.

`DR-001` re-baselined the ladder (T2 = 75.4%, T3 = 82.8% as reference only, T0 still pending exact
figures), made the clause-size budget a default on every arm, cut the 80,000-clause `ctm-vanilla`
reproduction (no CIFAR-10 number exists to score it against), fixed a ~33 GPU-h Tier 0-1 P3
shortlist of the ~120 GPU-h envelope, and replaced "does capacity saturate?" with a slope
statistic (+2.18 pp per doubling of clauses, decaying) and a promotion bar of two doublings
(~+4 pp) for any P5 mechanism candidate. The ladder re-baseline needs the user's confirmation; the
promotion bar itself is not yet tested on this programme's own harness.
