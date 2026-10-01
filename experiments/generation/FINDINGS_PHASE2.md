# Phase 2 — 12-lead ECG generation (Stage A / gate G2)

**Question.** Can a block codebook represent 12-lead ECG well enough that a Tsetlin machine
generating *token sequences* could produce usable waveforms? Reconstruction bounds everything
downstream, so `PLAN.md` §3.3 resolves it before any TM is trained.

**Data.** PTB-XL v1.0.1 at `/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1`.
21 837 records / 18 885 patients, 10 s at 500 Hz, 12 leads in mV. Official `strat_fold`:
1–8 train, 9 validation, 10 test, patient-disjoint. Stage A uses 4 000 / 500 / 500 records.
Metrics are computed in physical units (mV) on **held-out fold-10 records**.

Code `code/phase2/`, records in `results/phase2/`. `src/torchtsetlin` unmodified.

---

## Confirmed first: only 8 of the 12 leads are independent (VALIDITY.md P3)

Measured on real PTB-XL test data, the Einthoven/Goldberger identities hold to within
quantisation noise:

| identity | relative mean-abs residual |
|---|---|
| III = II − I | 0.0016 |
| aVR = −(I+II)/2 | 0.0041 |
| aVL = I − II/2 | 0.0049 |
| aVF = II − I/2 | 0.0044 |

Reconstructing all 12 leads from the 8 independent ones (I, II, V1–V6) costs **0.085 %**
round-trip error. Two consequences, both adopted: the codebook quantises **8 channels, not 12**
(a third less to represent, and cross-lead consistency becomes exact by construction), and the
identity residual becomes a free validity metric on generated output.

## The design in the idea document fails immediately, and not for a tuneable reason

The document's specification — non-overlapping blocks of B=50 (100 ms), a joint 12-lead
codebook, K in the hundreds — gives:

| config | SNR mean | min-lead | seam ratio | QRS amp err |
|---|---|---|---|---|
| B=50, K=256, k-means | 4.3 dB | 0.1 dB | **92.9** | 0.62 |
| + R-peak-aligned grid | 4.2 dB | 0.1 dB | 90.0 | 0.66 |
| + high-pass (baseline removal) | 4.8 dB | −0.1 dB | 65.6 | 0.66 |
| + product quantiser (4 groups, 32 bits) | 7.5 dB | 3.8 dB | 72.9 | 0.43 |

against a gate of ≥ 20 dB and seam ≤ 2. Two diagnoses, in order:

**1. Baseline wander dominates each block.** High-pass filtering at 0.5 Hz cut the seam ratio
from 93 to 66 — i.e. adjacent centroids were largely disagreeing about DC level rather than
morphology. Necessary, nowhere near sufficient. (Note this changes the modelling target to the
high-passed signal, which is standard ECG practice but should be stated, not assumed.)

**2. The real obstacle is dimensional, and it is decisive.** Principal-component analysis of the
blocks themselves:

| block | dimensions | PCA components for 90 % | for 99 % |
|---|---|---|---|
| B=10, 8 leads | 80 | 5 | **14** |
| B=25 | 200 | 8 | **26** |
| B=50 | 400 | 14 | **48** |
| B=100 | 800 | 25 | **90** |
| whole beats (resampled, 8×128) | 1 024 | 31 | **130** |

A flat vector quantiser has `log2(K)` bits — 8 bits at K=256, 10 at K=1024 — to locate a point on
a manifold with 14–130 significant directions. **No feasible K closes that gap**, and note that
R-peak-aligned *beats*, the obvious "more structured" tokenisation, are the worst case of all at
130 components. This refutes the flat-codebook architecture on information-theoretic grounds
rather than by failed tuning, which is exactly what Stage A exists to establish cheaply.

## Residual quantisation rescues fidelity — and yields the token the TM actually wants

Multi-stage (residual) VQ quantises the residual of the previous stage, spending
`M × log2(K)` bits and reducing error geometrically. It is the standard construction in neural
audio codecs, and its token is a *tuple* of M sub-symbols — which is precisely the
**compositional encoding** `VALIDITY.md` P2 argued the TM needs in order to generalise across
similar morphologies instead of memorising exact n-grams.

Depth sweep (B=25, K=1024, 8 leads, high-passed):

| stages M | bits/token | SNR mean | min-lead | seam | QRS err |
|---|---|---|---|---|---|
| 1 | 10 | 5.7 dB | 0.3 | 48.1 | 0.661 |
| 2 | 20 | 7.8 | 1.8 | 32.1 | 0.561 |
| 4 | 40 | 10.6 | 4.4 | 20.2 | 0.425 |
| 8 | 80 | 14.2 | 7.9 | 12.4 | 0.294 |
| 16 | 160 | 18.1 | 11.9 | 7.2 | 0.185 |
| 32 | 320 | 22.9 | 16.6 | 4.88 | 0.095 |

Roughly **+3.5 dB per doubling of depth**, with no sign of saturation.

Block length at fixed depth (M=16):

| B | tokens / 10 s | SNR mean | min-lead | seam | QRS err |
|---|---|---|---|---|---|
| **10** | 500 | **23.6 dB** | **17.4** | 4.98 | 0.098 |
| 25 | 200 | 18.1 | 11.9 | 7.18 | 0.185 |
| 50 | 100 | 15.0 | 9.0 | 9.86 | 0.226 |
| 100 | 50 | 12.9 | 6.8 | 12.99 | 0.255 |

Shorter blocks win on every fidelity metric, at the cost of a longer token sequence — the
rate-distortion trade the document anticipated, now quantified. The document's own B=50 sits in
the worst third of this range.

## The seam criterion: my gate threshold was wrong, not the architecture

The best configuration reached seam ratio 4.51 against a pre-registered gate of ≤ 2, which looked
like a clear failure. Before recording it as one I computed the metric's null — what **real**
held-out ECG scores on the same statistic:

| nominal B | seam ratio of REAL fold-10 ECG |
|---|---|
| 10 | 4.61 |
| 25 | 4.63 |
| 50 | 4.64 |

The metric divides a *mean* at block joins by a *median* within blocks, and |Δx| on ECG is heavy
tailed (mean/median = 4.60). **Its floor is ~4.6, not 1.0, so the ≤ 2 threshold was unachievable
by real ECG itself.** I set that threshold when writing `PLAN.md` without computing the null.
The metric now reports `seam_ratio_real` and `seam_excess = seam_hat / seam_real` alongside, so
it is self-normalising; 1.0 means block joins are statistically indistinguishable from real
signal. This is the third time in this programme that a pre-registered number needed checking
against its own control before it could be believed.

## Two fixes tested separately, so the credit is attributable

| config | SNR mean | min-lead | seam | QRS err |
|---|---|---|---|---|
| B=10, M=16 (previous best) | 23.6 | 17.4 | 4.98 | 0.098 |
| **B=10, M=32** | **30.8** | **24.5** | **4.51** | **0.040** |
| B=10, M=16 + overlap-add | 20.6 | 14.4 | 3.81 | 0.142 |
| B=10, M=32 + overlap-add | 25.7 | 19.4 | 3.97 | 0.076 |

**Residual depth is the fix; overlap-add is not.** Crossfading cost 5 dB (each block must now
cover 2B samples on the same bit budget) and bought only 0.5 on a statistic whose floor is 4.6.
So seams here are not a boundary-disagreement artefact that windowing can repair — they simply
scale with reconstruction error, and the way to remove them is to reconstruct better.

---

# Gate G2 — verdict: **PASSED**

Winning configuration: **8 independent leads, high-passed at 0.5 Hz, non-overlapping blocks of
B = 10 samples (20 ms), K = 1024, M = 32 residual stages.**

| G2 criterion | required | measured | verdict |
|---|---|---|---|
| per-lead held-out SNR | ≥ 20 dB | **24.5 dB** (min lead); 30.8 dB mean | passes |
| seam discontinuity | ≤ 2 *(threshold invalid)* | **4.51 vs 4.61 on real data = 0.98× real** | passes, against the corrected null |
| QRS amplitude error | — | 0.040 (4 %) | — |
| lead-identity residual | — | exact by construction (4 leads derived) | — |

## What passing costs, and the Stage B operating-point decision

Fidelity was bought with bits, and the bill lands on the token sequence the TM must generate.
Per 10-second record (8 leads, raw = 640 kbit):

| operating point | tokens / 10 s | sub-symbols/token | **symbol predictions** | min-lead SNR | compression |
|---|---|---|---|---|---|
| B=10, M=32 *(gate-passing)* | 500 | 32 | **16 000** | 24.5 dB | 4 : 1 |
| B=10, M=16 | 500 | 16 | 8 000 | 17.4 dB | 8 : 1 |
| B=25, M=32 | 200 | 32 | 6 400 | 16.6 dB | 10 : 1 |
| B=25, M=16 | 200 | 16 | 3 200 | 11.9 dB | 20 : 1 |
| B=25, M=8 | 200 | 8 | 1 600 | 7.9 dB | 40 : 1 |

Phase 1's MNIST arm made 784 binary decisions per sample. The gate-passing ECG point needs
**16 000 predictions from a 1024-way alphabet** — roughly 20× the sequence burden, and it
compresses the signal only 4 : 1, which is a frank admission that VQ on a raw waveform is an
inefficient code.

Two honest consequences:

1. **Interpretability erodes at the gate-passing point.** A clause over residual stage 27 of a
   1024-entry codebook is not a human-readable rule, and interpretability is the whole reason to
   use a TM here. The cheap operating points keep that property; the accurate one does not.
2. **This is a genuine trade, not a tuning choice**, so it should be made deliberately rather
   than silently. My recommendation is to run Stage B first at **B=25, M=8** (1 600 predictions,
   ~8 dB) as a *rhythm-and-morphology* demonstration — Phase 1 established that sample quality
   and likelihood diverge, and for ECG the clinically meaningful criteria are RR-interval
   statistics and QRS morphology, not waveform SNR — and to treat the 24.5 dB point as the
   fidelity ceiling that a later, better-resourced arm targets.

---

# Stage B — token models

Operating point chosen by the user from the trade-off table above: **B=25, K=1024, M=8**
(200 tokens per 10 s, 8 sub-symbols each, 1 600 symbol predictions per record).

## The ceiling, and the risk attached to that choice

`ecg-roundtrip` — encode real held-out ECG through the codec and decode it. Nothing generated can
beat this.

| metric | value |
|---|---|
| R-peak recall (±50 ms) | **0.971** (2 369 of 2 401) |
| RR distribution KS vs real | **0.029** (mean 0.824 s vs 0.809 s) |
| SNR mean / min-lead | 14.1 / 7.9 dB |
| QRS amplitude error | 0.289 |
| codebook usage per stage | 999–1020 of 1024 (no dead entries) |

The stated risk of the cheap operating point was that QRS morphology might not survive ~8 dB.
It resolves cleanly: **rhythm survives almost perfectly, amplitude does not.** The codec preserves
*when* beats occur and how they are spaced, and blurs *how tall* they are. That is the right
profile for a rhythm-and-morphology demonstration, and it fixes what Stage B can be judged on:
R-peak recall and RR statistics, not waveform SNR.

## Baselines, and a structural finding that reshapes the arm

| model | NLL nats/sub-symbol | bits | R-peak recall | RR KS |
|---|---|---|---|---|
| uniform | 6.931 | 10.00 | — | — |
| `ecg-marginal` | 6.440 | 9.29 | 0.104 | 0.293 |
| **`ecg-markov1`** | **6.304** | **9.10** | 0.096 | 0.334 |
| `ecg-markov2` | 6.590 | 9.51 | 0.096 | 0.281 |
| *`ecg-roundtrip` (ceiling)* | — | — | *0.971* | *0.029* |

Two things follow, and the second changes the design.

**1. Every baseline produces rhythmic garbage.** R-peak recall ~0.10 against a ceiling of 0.971.
So the floor is genuinely low and there is real room for a sequence model to show value —
sampling tokens without temporal structure does not yield ECG.

**2. All the predictability is in residual stage 0.** Per-stage NLL:

| model | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 |
|---|---|---|---|---|---|---|---|---|
| `ecg-marginal` | 6.678 | 6.593 | 6.499 | 6.402 | 6.353 | 6.337 | 6.273 | 6.385 |
| `ecg-markov1` | **4.429** | 6.390 | 6.527 | 6.556 | 6.562 | 6.601 | 6.577 | 6.791 |

Stage 0 is strongly predictable (4.43 vs 6.68 nats); stages 1–7 sit *at or above* their own
marginal, i.e. the residuals are close to noise by construction — which is exactly what a residual
quantiser is designed to produce. **Averaging NLL over all 8 stages therefore dilutes any real
result eightfold**, and the meaningful target for the TM is stage-0 NLL, where the number to beat
is **4.429**.

## The TM arm's first configuration collapsed, and the diagnosis was already on file

The first run carried Phase 1's winning `T=8000` across to this model. It failed exactly as G0
predicts when `T` is transplanted rather than swept:

| head | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| empty clauses | **41.5 %** | 0.1 % | **36.9 %** | 1.1 % | 0.0 % | **13.2 %** | 0.0 % | **38.0 %** |
| NLL (nats) | 19.94 | 6.93 | 19.08 | 6.87 | 9.04 | 19.13 | 6.93 | 6.93 |

Overall 11.86 nats/sub-symbol — **worse than uniform**. Heads are either confidently wrong (~19
nats, the collapse signature) or pinned at exactly uniform (6.93). The empty-clause gate of G0
Finding 4 flagged every failing head and none of the healthy ones, which is the fourth independent
confirmation of that diagnostic.

Also measured: the coalesced multi-label update costs `batch × K × clauses`, so at K=1024 it runs
at **~15 updates/s** versus Phase 1's ~500 — a 30× slowdown that makes exhaustive sweeps expensive
and is worth recording as a scaling property of the library's multi-label path.

Both facts point the same way, so the arm was narrowed to **stage 0 only** — where the
predictability is — with a proper `T` sweep. That is 8× cheaper and tests the question that
matters.

## The real cause: `negative_scale`, and it was my own G0 rule applied out of its domain

Two hypotheses were tested for the TM's loss, in order.

**Hypothesis 1 — the compositional encoding handicapped it (VALIDITY.md P2). REJECTED.**
Giving the TM exactly what `ecg-markov1` conditions on (the previous token's stage-0 id as a
1024-wide one-hot, with the waveform summary removed) scored **6.737** — no better than the
waveform encoding's 6.696. Adding both together scored 6.921. Identity information was not the
missing ingredient.

**Hypothesis 2 — capacity (G0 Finding 6). REJECTED as the primary cause.** Shrinking the alphabet
by clustering the stage-0 centroids into N coarse symbols, and running the TM against a matched
count table on the *same* N-way problem, showed the TM losing to even the **marginal** at N=16 —
16 contexts, 16 outputs, 2000 clauses, zero empty clauses. That is wildly over-provisioned, so
capacity was not what was binding.

**The actual cause was `negative_scale = 1.0`.** With a one-hot target there are `K−1` negative
outputs per positive, so at `negative_scale = 1.0` the Type II feedback on negatives swamps the
positive signal. At N=32 (count table 1.270, marginal 1.762):

| `negative_scale` | TM NLL (T=2000) | gap to count table |
|---|---|---|
| 1.0 | 2.718 | +1.448 |
| 0.3 | 1.881 | +0.611 |
| 0.1 | 1.641 | +0.371 |
| **0.03 (≈ 1/N)** | **1.585** | **+0.315** |
| 0.01 | 1.612 | +0.342 |

The optimum sits at **≈ 1/N**, which is exactly the positive:negative ratio, and the rule transfers:

| N | clauses | `ns` | TM | count table | marginal | gap |
|---|---|---|---|---|---|---|
| 64 | 2 000 | 1/64 | 2.035 | 1.554 | 2.269 | +0.480 |
| 64 | 8 000 | 1/64 | 1.897 | 1.554 | 2.269 | +0.343 |
| 256 | 2 000 | 1/256 | 3.152 | 2.620 | 3.927 | +0.532 |
| 256 | 8 000 | 1/256 | **2.959** | 2.620 | 3.927 | **+0.339** |
| 256 | 2 000 | 1.0 | 4.956 | 2.620 | 3.927 | +2.336 |
| 256 | 8 000 | 1.0 | 4.678 | 2.620 | 3.927 | +2.058 |

**`negative_scale` is worth about six times more than clause budget here.** At N=256, correcting it
buys −1.80 nats; quadrupling the clauses at the wrong setting buys −0.28.

On the full 1024-symbol alphabet the correction takes stage-0 NLL from **6.696 → 6.039**, which
now clearly beats the marginal (6.678), though it remains above the count table (4.429) at a
clause budget the N=256 results say is under-provisioned. Both full-alphabet runs were healthy
(0 empty clauses) and `T` was re-checked at the corrected `negative_scale`: T=2000 gives 6.039,
T=8000 gives 6.462, so T=2000 stands.

### This supersedes G0 Finding 8

G0 Finding 8 concluded "keep `negative_scale = 1.0`", because that parameter skews the Type II rate
and would bias `p̂` away from `(v+T)/(2T)`. That is correct for *calibration fidelity* on the
balanced synthetic task it was measured on, and wrong for one-hot sequence targets, where the
class imbalance is `K−1 : 1` by construction. A rule derived on a balanced task was carried to an
extremely imbalanced one without re-deriving it — the same category of error as transplanting `T`
between models, which this programme has now made three times. `FINDINGS_G0.md` carries a pointer
to this correction.

---

# Stage B — verdict

| model | stage-0 NLL |
|---|---|
| uniform | 6.931 |
| `ecg-marginal` | 6.678 |
| **TM (corrected `ns`, 2 000 clauses)** | **6.039** |
| **`ecg-markov1`** | **4.429** |

**H4 is not met: the TM does not beat the count-table baseline on next-token NLL.** It is a
genuine model — well clear of the marginal at every alphabet size tested, with the gap to the
count table down to ~0.34 nats at N=256 — but on a first-order Markov problem a count table is
close to the optimal estimator, and the TM approaches without overtaking it.

**Where the TM's opportunity actually lies, and the next experiment.** The count table's advantage
is memorisation of exact bigrams, and it breaks as soon as contexts get sparse: at the full
alphabet `ecg-markov2` (two-token context) scores **6.590, worse than `ecg-markov1`'s 6.304**,
because 1024² contexts cannot be estimated from 800 000 observations. A TM generalises across
contexts instead of memorising them, so the regime where it should win is **longer context**, not
the bigram regime where it was tested. That is the decisive experiment: sweep context length L for
both models and find the crossover, if there is one.

## Status

Stage A complete, **G2 passed**. Stage B has its ceiling, floor, baselines and a working — but not
yet winning — TM arm. Not done: the context-length crossover above, generation and rhythm
evaluation from the TM arm, seed bands, and conditioning on diagnostic labels.
