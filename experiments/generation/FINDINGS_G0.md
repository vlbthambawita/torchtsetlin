# Gate G0 — results

**Question.** Both generation designs assume a Tsetlin machine's vote sum can be read as a
conditional probability. `VALIDITY.md` §1 derived, from the feedback rule, that it should:
`v* = T(2p − 1)`, so `p = (v + T)/(2T)`. G0 tests that on synthetic data where `P(y=1 | context)`
is **known exactly**, so calibration error is measured rather than estimated from bins.

**Setup.** 112 distinct Boolean contexts (F=10), each assigned a true probability from
{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}; labels sampled Bernoulli; 2 000 000 training examples at
batch 50 unless stated. This is deliberately hard — the contexts are arbitrary, so nothing
generalises from one to another and the machine must almost memorise them. `mae` is the mean
absolute error of the predicted probability against the known truth over all 112 contexts.
Calibrators are fitted on a held-out 400 000-example stream of *sampled labels*, never on `p_true`
— the protocol a real generator would have to use.

Code `code/g0/`, one JSON record per run in `results/g0/`, tables via
`python code/g0/summarize_g0.py`. No file under `src/torchtsetlin` was modified.

---

## Verdict: **G0 PASSES**, decisively.

| | gate | best measured |
|---|---|---|
| calibration error, analytic map `(v+T)/2T` | ≤ 0.05 | **0.0175 ± 0.0011** |
| calibration error, isotonic on validation | — | **0.0029 ± 0.0005** |
| fixed-point slope (`v/T` vs `2p−1`; theory says 1) | — | **0.963 ± 0.003** |
| r² | — | **0.997** |

Best configuration: **12 800 clauses/class, T = 400, s = 40, batch 50**, 3 seeds, zero empty clauses
in every run (`|v|max/T` = 0.90–0.93). The vote sum tracks the
conditional probability along the predicted line, and the residual is a mild monotone compression
that a fitted calibrator removes. **Both generation ideas may proceed to Phase 1.**

The cost of that verdict is that five hyperparameter choices turn out to be load-bearing, four of
them in ways neither idea document anticipated. Each has a sharp failure mode that produces a
*constant* `p̂` — a generator that emits the same image or the same waveform every time — while
still classifying acceptably. They are listed below in order of how much damage they do.

---

## Finding 1 — `T` must match the achievable vote amplitude. This dominates everything else.

`VALIDITY.md` called `T` the "probability resolution knob". The measured control variable is the
**ratio `|v|max / T`**, where `|v|max` is the largest vote the clause pool can actually produce:

* `T` at the largest value the pool can still reach (`|v|max/T = 1`) → calibrated.
* `T` above what the pool can reach → votes compress toward 0 and every `p̂` is dragged toward 0.5.
  At C=200, `T=200` gives slope **0.075** and mae 0.29 — nearly uninformative, though trained
  correctly and with no other fault.
* `T` far below the pool's capacity → votes clip, resolution is lost, and clauses die (C=3200 at
  `T=5`: 400 empty clauses, mae 0.44).

Since `|v|max` grows with the clause budget, **the clause budget sets the attainable probability
resolution and `T` must then be tuned to it.** Measured optimum: `T* ≈ C/32` (C = clauses per
class) across three decades of budget.

### p01 — calibration vs clause budget and T (2 M examples, s=10, seed 0)

| clauses/class | T | \|v\|max | \|v\|max/T | analytic | isotonic | platt | slope | r² | empty |
|---|---|---|---|---|---|---|---|---|---|
| 200 | 5 | 5 | 1.00 | 0.2830 | 0.1915 | 0.2163 | 0.745 | 0.424 | 16 |
| 200 | **10** | 10 | 1.00 | 0.0996 | 0.0824 | 0.0846 | 0.815 | 0.871 | 0 |
| 200 | 25 | 22 | 0.88 | 0.1587 | 0.0873 | 0.1021 | 0.545 | 0.818 | 0 |
| 200 | 50 | 31 | 0.62 | 0.2277 | 0.1264 | 0.1428 | 0.304 | 0.677 | 0 |
| 200 | 100 | 40 | 0.40 | 0.2713 | 0.1429 | 0.1646 | 0.152 | 0.586 | 0 |
| 200 | 200 | 51 | 0.26 | 0.2931 | 0.1606 | 0.1923 | 0.075 | 0.467 | 0 |
| 800 | 10 | 10 | 1.00 | 0.1183 | 0.0797 | 0.0888 | 0.938 | 0.858 | 0 |
| 800 | **25** | 25 | 1.00 | 0.0489 | 0.0290 | 0.0460 | 0.954 | 0.969 | 0 |
| 800 | 50 | 46 | 0.92 | 0.0875 | 0.0445 | 0.0582 | 0.784 | 0.943 | 0 |
| 800 | 100 | 82 | 0.82 | 0.1593 | 0.0785 | 0.0952 | 0.532 | 0.836 | 0 |
| 800 | 200 | 134 | 0.67 | 0.2278 | 0.1168 | 0.1404 | 0.300 | 0.674 | 0 |
| 3200 | 25 | 25 | 1.00 | 0.0863 | 0.0637 | 0.0738 | 0.971 | 0.911 | 0 |
| 3200 | 50 | 50 | 1.00 | 0.0527 | 0.0260 | 0.0437 | 0.963 | 0.967 | 0 |
| 3200 | **100** | 100 | 1.00 | 0.0449 | 0.0181 | 0.0415 | 0.929 | 0.977 | 0 |
| 3200 | 200 | 187 | 0.93 | 0.0811 | 0.0366 | 0.0520 | 0.786 | 0.951 | 0 |
| 3200 | 400 | 323 | 0.81 | 0.1563 | 0.0812 | 0.0985 | 0.547 | 0.826 | 0 |

(Rows at `T` far below the budget's optimum are omitted; they all collapse — see Finding 4.)

### The scaling law (seed-confirmed)

The optimum is not a plateau: buying clauses buys probability resolution, and `T*` tracks the
budget on a clean line.

| clauses/class | T* | analytic | isotonic | slope | r² | seeds |
|---|---|---|---|---|---|---|
| 200 | 10 | 0.0996 | 0.0824 | 0.815 | 0.871 | 1 |
| 800 | 25 | 0.0489 | 0.0290 | 0.954 | 0.969 | 1 |
| 3 200 | 100 | 0.0449 | 0.0181 | 0.929 | 0.977 | 4 |
| 12 800 | 400 | 0.0372 | 0.0096 | 0.934 | 0.987 | 3 |
| 51 200 | 1 600 | 0.0335 | 0.0117 | 0.955 | 0.988 | 1 |

At every optimum `|v|max / T = 1.00` exactly.

**Rule for Phases 1–2:** choose the clause budget, then set `T ≈ C/32`, then *verify* `|v|max/T ≈ 1`
and `empty_clauses == 0` on a held-out batch. A wrong `T` costs an order of magnitude more
calibration error than anything else measured.

## Finding 2 — the analytic map is the right form; isotonic removes a residual compression

At every well-specified setting the fitted isotonic map roughly halves the analytic error, and
Platt lands between. The residual is a *monotone under-dispersion* (slope slightly below 1) —
exactly what a fixed point with unequal up/down step sizes predicts, and exactly what a monotone
calibrator is for.

Use `(v + T)/(2T)` as the reference and report the gap to the fitted map as a diagnostic: it is the
theory check. Both idea documents proposed fitting a calibrator; that is **necessary but not
sufficient** — isotonic cannot rescue a badly chosen `T` (at C=200, `T=200` it still leaves 0.16).

## Finding 3 — specificity `s` has a narrow usable band, and the ends are catastrophic

Measured at each budget's tuned `T` so Finding 1 cannot confound it (2 M examples, batch 50):

| s | C=800, T=25 | C=3200, T=100 | C=12800, T=400 |
|---|---|---|---|
| 2 | **collapse** (765 empty) | **collapse** (3084 empty) | — |
| 5 | **collapse** (707 empty) | **collapse** (2842 empty) | — |
| 10 | 0.0489 / iso 0.0290 | 0.0392 / iso 0.0111 | — |
| 20 | 0.0411 / iso 0.0200 | 0.0291 / iso 0.0041 | 0.0225 / iso 0.0038 |
| 40 | — | 0.0244 / iso 0.0067 | **0.0179 / iso 0.0031** |
| 80 | — | 0.1283 (slope 0.60) | 0.1284 (slope 0.60) |
| 160 | — | 0.1751 (slope 0.45) | 0.1752 (slope 0.45) |

Usable band **s ∈ [10, 40]**, optimum at **s = 40**. Below that clauses forget faster than they
memorise and die; above it they over-specialise and the vote range compresses (slope 0.60 → 0.45).
The default `s=10` is inside the band but not optimal — `s=40` at C=12 800 is the best result in
G0. This supersedes an earlier reading of these data taken at an untuned `T`, where `s` appeared not
to matter; it appeared not to matter only because every setting was already broken.

## Finding 4 — `empty_clauses > 0` detects every collapse, and it is free to check

Pooling every run across all sweeps, the empty-clause count separates working from collapsed models
with **no overlap**:

| sweep | setting | empty | mae | `p̂` mean |
|---|---|---|---|---|
| smoke | C=800, T=50, N=100 k | 213 / 1600 | 0.438 | 0.938 |
| p01 | C=3200, T=5 / T=10 | 400 / 356 | 0.440 / 0.290 | — |
| p01 | C=12800, T=50 | 687 | 0.409 | — |
| p01 | C=51200, T=200 | 6170 | 0.305 | — |
| p02 | batch 70–2000 | 1067–12761 | 0.31–0.50 | ~0.5 / pinned |
| p03a | 1 000 / 10 000 / 100 000 contexts | 84 / 112 / 168 | ~0.50 | 0.998 / 1.000 / 0.000 |
| p07 | s = 2 or 5 | 707–3084 | 0.29–0.47 | pinned |
| **every healthy run** | — | **0** | 0.018–0.096 | ≈ 0.49 |

The signature is always identical: `p̂` pins to 0.000 or 1.000 and the slope goes to zero — the
machine emits a **constant probability**. A generator built on such a model produces a constant
image or a constant waveform, and it can do so while still classifying acceptably. *Accuracy is not
a safety check for a generative TM.*

**Add `assert (model.include_count == 0).sum() == 0` as a per-epoch training gate.** It costs one
tensor reduction, needs no labels, and it flagged every failure in this gate.

### Differential diagnosis — same symptom, three distinct causes

The counter tells you *that* the model is dead, not *why*. G0 separated three causes, and getting
this wrong is easy (I misdiagnosed two of them mid-run before the controls reported):

| cause | control that identifies it | fix |
|---|---|---|
| **undertraining** | more data fixes it (100 k → 500 k took mae 0.44 → 0.096, empty 213 → 0) | train longer |
| **batch too large** | *not* fixed by matching update count (Finding 5) | batch ≤ 60 |
| **capacity exceeded** | *not* fixed by 16× more data (Finding 6) | more clauses, or structured context |

## Finding 5 — batched feedback collapses above a threshold set by `s`, and near it the failure is **stochastic**

Holding `update()` calls fixed at 20 000 (so batch size is not confounded with learning steps — this
control matters; the naive experiment that fixes *examples* instead confounds the two and I misread
it at first), at C=3200, T=100:

| batch | s=10 | s=20 | s=40 | s=80 |
|---|---|---|---|---|
| 40 | 0.0386 | 0.0249 | 0.0255 | — |
| 60 | 0.0545 | 0.0278 | 0.0251 | — |
| 80 | **0.500 (2153 empty)** | 0.0299 | 0.0239 | — |
| 120 | **0.322 (3087)** | **0.394 (322 empty)** | 0.0239 | — |
| 160 | **0.313 (3175)** | **0.500 (1969)** | 0.0244 | — |
| 240 | **0.313 (3190)** | **0.294 (2918)** | 0.0285 | 0.132 (slope 0.58) |
| 480 | — | — | 0.0413 (slope 0.88) | 0.134 (slope 0.58) |
| 960 | — | — | **0.261 (2198)** | 0.142 (slope 0.56) |

**`s` sets the tolerance.** The cliff moves from ~70 at `s=10`, to ~100 at `s=20`, to ~500 at
`s=40`. It is *not* a clean proportionality (ratios 7, 5, 12), so it cannot be turned into a formula
— it must be measured per configuration. `s=80` never collapses but is useless anyway (slope 0.58,
Finding 3).

**It is not `n_states` and not the clause budget.** Swept `n_states` over 32/64/128/256/512: the
cliff sits between 40 and 80 in every case, with near-identical damage (2152–2279 empty clauses at
batch 80). This **refutes** the natural mechanism hypothesis — that large accumulated Type I
increments overshoot the include boundary, which would predict a cliff proportional to `n_states`.
It is likewise identical at C=800, 3200 and 12800. Combined with the `s` dependence, the surviving
explanation is a gain instability in the forget channel (`dec = binomial(n_false + n_ib, 1/s)`,
unbounded and scaling with batch, where Type II is explicitly clamped to the boundary). That
explanation is **not confirmed**; see `LIBRARY_GAPS.md` LG-G01.

**Near the cliff the failure is stochastic across seeds.** The best recipe at batch 240 ran 3 seeds:
two gave mae 0.028 with zero empty clauses, and **one collapsed outright** (mae 0.492, slope 0.028,
1163 empty). A single-seed "it works at this batch size" result is therefore not evidence.

**Recommendation.** Use **batch ≤ 50**, where every configuration tested was stable. If throughput
demands more, `s=40` with batch ≤ 240 is available *only* with the per-epoch empty-clause assert of
Finding 4 and multiple seeds — treat it as an optimisation to be re-validated, not a default.

**Cost consequence for Phases 1 and 2.** At batch 50 the harness sustains ~570 `update()` calls/s
here, i.e. ~29 k examples/s. MNIST-AR at all 784 positions per image is 47 M examples/epoch →
~27 min/epoch. This is why `PLAN.md` §3.2 samples ~100 positions per image per epoch (6 M examples →
~3.5 min/epoch); that decision is now forced rather than merely prudent. Sequential mode is not an
escape: 500 s for 500 k examples (40× batch-50) for mae 0.081 versus batch-50's 0.096.

## Finding 6 — there is a real capacity limit, and more data does not cure it

`p03a`, with the number of distinct arbitrary contexts varied at fixed total examples (F=20,
C=800):

| distinct contexts | repetitions each | mae | slope | empty |
|---|---|---|---|---|
| 16 | 125 000 | 0.0587 | 0.969 | 0 |
| 112 | 17 857 | 0.0325 | 0.936 | 0 |
| 1 000 | 2 000 | 0.4989 | 0.000 | 84 |
| 10 000 | 200 | 0.4997 | 0.000 | 112 |
| 100 000 | 20 | 0.4999 | 0.000 | 168 |

The control: 1 000 contexts at the tuned `T`, trained 16× longer.

| examples | mae | empty | mean literals included |
|---|---|---|---|
| 2 M | 0.4996 | 257 | 11.8 |
| 8 M | 0.4996 | 219 | 11.9 |
| 32 M | 0.5005 | 178 | 12.8 |

**16× more data does not fix it.** Nor does 16× more clauses (1 000 contexts at C=12 800, T=400:
mae 0.4996, 4 073 empty). This is a representability limit: as the number of arbitrary contexts
grows, clauses become over-specific (mean included literals 4.3 → 12.8), match almost nothing, and
die.

**But this is the worst case by construction** — arbitrary context→probability maps with no shared
structure. Finding 7 measures the realistic case, and it behaves completely differently.

## Finding 7 — irrelevant context bits degrade calibration *gracefully* (H1)

`p03b`: `P(y=1)` depends on 4 signal bits; the context is padded with D irrelevant random bits.
Probed on every signal pattern × 64 random noise completions. C=800, T=50, 2 M examples, 3 seeds.
`within_group_spread` is the spurious variation in `p̂` caused purely by noise bits — ideally 0.

| irrelevant bits | mae | within-group spread | slope | r² |
|---|---|---|---|---|
| 0 | 0.0375 | 0.0000 | 0.956 | 0.993 |
| 4 | 0.0415 | 0.0286 | 0.946 | 0.983 |
| 8 | 0.0519 | 0.0325 | 0.912 | 0.978 |
| 16 | 0.0585 | 0.0405 | 0.905 | 0.967 |
| 32 | 0.1175 | 0.0461 | 0.719 | 0.856 |

With **8× more noise than signal** the machine still calibrates to mae 0.118 with slope 0.72. No
cliff. Taken with Finding 6: **what matters is whether the context→probability map is compressible
by clauses, not how wide the context is.** Structured maps tolerate large irrelevant padding;
unstructured maps fail once they exceed the pool's capacity, regardless of data.

Consequences for Phase 1: the window arm stays the primary, but the **canvas arm is not doomed** —
E3 predicted context breadth would be costly, and it is, by roughly 3× in mae, not
catastrophically. And `within_group_spread` is a cheap new diagnostic worth carrying forward: hold
the causal window fixed, vary the pixels outside it, and the spread in `p̂` measures directly how
much irrelevant context is leaking in.

## Finding 8 — P7 confirmed: coalesced + `multi_label=True`, at a far larger `T`

`VALIDITY.md` P7 argued the ECG token model must use `multi_label=True`, since multi-class updates
only the target and one random negative, leaving the other `K−2` outputs uncalibrated — survivable
for argmax, fatal for sampling. Measured on contexts with known sparse categorical distributions
over `K` tokens (`p04`, 1 M examples, `T` swept 25 → 6400 per variant):

| K | variant | best T | TV ↓ | KL ↓ | top1 ↑ | support mass ↑ |
|---|---|---|---|---|---|---|
| 64 | class-owned multi-class | flat | 0.935 | 2.90 | 0.59 | **0.065** |
| 64 | coalesced multi-class | 400 | 0.252 | 0.184 | 0.53 | 0.974 |
| 64 | **coalesced multi-label** | 3200 | **0.123** | 0.325 | **0.98** | 0.984 |
| 256 | class-owned multi-class | flat | 0.983 | 4.25 | 0.33 | **0.017** |
| 256 | coalesced multi-class | 400 | 0.267 | 0.218 | 0.28 | 0.956 |
| 256 | **coalesced multi-label** | 3200 | **0.110** | **0.184** | **1.00** | 0.982 |

**The class-owned machine is exactly as bad as P7 predicted**: at K=256 it places **1.7 %** of its
probability mass on the tokens that actually have support, at every `T`, while still producing a
usable argmax. Token accuracy tells you nothing about whether a TM can be sampled from.

**Multi-label needs ~8× the `T` that multi-class does, and this nearly caused a wrong call.** Swept
only to `T=400`, multi-class looks clearly better and I recorded exactly that conclusion mid-run.
It was wrong: multi-class peaks at `T=400` and then degrades (K=256: TV 0.267 → 0.520 by `T=3200`,
support mass 0.956 → 0.523), while multi-label improves monotonically to `T=3200` and overtakes
decisively, then falls off by `T=6400` (TV 0.426). The cause is Finding 1 — in multi-label mode
every output is updated on every example, so the per-output vote scale is much larger.

**The general warning:** any TM comparison run at a single `T`, or over a `T` range tuned for one
arm, is not a comparison. Both idea documents propose comparing variants; every such comparison must
sweep `T` per variant until each has peaked *and turned over*.

**Phase 2B:** `CoalescedTsetlinMachine(n_outputs=K, multi_label=True)`, `T` swept around 3200 for
K≈256 (rescaled with the clause budget per Finding 1).

> **Superseded in Phase 2 — `negative_scale = 1.0` is wrong for one-hot sequence targets.**
> This finding fixed `negative_scale` at 1.0 because that hyperparameter skews the Type II rate
> for negative outputs and would bias `p̂` away from `(v+T)/(2T)`. That reasoning holds for
> *calibration fidelity* on the balanced synthetic task measured here. It does **not** transfer to
> next-token prediction, where a one-hot target gives `K−1` negatives per positive and the negative
> feedback swamps the signal. Measured on ECG tokens at K=32: the gap to a count-table baseline
> falls from **+1.45 nats at `negative_scale=1.0` to +0.32 at `negative_scale≈1/K`**, and the model
> goes from losing to the marginal to clearly beating it. See `FINDINGS_PHASE2.md`.

## Finding 9 — sampling adds no error, and the train/eval ambiguity dissolves

* `p06`: drawing 10 000 Bernoulli samples per context from `p̂` reproduces `p̂` to **0.0034** while
  sitting 0.087 from `p_true` — the entire sampling error *is* the calibration error. The generative
  step itself is sound; everything rests on calibration.
* `p05`: at a healthy setting (`empty_clauses = 0`) the mean absolute train-vs-eval vote shift is
  **exactly 0.000** and both semantics give identical mae. **Correction to `VALIDITY.md` E4:** the
  hazard is not an intrinsic mode ambiguity to be resolved by choosing `eval()`; it is entirely
  mediated by empty clauses. Drive that count to zero (Finding 4) and the question disappears.

---

## Consolidated recipe for Phases 1 and 2

1. Choose the clause budget for the probability resolution you need (Finding 1 scaling table).
2. Set `T ≈ C/32`; verify `|v|max/T ≈ 1` on a held-out batch. Re-tune `T` per model variant — never
   compare variants at a shared `T` (Finding 8).
3. Set `s = 20–40`. Never below 10 (Finding 3).
4. **Batch ≤ 50** (Finding 5). Budget the compute accordingly; subsample positions rather than
   raising the batch. Larger batches are available at `s=40` but must be re-validated on several
   seeds — the failure there is stochastic.
5. Assert `empty_clauses == 0` every epoch; treat a non-zero count as a failed run, then use the
   differential diagnosis in Finding 4 to decide whether to add data, cut the batch, or add clauses.
6. Report analytic and isotonic side by side — the first is the theory check, the second is what you
   sample from.
7. Keep contexts compressible. Irrelevant width is affordable (Finding 7); unstructured context
   beyond the pool's capacity is fatal and no amount of data fixes it (Finding 6).

## Corrections this gate forced on `VALIDITY.md`

| item | as written | as measured |
|---|---|---|
| E4 (train/eval semantics) | "score under `eval()` at both calibration and generation" | Irrelevant once `empty_clauses == 0`; that is the real invariant (Finding 9) |
| `T` as "resolution knob" | roughly right | The controlling quantity is `|v|max/T`; `T` alone is meaningless (Finding 1) |
| P7 (multi-label) | asserted without evidence | Confirmed, but only at ~8× the `T` multi-class wants (Finding 8) |
| "`s` second order" | claimed from an untuned-`T` sweep | False: `s ≤ 5` collapses, `s ≥ 80` compresses; optimum 40 (Finding 3) |
| H1 (repetition vs breadth) | posed as one dichotomy | Two separate effects: graceful with irrelevant width, hard limit on unstructured capacity (Findings 6, 7) |

## Status

All G0 sweeps are complete: **238 runs**, every one recorded as a JSON record in
`results/g0/` (p01 52, p02 79, p03a 11, p03b 15, p04 63, p05 1, p06 1, p07 16). One library gap raised
(`LIBRARY_GAPS.md` LG-G01, the batch cliff) plus one ergonomic gap (LG-G02, no public access to
unclamped votes). `src/torchtsetlin` unmodified throughout.
