# Generative Tsetlin Machines — experimental plan

**Status: Gate G0 PASSED and Gate G1 PASSED (2026-09-24) — see `FINDINGS_G0.md` and
`FINDINGS_PHASE1.md`.** Phase 2 (ECG) is cleared to start. `src/torchtsetlin` unmodified
throughout.

**Goal.** Test, by measurement, whether a Tsetlin machine can be used as an autoregressive
generative model — first on binarised MNIST, then on 12-lead ECG — and publish the result either
way. The literature contains no TM image generator and no TM ECG generator
(`source_documents/papers/07_generative_sequence_tm/README.md`), so a negative result that is
properly controlled is also a publishable result.

Read `VALIDITY.md` first. It establishes the one fact the whole programme rests on — the TM vote
sum has fixed point `v* = T(2p − 1)`, so `p̂ = (v + T)/(2T)` — and lists the 12 design corrections
(E1–E5, P1–P8) already folded into the arms below.

---

## 1. Hypotheses

Each is falsifiable, has a named metric, and a gate that can stop the programme.

| # | Hypothesis | Falsified if |
|---|---|---|
| **H0** | A TM's vote sum converges to `T(2p−1)`, so `(v+T)/(2T)` is a calibrated probability without a fitted calibrator. | Best-setting ECE > 0.05 on synthetic data with known `p` (§3.1). |
| **H1** | Calibration quality is governed by **context repetition**, not context breadth. | A unique-context arm calibrates as well as a repeating-context arm at matched capacity. |
| **H2** | An autoregressive TM over a local causal window generates recognisable MNIST digits. | Generated-digit accuracy under a held-out CNN judge < 40% (chance = 10%), or samples collapse to ≤ 3 distinct images per class. |
| **H3** | The TM beats a matched-feature logistic regression and a lookup-table marginal on held-out next-pixel NLL. | It does not, on 3 seeds. |
| **H4** | A block-code TM beats a count-based n-gram on held-out ECG token NLL. | It does not. Then the TM adds only interpretability, and we say so. |
| **H5** | Generated 12-lead ECGs reproduce real RR-interval statistics and satisfy the Einthoven/Goldberger lead identities. | KS distance of generated vs real RR > 0.3, or lead-identity residual exceeds codebook reconstruction error. |

**What we will not claim:** clinical validity, diagnostic utility, or privacy safety. Memorisation
is measured (§5.4), not asserted away.

## 2. Hard constraints

| # | Constraint | Enforcement |
|---|---|---|
| C1 | **No edits to `src/torchtsetlin/**` without explicit user confirmation.** | `git diff --stat src/` must be empty at every gate. All code lives in `experiments/generation/code/`. |
| C2 | Missing or wrong library behaviour is **reported, not fixed.** | Numbered entry in `LIBRARY_GAPS.md` with a failing reproduction, impact estimate and patch *sketch*. |
| C3 | No method is adopted without a measured result against a named baseline. | Every arm in `ARMS.md` has a control; any arm without one is not scored. |
| C4 | 3 training seeds minimum for any reported number; sampling seeds separate from training seeds. | `record.py` refuses to write a summary row with < 3 seeds. |
| C5 | Test split touched once, at the end of each phase. Calibration and tuning use validation only. | Split loader raises unless `allow_test=True`. |

---

## 3. Phases and gates

### 3.1 Phase 0 — Does the probability interpretation hold? (**gate G0**)

The cheapest and most decisive phase. Synthetic data with *known* `P(y=1|x)`, so calibration error
is measurable exactly rather than estimated. ~2 GPU-hours total; run this before anything else.

Construction: `F` Boolean features; a fixed random assignment maps each of `M` distinct contexts to
a target probability drawn from `{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}`; labels are sampled
Bernoulli. Train, then compare measured `v/T` against `2p − 1`.

| ID | Sweep | Question | Why it matters |
|---|---|---|---|
| P0.1 | `T ∈ {10, 25, 50, 100, 200}` | Does `v/T` track `2p−1` with slope 1? What is the achievable ECE? | **H0.** Also fixes `T` as the *probability resolution* knob (steps of ~`1/2T`), not a margin. |
| P0.2 | batch `∈ {1, 10, 50, 100, 500}`, `feedback_mode ∈ {batch, sequential}` | Does batched feedback aggregation bias the fixed point? | `CLAUDE.md` warns fidelity degrades with batch size. Both ideas want large batches over millions of examples. If batch 100 biases `p̂`, every downstream arm must use small batches and the budget triples. |
| P0.3 | context repetition `∈ {1, 10, 100, 1000}` occurrences | How much repetition does calibration need? | **H1**, and the decision rule for MNIST window size and ECG token encoding (E3, P2). |
| P0.4 | `K ∈ {16, 64, 256}`, multi-class vs `multi_label=True` | Are *all* `K` outputs calibrated, or only the argmax? | **P7.** Sampling needs the whole distribution. Settles the ECG model choice empirically. |
| P0.5 | `train()` vs `eval()` scoring | Vote shift from empty-clause semantics. | **E4.** Confirms the silent-bug risk and sets the empty-clause diagnostic threshold. |
| P0.6 | sample 10⁶ draws from `p̂` | KL(empirical ‖ true) | Confirms sampling, not just scoring, is faithful. |
| P0.7 | `s ∈ {2, 5, 10, 20}`, clauses `∈ {50, 200, 800}` | Capacity/specificity effect on calibration. | Sets defaults for Phases 1–2 instead of guessing. |

**Gate G0 — PASSED.** P0.1 reached analytic mae **0.0175 ± 0.0011** (gate: ≤ 0.05) and isotonic
**0.0029**, with fixed-point slope 0.963 and r² 0.997 on 3 seeds. P0.4 identified the mode in which
all `K` outputs calibrate (coalesced, `multi_label=True`, large `T`).

**Programme defaults recorded from G0** (`FINDINGS_G0.md` for the evidence):

| knob | value | why |
|---|---|---|
| clause budget `C` | as large as the compute allows | sets attainable probability resolution |
| `T` | `≈ C/32`, then verify `\|v\|max/T ≈ 1` | Finding 1 — dominates everything |
| `s` | **40** (usable band 10–40) | Finding 3 — `s ≤ 5` collapses, `s ≥ 80` compresses |
| batch | **≤ 50** | Finding 5 — cliff above, stochastic near it |
| calibration | analytic `(v+T)/2T` reported beside isotonic | Finding 2 |
| per-epoch assert | `empty_clauses == 0` | Finding 4 — detects every failure mode |
| model for K-way tokens | `CoalescedTsetlinMachine(multi_label=True)` | Finding 8 — class-owned is unusable for sampling |

**Comparisons between model variants must sweep `T` per variant until each peaks and turns over**
(Finding 8). A comparison at a shared `T` is not a comparison.

### 3.2 Phase 1 — MNIST autoregressive generation (**gate G1**)

Data: `load_mnist_boolean` at `pixel > 0.3` (the repo's existing MNIST convention). Split by image:
50k train / 10k validation / 10k test. Raster order. All scoring and generation under `model.eval()`.

1. **P1.0 baselines first** — `mnist-marginal` (lookup table, no model) and `mnist-logreg-window`.
   These take minutes and define the NLL floor. No TM arm is scored before they exist.
2. **P1.1 `mnist-tm-window` (primary)** — causal window: previous 2 rows + 4 pixels left (~60
   pixels), `row`/`col` thermometers (28+28), digit one-hot (10) ≈ 126 features. Shuffle at
   `(image, position)` level (E2); sample 100 positions/image/epoch to cut epoch cost 8× and
   decorrelate batches further.
3. **P1.2 `mnist-tm-canvas` (ablation)** — the original full-canvas design with E1 applied
   (thermometer position, no redundant one-hot). Tests H1 directly: breadth vs repetition.
4. **P1.3 window-size sweep** — {1 row+2, 2 rows+4, 4 rows+8}. The H1 curve.
5. **P1.4 calibration comparison** — analytic `(v+T)/(2T)` vs isotonic vs Platt, on validation.
   Reported as a reliability diagram. Deviation from the analytic map *is* a result.
6. **P1.5 `mnist-tm-nocal` (control)** — sampling from raw thresholded votes, to show what
   calibration buys.
7. **P1.6 `mnist-tm-gray16` (extension)** — 16 intensity bins via
   `CoalescedTsetlinMachine(multi_label=True)`, per the mode chosen at G0.

**Gate G1 — PASSED (narrowly on sample quality).** Winning arm: `WindowContext(4,8)`, weighted,
T=8000, s=40, 2 000 clauses/class, batch 240, 4 M examples, 3 seeds.

| criterion | required | measured |
|---|---|---|
| NLL vs `mnist-marginal` | < 177.29 | **77.39 ± 0.49** |
| NLL vs `mnist-logreg-window` | < 127.45 | **77.39 ± 0.49** |
| judge accuracy | ≥ 0.40 | **0.405 ± 0.020** (band straddles the gate) |
| distinct samples/digit | ≥ 20 | **100** |

**Rule change forced by Phase 1:** checkpoint selection on validation NLL (§3.2 step 5) is wrong for
a generator. Held-out likelihood improves monotonically with training while sample quality peaks and
then declines (2 M / 4 M / 8 M → judge 0.272 / 0.405 / 0.386 at NLL 83.2 / 77.4 / 76.7).
**Early-stop on a sample-quality metric.** For Phase 2 that means RR-interval and morphology
statistics, not token NLL.

### 3.3 Phase 2 — 12-lead ECG (**gates G2, G3**)

Data: **PTB-XL v1.0.1**, already on disk at
`/work/vajira/DATA/EXG_PTB_XL/physionet/files/ptb-xl/1_0_1` — verified 2026-09-24: **21 837 records,
18 885 patients**, `records500/` present, `strat_fold` present with 10 balanced folds (2 175–2 203
records each). Use folds 1–8 train, 9 validation, 10 test (patient-disjoint; 9–10 human-validated).
Canonical lead order saved; units and sampling rate fixed per model.

**Stage A — codebook study (no TM at all).** Reconstruction bounds everything downstream, so
resolve it before spending a GPU-hour on clauses. Grid: alignment {fixed grid, R-peak aligned},
`B ∈ {25, 50, 100}`, `K ∈ {64, 256, 512, 1024}`, channels {8 independent, all 12}, quantiser
{k-means, product quantiser}. Metrics: per-lead reconstruction SNR and RMSE on held-out records,
QRS amplitude and duration error, seam discontinuity ratio (|Δ| at block joins ÷ median |Δ| within
blocks).

> **Gate G2.** Proceed iff some configuration reaches **≥ 20 dB** per-lead SNR on held-out data with
> seam discontinuity ratio ≤ 2. If the best fixed-grid config fails and R-peak alignment passes,
> that comparison is itself the finding (P1). If both fail, the block-code architecture is refuted
> and Phase 2 stops here — cheaply, which is the point of ordering it first.

**Stage B — token models.** Baselines before TMs, again: `ecg-marginal` (i.i.d. token sampling) and
`ecg-markov-L` (count-based n-gram with backoff — **the arm that can falsify H4**). Then
`ecg-tm-onehot` (the document's literal design, as the control for P2), `ecg-tm-compositional`
(product-quantiser sub-code context), `ecg-tm-phase` (+ cardiac-phase and RR conditioning, `L ≥ 20`),
`ecg-tm-8lead` (8 independent leads, 4 derived).

**Gate G3.** Report as success only if a TM arm beats `ecg-markov-L` on held-out token NLL on 3
seeds, and generated RR-interval distributions have KS distance ≤ 0.3 against real held-out records.

### 3.4 Phase 3 — report

LaTeX → PDF under `experiments/generation/report/`, following `whitepaper/` conventions: theory
(the fixed-point derivation of §1 in `VALIDITY.md`, which is the paper's actual contribution),
method, arms table, results with seed bands, negative results kept in. Plus a `LIBRARY_GAPS.md`
disposition and — only if the results warrant it, and only with user confirmation — a proposal for
`src/torchtsetlin`.

---

## 4. Arms

See `ARMS.md`. Nothing runs that is not registered there first.

## 5. Metrics

**5.1 Density (both phases).** Held-out NLL — nats/image for MNIST (anchored against the standard
binarised-MNIST ladder, NADE ≈ 88 / PixelRNN ≈ 79 [VERIFY]), nats/token for ECG. This is the only
metric that compares model to baseline on equal terms.

**5.2 Calibration.** ECE and reliability diagrams on validation; deviation of the fitted calibrator
from the analytic `(v+T)/(2T)`; empty-clause count in both modes.

**5.3 Sample quality.** MNIST: accuracy of a separately-trained CNN judge on generated digits;
ink-fraction and connected-component distributions vs real. ECG: RR-interval KS distance, heart-rate
range, QRS duration/amplitude distributions, seam discontinuity ratio, **lead-identity residual**
(`III − (II − I)` etc., free by P3), and a TSTR check — train a rhythm classifier on synthetic,
test on real.

**5.4 Diversity and memorisation.** Distinct-sample count; mean pairwise Hamming (MNIST) / DTW
(ECG) distance within a class; nearest-training-neighbour distance distribution against the
train-to-train baseline. Reported as a measurement, never as a privacy guarantee.

**5.5 Cost.** Wall-clock per epoch, examples/s, peak memory, seconds per generated sample.

## 6. Protocol

Splits fixed once by seed and saved. 3 training seeds × ≥ 100 sampling seeds per reported number.
Every run writes a JSON record (arm, config hash, git SHA, device, seed, metrics, wall-clock) via
`code/record.py`; summaries are derived from records only — no hand-entered numbers, following
`experiments/convtm`. Test split loaded once per phase with `allow_test=True`.

## 7. Budget [ESTIMATE, anchored on `docs/benchmarks.md`, RTX 3090]

| Phase | Work | GPU-hours |
|---|---|---|
| 0 | 7 sweeps × small models | ~2 |
| 1 | baselines + 6 TM arms × 3 seeds × ~20 epochs | ~25 |
| 2A | codebook grid (CPU-heavy, no TM) | ~6 (mostly CPU) |
| 2B | baselines + 4 TM arms × 3 seeds | ~30 |
| 3 | figures, report | ~2 |
| | **total** | **~65 GPU-hours** |

Both ideas are compute-feasible; see `VALIDITY.md` §5. The dominant practical constraints are
building contexts on the fly (never materialise 47 M × F bools) and batching *samples* rather than
steps during generation, which is inherently serial.

## 8. Deliverables

1. `VALIDITY.md` — done (this review).
2. `code/` — arms, runner, record format, baselines, metrics. Nothing under `src/`.
3. `results/` — one JSON record per run; `tables.md` derived.
4. `report/` — LaTeX → PDF.
5. `LIBRARY_GAPS.md` — numbered, with reproductions.
6. A go/no-go recommendation on each of H0–H5, with the seed bands that support it.

## 9. Risks and kill criteria

| Risk | Detect | Kill / mitigate |
|---|---|---|
| Fixed point does not hold in practice | P0.1 ECE | **Kill at G0**; pivot to PTM or ensemble estimator and report why |
| Batched feedback biases calibration | P0.2 | Small batches; budget ×3; recorded as a library finding |
| Unique contexts → no calibration | P0.3, P1.2 vs P1.1 | Window arm is already primary (E3) |
| Exposure bias: sampled prefixes drift off-distribution | Compare teacher-forced vs free-running NLL | Scheduled sampling; report the gap rather than hide it |
| Codebook cannot represent QRS | Stage A SNR | **Kill at G2**, or switch to beat-level tokens |
| TM only matches an n-gram | `ecg-markov-L` | Report honestly (H4); compositional encoding is the designed response (P2) |
| Generation too slow to evaluate | s/sample in P1.1 | Batch independent samples; subsample positions |
| Silent train/eval mode bug | P0.5 + empty-clause logging | Assert `not model.training` inside scoring and sampling entry points |

## 10. Layout

```
experiments/generation/
  VALIDITY.md   PLAN.md   ARMS.md   LIBRARY_GAPS.md
  code/    arms.py record.py data_mnist.py data_ptbxl.py context.py
           calibrate.py sample.py metrics_*.py baselines/ codebook/ run_arm.py
  results/ *.json  tables.md
  report/  main.tex sections/ figures/
```

Commands are registered per arm in `ARMS.md`. **None have been run.**
