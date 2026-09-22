# ROUND-1 crossfire

Compiled by the orchestrator from the three position papers. Claims are grouped by topic and
stripped of attribution framing. **C-n** marks a contradiction that the round must resolve or
explicitly defer with a named measurement. Where a measurement has since landed, it is shown under
the claims it bears on.

---

## A. The target ladder — the plan's T0/T2 are set from a number that is not a CIFAR-10 result

1. `[FACT: arXiv:2406.00704 §V + ref. 14]` 60.7% traces to an MSc thesis using a **modified**
   CIFAR-10, cited at third hand; its clause budget, window, T, s and epochs are unknown.
   *Orchestrator: independently verified against the PDF.*
2. `[FACT: arXiv:1905.09688 §5]` The 2019 Convolutional TM paper reports **no CIFAR-10 number**.
   *Orchestrator: independently verified.*
3. `[FACT: Sharma et al. AAAI 2023 Table 2]` Vanilla CTM = **69.3%** at 60 000 clauses; Drop Clause
   p=0.5 = **75.1 ± 0.4%**. No window, no epoch count stated.
4. `[FACT: arXiv:2406.00704 Table IV]` Best single model = **75.4 ± 0.09%**, 64 000 clauses, 250
   epochs, T=3000, s=5.0, weighted, 5×5 colour thermometers.
5. `[FACT: arXiv:2406.00704 Tables III–IV]` 82.8% is a **22-specialist composite**, each 64 000
   clauses × 250 epochs, 7 Booleanizations. Composite at 2 000 clauses is already 79.5%.
6. `[FACT: arXiv:2309.04801 Eqs 6–8]` The composition rule normalises by `α_t` computed **over the
   evaluated set** — the published protocol is **transductive**.

**No contradiction between members here.** The consequence is a decision for the user: T0's
reproduction target becomes ~69%, not 55–61%, and T2 rises from 60.7% to 75.4%.

**Open question O-1** — the composite's transductive `α_t`. Our reproduction must fix `α_t` on
validation and report both; does the published number survive that? Nobody has costed this.

---

## C-1. Is the counting pool a standalone candidate? — **the measurement has landed and it is between the two positions**

| | claim |
|---|---|
| tm-theorist | `[HYPOTHESIS]` A counting pool is a **no-op** at the densities the standard feedback rule produces; it pays off only jointly with density control. Falsifier stated in advance: **median \|M_j\| ≥ 5 for a firing clause**. |
| dl-expert | `[HYPOTHESIS]` OR gives the vote `C` bits, counting gives ~`9.7C` — a ~10× information increase at identical automata budget and identical interpretability. Predicted **5–12 points**. Pre-committed: if the TM histogram shows **median ≈ 1 given firing**, the standalone prediction is withdrawn and the mechanism merges with density control into one paired candidate. |

**`[MEASURED: results/ctm-small_seed*.json → diagnostics.match_count]`** and
**`[MEASURED: screen/cnn-ctmshape-max_seed0.json]`**:

| | `ctm-small` (TM, trained) | `cnn-ctmshape-max` (CNN) |
|---|---|---|
| fire_frac | 0.070 | 0.0096 |
| **median given firing** | **2** | 40 |
| mean given firing | 6.1 | 86.6 |
| **P(\|M\| ≥ 5 \| fires)** | **0.303 ± 0.002** | 0.883 |
| P(\|M\| = 1 \| fires) | 0.356 | 0.038 |

**Neither pre-registered condition is met.** The theorist's falsifier (median ≥ 5) fails. The DL
expert's withdrawal condition (median ≈ 1) also fails — median is 2, and 30% of firing clauses match
five or more positions. The distribution is **long-tailed, not degenerate**.

Also `[MEASURED]`: a CNN filter is a **rare, broad** detector; a TM clause is a **common, narrow**
one — it fires on 7× more pairs but matches 20× fewer positions. So the CNN-side 12.2-point proxy
(`cnn-ctmshape-max` 38.86% vs `-sum` 51.02% test, registered screens) is measured on a substrate
whose match distribution is not the TM's, which weakens its transfer.

**Both members must now answer**: does your position survive a median of 2 and
`P(≥5 | fires) = 0.303`? Specifically — is the counting pool (a) standalone, (b) admissible only
paired with density control and screened as a unit against a density-only control, or (c) dead?
Name the arm structure you would now screen.

---

## C-2. Capacity-limited or mechanism-limited? — a direct contradiction

| | claim |
|---|---|
| dl-expert | `[HYPOTHESIS]` The capacity curve **saturates below 70% by ~8–16 k clauses**; CIFAR-10 CTMs are mechanism-limited, not capacity-limited. "This must be tested before P5, not after." |
| tm-theorist | `[FACT: arXiv:2406.00704 Table IV]` 5×5 colour thermometers go **64.5 → 75.4 from 2 000 to 64 000 clauses and are still climbing at 64 000** (+1.7 from 32k). HOG, by contrast, is flat at 67.5 by 32 000. `[HYPOTHESIS]` Saturation is at the **Booleanization's** information ceiling, not the clause mechanism's. |

These cannot both hold. The published curve for the best encoding is still rising exactly where the
DL expert predicts saturation. If the theorist is right, the honest next move for accuracy is *buy
clauses and better encodings*, and mechanism work is a separate contribution about efficiency and
interpretability rather than accuracy. Both members rank this measurement first or near-first, and
the engineer has priced it.

**Falsifier already named by the theorist**: `cnn-boolean` on HOG bits substantially beating 67.5%
would show the ceiling is the TM's, not the encoding's. **Resolve with: our own capacity curve over
≥2 encodings** (P4.1, ~25 GPU-h at the measured price list), plus that `cnn-boolean` probe.

---

## C-3. Augmentation — a direct contradiction about a cheap arm

| | claim |
|---|---|
| dl-expert | Rank 2, predicted **+1.5–4 points**, **zero new code**; augmentation is worth 3–7 points to CNNs `[FACT: arXiv:1708.04552 Table 1 — ResNet-18 4.72% aug vs 10.63% no-aug]`. |
| tm-theorist | `[FACT: arXiv:2406.00704 Table IV]` Static flip augmentation **hurts** the best specialist by 1.3 points and raises its variance ~15×, while helping the edge/threshold specialists; and the published comparison is **not data-budget-matched** (100 000 vs 50 000 images at the same epoch count). |

Both cite real numbers; they are about different substrates (gradient-trained CNN vs TM) and the TM
evidence is confounded by data budget. **Resolve with a data-budget-matched augmentation arm** — the
one design neither paper ran.

---

## C-4. What is `ctm-vanilla`, and what does faithfulness cost?

1. `[FACT: arXiv:1905.09688 Table 1; arXiv:2301.08190 fn. 6]` "Clauses" means **per class** in the
   CTM and CSC-TM papers. `[FACT: arXiv:2309.04801]` TM Composites uses **both conventions in one
   paper**. The Toolbox and GraphTM are unqualified. **A factor of 10 in automata budget is at
   stake.**
2. `[FACT]` `ctm-vanilla` is implemented as `ConvTsetlinMachine` (class-owned), not coalesced: the
   2019 paper convolves the 2018 multi-class TM, and using the coalesced model would reproduce the
   wrong paper *and* break matched-budget comparisons against `ctm-coalesced`. Both members agree.
3. `[MEASURED: calib_large_gpu0]` The faithful reading is **80 000 total clauses → 632 s/epoch →
   10.5 GPU-h per seed, 31.6 GPU-h for three** — a quarter to a third of the P3 envelope. The
   unqualified 8 000-clause reading is **3.9 GPU-h for three seeds**.
4. `[MEASURED: smoke_vanilla80k]` Above ~16 000 clauses the per-epoch probes stop being free: +23%,
   turning a 10.5-hour seed into 13 hours. Mitigated by `--eval-n 2000 --m2-n 500`.

**Engineer's recommendation**: run the 8k reading first — it costs 4 GPU-h and could save 28.
**Question for the round**: does an 8 000-clause `ctm-vanilla` that lands near the Toolbox's own
8 000-clause column discharge T0, or does T0 require the 80 000-clause arm?

---

## C-5. Our protocol makes every reproduction look worse — and nobody has priced it

`[FACT: LITERATURE_TM.md §2.5]` **No TM image paper uses a validation split.** Reported accuracy is
the mean of the last 25 or 100 epochs, or a peak. Under PLAN §7.1 our numbers will be *lower* for
the same configuration — `[HYPOTHESIS]` by ≤ 1 point for a converged CTM.

This interacts with the §7.5 reproduction tolerance (3 points): a systematic protocol penalty eats a
third of the tolerance band before any real discrepancy appears.

**Proposed control, already suggested by the theorist**: log the paper's own statistic (last-25-epoch
mean, and peak) **alongside** the validation-selected number, for every `existing`-family arm. Costs
nothing — both are computable from the recorded curve. The engineer should confirm the curve carries
what is needed.

Related `[MEASURED: results/calibration_seednoise.json]`: `ctm-small` plateaus by epoch 2–9 then
oscillates; 7–15 of its 30 epochs sit within 0.5 pp of the best, and selected epochs were 29, 8, 14,
13 and **2**. `selected_epoch` must not be read as "when the model was best".

---

## C-6. `ctm-clausesize` cannot be reproduced as published

`[MEASURED: code/repro/lg003.py, LG-003]` Budget 8 → median clause size **10.5 at batch 50**, 8.0 at
batch 5, 7.0 sequential. The budget does not bind under batched feedback. The arm must use sequential
feedback (~10× slower) or a ported density controller — **and which one it uses changes what "we
reproduced that paper" means.**

`[FACT: arXiv:2301.08190 Table 3]` The mechanism is worth **+0.06** on CIFAR-2 and +0.05 on MNIST —
an interpretability and hardware mechanism, not an accuracy one — **yet it is part of two SOTA
recipes** (budget 32 in both TM Composites and GraphTM).

**Both members appear to agree it is a control, not a candidate.** Confirm, and state whether the
control must bind (sequential, 10× cost) or whether recording that it does not bind is sufficient.

---

## D. Agreements worth recording as settled

- Since 2019 **nobody has changed the CTM's convolution mechanism**. Swept across the corpus; the
  only exception is CTM-UNet, on segmentation, as a backprop hybrid, with no CIFAR-10 number.
- Published CIFAR-10 progress decomposes into Booleanization (**16.7-point spread at fixed 2 000
  clauses**), clause count (+10.9 from 2k→64k), composition (+7.4 to +11.6) and Drop Clause (+5.8).
- **Binary computation is not the bottleneck**: best fully binary net = 93.75% vs 94.8% FP; binary
  *weights* cost nothing (BWN 56.8 vs FP 56.6 on ImageNet), binary *activations* cost 12.4.
- **GraphTM's CIFAR-10 advantage is not a depth effect** — its CIFAR-10 run is depth 1 and the
  authors attribute the gain to multi-view input. A cheap multi-view CoTM arm tests the same claim.
- `T_ratio = 0.8` is **not** a safe default: `T/(total clauses)` spans **0.025 to 1.5** across the
  bibliography. Every arm carries its paper's own T and s.
- Memory is never the constraint (0.9–1.5 GB from 640 to 8 000 clauses; 8.4 GB at 80 000). The 3080
  is a **slower card, not a small one** — 1.08× at 8k, **1.46× at 80k**. That is the device policy.
- The seed band: **two 3-seed means must differ by ≥ 1.00 pp to be a difference.** Selection adds
  only ~0.09 pp of it; validation-only selection is not what makes it wide.
- `mctm-skip` is feasible with **no subclass at all** if the skip tensor is resampled to the
  clause-map grid; the chunk budget is then automatically correct.
- No arm uses a non-default chunk budget until the neutrality check lands; thereafter arms are
  compared only at equal values.

---

## What each member must return

1. Which claims above you **accept**, which you **reject**, and on what grounds.
2. **C-1 and C-2 are the two that must be settled now** — they gate the P5 branch and the whole
   framing of the programme. Answer them directly; do not defer.
3. For each remaining disagreement, **the single measurement that would settle it**, with the
   engineer's price list applied.
4. Your final ranked P3 shortlist with costs, given the measured price list and a ~120 GPU-h P3
   envelope. State what you would cut first.
