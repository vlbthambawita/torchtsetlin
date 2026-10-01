# Validity review — the two TM generation ideas

Reviewed: `ideas/generations/mnist_tsetlin_generation_pseudocode.md`,
`ideas/generations/tsetlin_12_lead_ecg_generation_pseudocode.md`. Date: 2026-09-23.
Nothing was executed. Claims tagged **[DERIVED]** (from the library source), **[MEASURED]**
(from `docs/benchmarks.md`), **[ESTIMATE]** (arithmetic on a measured anchor), **[VERIFY]**
(needs checking before it is relied on).

## Verdict

| | Idea 1 — MNIST | Idea 2 — 12-lead ECG |
|---|---|---|
| Core mechanism sound? | **Yes** | **Yes** |
| Rests on a false assumption? | No | No |
| Fatal flaw? | None found | None found |
| Design errors to fix first | 5 (E1–E5) | 7 (P1–P7) |
| Compute feasible on this repo's hardware? | Yes | Yes |
| Biggest risk | context sparsity → miscalibration | codebook phase misalignment |

Both are **worth running**. Neither is novel-by-assertion: §1 shows the probability
interpretation they depend on is a property of the TM feedback rule, not a hope. The fixes below
are design corrections, not rescues — and three of them (E3, P2, P8) change what gets run first.

---

## 1. The load-bearing assumption is true, and better than either document claims

Both documents hedge: *"an ordinary classifier's raw clause votes are not inherently
well-calibrated probabilities"* (MNIST), *"the TM itself is not a likelihood model by default"*
(ECG). That hedge is too pessimistic, and the correct statement is much stronger.

`src/torchtsetlin/functional.py:112` gives the feedback probability exactly:

```python
def feedback_probabilities(votes, T, target):
    v = votes.clamp(-T, T)
    if target: return (T - v) / (2*T)     # target class
    return (T + v) / (2*T)                # negative class
```

Take a context that occurs with `P(y=1) = p`. Class 1's vote sum `v` is pushed **up** when the
example is positive — which happens with rate `p·(T−v)/(2T)` — and **down** when negative, at rate
`(1−p)·(T+v)/(2T)`. Setting the drift to zero:

```
p(T − v) = (1 − p)(T + v)   ⟹   v* = T(2p − 1)   ⟹   p = (v* + T) / (2T)
```

**The TM's vote sum is a fixed-point estimator of the conditional probability, and the feedback
rule is what drives it there.** [DERIVED]

Three consequences that neither document exploits:

1. **The correct calibrator is known analytically, not just fittable.** `p̂ = (v + T)/(2T)` is
   already implemented as `functional.confidence_from_votes` and as
   `TsetlinMachine.predict_proba(method="linear")` (`models/classifier.py:127`), citing Abeyrathna
   et al. So a fitted calibrator should be *compared against* the analytic map, not used blind. If
   isotonic regression on validation data recovers something close to the identity, the model is
   behaving as theory says and the result is trustworthy. If it does not, that discrepancy is the
   finding.
2. **The "saturation kills diversity" worry is unfounded.** Feedback pressure vanishes as
   `v → ±T`, but the fixed point is *interior* for every `p ∈ (0,1)`: `p=0.5 → v*=0`,
   `p=0.9 → v*=0.8T`. The TM does not collapse to a hard decision on genuinely ambiguous contexts;
   it parks at the vote sum that encodes the ambiguity. `T` is therefore the **probability
   resolution knob** — the distribution is quantised to steps of roughly `1/(2T)` — and should be
   tuned as such, not as a margin.
3. **It holds for the coalesced model too.** `models/coalesced.py:24` uses
   `d_i = |q_i − v_i|/(2T)` with `q_i = ±T`, which is the same two expressions, so the same fixed
   point applies per output. [DERIVED]

**Where it can break, and this is the real risk.** The fixed point is reached only for contexts
*seen repeatedly*. A clause covers a region of context space, so what actually equilibrates is
`P(ink | region covered by the matching clauses)`. If every training context is effectively unique,
there is no repetition to average over and the vote sum is an extrapolation, not an estimate. This
single mechanism is why E3 and P2 below matter more than anything else in either document.

---

## 2. Idea 1 — MNIST. Correct in outline; five fixes

**Right:** raster factorisation is exact; teacher forcing with `observed` bits is the standard
construction and the assertions in `ENCODE_CONTEXT` genuinely prevent target leakage; splitting by
image before building pixel examples is correct; fitting the calibrator on validation only is
correct; feature arithmetic checks out (784+784+10+784 = 2362).

**E1 — `observed` is a deterministic function of `position`, so 784 features carry no information.**
The pseudocode asserts this itself (`assert all(observed[j]==1 for j<position)`). Under teacher
forcing in raster order, `observed` *is* the thermometer encoding of `position`, and it sits
alongside a 784-wide one-hot of the same quantity. That is defensible — a thermometer gives clauses
a cheap `position ≥ j` predicate that one-hot cannot express in one literal — but it should be a
stated choice, not justified by a "known zero vs unknown" ambiguity that raster order has already
removed. Drop the one-hot, keep the thermometer, and encode `row`/`col` separately (28+28 bits);
2362 features → ~850, which is a 2.8× throughput gain for no information loss. [DERIVED]

**E2 — the batching correlates examples, and this repo is on record that that hurts.**
`MAKE_TRAINING_EXAMPLES` yields 784 consecutive positions of one image, and `BATCH(stream, …)` then
slices that stream — so every update batch is ~100 adjacent pixels of a *single* digit. Batch-mode
feedback aggregates all events of a batch and applies them once (`CLAUDE.md`: fidelity degrades as
batch size grows; 10–50 fine, 200 noticeably worse). Aggregating near-duplicate examples is the
worst case for that. **Shuffle at the `(image, position)` level**, or sample random `(image,
position)` pairs per step. This also makes position-subsampling (below) natural.

**E3 — the priorities are inverted: the local window should be the primary arm.**
The document presents the full 2362-bit canvas as the main design and a local window as a cheaper
"practical variant" that "loses distant context". By §1 the opposite ordering is right. A full-canvas
context is *unique to one image* after a few dozen pixels, so contexts essentially never repeat and
the equilibrium never forms. A causal window (e.g. previous two rows plus four pixels left, ~60
pixels) repeats constantly across the 60 000 training images, which is exactly the condition the
fixed point needs — and it is ~19× cheaper. Run the window first; the full canvas becomes the
ablation that tests whether context breadth or context repetition matters more.

**E4 — `train()`/`eval()` mode will silently invalidate the calibrator.**
An empty clause (no included literals) evaluates **True** while training and **False** when
predicting, keyed off `self.training` (`CLAUDE.md`; `models/base.py:271`). Calibrating in one mode
and generating in the other shifts every vote sum by the number of empty clauses. `SCORE_ALL_EXAMPLES`
and `GENERATE_DIGIT` must both run under `model.eval()`, and empty-clause count should be logged as
a diagnostic. This is a silent wrong-answer bug, not a crash.

**E5 — no baselines, so no claim is falsifiable.** Needed, cheapest first: (i) per-position marginal
`P(ink | position, digit)` — a lookup table, no model; (ii) logistic regression on the identical
feature vector, which isolates "TM" from "these features"; (iii) a small PixelCNN for the NLL ladder.
Anchor the NLL in nats/image against the standard binarised-MNIST ladder (NADE ≈ 88, PixelRNN ≈ 79)
[VERIFY].

**Minor:** "trained incrementally … the TM adapter must preserve automaton state across
`update_batch` calls" — no adapter is needed, `update()` mutates persistent buffers by construction
(`CLAUDE.md`, "The learning state"). And `update()` already **returns the pre-update vote sums**, so
calibration data can be harvested during training instead of by a second scoring pass.

---

## 3. Idea 2 — 12-lead ECG. Correct in outline; seven fixes

**Right:** patient-level splitting before tokenisation; fitting scaler and codebook on train only;
gating generation on codebook reconstruction error *first* (the document is admirably firm that the
generator cannot recover what the codebook discarded); BOS distinct from every code ID; declaring
that block joins are left visible in the baseline rather than hidden.

**P1 — nonoverlapping blocks are not phase-aligned, and this is the gate.**
With `B=50` (100 ms) cut on a fixed grid, a QRS complex lands at an arbitrary offset inside its
block. The codebook must therefore spend its capacity on ~50 shifted copies of every morphology
before it represents a single one well, so either `K` explodes or reconstruction is poor — and a
poor codebook caps everything downstream. Fix, in order of preference: (a) **R-peak-aligned
tokenisation** — detect R peaks, tokenise beat-relative segments, and model the RR interval as its
own token stream; (b) keep the fixed grid but add **cardiac phase (samples since last R peak,
thermometer-encoded)** to the context; (c) overlap-and-add decoding. Measure reconstruction SNR per
lead under (a) and (b) before committing.

**P2 — one-hot token context makes the TM an n-gram lookup table.**
`ONE_HOT(token, K+1)` per context slot means two codes that decode to near-identical waveforms share
no bits. A clause can then only match *exact* token sequences, so the model memorises n-grams and
cannot generalise across similar morphologies — and it will be beaten, or matched, by a count-based
Markov chain that does the same thing without a TM. **Encode each context token compositionally**:
product-quantiser sub-codes, or a thermometer over a few centroid descriptors (peak amplitude,
dominant lead, energy, slope). Then clauses express "previous block was high-amplitude in V1-ish"
and generalisation becomes possible. This is the difference between a result and an expensive
re-derivation of a bigram table.

**P3 — four of the twelve leads are exact linear functions of the other two.**
`III = II − I`, `aVR = −(I + II)/2`, `aVL = I − II/2`, `aVF = II − I/2`. So a 12-lead ECG has only
**8 independent** channels (I, II, V1–V6). Two consequences the document misses: the codebook should
be `[K, 8, B]` and the remaining four leads *derived* (33% less to quantise, and cross-lead
consistency becomes exact by construction); and if you do model all 12, the identity residual is a
free, sharp, automatic validity metric on generated output. Either way this replaces "does not
guarantee physiological cross-lead relationships" with something quantitative.

**P4 — `L` is too short for rhythm.** At `B=50`, `L=8` is 800 ms — barely one beat, so the model
cannot see RR regularity at all, and sustained AF is out of reach. Use `L ≥ 20` (2 s), or carry
explicit RR/phase state (P1b), which is far cheaper than a long window.

**P5 — no dataset is named.** Use **PTB-XL**: 21 799 records, 18 869 patients, 10 s, 500 Hz, 12-lead,
with official patient-disjoint `strat_fold` — folds 1–8 train, 9 validation, 10 test, where 9 and 10
carry human-validated labels. It satisfies the document's own patient-splitting requirement out of
the box, so do not invent a split.

**P6 — missing the one baseline that can falsify the idea.** A count-based **n-gram / Markov chain
over the same tokens** is the direct competitor: same tokenisation, same context length, closed-form
probabilities, seconds to fit. If the TM does not beat it on held-out token NLL, the TM contributes
nothing but interpretability and the honest result is to say so. Add also: i.i.d. sampling from the
token marginal (floor), and codebook round-trip of *real* held-out ECGs (ceiling — nothing generated
can beat it).

**P7 — use multi-label mode, not multi-class, or the sampling distribution will be wrong.**
This is library-specific and decisive. In multi-class mode only the target and **one random
negative** class are updated per example (`models/coalesced.py:30`, `classifier.py` via
`sample_negative_classes`), so the other `K−2` vote sums are updated rarely and never reach their
fixed point. Argmax accuracy survives that; a *sampling* distribution does not — you need all `K`
probabilities calibrated, not just the winner. `CoalescedTsetlinMachine(multi_label=True)` with
one-hot targets updates **every** output with its own probability, which is exactly the regime §1
describes. Keep `negative_scale = 1.0`, since that hyperparameter deliberately skews the Type II
rate for negative outputs and would bias `p̂` away from `(v+T)/(2T)`. [DERIVED]

**P8 — the codebook is where the risk is, so run the codebook study before any TM training.**
Reconstruction quality bounds everything and costs no TM time. If R-peak-aligned `K=512` cannot
reconstruct held-out ECGs to a clinically unembarrassing SNR, the architecture is refuted early and
cheaply.

**Minor:** `max_blocks` and "predict `ceil(T/B)` blocks then trim" is right but means the position
feature must cover the longest generation, not the training length — otherwise generation walks
off the encoding. Assert it.

---

## 4. What both documents are missing

1. **Baselines** (E5, P6). Without them neither result can be interpreted.
2. **Named gates.** Both list metrics to "report". Neither states a number that would stop the work.
   Phase gates are in `PLAN.md` §3.
3. **Compute budget.** Both are silent; §5 shows both fit comfortably.
4. **Seeds.** Single-seed generative results are not evidence — `experiments/convtm` already
   established a ±0.6 pp seed band on a *classification* metric; sampling adds a second noise source.
   Three training seeds × ≥100 samples each.
5. **`s`, not `max_included_literals`, is the clause-size control** in this library — a recorded
   finding from earlier work here.

## 5. Feasibility

Anchor [MEASURED, `docs/benchmarks.md`, RTX 3090]: `TsetlinMachine`, 5 000 clauses total,
F=784 (2F=1568), batch 100 → **39 000 examples/s**. Cost scales as `n_clauses × 2F`.

| Arm | clauses × 2F | throughput [ESTIMATE] | examples/epoch | s/epoch [ESTIMATE] |
|---|---|---|---|---|
| MNIST window (~126 feat, 800 clauses) | 2.0e5 | overhead-bound, ~1–3e5/s | 47.0 M (all positions) | 150–500 |
| — same, 100 sampled positions/image | | | 6.0 M | 20–60 |
| MNIST full canvas (2362 feat, 1000 clauses) | 4.7e6 | ~65 000/s | 47.0 M | ~720 |
| ECG coalesced (4 222 feat, 4 000 shared) | 3.4e7 | ~9 000/s | 1.74 M | ~190 |
| ECG class-owned (K=256 × 100 clauses) | 2.2e8 | ~1 400/s | 1.74 M | ~1 250 |

**Compute is not the blocker for either idea.** Both fit in hours per seed on one GPU. The blockers
are context repetition (E3) and codebook phase (P1). Two practical constraints: contexts must be
built on the fly — materialising 47 M × 2362 bools is ~111 GB — and generation is inherently serial
(784 forward passes per image, 100 per ECG), so batch the *samples*, not the steps.

## 6. Library status: nothing needs to be written

| Need | Provided by |
|---|---|
| Persistent automata across updates | `update()` mutates buffers (`models/base.py`) |
| Vote sums for calibration | `forward()`; `update()` also returns pre-update votes |
| Analytic calibrator | `functional.confidence_from_votes`, `predict_proba(method="linear")` |
| Softmax calibrator | `functional.predict_proba_from_votes(votes, T, temperature)` |
| All-output updates (P7) | `CoalescedTsetlinMachine(multi_label=True)` |
| Thermometer / one-hot / bit-plane context encoding | `data/encoders.py` |
| Sequence model over Boolean channels | `Conv1dTsetlinMachine` |
| Clause inspection for the interpretability claim | `interpret.py` (`rules`, `explain`, `explain_pixel`) |
| Checkpointing | `ModelCheckpoint`; `load_state_dict` post-hook refreshes `include` |

No library gap identified at review time. Per the standing constraint, anything found during
execution goes to `LIBRARY_GAPS.md` as a numbered proposal with a failing reproduction — it is not
patched into `src/`.
