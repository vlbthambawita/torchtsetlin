# Library gaps found by the generative-TM programme

Per constraint C2 these are **reported, not patched**. Nothing under `src/torchtsetlin` has been
modified (`git diff --stat src/` is empty). Each entry carries a reproduction and an impact
estimate; patch sketches are sketches.

---

## LG-G01 — batched feedback collapses above batch ≈ 64, silently, killing most of the clause pool

**Severity: high for probabilistic use, moderate for classification.** Found in gate G0
(`FINDINGS_G0.md` Finding 5).

### What happens

With `feedback_mode="batch"` (the default), raising the mini-batch from 60 to 70 takes a
well-calibrated machine to a dead one in a single step. Roughly **two thirds of the clause pool
becomes empty** (`include_count == 0`), the vote sum stops varying with the input, and the machine
emits a constant probability. No exception, no warning.

Measured at C=3200 clauses/class, T=100, s=10, F=10, holding `update()` calls fixed at 20 000 so
batch size is not confounded with learning steps:

| batch | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|
| calibration mae | 0.044 | 0.055 | **0.499** | **0.500** | **0.500** | 0.388 |
| empty clauses | 0 | 0 | **1067** | **2153** | **2518** | **2825** |

### It is not the obvious things

* **Not step starvation.** The naive experiment holds total *examples* fixed, which also cuts the
  number of `update()` calls — that confound is removed above by fixing update count instead, and
  the cliff survives.
* **Not `n_states`.** Swept 32 / 64 / 128 / 256 / 512: the cliff is between 40 and 80 in *every*
  case, with near-identical damage (2152–2279 empty clauses at batch 80). This refutes the natural
  hypothesis that large accumulated Type I increments overshoot the include boundary, which would
  predict a cliff proportional to `n_states`.
* **Not the clause budget.** Identical at C=800, 3200 and 12800.
* **It *is* `s`.** The threshold moves with specificity — ~70 at `s=10`, ~100 at `s=20`, ~500 at
  `s=40` — though not proportionally, so it cannot be reduced to a formula. This points at the
  forget channel, whose magnitude is `binomial(n_false + n_ib, 1/s)`.
* **Near the threshold it is stochastic.** At C=12800, T=400, s=40, batch 240, two of three seeds
  trained cleanly (mae 0.028, 0 empty) and the third collapsed (mae 0.492, 1163 empty). Any
  single-seed check of "is this batch size safe" is unreliable.

### Where it probably comes from

`functional.apply_feedback` treats the three feedback channels asymmetrically. Type II is clamped to
the distance to the boundary —

```python
room = torch.sub(N, state).clamp_(min=0)
inc2 = torch.minimum(n2, room.to(n2.dtype), out=n2)   # "never beyond the include boundary"
```

— while Type Ia memorisation and Type Ib forgetting are not:

```python
inc = _binomial(n_true, p_mem, generator)     # p_mem == 1.0 when boost_true_positive
n_false.add_(n_ib.unsqueeze(1))
dec = _binomial(n_false, p_forget, generator)
delta = inc.sub_(dec).add_(inc2)
```

With `boost_true_positive=True` (the default) `inc` is the *raw count*, so the applied increment
grows linearly with the batch, whereas the sequential algorithm re-evaluates clause matching after
every example. Batch mode therefore takes one large explicit step computed entirely at the
pre-update operating point. That is the usual setting for an explicit-integration instability, and
the observed independence from `n_states` argues the instability is in the *gain* (how much
feedback is applied per commit relative to how fast the clause's matching set changes), not in the
state range. **This explanation is not yet confirmed** — the `n_states` test refuted the first
mechanism hypothesis, and no further mechanism test has been run.

### Impact

`CLAUDE.md` already warns that fidelity "degrades as batch size grows — batch 10–50 matches
sequential on Noisy XOR, 200 degrades noticeably". The measurement here says the effect is (a) a
cliff, not a gradient, (b) located at ~64, (c) invariant to `n_states` and clause budget, and
(d) **detectable for free** via the empty-clause count. A user following the documented guidance and
picking batch 100 for throughput gets a silently dead model. For classification the damage is partly
masked by argmax; for anything that reads the vote sum as a probability it is total.

### Suggested response (sketch — not applied)

1. **Cheapest and most valuable:** warn at construction or on the first `update()` when
   `batch > 64`, and expose the empty-clause count as a first-class diagnostic (e.g. an
   `empty_clause_fraction` property plus a one-line mention in the docs). This alone converts a
   silent failure into a visible one.
2. Tighten the `CLAUDE.md` note from "degrades noticeably at 200" to the measured cliff.
3. Possible fix to test: clamp the Type Ia/Ib deltas per commit the way Type II already is, or
   scale accumulated counts by a factor that keeps the per-commit step bounded. Both change learning
   dynamics and would need the existing test suite plus a calibration check before adoption.
4. A regression test in the spirit of `tests/test_models.py`: train on a two-context Bernoulli
   problem at batch 32 and batch 128, and assert no clause is empty.

### Reproduction

```bash
cd experiments/generation/code/g0
python run_g0.py --sweep p02 --device cuda:0 --seeds 0 --clauses 3200 --T 100 \
    --batch-values 50 60 70 80 --fixed-updates 20000
```
Records land in `experiments/generation/results/g0/`; the `empty_clauses` field is the signal.

---

## LG-G02 — no supported way to read a clause's raw (unclamped) vote

**Severity: low (ergonomics).** `forward()` returns vote sums already clamped to `[-T, T]`
(`_votes(out)` with `clamp=True`). Gate G0 needed the *unclamped* amplitude to discover that
calibration is governed by `|v|max / T` (Finding 1) — the single most important tuning rule the gate
produced — and had to reach into private API to get it:

```python
out, _ = m._evaluate(m._encode(m._prepare(X)), empty_value=False)
raw = m._votes(out, clamp=False)[:, 1]
```

Since `_votes` already takes a `clamp` argument, a public wrapper (e.g. `forward(x, clamp=False)` or
a `raw_votes(x)` method) would cost almost nothing and makes the `|v|max/T ≈ 1` check — which we now
recommend as standard practice before trusting any TM probability — a supported operation.

---

## Non-gaps (checked, working as documented)

* `predict_proba(method="linear")` and `confidence_from_votes` implement exactly the analytic
  calibrator the fixed point predicts, and it is accurate to mae 0.018 at a tuned setting.
* `CoalescedTsetlinMachine(multi_label=True)` behaves as documented and is the right model for
  next-token distributions (Finding 8).
* The empty-clause semantics documented in `CLAUDE.md` are correct, and cause no train/eval
  discrepancy once no clause is empty (Finding 9).
