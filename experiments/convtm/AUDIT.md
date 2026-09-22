# Audit log

Maintained by the research engineer. One entry per audit: what was checked, how, what was found.
Run at every gate. An audit that found nothing is still recorded — that is what makes the ones that
found something meaningful.

Standing checks: `src/` guard, leakage, reproducibility re-run, budget matching, the six silent
failure modes in CHARTER.md, crash accounting.

---

# G0 — P0 harness gate (2026-09-20)

## A1 — C1 `src/` guard

`git diff --stat -- src/` at the **start** of P0: empty. At the **end** of P0: empty (see A9).
No file under `src/torchtsetlin/` was opened for writing at any point; the harness reaches library
internals only through read-only attribute access, and every such use is recorded in
`LIBRARY_GAPS.md` (LG-005).

**Found**: nothing. Clean.

## A2 — Leakage: no test data in any selection path

Checked by construction, not only by grep:

1. `run_arm.py` trains with `data.load_boolean(..., splits=("train","val"))`. **The test tensor is
   not loaded while the training loop runs**, so no test accuracy exists to select on.
2. Selection is `record.select_epoch(curve)` = `argmax(val_acc)`, earliest epoch on a tie. It is
   the only selection function in the harness; `run_arm.py` asserts that it agrees with the
   best-validation snapshot taken during the loop (two independent paths, cross-checked).
3. The best-validation automata state is restored from a CPU snapshot, and *only then* is the test
   split loaded and evaluated — once.
4. `record.validate()` **recomputes** `argmax(val_acc)` and refuses to write a record whose
   `selected_epoch` disagrees; it also refuses a record with a `test_acc` key inside any curve
   entry. `record.py`'s self-test asserts both refusals fire (`python code/record.py`).
5. `grep -n "test\|xte\|yte" code/run_arm.py`: every occurrence is either a docstring, the
   `n_test` metadata field, or below the `--- selection is over ---` line at `run_arm.py:138`.
6. The CNN trainer written concurrently (`code/cnn/train.py`) goes through the same
   `record.select_epoch` / `record.finalize` / `record.write`, so the same guard binds it.

**Found**: nothing. The prior programme's `best_test_acc` field does not exist in this schema and
cannot be reconstructed from a record, because per-epoch test accuracy is never computed.

## A3 — Split determinism and cache integrity

* `split_hash() = 8a08ca1533b36c692861b471c316ab3053917403` (`SPLIT_SEED=1234`, 45000/5000,
  500 validation images per class). Recorded in every result record as `data.split_hash`;
  `summarize_p0.py` asserts the hashes agree across seeds.
* Nested subsets verified: `subset(1000) ⊂ subset(5000) ⊂ subset(10000)`, stratified exactly
  (100/500/1000 per class). `python code/data.py` prints this.
* `.cache/cifar10_therm{4,8}.pt` were **re-split, not re-encoded**. `data._boolean_full` checks the
  cached label vectors against the raw torchvision order on every load and raises if they diverge —
  if that check ever fires, every split in the programme is silently wrong, so it is checked rather
  than assumed. It passes for therm4 and therm8.
* `adaptive` (Z=3) was encoded once and cached as `.cache/cifar10_adaptive.pt`.

**Found**: nothing.

## A4 — Silent failure mode: `max_included_literals` does not bind (LG-003)

Reproduction written at `code/repro/lg003.py`. Result recorded in A10 below.
Consequence for the programme: **the `ctm-clausesize` arm cannot be reproduced through
`max_included_literals` under batched feedback** and must use sequential feedback or an explicit
size dial; whichever it uses goes in its registry entry.

## A5 — Silent failure mode: the chunk budget (new — LG-004)

The repo `CLAUDE.md` warns that a wrong `_chunk_elements_per_example` gives "a 4-20x slowdown with
correct answers". Measured here: the **default** `max_chunk_elements = 2**27` already produces that
condition for the configurations this programme is built around. At 10x10 patches, thermometer-4,
batch 50 (`python code/repro/lg004.py`, CPU, no data):

| clauses | `2**27` (default) | `2**29` | `2**31` | `2**33` |
|---|---|---|---|---|
| 640 | 24 | 50 | 50 | 50 |
| 2000 | 9 | 37 | 50 | 50 |
| 8000 | **2** | 9 | 39 | 50 |
| 32000 | **1** | 2 | 10 | 40 |

i.e. `ctm-vanilla` at its PLAN §6.2 configuration executes a mini-batch of 50 as 25 sequential
chunks, on a card with 23 GB free. The throughput consequence is measured in A7. Because it is a
throughput dial and not an algorithm change, **arms are only compared at equal
`max_chunk_elements`**, and the value is recorded in `hp`.

## A6 — Silent failure modes: the remaining four

| mode | check | found |
|---|---|---|
| missing `model.eval()` | every evaluation path (`TMArm.predict`, `diagnostics.firing_stats`, `patch_firing_stats`) sets `eval()`, restores the previous mode in a `finally`, and is the only route to a prediction | clean |
| `.to(cpu, non_blocking=True)` | `grep -rn non_blocking code/` — the only host-bound copy is `TMArm.snapshot`, which uses a plain blocking `.to("cpu", copy=True)`. `code/cnn/arms_cnn.py` uses `non_blocking=self.device.type == "cuda"`, i.e. only for CUDA destinations, which is the correct pattern | clean |
| `position_encoding=True` on translation-invariant data | CIFAR-10 is not one of the translation-invariant toys; the flag is `True` (faithful to Granmo 2019, which makes patch coordinates part of the patch) and is recorded in `hp`, and `--position-encoding false` is available as an ablation | recorded, not a defect |
| batch-size fidelity | `batch_size` is mandatory in `hp` (`validate()` rejects a record without it); the whole programme runs at 50, inside the 10–50 band where batched feedback tracks the sequential algorithm | clean |

## A7 — Throughput and memory calibration (P0.4)

Measured, not estimated. `code/calibrate.py`, RTX 3090 and RTX 3080, 45 000 training images,
batch 50, one suite at a time so a crash costs only the running cell. Raw records:
`results/calib_grid_gpu0.json` (full-scale, 3 epochs per cell), `results/calib_chunk_gpu0.json`
and `results/calib_large_gpu0.json` (probe mode). Merged and derived:
`results/calibration_throughput.json`. Tables regenerated by `python code/summarize_p0.py`
into `results/tables_p0.md` — **do not hand-edit the numbers below** (C4).

**The cost rule.** s/epoch ≈ (total clauses / 1000) × {3.7 (4×4), 7.3 (8×8), 10.1 (10×10)}, to
within 10% from 640 to 16 000 clauses at the library-default chunk budget. Above that the chunk
budget collapses (A5 / LG-004) and the rule over-predicts; price large arms from the probe table.

**Memory is not the constraint; time is.** At the default chunk budget, peak reserved memory is
0.9–1.5 GB everywhere from 640 to 8 000 clauses and 4×4 to 10×10, of which ~0.7 GB is the
resident Boolean dataset. The automata are small: 8 000 clauses at 10×10 thermometer-4 is
19.9 M automata ≈ 80 MB of `ta_state` plus an equal float32 include cache.

**The largest arm that fits the 10 GB card — measured on the 10 GB card.**
[MEASURED: calib_gpu1_3080] Every configuration this programme plans to run fits, including the
largest: **80 000 clauses at 10×10 thermometer-4 — 199 M automata — ran to completion on the
RTX 3080 at 6 678 MB allocated / 8 432 MB reserved**, against the card's 9 871 MB usable
(note: usable, not the nominal 10 240). ~1.4 GB of headroom. Everything up to 40 000 clauses
stays under 4.6 GB reserved.

[MEASURED: calib_gpu1_3080 vs calib_grid_gpu0] **The 3080 is not a small card, it is a slower
one, and the gap widens with size**: 1.09× slower at 2 000 clauses / 4×4 (8.4 vs 7.7 s/epoch),
1.08× at 8 000 / 10×10 (80.8 vs 74.7), 1.31× at 40 000 (430 vs 329) and **1.46× at 80 000
(922 vs 632 s/epoch)** — 15.4 GPU-hours per seed against 10.5. Small arms cost almost nothing
to move to the 3080; large ones cost a third to a half.

Two caveats that decide the device policy anyway:

* **The chunk dial is what breaks it.** At `2**33` the same 80 000-clause arm reserves
  **10 520 MB** and will OOM a 3080. So the 10 GB card can hold the biggest planned arm only at
  the default chunk budget, i.e. only at its slowest setting.
* **Time, not memory, is the reason to put it on the 3090** — 10.5 GPU-hours per seed against
  15.4.

**Device policy: `ctm-vanilla` (80 000 clauses) on the 3090; every other arm in PLAN §9.1 runs
on either card, and no arm in the plan is memory-limited.**

**The cost of the faithful `ctm-vanilla`.** 80 000 clauses (8 000 per class, LG-010 A1) at 10×10
measures **632 s/epoch** at the default chunk budget and **550 s/epoch** at its best setting
(`2**33`, 10.5 GB). At 60 epochs that is **9.2–10.5 GPU-hours per seed, 28–32 GPU-hours for
three seeds** — a quarter to a third of the entire P3 envelope for one arm. `ctm-vanilla-8k`
(the unqualified reading, 8 000 total) is **78 s/epoch**, 1.3 h/seed, 3.9 GPU-h for three.
This is a DR-001 decision, not an engineering one, and it is now priced.

**Booleanization cost.** thermometer-8 doubles the literal space and roughly doubles the cost:
8 000 clauses at 10×10 is 78 s/epoch on therm4 against 152 s/epoch on therm8.

## A8 — Chunk-budget sweep, and why it is now recorded in `hp`

[MEASURED: calib_chunk_gpu0] Raising `max_chunk_elements` above the library default buys
**1.18–1.51×**, saturating between `2**29` and `2**31`, at up to 13× the peak memory:

| configuration | default `2**27` | `2**29` | `2**31` | `2**33` |
|---|---|---|---|---|
| 8000 cl, 4×4 | 29.8 s (chunk 4) | 21.2 s (18) | 19.8 s (50) | 19.9 s (50) |
| 8000 cl, 10×10 | 74.7 s (2) | 58.8 s (9) | 55.9 s (39) | 55.4 s (50) |
| 2000 cl, 10×10 | 20.9 s (9) | 17.7 s (37) | 17.0 s (50) | 16.9 s (50) |
| 80000 cl, 10×10 | 632 s (1) | 640 s (1) | 571 s (4) | 550 s (16) |

**Programme standard: `2**29`** — most of the available speedup for ≤ 1.9 GB peak reserved, so
it still fits the 10 GB card. `2**31` buys a further 0.05–0.11× for 3–6 GB; `2**33` buys nothing
for 9–13 GB. The dial does *not* rescue the 80 000-clause arm: 1.15× at 13× the memory.

It is recorded in `hp.max_chunk_elements` and `hp.chunk_size_at_batch`, and the runner prints a
warning whenever the chunk size is below the batch size, because an arm run at a lower chunk
budget than the arm it is compared against looks expensive for a reason that has nothing to do
with the method. **Open item, carried into P2**: the dial is a pure batch partition
[FACT: `models/base.py:305-322`] and should therefore be accuracy-neutral, but the per-chunk
random draws mean the RNG realisation differs. A 3-seed check against the P0 seed band
(`jobs/p0_chunkneutral.txt`, ~25 GPU-min) settles it; until it lands, **no arm uses a
non-default chunk budget**.

## A9 — LG-003 reproduced here (the clause-size budget)

`python code/repro/lg003.py` — 200 clauses, 4×4 patches, `max_included_literals=8`, 2 epochs on
2 000 CIFAR-10 images:

| feedback | median clause size | mean | p95 | max | budget |
|---|---|---|---|---|---|
| batch 50 | **10.5** | 13.9 | 35.1 | 86 | **exceeded** |
| batch 5 | 8.0 | 7.7 | 13.0 | 23 | held |
| sequential | 7.0 | 7.2 | 12.1 | 18 | held |

The direction reproduces the recorded behaviour exactly. The **magnitude** is smaller than the
9× (budget 8 → median 75) in the repo `CLAUDE.md`, because this is a 2-epoch run on a small
configuration and the overshoot accumulates with training. Recorded honestly rather than
rounded to the known number: an arm that trains for 60 epochs should expect the larger end, and
**every arm relying on the budget must verify per run that it binds**, from the
`diagnostics.clause_len` series the harness already records each epoch.

## A10 — M2, the per-clause match count, and what it costs to get wrong

Added mid-P0 at the orchestrator's request and then re-specified: the decisive statistic is
**conditional**, `P(|M_j| ≥ 5 | |M_j| ≥ 1)`, not the mean. A mean near 1 is equally consistent
with "almost every clause matches one patch" (a count carries nothing an indicator does not, and
the P5 counting-pool branch is dead) and with "a few clauses match many, the rest match none"
(a count carries a great deal). Field names mirror
`code/cnn/arms_cnn.py::CnnArm.match_count_stats` so the TM-side and CNN-side numbers sit in one
schema; the two definitions of "match" differ (a satisfied conjunction here, a thresholded real
activation there) and that difference is recorded in `definition`, not hidden in a key name.

**Process note, recorded because it is the kind of thing that quietly corrupts a result set.**
M2 must be collected *during* a run. Adding it mid-P0 therefore invalidated the seed-noise
records already produced, and rather than mixing two schemas in one band I stopped the queue,
archived those records to `results/superseded/` with a README, and re-ran. Two independent
checks confirm the change is result-neutral, which is what made re-running cheap rather than
alarming:

* the pre-M2 and post-M2 smoke runs of `ctm-small` agree to every digit printed
  (val 0.2026, test 0.1908, selected epoch 2), as do the val-probe and train-probe variants;
* M2 draws no random numbers — it reduces the `(B, P, C)` tensor `_evaluate` already returns,
  and the random matching patch is drawn in `_feedback_counts`, which is the property
  `tests/test_models.py::test_conv_prediction_draws_no_random_patches` pins.

It is probed on **validation**, matching the CNN side, so the two numbers are comparable.
Validation already drives selection, so a read-only probe of it reveals nothing that selection
does not; the test split is still untouched until the training loop has ended.

## A11 — Crash accounting and queue hygiene

* `queue_runner.py` logs every job's return code, checks that the result file exists, and
  prints a failure table with the tail of each failing log at the end of the queue. A queue
  that finishes suspiciously fast now says so.
* Queues stopped deliberately during P0 (twice, for the schema changes in A10) left their
  partial records behind; those were deleted, not left to be picked up by a later `skip-if-
  exists`. The per-GPU lock file was released each time and re-checked before relaunch.
* **`results/preds/` hygiene — the same bug existed on my side, and is now fixed rather than
  cleaned.** The DL expert reported that CNN smoke runs had been writing stray `.npy` files
  there. Checking my own harness found the identical defect and the identical cause:
  `record.finalize` wrote predictions to a module-level `results/preds/`, so a run whose
  **record** went to a temporary directory still dropped its **predictions** into the real
  results tree, where they are indistinguishable from the artefacts of an admissible result.
  Cleaning is not a fix, because the next smoke run recreates them. `run_arm.py` now derives
  the prediction directory from the record's own path (`<dirname of --out>/preds`), and
  `code/smoke.py` asserts that a smoke run adds nothing to `results/preds/`. Verified: the
  directory holds exactly the five band prediction vectors and nothing else.
* **One timing measurement was taken under contention.** The orchestrator authorised two short
  CNN screens on GPU 1 that overlapped `ctm-small seed 0` for ~110 s. Accuracy is unaffected
  (the harness is deterministic in the seed — see A10); `epoch_s` and `wall_s` for that seed are
  inflated. It is re-timed on a clear card in A12 and the cost figures in `ARMS.md` come from
  the calibration grid, which was measured with one suite at a time.

## A12 — Smoke test of every harness path (P0.6)

`python code/smoke.py` — 2 000-image stratified subset, 2 epochs, every registered arm.
**0 failures.** ~30 s for the small arms, ~13 min for the full registry (dominated by
`ctm-vanilla`, whose 80 000 clauses evaluate at 55 img/s on the 3080). What it asserts, and why
each one is there:

| group | checks | result |
|---|---|---|
| data | 45 000/5 000 split; 500 val images per class; train/val disjoint; split hash stable; subsets nested and stratified; every Booleanization in the registry loads as bool; `load_float` + on-device augmentation | pass |
| record | `record.py` self-test — `validate()` must *refuse* a test-selected epoch, a `test_acc` inside a curve entry, a missing capacity key, `n_val != 5000`, empty diagnostics, a malformed prediction hash and a wrong per-class length | pass |
| each arm | runs; record validates; `selected_epoch == argmax(val_acc)`; no `test_acc` in any curve entry; all five diagnostics present; predictions saved; `n_test == 10000`; split hash recorded; skip-if-exists | pass for `ctm-small`, `ctm-coalesced-small`, `ctm-vanilla-8k` |
| model semantics | prediction is **bit-identical across two calls in eval mode** (the RNG-free prediction path); **empty clauses flip between train and eval** — clause output density 1.000 in `train()` against 0.000 in `eval()` at initialisation, which is the failure mode that silently changes predictions if `model.eval()` is missed; snapshot lives on the host; restore reproduces predictions exactly | pass |
| queue runner | builds the right command line; releases its per-GPU lock | pass |

`ctm-vanilla` (80 000 clauses) is smoke-tested at its real clause budget and patch size, with
its per-epoch probes capped through the new `ArmSpec.smoke` field (`--eval-n 500 --m2-n 200`) —
a smoke test exists to prove the path works, and at 55–78 img/s the *default* probes otherwise
dominate it (A7).

**The smoke test earned its keep immediately: it found two harness bugs of my own.**

1. **Predictions escaping into `results/preds/`.** `record.finalize` wrote to a module-level
   directory, so a run whose record went to a temporary directory still dropped its `.npy`
   into the real results tree — the identical defect the DL expert reported on the CNN side.
   Fixed at the cause (predictions now live beside their record) rather than cleaned, and the
   smoke test now asserts that it adds nothing to `results/preds/` (A11).
2. **`--tag -foo` silently unusable.** `queue_runner.py --tag -chunk29` fails with an argparse
   error, because a leading-dash value is read as another option; it must be written
   `--tag=-chunk29`. This would have hit the orchestrator on the first sweep of a single arm.
   Documented in the runner's help and its module docstring, and the smoke test now exercises
   the tag path (which is also what covers `--label`, the flag that stops a sweep of one arm
   from overwriting another's prediction vectors).

## A13 — The seed-noise band (P0.5)

Five seeds of `ctm-small` (2 000 clauses, 4×4, T=160, s=10, batch 50, 30 epochs, thermometer-4)
under the full protocol — validation-selected epoch, test evaluated once. Queue: 5 ok,
0 skipped, 0 failed. Record: `results/calibration_seednoise.json`.

**test accuracy 37.08 ± 0.61 pp** (individual seeds 36.35, 36.67, 37.89, 37.04, 37.47;
range 1.54 pp). Validation 37.09 ± 0.64 pp.

**The number the programme uses**: two 3-seed means must differ by at least
**1.00 percentage points** before the difference is a difference (2 sd × √(2/3)). Anything
smaller is inside the noise of this harness and I will reject a claim that rests on it.

**Where the band comes from.** Test accuracy at a fixed epoch is deliberately never recorded —
that is the leakage guard — so the decomposition is done on validation:

| quantity | sd across 5 seeds |
|---|---|
| val at a **fixed** epoch (the last), selection-free | 0.55 pp |
| val at the **selected** epoch | 0.64 pp |
| test at the selected epoch | 0.61 pp |

So the band is **dominated by genuine seed-to-seed variation**, and `argmax(val_acc)` selection
adds only ~0.09 pp on top. Validation-only selection is not what is making the band wide.

**But the selected epoch itself is close to meaningless here, and that is worth knowing before
P3 sets its epoch budgets.** The validation curve plateaus early and then oscillates: the
selected epochs are 29, 8, 14, 13 and **2**, and between **7 and 15 of the 30 epochs are within
0.5 pp of the best** in every seed, with the first such epoch arriving at epoch 2–9. This arm
is trained roughly three times longer than it needs. Two consequences for P3:

* an epoch budget should be justified against a plateau measurement, not assumed — for arms of
  this size much of the 30-epoch schedule is buying nothing;
* a reported `selected_epoch` should not be read as "the epoch at which the model was best". It
  is the argmax of a nearly flat noisy curve.

I am not changing the selection rule on my own authority — it is the protocol — but a smoothed
or minimum-epoch-floored selector is a cheap thing for the orchestrator to consider, and the
data to decide it is now in every record.


## A14 — Standing checks at G0, and what remains open

| check | result |
|---|---|
| C1 `git diff --stat -- src/` at start | empty |
| C1 `git diff --stat -- src/`, `git status --porcelain -- src/`, untracked files under `src/` at close | all empty |
| leakage (A2) | clean — test tensors are not loaded until selection has ended, and `validate()` refuses a record selected any other way |
| split determinism (A3) | one hash, `8a08ca15…`, recorded in every record; cache order verified against raw CIFAR-10 on every load |
| the six silent failure modes (A4–A6, A9) | two are live and documented as LG-003 and LG-004; four are clean |
| crash accounting (A11) | every return code logged; failures print a table with log tails |
| `results/preds/` hygiene (A11) | clean; smoke output goes to a temporary directory |
| reproducibility re-run (PLAN §7.4.6) | **deferred to P2 by design** — it needs the seed band, which P0 only just produced. First P2 action: re-run one band seed from its recorded `argv` and confirm it lands inside the band. |
| chunk-budget accuracy neutrality | **open** — until it lands, no arm uses a non-default `max_chunk_elements` (A8) |

---

# ROUND-1 addenda

## A15 — The per-GPU lock fired in anger, on its first real opportunity

The orchestrator, having concluded (wrongly, see A16) that the chunk-neutrality queue had died,
launched a second queue on GPU 1. `queue_runner.py` refused it:

```
gpu 1 is already driven by queue pid 1836831 (logs/.gpu1.lock)
```

This is the exact damage the lock was carried over from `experiments/mctm` to prevent — two
queues on one card, OOMing each other, each crashed job writing no result file, so the queue
burns its whole list in seconds and leaves only tracebacks. It prevented it. Recorded because a
guard that has never fired is indistinguishable from a guard that does not work.

## A16 — A wave log cannot distinguish "working" from "all jobs crashed"

Standing check 6 says *"a queue that finishes suspiciously fast is a queue whose jobs all
crashed"*. It has a converse that cost the orchestrator a wrong diagnosis: **a queue that
appears to have produced nothing may simply be working.** `queue_runner.py` prints its per-job
`rc=` line only *after* a job returns, so an in-progress wave and a wave whose every job died
look identical in `logs/<wave>.log` — which is the first log a supervisor checks.

*(No failure occurred. The orchestrator initially reported that the queue had been reaped at
epoch 8 and then retracted it: the run never died, no work was lost, nothing was re-run. The
retraction is recorded rather than the original claim, because an audit log with a fabricated
failure mode in it is worse than one with a gap.)*

**Fixed** — `queue_runner.py` now prints a `[start] <label> seed <k> pid=<n> -> <log>` line
*before* handing off to the subprocess, so the wave log itself distinguishes the two states, and
names the pid and the per-job log that are the real discriminators (alongside the per-job log's
mtime, `ps`, and the lock file). Cheap, and it removes the ambiguity from the log a supervisor
reads first.

## A17 — Pricing the configurations the round actually argues about (5×5 window)

My P0 calibration priced 4×4 to 10×10 windows at up to 80 000 clauses, because that is what
PLAN §9.1 implied. ROUND-1 moved the target ladder onto **real CIFAR-10 results at 5×5 and
60 000–64 000 clauses**, which I had not priced. `results/calib_ladder5x5_gpu0.json`, 9 cells,
RTX 3090, 45 000 train, batch 50, default chunk budget:

| clauses | chunk/50 | therm4 s/epoch | 250 ep | therm8 s/epoch | 250 ep |
|---|---|---|---|---|---|
| 2 000 | 16 / 12 | 9.8 | 0.68 h | 17.8 | 1.24 h |
| 8 000 | 4 / 3 | 32.7 | 2.27 h | 59.4 | 4.12 h |
| 16 000 | 2 | 65.0 | 4.51 h | — | — |
| 32 000 | 1 | 133.8 | 9.29 h | 221.6 | 15.4 h |
| 64 000 | 1 | 262.2 | **18.2 h** | 395.5 | **27.5 h** |

Three consequences, all of which change what the programme can commit to:

1. **The best published single model (75.4%, 64 000 clauses, 250 epochs) costs 18.2–27.5 GPU-h
   per seed, 55–82 for three.** Reproducing T2 is half to two-thirds of the P3 envelope for one
   arm.
2. **T3 is not reproducible on this machine at any budget.** The 82.8% composite is 22
   specialists × 64 000 clauses × 250 epochs ≈ **400–600 GPU-h for one seed**, against a
   whole-programme budget of 220–300. It must be cited, not reproduced. Recommended for
   `DECISIONS.md` as a standing constraint.
3. **The 2 000-clause composite — reportedly 79.5%, above the best single model — costs
   0.68–1.24 GPU-h per specialist, i.e. 15–27 GPU-h for all 22 at one seed.** Affordable, and
   nobody had priced it. Its blockers are engineering (four missing Booleanizations, LG-008) and
   protocol (the transductive `α_t`, O-1), not compute.

Note also that **everything from 8 000 clauses up at 5×5 is chunk-serialised** (chunk 4 at 8 000,
chunk 1 at 32 000). That kills the `2**29` chunk-budget recommendation I made at G0 — it is worth
1.2–1.4× only in the 2 000–8 000 range — and it is what makes the batch-5 route to a binding
clause-size budget cheap (A18).

## A18 — C-5: the recorded curve did not carry the paper's own statistic

`validate()` refuses `test_acc` inside a curve entry — that refusal is the leakage guard — so the
statistic every TM image paper actually reports (a mean over the last 25/100 epochs, or a peak,
of **test** accuracy) was **not computable from a completed run**. Same class of mistake as M2:
cheap before P3, a full re-run after it.

**Fixed without reintroducing the hazard.** `run_arm.py --test-curve-last N` keeps the last N
**automata snapshots** on the host and scores them **after the training loop has ended**. No test
accuracy exists while any decision is being made; `validate()` still refuses test inside the
curve; the results land in `diagnostics.paper_statistics` with the series in
`diagnostics.test_curve`. Cost: N test evaluations + N × `capacity.state_bytes` of host RAM —
**+1.7% on a 250-epoch 64 000-clause arm** (4.5 GB of 121 GB), +7% on a 60-epoch 8 000-clause
arm. Recommended for every `existing`-family arm at N=25.

**My recommendation on the §7.5 tolerance: do not widen it.** Widening a tolerance to absorb a
bias that can be measured is how a reproduction becomes unfalsifiable. Score the reproduction
against the paper's statistic computed our way, report the validation-selected number as the
headline, and let the measured penalty be a reported quantity rather than hidden slack.

## A19 — Four ROUND-1 waves, and the finding P0's diagnostics should have flagged

| wave | question | result | vs the 1.00 pp band |
|---|---|---|---|
| `ctm-small-chunk29`, 3 seeds | is `max_chunk_elements` accuracy-neutral? | 37.15 ± 0.85 against the band's 36.97 ± 0.81 — **+0.18 pp** | **inside — dial released** |
| `ctm-small-protocol`, 3 seeds, GPU 0 | same arm, different card | 37.25 ± 0.38 — **+0.28 pp** | **inside — arms may span both cards** |
| same, `--test-curve-last 25` | how big is our protocol penalty? | −0.04 pp vs a last-25 mean, **+0.25 pp vs a peak** | **4× below the band, 12× below the §7.5 tolerance** |
| `ctm-small-budget32` | does the clause-size cap bind, and what does it cost? | **+7.7 pp accuracy, 5.4× fewer literals/image** | **8× the band** |

**Determinism, restated precisely.** `ctm-small` seed 0 is bit-identical on a re-run on the *same*
card (verified three times), but on a *different* card it diverges at epoch 1 and finishes 1.2 pp
away — floating-point non-associativity in the feedback reductions forks the trajectory. So
**`seed` alone is not a reproducibility key; `device + seed` is** (`env.gpu` already records it),
and PLAN §7.4.6's re-run check must be "lands inside the band", which is what it says — now
measured rather than assumed. The chunk budget forks it the same way, which is why neutrality had
to be judged on 3-seed means rather than on a single re-run.

**The finding I should have escalated at G0 and did not.** Every P0 record carries
`clause_len.median = 175` against 248 available features, with `negation_fraction = 0.51`. I
reported it in A13 as a curiosity about flat validation curves. It was a **defect**: a clause
asserting 175 of 248 features discriminates almost nothing. Capping it at 32 — one constructor
flag, +4% wall-clock — is worth **+7.7 points** and **5.4× fewer included literals per image**.
The diagnostic that detects this was built in P0, collected from the first run, and printed on
every epoch line; what was missing was a threshold at which it is escalated rather than logged.

**Consequence adopted**: I now treat `clause_len.median > 0.5 × n_features` as a *defect
signature*, not an observation, and it is called out in the run's log line and in the round
report rather than left in the JSON. Any arm showing it is mis-tuned until proven otherwise.

**Consequence for others**: `CHARTER.md` silent-failure mode 2 tells three agents that
`max_included_literals` does not bind under batched feedback. Measured, it holds at 16/32/64 and
fails only at 8. The rules card is steering the team away from a mechanism worth 7.7 points.
Flagged to the orchestrator; `LIBRARY_GAPS.md` LG-003 has been narrowed accordingly (it is mine
to correct, `CHARTER.md` is not).

## A20 — `ARMS.md` is a shared file and was clobbered by a concurrent edit

While restoring the ROUND-1 price list I found that several sections I had written to `ARMS.md`
at G0 and during this round were **missing**: the `ctm-vanilla` 80 000-clause correction, the
`ctm-vanilla-8k` row, the seed-noise band section and the measured cost model. The DL expert had
independently added `cnn-ctmshape-*` rows and a screen-records section, writing back a version
based on an earlier read. Some of my edits survived and some did not, interleaved.

**My own contribution to the failure**, which is the part worth recording: I was applying those
edits with Python `str.replace` and printing `"ok"` unconditionally. `str.replace` is a **silent
no-op when the target is absent**, so I reported success for edits that had not applied. That is
the same class of error as a queue whose jobs all crash while the wave log stays clean (A16) —
a success message that is not evidence of success.

**Fixed**: restored by *appending* (collision-safe) and by a single targeted table-row edit that
fails loudly on a missed match, then verified by reading the file back rather than trusting the
writer's exit status. `ARMS.md` now carries a header saying it has multiple authors and that
edits should append rather than rewrite.

**Recommendation to the orchestrator**: `ARMS.md`, `LIBRARY_GAPS.md` and `RISKS.md` all now have
more than one author. Either assign each a single owner, or require append-only edits with a
`##` heading per contribution. `LIBRARY_GAPS.md` already works this way by convention (numbered
LG entries) and has had no collisions; `ARMS.md` has a table that invites rewriting, and did.

## A21 — `T` was inert in the arm that defined the programme's seed band

The tm-theorist found it; I am recording it because it is mine. `ctm-small` was registered with
**`T = 160`** against **2 000 total clauses = 200 per class**, of which only the **100 positive**
ones can raise a class sum, each by at most 1 (the arm is unweighted). **The achievable class sum
is 100, so `T = 160` is unreachable**: `(T − v)/2T` never falls below ~0.48, the margin never
anneals, and `T` does nothing at all. Measured feedback probability 0.478 unconstrained and
0.450 at budget 32 — i.e. essentially the 0.5 of an untuned machine.

This is precisely the failure `ARMS.md` convention 2 exists to prevent, sitting undetected in the
arm that produced the seed band, the chunk-neutrality verdict, the GPU-divergence verdict, the
protocol-penalty measurement and the +7.50 pp clause-budget result.

**Cause of my error**: I set `T = 0.8 × n_clauses_per_class`, treating "clauses per class" as the
vote capacity. For a class-owned model with fixed polarity, half the clauses vote *against* the
class, so the capacity is `n_clauses_per_class / 2`. I had written the correct rule into
`arms.py`'s module docstring — *"`T` must be commensurate with the achievable vote sum"* — and
then implemented a different one.

**Fixed**: `arms._vote_capacity()` now computes the achievable sum explicitly (positive half for
class-owned, the whole pool for coalesced), `_resolve_T` reports `T_ratio` against it, the value
is recorded as `hp.vote_capacity` in every record, and **the runner prints a hard warning when
`T` exceeds the capacity**, which is the guard that would have caught this on the first run:

```
!! T=160 EXCEEDS the achievable class sum (100): the margin is unreachable,
   (T-v)/2T stays near 0.5 for every example and T is inert.
```

`ctm-small` is re-registered at `T = 80`. **What this invalidates**: nothing that is a
*difference measured between two arms sharing the fault* — the chunk, GPU and protocol-penalty
verdicts were all same-config comparisons and stand. What it does put in question is the **size**
of the +7.50 pp clause-budget effect, because a budget cap and an inert margin are two ways of
constraining the same thing. That pair is being re-measured at `T = 80`, alongside the
tm-theorist's registered prediction P3 that the effect is largely a batched-feedback artefact.

## A22 — my HOG encoder was wrong in every parameter that mattered

Written from my own reading of Dalal & Triggs while the specification was being sought: 8 unsigned
orientations, 4×4 cells, 2×2 L2-Hys blocks, thermometer-4, emitted as a **convolutional** 128-plane
7×7 map. The reference scripts specify **18 signed** bins, 12×12 blocks with 4×4 stride, gamma
correction, a single 32×32 window and a plain `>= 0.1` threshold, giving a **flat 5 832-bit
vector**. Every parameter I guessed was wrong, and so was the shape: I had built a convolutional
encoder for a specialist that is not convolutional.

It was caught by the theorist supplying the spec, not by anything in my process — my only defence
was recording the reading as an explicit ambiguity (A5) rather than presenting it as faithful,
which at least meant `admissible_as_reproduction()` was already refusing it.

**Replaced** with the reference configuration verbatim, as named constants; verified to produce
exactly 5 832 features; re-cached (5 s). The arm now runs `patch_size=1` over an
`(N, 5832, 1, 1)` tensor, which expresses a flat TM through the convolutional builder with no
second code path.

**Standing rule adopted** (the orchestrator's, and I would have benefited from it a day earlier):
**read the reference code before declaring an arm a `GAP`** — for this paper family the modal
cause of a shortfall is an undocumented default, not an implementation fault. Three of its
load-bearing settings appear in no table: the budget of 32, the absence of a validation split,
and the transductive `α_t`.

## A23 — Repricing at the per-class convention, and two new hard walls

`[MEASURED: calib_perclass_gpu1, calib_hog40k_gpu1]` I re-priced everything at the corrected
convention rather than scaling my old numbers, and it was worth it twice over: my extrapolation
for the 20 000-total thermometer cell was **135 s/epoch against a measured 180.7** (34% low), and
the HOG arm turned out to have a memory wall I would not have predicted at all.

| total clauses | 5×5 therm8 s/epoch | HOG (flat) s/epoch |
|---|---|---|
| 20 000 | 180.7 | 146.6 (9.5 GB reserved) |
| 40 000 | 345.5 | **OOM (24 GB)** |
| 80 000 | 645.8 | **OOM (24 GB)** |

**Wall 1 — the paper's best cell is unreachable.** 64 000 per class = 640 000 total ≈ 72 GPU-h
per seed at 50 epochs. Cite-only, alongside the 22-specialist composite.

**Wall 2 — HOG is one point, not a curve, and this changes the C-2 design.** HOG holds 11 664
literals per clause, and the feedback accumulator is four `(C, 2F)` tensors on top of `ta_state`
and `include`. The published HOG plateau starts at 32 000 **per class** — 32× beyond what fits —
so **we cannot observe HOG's saturation directly.** The C-2 discriminator as designed (two
capacity curves of opposite shape) is not runnable on this hardware.

**Recommended C-2 redesign, for the orchestrator's decision** (it is a design change forced by a
measurement, not mine to take alone):

1. **Thermometer capacity ladder on the TM** at {20 000, 40 000, 80 000} total — the reachable
   part of the published curve, over which thermometers are predicted to gain ~5.5 points.
   16.3 GPU-h at 1 seed, 32.6 at 2.
2. **`cnn-boolean` on HOG bits** becomes the *primary* HOG evidence rather than a side check.
   It is the theorist's own named falsifier, it tests the encoding-ceiling claim directly, and a
   CNN over 5 832 bits has none of the TM's `C × 2F` memory problem. ~2 GPU-h.
3. **HOG TM at its single reachable point (20 000 total)** as a level comparison against the
   thermometer at the same point — where the published values are 64.2 vs 64.5, i.e. a
   prediction of near-equality that is itself worth testing. 2.04 GPU-h.

**Also recorded**: `compute_dtype=torch.float16` is **not** a safe way to buy headroom here.
`functional.clause_violations` sums up to `n_literals` 0/1 terms (11 664 for HOG) and float16 is
exact on integers only to 2 048, so it would silently miscount violations on precisely the arms
that need the memory. `state_dtype=torch.int16` is safe (states ≤ 255) and halves `ta_state`,
but does not close a 2× gap on its own.

---

## A24 — Three mechanism diagnostics landed **before** the sweep that needs them; the harness was
re-verified across the change

**Date**: 2026-09-21, 20:20–21:00. **Auditor**: research-engineer.
**Trigger**: the theorist's note on `jobs/p3_budget_predictions.txt` — *"If D-TA and D-FIRE land
before this queue runs, the sweep tests the mechanism directly instead of only its accuracy
consequence. If they land after, every arm here has to be re-run to get them."* Same argument as
M2 before P3, and it was right both previous times.

### What was added (all in `code/diagnostics.py`, under the existing `diagnostics` key)

| id | field | what it is | measured cost |
|---|---|---|---|
| **D-TA** | `diagnostics.ta_state` | full `0..2N-1` bincount over `ta_state`, reported as `frac_at_boundary` (`state == N`), `frac_deep` (`>= 1.5N`), `frac_saturated`, `mean/median_depth`, a 16-bin histogram of the included mass | **0.003 s** |
| **D-FIRE** | `diagnostics.firing_rate_eligible` | `P(fire | y is Type-I-eligible for that clause)` — `y == c` for a positive clause of class `c`, `y != c` for a negative one — with `s`, `predicted_1_over_1_plus_s` and `ratio_to_prediction` in the field | **+0.015 s** on a probe pass that already happens |
| **D-ORDER** | `diagnostics.order` | match probability of the top-`k` prefix of each clause's included literals ordered by TA state, `k ∈ {8,16,32,all}`, with `frac_tied_at_k` | **0.373 s** |

[MEASURED: timed on GPU 0 against a *contended* 21.25 s `ctm-small` epoch] the three together are
**1.85% of an epoch**, against the pre-existing probes' 5.2%. The price in
`jobs/p3_block2_s_sweep.txt` (0.107 h/job against the measured 0.104) is therefore ~1 pp
conservative, not optimistic.

### Verification, in the order it was done

1. **`src/` guard**: `git diff --stat -- src/` **empty** at start and at end of the session. C1 holds.
2. **Independent recomputation of every statistic.** Each diagnostic was checked against a
   separate, deliberately naive implementation on the same model and probe:
   - D-TA `frac_at_boundary` vs `(ta_state[included] == N).float().mean()` — exact match, CPU and GPU.
   - D-FIRE `mean` vs a per-clause Python loop building the eligibility mask one clause at a
     time — agrees to float32 rounding (`0.0009382284618914127` vs `…479215741` on CPU; exact on GPU).
   - D-ORDER `k=all` `p_image` vs `model.evaluate_clauses(...).float().mean()` on the same probe —
     exact (`0.0009375` both).
3. **RNG neutrality, the check that actually mattered.** A diagnostic that consumed a CUDA random
   draw would silently shift every subsequent feedback event, because `conv._feedback_counts`
   draws its matching patch from the global CUDA stream. Trained the same arm twice — once with
   the probes off, once with all three on after every update — and compared the sha1 of
   `ta_state`: **identical on CPU and on CUDA**. The only random draw in the new code is the
   D-ORDER clause subsample, taken from an explicit CPU `torch.Generator`.
4. **End-to-end**, GPU: `run_arm.py --arm ctm-small --label diagcheck --epochs 2 --subset 2000`
   completes, record validates, all three fields present in every curve entry and in the summary.
5. **End-to-end**, CPU: the same for `ctm-small-cifar2-b32`; record validates with
   `data.n_classes = 2` and a length-2 `test_acc_per_class`.
6. **`record.py` self-test** extended and passing, including a new pair of cases that pin the
   two-class schema (a 10-entry per-class list on a 2-class record is refused).

### Two things stated plainly, because they are weaknesses

* **`frac_at_boundary` is an upper bound on C5.3's Type-II fringe, not an estimate of it.** A
  literal climbing under Type Ia also passes through state `N`, and with `init="constant"` every
  automaton starts at `N-1`, one step below. The theory's claim is that the fringe is *at most*
  this large and the deep core *at least* `frac_deep`; that is what the field supports.
* **D-ORDER's ordering degenerates exactly where C5.3 says it should.** Ties at `2N-1` are broken
  by index, so within a saturated core the top-`k` prefix is an arbitrary `k`-subset. The field
  reports `frac_tied_at_k` so a reader can see when that is happening — in the first trial runs it
  was 0.96–1.00 on a barely-trained model. **A D-ORDER level with `frac_tied_at_k` near 1 is a
  statement about a random subset of the core, not about "the first k literals admitted".**

### Failure policy, and why it is not a silent-failure hazard

Each diagnostic is wrapped by `diagnostics._safe`: on an exception it records
`{"error": "<type>: <msg>"}` under its own key and prints `!! diagnostic <name> FAILED` into the
job log. This was a deliberate choice, taken because the change lands **mid-queue**: GPU 0's
capacity ladder starts its 80 000-clause point (8.97 GPU-h) hours from now and will pick up the
new code, and an OOM in an observer must not destroy the arm it is observing. The error is in the
record and in the log, so it is visible to crash accounting and to this audit — **it is not
allowed to become a missing field nobody notices.** Standing check for the next gate: grep
`results/*.json` for `"error"` inside `diagnostics`.

### Arms that will NOT have these fields — recorded so the gap is not rediscovered

`ctm-therm5-40k_seed0` (running since 19:53) and everything in `results/` before today. The
80 000 point, the CIFAR-2 pair and all of `p3_block2_s_sweep.txt` / `p3_budget_predictions.txt`
will have them. **The s = 10 column of Block 2's factorial is therefore re-run under new labels**
— see A25.

---

## A25 — Block 2's s = 10 column re-bought, and used as the phase's reproducibility check

**Date**: 2026-09-21. **Auditor**: research-engineer.

The theorist's Block 2 buys four of the six cells of a `{s = 2, 10, 20} × {budget none, 32}`
factorial, on the grounds that "the s = 10 column already exists". It exists **without D-FIRE**,
and D-FIRE is the diagnostic the block's central prediction is stated over: C5.2 says the
unconstrained equilibrium firing rate is pinned at `1/(1+s)`, which moves **7×** across the sweep,
and the theorist's own §5.5.2 says the existing global rate "mixes the two polarities and cannot
test it". A three-point curve with a hole in the middle cannot be read against that.

So `ctm-small-T80-s10-{unc,b32}` were added, +0.62 GPU-h (Block 2: 1.25 → 1.90 h). `--s 10.0` is
the arm default, the seeds and the card (3080) are the same, so **they must reproduce
`ctm-small-T80-{unc,b32}` exactly**:

| label | must reproduce | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| `ctm-small-T80-s10-unc` | `ctm-small-T80-unc` | 37.50 | 37.93 | 39.50 |
| `ctm-small-T80-s10-b32` | `ctm-small-T80-b32` | 47.80 | 48.07 | 47.44 |

This is **PLAN §7.4 / standing check 3 for P3**: a result re-run from its recorded `argv` and
`seed`, on the recorded device, *across a harness change* — `diagnostics.py`, `arms.py`,
`run_arm.py`, `record.py` and `data.py` all changed today. Because the diagnostics are measured
RNG-neutral the tolerance here is **not** the seed band, it is **zero**: the numbers must be
identical to the digit. If they are not, the RNG-neutrality finding in A24 is wrong and every arm
run after 2026-09-21 is suspect. It runs **first** in each seed, so the check lands ~6 minutes
into the wave rather than at the end of it.

---

## A26 — Two contended-GPU probes, declared

Two short jobs were run on GPU 0 while `ctm-therm5-40k_seed0` held the lock: a ~40 s diagnostic
validation and a ~12 s `run_arm.py` end-to-end, plus a ~60 s timing measurement. They were run
**without taking the lock**, deliberately, and they contend.

Declared rather than hidden because the lock exists precisely so that throughput numbers mean
something: **`ctm-therm5-40k_seed0`'s `epoch_s` for epochs 6–8 and its `wall_s` are inflated by
roughly 2 minutes on a 3.75-hour run (<1%).** Its *accuracy* is unaffected. The alternative was to
launch a 1.9 GPU-h unattended sweep on code that had never executed a GPU kernel, which is the
trade this programme has already been bitten by once (A16). The 40 000-clause point is a capacity
*level*, not a throughput measurement; the price list is taken from `calib_perclass_gpu1`, which
is untouched.

---

## A27 — `jobs/p3_budget_predictions.txt`: what I think is mis-priced

Reviewed as the owner of the price list. The theorist priced it from my own records and the
arithmetic is right; three inputs are not.

1. **CIFAR-2 at 0.62 GPU-h is low, probably by ~2×.** The estimate scales `ctm-small`'s measured
   0.104 h/seed by nothing. But the multi-class rule gives feedback to the target class's clauses
   **plus one sampled contrast class's** [FACT: `src/torchtsetlin/models/classifier.py:25-30`],
   i.e. `2 / n_classes` of the pool per example, and `conv._feedback_counts` gathers one `(2F,)`
   literal row per selected `(example, clause)` event [FACT: `models/conv.py:167-176`]. At
   `n_classes = 2` that is the **whole** pool instead of a fifth of it, so the feedback half of an
   epoch grows ~5×. [HYPOTHESIS] 0.15–0.25 h/seed → **0.9–1.5 GPU-h** for the pair at 3 seeds.
   To be replaced by the measured `wall_s` of seed 0.
2. **Block 2 at 1.25 GPU-h buys an incomplete factorial.** A25: the real price of the experiment
   as specified is **1.90 GPU-h**, and the extra 0.62 h is not overhead — it is the middle point
   of the curve and the phase's reproducibility check.
3. **The `F·P` scaling for blocks 3 and 4 is the right model but is quoted more precisely than it
   is known.** The supporting measurement is `calibration_throughput.json`'s therm8/therm4 ratio
   at 10×10 (1.95× for an `F` ratio of 1.96×) — one point, at a different window, at a different
   clause count. My own price list already records that cost is **not** proportional to `F`,
   because per-batch overhead does not scale with it and the gap closes as the arm grows
   (ARMS.md, "Measured price list"); at `ctm-small`'s size the overhead share is *largest*. So
   block 3 (`×0.72`) is likely optimistic and block 4 (`×1.77`) likely pessimistic. Net effect on
   the 3.33 h total is small and in opposite directions; I am not asking for a re-price, only
   recording that those two numbers are ±25%, not ±2%.

Not mis-priced, and worth saying: the **ordering rule** ("the cheapest complete answer to a
registered prediction comes first", blocks 3+4 indivisible, keep Block 2 if the envelope is cut)
is exactly right, and Block 2 is correctly identified as the only block that can overturn our own
headline.

---

## A28 — Three corrections and one clean bill, all made before Block 2 took the card

**Date**: 2026-09-21, 20:50. **Auditor**: research-engineer. Appended rather than folded into
A24–A27, which are append-only.

### 1. Correction to A24's fail-soft scope — it was too broad, and would have made a *worse*
failure mode than the one it fixed

As first written, `_safe` wrapped `firing_stats` and `match_count_stats` as well. Those produce
`firing_rate`, `clauses_used_frac` and `match_count`, all of which `record.validate()` **requires**
for a clause arm. Swallowing a failure there would have converted an immediate crash into an
**invalid record written after hours of training** — the job would run to completion, then be
refused at `write()`, and the whole arm would be lost. That is strictly worse than failing fast.

`_safe` now wraps only the **additive** fields: `ta_state` (D-TA), `order` (D-ORDER) and the
`firing_rate_eligible` sub-computation (D-FIRE). The required fields keep their original
crash-on-failure semantics, unchanged from P0. Re-verified end to end after the change.

### 2. Precise list of arms that will NOT carry the mechanism fields

A24 said "`ctm-therm5-40k_seed0` and everything before today", which is not complete:

| arm | started | has D-TA/D-FIRE/D-ORDER? |
|---|---|---|
| everything in `results/` as of 2026-09-21 20:00 (40 records) | — | **no** |
| `ctm-therm5-40k_seed0` (GPU 0, capacity ladder) | 19:53 | **no** |
| `ctm-therm5-20k-unc_seed0` (GPU 1, the 20 000 budget control) | 19:54 | **no** |
| `ctm-therm5-80k_seed0` (GPU 0, starts ≈ 23:40 when 40k finishes) | — | **yes** |
| everything in `p3_block2_s_sweep.txt`, `p3_cifar2.txt`, `p3_budget_predictions.txt` | — | **yes** |

**The consequence that matters**: THEORY.md B2 (the budget gap should not close with clause count)
is tested by the 20 000 and 80 000 unconstrained points, and **only the 80 000 one will have
D-FIRE**. The C5.2 prediction `f* = 1/(1+s)` therefore has a one-point test at ladder scale, not a
two-point one. Buying it at 20 000 costs a 2.5 GPU-h re-run and I am **not** buying it
unilaterally; it is a 2.5 h decision for the orchestrator, and the free alternative is to read B2
off `clause_len` (which both points do have) and treat the firing-rate half as `ctm-small`-scale
evidence only.

A free observation from the in-flight logs, worth flagging to the theorist and **not** worth a
run: at epoch 13 `ctm-therm5-20k-unc` (20 000 clauses, 5×5 therm8, `s = 5.0`, unconstrained) sits
at median clause length **140 and still rising**, against the budget-32 pre-flight's achieved mean
of 27.8 — i.e. **β ≈ 5 at 20 000 clauses too**, the same ratio as at 2 000. That is B2's own
prediction, seen early, at a different encoding and a different `s`. Its global image firing rate
is **0.154** against `1/(1+s) = 0.167`, a 7% shortfall where `ctm-small` at `s = 10` showed 21% —
consistent with C5.2 and with the polarity-mixing explanation, but the global rate cannot settle it
and this arm will not have D-FIRE.

### 3. Budget matching, all 23 queued cells, checked by resolving the command lines rather than
by reading them

[MEASURED: `hp` of `ctm-small-T80-unc_seed0` vs the resolved config of every job line]

* **Block 2 (6 cells)**: every cell differs from the reference arm in `s` and/or
  `max_included_literals` and in **nothing else** — `n_clauses` 2000, `patch` 4, `stride` 1,
  `T` 80.0, `T_ratio` 0.8, `batch_size` 50, `position_encoding` True, `feedback_mode` batch,
  `weighted` False, `booleanization` therm4, `epochs` 30, `max_chunk_elements` None (hence
  `chunk_size_at_batch` 18 in both). `ctm-small-T80-s10-unc` differs in *nothing at all*, which is
  what makes it the reproducibility check.
* **Blocks 1, 3, 4 (17 cells)**: all resolve; the literal counts the theorist's file asserts are
  correct to the digit — `F = Z·k² + 2(32−k)` gives **166 / 248 / 440** at (patch 3, therm4) /
  (patch 4, therm4) / (patch 4, therm8) and `P = (33−k)²` gives **900 / 841 / 841**. `T` correctly
  stays at 80 across all three encodings: vote capacity depends on clause count alone, not on the
  window or the bit depth.
* **CIFAR-2 pair**: differs from the reference in the label map and in `T` (400 vs 80) — and the
  `T` difference is the *same rule* applied to a 5× larger vote capacity (ARMS.md A10), not a free
  choice.
* **Label collisions**: the four `s2`/`s20` labels appear in both job files by construction, so
  wave 3 will skip them as "exists". The two `s10` labels appear only in mine. No job in either
  file can silently overwrite another's output.

### 4. Clean bills

* `git diff --stat -- src/` **empty** (C1).
* All **40** existing records re-validate under the amended `record.validate()` (the
  `test_acc_per_class` length check now reads `data.n_classes`, defaulting to 10 when absent).
  `data.split_hash()` is unchanged at `8a08ca1533b36c692861b471c316ab3053917403`, the single value
  carried by every record — so adding the CIFAR-2 task did **not** move the split.
* Leakage (standing check 2): `grep -n test code/run_arm.py` — every access to test *data* is at
  line 241 or below, under the `--- selection is over ---` marker at line 237. The new
  diagnostics probe **train** images and train labels (`probe_x`, `probe_y` = `xtr/ytr[:diag_n]`);
  M2 continues to probe validation, which selection already reads. No new path.
* `grep -rn non_blocking code/` — the only hits are `arms.py`'s explicit `# never non_blocking to
  the host` in `snapshot()` and the CNN arms' `non_blocking=self.device.type == "cuda"`, which is
  correct (the hazard is device-to-**host**).
* `model.eval()` is forced and restored in a `try/finally` by all four probing diagnostics
  (`firing_stats`, `match_count_stats`, `order_stats`, `patch_firing_stats`); D-TA reads the
  automata and needs no forward pass.
* Crash accounting: both in-flight jobs alive with fresh log mtimes at 20:48
  (`ctm-therm5-40k` epoch 10/50, `ctm-therm5-20k-unc` epoch 13/50); the CNN queue and the new
  Block 2 queue are both in their poll loops, neither holding the lock.

---

## A29 — Orchestrator rulings on A24–A28, and the reproducibility check made automatic

**Date**: 2026-09-21, 21:00. **Recorded by**: research-engineer, from the orchestrator's ruling.

### Decisions taken (none of them mine to take alone)

1. **B2's D-FIRE gap is NOT bought back.** The 2.5 GPU-h re-run of `ctm-therm5-20k-unc` with the
   mechanism diagnostics is refused, and the reasoning is recorded so it is not relitigated:
   `clause_len` is present at **both** 20 000 and 80 000, so B2's primary test — *does the budget
   gap close with scale* — runs for free; D-FIRE would upgrade only the **firing-rate half**,
   which tests the theorist's C5.2 rather than the thing DR-004 rests on. With the sweep at
   7.4–8.0 h and P2 at ~47 h, that is not where the time goes. **The firing-rate evidence is
   therefore `ctm-small`-scale only, and the report must say so in those words.** If the s-sweep
   makes the firing-rate mechanism load-bearing for the theory section, it is revisited with a
   cheaper targeted arm, not a 2.5 h re-run.
2. **The free B2 observation is promoted to report material**: `ctm-therm5-20k-unc` at median
   clause length **145 and still rising** at epoch 15 against the pre-flight's achieved **27.8**
   is **β ≈ 5 at 20 000 clauses**, matching β ≈ 5 at 2 000 — at a different encoding, a different
   `s`, and a 10× clause count. B2's own prediction, observed at no cost.
3. **Buying the two `s = 10` cells (A25) is endorsed**, both as the missing middle of the
   factorial and as a zero-tolerance reproducibility check. Standing instruction attached:
   **if the re-run does not return 37.50 / 37.93 / 39.50 and 47.80 / 48.07 / 47.44 exactly, the
   queue stops and the orchestrator is told before anything else runs.**
4. **The `_safe` scope correction (A28.1) is to be read in its general form**, and it is the
   entry the orchestrator wanted kept: wrapping the *required* diagnostic fields fail-soft would
   have converted an immediate crash into an **invalid record written after hours of training** —
   *a silent failure manufactured by the safety mechanism itself*. Same family as the `str.replace`
   no-op and the wave log that cannot distinguish "running" from "all crashed": **a mechanism
   whose success signal is not evidence of success.** The grep for `"error"` inside
   `diagnostics` in `results/*.json` is **adopted as a standing gate check**.
5. **ARMS.md A9 goes in the report's method section, not a footnote.** Our CIFAR-2 Δ may be placed
   beside CSC-TM's Δ; **our absolute accuracy may not be placed beside their 94.18 % unless the
   caveat is in the same sentence.**
6. **A26's practice is endorsed and generalised**: short unlocked GPU probes to validate code
   before an unattended launch are correct, and are to be declared every time.
7. **Pricing accepted** at 7.4–8.0 GPU-h for the sweep; the ±25 % on blocks 3/4's `F·P` scaling
   stays a recorded uncertainty rather than a re-price.

### `code/repro_guard.py` — the check of decision 3, automated

Decision 3's instruction fires at roughly **22:56**, six minutes into the wave. **A check that
depends on somebody being awake at 22:56 is not a check**, so it is a detached program
(`setsid nohup python code/repro_guard.py`, pid 2008707, log `logs/repro_guard.log`).

* Compares only what **must** be identical: the full `train_acc` / `val_acc` curve, `selected_epoch`,
  `val_acc`, `test_acc`, `test_predictions_sha` (a sha1 over all 10 000 test predictions — the
  sharpest single discriminator available), plus a free budget-matching cross-check over 14 `hp`
  keys and `env.gpu`.
* Ignores what legitimately differs: `wall_s`, `epoch_s`, `started`, `argv`, the label,
  `hp.task` (a key that did not exist before today) and the new diagnostic fields.
* On mismatch: SIGTERMs the **launcher first** (so wave 2 cannot start), then the queue and its
  running job, writes `logs/REPRO_FAILURE.md` with the differences and a three-step diagnosis
  path, and leaves the partial records in place. On success: `logs/repro_check_PASS.txt`.
* **It cannot touch GPU 0's ladder or the CNN queue.** Before signalling the lock owner it reads
  `/proc/<pid>/cmdline` and requires it to name `p3_block2_s_sweep.txt`. Verified now, while the
  GPU-1 lock is held by `p3_budget_control_20k.txt`: the guard reports `would signal? False`.

**Unit-tested before launch**, against a real record: identity → `[]`; a **1e-9** perturbation of
one curve point → caught; perturbed sha / `selected_epoch` / `hp.s` / `env.gpu` → each caught with
its own message; and `wall_s`, `argv`, `arm`, `started`, `hp.task` and injected `ta_state` fields
all → `[]`. The tolerance is zero and the guard demonstrably enforces zero.

**Why zero and not the seed band**: the licence is the RNG-neutrality result of A24, not the
protocol. The band exists because two seeds differ; here the seed, the device, the configuration
and the code path are identical, so any difference at all is a defect.

---

## A30 — Block 2 landed. The reproducibility check passed; the verdict; and the mechanism
diagnostics paid for themselves twice

**Date**: 2026-09-22, 00:40. **Auditor**: research-engineer.
All numbers below are generated by `python code/summarize_budget.py --family s` (C4). Four cells
of seed 2 were still running at the time of writing; seed counts are stated per cell.

### 1. Standing check 3 — reproducibility across the 2026-09-21 harness change: **PASS**

`logs/repro_guard.log`, five of six pairs verified at the moment each landed, zero tolerance:

| re-run | reproduced | test | sha1 over 10 000 predictions |
|---|---|---|---|
| `ctm-small-T80-s10-unc` seed 0 | `ctm-small-T80-unc` seed 0 | 37.50 | `1fccbda1f25d` |
| `ctm-small-T80-s10-b32` seed 0 | `ctm-small-T80-b32` seed 0 | 47.80 | `2ebc312a27c6` |
| `ctm-small-T80-s10-unc` seed 1 | ... seed 1 | 37.93 | `99f051fa90e4` |
| `ctm-small-T80-s10-b32` seed 1 | ... seed 1 | 48.07 | `0ab98320789f` |
| `ctm-small-T80-s10-unc` seed 2 | ... seed 2 | 39.50 | `0693f6b5d37c` |

**Identical on the full train/val curve, `selected_epoch`, `val_acc`, `test_acc`, all 14 matched
`hp` keys, `env.gpu`, and the sha1 of every one of the 10 000 test predictions.** `diagnostics.py`,
`arms.py`, `data.py`, `run_arm.py` and `record.py` all changed on 2026-09-21 and **not one of them
perturbs training.** No `REPRO_FAILURE.md`; the sixth pair (`s10-b32` seed 2) is queued.

**Crash accounting**: 14 finished jobs, **14 × `rc=0`**, 14 result files, 0 failures. Wall 369–376 s
against the pre-change arms' 361–372 s, i.e. the three new diagnostics cost **1.5–2.5 % in situ**,
consistent with the 1.85 % measured on the bench (A24).

### 2. The pre-registered verdict (DR-006 Decision 1): **`s` does NOT substitute for the budget**

    ctm-small-T80-s2-unc  36.78 +- 0.99 (3 seeds)
    ctm-small-T80-s10-b32 47.93 +- 0.19 (2 seeds) [= ctm-small-T80-b32 47.77 +- 0.32, 3 seeds]
    gap +11.15 pp against a 1.00 pp band  ==>  DR-004 STANDS.

The falsifier failed in the safe direction and then some: `s = 2` unconstrained is **worse** than
`s = 10` unconstrained (36.78 vs 38.31), not better. The repository's own prior note
(`MEMORY.md → tsetlin-clause-budget-batched`, *"use `s` instead"*) is **measured wrong**, and it was
written before any of this existed.

### 3. The `L₃₀` question — the answer is neither branch, and the reason is the finding

I framed it as: *`s=2` reaching budget-32's accuracy at a longer `L₃₀` means `s` is the better dial;
at a shorter `L₃₀` means the cap selects better literals.* Neither happened, because **`s` is barely
a length dial at all**:

| `s` | `L₃₀` unconstrained | `L₃₀` at budget 32 | β | Δ (budget effect) | n |
|---|---|---|---|---|---|
| 2 | **160.40 ± 0.49** | 31.95 ± 0.01 | 5.02 | **+11.47 pp** | 3 / 2 |
| 10 | **170.36 ± 0.28** | 32.15 ± 0.03 | 5.30 | **+9.62 pp** | 3 / 2 |
| 20 | **172.78 ± 1.35** | 32.36 ± 0.01 | 5.34 | **+9.42 pp** | 2 / 2 |

A **10× range of `s`** — a 7× movement in the admission threshold `1/(1+s)`, 0.333 → 0.048 — moves
the unconstrained equilibrium length by **12.4 literals, 7.7 %**. It costs 1.53 pp of accuracy to
buy those 10 literals at `s = 2`. `s` thresholds literal *frequency*; it does not control literal
*count*, and this is the measurement that says so.

### 4. THEORY B1 — direction right, magnitude wrong by 5×, and the sweep cannot test the bloat law

`[FACT: THEORY.md 5.5.6 B1]` predicted *"at `s = 2` … `L_unconstrained` should fall **below 32**,
β ≈ 1, and **Δ should vanish into the seed band**"*. Measured: `L_unc(s=2) = 160.40`, **β = 5.02**,
**Δ = +11.47 pp** — the *largest* Δ in the sweep. The monotone rise of `L_unc` with `s` is
confirmed; everything quantitative about it is not.

B1's second falsifier — *"Δ(s=20) < Δ(s=10) while L_unc(s=20) > L_unc(s=10)"* — is **literally
satisfied** (9.42 < 9.62, 172.78 > 170.36) and **I decline to call it fired**: 0.20 pp is a fifth of
the seed band and both cells are n = 2. What *is* measurable is the opposite of B1's trend — Δ is
largest at the smallest `s`, and the `s=2`-vs-`s=10` difference of 1.85 pp is outside the band.

**The consequence that matters for the report**: β spans only **5.02–5.34** across the whole sweep.
The `s` axis has **no leverage on β**, so *these runs cannot test the bloat law at all* — a
near-constant Δ against a near-constant β is consistent with it and with almost anything else. The
theorist priced Block 2 as a test of B1; it is a decisive test of the `s`-substitution risk and a
**non-test** of the bloat law. `ctm-small-cifar2-{unc,b32}` (wave 2) is therefore not one of two
tests of the bloat law — **it is the only one we have designed.**

### 5. D-FIRE — C5.2 confirmed, and the old diagnostic alone would have got it WRONG

| `s` | `1/(1+s)` | **eligible** rate | ratio | *global* rate | ratio |
|---|---|---|---|---|---|
| 2 | 0.3333 | 0.3757 | **1.127** | 0.3069 | 0.921 |
| 10 | 0.0909 | 0.1016 | **1.117** | 0.0718 | 0.790 |
| 20 | 0.0476 | 0.0531 | **1.115** | 0.0364 | 0.763 |

The prediction moves **7×** across the sweep and the eligible rate tracks it with a ratio flat to
**1.1 %** (1.127 / 1.117 / 1.115, per-seed sd ≤ 0.02). That is a confirmation of C5.2, and the
residual is a *constant* 12 % excess — systematic, not noise. `[HYPOTHESIS]` the ungated Type II
inflow, which the first-order Type-I fixed point omits.

**The global rate drifts 0.921 → 0.763 over the same sweep.** Read with the pre-existing diagnostic
alone, this experiment would have reported a firing rate 8–24 % *below* prediction and *diverging
with `s`* — i.e. it would have **falsified C5.2 incorrectly.** This is the concrete answer to the
question of whether landing the diagnostics before the sweep was worth it: it was not a matter of
avoiding a re-run, it was a matter of avoiding a **wrong published conclusion**.

Under the cap the pinning breaks, increasingly with `s`: ratios **1.76 / 2.35 / 2.86** at budget 32.
The cap holds the clause far above the firing rate its own feedback loop would settle at.

### 6. D-TA — C5.3's population claim is **falsified**, and the replacement is better

C5.3 predicted a **bimodal** TA-state histogram — a deep core at `2N−1` plus a Type-II fringe
parked at exactly `N` — with *"a budget strips the fringe first"*.

| | `frac_at_boundary` (== N) | `frac_deep` (≥ 1.5N) | `frac_saturated` (== 2N−1) | `mean_depth` |
|---|---|---|---|---|
| unconstrained `s=10` | **0.0021** | 0.778 | 0.028 | 0.694 |
| budget 32, `s=10` | **0.0335** | 0.169 | 0.001 | 0.273 |

* **There is no population at the boundary.** 0.2 % unconstrained. The distinguishable Type-II
  fringe does not exist at equilibrium.
* **The histograms are unimodal and monotone, in opposite directions.** Unconstrained rises
  0.017 → 0.169 across the 16 depth bins; budgeted falls 0.178 → 0.007.
* **The cap therefore does not strip a shallow fringe off a deep core — it inverts the entire depth
  distribution**, and the budgeted arm has *16× more* boundary mass than the unconstrained one, the
  opposite of the prediction. `[HYPOTHESIS]` because every Type Ia event of an oversize clause is
  converted to Type Ib `[FACT: functional.py:232-242]`, a budgeted clause is under permanent net
  downward pressure on *all* its included literals, so its whole membership churns just above the
  include boundary instead of ratcheting to the ceiling.

Note what this does **not** overturn: CSC-TM's own remark that an over-budget clause's included
literals sit *"in the middle states"* `[FACT: arXiv:2301.08190]` is **confirmed** — that is exactly
the budgeted arm's shallow distribution. What is refuted is the inference drawn from it, that the
cap's benefit comes from *shedding* that shallow population. The shallow population is what the
capped clause **is made of**.

### 7. D-ORDER — the tie flag I built in is load-bearing, and it disqualifies the headline statistic

`p_image_ratio_32_over_all` reads 2.41 / 7.17 / 11.69 on the unconstrained arms and 1.01–1.08 on the
budgeted ones. **It must not be quoted**, for the reason the field was designed to expose:
`frac_tied_at_k` at k = 32 is **0.72 / 0.70 / 0.61** on the unconstrained arms — for two thirds of
clauses the 32nd and 33rd TA states are equal, so the "top 32" is an *arbitrary* 32-subset of a
160–173-literal clause. The ratio is then mostly a statement about 32 versus 170, not about
inclusion order. (On the budgeted arm the tie fraction is 0.12 at k = 8, so *there* the ordering is
real.)

**The statistic that survives the caveat** is the length-normalised per-literal selectivity
`−ln q = −ln(p_patch)/L`, which is an average over the prefix rather than a claim about the identity
of its last member:

| prefix | `s=10` unconstrained | `s=10` budget 32 |
|---|---|---|
| top 8 | 0.2284 | 0.3230 |
| top 16 | 0.1798 | 0.2740 |
| top 32 | 0.1310 | 0.2099 |
| whole clause | **0.0442** (L = 170.6) | **0.2086** (L = 32.1) |

Within a single trained unconstrained clause, per-literal selectivity **decays 5.2×** from the top 8
to the full clause and does so monotonically; under the cap it decays only **1.55×**. That is
THEORY 5.5.3's claim — late-arriving literals contribute progressively less — in the strongest form
this diagnostic can support, and it is **3.3× weaker under the cap**. Free cross-arm reading, also
tie-robust because it is length-normalised: at a matched top-8 prefix the budgeted clause is **41 %
more selective per literal** (0.3230 vs 0.2284), which is the top-*b*-selection versus
low-pass-filter distinction, measured.

**Internal consistency check on D-ORDER**: for the budgeted arm k = 32 and k = all agree to
`p_image` 0.1533 vs 0.1444 and `−ln q` 0.2099 vs 0.2086 at `mean_len` 31.5 vs 32.1 — they should,
because the clause *is* 32 literals, and they do.

### 8. B2, free, from the budget control that finished at 23:10 — and it is stronger than expected

[MEASURED: `ctm-therm5-20k-unc_seed0` vs `ctm-therm5-preflight_seed0`, **1 seed each**] 20 000 total
clauses, 5×5 therm8, `s = 5`, `T = 3000`, weighted, 50 epochs — matched on every one of those:

    unconstrained  test 58.67  val 61.28  selected epoch 13/50  L_last 194.18  L_selected 147.14
    budget 32      test 65.10  val 65.82  selected epoch 45/50  L_last  27.77  L_selected  27.83
    Delta = +6.43 pp    beta = 6.99   (against +9.46 and beta 5.30 at 2 000 clauses / therm4 / s=10)

**The gap does not close with scale** (B2's falsifier was "the gap at 80 000 is less than half the
gap at 20 000"; the 2 000 → 20 000 segment loses a third of it) and **β grew, 5.30 → 6.99**.
Caveat stated up front: the two scales differ in encoding, window, `s`, `T`, weighting and epochs,
so this is DR-003's "has the gap changed" test, not a clean scale ladder, and it is n = 1.

**The ratchet, at ten times the clause count and now visibly harmful:**

| epoch | 1 | 5 | **13** | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| val % | 49.66 | 57.82 | **61.28** | 60.14 | 59.52 | 59.30 | 58.98 |
| `L` | 63.49 | 108.58 | **147.14** | 163.36 | 179.09 | 189.10 | 194.18 |
| fire | 0.1660 | 0.1669 | 0.1674 | 0.1694 | 0.1679 | 0.1695 | 0.1695 |

From the selected epoch to the last, the clause **grows 32 %** (147 → 194) while the firing rate
moves **+1.3 %** and validation **falls 2.30 pp**. At 2 000 clauses the late literals were neutral
(+0.32 pp, inside the band); **at 20 000 they are actively harmful.** This is the clearest single
picture of the inclusion ratchet the programme has, and it cost nothing.

### 9. Cost note for the 80 000 point

`--test-curve-last 25` on the 40 000-clause arm added **~25 minutes** after epoch 50 (25 restore +
full-test evaluations). At 80 000 expect **~45–50 minutes** on top of the 8.97 h estimate. Not a
fault — the protocol control is worth it — but it is not in the price list and the ladder's ETA
should carry it.

### A30 — ACTIONS REQUIRED (raised here because `SendMessage` is disabled and my one hand-off is spent)

Numbers persisted at `results/summary_block2.json` and `results/summary_budget_family.json`,
regenerable with `python code/summarize_budget.py --family s|all --json <path>`. Nothing below is
hand-typed.

**For the orchestrator (`DECISIONS.md` is yours; I do not edit it):**

1. **DR-006 Decision 1's falsifier has resolved in DR-004's favour.** The pre-registered rule was
   *"if `ctm-small-T80-s2-unc` lands within the 1.00 pp band of `ctm-small-T80-b32` (47.77),
   DR-004's claim is rewritten"*. It landed at **36.78 ± 0.99, a gap of +11.15 pp** — eleven times
   the band, and on the far side of the unconstrained baseline rather than the budgeted one.
   **DR-004's wording stands unchanged**: the clause-size budget is worth ~9 points, and it is not
   one of two dials.
2. **Report guard, to be enforced like ARMS.md A9**: `diagnostics.order.p_image_ratio_32_over_all`
   **may not appear in the report for an unconstrained arm**. Its own tie flag
   (`frac_tied_at_k` = 0.61–0.72 at k = 32) says the prefix is an arbitrary subset there. The
   length-normalised `−ln q` decay is the version that survives and is what should be quoted.
3. **Ladder ETA correction**: `--test-curve-last 25` costs **~25 min** after epoch 50 at 40 000
   clauses and should be budgeted at **~45–50 min** at 80 000, on top of the 8.97 h. Not a fault,
   not in the price list.
4. **The CIFAR-2 pair just became more valuable, not less.** Block 2 was priced as a test of B1;
   it turns out the `s` axis moves β by only 5.02 → 5.34 and therefore **cannot test the bloat law
   at all**. `ctm-small-cifar2-{unc,b32}` (wave 2, next on GPU 1) is now the *only* designed test
   of the explanation the report gives for +9.46 against CSC-TM's +0.06.

**For the tm-theorist — three corrections to THEORY §5.5, all measured, none of them mine to write:**

* **B1 (§5.5.6)**: the direction is confirmed (`L_unc` rises monotonically with `s`) and every
  quantity is wrong. Predicted `L_unc(s=2) < 32`, β ≈ 1, Δ ≈ 0; measured **160.40, β = 5.02,
  Δ = +11.47 pp** — the largest Δ in the sweep. B1's second falsifier is *literally* satisfied
  (Δ(s=20) 9.42 < Δ(s=10) 9.62 while `L_unc` rose) and I have **declined to call it fired** at
  0.20 pp on n = 2 cells; the record says so explicitly so that nobody later reads the decision as
  the theory surviving a test it did not take.
* **C5.3 (§5.5.2)**: the *population* claim is falsified. `frac_at_boundary` is **0.0021**
  unconstrained and **0.0335** budgeted — there is no Type-II fringe at `N`, and the cap **inverts**
  the depth distribution instead of stripping a shallow layer off a deep core. CSC-TM's own
  "middle states" remark is confirmed; the inference drawn from it is not. Suggested replacement,
  offered as a `[HYPOTHESIS]` for the theorist to take or reject: the cap's conversion of every
  Type Ia event into Type Ib on an oversize clause `[FACT: functional.py:232-242]` keeps the whole
  membership churning just above the include boundary.
* **C5.2 (§5.5.2) is confirmed and should be stated at full strength** — eligible firing rate /
  `1/(1+s)` = **1.127 / 1.117 / 1.115** across a 7× movement, per-seed sd ≤ 0.02, with a *constant*
  12 % excess. Please state in the same paragraph that the **global** rate drifts 0.921 → 0.763
  over the same sweep, i.e. that this prediction is only testable with D-FIRE. That sentence is the
  programme's evidence that a diagnostic can be the difference between a right and a wrong
  conclusion, not merely between measuring and re-running.

**Still open when this was written** (00:45): four seed-2 cells of Block 2 (~01:05), then wave 2
(CIFAR-2), then wave 3 (blocks 1, 3, 4) at equal priority with the CNN queue. `code/repro_guard.py`
is still watching for the sixth reproducibility pair and will stop the queue and write
`logs/REPRO_FAILURE.md` if it disagrees. Re-run `python code/summarize_budget.py --family s` once
seed 2 completes so every cell carries 3 seeds; **no conclusion above depends on it** — the decisive
cell already has 3.
