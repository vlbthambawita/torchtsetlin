# Team charter — read this first, every time

One page. The full contract is `PLAN.md`; this is what you must not get wrong.

## The six hard constraints

| # | Rule |
|---|---|
| **C1** | **`src/torchtsetlin/**` is READ-ONLY** until the user manually confirms otherwise. All code lives in `experiments/convtm/`. `git diff --stat -- src/` must be empty at every gate. |
| **C2** | Missing or wrong library behaviour is **reported in `LIBRARY_GAPS.md`, never fixed**: symptom, minimal reproduction, impact, fix *sketch*. |
| **C3** | **No method is chosen without experimental results.** See claim tags below. |
| **C4** | Every number in the report is generated from `results/*.json`. Nothing is hand-typed. |
| **C5** | Negative results are findings. They are reported with the same care as positive ones. |
| **C6** | No result is admissible without `argv`, `seed`, `git_sha` and `env` in its record. |

## Claim tags — use them in everything you write

| Tag | Means | Can justify a decision? |
|---|---|---|
| `[FACT: <paper section / file:line>]` | Citable | Yes, for what exists or was published |
| `[MEASURED: <result-id>]` | A record in `results/` or `experiments/mctm/results/` | **Yes — the only tag that can justify choosing a method** |
| `[HYPOTHESIS]` | Everything else, however plausible | No |

Untagged text is read as `[HYPOTHESIS]`. A decision record with no `[MEASURED]` entry is rejected.

## Experimental protocol (Section 7 of PLAN.md, compressed)

- **Split**: CIFAR-10, fixed stratified **45 000 train / 5 000 val**, `split_seed=1234`, identical for
  every arm, TM and CNN alike. Official 10 000 test.
- **Selection is on validation. Always.** Test is evaluated **once** per (arm, seed), at the epoch
  validation selected. Per-epoch test may be logged for curves and may never influence any choice.
  (The prior `experiments/mctm` programme reported `best_test_acc`; this programme does not.)
- **Seeds**: 3 for any arm in a comparison, 5 for a finalist and whatever it claims to beat. Report
  mean ± sd over seeds; never a single seed in a headline.
- **Seed noise band** is measured in P0. A difference smaller than the band is not a difference.
- **Budget matching**: any "A beats B" states the axis, and at least two of — clause budget, automata
  budget (`n_clauses_total x 2 x n_features`), literals evaluated per image, wall-clock, data budget.
- **Significance**: paired McNemar on shared test predictions + bootstrap CI + across-seed spread.
  All three reported; superiority needs the seed intervals disjoint *and* the paired test to agree.
- **Reproduction tolerance** for a literature arm: within 3 points of the paper (or its own spread,
  whichever is larger), else marked `GAP` with a documented cause.

## Silent failure modes in this stack (from the repo CLAUDE.md — all have bitten before)

1. Missing `model.eval()` — empty clauses evaluate True while training, False when predicting.
2. `max_included_literals` **binds at the budgets papers actually use** — this entry previously said
   it does not, and that was measurably too strong. `[MEASURED: ctm-small-budget32, 3 seeds]` the
   overshoot is **absolute (+2.5 literals) and only at budget 8**: median clause size lands at
   10.5 / 15.0 / 20.0 / 25.0 for budgets 8 / 16 / 32 / 64 at batch 50. No sequential feedback needed.
   **Budget 32 is worth +7.50 pp on CIFAR-10** (44.58 +- 0.55 vs 37.08 +- 0.61 unconstrained, seed
   intervals disjoint) at 4.8x fewer literals per image and +1.7% wall-clock. Treat a clause-size
   budget as a **default on every arm**, not a hazard. Still true: at budget 8 it overshoots, and
   the repo's `CLAUDE.md` and the older `experiments/mctm` notes state the strong form — they were
   derived from a budget-8 measurement and do not generalise.
3. `_chunk_elements_per_example` must scale with `P*C` for conv models — wrong gives a 4-20x
   slowdown with correct answers, i.e. a wrong throughput number.
4. Any `.to(cpu, non_blocking=True)` silently Booleanizes garbage (~40% wrong bits, no error).
5. `position_encoding=True` pins clauses to locations on translation-invariant tasks.
6. Batched feedback fidelity degrades with batch size (10-50 fine, 200 degrades). Compare arms only
   at equal batch size.
7. A queue that finishes suspiciously fast is a queue whose jobs all crashed. Check return codes.
   The converse also holds: `queue_runner` prints `rc=` only *after* a job returns, so an in-progress
   wave and a wave whose jobs all died look identical **in the wave log**. Discriminate with the
   per-job log's mtime, `ps`, and the GPU lock file.
8. **Never set `compute_dtype=float16` to buy memory headroom.** `functional.clause_violations` sums
   up to `n_literals` 0/1 terms — 11 664 for the HOG specialist — and float16 represents integers
   exactly only to 2 048. It would **silently miscount violations on exactly the arms big enough to
   need the memory**. `state_dtype=torch.int16` is safe but only buys 2x.
9. **Large-literal arms OOM through the feedback accumulator, not `ta_state`.** The accumulator is
   four `(C, 2F)` tensors on top of `ta_state` and `include`, so memory scales with
   *clauses x literals*. The HOG specialist (F = 5 832) reserves 9.5 GB at 20 000 clauses and OOMs a
   24 GB card at 40 000. Price memory from `C x F`, not from clause count.

## Where things go

`LITERATURE_TM.md` `LITERATURE_CNN.md` `THEORY.md` `MECHANISMS.md` `LIMITATIONS.md` `CANDIDATES.md`
`ARMS.md` `AUDIT.md` `LIBRARY_GAPS.md` `RISKS.md` `DECISIONS.md` (append-only) `MEETINGS.md`
`DIGESTS/round-<k>.md` `rounds/ROUND-<k>/*.md` `code/` `results/*.json` `screen/*.json` `report/`

## Machine

2 GPUs: RTX 3090 (24 GB) for large arms, RTX 3080 (10 GB) for screens/diagnostics/CNNs. 32 cores,
121 GB RAM. torch 2.10.0+cu128, torchvision 0.25. Boolean CIFAR-10 caches already exist at
`.cache/cifar10_therm{4,8}.pt`; raw dataset at `.data/`. Do not re-download or re-encode.
LaTeX: `PATH=/work/vajira/DL2026/.texlive/tl/bin/x86_64-linux:$PATH` (no system LaTeX on this box).
