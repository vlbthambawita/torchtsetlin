# compgen: the compositional generalization benchmark (issue #15) on torchtsetlin

**Questions.** Does torchtsetlin reach 100 % test accuracy on the benchmark of
[issue #15](https://github.com/vlbthambawita/torchtsetlin/issues/15), how does the batch size
change the learning dynamics, and what does the standard Tsetlin machine (both literal kinds) do
on it? The benchmark is `examples/CompositionalGeneralizationBenchmark.py` (was `SequenceDemo.py`)
of [cair/PyHierarchicalTsetlinMachineCUDA](https://github.com/cair/PyHierarchicalTsetlinMachineCUDA)
(MIT, © Granmo and the University of Agder).
W&B mirror: https://forge.coreweave.com/wandb/simulamet_mlc/TM_COMPGEN (runs `cg-…`).

## Quick start (server with 4 GPUs, one GPU per tmux session)

```bash
git clone -b compgen https://github.com/vlbthambawita/torchtsetlin.git
cd torchtsetlin/experiments/compgen
./setup.sh          # once, in any session: environments, GPU check, data, checks, smoke test (~15 min)
. .venv/bin/activate && wandb login --host https://api.forge.coreweave.com   # once
./run_gpu0.sh       # tmux session 1
./run_gpu1.sh       # tmux session 2
./run_gpu2.sh       # tmux session 3
./run_gpu3.sh       # tmux session 4
./status.sh         # progress and the cost table
```

A runner uses whichever single GPU its session sees; the number only names it (`logs/gpuK.log`).
All four must run in the same folder: they share one job list through `.cache/claims/`. Each
`run_gpuK.sh` starts a background runner (one log per run in `logs/runs/`); `CG_FG=1 ./run_gpu0.sh`
keeps it in the foreground of the tmux pane instead.

The runners work through **price** (every cell for 3 epochs, gives the cost table and the job
order) → **phase 1** (18 runs) → **phase 2** (12 runs) and exit when nothing is left. Re-running
a script resumes: finished runs are skipped, interrupted runs continue from their checkpoint
(every 25 epochs). `./stop.sh` stops after the current runs, `./stop.sh --now` immediately;
remove `STOP` to continue.

Options (use the same in all four sessions):

| Variable | Effect |
|---|---|
| `CG_PHASES=price,p1` | only these phases (`price`, `p1`, `p2`) |
| `CG_SEEDS=0` | one seed instead of 0, 1, 2 |
| `CG_REF=0` | no reference-library runs (when pycuda cannot be built) |
| `CG_BOTH_MAX_EPOCHS=500` | shorter cap for the standard-TM runs of phase 2 (default 3 000) |
| `CG_SLOTS=2` | two runners per GPU (the models need well under 1 GB) |
| `CG_WANDB=0`, `CG_WANDB_PROJECT=…` | no W&B, or another project |
| `CG_VENV=none`, `PYTHON=python3.12`, `TORCH_INDEX=<wheel index>`, `CUDA_ROOT=<toolkit>` | environment |

Results come back with `rsync -a <server>:<path>/results/ results/` (or a commit); then
`python code/report.py` writes `REPORT.md` and `figures/`.

## Rules

| # | Rule |
|---|---|
| R1 | `src/torchtsetlin/**` is **read-only** here. Library defects go to `LIBRARY_GAPS.md`. |
| R2 | Every number comes from a `results/<phase>/<tag>.json` record carrying `argv`, `seed`, `git_sha`, `env` and the data hash. W&B is a mirror. `REPORT.md` is generated; do not hand-edit. |
| R3 | Nothing is tuned: all hyper-parameters are the issue's. |
| R4 | There is no validation set and nothing is selected. A run stops by a convergence rule on the test accuracy; the statistic is the time to reach it. |

## The benchmark

A sequence of 5 positions holds a permutation of 5 concepts, each shown as one of 5 synonyms:
5 × 5 × 5 = 125 one-hot features, 5 of them set. The label is the parity of the permutation.
Of the 375 000 possible inputs, 40 000 are drawn for training (1 % label noise) and 40 000 for
testing, so most test inputs were never seen. The issue's command:

```
--sequence-length 5 --s 25.0 --T 10 --noise 0.01 --number-of-examples 40000 --number-of-state-bits 8
--q 1.0 --clause-weight-update-ratio 1.0 --literal-budget 100 --number-of-or-alternatives 1
--number-of-clauses 200 --vanilla --number-of-synonyms 5 --and-group-normalization --negate-input
```

The input is negated and negated literals are not appended, so a clause can only say "this
symbol is *not* here" and must find a concept by elimination.

## Design

| Arm | Model | Literals | Clauses |
|---|---|---|---|
| `ref` | the reference library at commit `226d8b64`, built as the script's `--vanilla` branch | negated input only | 200, one output |
| `neg` | `tt.TsetlinMachine`, 2 classes, `literals="negated"` (torchtsetlin ≥ 0.7.4) | negated only | 2 × 100 |
| `both` | `tt.TsetlinMachine`, 2 classes | both kinds (the standard TM) | 2 × 100; 2 × 1 000 and 2 × 10 000 with T × 10, × 100 |

All: T = 10, s = 25, 256 states per automaton, unweighted, boost true positive, literal budget 100.

| Phase | Runs |
|---|---|
| `price` | every cell below for 3 epochs, seed 0 |
| `p1` | `neg` × {sequential, batch 8, 32, 128, 512} × seeds 0–2, and `ref` × seeds 0–2 |
| `p2` | `both` × {sequential, batch 32} × seeds 0–2, and `both` at 1 000 and 10 000 clauses per class × batch 32 × seeds 0–2 |

**Training.** One pass over the training set per epoch. torchtsetlin runs shuffle every epoch;
the reference library, as in its script, sees the examples in file order. A run stops when the
test accuracy is ≥ 99.9 % for 20 epochs in a row, after 3 000 epochs, or after 24 h.

**Statistics.** Epochs, updates and training time to the first 10-epoch stretch at ≥ 90 %,
≥ 99 % and ≥ 99.9 % test accuracy; the final accuracy; seconds per epoch.

**Metrics** (the benchmark's own, on the test set; every epoch up to 200, then every 5th): share
of examples matched by at least one clause, clause precision and recall (percentiles 1, 5, 10,
50, 90, 95, 99 and the mean), the highest precision among the clauses matching each example, and
matching clauses per example. Each run also writes `<tag>.stats.txt` in the reference script's
23-column format.

**Data.** `code/data.py` is a line-by-line port of the script's generator; with the same NumPy
seed it returns the same arrays (check M0 executes the original block and compares). The data
seed is the run seed; `results/data/seed<k>.json` holds the hash and the pattern statistics.

## Known differences from the reference

- Two class sums of 100 clauses against one output of 200 (`LIBRARY_GAPS.md`, CG-2).
- torchtsetlin shuffles the training set every epoch.
- Pattern statistics: the issue reports 29 424 distinct training inputs and 9 044 shared with the
  test set; the generator gives about 37 900 and 3 700 here (uniform sampling predicts these).

## Layout

| Path | What |
|---|---|
| `setup.sh`, `setup_ref.sh`, `env.sh` | one-time setup; the reference library's environment; shared environment |
| `run_gpu0.sh` … `run_gpu3.sh`, `run_gpu.sh`, `gpu_device.sh` | one runner per GPU slot |
| `status.sh`, `stop.sh` | progress; graceful / immediate stop |
| `code/common.py` | arms, grid, tags, paths, records, stop rule and statistics |
| `code/data.py` | the benchmark's generator, hashes, pattern statistics |
| `code/metrics.py`, `code/metrics_np.py` | the benchmark's metrics on the GPU; the script's formulas in NumPy |
| `code/train.py` | one torchtsetlin run |
| `code/run_ref.py` | one reference-library run (its own environment, same record format) |
| `code/checks.py` | M0 generator, M1 metrics, M2 literal restriction, M3 resume |
| `code/queue_runner.py` | shared job list, claims |
| `code/tracking.py` | W&B mirror, id `cg-<phase>-<tag>`, group `cg-<arm>` |
| `code/report.py` | `REPORT.md`, figures, `results/summary.json`; `--status` for progress |
