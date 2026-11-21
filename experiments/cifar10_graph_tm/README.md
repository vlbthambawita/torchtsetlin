# cifar10_graph_tm: the graph Tsetlin machine on CIFAR-10

**Questions.** A graph TM of depth 1 on image patches is the convolutional coalesced TM (checked
here bit for bit). What a graph TM adds is message passing between nodes. (1) Does message
passing between neighbouring patches (depth 2, 3) raise CIFAR-10 accuracy over depth 1 at the
same clause count? (2) Is a gain due to the spatial edges or only to the extra literals? (3) Does
a best-effort rebuild of the graph TM paper's CIFAR-10 setting (arXiv:2507.14874: 70.28 %, CoTM
66.42 %) reach its number? Plan (hypotheses, cells, cost, risks):
`torchtsetlin_plans/experiments/cifar10_graph_tm/PLAN.md`; todo: issue #20. W&B mirror:
https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR (runs `cgt-…`, groups `cgt-pilot`,
`cgt-depth`, `cgt-paper`).

Builds on `../cifar10_batch_clauses` (CIFAR-10 files, the 45 000 / 5 000 split) and
`../cifar10_conv_reg` (runner, early stopping, figures).

## Quick start (server with four V100s, one per tmux session)

```bash
git pull                                             # branch joint-wide-tm
cd torchtsetlin/experiments/cifar10_graph_tm
wandb login --host https://api.forge.coreweave.com   # once per server
./setup.sh          # once: environment, GPU check, data, phase-0 checks, smoke test (about 5 min)
./run_gpu.sh 0      # tmux session 1
./run_gpu.sh 1      # tmux session 2
./run_gpu.sh 2      # tmux session 3
./run_gpu.sh 3      # tmux session 4
./status.sh         # progress, cost table, gate decisions, chosen T and s
```

If a session sees exactly one GPU, `run_gpu.sh K` uses that GPU and K only names the runner (it
must differ between sessions). If one shell sees all four, K is the GPU index. `setup.sh` reuses
the Python environment of `../cifar10_batch_clauses/.venv` when it exists (it adds matplotlib and
OpenCV if they are missing) and creates `.venv` here otherwise.

All runners must run in this folder: they share one job list through `.cache/claims/`. Each
starts a background process (log `logs/gpuK.log`, one log per run in `logs/runs/`);
`CGT_FG=1 ./run_gpu.sh 0` keeps it in the foreground. Re-running a script resumes: finished runs
are skipped, interrupted runs continue from their checkpoint. `./stop.sh` stops after the current
runs, `./stop.sh --now` immediately; `rm STOP` to continue.

## What runs

| Order | Phase | Runs | What |
|---|---|---|---|
| 1 | `pricing` | 11 timings (100 updates each) | seconds per epoch and peak memory of every cell; the **cost gate** |
| 2 | `paper` | up to 2 | the paper's setting: GraphTM and its CoTM baseline |
| 3 | `pilot` | 6 | depth 1, 20 000 clauses: T ∈ {3 750, 10 000, 30 000} × s ∈ {5, 20}, at most 60 epochs; chooses T / C and s |
| 4 | `depth` | up to 9 | the depth study, longest first; waits for the pilot and for pricing |

**Cost gate.** A cell that needs more than 1 500 s per epoch (training plus evaluation), or that
does not fit the GPU, is not run as planned. For the depth study the next smaller option is
priced and run instead: inbox 1 024, then 40 000 clauses; if nothing passes, the cell is dropped.
A paper-setting run above the gate is not started. Every decision is listed by `./status.sh` and
in `REPORT.md`. Overrides: `CGT_GATE_S=3000` (another limit), `CGT_GATE=off` (run everything as
planned), `CGT_PAPER_FORCE=1` (run the paper arm whatever it costs), `CGT_TIME_CAP_H=120`
(GPU-hours per run, default 48).

**The paper-setting GraphTM will not pass the gate.** A functional check on an RTX 3090 measured
about 1.4 s per update for it (80 000 clauses × 625 nodes × 1 778 features, 6 250 updates per
epoch): about 2.4 h per epoch, 70 h for the 30 epochs, more on a V100. The CoTM baseline needs
about 1 minute per epoch. Options are in the plan (§10, D1 and D7).

### Depth study (`depth`)

Each image is a graph of 225 nodes: the 15 × 15 windows of size 4 × 4 at stride 2 of the 8-level
colour thermometer (384 window bits + 28 position bits, exactly the patch features of the
convolutional models). Every node sends a message to its right, left, lower and upper neighbour
over four edge types (840 directed edges). `GraphTsetlinMachine`: shared clauses with signed
weights per class, soft literal budget 32, batch 32, hashed messages (2 048 inbox positions per
message layer, 2 per message), seed 0.

- Main cells: depth 1, 2, 3 × 20 000 and 80 000 clauses.
- Extra cells at depth 2, 20 000 clauses: inbox 512 and 8 192; **rewired edges** (every in-edge
  keeps its destination and type, its sender is a random other node; one fixed rewiring).
- T = (T / C of the pilot) × clauses, s of the pilot, the same for every depth.
- Protocol: 45 000 training / 5 000 validation images, early stopping on validation accuracy with
  patience 50 (minimum 30, maximum 300 epochs, 48 h per run). Statistic: test accuracy at the
  best-validation epoch. No augmentation.

### Paper setting (`paper`), best effort

80 000 clauses, T = 15 000, s = 20, 8 × 8 windows at stride 1 (625 nodes), literal budget 32,
depth 1, no edges. GraphTM node features: adaptive Gaussian threshold per channel (192) + 8-level
colour thermometer (1 536) + one-hot row and column (50) = 1 778 symbols, one feature each. CoTM
baseline: `ConvCoalescedTsetlinMachine` on the threshold planes. The paper's protocol, not ours:
all 50 000 training images plus their horizontal flips, 30 epochs, statistic = mean test accuracy
of the last 5 epochs. Batch 16 and the library's default multi-class feedback, as in the MNIST
and Fashion-MNIST reproductions (`../paper_experiments/code/graphtm/`). The symbol set is our
reading of the paper (its Table 16 says 2 500 symbols, in hypervectors), so a difference from
70.28 % cannot be attributed to the implementation alone.

## Phase-0 checks (`code/checks.py`, run by `setup.sh`)

1. Depth 1 on this programme's batches equals `ConvCoalescedTsetlinMachine` (4 × 4, stride 2)
   bit for bit: vote sums, automata and weights after each of 20 updates, and predictions.
2. Updates at depth 1, 2 and 3 are committed by the Triton kernel, and a warmed-up depth-2 update
   never synchronises with the host.
3. The rewired graph has the same in-edges per node and type as the grid.

`setup.sh` stops if one fails; the result is in `results/checks.json`.

## Outputs

- `results/<phase>/<tag>.json`: one record per run (settings, git sha, environment, the curve,
  the statistic). **Every reported number comes from these.**
- `results/pilot/choice.json`: the chosen T / C and s.
- `models/<phase>/<tag>.pt` (git-ignored): the best-validation model of each depth run, the final
  model of each paper run. Load with `train.load_model(path, device)`.
- `figures/<phase>/<tag>/`: every 20 epochs (`eNNN_*`) and for the best model (`best_*`):
  - `panel`: where the predicted class gets its votes, for one validation image per class;
  - `health`: literals per clause and layer, clause weights, firing rates, vote margins;
  - `rules`: the three strongest clauses as rules, with their message literals (`.png`, `.txt`);
  - `curves`: training and validation accuracy and their gap.
- `REPORT.md` and `figures/accuracy_vs_depth.png`: written by `python code/report.py`.

Tags: `gtm_d<depth>_c<clauses>[_m<inbox>]_<grid|rewired>_s0`, pilot runs add `_T<T>_spec<s>`;
`paper_graph_s0`, `paper_cotm_s0`.

Reading a rule: `R>=3[1,2]` is "red is at least thermometer level 3 at row 1, column 2 of the
window"; `row>4` and `col>4` are the window's position. `right1:57` is a message literal of layer
1: clause 57 held at the node a right-going edge comes from (the left neighbour). Messages are
hashed into the inbox, so a decoded message may also be another clause with the same two
positions; positions that decode to no message are printed raw (`m1[k]`).

## Single runs

```bash
. ./env.sh && cd code
python train.py --kind grid --depth 2 --clauses 20000 --T 10000 --s 5 --phase extra --device cuda:0
python train.py --kind grid --depth 2 --clauses 20000 --T 10000 --s 5 --phase pricing --calibrate 100
python viz.py ../models/depth/<tag>.pt        # figures of a kept model -> figures/saved/<tag>/
python queue_runner.py --gpu 0 --dry-run      # what would run next
```

`CGT_TS=0.5,5` fixes T / C and s without a pilot; `CGT_PHASES=pricing,pilot` restricts a runner;
`CGT_WANDB=0` switches the W&B mirror off.

## Rules

- `src/torchtsetlin/**` is read-only here; library defects go to `LIBRARY_GAPS.md`.
- Nothing is tuned beyond the pilot, and that choice uses validation only.
- The paper arm follows the paper's protocol and is never mixed into the depth table.
- Seed 0 only: differences below 0.5 pp are ties.
