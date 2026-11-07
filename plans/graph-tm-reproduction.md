# Graph Tsetlin machine: reproducing the paper's numbers (graph-tm.md, step 10)

*Status: plan, 2026-11-07. Not started. **Every run in it is benchmark-class and needs the user's
go-ahead** (memory note: no timing or seed runs without it; smoke runs are fine). Parent plan:
`plans/graph-tm.md`; fast paths: `plans/graph-tm-fast-paths.md` (done, `f92c813`). Tracks
GitHub issue #8.*

## TL;DR

Reproduce, with `GraphTsetlinMachine`, the experiments of Granmo et al. (arXiv:2507.14874, v2)
whose inputs and settings are published:

| ID | Experiment (paper section) | Reported | Feasible on one RTX 3090? |
|---|---|---|---|
| G0 | Three consecutive A's, depth 2, 4 clauses (Sec. 3.1) | 100% test | yes, seconds |
| G1 | MNIST, disconnected nodes (Sec. 3.2, Tables 3 and 16) | GraphTM **98.42 ± 0.05**, CoTM 98.93 ± 0.02 | yes, current code |
| G2 | Fashion-MNIST, disconnected nodes | GraphTM **89.49 ± 0.09**, CoTM 91.05 ± 0.09 | only after library step L1 |
| G3 | CIFAR-10, disconnected nodes, two views | GraphTM **70.28 ± 0.17**, CoTM 66.42 ± 0.19 | after L1; the preprocessing is under-specified (§3.4) |
| G4 | Multivalue noisy XOR (Sec. 3.8, Fig. 8, Table 22) | curves only (no table) | yes |

Each experiment is run twice: the GraphTM on its symbol-hypervector graphs, and, where the paper
has one, the CoTM baseline as `ConvCoalescedTsetlinMachine` on the same Boolean images.

**One library step first (L1).** At depth 1 a graph clause *is* a convolutional clause over the
node features. The existing Triton kernel `_triton.conv_eval` evaluates and draws on bit-packed
literals without ever forming the `(B, P, C)` match tensor. Today's graph path forms it twice
(the matches and the dense node draw). Per epoch that is:

* MNIST: ~1.3 TB of memory traffic, fine;
* F-MNIST (40,000 clauses, 676 nodes): ~39 TB, hours per epoch;
* CIFAR-10 (80,000 clauses, 625 nodes, 100k images with flips): ~120 TB, hours per epoch.

Routing depth-1 graphs with no padding nodes through `conv_eval` makes G2 and G3 feasible.

## 1. Rules

The programme follows the rules of `experiments/paper_experiments/PLAN.md` §1 (P1–P7). It is
a new benchmark row there, the GraphTM row; that plan currently says the GraphTM "publishes no
numbers to compare against", and that line is updated.

* **P1, the library is read-only during runs.** L1 is library work and happens *before* the
  programme starts. It is committed and tested, and the version is pinned in every record. A
  defect found during the runs goes into `LIBRARY_GAPS.md`.
* **P2, numbers come from `results/*.json` records only** (argv, seed, git SHA, versions, env,
  per-epoch logs).
* **P3, no tuning.** The settings are Table 16 and Table 22 and, where those are silent, the
  reference repository's demo at `cair/GraphTsetlinMachine@c6129e1` (the same commit as the
  parent plan). Every choice the paper leaves open is listed in §3 *before* the first run.
* **P4, statistics.** The paper reports the **mean test accuracy over the final 5 of 30
  epochs**; its `±` is over those epochs, and the number of runs is not stated. We report
  exactly that statistic, plus the final-epoch accuracy (the programme's headline rule), plus
  the mean ± sd over seeds. Test accuracy is computed every epoch, but nothing is selected on
  it.
* **P5, identical inputs.** Each dataset is Booleanised once into the programme's
  `.cache/data`. The MNIST threshold matches B4/B7. The GraphTM graphs and the CoTM inputs are
  built from the same bits.
* **P6, timing runs are exclusive.** Load and `nvidia-smi` are logged before and after; the
  machine is shared right now, so we wait for it to be free.

## 2. Library step L1: depth-1 graphs on `conv_eval` (before the programme)

Scope:

* `_GraphMixin._evaluate` and `_evaluate_learning` take the convolutional kernel path when all
  of these hold: `depth == 1`, CUDA with Triton, and **no padding nodes**. A batch knows it has
  no padding nodes when every graph has `P` nodes; `from_images` and the image builders of §3
  set a `full_nodes=True` flag, so the check needs no sync.
* Literals are bit-packed per node, as `_ConvMixin._neg_bits` does. `conv_eval(..., draw=True)`
  returns the clause outputs plus the Type Ia and Type II nodes. Events mode then works
  unchanged (`fused_commit` CONV).
* Node types (`n_node_types > 1`) stay on the dense path: the kernel has no clause-ownership
  mask.
* Prediction uses `conv_eval(draw=False)`, so it never builds the dense tensor either.

Tests:

* Evaluation equals the dense path exactly.
* Draws are uniform over matching nodes (as in `test_conv_patch_draws_are_uniform_and_independent`).
* With `from_images` graphs and the same events, a depth-1 graph model updates bit for bit like
  `ConvCoalescedTsetlinMachine`.
* A padded or typed batch falls back to the dense path.

The smoke measurement afterwards is one epoch on a 1,000-graph F-MNIST subset. Estimate:
2–3 days.

## 3. Experiments and the choices the paper leaves open

### 3.1 G0: three consecutive A's (Sec. 3.1)

* 40,000 sequences of 5 letters (13,330 positive), 1% training-label noise, a noise-free test
  set.
* Depth 2, 4 clauses, 4 epochs.
* **Open choices:**
  * `T` and `s` are not stated. We use those of `tests/test_graph.py` (T=40, s=2.0).
  * The train/test split is not stated. We use 80/20.
* **Pass:** 100% test accuracy. This is a sanity check before anything big.

### 3.2 G1: MNIST (Table 16)

* **GraphTM:** 2,500 clauses, T=3,125, s=10, depth 1, 30 epochs, no budget.
* **Graphs:** 10×10 windows at stride 1 give 361 nodes.
* **Node symbols** follow `MNISTConvolutionDemo.py` exactly:
  * the 100 pixel positions that are on;
  * `C:q` and `R:r` for the window's row and column, one symbol each (not thermometer).

  That is 138 symbols, the paper's count. They are encoded in a 128-bit hypervector with 2 bits
  per symbol (`SymbolTable(encoding="random", dim=128, bits=2)`, the reference's encoding,
  tested in the parent plan).
* **Threshold:** the paper says "stationary threshold" without a value; we use the demo's
  `> 75`, which is also the programme's B4/B7 rule.
* **CoTM:** `ConvCoalescedTsetlinMachine` with the same `C`, `T`, `s` and patch, on the raw
  thresholded pixels, with position encoding on.
* Settings fixed in advance:
  * state bits 8;
  * `negative_scale` and `multi_label` as in the reference rule
    (`docs/concepts/graph-tm.md`, differences table);
  * batched feedback with batch 25;
  * one sequential run of seed 0 to check fidelity.

### 3.3 G2: Fashion-MNIST

* **Settings:** 40,000 clauses, T=15,000, s=10, 3×3 windows (676 nodes), depth 1, 30 epochs.
* **Symbols:** an 8-bin thermometer per pixel (Grønningsæter et al., 2024), so 9 × 8 = 72
  pixel-level symbols, plus 26 + 26 position symbols. That is 124, the paper's count.
* **Open choices:**
  * The thermometer thresholds are not given. We use the uniform levels `k · 256 / 9` for
    k = 1…8.
  * The position style is not given. We use the MNIST demo's one symbol per row and column.
* **CoTM:** the same thermometer bits, 3×3 patches.

### 3.4 G3: CIFAR-10 (two views)

* **Settings:** 80,000 clauses, T=15,000, s=20, budget 32, 8×8 windows (625 nodes), depth 1,
  30 epochs.
* **Training set:** expanded with horizontal flips, giving 100,000 images.
* **GraphTM nodes:** two views per patch, as node symbols:
  * adaptive Gaussian threshold per channel, `cv2.adaptiveThreshold(GAUSSIAN_C, 11, 2)` as in
    `CIFAR10Demo.py`: 8 · 8 · 3 = 192 symbols;
  * an 8-bin colour thermometer: 8 · 8 · 3 · 8 = 1,536 symbols;
  * position symbols.
* **CoTM:** the adaptive threshold only, as the paper says.
* **Known gap:** the paper's Table 16 says **2,500 symbols**, which this description does not
  produce (it gives 1,778 with one symbol per row and column). The public `CIFAR10Demo.py` uses
  one view, 40,000 clauses, T=7,500 and thermometer positions, so it is not the paper's setup
  either.

  We run the description as written and report the symbol count, so G3 is a **best-effort**
  reproduction and is labelled as such. Before running it, we email or open an issue with the
  authors asking for the CIFAR-10 preprocessing (decision D3).

### 3.5 G4: multivalue noisy XOR (Sec. 3.8, Table 22)

* **Data:** `x1, x2 ∈ {0…499}` (500 symbols); `y = (x1 + x2) mod 2`; 1% training-label noise.
  50,000 training and 5,000 test examples.
* **Graph:** two nodes, an edge each way, one edge type (`NoisyMultiValueXORDemo.py`).
* **Settings:** depth 2, s=2.2, T=10 × clauses, node hypervector 2048 bits.
  * Clauses ∈ {4, 50, 200, 1,000, 2,000}.
  * Message size ∈ {2,048, 4,096, 8,192}; Table 22 prints 8,196, a typo for Fig. 8's 8,192.
  * Epochs: Fig. 8 runs to ~50.
  * Five trials, as the figure caption says.
* **Open choices:** bits per symbol and per message are not stated. We use the demo defaults,
  2 and 2, with random (not one-hot) messages, as in the paper.
* **There are no reported numbers, only curves.** The pass criteria are the paper's three
  qualitative findings:
  1. accuracy rises with the clause count;
  2. larger messages help at many clauses;
  3. at message size 8,192, 1,000 clauses beat 2,000 (collisions).

  We also report `message_collision_rate()` per cell, which tests the paper's explanation of
  (3) directly.
* **Grid:** 15 cells × 5 trials = 75 runs. The 2,000-clause cells dominate the cost.

### 3.6 Excluded, with reasons (stated in the report)

| Paper experiment | Why not |
|---|---|
| MNIST superpixels (Sec. 3.3) | The settings are inconsistent: Table 17 prints 60,000 clauses and depth 35, while Table 4 says 2,000 clauses. The k-NN edges and node symbols are under-specified. |
| IMDB / Yelp / MPQA (Sec. 3.4) | The tokenisation and graph construction (word distances as edge types) are not published. T=100,000 with 10,000 clauses at depth 2 on full text is beyond this budget. |
| Tangram, Amazon, MovieLens, viral genomes (Sec. 3.5–A.5) | Modified or noised datasets, or baselines copied from other papers. Not reproducible from the text. |
| Scalability study (Sec. 3.7) | Settings missing (the sidecar notes this). |

## 4. Pass criteria (fixed now)

* **G1 and G2:** the tt GraphTM is within **max(1.0 pp, 2 × seed sd)** of the reported value on
  the paper's statistic (programme rule §3.1). The same goes for tt's CoTM against the paper's
  CoTM.
  * The paper's own *ordering* must also hold: CoTM ≥ GraphTM on MNIST and F-MNIST.
  * The hashed 128-bit features cost the GraphTM about 0.5 pp; that is the paper's explanation,
    and our depth-1 equivalence tests make it checkable.

  An extra control isolates the encoding: the depth-1 GraphTM on **raw** patch bits must match
  the CoTM, because it is the same machine (tested bit for bit). Any remaining gap is then due
  to the hypervector encoding alone.
* **G3:** reported with its gap (§3.4). Passing means the GraphTM beats the CoTM, which is the
  paper's claim; the absolute value is informative only.
* **G4:** the three qualitative findings (§3.5).
* **A failed criterion is reported, not tuned away.** Likely causes are checked in this order:
  1. batched vs sequential feedback (rerun one seed sequentially);
  2. the reference's negative-class rule vs ours (`multi_label` + `negative_scale`);
  3. the per-event vs shared node draw (`shared_node_draw=True`).

## 5. Phases and budget

| Phase | Work | Runs | Gate |
|---|---|---|---|
| A | L1 (§2), tests, smoke | smoke only | — |
| B | Data and graph builders for G0–G4 in `experiments/paper_experiments/code/graphtm/`, written once to `.cache/data/graphtm_*.npz` with checksums (P5) | none | — |
| C | Smoke: every cell for 1 epoch on 1,000 graphs; the epoch time is extrapolated to a budget table per experiment | smoke only | **User approves the budget and the open choices (§3)** |
| D | Seed-0 runs of G0–G4, exclusive machine, timings recorded | benchmark | report to the user |
| E | Remaining seeds (G1: 5; G2 and G3: as the budget allows, at least 3; G4: 5 trials per cell) | benchmark | — |
| F | Report: tables and figures generated from the records (a GraphTM section in `REPORT.md`, Fig. 8 redrawn next to ours), `config_map.md` entries, literature DB cross-check against the sidecar | none | user review |

**Rough budget** (to be replaced by Phase C's measurements):

* G0: minutes.
* G1: ~1–2 min per epoch, so ~1 h per seed.
* G2 and G3 after L1: about the conv B7/B8 cost scaled by clauses × patches; expect several hours
  per seed for G2 and ~1 day per seed for G3 on the 3090.
* G4: dominated by the 2,000-clause cells, ~10 min each, so ~10 h for the grid.

## 6. Decisions for the user (before Phase D)

* **D1. Run the reference implementation too?** `cair/GraphTsetlinMachine` (PyCUDA) on G1 and
  G4 would be a true cross-library row in the programme ("run every library that implements the
  model") and a reproduction check of the paper by its own code. Cost: one more venv and about
  double the GPU time for those cells. Recommended for G1 and G4 only.
* **D2. Seeds for G2 and G3.** The budget may allow only 3 seeds, or 1 for G3.
* **D3. Contact the authors** about the CIFAR-10 preprocessing (2,500 symbols) and the run
  count behind the `±` values. Otherwise G3 stays labelled best-effort.

## 7. Files

* **Library (L1):**
  * `src/torchtsetlin/models/graph.py`: the depth-1 `conv_eval` path;
  * `src/torchtsetlin/data/graphs.py`: the `full_nodes` flag;
  * `tests/test_graph.py`;
  * docs.
* **Experiment:** `experiments/paper_experiments/code/graphtm/` holds:
  * `prepare.py` (graph builders);
  * `run_graphtm.py`;
  * `configs.py` entries G0–G4;
  * `results/graphtm/*.json`.
* **Reporting:**
  * `experiments/paper_experiments/PLAN.md`: the benchmark row and the "not run" table entry;
  * `config_map.md`;
  * `REPORT.md`.
