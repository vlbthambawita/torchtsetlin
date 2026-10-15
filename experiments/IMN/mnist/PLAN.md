# MNIST study: how explainable are IMN and IMN-TM on images?

Part of `experiments/IMN` (programme rules C1–C3 apply: `src/torchtsetlin/**` is read-only,
reported numbers come only from `results/mnist*/**.json` records, model selection uses
validation data only). Written 2026-10-15, before any MNIST run.

## 1. Questions

| # | Question | Answered by |
|---|---|---|
| M-Q1 | How accurate are IMN and IMN-TM on MNIST, relative to the same network without an interpretable local model (the *hypernetwork-only* reference) and to the source TM? | test accuracy, §5.1 |
| M-Q2 | What do their explanations look like on a digit, and how do they differ from post-hoc explanations of the black-box reference? | heatmap figures, §5.2 |
| M-Q3 | Are the explanations *faithful*: does removing the pixels they rank highest actually destroy the prediction? | pixel-deletion curves, §5.3 |
| M-Q4 | Do the explanations depend on the trained model at all, or do they mostly redraw the input? | model-randomisation sanity check, §5.4 |
| M-Q5 | When a model is known to use a spurious cue, does its explanation show it? | Decoy-MNIST ground truth, §5.5 |
| M-Q6 | What does a reader have to read: how big and how locally valid is each explanation? | explanation size, local fidelity, §5.6 |

## 2. Arms

| Arm | Model | Role | Native explanation |
|---|---|---|---|
| `tabresnet` | the IMN hypernetwork backbone (TabResNet, 2 blocks × 128) with a plain linear head that predicts the logits directly | **hypernetwork-only reference**: the same network with no interpretable local model; upper reference for accuracy at this capacity | none, so we use post-hoc methods: occlusion (removal) and Integrated Gradients |
| `imn` | hypernetwork → per-image linear model over the 784 pixels | the paper's method | the generated weight map `w_k(x)` and the impacts `w_km(x)·x_m` |
| `imn_tm` | hypernetwork → per-image weights over frozen TM clauses | our method | the active rules with their weights, plus a pixel *clause-credit* map (§4) |
| `tm` | the source Tsetlin machine (global clause vote) | the interpretable model IMN-TM builds on | clause-credit map with the global votes |
| `clause_lr` | one global, re-fitted weight per frozen clause | control: is the per-image hypernetwork needed? | clause-credit map with the global weights |
| `lr` | multinomial logistic regression | global linear reference | `w_k·x` |

IMN and IMN-TM use the main-stage settings, without re-tuning:
- `imn`: `l1_mode="all"`, `weight_norm=0.1`, the paper's default;
- `imn_tm` and `clause_lr`: `l1_mode="prior"`, `weight_norm=1`, chosen in Phase 1 (D19);
- all three arms keep the remaining defaults of `NN_DEFAULTS`.

The training budget is the one exception and is chosen on validation data (§3, M0).

## 3. Data and protocol

- **Data:** the official MNIST split read with torchvision from the repository's `data/MNIST/raw`:
  60,000 training and 10,000 test images.
  - The **test** set is the official 10,000 images, the same for every seed.
  - **Validation** is 10,000 images taken from the official training set (stratified, by split
    seed). The remaining 50,000 images are used for training.
  - Seeds 0–4 change the validation split, initialisation and data order.
- **Preprocessing:** pixel intensities are divided by 255, giving values in [0, 1], with no
  per-pixel standardisation. Decision MD1:
  - Standardising the ~70 always-black border pixels would divide by a near-zero spread.
  - With 0 = black, "remove pixel m" (set `x_m = 0`) means "remove the ink", the usual image
    baseline. IMN's impact `w_m(x)·x_m` is then exactly the removal attribution (D-paper Eq. 2).
- **Booleanisation for the TM:** fixed intensity thresholds instead of quantiles. A pixel bit is
  `x_m ≥ t`, and constant pixels are dropped. Decision MD2: quantile thresholds of mostly-zero
  pixel columns are degenerate, while the TM-MNIST literature uses `x ≥ 0.3`.
- **M0 pilot (seed 0, validation only):**
  - *TM:* thresholds `{[0.3], [0.25, 0.5, 0.75]}` × clauses per class `{200, 500}` × `s ∈ {5, 10}`,
    with `T = 0.5·clauses`, at most 30 epochs and patience 5. That is 8 configurations, and the one
    with the best validation accuracy becomes the source TM of every clause arm.
  - *Budget:* `tabresnet`, `imn` and `imn_tm` with `max_steps ∈ {50k, 200k}`, about 64 and 256
    epochs at batch 64. Each arm keeps the budget with the higher validation accuracy. This avoids
    the under-training artefact found in Tier 2.

## 4. Explanations

Every arm uses the same *removal attribution* `φ_m(x)`: the drop in the explained logit (that of
the predicted class) when pixel `m` is set to 0.
- For `imn` and `imn_tm`, the local model is held fixed, as in the paper. IMN-TM re-evaluates its
  clauses on the modified image.
- For the other arms, the full model is re-evaluated (occlusion).

| Arm | Explanations computed |
|---|---|
| `imn` | weight map `w_k(x)` (784 numbers, a per-image *template*), impacts `φ = w_k(x) ⊙ x` |
| `imn_tm` | `φ` (removal); **clause credit** `ψ_m = Σ_{j active} W_kj(x) / L_j · [pixel m is in clause j]` (each active clause spreads its weight evenly over its `L_j` literals; `Σ_m ψ_m = z_k − b_k` exactly); the top active clauses drawn as literal masks |
| `tm`, `clause_lr` | `φ` (occlusion) and `ψ` with the global weights |
| `tabresnet` | `φ` (occlusion) and **Integrated Gradients** (black-image baseline, 64 steps, averaged over the snapshot ensemble) |
| `lr` | `φ = w_k ⊙ x` |

Clause credit shows *absence* evidence ("this pixel must stay blank") on background pixels, which
no removal attribution with a black baseline can show.

## 5. Metrics

1. **Accuracy (M-Q1):** test accuracy and log-loss, plus the validation accuracy used for selection.
2. **Pictures (M-Q2):** ten *showcase* test images, the first test image of each class. They are
   explained by every arm and seed, and their maps are stored in `.npz` files next to the records.
3. **Faithfulness (M-Q3), pixel deletion (MoRF):**
   - For each of 500 explained test images (a seeded random subset), rank the *ink* pixels
     (`x_m > 0`) by attribution, highest first.
   - Set the top fraction `f ∈ {0, 0.05, …, 1}` of them to 0 and read the **full** model's
     probability of the originally predicted class.
   - Report the area under that curve (**deletion AUC**, lower is more faithful) and the same
     area for a random pixel order. Their difference is the **deletion gain**.
   - Each explanation is scored on the model it explains.
4. **Sanity (M-Q4):** the same architecture with freshly initialised, untrained parameters. For
   `imn_tm`, the clauses are also replaced by random data-anchored clauses of the same lengths.
   The model is explained on the 500 images, and we report the Spearman correlation between
   `|φ|` from the trained model and from the random model (Adebayo et al. 2018).
   - **Low** means the explanation depends on what the model learned.
   - **High** means it mostly redraws the input.
   - This applies to `tabresnet` (both methods), `imn` and `imn_tm`.
5. **Ground truth (M-Q5), Decoy-MNIST:**
   - *Construction:* a one-hot "barcode" of ten pixels on the top row (row 0, columns 9–18,
     always black in MNIST). In training and validation images the pixel for the **true** class
     is lit with probability 0.9, and the pixel for a random class otherwise. The cue is
     informative but noisier than the digit.
   - *Test sets:* the barcode is set to the true class (**consistent**), set to a different random
     class (**swapped**), or left out (**clean**).
   - *Reliance:* the share of swapped images whose prediction follows the barcode.
   - *Ground truth per image:* if a swapped image is predicted as the barcode class, the
     decision used the barcode, so a correct explanation must rank that lit barcode pixel among
     its top 10 pixels (**barcode hit rate@10**). We also report the share of `|φ|` that falls on
     the barcode in consistent images.
6. **Readability (M-Q6):**
   - the number of terms that cover 90% of `|contribution|`: pixels for `imn`, clauses for `imn_tm`;
   - active clauses and clause length;
   - local fidelity: the accuracy of an image's local model on its k nearest explained
     neighbours from each class, as in paper Table 1;
   - the cosine similarity between local-model and full-model removal attributions.

## 6. Phases, budget and outputs

| Phase | Runs | Estimated wall time (2 GPUs, 3 workers each) |
|---|---|---|
| M0 pilot | 8 TM + 6 NN (seed 0) | ~1 h |
| M1 main | 6 arms × seeds 0–4 = 30 | ~1.5 h |
| M2 decoy | `lr tm tabresnet imn imn_tm` × seeds 0–2 = 15 | ~1 h |
| Report | `code/mnist_report.py` → `mnist/REPORT.md`, `mnist/figures/*.png` | minutes |

- Results go to `results/mnist_pilot/`, `results/mnist/` and `results/mnist_decoy/`, with one
  JSON record per (arm, seed), plus `.npz` files holding the showcase maps and the deletion curves.
- `mnist/REPORT.md` is rendered from `mnist/report_template.md`. Every number is a placeholder
  filled from the records (C2), and only the prose is written by hand.

**Statistics.** Tables give the mean ± sd over seeds. A paired comparison reports the mean
per-seed difference and the number of seeds in which one arm beats the other. For the per-image
explanation metrics, we use a Wilcoxon signed-rank test over the 500 images in each seed and
report the largest p-value across seeds. With 5 seeds, no seed-level test can reach p < 0.05
after correction, so none is claimed.

## 7. Risks and limits fixed in advance

- **Deletion produces off-manifold images.** All arms are scored on the same kind of
  perturbation, which makes the comparison relative and not absolute. ROAR, which retrains,
  costs too much here.
- **A one-hot barcode is easy to read for every arm**, so reliance may approach 1 for all of
  them. The 0.9/0.1 noise is meant to prevent this, and we report the reliance, not assume it.
- **Large clause sets.** With 500 clauses per class, the generated clause block has
  128 × 5001 × 10 ≈ 6.4 M head parameters, which is feasible. If the pilot picks it and memory
  fails, the row chunks of D27/D28 apply.
- **MNIST is not tabular.** Like IMN, IMN-TM treats pixels as independent features. A
  convolutional TM would be the natural clause source and is out of scope.
