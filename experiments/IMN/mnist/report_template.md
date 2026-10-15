# IMN and IMN-TM on MNIST: accuracy and explainability

*IMN × TM research programme, `experiments/IMN/mnist`. Plan: [`PLAN.md`](PLAN.md); decisions MD1–MD6 in
[`../DECISIONS.md`](../DECISIONS.md). Every number below is filled in from `results/mnist*/` by
`python -m experiments.IMN.code.mnist_report` (rule C2).*

## Summary

- **Accuracy.**
  - IMN-TM is the most accurate arm ({{acc_imn_tm}} %).
  - It beats the hypernetwork-only reference by {{dacc_imn_tm_tabresnet}} and IMN by {{dacc_imn_tm_imn}}.
  - It ties Clause-LR ({{dacc_imn_tm_clause_lr}}). On MNIST, the gain comes from the frozen TM
    clauses, not from making their weights per-image.
  - IMN is *less* accurate than the same network without a local model ({{dacc_imn_tabresnet}}).
- **Faithfulness.** On pixel deletion, IMN's impacts ({{retgain_imn_removal}}) and the post-hoc
  explanations of the black box ({{retgain_tabresnet_ig}}) are the most faithful.
  - IMN-TM's removal attribution is clearly lower ({{retgain_imn_tm_removal}}), but well above the
    source TM ({{retgain_tm_credit}}).
  - The clause models spread their evidence over many redundant rules, and most of it says
    *"this pixel must stay blank"*. Deleting ink cannot remove that kind of evidence.
- **Sanity.** IMN's weight map and IMN-TM's clause credit change completely when the model is
  randomised (ρ = {{sanity_imn_weights}} and {{sanity_imn_tm_credit}}).
  - Every ink-only attribution keeps a sizeable correlation, the hypernetwork-only post-hoc maps
    included (IG {{sanity_tabresnet_ig}}, IMN impacts {{sanity_imn_removal}}, IMN-TM removal
    {{sanity_imn_tm_removal}}).
  - Part of what those maps show is the digit's own ink.
- **Spurious cue.** On Decoy-MNIST, every explanation of every arm finds the barcode whenever the
  barcode decided the prediction (hit rate@10 of {{hit_imn_tm_removal}} % for IMN-TM, likewise for
  every other arm and method; Section 7).
  - The test separates the *models* rather than the explanations.
  - IMN-TM relies least on the barcode ({{rel_imn_tm}} % of swapped images follow it), against
    {{rel_tabresnet}} % for the hypernetwork-only model and {{rel_imn}} % for IMN.
- **Readability.** IMN-TM explains an image with {{read_imn_tm_active_clauses_mean}} active rules,
  and {{read_imn_tm_clause_terms_90}} of them are needed for 90 % of the logit.
  - Single rules are readable. The full rule set of one prediction is not.
  - IMN-TM's local models are almost global: they predict an image's neighbours at
    {{read_imn_tm_local_acc_k50}} %, against {{read_imn_local_acc_k50}} % for IMN, and its local and
    full-model attributions coincide (cosine {{read_imn_tm_local_vs_full_removal_corr}}).

## 1. What is compared

The main report (`../report/report.pdf`) tested IMN-TM on tabular benchmarks. In IMN-TM, a hypernetwork
generates per-example weights for the frozen clauses of a Tsetlin machine. This study asks how IMN-TM
behaves on images, where explanations can be *looked at*. Every arm sees an MNIST image as 784
independent pixel features, exactly as IMN sees a tabular row.

| Arm | What it is | How it is explained here |
|---|---|---|
| **Hypernetwork-only** (`tabresnet`) | The IMN hypernetwork (TabResNet, 2 residual blocks of width 128) with an ordinary output layer. It predicts the class directly, with no local model, so it is a black box. This is the accuracy reference: same network, same training, no interpretability constraint. | Post-hoc only: **Integrated Gradients** (black-image baseline, 64 steps) and **occlusion** (set one pixel to 0 and record the change of the predicted-class logit). |
| **IMN** (`imn`) | The same network outputs, for each image, a linear model over the 784 pixels: `z_k(x) = Σ_m w_km(x) x_m + b_k(x)`. | **Impacts** `w_km(x)·x_m`, which equal the removal attribution exactly, and the generated **weight map** `w_k(x)`, a per-image template. |
| **IMN-TM** (`imn_tm`) | The same network outputs, for each image, one weight per frozen TM clause: `z_k(x) = Σ_j W_kj(x) c_j(x) + b_k(x)`. A clause is a conjunction of pixel tests (`x_m ≥ t` or `x_m < t`). | The **active rules** with their weights; **removal** attribution (clauses re-evaluated with the pixel removed, weights fixed); and **clause credit**, where each active rule spreads its weight evenly over the pixels it tests. Clause credit sums exactly to the clause part of the logit. |
| TM (`tm`) | The source Tsetlin machine whose clauses IMN-TM freezes: one global vote per clause. | Clause credit with the global votes, and occlusion. |
| Clause-LR (`clause_lr`) | The same frozen clauses with one global, re-fitted weight each (no hypernetwork). | Clause credit and occlusion. |
| Logistic regression (`lr`) | Global linear reference. | Occlusion. |

### Data and training

- **Data:** official MNIST. Training uses 50,000 images; 10,000 of the official training images are held
  out for validation by split seed; the test set is the official 10,000 images. Pixels are scaled to
  [0, 1] (MD1), so "removing" a pixel means turning it black.
- **TM input:** pixel `m` becomes the bits `x_m ≥ t` for a few fixed intensity thresholds `t` (MD2).
- **Hyper-parameters:** IMN and the hypernetwork-only model use the paper's defaults. IMN-TM and
  Clause-LR use the setting chosen in Phase 1 of the main study. Only the TM configuration and the
  training budget were chosen here, on validation accuracy of seed 0 (Section 2).
- **Seeds:** {{n_seeds}} seeds. Each seed changes the validation split, the initialisation and the data
  order.

### How explanations are evaluated

1. **Looking at them:** ten showcase test images, the first of each digit, explained by every arm.
2. **Faithfulness (pixel deletion):**
   - Remove the image's ink pixels in the order the explanation ranks them, most important first,
     and track how long the model keeps its original prediction.
   - The area under this *retention* curve is low for a faithful explanation.
   - The **gain** is the random-order area minus the explanation's area, for the same model. Higher is
     better, and it is the primary number (MD4).
   - Each explanation is scored on the model it explains.
3. **Sanity check (model randomisation):**
   - Explain the same images with the same architecture, untrained and with random parameters.
     For IMN-TM, the clauses are random too.
   - Report the Spearman correlation between the two maps over the ink pixels.
   - An explanation that hardly changes when the model is destroyed mostly redraws the input.
     *Lower is better.*
4. **Ground truth (Decoy-MNIST):**
   - A one-hot "barcode" of ten pixels on the always-black top row shows the true class in 90% of
     training images, so the models can learn to use it.
   - On test images where the barcode shows a *wrong* class and the model follows it, the barcode
     decided the prediction. A correct explanation must therefore put the lit barcode pixel in its
     top 10 pixels.
5. **Readability:**
   - how many pixels or rules carry 90% of an explanation;
   - how many rules are active;
   - how well an image's local model still predicts its neighbours (local fidelity).

## 2. Pilot (seed 0, validation only)

**TM configuration** (chosen: {{sel_tm}}):

{{table_pilot_tm}}

**Training budget** (chosen: hypernetwork-only {{sel_steps_tabresnet}}, IMN {{sel_steps_imn}},
IMN-TM {{sel_steps_imn_tm}}; Clause-LR takes IMN-TM's budget):

{{table_pilot_nn}}

- **TM.** 500 clauses per class beat 200 at every threshold setting. The three-threshold
  encoding with `s = 5` was best. The kept clause count after empty-clause pruning is in the
  table.
- **Budget.** Every arm preferred the larger budget, 200k steps, about 255 epochs. IMN-TM at 50k
  steps already beats the other two networks at 200k.
- **Caveat.** Because all arms picked the upper candidate, their accuracy may still rise with more
  training. The comparison is at an equal, validated budget, not at convergence.

## 3. Accuracy

{{table_accuracy}}

Paired by seed: IMN-TM − IMN {{dacc_imn_tm_imn}}; IMN-TM − hypernetwork-only {{dacc_imn_tm_tabresnet}};
IMN − hypernetwork-only {{dacc_imn_tabresnet}}; IMN-TM − TM {{dacc_imn_tm_tm}};
IMN-TM − Clause-LR {{dacc_imn_tm_clause_lr}}.

- **Ranking.** The clause models lead, ahead of the black-box reference.
  - IMN-TM and Clause-LR are statistically indistinguishable. Both improve on the source TM
    ({{dacc_imn_tm_tm}}) and on the hypernetwork-only network ({{dacc_imn_tm_tabresnet}}).
  - The frozen clauses act as a good fixed feature map of binarised pixel patterns. A linear layer
    on top of them is already enough.
  - On MNIST, the per-image re-weighting adds no accuracy.
- **IMN** pays an accuracy cost for its interpretability on images ({{dacc_imn_tabresnet}} against
  the same network without a local model). A per-image linear model over 784 pixels is a harder
  output than ten logits.
- **Log-loss.** The TM's high log-loss is not a modelling failure. Its probabilities are
  `softmax(votes / T)`, which is very flat. This is why the deletion analysis uses retention of the
  prediction rather than probability (MD4).
- **Absolute level.** These are tabular models given the preregistered tabular defaults, with no
  convolution or augmentation. None comes close to a CNN's ~99.5 %, and none was meant to.

## 4. What the explanations look like

{{fig_gallery}}

The gallery reads column by column:

- **IMN weight map.** A smooth, digit-shaped *template*: red where ink supports the prediction,
  blue where it would oppose it.
  - It covers the background too, because it shows what the local model *would* do with ink there.
  - It is the most informative IMN view: for the 5 that IMN misreads as a 6, the template is
    shaped like a 6.
- **Ink-pixel attributions.** IMN's impacts, IMN-TM's removal and the hypernetwork-only occlusion
  and Integrated Gradients look alike: they highlight the strokes that distinguish the digit.
  - These maps can only be non-zero on ink, because removing a black pixel changes nothing.
- **Clause credit.** IMN-TM and the TM spread credit over the whole image, including the
  background.
  - This is the "must stay blank" evidence of the clauses: for the 0, the empty centre carries
    credit, and for the 7, the blank space left of the stem.
  - The TM's credit is noisier: its global votes are large and unevenly spread across clauses.
- **LR occlusion.** Sparse and dominated by a few pixels.

### The rules behind an IMN-TM prediction

{{fig_clauses}}

- **What the rules test.**
  - Most of the top active rules for the predicted class test that pixels are *blank* (blue);
    few test that a pixel is ink (red).
  - A rule like "all of these few pixels are empty" is readable on its own: it says where the
    digit is *not*. Combined, such rules carve out the digit's shape.
- **How the weights differ.**
  - IMN-TM's generated weights are smaller and more even than the TM's global votes, the effect of
    the L1 `prior` penalty and of training all weights jointly.
  - So no single rule dominates. This matches the readability numbers (Section 8): hundreds of
    rules are needed to cover most of a logit.

## 5. Faithfulness: pixel deletion

{{fig_deletion}}

{{table_deletion}}

Paired per image (same 500 test images per seed; gain of A minus gain of B):

{{table_deletion_pairs}}

- **Ranking.** IMN's impacts and the hypernetwork-only explanations (occlusion or IG) are the most
  faithful, and IMN's impacts are slightly ahead of IG ({{pg_imn_removal__tabresnet_ig}}). IMN-TM's
  removal attribution follows ({{pg_imn_tm_removal__imn_removal}} against IMN). It is still far
  better than the TM's own explanation ({{pg_imn_tm_removal__tm_credit}}).
- **Why IMN-TM scores lower.** The clause curves level off above zero: some images keep their
  prediction even with all ink removed. Ink deletion cannot reach the evidence of a clause model.
  - {{read_imn_tm_credit_background_share}} % of IMN-TM's clause credit lies on background pixels,
    which deletion never touches.
  - Removing ink can make "must be blank" literals *true* and so switch clauses *on*.
  - About {{read_imn_tm_active_clauses_mean}} active clauses overlap heavily, so removing any one
    pixel changes little. Single-pixel removal scores then rank pixels poorly for multi-pixel
    deletion.
  - This is an interaction effect between conjunctions and the deletion protocol. It is not
    evidence that IMN-TM's *rules* are unfaithful: they are its exact computation.
- **Clause credit.** It ranks ink pixels less well than removal does
  ({{pg_imn_tm_credit__imn_tm_removal}}). It describes how the logit is composed, not what happens
  when a pixel is removed.
- **LR.** It has the lowest deletion AUC of all, because one removed stroke flips it. Its large
  gain reflects that fragility as much as the explanation.
- **Comparing gains.** The random-order area itself differs between models: IMN is the most robust
  to random deletion and the TM the least. Gains are therefore best compared within a model, or
  between models of similar robustness.

## 6. Sanity check: do the explanations depend on the model?

{{table_sanity}}

{{fig_sanity}}

- **Ink-confined maps.** For removal-type explanations, the trained and random maps still
  correlate: IMN's impacts {{sanity_imn_removal}}, IMN-TM's removal {{sanity_imn_tm_removal}},
  hypernetwork-only occlusion {{sanity_tabresnet_removal}} and IG {{sanity_tabresnet_ig}}.
  - These maps are confined to ink and scale with pixel intensity, so part of every such map is
    the input itself.
  - The figure shows the effect: the random model's maps still trace the digit.
  - The black-box reference does not do better than the interpretable models here.
- **Maps that pass clearly.** IMN's weight map ({{sanity_imn_weights}}) and IMN-TM's clause credit
  ({{sanity_imn_tm_credit}}) are uncorrelated with their randomised versions: what they show was
  learned.
  - Caveat for clause credit: for IMN-TM, randomisation replaces the clauses as well as the
    weights. Its near-zero ρ therefore tests the whole model, not the hypernetwork alone.

## 7. Ground truth: Decoy-MNIST

{{table_decoy_acc}}

{{table_decoy_expl}}

{{fig_decoy}}

- **Reliance.** All arms learned to use the barcode a little, but none mainly.
  - Consistent accuracy is near 100 %, and swapped accuracy stays far above chance.
  - IMN-TM ({{rel_imn_tm}} %) and the TM ({{rel_tm}} %) rely on it least. The hypernetwork-only
    model relies on it most ({{rel_tabresnet}} %).
  - The clause models' "must be blank" rules on the always-black top row may explain the
    difference: a lit barcode pixel breaks those rules for every class, not just one. This is an
    interpretation, not tested here.
- **Ground truth.** Whenever a prediction followed the barcode, *every* explanation placed the lit
  barcode pixel in its top 10, and almost never did so when the prediction did not follow.
  - This is a ceiling effect: a single isolated pixel that decides the class is easy for every
    method to find. The test confirms that none of the explanations hides a spurious cue.
  - It cannot rank the methods.
- **Barcode share in consistent images.** Removal-type maps put a sizeable part of their mass on
  the barcode ({{share_imn_tm_removal}} % for IMN-TM's removal, {{share_tabresnet_ig}} % for IG),
  while clause credit puts less ({{share_imn_tm_credit}} % for IMN-TM). Credit spreads over
  the many blank-pixel literals.
- **Figure.** It shows swapped images whose prediction follows the barcode (bold labels).
  - The strip under each map enlarges the ten barcode pixels on the map's own colour scale.
  - In every method, the lit pixel (outlined) is the strongest attribution of the whole image. It
    is so strong that the digit itself fades to near-white.

## 8. How much has a reader to read?

{{table_readability}}

- **Pixel maps.** To cover 90 % of an attribution, every pixel map needs about half of the
  digit's ~{{read_imn_ink_pixels_mean}} ink pixels. IMN-TM's removal map is the most concentrated
  ({{read_imn_tm_terms90_removal}}), IMN ({{read_imn_terms90_removal}}) and the hypernetwork-only IG
  ({{read_tabresnet_terms90_ig}}) are close.
- **Rules.** IMN-TM's native explanation is not small: {{read_imn_tm_active_clauses_mean}} active
  rules of about {{read_imn_tm_active_clause_len_mean}} literals each, and
  {{read_imn_tm_clause_terms_90}} rules for 90 % of the logit.
  - A reader can inspect the top rules (Section 4) but not the whole decision.
  - A smaller, sparser clause set is the obvious lever: fewer clauses per class, or a stronger L1
    on the generated weights. It was not tuned here.
- **Locality.** IMN-TM's per-image model behaves almost like a global one.
  - Applied to an image's 50 nearest neighbours per class, it is right
    {{read_imn_tm_local_acc_k50}} % of the time, the same as the global Clause-LR
    ({{read_clause_lr_local_acc_k50}} %).
  - Its local attributions agree perfectly with the full model's.
  - IMN's local models are genuinely local ({{read_imn_local_acc_k50}} %,
    cosine {{read_imn_local_vs_full_removal_corr}}). IMN-TM's are stable and transferable, which
    matches the tie with Clause-LR in accuracy.

## 9. Conclusions and limitations

**What the study shows.**

1. **IMN-TM is the most accurate interpretable model on MNIST**, and more accurate than the black
   box it is built on. The frozen TM clauses carry this gain, and a global linear layer over them
   (Clause-LR) does as well.
2. **Its explanations differ in kind from pixel maps.**
   - They are rules, mostly about where the digit is *not*.
   - Clause credit passes the randomisation sanity check cleanly and shows absence evidence that no
     ink-based map can show.
   - On ink-deletion faithfulness, it trails IMN and the post-hoc explanations of the black box.
     The main reason is that clause evidence on the background cannot be deleted.
3. **IMN keeps per-image locality and the best deletion faithfulness**, at an accuracy cost
   against the same network without a local model. Its weight map is the most informative single
   picture in the gallery.
4. **The hypernetwork-only reference is not more explainable post hoc.** IG and occlusion on the
   black box are about as faithful as IMN's built-in impacts. They fail the randomisation check to
   the same degree as the other ink-only maps. They need a separate, approximate method, where IMN
   and IMN-TM are explained by their own computation.
5. **On a spurious cue**, all explanations reveal it when it is used. IMN-TM uses it least.

**Limitations.**

- **Training settings.** Tabular defaults were used, with no tuning beyond the TM configuration
  and the budget. All arms chose the largest budget offered, so the accuracies may not be
  converged.
- **Input representation.** Pixels are independent features with no spatial prior. A
  convolutional TM would be the natural clause source for images and is out of scope.
- **Deletion.** It produces off-manifold images and only removes ink. It therefore systematically
  under-credits explanations built on absence evidence. ROAR, which retrains, or an insertion
  metric on the background would be complementary.
- **Decoy.** The one-hot barcode is too easy to find to rank explanation methods. A distributed or
  texture cue would be harder.
- **Seeds and statistics.** Five seeds for the main study and three for Decoy-MNIST. The per-image
  Wilcoxon tests are within seeds, and no seed-level significance is claimed.
- **Decoy seed-0 runs.** The four decoy runs shown in the figure (IMN, IMN-TM, hypernetwork-only,
  TM; seed 0) were re-run after the main phase only to store maps for all explained swapped
  images. Their records were rewritten by the same code path.
