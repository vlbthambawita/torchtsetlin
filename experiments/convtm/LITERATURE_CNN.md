# LITERATURE_CNN.md — the CNN reference table for CIFAR-10

**Owner**: dl-expert · **Phase**: P1 · **Status**: complete for Gate G1 · **Date**: 2026-09-20

Every number here is tagged and carries the configuration that produced it. Nothing in this file is
a measurement of ours — our own numbers arrive in P2 and land in `results/*.json`. Where the
literature does not report something we need, this file says **not reported** and names the P2 arm
that will measure it.

**Read the three warnings in §0 before quoting any number out of this file.**

---

## 0. Three traps that make most CNN-vs-TM comparisons wrong

**(0.1) "CIFAR-10 accuracy" is at least four different numbers.** The augmentation regime moves a
CNN by 3–7 points, which is larger than most of the effects this programme is trying to detect.
[FACT: DeVries & Taylor 2017, arXiv:1708.04552 Table 1] a ResNet-18 scores **4.72 % error with
standard crop+flip and 10.63 % without** — a 5.9-point swing from augmentation alone, same
architecture, same recipe, five runs each. [FACT: Huang et al. 2017, arXiv:1608.06993 Table 2] the
same swing for ResNet-110 is 13.63 → 6.41 (7.2 points, as reported by the Stochastic Depth paper).
**The TM arms in this programme start without augmentation. The honest comparison is against the
no-augmentation column.**

**(0.2) "No augmentation" still means "with dropout" in almost every published number.**
[FACT: arXiv:1608.06993 Table 2 caption] *"All the results of DenseNets without data augmentation
(C10, C100, SVHN) are obtained using Dropout"* — specifically dropout 0.2 after every conv but the
first. [FACT: arXiv:1708.04552] Cutout's no-aug WRN-28-10 keeps dropout p = 0.3. [FACT:
arXiv:1312.4400] Network-in-Network's 10.41 % is *with* dropout; **without dropout the same net
scores 14.51 %** — dropout alone is worth 4.1 points in the no-augmentation regime. There is no
widely cited CIFAR-10 CNN number with neither augmentation nor regularisation. Our `cnn-*-noaug`
arms keep weight decay and the architecture's own BatchNorm but add no dropout, so they are, if
anything, a *conservative* no-augmentation baseline. This is recorded, not hidden.

**(0.3) The binary-net literature's comparison tables silently mix regimes.** [FACT:
arXiv:1511.00363 §3 and arXiv:1602.02830 §2] BinaryConnect, BNN and XNOR-Net all train CIFAR-10
**with no data augmentation at all** — BNN verbatim: *"We do not use any preprocessing or
data-augmentation (which can really be a game changer for this dataset)."* Every modern binary net
(IR-Net onwards) uses crop+flip. The "VGG-Small" rows in IR-Net / RBNN / ReCU / BiPer tables are
just the 2015–16 numbers arithmetically converted to accuracy, so those tables compare
no-augmentation 2016 methods against augmented 2020s methods in the same column. Do not reuse those
tables. [HYPOTHESIS] this inconsistency is why the binarisation gap looks larger in old work than
in new.

---

## 1. Master reference table — CIFAR-10

Error rates unless a cell says "acc". `—` = not reported by that source.
**no-aug** = trained with no data augmentation. **aug** = crop and/or flip, per the paper.

| # | Model | Source | Params | no-aug err | aug err | Augmentation | Optimiser / schedule / length |
|---|---|---|---|---|---|---|---|
| **Simple CNNs** |
| 1 | cuda-convnet `layers-18pct` (4 weight layers) | [FACT: Krizhevsky, cuda-convnet `example-layers/layers-18pct.cfg` header] ⚠ config comment, no paper | ~0.09 M (derived) | **18.0** | — | none | not documented |
| 2 | Model A (5 weight layers) | [FACT: Springenberg et al. ICLR-W 2015, arXiv:1412.6806 Table 3] | ≈0.9 M | **12.47** | — | none | SGD m 0.9, ×0.1 @200/250/300, 350 ep, wd 1e-3, dropout 20 %/50 % |
| 3 | Model C (9 conv) | same, Table 3 | ≈1.3 M | **9.74** | — | none | same |
| 4 | Strided-CNN-C | same, Table 3 | ≈1.3 M | **10.19** | — | none | same |
| 5 | ConvPool-CNN-C | same, Table 3 | ≈1.4 M | **9.31** | — | none | same |
| 6 | **All-CNN-C** | same, Tables 3 & 4 | ≈1.3–1.4 M ⚠ | **9.08** | **7.25** | flip + ≤5 px translate, on whitened/contrast-normalised images | same |
| 7 | Maxout | [FACT: Goodfellow et al. ICML 2013, arXiv:1302.4389 Table 4] | not reported (>6 M per All-CNN) | **11.68** | **9.38** ⚠ | translations + horizontal reflections | GCN + ZCA preprocessing |
| 8 | Network in Network | [FACT: Lin et al. ICLR 2014, arXiv:1312.4400 Table 1] | not reported (≈1 M per All-CNN) | **10.41** (14.51 without dropout) | **8.81** | translation + horizontal flip | GCN + ZCA |
| 9 | Deeply Supervised Net | [FACT: arXiv:1409.5185] / AISTATS 2015 ⚠ | ≈1 M | **9.78** (arXiv) / 9.69 (AISTATS) | **8.22** (arXiv) / 7.97 (AISTATS) | zero-pad 4, corner crop, random flip | — |
| **VGG** |
| 10 | VGG-16 config D, **ImageNet only** | [FACT: Simonyan & Zisserman ICLR 2015, arXiv:1409.1556 Tables 1–2] | 138 M | **CIFAR-10 not reported — 0 occurrences of "CIFAR" in the paper** | | 224 crop + flip + RGB shift | SGD m 0.9, lr 1e-2 ÷10 ×3, 370 k iters (74 ep), batch 256, wd 5e-4, dropout 0.5 |
| 11 | VGG-16 (13 conv + 2 FC, BN) | [FACT: Li et al. ICLR 2017, arXiv:1608.08710 Table 1] — the best peer-reviewed *literal* VGG-16 CIFAR-10 baseline | 1.5×10⁷ | — | **6.75** (93.25 % acc) | He recipe: pad 4 + crop + flip | He recipe, ~160 ep, batch 128 |
| 12 | "VGGNet" — **actually VGG-19**-BN ⚠ | [FACT: Liu et al. ICCV 2017, arXiv:1708.06519 Table 1] | 20.04 M | — | **6.34** (93.66 % acc) | shift + mirror, channel mean/std | SGD Nesterov 0.9, 0.1 ÷10 @50 %/75 %, 160 ep, batch 64, wd 1e-4 |
| 13 | VGG+BN+dropout (15 weight layers) | [FACT: szagoruyko/cifar.torch README] ⚠ repo, not peer-reviewed | ~15 M (derived) | **8.7** (91.3 % acc) | **7.55** (92.45 % acc) | horizontal flips **only** | SGD m 0.9, lr 1.0 halved /25 ep, 300 ep, batch 128, wd 5e-4 |
| **ResNet — CIFAR variants (the 6n+2 family)** |
| 14 | ResNet-20 | [FACT: He et al. CVPR 2016, arXiv:1512.03385 Table 6] | 0.27 M | — | **8.75** | pad 4 + 32×32 crop + flip | SGD m 0.9, lr 0.1 ÷10 @32 k & 48 k of 64 k iters, batch 128, wd 1e-4 |
| 15 | ResNet-32 | same | 0.46 M | — | **7.51** | same | same |
| 16 | ResNet-44 | same | 0.66 M | — | **7.17** | same | same |
| 17 | ResNet-56 | same | 0.85 M | — | **6.97** | same | same |
| 18 | ResNet-110 | same | 1.7 M | **13.63** ‡ | **6.43** best, 6.61 ± 0.16 over 5 runs | same (+ lr 0.01 warm-up to 80 % train err) | same |
| 19 | ResNet-1202 | same | 19.4 M | — | **7.93** | same | same |
| 20 | plain-20/32/44/56 (no shortcuts) | same, Fig. 6 left | identical to ResNets | — | **not reported numerically — curves only**; only hard fact is *"the error of plain-110 is higher than 60 %"* | same | same |
| 21 | ResNet-164 pre-act | [FACT: arXiv:1608.06993 Table 2, run by the DenseNet authors] | 1.7 M | **11.26** | **5.46** | mirror/shift | — |
| **ResNet-18 (4-stage ImageNet-style, CIFAR stem)** |
| 22 | ResNet-18 | [FACT: kuangliu/pytorch-cifar README] ⚠ repo, and ⚠ **model selected on the test set** | 11,173,962 (derived exactly from the repo's `models/resnet.py`) | — | **6.98** (93.02 % acc) | pad 4 + crop + flip | SGD 0.1 cosine T=200, 200 ep, batch 128, wd 5e-4 |
| 23 | ResNet-18 | [FACT: DeVries & Taylor 2017, arXiv:1708.04552 Table 1, mean of 5 runs] | not stated | **10.63 ± 0.26** | **4.72 ± 0.21** | pad 4 + crop + 50 % flip | SGD Nesterov 0.9, 0.1 ÷5 @60/120/160, 200 ep, batch 128, wd 5e-4 |
| 24 | PreAct ResNet-18 | [FACT: Zhang et al. ICLR 2018, arXiv:1710.09412 Fig. 3a, ERM baseline] | not stated | — | **5.6** | standard | SGD m 0.9, 0.1 ÷10 @100/150, 200 ep, wd 1e-4 |
| **Wide / dense** |
| 25 | WRN-28-10 | [FACT: Zagoruyko & Komodakis BMVC 2016, arXiv:1605.07146 Table 5, median of 5] | 36.5 M | — | **4.00** (mean/std norm) / **4.17** (Table 4, ZCA) | **reflect**-pad 4 + crop + flip | SGD Nesterov 0.9, 0.1 ×0.2 @60/120/160, 200 ep, batch 128, wd 5e-4 |
| 26 | WRN-28-10 | [FACT: arXiv:1708.04552 Table 1] | 36.5 M | **6.97 ± 0.22** (dropout 0.3) | **3.87 ± 0.08** | pad 4 + crop + flip | as row 23 |
| 27 | DenseNet-BC-100 k=12 | [FACT: arXiv:1608.06993 Table 2] | 0.8 M | **5.92** | **4.51** | mirror/shift | SGD Nesterov 0.9, wd 1e-4; no-aug runs add dropout 0.2 |
| 28 | DenseNet-BC-250 k=24 | same | 15.3 M | **5.19** — the best no-aug number in that table | **3.62** | same | same |
| 29 | DenseNet-BC-190 k=40 | same | 25.6 M | — | **3.46** | same | same |

‡ ResNet-110's 13.63 no-aug figure is [FACT: arXiv:1608.06993 Table 2, attributed there to the
Stochastic Depth paper], not to He et al. themselves.

⚠ **All-CNN-C's own paper gives ≈1.4 M in Table 3 and ≈1.3 M in Table 4.** ⚠ **Maxout's prose says
9.35 % with augmentation, its Table 4 says 9.38 %**; the literature uniformly cites 9.38.
⚠ **DSN: arXiv:1409.5185 reports 9.78/8.22; the AISTATS 2015 version reports 9.69/7.97.** DenseNet's
table cites the latter. ⚠ **Liu et al.'s "VGGNet" row (6.34 %) is VGG-19, not VGG-16** — confirmed by
reproducing its 20,040,522 parameter count; a 16-layer version gives 14.73 M, which does not match.
Do not cite 6.34 % as a VGG-16 number.

### 1.1 He et al. and the 45k/5k split — the exact wording

This matters because our protocol (PLAN §7.1) also uses 45k/5k, but for a different purpose.

[FACT: arXiv:1512.03385 §4.2] *"We start with a learning rate of 0.1, divide it by 10 at 32k and 48k
iterations, and terminate training at 64k iterations, **which is determined on a 45k/5k train/val
split**."*

He et al. use the split **only to fix the schedule**; Table 6's errors come from models retrained on
the full 50 k. **We do not do that.** Our arms train on 45 k and never see the held-out 5 k, so our
reproduction of ResNet-20 should be expected to land slightly *worse* than 8.75 % on 10 % less data,
and the P2 record will say so rather than quietly training on 50 k to match the paper.
[HYPOTHESIS] the 45 k-only penalty for ResNet-20 is 0.3–0.8 points. Falsifier: `cnn-resnet20`
reproduces within the §7.5 3-point band either way, so this does not gate anything; it is a caveat
for the reproduction table, not a claim.

---

## 2. Binarised networks — the "same computational substrate" reference

This is the most important section for the programme's argument. A Tsetlin machine computes with
Boolean literals and integer votes; the fairest neural reference is not a float CNN but a net whose
weights **and activations** are one bit. That bounds what binary computation achieves on this task.

### 2.1 The original three (all trained **without any augmentation**)

| Method | Source | What is binarised | CIFAR-10 err | FP baseline *in the same paper* | Gap |
|---|---|---|---|---|---|
| BinaryConnect (stochastic) | [FACT: Courbariaux, Bengio & David, NIPS 2015, arXiv:1511.00363 Table 1] | **weights only**, activations real | **8.27** | **10.64** ("No regularizer", identical arch) | **−2.37** — binary is *better* |
| BinaryConnect (deterministic) | same | same | **9.90** | 10.64 | −0.74 |
| BNN (Torch7) | [FACT: Courbariaux, Hubara et al., arXiv:1602.02830] | **weights + activations**, 1 bit | **10.15** | none of their own arch | vs BC's 10.64: −0.49 |
| BNN (Theano) | same | same | **11.40** | " | vs 10.64: +0.76 |
| XNOR-Net BWN | [FACT: Rastegari et al. ECCV 2016, arXiv:1603.05279] | weights only + per-filter α | **9.88** | **not reported for CIFAR-10** | n/a |
| XNOR-Net | same | weights + activations + α, β | **10.17** | " | n/a |

**Architecture, common to all of them** [FACT: arXiv:1511.00363 §3]:
`(2×128C3)-MP2-(2×256C3)-MP2-(2×512C3)-MP2-(2×1024FC)-10SVM`, BatchNorm throughout, square hinge
loss. Parameter count is **not reported by any of these papers**; ≈14.0 M derived from the
architecture — **our implementation of it gives exactly 14,029,726** [MEASURED: code health check,
`code/cnn/models.py::bnn_vgg`, §5 below], of which 14,008,320 are binary weights and 21,406 are
float (the full-precision first conv, the classifier and every BatchNorm).

**Training** [FACT: arXiv:1511.00363 / arXiv:1602.02830]: ADAM, exponentially decaying lr, **500
epochs**, batch 50 (Theano) or 200 (Torch7, shift-based AdaMax, lr halved every 50 epochs).
Preprocessing: BinaryConnect uses GCN + ZCA; **BNN drops even that** — verbatim: *"We do not use any
preprocessing or data-augmentation."*

**First and last layers** [FACT: arXiv:1603.05279]: *"This motivates us to avoid binarization at the
first and last layer of a CNN. In the first layer the channel size is 3 and in the last layer the
filter size is 1×1."* BNN likewise: *"all the layers inputs are binary, with the exception of the
first layer."* BinaryConnect does **not** mention any exemption. Our `cnn-binary*` arms keep the
first conv and the classifier full precision by default and expose `binary_first_last` so the cost
of *not* doing so can be measured rather than assumed.

**What the speedup claims actually cover** [FACT: arXiv:1603.05279]: *"58× faster convolutional
operations … and 32× memory savings."* The 58× is the high-precision **operation count of one
convolution** on a CPU doing 64 binary ops/clock, *"excluding the process for memory allocation and
memory access"* — not an end-to-end network speedup. BWN alone gets ~2×. The 32× is weight storage.
This is the right precedent for how our own efficiency claims should be scoped.

### 2.2 The modern binary nets (all **with** crop+flip)

| Method | Source | ResNet-18 acc | ResNet-20 acc | VGG-Small acc | FP baseline in that paper |
|---|---|---|---|---|---|
| IR-Net | [FACT: Qin et al. CVPR 2020, arXiv:1909.10788] | 91.5 | 86.5 | 90.4 | 93.0 / 91.7 / 91.7 |
| RBNN | [FACT: Lin et al. NeurIPS 2020, arXiv:2009.13055] | 92.2 | 87.8 | 91.3 | 93.0 / 91.7 / 91.7 |
| ReCU | [FACT: Xu et al. ICCV 2021, arXiv:2103.12369] | 92.8 | 87.4 | 92.2 | 94.8 / 92.1 / 94.1 |
| **BiPer** | [FACT: Vargas et al. CVPR 2024, openaccess CVPR2024 BiPer, Table 1] | **93.75** | 87.5 | 92.11 | 94.8 / 92.1 / 94.1 |
| SURGE | [FACT: arXiv:2605.10989, ICML 2026] ⚠ very recent, unverified by a second source | 93.1 | **88.0** | **92.5** | 94.8 / 92.1 / 94.1 |

**The single number this programme needs**: the best known fully binary (1-bit weights, 1-bit
activations) CIFAR-10 accuracy is **93.75 %** [FACT: BiPer, CVPR 2024], on a CIFAR-adapted ResNet-18
with crop+flip, against a 94.8 % full-precision baseline in the same paper — a **binarisation gap of
about 1 point**. Note this keeps the first and last layers in full precision and uses augmentation.

**Why this matters more than anything else in this file.** Published single-model CTM best on
CIFAR-10 is **60.7 %** and TM SOTA (a composite) is **82.8 %** [FACT: PLAN.md §2 / §3.3, to be
verified by tm-theorist in `LITERATURE_TM.md`]. A network with the same 1-bit substrate reaches
93.75 %. **So the TM's ~33-point gap to CNNs is not explained by binary computation.**
[HYPOTHESIS] at most ~1–5 points of the gap is attributable to 1-bit arithmetic; the remainder is
Booleanized input, architecture, and the learning algorithm. This decomposition is the report's
spine and P2 measures each term.

### 2.3 ImageNet, for scope

[FACT: arXiv:1603.05279] AlexNet: full precision 56.6/80.2 top-1/top-5; **BWN 56.8/79.4** (binary
weights cost *nothing* on top-1); **XNOR-Net 44.2/69.2** (binarising activations too costs 12.4
points). ResNet-18: FP 69.3/89.2, BWN 60.8/83.0, XNOR 51.2/73.2.
⚠ XNOR-Net's BinaryConnect (35.4) and BNN (27.9) AlexNet rows are **their own 16-epoch
reproductions**; the BNN authors report 41.8 top-1 for the same setting [FACT: arXiv:1609.07061].
This is the most mis-cited pair of numbers in the binary-net literature.

[FACT: Liu et al. ECCV 2020, arXiv:2003.03488] ReActNet-A reaches **69.4 % ImageNet top-1** at 1/1,
against 72.4 % for a real-valued ResNet-18. **Neither Bi-Real Net nor ReActNet reports CIFAR-10.**

[HYPOTHESIS, and the load-bearing one] the *activation* binarisation is where the cost lives, not
the weight binarisation: BWN ≈ FP on AlexNet top-1, XNOR −12.4. A Tsetlin machine binarises both,
and additionally binarises the *input*. Falsifier in P2: if `cnn-binaryconnect` (binary weights,
real activations) and `cnn-binary` (both) land within 1 point of each other on CIFAR-10, this
generalisation from ImageNet does not hold at CIFAR scale and the claim is withdrawn.

---

## 3. Sample efficiency — CIFAR-10 with few labels

TMs are widely claimed to be sample-efficient; PLAN §2 lists this as a secondary success axis, and
notes nobody in the local bibliography measures it against a CNN under one protocol. Here is what
the CNN literature actually reports. **All error rates, supervised-only (labelled data only).**

| Source | Architecture | Aug on the baseline | 1 000 | 2 000 | 4 000 | 50 000 |
|---|---|---|---|---|---|---|
| [FACT: Laine & Aila ICLR 2017, arXiv:1610.02242 Table 1] | 13-layer ConvNet | none | — | — | 35.56 ± 1.59 | 7.33 ± 0.04 |
| same | same | flips + translations | — | — | 34.85 ± 1.65 | 6.05 ± 0.15 |
| [FACT: Tarvainen & Valpola NIPS 2017, arXiv:1703.01780 Table 5] | 13-layer ConvNet | none | 48.38 ± 1.07 | 36.07 ± 0.90 | 24.47 ± 0.50 | 7.43 ± 0.06 |
| same, Table 2 | same | flips + ±2 px + noise | 46.43 ± 1.21 | 33.94 ± 0.73 | 20.66 ± 0.57 | 5.82 ± 0.15 |
| [FACT: Oliver et al. NeurIPS 2018, arXiv:1804.09170] | WRN-28-2 | standard | — | — | 20.26 ± 0.38 | — |
| [FACT: Cubuk et al. arXiv:1805.09501 Table 2, "Reduced CIFAR-10"] | WRN-28-10 | flips + pad/crop | — | — | 18.8 | — |
| same | WRN-28-10 | + AutoAugment | — | — | 14.1 ± 0.3 | — |
| [FACT: Sohn et al. NeurIPS 2020, arXiv:2001.07685 Table 9] | WRN-28-2, 1.5 M | RandAugment (**strong**) | — | — | 12.74 ± 0.29 | — |
| [FACT: arXiv:1905.02249] | WRN-28-2, 1.5 M | crop+flip | — | — | — | 4.17 |

**Gaps in the literature we must fill ourselves**: **no CIFAR-10 supervised-only number at 5 000 or
10 000 images is reported in any of these papers.** The only directly citable 1 000-image
supervised-only pair is Mean Teacher's 46.43 (aug) / 48.38 (no aug). Our sample-efficiency arms
(`--subset 1000/5000/10000`) therefore produce numbers that do not exist in the literature at the
5 k and 10 k points, under a protocol where the TM arms are measured identically.

**The biggest single caveat**: [FACT: arXiv:1610.02242 §4, quoting Sajjadi et al. 2016b] a
supervised-only 4 000-label baseline ranges from **13.60 % to 35.56 % error across papers**, purely
from augmentation strength and architecture. Laine & Aila say so explicitly: *"They quote an error
rate of only 13.60 % for supervised-only training with 4000 labels, while our corresponding baseline
is 34.85 %."* **A "CNN sample-efficiency curve" quoted from the literature is worthless for our
purposes.** We must measure our own, and we do, with the same split, the same augmentation setting
and the same selection rule as the TM arms. This is `cnn-*` × `--subset {1000,5000,10000,50000}`.

**Augmentation's value grows as data shrinks — but the sources disagree on how much.**
Mean Teacher: 3.81 points at 4 000 labels vs 1.61 at 50 000. Temporal Ensembling: 0.71 points at
4 000 vs 1.28 at 50 000 — the *opposite* ordering, on a similar backbone.
[HYPOTHESIS] the disagreement is augmentation *strength*, not data size.

---

## 4. What the literature does **not** report, and which P2 arm supplies it

| Missing number | Why it matters here | P2 arm |
|---|---|---|
| CNN accuracy on a **Booleanized** (12-plane thermometer) CIFAR-10 | Isolates the cost of Booleanization — the TM's input handicap — from everything else. **Nothing in either literature measures this.** | `cnn-boolean`, `cnn-boolean-therm8`, `cnn-boolean-resnet18` |
| CNN accuracy with binary weights + binary activations **on Boolean input** | The closest thing to a TM on the same substrate | `cnn-binary-boolean` |
| A **CTM-shaped** CNN: one conv layer, global max over patch positions, linear vote | Separates *architecture* from *learning algorithm*. The CTM is exactly this network, trained differently. | `cnn-ctmshape-max` |
| The same with a **counting** (mean) pool | Prices PLAN §9.2(1) on a substrate that can be trained exactly, for ~10 GPU-min instead of hours | `cnn-ctmshape-sum` |
| Supervised-only CIFAR-10 at **5 000 and 10 000** images | Two of the four sample-efficiency points the programme needs | every arm × `--subset` |
| ResNet-20 / VGG-16 **without augmentation** | The like-for-like comparison against the TM arms | `cnn-resnet20-noaug`, `cnn-vgg-noaug` |
| Any CNN number trained on **45 k only**, never touching the val 5 k | Our protocol; every literature number trains on 50 k | all of them |

---

## 5. Our implementations — parameter counts verified against the papers

`code/cnn/models.py`, checked on this machine today. This is a code-health measurement, not a
result; the accuracies arrive in P2.

| Our builder | Params (ours) | Params (literature) | Match |
|---|---|---|---|
| `resnet_cifar(depth=20)` | 269,722 | 0.27 M [FACT: arXiv:1512.03385 Table 6] | ✓ |
| `resnet_cifar(depth=56)` | 853,018 | 0.85 M, same | ✓ |
| `resnet_cifar(depth=110)` | 1,727,962 | 1.7 M, same | ✓ |
| `resnet18_cifar()` | 11,173,962 | 11,173,962 (derived from kuangliu/pytorch-cifar) | ✓ exact |
| `vgg_cifar('vgg16')` | 14,986,698 | 1.5×10⁷ [FACT: arXiv:1608.08710 Table 1] | ✓ |
| `bnn_vgg()` | 14,029,726 (14,008,320 binary + 21,406 float) | not reported; ≈14.0 M derived from the arch string | ✓ |
| `cnn_small(width=64)` | 1,148,874 | — (ours; comparable to All-CNN-C's ≈1.3 M) | n/a |
| `ctm_shaped(2000 filters, 4×4)` | 408,010 | — (ours) | n/a |

MACs per image (ours, conv+linear only): cnn_small 152.8 M, vgg16 313.5 M, resnet20 40.6 M,
resnet56 125.5 M, resnet18 555.4 M, ctm_shaped(2000, 4×4, stride 1) 323.0 M.

---

## 6. The targets our P2 arms must hit (Gate G2)

Gate G2 requires `cnn-resnet18` ≥ 93 % and `cnn-small` ≥ 85 %, both with augmentation.

| Arm | Predicted test acc | Basis | Falsifier |
|---|---|---|---|
| `cnn-resnet18` (aug, 160 ep cosine, 45 k) | **94.0–95.5 %** | [FACT: arXiv:1708.04552] 95.28 % at 200 ep with a step schedule on 50 k | < 93 % → G2 fails, investigate before proceeding |
| `cnn-resnet18-noaug` | **88–90 %** | [FACT: arXiv:1708.04552] 89.37 % no-aug, but *with* their regularisation | — |
| `cnn-vgg` (aug) | **93–94 %** | [FACT: arXiv:1608.08710] 93.25 % | — |
| `cnn-resnet20` (aug, He recipe, 45 k) | **90.5–91.5 %** | [FACT: arXiv:1512.03385] 91.25 % on 50 k | outside §7.5's 3-point band → marked `GAP` |
| `cnn-small` (aug) | **91–93 %** | comparable to All-CNN-C's 92.75 % aug at similar size | < 85 % → G2 fails |
| `cnn-small-noaug` | **85–88 %** | [FACT: arXiv:1412.6806] All-CNN-C 90.92 % no-aug **with dropout**; we have none | — |
| `cnn-binary` (BNN, no aug, 200 ep not 500) | **86–89 %** | [FACT: arXiv:1602.02830] 89.85 % at 500 epochs | < 85 % → the 200-epoch budget is the cause; report it, do not hide it |
| `cnn-boolean` (no aug) | **83–87 %** | [HYPOTHESIS] thermometer-4 keeps most of the signal | this is the number the whole decomposition rests on — 3 seeds, no shortcuts |

**Deviations from the papers, declared in advance** (each goes into the record's `notes`):
1. **45 k training images, not 50 k.** Protocol, PLAN §7.1. Costs an estimated 0.3–0.8 points.
2. **Reflect padding** in augmentation (`data._augment_batch`), where He et al. zero-pad and WRN
   reflect-pads. One implementation for the whole programme beats per-arm fidelity.
3. **Cosine schedules** for the non-ResNet arms, where the papers use step schedules.
   `cnn-resnet20`/`cnn-resnet56` use `schedule="he-step"` for reproduction fidelity.
4. **200 epochs for the binary arms, not 500.** A 2.5× budget cut; if `cnn-binary` under-performs
   the paper, this is the first suspect and the record says so.
5. **No dropout in `cnn-*-noaug`.** See §0.2. Makes our no-aug numbers conservative.
6. **`cnn-boolean` uses ±1 rather than 0/1 plane encoding.** Better conditioned for BatchNorm; the
   `bool_scale` flag makes the alternative a one-line change if it is ever contested.

---

## 7. Sources

ResNet https://arxiv.org/abs/1512.03385 · VGG https://arxiv.org/abs/1409.1556 · All-CNN
https://arxiv.org/abs/1412.6806 · NiN https://arxiv.org/abs/1312.4400 · Maxout
https://arxiv.org/abs/1302.4389 · DSN https://arxiv.org/abs/1409.5185 · WRN
https://arxiv.org/abs/1605.07146 · DenseNet https://arxiv.org/abs/1608.06993 · Cutout
https://arxiv.org/abs/1708.04552 · AutoAugment https://arxiv.org/abs/1805.09501 · mixup
https://arxiv.org/abs/1710.09412 · Network Slimming https://arxiv.org/abs/1708.06519 · Pruning
Filters https://arxiv.org/abs/1608.08710 · BinaryConnect https://arxiv.org/abs/1511.00363 · BNN
https://arxiv.org/abs/1602.02830 · QNN/JMLR https://arxiv.org/abs/1609.07061 · XNOR-Net
https://arxiv.org/abs/1603.05279 · Bi-Real https://arxiv.org/abs/1808.00278 · ReActNet
https://arxiv.org/abs/2003.03488 · IR-Net https://arxiv.org/abs/1909.10788 · RBNN
https://arxiv.org/abs/2009.13055 · ReCU https://arxiv.org/abs/2103.12369 · BiPer (CVPR 2024
openaccess) · SURGE https://arxiv.org/abs/2605.10989 · Mean Teacher
https://arxiv.org/abs/1703.01780 · Temporal Ensembling https://arxiv.org/abs/1610.02242 · MixMatch
https://arxiv.org/abs/1905.02249 · FixMatch https://arxiv.org/abs/2001.07685 · Realistic SSL
https://arxiv.org/abs/1804.09170 · kuangliu/pytorch-cifar · szagoruyko/cifar.torch · cuda-convnet
`example-layers/*.cfg`
