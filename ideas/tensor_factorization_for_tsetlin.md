# Tensor factorization and the Tsetlin pipeline — literature review, theory, and a go/no-go

**Date**: 2026-09-22 · **Status**: scoping note, *not* a programme artifact.
No method is chosen here (`experiments/convtm/CHARTER.md` C3). Claims carry the house tags:
`[FACT: source]`, `[MEASURED: result-id]`, `[HYPOTHESIS]`.

---

## 0. The answer, first

1. **As a modelling device, tensor factorization is not an addition to a Tsetlin machine — it is
   what a Tsetlin machine already is.** The TM decision function is literally a CP (rank-C)
   decomposition of a multilinear tensor, with the clauses as the rank-1 terms and the
   include-matrix as Boolean factor matrices (§3.1). "Adding factorization" therefore means
   *relaxing the factors from {0,1} to the reals*, which buys expressivity and costs the two
   things the TM exists for: literal-level interpretability and multiply-free hardware.
2. **As a compression or speed device, it is exact-but-dominated.** The clause match test admits an
   exact factorized form (§3.4) — no approximation needed, which is a nice result — but its ceiling
   on our own flagship is ~1228/r×, whereas plain sparsity offers **47×** on the same measurement
   `[MEASURED: ctm-therm5-preflight_seed0]`, and four published TM papers already take that route.
3. **Exactly one question in this space is live and cheap**: is the *linear* vote over the clause
   presence vector a binding constraint on CIFAR-10? A low-rank multilinear head (FM = order-2 CP;
   Exponential Machines = TT) is the cheapest known way to lift it, and — unlike a second TM layer —
   it is gradient-trainable, so it walks around the wall that killed `experiments/mctm`.
   **Cost to answer: ~4–6 GPU-hours (§5). Do that, and nothing else, before spending more.**

---

## 1. The literature, sorted into five families

### 1.1 Classical multilinear decompositions
CP/PARAFAC, Tucker, Tensor Train (TT/MPS), Hierarchical Tucker. The canonical reference is Kolda &
Bader's review `[FACT: kolda.net/publication/TensorReview.pdf]`. In deep learning these are used for
**post-hoc compression**: a conv kernel `(C_out, C_in, k, k)` is replaced by 3 smaller layers via
partial Tucker or CP, ranks chosen by singular-value thresholding, followed by fine-tuning
`[FACT: dl.acm.org/doi/10.1145/3702641 (TEC-CNN, 2024); S156849462200905X (HT-2)]`. Known failure
mode: CP gives better compression when it works but is numerically unstable and becomes infeasible
above ~512×512 layers; this instability is what makes fine-tuning after decomposition hard
`[FACT: mdpi.com/2076-3417/14/4/1491, Feb 2024]`.

**Relevance here: near zero.** Every one of these methods factorizes a *real-valued, differentiable*
parameter tensor and repairs the damage with gradient fine-tuning. A TM has neither.

### 1.2 Tensor networks as the model itself
Stoudenmire & Schwab map inputs through a tensor-product feature map and represent the resulting
exponentially long weight vector as an MPS, reaching <1% MNIST error
`[FACT: arXiv:1605.05775]`. Novikov et al.'s **Exponential Machines** make the framing explicit: the
model has one weight per *subset* of features — 2^N of them — and the TT format stores that tensor
in O(N r²), fitted by Riemannian optimization; they report fitting tensors with 2^160 entries
`[FACT: arXiv:1605.03795]`. Also tree tensor networks with CP-rank constraints and tensor dropout
`[FACT: arXiv:2305.19440]`, and TN-constrained kernel machines `[FACT: arXiv:2403.19500]`.

**This family is the one that actually talks about the same object a TM does** — see §3.1.

### 1.3 Factorization machines and polynomial networks
Rendle's FM models every pairwise interaction with a shared low-rank embedding: O(pk) parameters
instead of O(p²), evaluable in O(pk). Higher-order FMs represent the order-d interaction tensor in
**CP format** `[FACT: ismll.uni-hildesheim.de FreudenthalerRendle]`. Blondel et al. unify FMs and
polynomial networks as low-rank symmetric tensor estimation, multi-convex even at higher order
`[FACT: arXiv:1607.08810, arXiv:1705.07603]`.

**This is the practically relevant family**: it is tensor factorization at a scale and cost that
fits on top of a 20 000-dimensional Boolean vector.

### 1.4 Boolean / discrete factorization
The discrete analogue, where 1+1=1. Miettinen's Discrete Basis Problem and the Asso algorithm,
MDL4BMF for choosing the rank `[FACT: dl.acm.org/doi/10.1145/2601437]`, Boolean CP/Tucker via
Walk'n'Merge `[FACT: arXiv:1310.4843]`, GETF at O(n) `[FACT: NeurIPS 2020, 1def1713ebf1]`, and
Bayesian/EM variants `[FACT: arXiv:1702.06166, arXiv:1905.12766]`.

Complexity, and it matters: **Boolean rank is NP-hard to compute, and minimum-error fixed-rank
Boolean CP cannot be approximated within any polynomially computable factor**
`[FACT: mpi-inf.mpg.de/~pmiettin/btf/; ijcai.org/proceedings/2020/0685.pdf]`. The hardness descends
from the set-basis problem.

**Relevance: this is the only family whose algebra matches TM state.** §3.4 uses it — and finds that
the hardness is not the binding problem; the achievable gain is.

### 1.5 Name collisions to discard
**Logic Tensor Networks** `[FACT: arXiv:2012.13635]` are fuzzy first-order logic grounded in neural
networks; "tensor" there means "the data structure", not a decomposition. Likewise "tensorized NN
accelerators" `[FACT: arXiv:2511.17971, arXiv:2504.06474]`. Neither is about factorization of a TM.

### 1.6 The niche on the TM side is already occupied
Anyone proposing factorization *for efficiency* is competing with, and must beat, these — all exact,
none requiring a combinatorial search:

| Work | Mechanism |
|---|---|
| Weighted TM `[FACT: arXiv:1911.12607]` | one weighted clause replaces many duplicates |
| Clause indexing `[FACT: arXiv:2004.03188]` | index literals → skip non-matching clauses |
| Sparse TM `[FACT: arXiv:2405.02375]` | store active literals only, fixed memory bound |
| Contracting TM / absorbing automata `[FACT: arXiv:2310.11481]` | literals and clauses leave the model permanently |
| ETHEREAL `[FACT: arXiv:2502.05640]` | drop literals included in both polarities |
| Drop Clause `[FACT: arXiv:2105.14506]` | stochastic clause dropout |
| Omni TM-AE `[FACT: arXiv:2505.16386]` | reuses the full automaton state space as an embedding |

### 1.7 Prior art at the intersection
**None found.** Searches over TM + {tensor decomposition, low-rank, tensor train, factorization
machine, Boolean matrix factorization}, 2018–2026, return the two literatures side by side and never
joined. That is either an opening or a signal. §3 argues it is mostly a signal, and gives the
structural reason: *the join has already happened implicitly, and the TM is on the far side of it.*

---

## 2. Where a tensor actually lives in this pipeline

Shapes for the measured flagship, `ctm-therm5-preflight` (CIFAR-10, 20 000 clauses, 5×5 patches,
therm-5, budget 32) `[MEASURED: ctm-therm5-preflight_seed0]`:

| Object | Shape | Size here | Dtype | Notes |
|---|---|---|---|---|
| `ta_state` | (C, 2F) | 26.16 M automata, **104.6 MB** | int8 | the learning state `[FACT: base.py]` |
| `include` | (C, 2F) | 20 000 × 1 308 | 0/1 float | derived cache; **556 514 ones → 2.13 % dense** |
| literals | (B, P, 2F) | P = 784 | 0/1 float | `conv.py:_encode` |
| `matches` | (B, P, C) | the memory wall | bool | `conv.py:_evaluate` |
| Φ (clause presence) | (B, C) | 20 000 bits/image | bool | `matches.any(dim=1)` |
| weights | (C, K) | 20 000 × 10 | int | 80 KB — nothing to compress |

Two numbers from that record drive everything below:
`literals_evaluated_per_image = 20.51 G` versus `included_literals_per_image = 436 M` —
**the dense evaluation does 47× more work than the included literals require**.

---

## 3. Theoretical analysis

### 3.1 Proposition A — a Tsetlin machine *is* a CP decomposition

Write the literal vector ℓ(x) ∈ {0,1}^{2F} and the include matrix A ∈ {0,1}^{C×2F}. The class vote is

    v_i(x) = Σ_{j=1..C} w_{j,i} · Π_{l=1..2F} ℓ_l(x)^{A_{jl}}                              (★)

`[FACT: functional.py:52-63 (match ⟺ no included literal false), classifier.py:99-112]`.

Each term of (★) is a **monomial in ℓ**, i.e. a rank-1 tensor in the tensor product space
⊗_{l=1..2F} R², whose l-th mode carries the 2-vector (1, ℓ_l). Therefore (★) is a sum of C rank-1
terms: **a canonical polyadic (CP) decomposition of the multilinear tensor representing v_i, of rank
at most C, whose factor matrices are constrained to {0,1} and whose core weights are the integer
clause weights.** The learning algorithm is a gradient-free, bandit-driven search over the factor
*support*; the weights are the coefficients.

This is not an analogy. It is the same object the Exponential Machines paper factorizes
`[FACT: arXiv:1605.03795 — "one weight per subset of features"]`, reached from the other side: **ExM
keeps all 2^{2F} coefficients and makes them low-rank; a TM keeps C of them and makes them sparse.**
Sparse-in-the-monomial-basis versus low-rank are the two classical ways to tame the same
exponentially large tensor.

*Consequence.* "Apply tensor factorization to a TM" is not a well-posed addition. Any real
factorization of (★) replaces the Boolean factor constraint with a real-valued one — at which point
the model is an FM or an ExM, and the clause is no longer readable as a propositional rule and no
longer executable as an AND gate. Those two properties are the TM's entire reason to exist.

### 3.2 Corollary — clause count is bounded below by tensor rank

Since (★) exhibits v_i as a CP decomposition with C terms, **C ≥ rank_R(T_i)**, the real CP rank of
the target's multilinear tensor, and generally C ≫ rank_R(T_i) because the factors are restricted
to {0,1}. This gives a principled lower-bound tool for `experiments/convtm/THEORY.md` §4 ("capacity"):
*any* clause budget is above the tensor rank of whatever the class function is. It also reframes the
measured capacity plateau — flat validation at 20k/40k/80k `[MEASURED: ctm-therm5-{20k,40k,80k}]` —
as evidence about the *reachable* part of the decomposition, not about the rank of the target.
`[HYPOTHESIS]` — the lower bound is sound; the reading of the plateau is not yet measured.

### 3.3 Proposition B — the whole cost is the Boolean factor constraint

Parity of k bits is the cleanest witness.

    parity(x) = ( 1 − Π_{i=1..k} (1 − 2x_i) ) / 2

The product term is a **single rank-1 multilinear term**: CP rank 2 with real factors. In the
conjunction basis with {0,1} factors, parity requires **2^{k−1} clauses**. So the two representations
are genuinely incomparable, and the gap is exactly the factor alphabet.

This lands precisely on the recorded limits of a CTM layer:
`THEORY.md` §3.3 Corollary 1 ("XOR of presences is unreachable"), §3.5 Corollary 3 ("blind to object
count") — and on the measured synthetic result that a hand-built 2-layer stack reaches 99.4 % on an
XOR-of-presences task where one layer is **provably capped at 75 %**
`[MEASURED: experiments/mctm, oracle layer-1 arm]`.

**So there is a real theoretical case for a factorized component, and it is narrow and specific:
parity-like and counting-like structure over clause presences.** The open question is whether
CIFAR-10 needs it — §5.

### 3.4 Proposition C — the match test factorizes *exactly*, and the factorization is frequent-itemset mining

A pleasant and, I believe, unpublished observation. The library computes
`violations = (1 − ℓ) @ includeᵀ` and tests `violations == 0` `[FACT: functional.py:52-63]`.

Choose sub-patterns L_1,…,L_r ⊆ [2F] and S ∈ {0,1}^{C×r} such that ∪_{k : S_{jk}=1} L_k = I_j for every
clause j (a **cover**, not an exact factorization). Let u_k(x) = Σ_{i∈L_k} (1 − ℓ_i(x)) ≥ 0. Then

    Σ_k S_{jk} u_k(x) = Σ_i m_{ji} (1 − ℓ_i(x)),   m_{ji} ≥ 1 for i ∈ I_j,  m_{ji} = 0 otherwise,

and since every term is non-negative, the sum is 0 **iff** every included literal is true. Double
counting from overlapping L_k is harmless *because the test is a zero test on a non-negative sum*.
The two-stage evaluation is therefore **exact**, not approximate — unusual, and worth stating.

Now the arithmetic, on our own flagship (2F = 1308, C = 20 000):

    speedup = 2F·C / ( r·(2F + C) ) = 1228 / r

- r = 100 → 12.3× ; r = 500 → 2.5×.
- The constraint that kills it: each L_k may only be used by clauses whose literal set **contains**
  it, so the L_k must be **frequent itemsets of the clause dictionary**. Beating sparsity's 47×
  requires r ≤ 26 — i.e. 20 000 clauses' 28-literal sets all expressible as unions from a family of
  26 sets over a 1308-literal alphabet. Not credible.
- And it competes against the measured 47× headroom that *plain sparsity* offers with no search at
  all — already taken by Sparse TM, Contracting TM, clause indexing (§1.6).
- And `include` changes every batch, so the cover must be re-mined (NP-hard exactly; greedy set
  cover gives ln n) at a cadence that amortizes.

**Verdict: correct, elegant, and dominated.**

### 3.5 Where approximation is structurally inadmissible

Anything that touches the learning loop must be exact, for reasons specific to this codebase:

- **The match test is an exact zero test.** Any approximate low-rank `include ≈ UV` makes
  `violations` a small non-zero number and every clause matches everything. There is no tolerance to
  tune: the quantity is a count of violated literals `[FACT: functional.py:52-63]`.
- **Inclusion is a ratchet.** Only Type Ib removes an included literal, at rate 1/s; admission asks
  only whether a literal is frequent in the clause's own match set `[MEASURED: THEORY.md §5.5.2,
  clause length 86.7 → 170.6 over 30 epochs]`. A perturbed state does not relax back.
- **`apply_feedback` uses the count tensors as in-place scratch** `[FACT: CLAUDE.md]` — no room to
  smuggle a factorized path in without touching `src/`, which is frozen under C1.
- **Batched feedback already trades fidelity for throughput** `[FACT: docs/benchmarks.md]`. A second
  approximation on top of it would be unattributable.

Therefore: factorization is admissible **before** the TM (features), **after** the TM (the head), or
as an **exact** reformulation (§3.4). Never inside the automata.

---

## 4. The five insertion points, scored

| # | Route | What it would do | Verdict |
|---|---|---|---|
| **A** | **Front-end**: Tucker/CP/NMF on the image tensor → thermometer → TM | new Booleanization, a TM Composites "specialist" | **Attacks the wrong loss.** Our own decomposition says Booleanization costs ~8.6 pp (cnn-small 92.71 → cnn-boolean 84.11) while the **TM mechanism costs ~19 pp** (84.11 → best CTM 65.44) `[MEASURED: cnn-small_seed0, cnn-boolean_seed0, ctm-therm5-40k_seed0]`. Ceiling is 8.6 pp, and rank-r global coefficients are a *worse* TM input than HOG: dense, global, non-local, and unreadable as a rule. Cheap to try; low expected value. |
| **B** | **Dictionary**: Boolean factorization / cover of `include` | exact compute reuse (§3.4) | **Correct and dominated.** 1228/r × vs sparsity's measured 47×. Engineering, not research. |
| **C** | **State**: low-rank `ta_state` | memory | **Dead.** Learning needs exactness (§3.5); for inference you need 1 bit/automaton, and the sparse form is 556 514 literal indices at int16 ≈ 1.11 MB against 104.6 MB of int8 state — ~94× by exact means. A 2.13 %-dense binary matrix has no useful low-rank structure to find. |
| **D** | **Weights**: factorize (C, K) | parameters | **Dead for this programme.** K = 10, so the matrix has rank ≤ 10 already, and it is 80 KB of an 104.6 MB model. Only interesting at large K (multi-label, vocabulary-scale NLP). |
| **E** | **Head**: replace the linear vote with a low-rank multilinear form over Φ (FM / TT) | lifts §3.3's limits | **The only live route.** See §5. |

---

## 5. The one experiment worth running — the frozen-Φ head ladder

**The question it answers.** Proposition 1 of `THEORY.md` says a CTM is *exactly* a linear model
over the Boolean presence vector Φ(x) ∈ {0,1}^C. Route E asks: **how much accuracy is the word
"linear" costing on CIFAR-10?** No amount of theory answers this; one afternoon of compute does.

**Why this is the right probe, and not a second TM layer.** A second TM layer is the same idea in
Boolean clothing, and it was tried: the 2-layer stack scored 15.37 ± 0.98 % against 34.36 ± 0.97 %
for its own frozen layer 1 alone, and **random** layer-1 clauses beat trained ones 37.60 vs 15.37
`[MEASURED: experiments/mctm, 10 arms × 3 seeds]`. The blocker was never representation — it was
credit assignment, which failed at four successive points, the last two irreparably
(no quantity sets the Ia:Ib ratio; the meaning of "channel k" drifts underneath the layer above).
**A gradient-trained head over a frozen Φ has no credit-assignment problem at all.** That is the
whole argument for doing E rather than re-attempting MCTM.

**Protocol.**

1. Re-run `ctm-therm5-preflight` (3.43 h, seed 0) `[MEASURED: wall_s = 12 340]`, and dump
   Φ ∈ {0,1}^{N×20000} for the fixed 45k/5k/10k split **with `model.eval()`** — empty clauses evaluate
   True while training and False when predicting, and getting this wrong silently changes every
   number `[FACT: CLAUDE.md; tests/test_models.py]`. 900 MB as uint8, 112 MB bit-packed.
2. Fit four heads on the *same* frozen Φ, selecting on the 5k val split only:
   - **H0 — the TM's own vote** (the arm's accuracy): 65.10 %.
   - **H1 — logistic regression**: the best *linear* head. Separates "the vote layer is badly fitted"
     from "the vote layer is badly *specified*". This distinction has, as far as I can find, never
     been measured in the TM literature.
   - **H2 — rank-r factorization machine** (r ∈ {8, 16, 32, 64}): all pairwise clause co-presences at
     O(Cr). This is the order-2 CP of §1.3 and directly supplies the cross-patch conjunction that
     `THEORY.md` §3.4 proves one layer cannot reach.
   - **H3 — a 2-layer MLP**: the *ceiling* of any nonlinear head. H3 − H1 upper-bounds what route E
     can ever be worth. If H3 ≈ H1, no factorized head helps, at any rank, ever.
3. Report mean ± sd over 3 seeds, paired McNemar on shared test predictions, per `CHARTER`.

**Cost**: ~4–6 GPU-h for seed 0, ~12–18 GPU-h for three seeds — about 5 % of the programme's budget.

**Pre-registered decision rule** (written before the measurement, per C3):

| Outcome | Reading | Action |
|---|---|---|
| H3 − H1 ≤ seed band (~0.8 pp) | Φ is the bottleneck; the linear vote is *not* | **Close route E permanently.** Publishable negative result: it converts `THEORY.md`'s Corollaries 1–3 from "unreachable in principle" to "empirically not what costs the points on CIFAR-10", and redirects all effort to the dictionary. |
| H3 − H1 ≈ 1–3 pp | marginal | Not worth a factorized head; note and stop. |
| H3 − H1 > 3 pp **and** H2 recovers most of it at r ≤ 32 | the linear vote is binding, and low rank suffices | Route E is alive. Next question — the hard one — becomes whether Φ can be *trained* against a factorized head without reopening MCTM's credit problem. |

**Falsifier for the whole direction**: H3 ≈ H1. One number, ~4 GPU-hours, kills or opens it.

**Bonus**: H1 alone is worth having regardless. It measures whether the TM's bandit-driven weight
learning even reaches the best linear fit over its own clauses — a diagnostic nobody in the TM
literature reports, and one that slots straight into the existing report's decomposition spine
(Booleanization loss + binary-computation loss + **vote-layer loss** + dictionary loss).

---

## 6. Final verdict

**Do not open a tensor-factorization workstream.** The reasons, in order of force:

1. **It is redundant at the level of theory.** §3.1: the TM is already a CP decomposition. The
   literature's factorizations differ from it in one respect — real factors instead of Boolean — and
   that difference *is* interpretability and multiply-free inference. You would be trading away the
   two claims that make the TM publishable to buy expressivity on parity-like structure.
2. **It aims at the wrong loss.** Our own numbers put ~8.6 pp on Booleanization and ~19 pp on the TM
   mechanism `[MEASURED: cnn-small / cnn-boolean / ctm-therm5-40k]`. Front-end tensor features
   (route A) are capped at the 8.6 and are a worse input than HOG.
3. **The compression case is arithmetically lost before it starts.** 1228/r × against a measured 47×
   available from sparsity alone, against four published TM papers that already took it (§1.6, §3.4).
4. **The hardness results bite where it would matter.** Boolean rank is NP-hard and inapproximable
   within any polynomial factor `[FACT: §1.4]`, and `include` changes every batch.

**Do run the frozen-Φ head ladder (§5).** It is not really an experiment about tensor factorization;
it is the experiment that tells you whether the *only* factorization worth building (a low-rank
multilinear head) has any headroom at all — and it returns a useful, publishable number in **either**
direction, for ~4 GPU-hours, without touching `src/`.

If you want the honest one-sentence version: *tensor factorization is not a new tool for Tsetlin
machines, it is the continuous mirror of what they already do — so the only question worth asking is
whether relaxing the Boolean constraint on the last layer buys anything, and that costs one
afternoon to find out.*

---

## 7. Sources

Tensor methods —
[Kolda & Bader, Tensor Decompositions and Applications](https://www.kolda.net/publication/TensorReview.pdf) ·
[Stoudenmire & Schwab, Supervised Learning with Quantum-Inspired Tensor Networks (arXiv:1605.05775)](https://arxiv.org/pdf/1605.05775) ·
[Novikov et al., Exponential Machines (arXiv:1605.03795)](https://arxiv.org/abs/1605.03795v3) ·
[Blondel et al., Polynomial Networks and Factorization Machines (arXiv:1607.08810)](https://arxiv.org/abs/1607.08810) ·
[Multi-output Polynomial Networks and FMs (arXiv:1705.07603)](https://arxiv.org/pdf/1705.07603) ·
[Freudenthaler & Rendle, Factorized Polynomial Regression](https://www.ismll.uni-hildesheim.de/pub/pdfs/FreudenthalerRendle_FactorizedPolynomialRegression.pdf) ·
[Tree tensor networks with CP-rank constraints (arXiv:2305.19440)](https://arxiv.org/pdf/2305.19440) ·
[Tensor-network-constrained kernel machines (arXiv:2403.19500)](https://arxiv.org/pdf/2403.19500)

CNN compression —
[TEC-CNN (ACM TOMM 2024)](https://dl.acm.org/doi/10.1145/3702641) ·
[Hierarchical Tucker-2 compression](https://www.sciencedirect.com/science/article/abs/pii/S156849462200905X) ·
[Stable low-rank CP decomposition (Applied Sciences 2024)](https://www.mdpi.com/2076-3417/14/4/1491)

Boolean / discrete factorization —
[Miettinen, Boolean tensor factorizations](https://www.mpi-inf.mpg.de/~pmiettin/btf/) ·
[Recent Developments in Boolean Matrix Factorization (IJCAI 2020)](https://www.ijcai.org/proceedings/2020/0685.pdf) ·
[MDL4BMF (ACM TKDD)](https://dl.acm.org/doi/10.1145/2601437) ·
[Walk'n'Merge (arXiv:1310.4843)](https://arxiv.org/pdf/1310.4843) ·
[Geometric All-Way Boolean Tensor Decomposition (NeurIPS 2020)](https://proceedings.neurips.cc/paper/2020/file/1def1713ebf17722cbe300cfc1c88558-Paper.pdf) ·
[Bayesian BMF (arXiv:1702.06166)](https://arxiv.org/pdf/1702.06166) ·
[Engineering Boolean Matrix Multiplication (arXiv:1909.01554)](https://arxiv.org/pdf/1909.01554) ·
[Method of Four Russians](https://en.wikipedia.org/wiki/Method_of_Four_Russians)

Tsetlin machines —
[Granmo, The Tsetlin Machine (arXiv:1804.01508)](https://arxiv.org/pdf/1804.01508) ·
[Weighted TM (arXiv:1911.12607)](https://arxiv.org/abs/1911.12607) ·
[Clause Indexing (arXiv:2004.03188)](https://arxiv.org/pdf/2004.03188) ·
[Sparse TM (arXiv:2405.02375)](https://arxiv.org/pdf/2405.02375) ·
[Contracting TM with Absorbing Automata (arXiv:2310.11481)](https://arxiv.org/pdf/2310.11481) ·
[ETHEREAL (arXiv:2502.05640)](https://pith.science/paper/2502.05640) ·
[Drop Clause (arXiv:2105.14506)](https://arxiv.org/abs/2105.14506) ·
[TM Embedding (arXiv:2301.00709)](https://arxiv.org/abs/2301.00709) ·
[Omni TM-AE (arXiv:2505.16386)](https://pith.science/paper/2505.16386) ·
[Optimized Toolbox / TM Composites, 82.8 % CIFAR-10 (arXiv:2406.00704)](https://arxiv.org/html/2406.00704v2) ·
[TMComposites (arXiv:2309.04801)](https://arxiv.org/abs/2309.04801) ·
[The Tsetlin Machine Goes Deep (arXiv:2507.14874)](https://arxiv.org/html/2507.14874v2)

Not relevant despite the name —
[Logic Tensor Networks (arXiv:2012.13635)](https://arxiv.org/abs/2012.13635)
