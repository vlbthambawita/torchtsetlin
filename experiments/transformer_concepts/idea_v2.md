# Prompt: positional encoding and key–query–value attention for Tsetlin machines (v2)

> Expanded from `idea.md` ("I want to test and do experiments with positional encoding and attention
> mechanism using Key, Query and Value concept to Tsetlin machine. Make a plan and write a
> mathematical proof, how it works into a .md file."). Paste everything below this line as the task.

---

## Role and goal

You are working in the `torchtsetlin` repository (`/work/vajira/DL2026/torchtsetlin`) inside the
research programme folder `experiments/transformer_concepts/`. Your job in this first round is to
**design, not run**: produce a research plan and a self-contained mathematical write-up that explains
how positional encoding and key–query–value (KQV) attention can be expressed *natively* in a Tsetlin
machine (TM), what each construction can and cannot represent, how it learns under TM feedback, and
which experiments would decide whether it helps.

Deliverables (all in `experiments/transformer_concepts/`):

1. `THEORY.md` — definitions, constructions, propositions and proofs (see "Theory requirements").
2. `PLAN.md` — the experiment plan with arms, tasks, baselines, metrics, preregistered decision
   rules, phases with gates, and a cost estimate (see "Plan requirements").
3. `DECISIONS.md` and `LIBRARY_GAPS.md` — started empty with the headers used by the other
   programmes (`experiments/transformer_like/` is the template).
4. A short `README.md` pointing at the three files above.

Do **not** start training runs, timing runs or seed sweeps. A smoke test of a few seconds to check
that a construction is implementable is fine. Full runs need the user's explicit go-ahead.

## Ground rules (same as the other programmes)

- **C1:** `src/torchtsetlin/**` is read-only. Build everything on top of the public API and the
  documented extension hooks (`docs/guides/extending.md`, `CLAUDE.md` "Architecture"). Library
  defects or missing hooks go into `LIBRARY_GAPS.md` with a minimal reproduction and a fix sketch.
- **C2:** every reported number must come from a `results/*.json` record written by code; nothing
  hand-typed. Reports are generated from the records.
- Deviations from the plan are logged in `DECISIONS.md` with a date and a reason.
- Test-set access is logged; selection uses validation data only.
- Mark clearly what is proven, what is conjecture, and what is an empirical question.

## What already exists — read first, do not repeat it

1. **`experiments/transformer_like/`** (finished 2026-10-17; read `report/REPORT.md`,
   `DECISIONS.md`, and §20 of `tm-query-key-attention-mnist-implementation-plan.md`).
   That programme tested *frozen Q/K TMs → real-valued softmax scores → value mixing →
   thresholds → head TM* on MNIST 7×7 patch tokens. Outcome: the attention model (86.3 %) lost to
   the flat TM (92.1 %) and the conv TM (93.7 %), and also to self-only attention and to direct Q/K
   features. QK attention put ~69 % weight on the token itself and replacing it by self-only
   attention at inference cost nothing. Lessons that apply here:
   - The Q/K "coordinates" were supervised class-score vectors, so query and key lived in the same
     10-d digit space; cross-patch similarity carried no extra information. **v2 must use a
     different Q/K coordinate system** (Boolean, learned by clauses, or position-aware), and must
     state why it is expected to carry information the token itself does not.
   - MNIST is a poor testbed for attention: a flat TM already sees all pixels at once, so there is
     nothing to route. **v2 must include tasks where content-dependent routing is necessary.**
   - `T/M = 0.5` made clauses bloat under batched feedback; `T/M ≈ 0.25` and weighted clauses
     were worth 5–10 points to every arm. Fix these before comparing architectures.
2. **Convolutional TM position encoding** (`src/torchtsetlin/models/conv.py`,
   `position_encoding=True`): thermometer-coded patch coordinates appended to each patch's
   literals. Known caveat: it pins clauses to locations and hurts on translation-invariant toys.
3. **Graph TM** (`src/torchtsetlin/models/graph.py`, `GraphTsetlinMachine`): layer-by-layer
   message passing where a node's inbox is the scatter-OR of clause matches at its in-neighbours.
   This is already a *hard, structure-given* attention: the "keys" are the edge list. Relate every
   attention construction you propose to it, and say whether the new construction is a graph TM
   with a learned or content-dependent edge set.
4. Memories / notes: `experiments/mctm/` (greedy layer-wise stacking fails; credit cannot be pushed
   down through clause inclusion) — relevant to any multi-layer attention design.

## Theory requirements (`THEORY.md`)

Write in plain Markdown with LaTeX math (`$...$`, `$$...$$`). Use the library's notation:
literals $\ell = [x, \lnot x] \in \{0,1\}^{2F}$, clause $C_j(x) = \bigwedge_{k \in I_j} \ell_k$,
include set $I_j$ from automaton states $\ge N$, polarity $p_j \in \{+1,-1\}$, vote sum
$v(x) = \sum_j p_j w_j C_j(x)$ clipped to $[-T, T]$, Type I / Type II feedback with specificity $s$.
State the empty-clause convention (True while learning, False when predicting) wherever it matters.

Cover at least the following, each with a definition, a claim, and a proof or a counterexample:

### A. Positional encoding for Boolean inputs

1. Define a sequence/grid input $X = (x_1, \dots, x_L)$, $x_i \in \{0,1\}^F$, and the encodings to
   compare: none (bag of tokens), one-hot position, thermometer (as in `conv.py`), binary / Gray
   code, and **relative** position (thermometer of $i - j$ or of $|i-j|$, used inside pairwise
   clauses).
2. Prove what a *single clause* can express under each encoding: e.g. with thermometer codes a
   clause can express any interval constraint $a \le i \le b$ with at most two literals; with
   binary codes an interval needs $O(\log L)$ clauses or literals in general; with one-hot a clause
   can select one position or *exclude* any set. Give the literal-count bounds.
3. Show how the convolutional OR over patches interacts with absolute position: a clause with
   position literals is a "located detector", without them it is translation-invariant. Prove the
   translation-invariance statement for the conv TM without position encoding, and characterise
   which functions require position literals (e.g. "token A appears before token B").
4. Discuss the Boolean analogue of sinusoidal/rotary encodings: is there a Boolean code for which
   relative offset is a fixed literal pattern independent of absolute position (shift
   equivariance)? Prove or refute for thermometer and Gray codes.

### B. Key–query–value attention as Boolean selection

Define attention in TM terms. Start from the standard form
$\mathrm{Attn}(i) = \sum_j \alpha_{ij} V(x_j)$ with $\alpha_{ij} = \mathrm{softmax}_j(\langle Q(x_i), K(x_j)\rangle)$
and give TM-native counterparts, for example:

1. **Clause-based Q and K.** $Q(x_i) \in \{0,1\}^{m}$ and $K(x_j) \in \{0,1\}^{m}$ are the outputs of two
   clause banks. Compatibility options: (a) AND-match $a_{ij} = \bigvee_r (Q_r(x_i) \land K_r(x_j))$;
   (b) Hamming/popcount score $\langle Q, K \rangle$ with a threshold or hard top-$k$;
   (c) a **pairwise clause** over the concatenated literals $[\ell(x_i), \ell(x_j), \mathrm{relpos}(i,j)]$,
   i.e. attention as a clause on pairs.
2. **Value and aggregation.** $V(x_j)$ = clause outputs or raw literals; aggregation by OR
   (existential, matches the conv TM), by counting with thermometer thresholds, or by hard
   argmax selection. Prove that OR-aggregation of a pairwise clause equals a clause of the form
   "$\exists j$: condition on $x_i$, $x_j$, $i-j$", and state when it reduces to a conv TM or a graph TM.
3. **Expressivity.** Prove at least: (i) a one-layer pairwise-clause attention TM can represent
   functions a flat TM over the same tokens cannot represent with $o(L^2)$ clauses (or show that
   it cannot, if that is the truth — the flat TM over the full sequence already sees everything);
   be careful and precise about what the separation is (clause count, literal count, or sample
   complexity), since a flat TM is a universal DNF learner; (ii) what attention adds over the conv
   OR (content-dependent pairing vs. a fixed window); (iii) multi-head = multiple clause banks,
   and when heads are redundant.
4. **Complexity.** Cost of each construction in clause evaluations: flat $O(M \cdot LF)$, conv
   $O(M \cdot P \cdot \text{patch})$, pairwise $O(M \cdot L^2)$. Say which fits the existing
   bit-packed kernels and which needs new code.

### C. Learning under TM feedback

This is the hard part and was the weak point of the previous programme (Q/K were frozen and
trained on proxy labels). For each construction:

1. State which automata receive feedback and from what signal. For an OR-aggregated pairwise
   clause, show that the conv TM's feedback rule applies unchanged (pick one matching pair at random
   for Type Ia, as `_feedback_counts` does for patches) and therefore inherits its convergence
   behaviour. Make this the default construction if it holds.
2. For two-stage Q/K/V banks, explain exactly how the class-level feedback reaches Q and K. If it
   requires a new feedback rule, derive it, state it as a new algorithm, and say what
   guarantee (if any) is lost. Use the MCTM result (credit cannot be pushed through clause
   inclusion) as a known obstacle and say how the design avoids it.
3. Discuss the interaction with batched feedback (`feedback_mode="batch"`), the clause budget
   (`max_included_literals`, soft by default) and `T/M`.

### D. Worked example

One small task (e.g. "is token B present somewhere after token A", or associative recall with
$L = 8$) worked by hand: the encoding, a clause set that solves it under the proposed
construction, and the minimum number of clauses / literals compared with a flat TM and a conv TM.

## Plan requirements (`PLAN.md`)

1. **Questions** stated as testable hypotheses (H1: positional encoding variant X improves Y;
   H2: KQV/pairwise attention beats the conv TM and flat TM on tasks needing content-dependent
   routing; H3: it does not hurt on tasks that do not; H4: learned attention patterns are
   interpretable as Boolean rules).
2. **Tasks**, cheapest first, with a sentence on why each needs (or does not need) attention:
   - synthetic sequence tasks with controllable length $L$: ordered-pair detection, associative
     recall (key→value lookup), copy/match-at-distance, majority/parity with positions, and a
     "needle" task where the relevant position varies per example;
   - a bag-of-tokens control where order is irrelevant (positional encoding should not help);
   - one real dataset only after the synthetic gate passes (e.g. sequential MNIST rows, a short
     text classification set, or a medical time series relevant to the user's cardiac MRI work);
     justify the choice.
3. **Arms:** flat TM, conv TM (with and without position encoding), graph TM with a fixed
   all-pairs/windowed graph, each proposed attention construction, plus ablations (shuffled
   positions, attention restricted to self, random pairing). Use weighted clauses and
   `T/M ≈ 0.25` unless the plan argues otherwise. Match automaton counts in a secondary comparison.
4. **Implementation map:** for each arm, which library class/hook it uses (`_ConvMixin`,
   `_GraphMixin`, a new `_encode`/`_evaluate`/`_feedback_counts` override in experiment code),
   expected memory/time on the GPU box, and whether it needs a new kernel. Flag any need to touch
   `src/` as a `LIBRARY_GAPS.md` entry instead.
5. **Protocol:** train/valid/test splits, seeds (5), early stopping on validation, frozen protocol
   file before any test read, test-access log, paired comparisons (McNemar + bootstrap CI), and the
   same "supported / reverse supported / inconclusive" rule as `transformer_like`.
6. **Phases and gates:** theory review → tiny implementation with unit tests against a naive
   reference (pairwise clause evaluation vs. a Python loop; OR-aggregated attention with $L=1$
   reduces bit for bit to the flat TM) → synthetic smoke → user go-ahead → search → freeze → final
   seeds → report. Each gate states its pass criterion.
7. **Risks and kill criteria:** e.g. stop if self-only attention matches full attention on the
   synthetic routing tasks (that was the failure mode last time), or if $O(L^2)$ cost makes
   $L \ge 64$ infeasible.
8. **Out of scope** for this round (e.g. next-token generation, causal masks for language
   modelling, differentiable surrogates) unless the theory shows they are needed.

## Style

- Precise and short; prefer one proved statement over three vague ones.
- Every proposition names its assumptions (input encoding, empty-clause convention, polarity,
  clipping). If a claim depends on batched vs. sequential feedback, say so.
- Cite papers from `source_documents/papers/` and the literature catalog
  (`benchmarks/literature/`) where relevant, and say when an idea is new rather than imply novelty.
- Use they/them for any person mentioned.

When finished, reply with: the list of files written, the main propositions (one line each, marked
proved / conjecture), the recommended first construction and task, and the estimated GPU cost of
the full plan — then wait for the go-ahead before running anything.
