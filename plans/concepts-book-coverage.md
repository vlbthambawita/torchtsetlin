# Concept pages vs. the book chapters: coverage audit and update plan

*Status: **implemented**, 2026-11-05, except the optional empirical check under "Open
questions", which needs a go-ahead for seeded runs.*

Implementation notes:

* `python scripts/make_concept_figures.py --check` reproduces every Chapter 2 value the book
  prints: single literal 0.04 / 0.87 / 0.9999 / 1.0, two literals 0.16 / 0.07 / 0.36 to two
  decimals.
* The two-rule example does not state `s`. `s = 5` reproduces the book's peaks (0.31 and
  0.16; computed 0.306 and 0.154). The book says the bad configuration has zero probability
  with the margin; the exact chain gives 0.3 %. The page says so.
* The library's `confidence()` clips at `T`, so it keeps the book's ranking only below `T`;
  `confidence.md` says to rank by the raw votes.
* `mkdocs build --strict` fails on two missing PDFs linked from `docs/cite.md`, which predate
  this work; the new pages add no warnings.

Sources audited (`source_documents/book_chapters/`): Granmo, *An Introduction to Tsetlin
Machines*, Chapter 1 (revised, plus its companion notebook), Chapter 2 (Classification),
Chapter 4 (Convolution), Chapter 7 (Confidence, Trustworthiness and Composites). Pages audited:
everything under `docs/concepts/`.

## TL;DR

The *mechanics* of Chapters 1 and 4 are covered: literals, clauses, voting, automata, the three
feedback types, the vote margin, convolution and position literals. What is missing is the
*explanation of why it works*:

| Chapter | Coverage | Biggest gap |
|---|---|---|
| 1 Your first TM | mechanics yes, intuition partly | the three strategies (frequent patterns / discrimination / dissection), `s` as a frequency threshold, how rules coordinate |
| 2 Classification | **none** | Markov-chain analysis, literal game, rule interaction and why `T` avoids local optima |
| 4 Convolution | ~90 % | clauses specialising on different parts of one image; why `T > 2` matters for that |
| 7 Confidence & composites | API only, in a *guide* | no concept page; library confidence differs from the book's; composite snippet normalises per batch |

Chapter 1's notebook adds nothing beyond Chapter 1's text.

## Gap list

### Chapter 1 → `concepts/tsetlin-machine.md`

Covered: Boolean features, AND/NOT clauses, literals, majority vote, rule memory and its
initialisation, Recognize/Erase/Reject (= Type Ia/Ib/II, with the mapping table), vote margin
and `(T - v)/2T`, resource allocation, specificity.

Missing:

1. **The three strategies framing.** The book's own summary: Type Ia/Ib = frequent pattern
   mining, Type II = discrimination, vote margin = data dissection. The page lists the
   feedback types but never says what each one is *for* as a whole.
2. **`s` as a frequency threshold, with the arithmetic.** The book's breast-cancer example:
   with memorize 0.2 / forget 0.8, a literal true for 2 of 3 patients gains 0.4 and loses 0.8
   per pass, so it is forgotten; at 0.5 / 0.5 it is kept. This is the clearest statement of
   "a literal survives if it is true in more than about `1/s` of the class's examples". The
   hyperparameters page has the specificity figure but not this reasoning.
3. **Book vs library feedback values.** Chapter 1 uses *Memorize + Forget = 1* (e.g. 0.1/0.9);
   Chapter 2 and the library use memorize `(s-1)/s` (or 1 with `boost_true_positive`) and
   forget `1/s`. One sentence plus the mapping-table row avoids confusion for book readers.
4. **Contradictions vanish on their own.** Including `x` and `NOT x` makes a clause that is
   always False; it then only receives Type Ib and is forgotten. Short remark.
5. **How clauses coordinate (Section 1.6 "Coordination example").** Type II only reaches
   clauses that voted for the wrong class, shielding correct ones; precise, frequent clauses get
   the most Type Ia and are memorised deepest; duplicated clauses are useful because they
   reinforce each other's vote. Add as a short subsection after "Feedback", reusing the
   existing vote-margin figure plus a small table of update probabilities at `T = 2`
   (`v = -2, 0, 1, 2` → `1, 0.5, 0.25, 0`).
6. **Initialisation does not matter in the limit**, which is why `init="boundary"` and
   `init="random"` both work. One sentence; links forward to the Chapter 2 page.
7. **Why Booleans** (bitwise hardware, propositional logic as the interpretability argument;
   universal approximator + rule-based + summation-based). Two sentences in the intro at most.

### Chapter 2 → new page `concepts/why-it-learns.md` ("Why a Tsetlin machine learns what it learns")

Not covered anywhere. This is the main deliverable.

1. **One literal is a Markov chain.** The four observation kinds `(L,Y)`, `(¬L,Y)`,
   `(L,¬Y)`, `(¬L,¬Y)` map to Recognize, Erase, Reject and no feedback. The transition matrix
   and its stationary distribution.
2. **What the stationary distribution says** (the book's six factors, Table 2.1): frequent
   literals are memorised, infrequent ones forgotten, `s` moves the frequency threshold,
   discriminative literals are preferred, and **each extra memory state sharpens the
   decision exponentially**. That last point is the real meaning of `n_states` and belongs on
   the hyper-parameters page too.
3. **The four single-literal scenarios** as one figure: memorise probability 0.04 → 0.87 when
   `s` goes 1 → 15, 0.9999 for a discriminative literal, ~1.0 with 8 states. Verified: a
   30-line numpy power iteration reproduces 0.043, 0.872, 1.000, 1.000.
4. **Literals play a game.** Two literals decide independently whether to join the clause;
   the Prisoner's Dilemma framing; the 16-state chain. Outcomes from Tables 2.5–2.8: picks the
   informative literal, keeps several alternatives alive, prefers the best one, combines both
   when `s` is raised to allow rarer, more precise patterns.
5. **Rules interact through `T`.** The `x1 OR x2` example: without a vote margin both clauses
   collapse to `x1 AND x2` (a local optimum); with `T = 2` they split into `x1` and `x2` and the
   bad configuration has probability zero. This is the formal reason behind "raise `T` with
   the number of clauses".
6. **Learning never stops.** The outcome is a distribution over solutions, independent of
   initialisation; the machine keeps revisiting alternatives. This explains run-to-run
   variance and why clause sets keep changing late in training.
7. **Practical takeaways box** linking each result to a parameter (`s`, `n_states`, `T`).
8. **Scope caveat.** The analysis is for the classical, sequential, unweighted algorithm. Say
   explicitly that batched feedback, clause weights and the clause budget are outside it, and
   link to [Batched vs sequential feedback](batching.md).

### Chapter 4 → `concepts/convolution.md`

Covered: patches, OR over patches, random matching patch for Type Ia/II, no patch for Type Ib,
position literals as thermometers, 1D convolution, thermometer vs binary trade-off (in
`booleanization.md`).

Missing (small):

1. **The counting argument up front**: without convolution, Circle needs one rule per
   location (4 in a 4×4 image with a 3×3 window), with it one rule regardless of image size.
   The page states this for 3D but not for the basic 2D case.
2. **Clauses specialise on different parts of the image** (the book's truck figure) and a
   larger `T` is what makes several of them cooperate on one image. One paragraph.
3. **"Not target, no match → no feedback"** is implicit; state the four cases as a small
   table (match/no match × target/not target), mirroring the book's Figure 4.5.
4. Position-literal motivation (the top stroke of a 7 vs the bottom of a 2) as the example
   for when `position_encoding=True` *is* right. Complements the existing "turn it off for
   translation-invariant tasks" tip.

### Chapter 7 → new page `concepts/confidence.md` ("Confidence and composites")

Today only `guides/evaluation.md` touches it, with API calls and no explanation.

1. **Confidence from the vote sum.** Book: `|v|` for one binary machine, `max_k v_k` for
   multi-class. Library: `model.confidence()` returns `(v_max + T) / 2T`. Explain that this is
   a monotone rescaling of the book's definition, so rankings and trustworthiness curves are
   unchanged, and say why the library rescales (a `[0, 1]` number across models with
   different `T`).
2. **Trustworthiness** as the 2×2 table (justifiably confident / overconfident /
   underconfident / justifiably uncertain) and the accuracy-above-threshold curve, worked on
   the book's six-student example; point to `metrics.trustworthiness_curve` and
   `viz.plot_trustworthiness`.
3. **Composites.** Specialists trained on different Booleanizations or populations output
   vote sums; each is divided by its own range `max - min`, the normalised sums are added,
   and the arg-max decides. Cite the CIFAR-10 reproduction
   (`experiments/multi_convtm/REPORT.md`: +11.8 pp over the best specialist, from its JSON).
4. **Fix the guide snippet.** The `composite_votes` example in `guides/evaluation.md`
   computes `v.max() - v.min()` over whatever batch it is given, so an example's prediction
   depends on the other examples in its batch, and batch size changes results. The book and
   the paper fix the factor once per machine over a reference set. Change the snippet to
   take precomputed factors (fit on a validation set), and say so on the concept page.
   `experiments/multi_convtm/PROPOSAL.md` Step 6 proposes a `tt.TMComposite`; if that lands,
   the page should use it instead, but the docs fix does not depend on it.

## Implementation steps

1. `tsetlin-machine.md`: items Ch1-1 to Ch1-7. Text only, plus one small table.
2. New `why-it-learns.md` for Chapter 2. Add generated figures to
   `scripts/make_concept_figures.py` (light + `_dark` pairs, static is fine; any animation
   must end on the finished picture), computed by a deterministic power iteration inside the
   script, not hand-typed:
   * `markov-literal` — the 4-state chain with its three feedback transitions;
   * `markov-scenarios` — bar charts of the four single-literal stationary distributions;
   * `literal-game` — 8×8 heat map of Table 2.5 vs 2.8 (pick-one vs combine);
   * `rule-interaction` — where the probability mass sits without vs with `T = 2`.
   The two-literal and two-rule chains need the full transition construction from
   Tables 2.4 and 2.9; check the computed peaks against the book's tables before drawing.
3. `convolution.md`: items Ch4-1 to Ch4-4. Text plus a 2×2 feedback table.
4. New `confidence.md` for Chapter 7, plus the snippet fix in `guides/evaluation.md`.
5. `hyperparameters.md`: one line each linking `s`, `n_states` and `T` to the Chapter 2 page.
6. `mkdocs.yml` nav: insert *Why it learns* after *How a Tsetlin machine learns*, and
   *Confidence and composites* after *Hyper-parameters*. Add both to the concept list in
   `docs/index.md`.
7. Check: `mkdocs build --strict`, both themes render the new SVGs, `ruff check` on the
   figure script.

No library code changes, no training or benchmark runs. The Chapter 2 numbers come from exact
Markov-chain computation, not from training a model.

## Open questions

* Should the Chapter 2 page also show an *empirical* check, e.g. a one-clause
  `TsetlinMachine` with `feedback_mode="sequential"` on scenario data, comparing how often the
  literal is included against the stationary probability? It would tie the theory to this
  library, but it is a seeded run, so it needs the go-ahead.
* Chapters 3 (regression), 5, 6 and 8+ are not in `source_documents/book_chapters/`, so the
  regression, coalesced and weighted variants could not be audited against the book.
