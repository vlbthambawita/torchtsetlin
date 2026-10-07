# How a Tsetlin machine learns

This page summarises the algorithm implemented by `torchtsetlin`, using the vocabulary of
Granmo's textbook *An Introduction to Tsetlin Machines* and the original paper
(Granmo, 2018). If you know the book, the mapping is:

| Book | Paper / code |
|---|---|
| Memorize / Forget value | `(s-1)/s` and `1/s` (specificity `s`) |
| Recognize feedback | Type Ia feedback |
| Erase feedback | Type Ib feedback |
| Reject feedback | Type II feedback |
| Vote margin | threshold `T` |
| memory position 1…2N | automaton state `0…2N-1`, included when `>= N` |

## Literals, clauses and votes

An example is a vector of `F` Boolean features. Its **literals** are the features and their
negations, `[x, NOT x]` (length `2F`). A **clause** is an AND of a subset of literals; it
*matches* an example when all its literals are True. Each clause belongs to a class and has a
**polarity**: positive clauses vote *for* the class, negative clauses vote *against* it.

The class sum (vote sum) is

$$v_k = \sum_{j \in \text{pos}(k)} w_j c_j(x) - \sum_{j \in \text{neg}(k)} w_j c_j(x),
\qquad \hat y = \arg\max_k \operatorname{clip}(v_k, -T, T)$$

with `w_j = 1` for the standard TM (integer weights for the weighted TM).

In everyday terms: the machine turns the input into a list of yes/no facts, each clause is a
small checklist that fires only when *every* item on it is true, and the class with the most
net votes wins.

<figure markdown="span">
  ![Four clauses checking a sunny, windy weekend and voting on whether it is a beach day](../assets/concepts/tm-clauses.svg#only-light){ loading=lazy } ![Four clauses checking a sunny, windy weekend and voting on whether it is a beach day](../assets/concepts/tm-clauses_dark.svg#only-dark){ loading=lazy }
  <figcaption>Deciding “beach day?” for a sunny, windy weekend. Each clause is an AND of literals; it votes only if all of its literals are true. Two clauses vote for, one against, so the total is +1 and the answer is yes.</figcaption>
</figure>

While **learning**, an empty clause (no included literals) evaluates to True so that it can
start memorising; while **predicting** it evaluates to False.

## Memory: Tsetlin automata

Every (clause, literal) pair owns a **Tsetlin automaton** whose state is an integer in
`[0, 2N-1]`. States `>= N` mean the literal is *included* (memorised) in the clause, states
`< N` mean it is *excluded* (forgotten). `N` (`n_states`) is the memory depth: deep states
are hard to change, states near the boundary flip easily. Automata start just below the
boundary (`N-1`), "about to be memorised but currently forgotten".


Think of each automaton as a token on a track: rewards push it towards *remember*,
penalties towards *forget*, and the literal is part of the clause only while the token sits
on the remember side. A token deep in either half needs many consistent signals to change
its mind — that is what makes learning robust to noise.

<figure markdown="span">
  ![A token moving along an eight-position track that decides whether Sunny is in the clause](../assets/concepts/tm-automaton.svg#only-light){ loading=lazy } ![A token moving along an eight-position track that decides whether Sunny is in the clause](../assets/concepts/tm-automaton_dark.svg#only-dark){ loading=lazy }
  <figcaption>One Tsetlin automaton with <code>N = 4</code> (8 states). The token starts just left of the line (<code>N-1</code>); each step right is a reward, each step left a penalty. <em>Sunny</em> is in the clause only while the token is on the right.</figcaption>
</figure>

## Feedback

For a training example of class `y`:

1. Evaluate all clauses and compute the clamped vote sums.
2. Draw a random negative class `q != y`.
3. Each clause of class `y` receives feedback with probability
   $(T - v_y) / 2T$: **Type I** if it is positive, **Type II** if it is negative.
4. Each clause of class `q` receives feedback with probability
   $(T + v_q) / 2T$: **Type II** if it is positive, **Type I** if it is negative.

The probabilities implement the *vote margin*: once a class already wins by `T` votes,
its clauses stop being updated, which frees them to specialise on other examples
(resource allocation).

<figure markdown="span">
  ![Chance of feedback falling with the correct class's vote sum and rising with a wrong class's](../assets/concepts/tm-vote-margin.svg#only-light){ loading=lazy } ![Chance of feedback falling with the correct class's vote sum and rising with a wrong class's](../assets/concepts/tm-vote-margin_dark.svg#only-dark){ loading=lazy }
  <figcaption>The vote margin <code>T</code>. The class of the example is taught with probability <code>(T - v)/2T</code> — always when it clearly says no, never once it wins by <code>T</code>. A randomly drawn wrong class is corrected with probability <code>(T + v)/2T</code>.</figcaption>
</figure>

**Type I feedback** makes a clause *recognise* frequent patterns:

* If the clause matches the example (**Type Ia** / Recognize): every True literal is
  memorised (state `+1`) with probability `(s-1)/s` — or with probability 1 when
  `boost_true_positive=True`, the default — and every False literal is forgotten (`-1`)
  with probability `1/s`.
* If the clause does not match (**Type Ib** / Erase): all literals are forgotten with
  probability `1/s`.

**Type II feedback** makes a clause *reject* examples of other classes: if the clause
matches, every False literal that is currently excluded moves one step towards inclusion
(deterministically). As soon as one such literal is included the clause no longer matches.

<figure markdown="span">
  ![Type I feedback teaching a clause Weekend from a beach day, and Type II feedback adding Weekend to reject a weekday](../assets/concepts/tm-feedback.svg#only-light){ loading=lazy } ![Type I feedback teaching a clause Weekend from a beach day, and Type II feedback adding Weekend to reject a weekday](../assets/concepts/tm-feedback_dark.svg#only-dark){ loading=lazy }
  <figcaption>The same clause, <em>Sunny</em>, under each kind of feedback. <strong>Type I</strong> (an example of its own class): true literals step towards remember, false ones occasionally towards forget. <strong>Type II</strong> (a look-alike from another class): false literals step towards remember, so the clause stops matching. Both end at <em>Sunny AND Weekend</em> by different routes.</figcaption>
</figure>

Larger `s` means forgetting is rarer, so clauses keep more literals and become more
specific. `T` controls how many clauses cooperate on each example; larger `T` needs more
clauses.

## Variants implemented

* **Weighted / integer-weighted TM** (`weighted=True`): each clause has an integer weight
  that grows by 1 on Type Ia and shrinks by 1 on Type II feedback.
* **Coalesced TM** (`CoalescedTsetlinMachine`): one pool of clauses shared by all outputs,
  each with a signed integer weight per output. Weights move away from zero on Type I and
  towards zero on Type II, so a clause can change sides. Supports multi-label targets.
* **Regression TM** (`RegressionTsetlinMachine`): all clauses positive, `v` in `[0, T]`
  mapped linearly to the target range; Type I when the prediction is too low, Type II when
  too high, with probability proportional to the error.
* **Convolutional TM** — see [Convolution](convolution.md).
* **Clause size constraint** (`max_included_literals`): clauses above the budget receive
  only forgetting feedback until they shrink (Abeyrathna et al., 2023). The budget is a hard
  limit: a batched commit never grows a clause past it, however many events it aggregates.
* **Drop clause / drop literal** (`drop_clause_p`, `drop_literal_p`): randomly excluded
  clauses neither vote nor learn for one batch or epoch (Sharma et al., 2023).
* **Focused negative sampling**: choose the negative class proportionally to its current
  vote sum, i.e. correct confusions first.

## Where each piece lives

| Step | Function / method |
|---|---|
| literals | `functional.to_literals` |
| clause evaluation | `functional.clause_outputs` (one matmul + comparison) |
| vote sums | `functional.vote_sums`, `model._votes` |
| feedback probabilities | `functional.feedback_probabilities` |
| which clause gets which feedback | `model._select_feedback` |
| counting feedback events over a batch | `functional.type_i_counts`, `functional.type_ii_counts` |
| moving the automata | `functional.apply_feedback` |

## References

* O.-C. Granmo, *The Tsetlin Machine – A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic*, arXiv:1804.01508 (2018).
* O.-C. Granmo, *An Introduction to Tsetlin Machines*, tsetlinmachine.org.
* A. Phoulady et al., *The Weighted Tsetlin Machine*, arXiv:1911.12607 (2019).
* S. Glimsdal, O.-C. Granmo, *Coalesced Multi-Output Tsetlin Machines with Clause Sharing*, arXiv:2108.07594 (2021).
* K. D. Abeyrathna et al., *Building Concise Logical Patterns by Constraining Tsetlin Machine Clause Size*, arXiv:2301.08190 (2023).
* J. Sharma et al., *Drop Clause: Enhancing Performance, Interpretability and Robustness of Tsetlin Machines*, AAAI (2023).
