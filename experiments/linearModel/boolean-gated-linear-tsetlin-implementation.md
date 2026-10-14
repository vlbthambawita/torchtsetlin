# Tsetlin Machines with Boolean clauses and finite-state linear outputs

**Implementation design and pseudocode — 14 October 2026**

## Purpose

Implement the original idea from our discussion using an existing Tsetlin Machine (TM): **keep the Boolean clause conditions, and replace each clause's fixed vote with a small linear model that uses the original numerical inputs.** Store each linear coefficient as a bounded integer state multiplied by a fixed step size.

The later proposal to replace conditions with linear gates is outside this design. Conditions remain Boolean conjunctions throughout.

This document is a specification, not an implementation. All algorithm blocks are language-independent pseudocode. Functions named `ExistingTM.*` are adapter responsibilities, not claims that a particular library exposes those exact methods. No MNIST results are claimed.

## Contents

1. [Design decisions](#1-design-decisions)
2. [Reuse and additions](#2-reuse-and-additions)
3. [Numerical inputs and Boolean inputs](#3-numerical-inputs-and-boolean-inputs)
4. [Prediction equations](#4-prediction-equations)
5. [Coefficient memory and model records](#5-coefficient-memory-and-model-records)
6. [Adapter to the existing TM](#6-adapter-to-the-existing-tm)
7. [Train or load the clauses, then freeze them](#7-train-or-load-the-clauses-then-freeze-them)
8. [Prediction pseudocode](#8-prediction-pseudocode)
9. [Training objective and exact state updates](#9-training-objective-and-exact-state-updates)
10. [Coefficient-training pseudocode](#10-coefficient-training-pseudocode)
11. [End-to-end pseudocode](#11-end-to-end-pseudocode)
12. [Worked examples](#12-worked-examples)
13. [Mathematical guarantees](#13-mathematical-guarantees)
14. [MNIST experiment and fair comparisons](#14-mnist-experiment-and-fair-comparisons)
15. [Cost, persistence, and diagnostics](#15-cost-persistence-and-diagnostics)
16. [Verification plan](#16-verification-plan)
17. [Optional extensions](#17-optional-extensions)
18. [Implementation checklist](#18-implementation-checklist)
19. [References](#19-references)

## 1. Design decisions

The first implementation follows this sequence:

```text
Train or load an existing TM
    -> export its learned Boolean clauses
    -> freeze their conditions
    -> attach a linear output to each clause
    -> learn bounded integer coefficient states
    -> sum active contributions and produce the prediction
```

| Decision | Required first version |
|---|---|
| Clause learning | Reuse the existing TM and its usual feedback |
| Clause conditions during coefficient training | Frozen |
| Input to conditions | Boolean features from a saved encoder |
| Input to linear outputs | Original normalized numerical features |
| Classification layout | Keep the existing class-specific clause banks |
| Clause contribution | Boolean activation multiplied by an affine output |
| Coefficient representation | Signed integer state times a fixed step size |
| Coefficient training | Exact full-training-loss comparison of one-state moves |
| Objective | Squared error plus ridge regularization |
| Initial coefficient states | All zero |
| First image task | Nonconvolutional MNIST classification |
| First correctness task | Small regression example with known fixed conditions |

An affine output is a linear combination plus an intercept. This document uses “linear output” in that familiar sense.

The coefficient optimizer is a finite-state coordinate search. It is **not** standard TM Type I/Type II feedback and does not inherit that algorithm's convergence claims. This separation makes the first prototype precise and testable.

Two phases are intentional: the existing TM finds conditions; the new optimizer learns numerical outputs for those conditions. Jointly learning both is a later research problem.

## 2. Reuse and additions

| Existing functionality to reuse | New functionality to add |
|---|---|
| TM training and checkpoint loading | Adapter that exposes frozen clause truth values |
| Include/Exclude automata and learned literals | Numerical input path alongside the Boolean input path |
| Boolean feature encoding, if already available | Signed integer states for intercepts and slopes |
| Clause evaluation in inference mode | Linear output evaluation and aggregation |
| Original prediction method for the baseline | Exact coefficient-state optimizer |
| Existing dataset and split utilities | Hybrid-model checkpointing and diagnostics |

Do not replace a literal's automaton state with a regression coefficient. The original literal states determine which conditions are included. The new coefficient states determine the numbers produced when those conditions match.

A standard TM already has multistate memory; the action selected by a literal automaton is Include or Exclude. The [original TM paper](https://arxiv.org/abs/1804.01508) describes this distinction.

For one rule:

```text
Numerical input x
    |
    +---- saved Boolean encoder ---- frozen AND clause ---- C(x): 0 or 1
    |
    +---- retained numerical values ---- b + sum(w[f] * x[f])

Rule contribution = C(x) * (b + sum(w[f] * x[f]))
```

Keep the source TM checkpoint as a separate baseline. Its training feedback, clause polarities, and original prediction behavior remain part of that source model.

## 3. Numerical inputs and Boolean inputs

### 3.1 Two views of each example

For each sample maintain:

- `X_numeric`: normalized numerical features used by the linear outputs.
- `B`: Boolean features used by the existing TM's conditions.

For MNIST:

\[
d=28\times28=784,\qquad x_f=\text{pixel}_f/255.
\]

Flatten in row-major order: pixel `(row, column)` maps to `28 * row + column`. The 784 grayscale intensities are preserved. Normalization preserves their original 256 levels; it does not add measured precision. The [MNIST loader documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data) gives the image shape, intensity range, and official split sizes.

For a new threshold encoder, a simple initial choice is:

\[
B_f(x)=\mathbf 1[x_f>0.5].
\]

A later experiment may use multiple thresholds per pixel, such as `0.25, 0.5, 0.75`. These values are starting choices, not tuned recommendations. Threshold encoding for numerical inputs is established TM practice; see the [continuous-input TM paper](https://arxiv.org/abs/1905.04199).

If loading a trained TM, **use that checkpoint's exact encoder**, including comparison operators, threshold values, feature order, and complement convention. Do not replace its thresholds with this document's examples.

### 3.2 Canonical literal mapping for a newly trained model

Let `L` be the number of thresholds per numerical feature, and `F = d * L` the number of positive Boolean features.

```text
u = feature_index * L + threshold_index
B[u] = (X_numeric[feature_index] > threshold[threshold_index])

Canonical literal (u, POSITIVE) means B[u]
Canonical literal (u, NEGATIVE) means NOT B[u]
```

Under this encoder, a negative literal means `x <= threshold`, including equality. The adapter translates library-specific literal IDs into these canonical pairs. Some libraries add complements internally; do not append them twice.

### 3.3 Data and target policies

- Split data before fitting any data-dependent preprocessing or encoder.
- Use training data for fitting, validation data for configuration/checkpoint selection, and held-out test data for final evaluation.
- Reject nonfinite numerical inputs in version 1. Add imputation only as an explicit fitted preprocessing step.
- Do not add training-time augmentation during fixed-feature coefficient optimization. If used later, freeze the augmented training set before optimizing.
- For classification, map class labels to indices `0..K-1` and use one-hot coefficient-training targets. Keep the label mapping for inference.
- For scalar regression, set `K = 1`. If target normalization is needed, fit it on training targets, save it, and invert it after prediction. Record separately any target transformation required by an existing regression TM during clause learning.
- An existing multiclass TM alone does not supply a native regression clause learner. Use an existing regression TM for that phase, or use supplied fixed clauses for the regression verification task.

## 4. Prediction equations

### 4.1 Keep class-specific rule banks

For output `k`, let there be `M_k` clauses. Preserve the source TM's class association. Define:

- `C_kj(x)`: Boolean output of clause `j` belonging to output `k`.
- `P_kj`: fixed numerical feature indices used by that clause's linear output.
- `beta_k`: global intercept for output `k`.
- `b_kj`: local intercept for clause `j`.
- `w_kjf`: coefficient of numerical feature `f` in that clause.

The score is:

\[
\boxed{
S_k(x)=\beta_k+\sum_{j=1}^{M_k}C_{kj}(x)
\left(b_{kj}+\sum_{f\in P_{kj}}w_{kjf}x_f\right).
}
\]

Use all `d` numerical features in each local model for the full formulation. A fixed subset is an explicitly recorded smaller variant. A slope can use a feature that does not occur in the clause condition.

For classification:

\[
\widehat y=\operatorname*{arg\,max}_{k=0,\ldots,K-1} S_k(x).
\]

Break exact ties by the lowest class index. These are scores, not probabilities. For MNIST, use ten outputs, not one scalar estimate of the digit number.

For regression:

\[
\widehat y=S_0(x),
\]

followed by inverse target normalization when applicable. Do not apply the original regression TM's vote-count scaling to the new linear output.

### 4.2 Polarity and overlap

Allow all new coefficients to be positive, zero, or negative. Source clause polarity is retained as metadata but is **not multiplied into the hybrid output again**. The signed coefficients determine whether a contribution is positive or negative.

Several clauses can match simultaneously. Sum their contributions; do not average by the active-clause count in version 1. If none matches for output `k`, its score is `beta_k`.

A shared rule bank across classes is possible, but is not the default here. Keeping class-specific banks makes integration with an existing multiclass TM direct and avoids attaching every extracted rule to every class.

## 5. Coefficient memory and model records

### 5.1 State-to-coefficient mapping

Every global intercept, clause intercept, and slope uses:

\[
q\in\{-Q,\ldots,0,\ldots,Q\},\qquad \theta=\delta q.
\]

Require `Q >= 1` and `delta > 0`. Start with a common `delta` for all coefficients.

Example configuration:

```text
coefficient_step delta = 1 / 64
state_bound Q = 256
available coefficient range = [-4, +4]
number of states per coefficient = 513
```

These are initial experiment settings. Choose numerical scaling and range together. A smaller step with unchanged `Q` also reduces the range. To compare precision fairly, hold `Q * delta` approximately constant.

`int16` can represent this example's states. Check bounds using a sufficiently wide temporary integer before storing an increment. Integer state storage does not mean prediction requires only bit operations: numerical multiplications and additions remain.

### 5.2 Logical records

```text
HybridModel:
    format_version
    task                         # CLASSIFICATION or SCALAR_REGRESSION
    input_dimension
    output_labels                # fixed label order; one output for regression
    numeric_preprocessor
    boolean_encoder
    target_transform             # identity unless regression scaling was fitted
    delta, Q, lambda
    frozen_rules[k][j]
    feature_indices[k][j]        # P_kj; fixed before coefficient fitting
    coefficient_layout[k]
    coefficient_states[k]       # integer vector q_k
    source_tm_identity           # implementation/version/checkpoint identity
    experiment_metadata

FrozenRule:
    source_output_id
    source_clause_id
    source_polarity              # metadata only in the hybrid
    source_clause_weight         # metadata only, if present
    included_literals           # canonical (Boolean feature ID, polarity) pairs
    empty_clause_output          # source prediction-mode semantics: 0 or 1
    semantic_fingerprint

CoordinateDescriptor:
    kind                        # GLOBAL_BIAS, CLAUSE_BIAS, or SLOPE
    clause_id                   # absent for GLOBAL_BIAS
    numeric_feature_id          # present only for SLOPE
```

Use this fixed coordinate order for each output: global intercept first, then each clause's local intercept followed by its slopes in ascending numerical feature order.

\[
D_k=1+\sum_{j=1}^{M_k}(1+|P_{kj}|).
\]

There are `D_k` integer states for output `k`. All fitted arrays, fingerprints, and ordering conventions must survive a save/load round trip.

## 6. Adapter to the existing TM

### 6.1 Required contract

The adapter must provide these capabilities, using existing APIs or read-only access to existing model state:

| Adapter operation | Required behavior |
|---|---|
| `TRAIN_OR_LOAD` | Return a trained source checkpoint, with its encoder and provenance |
| `LIST_OUTPUTS_AND_CLAUSES` | Return stable output and clause IDs |
| `GET_INCLUDED_LITERALS` | Return Include actions, translated to canonical literal pairs |
| `GET_CLAUSE_METADATA` | Report polarity, optional weight, and empty-clause inference semantics |
| `EVALUATE_RAW_CLAUSES` | Return Boolean clause outputs before polarity, weights, or vote clipping |
| `PREDICT_ORIGINAL` | Run the source model unchanged for baseline evaluation |
| `SNAPSHOT_IDENTITY` | Identify the source state sufficiently to detect accidental changes |

For example, the [pyTsetlinMachine interpretation example](https://github.com/cair/pyTsetlinMachine#interpretability-demo) exposes an accessor named `ta_action(class, clause, literal)`. Other implementations may require a different adapter. Pin the implementation version and verify its conventions; the pseudocode below deliberately avoids hard-coding that API.

If only final class predictions are exposed, that is insufficient to verify clause equivalence. Add a read-only clause inspection/evaluation adapter before fitting coefficients. Final predictions alone can conceal incorrect individual clauses.

### 6.2 Preserve clause semantics

Version 1 preserves all source clause slots and their class assignments. It does not silently deduplicate, prune, or reinterpret them.

- Empty clauses follow the source model's **prediction-mode** behavior, recorded per rule. Do not assume that training-mode and prediction-mode outputs agree for an empty clause.
- A clause containing contradictory literals evaluates to zero.
- A clause with zero observed training support remains in the frozen snapshot. Its zero-initialized coefficients receive no data-driven update; keeping it avoids silently changing the rule inventory.
- Duplicate clauses remain separate. Removing duplicates can change the regularization and available effective coefficient range even if their truth values agree.
- Disable stochastic clause/feature dropout when evaluating frozen gates. A deterministic inference snapshot is required.
- For custom predicates or convolutional clauses, an ordinary literal conjunction exporter is insufficient unless it reproduces those semantics. Use a matching adapter or defer that source model.

### 6.3 Frozen-clause extraction

```text
PROCEDURE FREEZE_RULES(source_tm, encoder, probe_inputs):
    identity_before <- ExistingTM.SNAPSHOT_IDENTITY(source_tm)
    banks <- empty ordered collection

    FOR each output k in saved output order:
        FOR each clause j in the source's stable clause order:
            literals <- ExistingTM.GET_INCLUDED_LITERALS(source_tm, k, j)
            metadata <- ExistingTM.GET_CLAUSE_METADATA(source_tm, k, j)

            ASSERT every literal references a valid Boolean feature
            ASSERT every literal has POSITIVE or NEGATIVE polarity
            ASSERT metadata.empty_clause_output is 0 or 1

            rule <- immutable copy of literals and metadata
            rule.semantic_fingerprint <- fingerprint(rule, encoder)
            append rule to banks[k]

    B_probe <- encoder.TRANSFORM(probe_inputs)
    expected <- ExistingTM.EVALUATE_RAW_CLAUSES(source_tm, B_probe,
                                               mode = DETERMINISTIC_INFERENCE)
    observed <- EVALUATE_FROZEN_BANKS(banks, B_probe)
    ASSERT observed equals expected element by element
    ASSERT ExistingTM.SNAPSHOT_IDENTITY(source_tm) equals identity_before

    RETURN immutable banks
```

Probe inputs include representative training examples plus valid inputs at, just below, and just above encoder thresholds when those values are representable. Test empty and contradictory clauses using adapter fixtures even if the trained model does not contain them. Never use test labels to choose clauses or fix preprocessing.

```text
FUNCTION EVALUATE_FROZEN_CLAUSE(rule, Boolean_sample):
    IF rule.included_literals is empty:
        RETURN rule.empty_clause_output

    FOR each (feature_id, polarity) in rule.included_literals:
        literal_value <- Boolean_sample[feature_id]
        IF polarity is NEGATIVE:
            literal_value <- NOT literal_value
        IF literal_value is FALSE:
            RETURN 0
    RETURN 1

FUNCTION EVALUATE_FROZEN_BANKS(banks, B):
    FOR each output k:
        allocate G[k] with shape (number_of_samples, number_of_rules_in_bank_k)
        FOR each sample i and rule j in banks[k]:
            G[k][i, j] <- EVALUATE_FROZEN_CLAUSE(banks[k][j], B[i])
    RETURN G
```

## 7. Train or load the clauses, then freeze them

```text
PROCEDURE PREPARE_SOURCE_TM(training, validation, encoder, source_configuration):
    B_train <- encoder.TRANSFORM(training.X_numeric)
    B_validation <- encoder.TRANSFORM(validation.X_numeric)
    source_training_targets <- training.source_targets

    IF a compatible trained checkpoint was supplied:
        source_tm <- ExistingTM.LOAD(checkpoint)
        VERIFY encoder, task, output labels, and input order match checkpoint
        VERIFY and record the checkpoint's training-data provenance
    ELSE:
        source_tm <- ExistingTM.CREATE(source_configuration)
        FOR each permitted TM training epoch:
            ExistingTM.TRAIN_ONE_EPOCH(source_tm, B_train, source_training_targets)
            measure source-model validation metric using B_validation
            retain the best source checkpoint under a declared tie rule
        source_tm <- restore best source checkpoint

    preserve a separate source checkpoint for baseline evaluation
    banks <- FREEZE_RULES(source_tm, encoder, training_probe_inputs)
    RETURN source_tm, banks
```

`source_training_targets` means class labels for a multiclass TM, or the targets expected by an existing regression TM. It is not automatically the one-hot matrix used by the new coefficient optimizer.

Only training data updates the source model. Use a validation criterion appropriate to its task: classification accuracy or regression error. Loading a checkpoint trained on the eventual test data invalidates a held-out comparison; record provenance rather than presenting such a run as independent evaluation.

After freezing, assign `P_kj` and build the coefficient layouts. The default is every numerical feature. Feature selection is an optional training-only operation completed before coefficient training starts. The frozen snapshot, encoder, preprocessing, feature subsets, target matrix, `delta`, `Q`, and regularization must remain fixed throughout one optimizer run.

## 8. Prediction pseudocode

### 8.1 Direct prediction

```text
FUNCTION PREDICT_SCORES(model, raw_inputs):
    X <- model.numeric_preprocessor.TRANSFORM(raw_inputs)
    ASSERT shape and numerical values are valid
    B <- model.boolean_encoder.TRANSFORM(X)
    G <- EVALUATE_FROZEN_BANKS(model.frozen_rules, B)
    S <- zero matrix with shape (number_of_samples, number_of_outputs)

    FOR each output k:
        beta <- model.delta * state_for_global_intercept(k)
        S[:, k] <- beta

        FOR each rule j in model.frozen_rules[k]:
            active_rows <- rows where G[k][:, j] equals 1
            local_output <- model.delta * state_for_local_intercept(k, j)

            FOR each feature f in model.feature_indices[k][j]:
                w <- model.delta * state_for_slope(k, j, f)
                local_output <- local_output + w * X[active_rows, f]

            S[active_rows, k] <- S[active_rows, k] + local_output

    RETURN S

FUNCTION PREDICT(model, raw_inputs):
    S <- PREDICT_SCORES(model, raw_inputs)
    IF model.task is CLASSIFICATION:
        indices <- row-wise argmax(S), ties to smallest index
        RETURN saved output labels at indices
    ELSE:
        RETURN model.target_transform.INVERSE(S[:, 0])
```

The scalar local intercept broadcasts across active rows. An inactive rule contributes exactly zero, even if its stored coefficients are nonzero. No source polarity, source weight, vote threshold, or vote-count normalization is applied after these calculations.

The `state_for_*` expressions look up the appropriate integer in `model.coefficient_states[k]` through its saved coordinate layout. They are lookup notation, not extra learned components.

### 8.2 Fixed design columns for training

For each output define:

\[
\phi_k(x)=\left[1, C_{k1}(x), C_{k1}(x)x_{P_{k1}},\ldots,
C_{kM_k}(x), C_{kM_k}(x)x_{P_{kM_k}}\right].
\]

Let `A_k` have rows `phi_k(x_i)`. Then:

\[
S_{:k}=A_k\theta_k,\qquad\theta_k=\delta q_k.
\]

This is linear in the coefficients while the gates are fixed, even though it can be nonlinear in the input.

Do not require a dense `A_k`. A column provider is sufficient:

```text
FUNCTION DESIGN_COLUMN(X, G, layout, k, r):
    descriptor <- layout[k][r]
    IF descriptor.kind is GLOBAL_BIAS:
        RETURN vector of ones, length number_of_samples
    IF descriptor.kind is CLAUSE_BIAS:
        RETURN numerical view of G[k][:, descriptor.clause_id]
    IF descriptor.kind is SLOPE:
        j <- descriptor.clause_id
        f <- descriptor.numeric_feature_id
        RETURN G[k][:, j] * X[:, f]             # elementwise product

FUNCTION SCORES_FROM_STATES(X, G, layout, q, delta):
    S <- zero matrix with shape (number_of_samples, number_of_outputs)
    FOR each output k:
        FOR each coordinate r in layout[k]:
            IF q[k][r] is not zero:
                S[:, k] <- S[:, k] + delta * q[k][r] * DESIGN_COLUMN(X, G, layout, k, r)
    RETURN S
```

Direct prediction and column-based prediction must agree. They are two evaluation paths for the same model, not separate architectures.

## 9. Training objective and exact state updates

### 9.1 Objective

For `N` training examples, targets `Y`, and `K` outputs:

\[
\boxed{
J(q)=\frac{1}{2N}\sum_{k=0}^{K-1}\|A_k\delta q_k-Y_{:k}\|_2^2
+\frac{\lambda}{2}\sum_{k=0}^{K-1}\|\delta q_k\|_2^2.
}
\]

For classification, `Y[i,k] = 1` for the true class and zero otherwise. For scalar regression, `Y[:,0]` contains the possibly normalized real-valued target.

Divide by `N`, **not `N*K`**, in both the objective and update formulas. Regularize all slopes and intercepts in version 1. Start with `lambda = 0.001` for experiments; `lambda = 0` is useful for exact arithmetic examples. Tune on validation data.

Squared error on one-hot targets is the chosen classification surrogate. Lower training loss does not guarantee higher classification accuracy at every step, nor calibrated probabilities.

### 9.2 Exact one-state loss change

Maintain residuals:

\[
R_{:k}=A_k\theta_k-Y_{:k}.
\]

For coordinate `r` of output `k`, let `a = A_k[:,r]` and:

\[
g=\frac{a^T R_{:k}}{N}+\lambda\theta_{kr},\qquad
h=\frac{a^Ta}{N}+\lambda.
\]

A state move `m` of `-1` or `+1` changes the coefficient by `s = delta*m`. Its exact objective change is:

\[
\boxed{\Delta J=s\,g+\frac{s^2}{2}h.}
\]

Evaluate both feasible moves, select the smaller loss change, and accept only if `DeltaJ < -accept_tol`. Start with `accept_tol = 1e-12` using float64 arithmetic; the tolerance is numerical, not a statistical confidence bound.

After acceptance:

```text
q[k][r] <- q[k][r] + m
R[:, k] <- R[:, k] + delta * m * a
```

This comparison evaluates the full supplied training partition. A mini-batch estimate is a different algorithm and does not inherit the exact full-loss descent guarantee.

The gradient expression is a shortcut for computing two trial losses. The method selects an integer move directly; it does not take a continuous gradient step and round afterward.

## 10. Coefficient-training pseudocode

### 10.1 Helpers

```text
FUNCTION OBJECTIVE(R, q, delta, lambda):
    N <- number of rows in R
    data_term <- sum of squared entries of R / (2 * N)
    penalty <- lambda / 2 * sum over all states of (delta * state)^2
    RETURN data_term + penalty

FUNCTION BEST_ONE_STATE_MOVE(current_state, g, h, delta, Q):
    best_move <- 0
    best_change <- 0
    FOR m in [-1, +1], in this order:
        proposed_state <- current_state + m, computed in a wide integer type
        IF proposed_state is outside [-Q, Q]:
            CONTINUE
        s <- delta * m
        change <- s * g + (s * s / 2) * h
        IF change < best_change:
            best_move <- m
            best_change <- change
    RETURN best_move, best_change

FUNCTION VALIDATION_KEY(task, scores, validation_targets):
    IF task is CLASSIFICATION:
        accuracy <- fraction of correct argmax predictions, using saved tie rule
        squared_error <- mean squared score error against one-hot targets
        RETURN (-accuracy, squared_error)       # smaller tuple is better
    ELSE:
        mse <- mean squared error against regression targets in training units
        RETURN (mse)
```

Compare validation keys lexicographically. A strict improvement replaces the best checkpoint. An identical key retains the earlier checkpoint. With a fixed affine target normalization, regression MSE ranking in training units and original units is the same.

### 10.2 Fit the numerical outputs

```text
PROCEDURE FIT_COEFFICIENTS(model_definition, training, validation, config):
    task, rules, encoder, preprocessing, feature subsets <- model_definition fields
    delta, Q, lambda, max_sweeps, accept_tol, seed <- config fields
    ASSERT delta, Q, lambda agree with the model definition
    ASSERT training has N > 0 samples and validation is nonempty
    ASSERT delta > 0, Q >= 1, lambda >= 0, max_sweeps >= 1, accept_tol >= 0
    ASSERT all targets, features, and configurations are valid

    freeze_id <- fingerprint(rules, encoder, preprocessing, feature subsets,
                             training sample IDs, targets, delta, Q, lambda)
    X_train, Y_train <- training.X_numeric, training.Y
    X_val, Y_val <- validation.X_numeric, validation.Y
    G_train <- EVALUATE_FROZEN_BANKS(rules, encoder.TRANSFORM(X_train))
    G_val <- EVALUATE_FROZEN_BANKS(rules, encoder.TRANSFORM(X_val))
    layout <- BUILD_LAYOUT(rules, feature subsets)

    q[k] <- integer zeros of length D_k, for every output k
    R <- -Y_train                               # initial scores are all zero
    J <- OBJECTIVE(R, q, delta, lambda)
    rng <- random generator initialized with seed

    FOR each output k and coordinate r:
        a <- DESIGN_COLUMN(X_train, G_train, layout, k, r)
        h[k][r] <- DOT(a, a) / N + lambda

    initial_val_scores <- zero matrix with shape (validation_size, output_count)
    best_key <- VALIDATION_KEY(task, initial_val_scores, Y_val)
    best_checkpoint <- deep copy of q at sweep 0
    trace <- initial objective, validation metrics, and state diagnostics
    stop_reason <- SWEEP_BUDGET

    FOR sweep from 1 through max_sweeps:
        accepted <- 0
        start_J <- J
        predicted_total_change <- 0

        FOR each output k in saved order:
            order <- seeded random permutation of all coordinates in layout[k]
            FOR r in order:
                a <- DESIGN_COLUMN(X_train, G_train, layout, k, r)
                theta <- delta * q[k][r]
                g <- DOT(a, R[:, k]) / N + lambda * theta
                m, change <- BEST_ONE_STATE_MOVE(q[k][r], g, h[k][r], delta, Q)

                IF m is not zero AND change < -accept_tol:
                    q[k][r] <- checked integer addition of m
                    R[:, k] <- R[:, k] + delta * m * a
                    predicted_total_change <- predicted_total_change + change
                    accepted <- accepted + 1

        # Rebuild residuals to remove accumulated floating-point drift.
        train_scores <- SCORES_FROM_STATES(X_train, G_train, layout, q, delta)
        R <- train_scores - Y_train
        J <- OBJECTIVE(R, q, delta, lambda)
        ASSERT J agrees with start_J + predicted_total_change
               within a declared absolute/relative numerical tolerance
        ASSERT J <= start_J within that numerical tolerance
        ASSERT current fixed-component fingerprint equals freeze_id
        # On failed numerical/integrity checks: stop with an error, not convergence.

        val_scores <- SCORES_FROM_STATES(X_val, G_val, layout, q, delta)
        key <- VALIDATION_KEY(task, val_scores, Y_val)
        IF key is strictly better than best_key:
            best_key <- key
            best_checkpoint <- deep copy of q, sweep, and selection metrics

        append objective, accepted moves, validation metrics,
               state-bound occupancy, and elapsed time to trace

        IF accepted equals 0:
            stop_reason <- NO_ACCEPTED_MOVES
            BREAK

    terminal_checkpoint <- deep copy of q and current sweep
    terminal_audit <- AUDIT_NEIGHBORS(terminal_checkpoint, training, model_definition)
    selected_audit <- AUDIT_NEIGHBORS(best_checkpoint, training, model_definition)

    RETURN best_checkpoint, terminal_checkpoint, trace, stop_reason,
           terminal_audit, selected_audit
```

`BUILD_LAYOUT` follows the coordinate order in Section 5. It returns one `GLOBAL_BIAS` descriptor, then one `CLAUSE_BIAS` and the configured `SLOPE` descriptors for each clause. No data-dependent layout changes occur in this procedure.

Within one output, moves are sequential and use the latest residual. Do not calculate independent updates for different columns from stale residuals and apply them simultaneously: cross terms invalidate their separate loss-change estimates. Different outputs have separable objectives, but a sequential reference implementation is simplest.

A zero column with a zero coefficient produces no update. If it somehow has a nonzero coefficient and `lambda > 0`, the regularization term can still move it toward zero; do not blindly skip zero columns in a generalized warm-start implementation.

### 10.3 Audit a checkpoint without modifying it

```text
FUNCTION AUDIT_NEIGHBORS(checkpoint, training, model_definition):
    q <- checkpoint's states
    R <- SCORES_FROM_STATES(training features and gates, layout, q, delta) - Y_train
    smallest_change <- 0
    improving_coordinates <- 0

    FOR each output k and coordinate r:
        a <- DESIGN_COLUMN(training features and gates, layout, k, r)
        g <- DOT(a, R[:, k]) / N + lambda * delta * q[k][r]
        h <- DOT(a, a) / N + lambda
        m, change <- BEST_ONE_STATE_MOVE(q[k][r], g, h, delta, Q)
        smallest_change <- MIN(smallest_change, change)
        IF change < -accept_tol:
            improving_coordinates <- improving_coordinates + 1

    RETURN:
        stationary_within_tolerance = (improving_coordinates equals 0)
        improving_coordinate_count = improving_coordinates
        best_available_objective_change = smallest_change
```

A budget stop is not convergence. A validation-selected checkpoint may precede the terminal state and need not be coordinate-stationary. Report its audit separately. Never label every returned model “converged” merely because training finished.

### 10.4 Numerical policy

Use float64 for the reference optimizer's dot products, residuals, and objective. `delta = 1/64` is exactly representable in binary floating point, but products with inputs can still round.

Declare drift-check tolerances separately from `accept_tol`; one checks numerical consistency, the other determines which state moves are accepted. If checks fail, investigate conditioning, summation, or residual maintenance. Do not silently increase tolerances until a failing run appears successful. Mathematical guarantees below refer to exact arithmetic; practical implementation requires these consistency checks.

## 11. End-to-end pseudocode

```text
PROCEDURE BUILD_AND_EVALUATE_LINEAR_OUTPUT_TM(raw_dataset, config):
    training_raw, validation_raw, test_raw <- declared disjoint split

    preprocessing <- fit numerical preprocessing on training_raw only
    training <- transform training_raw using preprocessing
    validation <- transform validation_raw using preprocessing

    encoder <- load source checkpoint's compatible encoder
               OR construct/fit the declared encoder using training only
    target_transform <- identity for classification
                        OR fit declared regression scaling using training only
    training.source_targets, validation.source_targets <- labels or targets
        prepared for the source TM under its saved target convention
    training.Y, validation.Y <- one-hot labels in saved output order
        OR column vectors of regression targets transformed by target_transform

    source_tm, banks <- PREPARE_SOURCE_TM(training, validation, encoder,
                                          config.source_tm)
    definition <- model schema with task, output labels, encoder, preprocessing,
                  target transform, frozen banks, delta, Q, and lambda
    definition.feature_indices <- fixed full-feature or declared subset policy
    definition.layout <- BUILD_LAYOUT(banks, definition.feature_indices)

    fit_result <- FIT_COEFFICIENTS(definition, training, validation, config)
    selected_model <- definition plus fit_result.best_checkpoint states
    SAVE_HYBRID(selected_model, fit_result and experiment metadata)

    reloaded_model <- LOAD_HYBRID(saved artifact)
    VERIFY reload predictions agree on a validation probe

    # Execute only after configuration and checkpoint selection are complete.
    test_source_predictions <- ExistingTM.PREDICT_ORIGINAL(source_tm,
        encoder.TRANSFORM(preprocessing.TRANSFORM(test_raw.inputs)))
    test_hybrid_predictions <- PREDICT(reloaded_model, test_raw.inputs)
    evaluate both predictions using untouched test targets
    report metrics, model sizes, timings, seeds, and limitations

    RETURN saved model artifact and experiment report
```

For a supplied fixed-rule synthetic task, replace the source-training/extraction stage with the explicit fixture. Do not invent a new rule-discovery algorithm when the intended integration already has a TM implementation.

For hyperparameter search, repeat fitting and selection using training/validation only. Freeze the complete comparison protocol before evaluating the test split. The pseudocode shows one chosen configuration, not a loop that repeatedly selects on test results.

## 12. Worked examples

### 12.1 One MNIST-style contribution

Use three illustrative normalized pixels:

\[
x_1=0.9,\qquad x_2=0.8,\qquad x_3=0.1.
\]

The frozen condition is:

\[
C(x)=\mathbf1[x_1>0.5\ \land\ x_2>0.5\ \land\ x_3\leq0.5].
\]

It activates. Let one rule for digit 7 have:

\[
f_7(x)=0.6x_1+0.3x_2-0.2x_3-0.1.
\]

Its contribution is:

\[
C(x)f_7(x)=0.54+0.24-0.02-0.1=0.66.
\]

With `delta = 0.1`, its slope states are `(6, 3, -2)` and intercept state is `-1`. Moving the second slope state from 3 to 4 changes that coefficient from 0.3 to 0.4 and the contribution from 0.66 to 0.74.

For an isolated one-sample example with target 1, no other contributions, and `lambda = 0`:

```text
residual before = 0.66 - 1 = -0.34
active design value a = 0.8
N = 1

g = 0.8 * (-0.34) = -0.272
h = 0.8^2 = 0.64
s = 0.1

DeltaJ = 0.1 * (-0.272) + (0.1^2 / 2) * 0.64 = -0.024
loss before = 0.5 * (-0.34)^2 = 0.0578
loss after  = 0.5 * (-0.26)^2 = 0.0338
```

The move is accepted in this isolated example. Actual training evaluates its effect on every supplied training sample and includes regularization. These coefficients and numbers illustrate mechanics; they are not trained MNIST results.

For pixels `(0.6, 0.6, 0.4)`, the same condition activates but the original illustrative linear output is `0.36`. A fixed Boolean vote would give the same contribution on both images; the linear output uses their different grayscale magnitudes.

### 12.2 Small regression fixture

Use an identity numerical preprocessor and `x` in `[-1, 1]`. Define the Boolean feature `B = [x > 0]`, with clauses:

\[
C_1(x)=\neg B=\mathbf1[x\leq0],\qquad C_2(x)=B=\mathbf1[x>0].
\]

Choose:

\[
\widehat y(x)=C_1(x)(1+2x)+C_2(x)(1-0.5x),\qquad\beta=0.
\]

With `delta = 0.5`, the exact local `(intercept, slope)` states are `(2, 4)` and `(2, -1)`.

| Input | Active clause | Output |
|---|---|---|
| -0.5 | First | 0 |
| 0 | First | 1 |
| 0.5 | Second | 0.75 |

This fixture verifies representability and prediction. Do not require a generic coordinate-search run to recover these exact states: redundant intercepts can give multiple equivalent parameterizations, and finite-state coordinate descent can stop in a coordinate-local solution. Test optimization separately with known one-coordinate or orthogonal fixtures.

## 13. Mathematical guarantees

### 13.1 Exact loss-change identity

Holding every other coefficient fixed, expand the affected residual:

\[
\frac{\|R_{:k}+s a\|^2-\|R_{:k}\|^2}{2N}
= s\frac{a^TR_{:k}}{N}+\frac{s^2}{2N}a^Ta.
\]

The penalty change is:

\[
\frac{\lambda}{2}\big((\theta_{kr}+s)^2-\theta_{kr}^2\big)
=\lambda s\theta_{kr}+\frac{\lambda s^2}{2}.
\]

Adding them gives exactly `DeltaJ = s*g + s^2*h/2`. Thus every feasible accepted move strictly lowers the stated objective in exact arithmetic.

### 13.2 Finite termination and its limits

Let `D_total = sum_k D_k`. There are `(2Q+1)^D_total` possible coefficient-state arrays. Strictly decreasing the same objective prevents a state array from being revisited. Therefore the algorithm has finitely many accepted moves.

If complete sweeps continue without a budget interruption, it eventually reaches a state with no improving feasible one-coordinate step, within the acceptance tolerance. This is not a useful runtime bound and is not a proof of a globally optimal integer-state solution. A convex continuous objective can still have coordinate traps on a discrete grid.

### 13.3 Interior quantization bound

For output `k`, let:

\[
H_k=A_k^TA_k/N+\lambda I,
\qquad \mu_k=\lambda_{\min}(H_k)>0.
\]

At a coordinate-stationary state where **both** `+delta` and `-delta` moves are feasible for every coefficient:

\[
|g_{kr}|\leq\delta(H_k)_{rr}/2
\]

when the acceptance tolerance is zero. Strong convexity implies, relative to the unconstrained continuous ridge optimum:

\[
J(q)-J(\Theta^*)
\leq\sum_k\frac{\delta^2}{8\mu_k}\sum_r(H_k)_{rr}^2.
\]

With acceptance tolerance `epsilon`, replace the gradient bound by:

\[
|g_{kr}|\leq\delta(H_k)_{rr}/2+\epsilon/\delta,
\]

and use `sum_k sum_r (delta*(H_k)_rr/2 + epsilon/delta)^2 / (2*mu_k)` for the corresponding objective-gap bound.

`lambda > 0` guarantees `mu_k >= lambda`. The bound can be loose. It does not apply as stated to coordinates at their state limits, unfinished optimization, or changing gates. No classification-accuracy or generalization guarantee follows from this objective-gap bound.

### 13.4 Why Boolean-gated linear models can be nonlinear

Different clauses activate in different input regions. Within each region where all clause truth values are fixed, the summed model is affine. Across regions, its slopes and intercepts can change. It can also have jumps at rule boundaries; continuity is not guaranteed.

If every gate were always active, the sum would collapse to one global affine model. Input-dependent Boolean conditions supply the regional structure.

For a scalar twice-differentiable target on `[0,1]` with `|f''| <= M`, choose interval gates of width at most `h` and midpoint Taylor models. Their approximation error is at most `M*h^2/8`. Rounding each local intercept and slope to the nearest grid point adds at most `delta`, provided the state range covers those rounded values. Hence an appropriate model exists with error at most:

\[
\sup_x|f(x)-\widehat f_\delta(x)|\leq M h^2/8+\delta.
\]

This is an existence argument for suitable conditions. It does not prove that a pretrained TM will learn those conditions or that the coefficient optimizer will find those parameters.

### 13.5 What this document does not prove

The derivation covers coefficient fitting with fixed gates and features. It does not establish convergence of joint clause/coefficient learning, superiority to the original TM, improved MNIST accuracy, reduced energy use, or novelty of local linear rule models. Those require separate analysis and experiments.

## 14. MNIST experiment and fair comparisons

### 14.1 Dataset protocol

Use the official 60,000-image training pool and 10,000-image test set. A practical declared split is 50,000 training and 10,000 validation images, stratified by digit, with saved sample indices and a fixed split seed. Reserve the official test set for final evaluation.

All variants must use the same split, numerical input scaling, and applicable Boolean encoder. Any data-dependent thresholds are fitted on the training partition only.

### 14.2 Small integration run

The following configuration is for debugging and estimating cost, not for demonstrating competitive accuracy:

| Setting | Initial debugging choice |
|---|---|
| Training examples | 2,000, stratified from the training partition |
| Validation examples | 1,000, stratified from the validation partition |
| Numerical input | All 784 normalized grayscale pixels |
| New Boolean encoder | One threshold at 0.5 per pixel |
| Source clauses | 2 per class if supported; otherwise the smallest valid count |
| Source TM training settings | Existing valid configuration, with its seed and epoch budget recorded |
| Numerical slopes per clause | All 784 features |
| Coefficient step | `delta = 1/64` |
| State limit | `Q = 256` |
| Ridge strength | `lambda = 0.001` |
| Optimizer sweep budget | 10 for integration testing; increase only after measuring cost |
| Acceptance tolerance | `1e-12`, float64 reference arithmetic |
| Design matrix | Lazy columns, not a full dense allocation |

The 2,000 examples are the complete training partition for this debugging run. The full-loss formula refers to these examples. Ten sweeps may be far from stationarity; report the audit rather than assuming convergence.

For a supplied trained TM, preserve its existing encoder and clause inventory. The table's source-training choices apply when deliberately training a new small debugging model.

### 14.3 Comparisons that answer different questions

| Model | Conditions | Output | Question answered |
|---|---|---|---|
| Original source TM | Original learned clauses | Original voting and prediction | What does the existing system achieve? |
| Frozen clauses with learned constants | Same frozen clauses | `beta_k + sum_j C_kj * b_kj` | How much comes from fitting vote magnitudes alone? |
| Global finite-state linear model | No conditional clauses | `beta_k + sum_f w_kf * x_f` | Are conditional local models useful? |
| Proposed model | Same frozen clauses | `beta_k + sum_j C_kj * (b_kj + w_kj^T x)` | What do local numerical slopes add? |
| Continuous ridge model, optional | Same frozen clauses and feature subsets | Same equation, continuous coefficients | What is lost through finite-state optimization and quantization? |

For the constant-output ablation, use `P_kj = empty` and the same integer-state optimizer. For the global linear baseline, use a global affine feature map `[1, x]`. Keep preprocessing, splits, loss normalization, and evaluation protocol aligned.

For continuous ridge with fixed gates, solve:

\[
(A_k^TA_k/N+\lambda I)\theta_k=A_k^TY_{:k}/N
\]

for each output. Use an appropriate solver rather than explicitly forming an inverse. A dense solve is suitable only for small dimensions; use a matrix-free method otherwise. The unconstrained continuous solution is a reference optimum for this objective, not a claim that the quantized learner must reach it. Rounding/clipping that solution would be a separate initialization experiment.

Do not pool every class's clauses into every other class for the proposed model while calling it the same class-specific comparison. Any shared-bank variant changes the parameterization and should be reported separately.

### 14.4 What to record

Report classification accuracy, per-class accuracy or confusion matrix, training surrogate loss, validation selection criterion, clause counts, literal counts, coefficient count, actual serialized size, peak working memory, and training/inference time on named hardware.

Report source TM learning time and new coefficient-fitting time separately, plus their combined total. If the source checkpoint was reused, distinguish added fitting cost from total model-building cost.

Record seeds, split indices, encoder details, all TM hyperparameters, coefficient hyperparameters, accepted-move counts, selected and terminal sweeps, stationarity audits, and fraction of coefficients at `-Q` or `+Q`.

After the prototype is correct, evaluate a predeclared set of seeds and configurations. Report variability, not just the best run. Better training loss alone is not evidence of better MNIST recognition.

## 15. Cost, persistence, and diagnostics

### 15.1 Computational cost

Let `D_total = sum_k D_k` and `M_total = sum_k M_k`.

- Coefficient storage is proportional to `D_total` integer states.
- A complete reference coefficient sweep costs approximately `O(N * D_total)` arithmetic operations, excluding constants and memory movement.
- Cached gates require `N * M_total` Boolean entries; packed storage can reduce bytes.
- Numerical inputs require `N * d` entries; training residuals require `N * K` entries.
- A full collection of dense design matrices requires `N * D_total` numerical entries. Avoid this for image-scale configurations.
- Prediction requires Boolean encoding and clause checks, followed by numerical dot products for active clauses.

Example: with 20 clauses per class, ten classes, and 784 slopes plus an intercept per clause:

\[
D_{\text{total}}=10\left[1+20(784+1)\right]=157{,}010.
\]

For 50,000 training examples, dense float64 design matrices alone require about **62.8 GB** in decimal units. The integer coefficient states occupy about **314 KB** with int16 storage, excluding gates, metadata, and preprocessing. Small coefficient storage does not imply small training memory or fast training.

Use a lazy column provider, cached Boolean activations, and incremental residuals. If even cached gates do not fit, evaluate columns over deterministic blocks of the same full training set, accumulating the full dot products before accepting a move. A fresh random mini-batch is not equivalent.

A full 784-feature output per clause can be expensive. Fixed local feature subsets and convolution are subsequent variants; their accuracy and cost must be measured rather than assumed.

### 15.2 Save and load

The authoritative learned coefficients are integer states. Save their exact values and the step size; do not save only rounded decimal coefficients.

```text
PROCEDURE SAVE_HYBRID(model, metadata):
    validate dimensions, finite scalars, integer types, and state bounds
    write a versioned record containing:
        task, output-label order, input shape, numeric preprocessing
        complete Boolean encoder and literal mapping
        frozen rule banks, empty-clause semantics, and source clause metadata
        feature subsets and coordinate layouts
        coefficient states, delta, Q, lambda, target transform
        source implementation/checkpoint identity and frozen-component fingerprints
        configuration, split identities, seeds, trace, and checkpoint selection data
    store a small non-sensitive validation fixture or its reproducible identity
    write artifact integrity checksums

FUNCTION LOAD_HYBRID(artifact):
    validate format version and artifact integrity
    restore preprocessing, encoder, frozen conditions, layouts, and states
    validate state bounds, label order, dimensions, and feature references
    verify reconstructed semantic fingerprints
    RETURN model ready for deterministic inference
```

A literal-export model with complete preprocessing can perform inference without a live source TM. If an adapter instead depends on the source library's evaluator, bundle the required source checkpoint and pin that dependency. Record which deployment mode is used.

For exact training resumption, also save terminal states, sweep number, random-generator state, optimizer configuration, training-set identity, and the validation-best checkpoint. Recompute residuals on resume. A validation-selected inference checkpoint alone does not contain the full training trajectory.

### 15.3 Diagnostics and likely causes

| Symptom | First checks |
|---|---|
| Extracted clause outputs disagree with the TM | Literal order, complement mapping, threshold equality, prediction mode, empty clauses |
| Scores unexpectedly change sign | Accidental extra source-polarity multiplication; coefficient sign decoding |
| Predicted update lowers loss but recomputed loss rises | Stale residuals, wrong normalization, missing ridge term, simultaneous column moves |
| Many coefficients hit their limits | Insufficient range, target/input scaling, correlated design columns |
| No coefficient moves from zero | Coarse step size, inactive gates, zero design columns, target mapping, tolerance |
| Training improves while validation worsens | Excess capacity, regularization, clause quality; inspect validation-selected checkpoint |
| All class scores are equal | Empty/inactive banks, zero states, class mapping, or failed fitting |
| Reloaded model gives different outputs | Missing encoder/preprocessor, layout order, state dtype, or different inference semantics |

Do not change `delta`, `Q`, features, or gates midway through a run without treating it as a new optimization problem and recomputing all dependent state.

## 16. Verification plan

These are specifications for tests to write during implementation. This document does not supply or claim executed model tests.

| Check | Fixture and expected result |
|---|---|
| Encoder boundaries | Values below, equal to, and above each threshold follow the saved comparison convention |
| Complement mapping | A Boolean feature and its negation always disagree |
| Adapter fidelity | Every frozen clause matches source raw inference outputs on probes |
| Empty/contradictory clauses | Empty behavior matches the source; contradictions never activate |
| State decoding | Positive, negative, and zero states map exactly to `delta*q` within floating arithmetic |
| State bounds | An outward move at a limit is unavailable; no overflow or wraparound |
| Active contribution | Section 12's example produces 0.66 before the move and 0.74 after it |
| Inactive contribution | A false gate produces zero contribution for arbitrary local coefficients |
| Overlapping clauses | Contributions add; they are not implicitly averaged |
| No active clauses | Output equals the saved global intercept |
| No clauses in a bank | Global-intercept-only prediction remains valid |
| Direct versus column prediction | Both evaluation paths agree for random valid states and inputs |
| Exact loss-change formula | For small fixed matrices, recomputed trial losses match the formula for every feasible neighbor, including intercepts and nonzero ridge |
| Residual maintenance | Incrementally updated residuals match full recomputation |
| Full-loss descent | Every accepted reference move strictly reduces the objective in the numerical verification fixture |
| Optimizer accuracy on a simple fixture | With `a = [1,2]`, target `[2,4]`, one coefficient, `delta = 0.5`, `Q >= 4`, and `lambda = 0`, starting at zero reaches coefficient 2 |
| Coordinate-local limitation | A deliberately correlated two-coordinate case can stop without reaching the best joint move; report stationarity correctly |
| Multiclass decoding | Ten scores use argmax and the saved class-label order; ties are deterministic |
| Regression transform | Forward target scaling and inverse prediction scaling are consistent |
| Save/load fidelity | States, gates, and predictions survive a round trip |
| Split isolation | Fitted preprocessing and updates consume training data only; selection consumes validation, never test labels |
| Frozen-state integrity | Gate, encoder, and feature fingerprints remain unchanged while coefficients train |
| Stop reporting | Budget termination and coordinate stationarity are distinct; selected and terminal checkpoints have separate audits |

A concrete coordinate-trap fixture is `A = [[1,1], [0,0.1]]`, `Y = [0,0.1]`, `N = 2`, `delta = 1`, `Q = 2`, and `lambda = 0`. From states `(0,0)`, every feasible single-state change increases the loss, but states `(-1,1)` fit exactly. This is an intended limitation, not an optimizer bug.

Regression representability, optimization mechanics, and learned-clause quality are separate questions. Keep their tests separate so a failed clause-learning experiment does not masquerade as a failed algebraic implementation.

## 17. Optional extensions

### 17.1 Convolution with fixed Boolean patch conditions

The original [Convolutional Tsetlin Machine](https://arxiv.org/abs/1905.09688) learns clauses over image patches. A later linear-output variant can reuse fixed patch conditions and share coefficients across patch positions.

For a fixed set of `P` patch positions, define `z_p` as the numerical patch vector and `C_kj(z_p)` as its Boolean activation. One explicit extension is:

\[
S_k(x)=\beta_k+\sum_j\frac{1}{P}\sum_{p=1}^{P}
C_{kj}(z_p)\left(b_{kj}+w_{kj}^Tz_p\right).
\]

The divisor is the fixed number of patch positions, including inactive positions. This pooling convention differs from an original clause's OR over patches and must be documented as a model change.

```text
FOR each image and frozen patch rule:
    pooled_intercept_feature <- mean over all fixed positions of C(patch)
    pooled_slope_features <- mean over all fixed positions of C(patch) * patch_values

Use these fixed pooled features with the same integer-state optimizer.
```

These features remain fixed while fitting coefficients, so the quadratic update derivation still applies. Position predicates or pooling regions must be preserved if they are part of the chosen source model; pooling all positions can discard useful spatial information. Do not substitute max pooling over coefficient-dependent linear outputs and assume the same fixed-design proof holds.

### 17.2 Other later experiments

- Fixed local feature subsets to reduce per-rule coefficient count.
- Different step sizes per coordinate, with the update formula using the appropriate local step.
- A shared clause bank across outputs, explicitly measured as a different parameterization.
- A continuous-ridge initialization followed by quantization and integer-state refinement, reported separately from learning from zero.
- Mini-batch or stochastic coefficient updates, with new acceptance analysis.
- Alternating clause and coefficient updates, with a separately specified gate-learning objective and cache invalidation policy.

These extensions are not prerequisites. In particular, ordinary TM feedback must not be assumed to minimize the hybrid squared-error objective when clauses change.

## 18. Implementation checklist

Complete in this order:

1. Identify and pin the existing TM implementation and its Boolean encoder.
2. Implement the adapter and verify raw clause equivalence, including edge cases.
3. Create immutable class-specific rule snapshots and coefficient layouts.
4. Implement state decoding, direct prediction, and lazy design columns.
5. Verify the worked examples and the regression representability fixture.
6. Implement the full-loss one-state optimizer and independent neighbor audit.
7. Verify the exact loss-change formula, residual updates, bounds, and stop reporting.
8. Implement checkpoint selection, serialization, and prediction round trips.
9. Run the small MNIST integration configuration and measure actual cost.
10. Run matched baselines and the full declared evaluation only after correctness is established.

The first version is complete when all of the following hold:

- The Boolean gates come from the existing TM or a declared fixed test fixture.
- The gates stay unchanged throughout coefficient optimization.
- Linear outputs use retained numerical values.
- Coefficients are stored as bounded integer states.
- Accepted moves follow the stated full-training objective and exact update formula.
- Classification and regression prediction paths follow their separate target conventions.
- Selected and terminal checkpoint statuses are recorded accurately.
- Saving and loading preserve predictions.
- Reported experimental outcomes are measured, with no assumed MNIST advantage.

## 19. References

These sources describe the existing TM components and dataset. The finite-state linear-output training design in this document is a proposed extension, not an algorithm attributed to these papers.

1. Granmo, **The Tsetlin Machine — A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic**. [Original TM paper](https://arxiv.org/abs/1804.01508).
2. **A Scheme for Continuous Input to the Tsetlin Machine with Applications to Forecasting Disease Outbreaks**. [Continuous-input encoding](https://arxiv.org/abs/1905.04199).
3. **The Regression Tsetlin Machine: A Tsetlin Machine for Continuous Output Problems**. [Original regression voting and normalization](https://arxiv.org/html/1905.04206#S2.SS3). The proposed affine outputs replace that output mechanism; the original normalization is not added afterward.
4. **The Convolutional Tsetlin Machine**. [Patch-based Boolean clauses](https://arxiv.org/abs/1905.09688).
5. CAIR, **pyTsetlinMachine**. [Maintainer repository and clause-inspection example](https://github.com/cair/pyTsetlinMachine#interpretability-demo). This is one adapter example; the design is library-independent.
6. TensorFlow, **MNIST data loader documentation**. [Dataset dimensions, labels, intensities, and splits](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data).
