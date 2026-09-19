# Multilayer Convolutional Tsetlin Machine (MCTM)

## Complete Pseudocode

### 1. Goal

The key idea is to construct a deep Convolutional Tsetlin Machine in
which each layer produces **spatial binary clause-activation maps**.
Instead of immediately OR-pooling each clause over all spatial
positions, these maps are preserved and used as the input channels to
the next CTM layer.

\[ X\^{(0)} `\rightarrow `{=tex}`\mathrm{CTM}`{=tex}\_1
`\rightarrow `{=tex}X\^{(1)} `\rightarrow `{=tex}`\mathrm{CTM}`{=tex}\_2
`\rightarrow `{=tex}X\^{(2)} `\rightarrow `{=tex}`\cdots`{=tex}
`\rightarrow `{=tex}`\mathrm{Classifier}`{=tex} \]

For layer (l),

\[ X\^{(l)} `\in `{=tex}{0,1}\^{H_l `\times `{=tex}W_l
`\times `{=tex}C_l} \]

where (C_l) is the number of clause feature maps produced by the layer.

------------------------------------------------------------------------

## 2. High-Level Architecture

``` text
INPUT:
    Training dataset D = {(X_i, y_i)}
    Number of CTM layers L
    Layer configurations:
        patch_size[l]
        stride[l]
        number_of_clauses[l]
        T[l]
        s[l]
        epochs[l]

OUTPUT:
    Trained multilayer CTM model

PROCEDURE TRAIN_MULTILAYER_CTM(D):

    current_dataset = D

    FOR layer l = 1 TO L:

        model[l] = INITIALIZE_CTM_LAYER(configuration[l])

        TRAIN_CTM_LAYER(
            model[l],
            current_dataset,
            epochs[l]
        )

        IF l < L:
            next_dataset = empty dataset

            FOR each (X, y) in current_dataset:

                feature_maps = EXTRACT_SPATIAL_CLAUSE_MAPS(
                    model[l],
                    X
                )

                ADD (feature_maps, y) TO next_dataset

            END FOR

            current_dataset = next_dataset

            FREEZE model[l]

    END FOR

    RETURN model
```

This is a **greedy layer-wise training strategy**. Each layer is trained
and then frozen before the next layer is trained.

------------------------------------------------------------------------

# 3. Data Representation

For the first layer, the input may be a binary image:

``` text
X.shape = [Height, Width, Channels]
```

For a grayscale binary image:

``` text
Channels = 1
```

For later layers:

``` text
X.shape = [Height, Width, NumberOfPreviousClauses]
```

Each channel is a binary clause-activation map.

Example:

``` text
Layer 1 output:

Clause 1 map:
0 0 1 0
0 1 1 0
0 0 0 0
0 0 1 0

Clause 2 map:
1 0 0 0
1 1 0 0
0 0 0 1
0 0 0 1

...

Stacked tensor:

X^(1).shape = [4, 4, NumberOfClauses]
```

------------------------------------------------------------------------

# 4. Extract Local Patches

``` text
FUNCTION EXTRACT_PATCHES(X, patch_height, patch_width, stride):

    patches = empty list

    H = height(X)
    W = width(X)
    C = channels(X)

    FOR row = 0 TO H - patch_height STEP stride:

        FOR col = 0 TO W - patch_width STEP stride:

            patch =
                X[
                    row : row + patch_height,
                    col : col + patch_width,
                    0 : C
                ]

            ADD (
                patch,
                row,
                col
            ) TO patches

        END FOR

    END FOR

    RETURN patches
```

The number of valid spatial positions is:

\[ H\_{out} = `\left`{=tex}`\lfloor`{=tex} `\frac{H-K_h}{S}`{=tex}
`\right`{=tex}`\rfloor `{=tex}+ 1 \]

\[ W\_{out} = `\left`{=tex}`\lfloor`{=tex} `\frac{W-K_w}{S}`{=tex}
`\right`{=tex}`\rfloor `{=tex}+ 1 \]

------------------------------------------------------------------------

# 5. Convert a Patch into Literals

Suppose a patch contains (N) binary features:

\[ x_1,x_2,`\ldots`{=tex},x_N \]

A Tsetlin clause can access both the original and negated literals:

\[ x_1,`\ldots`{=tex},x_N,
`\neg `{=tex}x_1,`\ldots`{=tex},`\neg `{=tex}x_N \]

``` text
FUNCTION PATCH_TO_LITERALS(patch):

    x = FLATTEN(patch)

    literals = empty vector

    FOR each binary feature x_i in x:

        APPEND x_i TO literals

    END FOR

    FOR each binary feature x_i in x:

        APPEND NOT(x_i) TO literals

    END FOR

    RETURN literals
```

Therefore, if:

``` text
patch size = 3 x 3
input channels = 64
```

then:

``` text
N = 3 * 3 * 64
  = 576
```

and the clause has access to:

``` text
2 * 576 = 1152 literals
```

------------------------------------------------------------------------

# 6. Tsetlin Automaton State

Each clause contains one Tsetlin Automaton for each possible literal.

Each automaton decides:

``` text
INCLUDE literal
or
EXCLUDE literal
```

``` text
STRUCTURE Clause:

    automata_states[number_of_literals]

    polarity
        POSITIVE
        or
        NEGATIVE
```

A simple action function is:

``` text
FUNCTION AUTOMATON_ACTION(state, N_states):

    IF state > N_states:
        RETURN INCLUDE

    ELSE:
        RETURN EXCLUDE
```

------------------------------------------------------------------------

# 7. Evaluate One Clause on One Patch

``` text
FUNCTION EVALUATE_CLAUSE_ON_PATCH(clause, patch):

    literals = PATCH_TO_LITERALS(patch)

    clause_output = 1
    included_literal_exists = FALSE

    FOR k = 1 TO number_of_literals:

        action = AUTOMATON_ACTION(
            clause.automata_states[k]
        )

        IF action == INCLUDE:

            included_literal_exists = TRUE

            IF literals[k] == 0:

                clause_output = 0

                BREAK

            END IF

        END IF

    END FOR

    IF included_literal_exists == FALSE:
        clause_output = EMPTY_CLAUSE_POLICY

    RETURN clause_output
```

Conceptually, a learned clause might represent:

\[ C_j = x_1 `\land`{=tex} `\neg `{=tex}x_5 `\land`{=tex} x\_{17}
`\land`{=tex} x\_{32} \]

The clause outputs `1` only when all included literals are satisfied.

------------------------------------------------------------------------

# 8. Apply One Clause Convolutionally

The same clause is evaluated at every spatial position.

``` text
FUNCTION APPLY_CLAUSE_CONVOLUTIONALLY(
    clause,
    X,
    patch_height,
    patch_width,
    stride
):

    patches = EXTRACT_PATCHES(
        X,
        patch_height,
        patch_width,
        stride
    )

    H_out = COMPUTE_OUTPUT_HEIGHT(...)
    W_out = COMPUTE_OUTPUT_WIDTH(...)

    activation_map =
        ZERO_MATRIX(H_out, W_out)

    FOR each (patch, row, col) in patches:

        r_out = row / stride
        c_out = col / stride

        activation =
            EVALUATE_CLAUSE_ON_PATCH(
                clause,
                patch
            )

        activation_map[r_out, c_out] =
            activation

    END FOR

    RETURN activation_map
```

Unlike a conventional CTM classification stage, **do not OR-pool this
map yet**.

------------------------------------------------------------------------

# 9. Produce All Clause Feature Maps

``` text
FUNCTION FORWARD_CTM_LAYER(layer, X):

    feature_maps = empty tensor

    FOR clause_index = 1 TO layer.number_of_clauses:

        clause =
            layer.clauses[clause_index]

        map =
            APPLY_CLAUSE_CONVOLUTIONALLY(
                clause,
                X,
                layer.patch_height,
                layer.patch_width,
                layer.stride
            )

        feature_maps[:, :, clause_index] =
            map

    END FOR

    RETURN feature_maps
```

Thus:

``` text
Input:
    H x W x C_in

Output:
    H_out x W_out x NumberOfClauses
```

------------------------------------------------------------------------

# 10. Optional Positional Encoding

Spatial position can be represented explicitly if desired.

``` text
FUNCTION ADD_POSITION_FEATURES(patch, row, col):

    row_features =
        BINARY_ENCODE(row)

    column_features =
        BINARY_ENCODE(col)

    RETURN CONCATENATE(
        FLATTEN(patch),
        row_features,
        column_features
    )
```

This allows clauses to distinguish between patterns occurring at
different locations.

------------------------------------------------------------------------

# 11. Clause-Level Spatial Aggregation for Classification

For a classification layer, the spatial activation map of each clause
can be OR-pooled:

``` text
FUNCTION SPATIAL_OR_POOL(feature_map):

    FOR every position (r, c):

        IF feature_map[r, c] == 1:
            RETURN 1

    RETURN 0
```

Therefore:

\[ z_j = `\bigvee`{=tex}\_{r,c} F_j(r,c) \]

where (z_j) indicates whether clause (j) matched anywhere.

------------------------------------------------------------------------

# 12. Class Voting

Assume positive and negative clauses.

``` text
FUNCTION CLASS_SCORE(layer, X, class_id):

    feature_maps =
        FORWARD_CTM_LAYER(layer, X)

    score = 0

    FOR each clause j assigned to class_id:

        clause_prediction =
            SPATIAL_OR_POOL(
                feature_maps[:, :, j]
            )

        IF clause[j].polarity == POSITIVE:

            score =
                score + clause_prediction

        ELSE:

            score =
                score - clause_prediction

    END FOR

    RETURN score
```

For multiclass classification:

``` text
FUNCTION PREDICT(layer, X):

    best_class = NONE
    best_score = -INFINITY

    FOR class_id = 1 TO number_of_classes:

        score =
            CLASS_SCORE(
                layer,
                X,
                class_id
            )

        IF score > best_score:

            best_score = score
            best_class = class_id

    END FOR

    RETURN best_class
```

------------------------------------------------------------------------

# 13. Training One CTM Layer

The following represents the general Tsetlin Machine learning mechanism.
Exact Type I/Type II update probabilities can follow the chosen CTM/TM
implementation.

``` text
FUNCTION TRAIN_CTM_LAYER(
    layer,
    dataset,
    number_of_epochs
):

    FOR epoch = 1 TO number_of_epochs:

        SHUFFLE(dataset)

        FOR each (X, y) in dataset:

            feature_maps =
                FORWARD_CTM_LAYER(
                    layer,
                    X
                )

            scores =
                COMPUTE_CLASS_SCORES(
                    layer,
                    feature_maps
                )

            FOR each clause j:

                feedback_type =
                    DETERMINE_FEEDBACK(
                        true_label = y,
                        class_scores = scores,
                        clause_polarity =
                            clause[j].polarity,
                        threshold = layer.T
                    )

                matching_positions =
                    FIND_MATCHING_POSITIONS(
                        feature_maps[:, :, j]
                    )

                selected_patch =
                    SELECT_PATCH_FOR_FEEDBACK(
                        X,
                        matching_positions,
                        layer.patch_size,
                        layer.stride
                    )

                IF feedback_type == TYPE_I:

                    APPLY_TYPE_I_FEEDBACK(
                        clause[j],
                        selected_patch,
                        layer.s
                    )

                ELSE IF feedback_type == TYPE_II:

                    APPLY_TYPE_II_FEEDBACK(
                        clause[j],
                        selected_patch
                    )

                END IF

            END FOR

        END FOR

    END FOR
```

------------------------------------------------------------------------

# 14. Type I Feedback

Type I feedback encourages clauses to recognize patterns associated with
the target class.

``` text
FUNCTION APPLY_TYPE_I_FEEDBACK(
    clause,
    patch,
    s
):

    literals =
        PATCH_TO_LITERALS(patch)

    clause_output =
        EVALUATE_CLAUSE_ON_PATCH(
            clause,
            patch
        )

    FOR literal k:

        action =
            AUTOMATON_ACTION(
                clause.automata_states[k]
            )

        IF clause_output == 1:

            IF literals[k] == 1:

                WITH high probability controlled by s:

                    REWARD_INCLUDE(
                        automaton[k]
                    )

            ELSE:

                WITH low probability:

                    PENALIZE_INCLUDE(
                        automaton[k]
                    )

        ELSE:

            WITH low probability controlled by s:

                ENCOURAGE_EXCLUSION(
                    automaton[k]
                )

        END IF

    END FOR
```

The precise probabilities should follow the selected Tsetlin Machine
learning rule.

------------------------------------------------------------------------

# 15. Type II Feedback

Type II feedback suppresses false-positive clauses.

``` text
FUNCTION APPLY_TYPE_II_FEEDBACK(
    clause,
    patch
):

    literals =
        PATCH_TO_LITERALS(patch)

    clause_output =
        EVALUATE_CLAUSE_ON_PATCH(
            clause,
            patch
        )

    IF clause_output == 1:

        FOR literal k:

            IF literals[k] == 0:

                IF automaton[k] currently EXCLUDES literal:

                    PENALIZE_EXCLUSION(
                        automaton[k]
                    )

                    // Moves automaton toward INCLUDE.
                    // Including this false literal in future
                    // makes the clause less likely to fire
                    // on this unwanted pattern.

                END IF

            END IF

        END FOR

    END IF
```

------------------------------------------------------------------------

# 16. Layer-Wise Deep Training

This is the central algorithm.

``` text
FUNCTION TRAIN_DEEP_CTM(
    original_dataset,
    layer_configs
):

    models = empty list

    current_dataset =
        original_dataset

    FOR l = 1 TO number_of_layers:

        PRINT("Training layer", l)

        layer =
            INITIALIZE_LAYER(
                layer_configs[l]
            )

        TRAIN_CTM_LAYER(
            layer,
            current_dataset,
            layer_configs[l].epochs
        )

        ADD layer TO models

        IF l < number_of_layers:

            transformed_dataset =
                empty dataset

            FOR each (X, y)
                in current_dataset:

                F =
                    FORWARD_CTM_LAYER(
                        layer,
                        X
                    )

                ADD (F, y)
                    TO transformed_dataset

            END FOR

            current_dataset =
                transformed_dataset

            FREEZE(layer)

        END IF

    END FOR

    RETURN models
```

------------------------------------------------------------------------

# 17. Inference Through the Complete Network

``` text
FUNCTION DEEP_CTM_PREDICT(models, X):

    representation = X

    FOR l = 1 TO number_of_layers - 1:

        representation =
            FORWARD_CTM_LAYER(
                models[l],
                representation
            )

    END FOR

    prediction =
        PREDICT(
            models[last_layer],
            representation
        )

    RETURN prediction
```

Therefore inference follows:

``` text
raw image
    |
    v
CTM Layer 1
    |
    v
binary clause maps
    |
    v
CTM Layer 2
    |
    v
binary higher-order clause maps
    |
    v
...
    |
    v
final CTM
    |
    v
class votes
    |
    v
prediction
```

------------------------------------------------------------------------

# 18. Example Two-Layer Architecture

Suppose:

``` text
Input:
    28 x 28 binary image
```

Layer 1:

``` text
Patch size:       3 x 3
Stride:           1
Clauses:          64
```

The output is approximately:

``` text
26 x 26 x 64
```

Layer 2:

``` text
Input:
    26 x 26 x 64

Patch size:
    3 x 3

Stride:
    2

Clauses:
    128
```

Each Layer-2 patch contains:

``` text
3 * 3 * 64
= 576 binary features
```

and therefore:

``` text
1152 possible literals
```

including negations.

Layer 2 produces:

``` text
12 x 12 x 128
```

binary clause maps.

A final classification layer can aggregate these maps and vote for the
output class.

------------------------------------------------------------------------

# 19. Optional Residual Connections

Deep Boolean representations may lose useful information. Residual or
skip connections can preserve earlier features.

Conceptually:

\[ X\^{(l+1)} = CTM_l( \[ X\^{(l)},, R(X\^{(l-1)})\] ) \]

where (R) spatially aligns the earlier representation.

``` text
FUNCTION RESIDUAL_CTM_LAYER(
    layer,
    current_features,
    earlier_features
):

    aligned_earlier =
        RESIZE_OR_POOL(
            earlier_features,
            target_spatial_size =
                current_features.spatial_size
        )

    combined =
        CONCATENATE_CHANNELS(
            current_features,
            aligned_earlier
        )

    output =
        FORWARD_CTM_LAYER(
            layer,
            combined
        )

    RETURN output
```

------------------------------------------------------------------------

# 20. Optional Pooling

Pooling can reduce spatial dimensionality between layers.

### OR pooling

``` text
FUNCTION OR_POOL(binary_map, pool_size):

    FOR every pooling region R:

        output[R] =
            OR of all bits in R

    RETURN output
```

For example:

``` text
0 1
0 0
```

becomes:

``` text
1
```

This is particularly natural for Boolean representations because it
means:

> The feature was detected somewhere inside this local region.

------------------------------------------------------------------------

# 21. Proposed Full Architecture

``` text
INPUT IMAGE
    |
    v
Binarization / Boolean Encoding
    |
    v
+-------------------------+
| CTM Layer 1             |
| Local raw-pixel clauses |
+-------------------------+
    |
    v
Binary clause feature maps
    |
    v
Optional OR Pooling
    |
    v
+-----------------------------+
| CTM Layer 2                 |
| Pattern-combination clauses |
+-----------------------------+
    |
    v
Higher-order binary maps
    |
    v
Optional OR Pooling
    |
    v
+-----------------------------+
| CTM Layer 3                 |
| Object-part / structure     |
| clauses                     |
+-----------------------------+
    |
    v
Binary semantic maps
    |
    v
Spatial OR aggregation
    |
    v
Positive / Negative
Clause Voting
    |
    v
CLASS PREDICTION
```

------------------------------------------------------------------------

# 22. Important Difference from a CNN

A CNN layer computes approximately:

\[ X\^{(l+1)} = `\sigma`{=tex}( W\^{(l)} \* X\^{(l)} + b ) \]

where (W) contains real-valued learned convolutional weights.

The proposed multilayer CTM instead computes:

\[ X\^{(l+1)}*j(r,c) = `\bigwedge`{=tex}*{k `\in `{=tex}I_j}
L_k(P\^{(l)}\_{r,c}) \]

where:

-   (P\^{(l)}\_{r,c}) is a local binary patch;
-   (L_k) is either a feature or its negation;
-   (I_j) is the set of literals selected by the Tsetlin Automata of
    clause (j).

Thus the hierarchy becomes:

``` text
CNN:

pixels
 -> weighted filters
 -> numerical features
 -> weighted filters
 -> numerical features


Multilayer CTM:

binary features
 -> logical clauses
 -> binary feature maps
 -> logical clauses
 -> binary higher-order features
```

------------------------------------------------------------------------

# 23. Important Research Questions

A practical implementation should investigate several design choices:

1.  **How should intermediate CTM layers be trained?**\
    Purely supervised training, auxiliary classifiers, layer-wise
    targets, self-supervised objectives, or another local objective may
    be required.

2.  **Should positive and negative clauses both be propagated?**\
    The next layer may benefit from both types, or polarity may only be
    needed at the classification stage.

3.  **Should spatial OR pooling be used between layers?**\
    Pooling provides local translation invariance but discards precise
    spatial information.

4.  **How many clauses should each layer contain?**\
    More clauses increase representational capacity but also increase
    the number of input channels and literals in later layers.

5.  **How should clause explosion be controlled?**\
    If Layer 1 contains (C) clauses and Layer 2 uses (K`\times `{=tex}K)
    patches, Layer 2 receives:

    \[ K\^2 C \]

    Boolean features and potentially:

    \[ 2K\^2C \]

    literals.

6.  **Should skip connections be used?**\
    They may preserve low-level information that disappears through
    successive Boolean transformations.

7.  **How should empty clauses be handled?**\
    The implementation must define the output of clauses containing no
    included literals.

8.  **How should spatial locations be encoded?**\
    Explicit position literals may improve spatial reasoning but reduce
    translation invariance.

9.  **Can the complete architecture be trained jointly?**\
    Standard backpropagation is not directly applicable to discrete
    Tsetlin Automata, so joint learning would require a different
    feedback mechanism.

------------------------------------------------------------------------

# 24. Minimal End-to-End Pseudocode

``` text
FUNCTION MULTILAYER_CTM_TRAIN(dataset):

    # --------------------------------
    # Layer 1
    # --------------------------------

    CTM1 =
        CREATE_CTM(
            patch = 3x3,
            clauses = 64
        )

    TRAIN(CTM1, dataset)

    D1 = empty dataset

    FOR each (image, label) in dataset:

        F1 =
            SPATIAL_CLAUSE_OUTPUTS(
                CTM1,
                image
            )

        ADD (F1, label) TO D1

    FREEZE(CTM1)


    # --------------------------------
    # Layer 2
    # --------------------------------

    CTM2 =
        CREATE_CTM(
            patch = 3x3,
            clauses = 128
        )

    TRAIN(CTM2, D1)

    D2 = empty dataset

    FOR each (F1, label) in D1:

        F2 =
            SPATIAL_CLAUSE_OUTPUTS(
                CTM2,
                F1
            )

        ADD (F2, label) TO D2

    FREEZE(CTM2)


    # --------------------------------
    # Final classifier
    # --------------------------------

    classifier =
        CREATE_TSETLIN_CLASSIFIER()

    TRAIN(
        classifier,
        SPATIAL_AGGREGATE(D2)
    )


    RETURN {
        CTM1,
        CTM2,
        classifier
    }
```

Inference:

``` text
FUNCTION MULTILAYER_CTM_INFERENCE(model, image):

    F1 =
        SPATIAL_CLAUSE_OUTPUTS(
            model.CTM1,
            image
        )

    F2 =
        SPATIAL_CLAUSE_OUTPUTS(
            model.CTM2,
            F1
        )

    Z =
        SPATIAL_AGGREGATE(F2)

    y_hat =
        model.classifier.PREDICT(Z)

    RETURN y_hat
```

------------------------------------------------------------------------

# 25. Core Concept

The essential architectural change is:

``` text
Standard CTM:

Image
 -> clause evaluates all patches
 -> OR across positions
 -> one clause bit
 -> class voting
```

versus:

``` text
Proposed Multilayer CTM:

Image
 -> clause evaluates all patches
 -> KEEP spatial clause outputs
 -> binary feature maps
 -> next CTM convolution
 -> higher-level binary feature maps
 -> ...
 -> aggregate
 -> class voting
```

In mathematical form:

\[ X\^{(0)} `\xrightarrow{CTM_1}`{=tex} X\^{(1)}
`\xrightarrow{CTM_2}`{=tex} X\^{(2)} `\xrightarrow{}`{=tex}
`\cdots`{=tex} `\xrightarrow{CTM_L}`{=tex} X\^{(L)}
`\xrightarrow{\text{aggregation + voting}}`{=tex} `\hat{y}`{=tex} \]

This preserves the central properties of Tsetlin Machines---binary
representations, propositional clauses, interpretable literals, and
automata-based learning---while introducing a hierarchical spatial
representation analogous to depth in convolutional neural networks.
