# Generating 12-Lead ECG Signals with a Tsetlin Machine

## Goal and proposed architecture

Generate an entire 12-lead waveform at a specified sampling rate and duration, optionally conditioned on a rhythm label or other attributes available in the training records. All leads describe the same cardiac activity. This design learns a **joint multilead block codebook** and uses a Tsetlin Machine (TM) to predict the next block code from previous codes.

For a 10-second recording at 500 Hz, the target shape is `[12, 5000]`. With a block length `B=50` samples (100 ms), there are 100 code predictions per ECG. The choice of `B=50` is illustrative. This is a research design to test; there is no presumption that simple block codes preserve diagnostic ECG details.

The codebook is a set of `K` representative blocks, each `[12,B]`. A code is one categorical token describing **all leads in a time block together**. The TM models the distribution of code tokens; the codebook decodes them into continuous-valued waveforms.

```text
P(codes | condition) = product over b of
                      P(code[b] | codes[0:b], condition).
waveform ≈ concatenate(codebook[code[b]] over all b).
```

This architecture guarantees equal time indexing of leads within a block. It does **not** guarantee physiological cross-lead relationships, smooth joins, beat continuity or realistic rhythm transitions; those require testing.

## Data contract and important choices

| Item | Specification |
| --- | --- |
| Input record | `signal[12,T]`, ordered by a saved lead list, known sampling rate and measurement units |
| Split unit | Patient; all records from one patient remain in a single split |
| Target generation | `signal[12, requested_seconds × sample_rate]` |
| Training block | `[12,B]`, nonoverlapping, after training-fitted preprocessing |
| Codebook | `K` jointly learned or clustered 12-lead prototypes `[K,12,B]` |
| TM target | Integer block ID `0..K-1` |
| Context | Last `L` block IDs, explicit beginning-of-sequence sentinel, current index and condition |
| Sampling | Validation-calibrated categorical distribution over all `K` IDs |

Use a canonical lead order, for example `I, II, III, aVR, aVL, aVF, V1, ..., V6`, and save it. Resampling, units conversion and quality filtering must be specified before training. Use one fixed sample rate per model. A block-code approach compresses data and may lose small clinically important features, so quantify reconstruction error **before** judging generation quality.

For a fixed-record baseline, require `T` divisible by `B` after preprocessing. For another requested duration, predict `ceil(T/B)` blocks and trim the final output. Conditioning attributes must be available at generation time; never include diagnostic labels inferred from the record being generated.

## Full pseudocode

```python
function PREPARE_DATA(records, sampling_rate, lead_order, B, split_seed):
    train, validation, test = SPLIT_BY_PATIENT(records, split_seed)
    assert PATIENT_SETS_ARE_DISJOINT(train, validation, test)

    for subset in [train, validation, test]:
        for record in subset:
            assert record.lead_names == lead_order
            assert record.units_are_known
            signal = CONVERT_UNITS_AND_RESAMPLE(record.signal, sampling_rate)
            assert signal.shape[0] == 12
            assert ALL_FINITE(signal)
            record.prepared_signal = signal

    return train, validation, test


function FIT_PREPROCESSING_AND_CODEBOOK(train, B, K, seed):
    # Fit fixed per-lead scales using training records only. Preserve actual
    # amplitude differences; do not normalize each beat or record separately.
    scaler = FIT_ROBUST_PER_LEAD_SCALER(train.prepared_signals)
    blocks = []
    for record in train:
        Z = scaler.transform(record.prepared_signal)
        blocks.extend(SPLIT_INTO_FULL_NONOVERLAPPING_BLOCKS(Z, B))

    codebook = FIT_JOINT_12_LEAD_CODEBOOK(blocks, K, seed)
    # E.g., clustering [12,B] blocks; prototypes are continuous arrays.
    reconstruction = DECODE(ENCODE_WITH_CODEBOOK(blocks, codebook), codebook)
    REPORT_RECONSTRUCTION_METRICS(blocks, reconstruction,
                                  including_morphology_and_seam_errors=True)
    return scaler, codebook


function TOKENIZE(record, scaler, codebook, B):
    Z = scaler.transform(record.prepared_signal)
    blocks = SPLIT_INTO_FULL_NONOVERLAPPING_BLOCKS(Z, B)
    return [NEAREST_JOINT_CODE_ID(block, codebook) for block in blocks]


function ENCODE_CONTEXT(previous_tokens, block_index, condition,
                        K, L, max_blocks):
    # Exactly L token slots; BOS is distinct from every valid code ID.
    context = LAST_L_TOKENS_PADDED_ON_LEFT(previous_tokens, L, BOS=K)
    assert all(token in 0..K for token in context)
    token_bits = CONCATENATE(ONE_HOT(token, K+1) for token in context)
    index_bits = ENCODE_POSITION(block_index, max_blocks)
    condition_bits = ENCODE_KNOWN_CONDITIONS(condition)
    return CONCATENATE(token_bits, index_bits, condition_bits)


function TOKEN_EXAMPLES(records, scaler, codebook, config):
    for record in records:
        tokens = TOKENIZE(record, scaler, codebook, config.B)
        assert length(tokens) <= config.max_blocks
        for b in 0..length(tokens)-1:
            X = ENCODE_CONTEXT(tokens[0:b], b, record.condition,
                               config.K, config.L, config.max_blocks)
            yield X, tokens[b]
            # Next example sees real earlier tokens (teacher forcing).


function TRAIN_GENERATOR(train, validation, config):
    scaler, codebook = FIT_PREPROCESSING_AND_CODEBOOK(
        train, config.B, config.K, config.seed)
    tm = INITIALIZE_MULTICLASS_TM(
        classes=range(config.K),
        feature_count=ENCODED_CONTEXT_SIZE(config),
        clauses=config.clauses, T=config.T, s=config.s,
        seed=config.seed)

    best_nll = +INFINITY
    no_improvement = 0
    for epoch in 1..config.max_epochs:
        SHUFFLE_RECORDS_WITH_SEED(train, config.seed, epoch)
        for X_batch, y_batch in BATCH(
                TOKEN_EXAMPLES(train, scaler, codebook, config),
                config.batch_size):
            tm.update_batch(X_batch, y_batch)  # Preserve automaton states.

        scores, targets = SCORE_ALL_TOKEN_EXAMPLES(
            tm, TOKEN_EXAMPLES(validation, scaler, codebook, config))
        calibrator = FIT_MULTICLASS_CALIBRATOR(scores, targets)
        probs = CLIP_AND_RENORMALIZE(calibrator(scores))
        val_nll = MEAN_CATEGORICAL_NLL(targets, probs)

        if val_nll < best_nll:
            best_nll = val_nll
            best = DEEP_COPY(tm.state, calibrator, scaler, codebook,
                             config, epoch, lead_order, sampling_rate,
                             units, condition_encoding)
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            break

    SAVE(best)
    return best


function GENERATE_ECG(best, duration_seconds, condition, sample_seed):
    assert condition_is_valid_for_training_encoding(condition)
    desired_samples = ROUND(duration_seconds * best.sampling_rate)
    n_blocks = CEIL(desired_samples / best.config.B)
    assert 1 <= n_blocks <= best.config.max_blocks

    rng = RANDOM_GENERATOR(sample_seed)
    tokens = []
    for b in 0..n_blocks-1:
        X = ENCODE_CONTEXT(tokens, b, condition,
                           best.config.K, best.config.L,
                           best.config.max_blocks)
        scores = best.tm.class_scores(X)
        probs = CLIP_AND_RENORMALIZE(best.calibrator(scores))
        tokens.append(SAMPLE_CATEGORICAL(probs, rng))

    scaled_blocks = [best.codebook[token] for token in tokens]
    scaled_signal = CONCATENATE_ALONG_TIME(scaled_blocks)[:, 0:desired_samples]
    signal = best.scaler.inverse_transform(scaled_signal)
    return signal  # [12, desired_samples], in the saved physical units.


function EVALUATE(best, held_out_records, generation_seeds):
    # First assess compression on held-out real ECGs.
    REPORT_CODEBOOK_RECONSTRUCTION_ERROR(held_out_records, best)
    REPORT_HELD_OUT_NEXT_TOKEN_NLL(held_out_records, best)

    for condition in PREDECLARED_EVALUATION_CONDITIONS:
        generated = [GENERATE_ECG(best, target_duration, condition, seed)
                     for seed in generation_seeds]
        CHECK_SHAPE_UNITS_AND_FINITE_VALUES(generated)
        REPORT_BLOCK_BOUNDARY_DISCONTINUITIES(generated, best.config.B)
        REPORT_HEART_RATE_RR_AND_RHYTHM(generated, condition)
        REPORT_P_QRS_T_MORPHOLOGY_AND_INTERVALS(generated)
        REPORT_CROSS_LEAD_CONSISTENCY(generated)
        REPORT_DIVERSITY_AND_NEAREST_TRAINING_RECORDS(generated)
```

## Reading the pseudocode

1. **Prepare records.** Group by patient before training and apply the same units, lead order and sampling rate. Fit scaling using training data only, so the generator can restore meaningful amplitude units.
2. **Learn a joint vocabulary.** Each codebook entry contains all 12 leads for one short interval. Encoding a real ECG yields a token sequence, while decoding tokens yields a waveform. Check reconstruction quality here: the generator cannot recover details discarded by the codebook.
3. **Train one next-token TM.** The model predicts one of `K` block IDs from the previous `L` IDs, its temporal position and permitted conditioning variables. Every valid class is registered at initialization even if a minibatch omits it.
4. **Calibrate.** Convert clause scores to a categorical distribution using validation data, save the best validation checkpoint and sample codes from that distribution. This is an estimated conditional distribution; the TM itself is not a likelihood model by default.
5. **Generate and decode.** Begin with `BOS` context, repeatedly sample code IDs, decode them into multilead blocks, concatenate them and reverse scaling. Generated blocks may have abrupt jumps at the joins; the baseline leaves them visible for assessment.

## Research considerations

- **Block boundaries:** Prototype centroids learned independently need not match at adjoining edges. Measure jumps and compare with real ECGs. If necessary, develop overlap-conditioned decoding or a separate continuity mechanism and evaluate whether it alters P waves, QRS complexes or ST segments.
- **Rhythm continuity:** A context of only `L` blocks may be too short to model beat timing or sustained AF. Try longer history or explicit conditioning on cycle phase/rhythm state, then compare with the baseline.
- **Codebook capacity:** Larger `K` improves reconstruction potential but increases the TM output classes and calibration burden. Tune using training and validation only.
- **Sampling costs:** Unlike deterministic argmax, categorical sampling yields variation. Inspect mode collapse and implausible rare token transitions.
- **Clinical interpretation:** Match recorded acquisition conditions; assess lead geometry, morphology, RR statistics and clinically salient features with domain experts before any clinical use. A plausible image of an ECG is insufficient evidence of signal fidelity.
- **Privacy:** Inspect exact and near-duplicate segments against training ECGs; patient splitting and good reconstruction do not themselves prevent memorization.
- **More precise alternative:** Predict quantized samples one time step at a time, jointly or in a fixed lead order. This avoids a lossy block codebook but makes a 10-second, 500-Hz, 12-lead sample require up to 60,000 token predictions and careful cross-lead modeling.
