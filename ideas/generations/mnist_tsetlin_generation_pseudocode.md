# Generating MNIST Images with a Tsetlin Machine

## Goal and model

Generate a new 28 × 28 grayscale handwritten digit, optionally conditioned on its class (0–9). This design starts with a **binary** digit image, produced by thresholding the grayscale MNIST image. It therefore generates a binary image, not the original 256-level grayscale image. A later extension can model multiple gray levels.

An autoregressive Tsetlin Machine (TM) predicts the probability that the next pixel is ink, conditional on its digit label, its position and all pixels already generated. Sampling those predictions gives a new image. This is a proposed generative use of a TM; an ordinary classifier's raw clause votes are not inherently well-calibrated probabilities.

For raster order, write the pixels as x[0], ..., x[783]. The model approximates

```text
P(image | digit) = product over i=0..783 of P(x[i] | x[0:i], digit).
```

Training uses real previous pixels (*teacher forcing*); generation uses previously sampled pixels. This difference can cause errors to accumulate.

## Input, output and fixed conventions

| Item | Specification |
| --- | --- |
| Training input | MNIST image `I[28,28]`, intensities in a documented range; label `digit` in 0–9 |
| Training target | One binary pixel `x[i]` at position `i`, obtained using `I[row,col] > ink_threshold` |
| Generated output | Binary image `[28,28]`; map values 0/1 to 0/255 only for display |
| Order | `i = 28*row + col`, from top left to bottom right |
| Unknown pixels | Have a separate `observed` bit, never confused with a known zero |
| Model | Binary TM classifier, trained incrementally on multiple pixel examples |
| Sampling | Bernoulli using validation-calibrated `P(pixel=1 | context)` |

Split by image before constructing pixel examples. Fit thresholds or calibration using only the training or validation subset respectively. Keep the encoding and generation order identical at training and sampling time.

### Context representation

The simplest explicit encoding includes the whole partial canvas and a mask that marks which positions are known, plus a one-hot digit and one-hot current position. This is large: `784 known bits + 784 observed bits + 10 digit bits + 784 position bits = 2,362 features` before negated literals. It makes the algorithm unambiguous but may be expensive. A smaller practical variant uses a fixed window of *earlier* pixels, a row/column encoding and the digit label. Never include the target pixel or any future pixel in the context.

## Full pseudocode

```python
CONSTANTS:
    HEIGHT = WIDTH = 28
    NUM_PIXELS = 784
    DIGITS = 10
    UNKNOWN = 0       # The canvas value is ignored if observed==0.


function BINARIZE(image, ink_threshold):
    assert image.shape == [28, 28]
    return integer_array(image > ink_threshold).flatten_in_raster_order()


function ENCODE_CONTEXT(partial_pixels[784], observed[784],
                        position, digit):
    assert 0 <= position < 784 and 0 <= digit < 10
    assert observed[position] == 0
    assert all(observed[j] == 1 for j < position)
    assert all(observed[j] == 0 for j >= position)

    # Fixed-size features with exactly the same ordering everywhere.
    known_value[j] = partial_pixels[j] if observed[j] else 0
                     for j in 0..783
    observed_bits = observed[0:784]
    digit_bits = ONE_HOT(digit, length=10)
    position_bits = ONE_HOT(position, length=784)
    return CONCATENATE(known_value, observed_bits,
                       digit_bits, position_bits)


function MAKE_TRAINING_EXAMPLES(images, labels, ink_threshold):
    for image, digit in ZIP(images, labels):
        target_pixels = BINARIZE(image, ink_threshold)
        partial = ZEROS(784)
        observed = ZEROS(784)

        for i in 0..783:
            features = ENCODE_CONTEXT(partial, observed, i, digit)
            yield features, target_pixels[i]

            # Reveal the true pixel AFTER constructing the example.
            partial[i] = target_pixels[i]
            observed[i] = 1


function TRAIN(train_images, train_labels, val_images, val_labels, config):
    tm = INITIALIZE_BINARY_TM(num_features=2362,
                              clauses=config.clauses,
                              T=config.T, s=config.s,
                              seed=config.seed)
    # The TM adapter must preserve automaton state across update_batch calls.
    best_state = NONE
    best_validation_nll = +INFINITY
    patience_counter = 0

    for epoch in 1..config.max_epochs:
        SHUFFLE_TRAIN_IMAGES_WITH_SEED(train_images, train_labels,
                                       config.seed, epoch)
        stream = MAKE_TRAINING_EXAMPLES(train_images, train_labels,
                                        config.ink_threshold)
        for features_batch, labels_batch in BATCH(stream, config.batch_size):
            tm.update_batch(features_batch, labels_batch)

        # Fit the score-to-probability map using validation examples only.
        validation_scores, validation_targets = SCORE_ALL_EXAMPLES(
            tm, MAKE_TRAINING_EXAMPLES(val_images, val_labels,
                                       config.ink_threshold))
        calibrator = FIT_CALIBRATOR(validation_scores, validation_targets)
        p = CLIP(calibrator(validation_scores), epsilon, 1-epsilon)
        validation_nll = MEAN_BINARY_NEGATIVE_LOG_LIKELIHOOD(
            validation_targets, p)

        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_state = DEEP_COPY(tm.state)
            best_calibrator = DEEP_COPY(calibrator)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    SAVE(tm_state=best_state, calibrator=best_calibrator,
         ink_threshold=config.ink_threshold,
         encoding='full partial canvas + observed + digit + position',
         generation_order='raster', best_epoch=best_epoch,
         seed=config.seed, model_config=config)


function GENERATE_DIGIT(saved_model, requested_digit, sample_seed):
    assert requested_digit in 0..9
    rng = RANDOM_GENERATOR(sample_seed)
    tm, calibrator = LOAD(saved_model)
    pixels = ZEROS(784)
    observed = ZEROS(784)

    for i in 0..783:
        features = ENCODE_CONTEXT(pixels, observed, i, requested_digit)
        raw_score = tm.score_for_ink(features)
        p_ink = CLIP(calibrator(raw_score), epsilon, 1-epsilon)
        pixels[i] = SAMPLE_BERNOULLI(p_ink, rng)
        observed[i] = 1

    return RESHAPE(pixels, [28, 28])


function EVALUATE_GENERATOR(saved_model, held_out_images,
                            held_out_labels, seeds):
    generated = []
    for digit in 0..9:
        for seed in seeds:
            generated.append(GENERATE_DIGIT(saved_model, digit, seed))

    # Compare generated and held-out images, never tune with the test set.
    REPORT_VISUAL_GRID_BY_DIGIT(generated)
    REPORT_DIGIT_CONSISTENCY_WITH_SEPARATELY_TRAINED_CLASSIFIER(generated)
    REPORT_DIVERSITY_AND_NEAREST_TRAINING_IMAGE_DISTANCES(generated)
    REPORT_HELD_OUT_NEXT_PIXEL_NLL(saved_model, held_out_images,
                                   held_out_labels)
```

## Reading the pseudocode

1. **Binarize.** A threshold converts each grayscale image into 784 binary targets. This intentionally loses gray level detail and is appropriate for a small proof of concept.
2. **Build a valid partial image.** At position `i`, positions `< i` contain actual observed values during training; positions `>= i` are marked unknown. `ENCODE_CONTEXT` cannot inspect the target before predicting it.
3. **Train one shared TM.** Each pixel position contributes a training example, but all positions use one model. Position and digit features let its clauses learn different behavior in different areas and classes.
4. **Calibrate and select.** Fit a mapping from TM scores to probabilities on validation examples and retain the checkpoint with best validation negative log likelihood. This metric measures next-pixel prediction; it does not alone establish visual quality.
5. **Generate sequentially.** Begin with an empty canvas; draw each next pixel from a Bernoulli distribution. Seeds control sampling reproducibly.

## Practical limitations and extensions

- The explicit context costs memory and computing time; 784 decisions per image also make sampling slow. A local past-only context is cheaper but loses distant context.
- Generated digits may become malformed when early sampled pixels put the model in contexts seldom seen during teacher-forced training. Inspect class fidelity and diversity together.
- A raw TM class decision or `argmax` does not yield varied samples; calibration should be assessed with held-out reliability metrics.
- For grayscale generation, model a quantized intensity class (for example 16 bins) at each position, calibrate a categorical distribution, and sample one bin. Record quantization and image reconstruction rules.
- Do not interpret nearest-neighbour distance alone as a privacy guarantee; inspect copying and memorization as separate questions.
