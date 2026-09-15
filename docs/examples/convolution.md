# Convolution on generated data

Three small scripts that each take a minute or less and need no download. They cover the
convolutional models that [MNIST](mnist.md) does not: 1-D sequences and convolutional
regression, plus a controlled look at what `position_encoding` actually does. See
[Convolution](../concepts/convolution.md) for the algorithm.

## Translation invariance

Full script: `examples/shapes_conv.py`.

A hollow 3 × 3 circle or a 3 × 3 cross sits at a random position in a 12 × 12 image — 100
possible positions. A flat machine has to memorise all of them; a convolutional one learns
one clause.

```python
import torch, torchtsetlin as tt

tt.seed_everything(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
x_train, y_train = tt.data.make_shapes(2000, size=12, seed=0)   # (N, 1, 12, 12) Boolean
x_test, y_test = tt.data.make_shapes(1000, size=12, seed=1)

model = tt.ConvTsetlinMachine(2, n_clauses=40, T=15, s=3.9, patch_size=3,
                              position_encoding=False, n_states=100)
model.class_names = ["circle", "cross"]
trainer = tt.Trainer(model, device=device, batch_size=16)
trainer.fit((x_train, y_train), epochs=10, val_data=(x_test, y_test))
print(trainer.test((x_test, y_test)))           # {'test_accuracy': 1.0}

# Clauses 0..39 vote for 'circle', 40..79 for 'cross'; rank them by how often they are right.
score = tt.interpret.clause_precision(model, x_test, y_test)
cross = (model.clause_class == 1) & (model.clause_polarity > 0) & (score["support"] > 0.05)
best = int(torch.where(cross, score["precision"], torch.zeros_like(score["precision"])).argmax())

print(model.clause_patch(best))                 # 1 = pixel on, -1 = off, 0 = don't care
print(model.clause_region(best))                # {'y': (0, 9), 'x': (0, 9)} — matches anywhere
```

The winner is not the whole cross but a *fragment* of it — and that is the point. The
script prints it with `#` = pixel must be on, `.` = must be off, blank = don't care:

```
clause 40 of 'cross': fires on 43% of images, 100% of them class 'cross'
  allowed window rows y=(0, 9), columns x=(0, 9)
|###|
|.# |
|...|
```

Three horizontally adjacent on pixels never occur in a hollow circle (`.#.` / `#.#` / `.#.`),
so the cross's arm plus an empty row below already separates the classes perfectly, and the
window range says the clause may match at any of the 100 positions. Nothing forces a clause
to learn the complete template when a shorter one is already precise.

Run `python examples/shapes_conv.py --flat` to train both models on the same images. The
convolutional one scored 1.000 on every seed tried; the flat one, with the same clause
budget, lands anywhere between 0.55 and 0.91.

!!! warning "Position literals are not free"

    `position_encoding=True` lets a clause demand *where* it matches. On a task where the
    shape can sit anywhere that is only a way to overfit: accuracy drops from 1.000 to about
    0.5 (`--position-encoding`). Digits are the opposite case — see
    [MNIST](mnist.md#variations-the-script-exposes) — so check this flag before suspecting
    the feedback code.

## Sequences and signals

Full script: `examples/conv1d_ramps.py`.

`Conv1dTsetlinMachine` expects `(B, Z, L)`: `Z` Boolean channels over `L` timesteps. A
thermometer encoder with `flatten=False` produces `(N, L, n_bits)`, so one `permute` puts the
bit planes where the channels belong.

```python
import torch, torchtsetlin as tt

# 48-sample signals of uniform noise with one 8-sample ramp hidden at a random offset:
# rising = class 1, falling = class 0 (see make_ramps() in the script).
x_train, y_train = make_ramps(3000, seed=0)
x_test, y_test = make_ramps(1000, seed=1)

enc = tt.data.ThermometerEncoder(n_bits=8, strategy="uniform",
                                 value_range=(0.0, 1.0), flatten=False)
b_train = enc(x_train).permute(0, 2, 1).contiguous()    # (N, 8, 48)
b_test = enc(x_test).permute(0, 2, 1).contiguous()

model = tt.Conv1dTsetlinMachine(2, n_clauses=50, T=30, s=5.0, kernel_size=8,
                                position_encoding=False, n_states=100)
trainer = tt.Trainer(model, device=device, batch_size=32)
trainer.fit((b_train, y_train), epochs=10, val_data=(b_test, y_test))
print(trainer.test((b_test, y_test)))           # {'test_accuracy': 0.996} from the script
```

A clause is a `(n_bits, 1, kernel_size)` pattern, so printing its bit planes with the highest
threshold on top draws the waveform it detects — here, a rising staircase:

```
clause 51 of 'rising': fires on 12% of signals, 100% of them class 'rising'
  ........
  .......#
   .....##
  #....###
  #....###
  #...####
  #..#####
  #.######
```

## Convolutional regression

Full script: `examples/conv_regression_blobs.py`.

`ConvRegressionTsetlinMachine` maps the vote sum linearly onto `y_range` instead of taking an
argmax. Here each image holds one solid square of side 2–6 and the target is that side
length — a quantity that does not depend on position, so `position_encoding=False` again.

```python
import torch, torchtsetlin as tt

x_train, y_train = make_squares(4000, seed=0)   # (N, 1, 12, 12) Boolean, float targets
x_test, y_test = make_squares(1000, seed=1)

model = tt.ConvRegressionTsetlinMachine(
    n_clauses=400, T=400, s=5.0, patch_size=7,  # the window must be wider than the square
    position_encoding=False,
    weighted=True,                              # one clause can vote several times
    y_range=(2.0, 6.0), n_states=100,
)
trainer = tt.Trainer(model, device=device, batch_size=16)
trainer.fit((x_train, y_train), epochs=30, val_data=(x_test, y_test))
print(trainer.test((x_test, y_test)))
# {'test_mae': 0.20, 'test_rmse': 0.27, 'test_r2': 0.96} from the script
```

Each clause adds `weight * (y_max - y_min) / T` to the prediction when it matches — 0.01 per
unit of weight at `T=400` over a range of 4 — and its patch shows what it is keying on. The
heaviest clause of a default run requires a three-pixel vertical run in one column and every
other pixel of the 7 × 7 window to be off — a three-tall edge with nothing else beside it:

```
clause 379: weight 47, 3 pixels required on, 46 required off
  #......
  #......
  #......
  .......
  .......
  .......
  .......
```

Read the next few and the tidiness stops: clauses 274 and 89 carry almost as much weight on a
scatter of unrelated pixels. A regression TM spreads its answer over hundreds of clauses, and
only some of them are individually legible.

!!! note "`weighted` and `T` have to be chosen together"

    At these settings `weighted=True` is not a refinement but a requirement. Unweighted
    clauses each contribute at most 1, so 400 of them can only reach a vote sum of 400 if
    *every* clause fires — and with `T=400` the same run collapses to R² −0.19. Either keep
    the weights, or drop `T` to roughly half the clause count (unweighted with `T=200`
    reaches R² 0.83).
