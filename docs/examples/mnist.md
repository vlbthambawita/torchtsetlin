# MNIST

Full scripts: `examples/mnist_flat.py` and `examples/mnist_conv.py` (require torchvision).

## Flat Tsetlin machine

```python
import torchtsetlin as tt

x_train, y_train = tt.data.load_mnist_boolean("./data", train=True, threshold=0.3, device="cuda")
x_test, y_test = tt.data.load_mnist_boolean("./data", train=False, threshold=0.3, device="cuda")

model = tt.TsetlinMachine(784, 10, n_clauses=2000, T=50, s=10.0, weighted=True)
trainer = tt.Trainer(model, device="cuda", batch_size=32,
                     callbacks=[tt.ModelCheckpoint("mnist_flat.pt"), tt.StateSummaryLogger()])
trainer.fit((x_train, y_train), epochs=30, val_data=(x_test, y_test))
```

With 500 clauses per class the model reaches about 95.8 % test accuracy after 5 epochs
(1.5–5 s per epoch on an RTX 3090 depending on the batch size, see
[Benchmarks](../benchmarks.md)); the literature reports ~98 % for 2000 clauses per class
after hundreds of epochs.

## Convolutional Tsetlin machine

```python
enc = tt.data.AdaptiveThresholdEncoder(block_size=11, C=2)          # book/paper recipe
x_train, y_train = tt.data.load_mnist_boolean("./data", train=True, encoder=lambda x: enc(x * 255), device="cuda")
x_test, y_test = tt.data.load_mnist_boolean("./data", train=False, encoder=lambda x: enc(x * 255), device="cuda")

model = tt.ConvTsetlinMachine(10, n_clauses=2000, T=2500, s=10.0, patch_size=10, weighted=True)
trainer = tt.Trainer(model, device="cuda", batch_size=32)
trainer.fit((x_train, y_train), epochs=30, val_data=(x_test, y_test))
tt.viz.plot_conv_clauses(model, range(32))                          # what the clauses look for
```

Images are `(N, 1, 28, 28)` Boolean tensors; each 10 × 10 window plus its thermometer-encoded
position forms a 136-feature patch (272 literals), and a 28 × 28 image has 19 × 19 = 361 such
windows. Adaptive thresholding turns the flat background on and the strokes off (as in the
reference implementations), which starts slower than a plain `pixel > 0.3` threshold
(≈ 74 % vs 96 % after one epoch with 200 clauses/class) but is the recipe used for the best
published results. `examples/mnist_conv.py` reaches 97.9 % with its defaults (200 clauses per
class, 10 epochs, ≈ 80 s on an RTX 3090).

### Variations the script exposes

```bash
python examples/mnist_conv.py --stride 2                 # 100 windows instead of 361, ~1.5x faster
python examples/mnist_conv.py --no-position-encoding     # translation invariant clauses
python examples/mnist_conv.py --coalesced --clauses 1000 # one shared clause pool, class weights
```

`--coalesced` builds a `ConvCoalescedTsetlinMachine`: instead of `n_clauses` clauses *per
class*, all classes share one pool and each clause carries an integer weight per class, so
`--clauses` is the total budget. Keep `T` proportional to the pool a class effectively uses.
Note that coalesced clauses are weighted by construction — there is no `weighted` argument.

Sharing pays here: `--coalesced --clauses 1000 --T 500` reaches 98.5 % in 50 s, against 97.9 %
in 80 s for the 2000 per-class clauses of the default run (10 epochs each, same RTX 3090).

Position literals earn their keep on MNIST — 97.2 % with them against 95.9 % without, after
three epochs of the default configuration. That is the opposite of a task where the pattern
can appear anywhere; `examples/shapes_conv.py` shows that side.
