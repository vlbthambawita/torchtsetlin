# Training loops and the Trainer

## The primitive: `model.update(x, y)`

Every model exposes one learning call. It evaluates the batch, decides feedback per clause,
updates the automata in place and returns the vote sums computed *before* the update:

```python
votes = model.update(x, y)            # (B, n_outputs)
batch_acc = (votes.argmax(1) == y).float().mean()
```

Anything you can do with a PyTorch training loop you can do here: custom schedules,
curriculum, mixing data sources, logging to TensorBoard or Weights & Biases, etc. Remember to
call `model.train()` before learning and `model.eval()` before measuring accuracy (empty
clause semantics differ).

## The `Trainer`

```python
trainer = tt.Trainer(
    model,
    device="cuda",                    # moves the model
    batch_size=32,
    callbacks=[
        tt.EarlyStopping(monitor="val_accuracy", patience=10, restore_best=True),
        tt.ModelCheckpoint("runs/best.pt", monitor="val_accuracy"),
        tt.CSVLogger("runs/log.csv"),
        tt.StateSummaryLogger(),      # mean literals / empty clauses per epoch
        tt.HyperparameterSchedule("s", lambda epoch: max(3.0, 10.0 - 0.2 * epoch)),
    ],
    metrics={"margin": lambda votes, y: tt.metrics.vote_margin(votes).mean().item()},
)
history = trainer.fit(train_data, epochs=50, val_data=val_data)
```

`train_data` / `val_data` can be:

* an `(x, y)` pair of tensors or arrays — kept on the device and indexed directly (fastest),
* a `torch.utils.data.Dataset` — wrapped in a `DataLoader`,
* a `DataLoader` — used as is (e.g. with on-the-fly Booleanization via
  `tt.data.TransformDataset`).

Per epoch the trainer logs `epoch_time`, `batch_accuracy` (or `batch_mae`), validation
metrics prefixed with `val_`, and anything callbacks add. `history` is a
[`History`](../api/train.md) with `history["val_accuracy"]`, `history.best(...)`,
`history.to_dataframe()` and `history.plot()`.

`trainer.evaluate(data)`, `trainer.test(data)` and `trainer.predict(data)` complete the
Keras-style trio; `tt.evaluate(model, data)` and `tt.predict(model, data)` are one-liners.

## Writing a callback

```python
class PrintRules(tt.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print(self.model.rules()[:3])
```

Hooks: `on_train_begin/end`, `on_epoch_begin/end`, `on_batch_begin/end`. Set
`self.trainer.stop_training = True` to stop early.

## Dropout granularity

`drop_clause_p` / `drop_literal_p` masks are resampled on every `update()` by default
(`drop_granularity="batch"`). The drop-clause paper resamples once per epoch; use
`drop_granularity="epoch"` and the `Trainer` will call `model.resample_dropout()` at the
start of each epoch (do the same in a manual loop).

## Reproducibility

`tt.seed_everything(seed)` seeds Python, NumPy and all PyTorch devices. Batched updates on
CUDA use `index_add_` for convolutional models, which is order-nondeterministic but exact
for the integer counts involved; results are reproducible up to sampling.
