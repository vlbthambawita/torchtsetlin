# Noisy XOR

The benchmark of the original paper: 12 Boolean inputs, `y = x0 XOR x1`, 40 % label noise
in the training set. Full script: `examples/noisy_xor.py`.

```python
import torch, torchtsetlin as tt

tt.seed_everything(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
x_train, y_train = tt.data.make_noisy_xor(5000, noise=0.4, seed=0)
x_test, y_test = tt.data.make_noisy_xor(5000, noise=0.0, seed=1)

model = tt.TsetlinMachine(12, 2, n_clauses=20, T=15, s=3.9, n_states=50)
trainer = tt.Trainer(model, device=device, batch_size=10,
                     callbacks=[tt.EarlyStopping(patience=30)])
trainer.fit((x_train, y_train), epochs=200, val_data=(x_test, y_test))
print(trainer.test((x_test, y_test)))
for rule in model.rules():
    print(rule)
```

Expected output (the two XOR patterns per class, plus their negations):

```
IF x0 AND NOT x1 THEN 1
IF x1 AND NOT x0 THEN 1
IF x0 AND x1 THEN NOT 1
IF NOT x0 AND NOT x1 THEN NOT 1
...
```

Try `feedback_mode="sequential"` to compare with the classical one-example-at-a-time
algorithm, or `max_included_literals=2` to force minimal rules.
