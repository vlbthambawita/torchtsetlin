# Saving and loading

Models are `nn.Module`s whose learnable state is stored in buffers, so the standard PyTorch
recipe applies:

```python
torch.save(model.state_dict(), "tm.pt")

model2 = tt.TsetlinMachine(784, 10, 2000, T=50, s=10)      # same hyper-parameters
model2.load_state_dict(torch.load("tm.pt", map_location="cpu"))
```

Lazily initialised models (`n_features=None`, or convolutional models without
`input_shape`) allocate their state while loading, so you can restore a checkpoint without
having seen data:

```python
model3 = tt.ConvTsetlinMachine(10, 200, T=100, patch_size=10)
model3.load_state_dict(torch.load("ctm.pt"))
model3.input_shape = (1, 28, 28)          # set explicitly if you did not pass input_shape
```

`ModelCheckpoint` stores `{"state_dict", "epoch", "logs"}` at the best validation score.

Encoders are modules too — save them next to the model so the Booleanization thresholds
travel with it:

```python
torch.save({"model": model.state_dict(), "encoder": enc.state_dict()}, "bundle.pt")
```

Whole-model pickling (`torch.save(model)`) also works but ties the file to the package
version; prefer `state_dict()`.
