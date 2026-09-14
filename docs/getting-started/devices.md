# Choosing a device

`torchtsetlin` follows the PyTorch device model exactly:

```python
model = tt.TsetlinMachine(784, 10, 2000, T=50, s=10).to("cuda")   # or .cuda(), .to("cuda:1")
model.device            # device of the automata state
model.to("cpu")         # move back
```

* All learnable state (`ta_state`, clause `weights`) and caches are registered buffers, so
  `.to()`, `.cuda()`, `.cpu()` and `.to(dtype)`-style calls behave like any `nn.Module`.
* Inputs are moved to the model's device automatically by `update()` / `forward()` (they are
  tiny Boolean tensors). For best throughput keep the whole Boolean dataset on the GPU and
  index it there — the `Trainer` does this when you pass `(x, y)` tensors.
* Lazily initialised models (`n_features=None`) allocate their state on the device the module
  was moved to before the first batch.
* Randomness uses the global PyTorch RNG of the device: `tt.seed_everything(0)` (or
  `torch.manual_seed`) makes runs reproducible per device.

## Multiple GPUs

Different models can live on different GPUs; a single model uses one device. Because Tsetlin
machine training is already massively parallel over clauses and literals, one modern GPU is
usually enough for tens of thousands of clauses.

## Memory

The state uses 4 bytes per automaton by default (`state_dtype=torch.int32`), i.e.
`4 * n_clauses_total * 2 * n_features` bytes plus a float copy of the include mask. A
2000-clause-per-class MNIST classifier (20 000 clauses × 1568 literals) needs about
250 MB. Intermediate tensors of an update are chunked automatically to stay within
`max_chunk_elements` (default 2^27 elements); lower it if you hit out-of-memory errors on a
small GPU.
