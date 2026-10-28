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

## Compiled kernels

Large updates run part of their work through `torch.compile`: on the CPU the forgetting
(Type Ib) step and the binomial draws of the commit, and in sequential mode on CUDA the whole
single-example step of the flat models. Together with the row-sparse commit, this makes CPU
training of an MNIST-sized model (20 000 clauses) about 14x faster than in 0.4.0, and GPU
sequential mode about 11x faster. The kernels draw from the same distributions as the eager
code.

* The first large update of a session compiles the kernels (a few seconds; the result is cached
  on disk). Small models stay on the eager code, where compiling would not pay off.
* Compiled kernels draw their random numbers from the default generator of the device, so
  `tt.seed_everything` still makes runs reproducible, but a compiled and an eager run with the
  same seed differ. Calls that pass an explicit `generator=` to the functional API stay eager.
* Where `torch.compile` does not work (no C++ compiler, unsupported Python), the library warns
  once and uses the eager code. Turn the kernels off with `TORCHTSETLIN_COMPILE=0` or
  `torchtsetlin.functional.set_compile(False)`.

## The CUDA fast path (Triton kernels and CUDA graphs)

On CUDA, with Triton available (it ships with the Linux CUDA builds of PyTorch), a batched
update runs through two hand-written kernels and is replayed as a CUDA graph:

* **One commit kernel** applies all of an update's feedback in a single pass over the automata:
  it computes the Type Ia / Type II counts from the update's events (on tensor cores for flat
  models), samples them, and writes the state, the include cache and the per-clause include
  counts. No count tensor is materialised.
* **Convolutional models** evaluate their clauses on bit-packed literals (32 per word) and OR
  over the patches in registers, so the `(batch, patches, clauses)` match tensor never exists.
  While learning, the same kernel picks each clause's random matching patch. Prediction of a
  10x10-patch MNIST model is 4.7x faster than in 0.5.1.
* **No host synchronisation**: labels that are already on the GPU are checked with a
  device-side assertion (an invalid label is reported as a CUDA error at the next
  synchronisation, as in PyTorch's own losses); labels from the host are checked there.
* **CUDA graphs**: an update shape seen for the third time is captured and replayed from then on
  (sequential mode in blocks of 64 examples). Replays give bit-identical results to eager
  execution and remove the kernel-launch overhead that dominates small models and sequential
  mode (Noisy XOR in sequential mode: 17x faster than 0.5.1). Models with clause or literal
  dropout are not graphed.

Turn the pieces off with `TORCHTSETLIN_TRITON=0` (Triton kernels), `TORCHTSETLIN_CUDA_GRAPHS=0`
or `model.cuda_graphs = False` (graphs), or `TORCHTSETLIN_COMPILE=0` (everything compiled). Every
fast path has the previous code as its fall-back, which is also what runs on the CPU, on MPS, and
when Triton is missing.

## Multiple GPUs

Different models can live on different GPUs; a single model uses one device. Because Tsetlin
machine training is already massively parallel over clauses and literals, one modern GPU is
usually enough for tens of thousands of clauses.

## Memory

The state uses 1 byte per automaton by default (`state_dtype=None` picks `uint8` for the
default `n_states=128`), i.e. `n_clauses_total * 2 * n_features` bytes, plus a copy of the
include mask in the evaluation dtype (2 bytes per automaton on CUDA, 4 on the CPU) and, on CUDA,
a bit-packed one (1 bit). The CUDA fast path needs no count buffers; the other paths keep three
float count buffers (12 bytes per automaton) between updates, which `model.eval()` releases. A
2000-clause-per-class MNIST classifier (20 000 clauses × 1568 literals) needs about 100 MB on
CUDA. Intermediate tensors of an update are chunked automatically to stay within
`max_chunk_elements` (default 2^27 elements); lower it if you hit out-of-memory errors on a
small GPU.
