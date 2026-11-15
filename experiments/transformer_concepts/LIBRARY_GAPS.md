# Library gaps (rule C1)

`src/torchtsetlin/**` is read-only in this programme. Missing hooks or defects found while
working here are recorded below with a minimal reproduction, impact and a fix sketch.

## LG-1 — Triton kernels fail for models on a non-current CUDA device (2026-11-08)

**Reproduction** (torchtsetlin 0.7.0, two GPUs, current device 0):

```python
import torch, torchtsetlin as tt
x = torch.rand(32, 64, device="cuda:1") > 0.5
y = torch.randint(0, 2, (32,), device="cuda:1")
m = tt.TsetlinMachine(64, 2, 20, 10, 5.0).to("cuda:1")
for _ in range(5):
    m.update(x, y)
# UserWarning: Triton kernel fused_commit failed (ValueError: Pointer argument (at 0) cannot be
# accessed from Triton (cpu tensor?)); torchtsetlin uses the torch path.
# ... followed by cudaErrorStreamCaptureInvalidated warnings from the CUDA-graph capture.
```

**Impact.** Learning stays correct: the library falls back to the torch commit and to eager
updates, as designed. It is slower, though: 1.95 s/epoch instead of 1.13 s for the flat TM on
T1, L = 16, 20k examples (`results/probe` runs). The Triton launch happens on the current device
(0) while the tensors live on device 1.

**Workaround used here.** Every worker pins its GPU with `CUDA_VISIBLE_DEVICES=<k>` and uses
`device="cuda"`.

**Fix sketch.** Wrap the Triton launches in `_triton.py` (`fused_commit`, `conv_eval`, …) in
`with torch.cuda.device(tensor.device):`, and add a test that runs one update on `cuda:1`
when two GPUs are present.

## Feature notes (not defects, no action requested)

- **Counting aggregation over patches.** Convolutional clauses OR over patches. A
  `[#matching patches >= theta]` read (THEORY §B.4) would need a custom `_evaluate` in experiment
  code. It is out of scope for this round.
