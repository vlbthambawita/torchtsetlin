"""Prototype: flat batched commit without fp32 count tensors (torch + torch.compile only).
Compares the 0.5.1 path (3 fp32 addmm + commit_dense + zero) with
  nt = mm(sel_i^T, lit) and n2 = mm(sel_ii^T, 1 - lit) in fp16 (exact: counts <= B < 2048),
  n_false derived in-kernel as n_ia[c] - nt, no zeroing pass."""
import sys
import time

import torch
from torchtsetlin import functional as F

dev = torch.device("cuda")
C, L, B = (int(v) for v in sys.argv[1:4])
N, s = 128, 10.0
g = torch.Generator(device=dev).manual_seed(0)
state = torch.randint(N - 3, N + 3, (C, L), device=dev, dtype=torch.uint8, generator=g)
include = (state >= N).half()
inc_count = (state >= N).sum(1)
lit = (torch.rand(B, L, device=dev, generator=g) < 0.5).float()
fire = torch.rand(B, C, device=dev, generator=g) < 0.3
sel = torch.rand(B, C, device=dev, generator=g) < 0.4
sel_i = (fire & sel).float()
sel_nofire = (~fire & sel).float()
sel_ii = (fire & (torch.rand(B, C, device=dev, generator=g) < 0.1)).float()


def old_path():
    nt = torch.zeros(C, L, device=dev)
    nf = torch.zeros(C, L, device=dev)
    n2 = torch.zeros(C, L, device=dev)
    nib = sel_nofire.sum(0)
    st = sel_i.t()
    nt.addmm_(st, lit)
    nf.addmm_(st, 1 - lit)
    n2.addmm_(sel_ii.t(), 1 - lit)
    return F.commit_dense(state, include, inc_count, nt, nf, nib, n2, n_states=N, s=s)


def _kernel(state, include, nt, n2, nia, nib, thr, N: int, width: int):
    ntf = nt.to(torch.float32)
    nf = nia.unsqueeze(1) - ntf + nib.unsqueeze(1)
    dec = F._bisect_binomial(nf, thr, False, width)
    old = state.to(torch.float32)
    inc2 = torch.minimum(n2.to(torch.float32), (N - old).clamp(min=0))
    new = (old + ntf - dec + inc2).clamp(0, 2 * N - 1)
    included = new >= N
    state.copy_(new.to(state.dtype))
    include.copy_(included.to(include.dtype))
    return included.sum(dim=1)


kern = torch.compile(_kernel, dynamic=True)
thr = F._wide_thresholds(1.0 / s, dev)


def new_path():
    lh = lit.half()
    nt = torch.mm(sel_i.t().half(), lh)          # (C, L) fp16, written once, never zeroed
    n2 = torch.mm(sel_ii.t().half(), 1 - lh)
    nia = sel_i.sum(0)
    nib = sel_nofire.sum(0)
    return kern(state, include, nt, n2, nia, nib, thr, N, 64)


for f in (old_path, new_path):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(30):
        f()
    torch.cuda.synchronize()
    print(f"C={C} L={L} B={B} {f.__name__}: {(time.perf_counter() - t) / 30 * 1e3:.3f} ms")
traffic = C * L * (1 + 1 + 2) / 1e9  # state r/w + include fp16 write: the floor of any dense commit
print(f"   floor (state r/w + include write, {traffic:.3f} GB) at 760 GB/s: {traffic / 760 * 1e3:.3f} ms")

from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(10):
        new_path()
    torch.cuda.synchronize()
for e in p.key_averages():
    if e.device_type.name == "CUDA" and e.device_time_total > 0 and "Call" not in e.key:
        print(f"   {e.device_time_total / 10 / 1e3:7.3f} ms  {e.key[:70]}")
