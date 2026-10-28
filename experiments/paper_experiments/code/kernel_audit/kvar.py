import time, torch
from torchtsetlin import functional as F
dev = torch.device("cuda"); C, L, N = 20000, 1568, 128
g = torch.Generator(device=dev).manual_seed(0)
state = torch.randint(N - 3, N + 3, (C, L), device=dev, dtype=torch.uint8, generator=g)
include = torch.zeros(C, L, device=dev, dtype=torch.half)
nt = torch.randint(0, 4, (C, L), device=dev, generator=g).half(); n2 = torch.randint(0, 2, (C, L), device=dev, generator=g).half()
nia = torch.full((C,), 6.0, device=dev); nib = torch.full((C,), 3.0, device=dev)
thr = F._wide_thresholds(0.1, dev)
def body(state, include, nt, n2, nia, nib, thr, mode: int, width: int):
    ntf = nt.to(torch.float32)
    nf = nia.unsqueeze(1) - ntf + nib.unsqueeze(1)
    if mode == 2:
        dec = nf * (torch.rand_like(nf) < 0.1)          # RNG only, no table
    elif mode == 3:
        dec = nf * 0.1                                  # no RNG, no table
    else:
        dec = F._bisect_binomial(nf, thr, False, width)
    old = state.to(torch.float32)
    new = (old + ntf - dec + torch.minimum(n2.to(torch.float32), (N - old).clamp(min=0))).clamp(0, 2 * N - 1)
    included = new >= N
    state.copy_(new.to(state.dtype)); include.copy_(included.to(include.dtype))
    if mode == 1:
        return None
    return included.sum(dim=1)
k = torch.compile(body, dynamic=True)
for mode, width, name in ((0, 64, "full (bisect 64 + row sum)"), (0, 16, "bisect 16"), (1, 64, "no row-sum"),
                          (2, 64, "rand only, no table"), (3, 64, "no rand, no table")):
    for _ in range(3): k(state, include, nt, n2, nia, nib, thr, mode, width)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): k(state, include, nt, n2, nia, nib, thr, mode, width)
    torch.cuda.synchronize(); print(f"{name:30s} {(time.perf_counter() - t) / 20 * 1e3:.3f} ms")
