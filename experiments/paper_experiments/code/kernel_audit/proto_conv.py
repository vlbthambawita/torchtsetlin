"""Prototype: fused convolutional clause evaluation (Triton), prediction semantics.
out[b, c] = OR_p ( sum_k neg[b, p, k] * inc[c, k] == 0 ), never materialising (B, P, C).
Compared with the 0.5.1 path: fp16 GEMM over (B*P, L) x (L, C), == 0, any over P."""
import time

import torch
import triton
import triton.language as tl


@triton.jit
def conv_or_kernel(neg_ptr, inc_ptr, out_ptr, P, L, C,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    n0 = tl.program_id(1) * BN
    rn = n0 + tl.arange(0, BN)
    hit = tl.zeros((BN,), dtype=tl.int32)
    for m0 in range(0, P, BM):
        rm = m0 + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, L, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(neg_ptr + (b * P + rm[:, None]) * L + rk[None, :],
                        mask=(rm[:, None] < P) & (rk[None, :] < L), other=1.0)
            w = tl.load(inc_ptr + rn[None, :] * L + rk[:, None],
                        mask=(rn[None, :] < C) & (rk[:, None] < L), other=0.0)
            acc += tl.dot(a, w)
        match = (acc == 0.0) & (rm[:, None] < P)
        hit = tl.maximum(hit, tl.max(match.to(tl.int32), axis=0))
    tl.store(out_ptr + b * C + rn, hit.to(tl.int8), mask=rn < C)


def fused(neg, inc, B, P):
    L = neg.shape[-1]
    C = inc.shape[0]
    out = torch.empty(B, C, dtype=torch.int8, device=neg.device)
    grid = (B, triton.cdiv(C, 128))
    conv_or_kernel[grid](neg, inc, out, P, L, C, BM=64, BN=128, BK=32, num_warps=8, num_stages=3)
    return out


def current(neg, inc, B, P):
    v = torch.matmul(neg, inc.t())  # (B*P, C) fp16
    return (v == 0).view(B, P, -1).any(dim=1)


dev = torch.device("cuda")
B, P, L, C = 18, 361, 272, 20000            # one prediction chunk of the B7 model (chunk = 18)
g = torch.Generator(device=dev).manual_seed(0)
lit = (torch.rand(B * P, L, device=dev, generator=g) < 0.5).half()
inc = (torch.rand(C, L, device=dev, generator=g) < 0.01).half()
inc[: C // 2] *= lit[torch.randint(0, B * P, (C // 2,), device=dev, generator=g)]  # matching clauses
neg = (1 - lit).contiguous()
ref = current(neg, inc, B, P)
got = fused(neg, inc, B, P).bool()
assert torch.equal(ref, got), (ref.float().mean(), got.float().mean())
print("exact:", True, " match rate", float(ref.float().mean()))
for f in (current, fused):
    for _ in range(3):
        f(neg, inc, B, P)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        f(neg, inc, B, P)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 20
    print(f"{f.__name__:8s}: {dt * 1e3:.3f} ms per chunk of {B} images -> {dt * 10000 / B:.2f} s per 10k")
