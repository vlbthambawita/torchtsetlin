import time, torch, triton, triton.language as tl

def current(neg, inc, B, P):
    v = torch.matmul(neg, inc.t())
    return (v == 0).view(B, P, -1).any(dim=1)


@triton.jit
def conv_or_i8(neg_ptr, inc_ptr, out_ptr, P, L, C, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0); n0 = tl.program_id(1) * BN; rn = n0 + tl.arange(0, BN)
    hit = tl.zeros((BN,), dtype=tl.int32)
    for m0 in range(0, P, BM):
        rm = m0 + tl.arange(0, BM)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, L, BK):
            rk = k0 + tl.arange(0, BK)
            a = tl.load(neg_ptr + (b * P + rm[:, None]) * L + rk[None, :], mask=(rm[:, None] < P) & (rk[None, :] < L), other=1)
            w = tl.load(inc_ptr + rn[None, :] * L + rk[:, None], mask=(rn[None, :] < C) & (rk[:, None] < L), other=0)
            acc += tl.dot(a, w)
        hit = tl.maximum(hit, tl.max(((acc == 0) & (rm[:, None] < P)).to(tl.int32), axis=0))
    tl.store(out_ptr + b * C + rn, hit.to(tl.int8), mask=rn < C)

@triton.jit
def conv_or_bits(neg_ptr, inc_ptr, out_ptr, P, C, W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # neg: (B*P, W) int32 bit-packed False literals; inc: (C, W) int32 bit-packed includes
    b = tl.program_id(0); n0 = tl.program_id(1) * BN; rn = n0 + tl.arange(0, BN)
    hit = tl.zeros((BN,), dtype=tl.int32)
    for m0 in range(0, P, BM):
        rm = m0 + tl.arange(0, BM)
        viol = tl.zeros((BM, BN), dtype=tl.int32)
        for w in tl.static_range(W):
            a = tl.load(neg_ptr + (b * P + rm) * W + w, mask=rm < P, other=-1)
            c = tl.load(inc_ptr + rn * W + w, mask=rn < C, other=0)
            viol = viol | (a[:, None] & c[None, :])
        hit = tl.maximum(hit, tl.max(((viol == 0) & (rm[:, None] < P)).to(tl.int32), axis=0))
    tl.store(out_ptr + b * C + rn, hit.to(tl.int8), mask=rn < C)

def pack(bits):  # (R, L) 0/1 -> (R, ceil(L/32)) int32
    R, L = bits.shape; W = (L + 31) // 32
    pad = torch.zeros(R, W * 32, dtype=torch.int64, device=bits.device); pad[:, :L] = bits.long()
    sh = torch.arange(32, device=bits.device, dtype=torch.int64)
    v = (pad.view(R, W, 32) << sh).sum(-1)
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32).contiguous()

dev = torch.device("cuda"); B, P, L, C = 18, 361, 272, 20000
g = torch.Generator(device=dev).manual_seed(0)
lit = (torch.rand(B * P, L, device=dev, generator=g) < 0.5).half()
inc = (torch.rand(C, L, device=dev, generator=g) < 0.01).half()
inc[: C // 2] *= lit[torch.randint(0, B * P, (C // 2,), device=dev, generator=g)]
neg = (1 - lit).contiguous(); ref = current(neg, inc, B, P)
neg8, inc8 = neg.to(torch.int8).contiguous(), inc.to(torch.int8).contiguous()
negb, incb = pack(neg), pack(inc); W = negb.shape[1]
def i8(): o = torch.empty(B, C, dtype=torch.int8, device=dev); conv_or_i8[(B, triton.cdiv(C, 128))](neg8, inc8, o, P, L, C, BM=64, BN=128, BK=64, num_warps=8, num_stages=3); return o
def bits(): o = torch.empty(B, C, dtype=torch.int8, device=dev); conv_or_bits[(B, triton.cdiv(C, 128))](negb, incb, o, P, C, W=W, BM=64, BN=128, num_warps=8); return o
def cur(): return current(neg, inc, B, P)
for f in (cur, i8, bits):
    assert torch.equal(f().bool(), ref), f.__name__
    for _ in range(3): f()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20): f()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20
    print(f"{f.__name__:5s}: {dt*1e3:.3f} ms/chunk -> {dt*10000/B:.2f} s per 10k (exact)")
