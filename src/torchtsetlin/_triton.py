"""Hand-written Triton kernels for the CUDA hot paths (optional).

Two kernels replace the GPU update's memory-bound passes:

* :func:`fused_commit` applies the feedback of a whole batched update in one pass over the
  automata. It does not read count tensors: each ``(clause, literal)`` tile computes its Type Ia
  and Type II counts from the update's events (a tensor-core product for flat models, a gather
  of each clause's random matching patch for convolutional ones), samples the binomials, and
  writes the state, the float16 include mask, the bit-packed include mask and the per-clause
  include counts.
* :func:`conv_eval` evaluates a convolutional machine on bit-packed literals (32 literals per
  ``int32`` word, ``AND`` instead of a multiply-add) and ORs over patches in registers, so the
  ``(B, P, C)`` match tensor never exists. While learning it also draws, per (example, clause),
  a uniformly random matching patch for Type Ia and an independent one for Type II.

Everything here is exact (evaluation) or equal in distribution (sampling) to the eager code in
:mod:`torchtsetlin.functional`, which stays the fall-back: without Triton, on other devices,
under ``TORCHTSETLIN_COMPILE=0``, or when a kernel cannot handle a shape, the callers take the
previous path. The random numbers come from a seed drawn with the CUDA default generator on the
device, so ``torch.manual_seed`` reproduces a run and CUDA-graph replays draw fresh numbers.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

try:  # pragma: no cover - depends on the installation
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAVE_TRITON = False

_FAILED: set = set()
#: rows of the binomial CDF table (counts up to TABLE_N - 1); shared with functional's table
TABLE_N = 256


def available(device: torch.device) -> bool:
    """Whether the Triton kernels can run on ``device``."""
    from . import functional as F

    return (_HAVE_TRITON and device.type == "cuda" and bool(F._COMPILE["enabled"])
            and os.environ.get("TORCHTSETLIN_TRITON", "1") != "0")


def _failed(name: str, err: Exception) -> None:
    _FAILED.add(name)
    warnings.warn(f"Triton kernel {name} failed ({type(err).__name__}: {err}); torchtsetlin uses the "
                  "torch path. Set TORCHTSETLIN_TRITON=0 to skip the attempt.", RuntimeWarning,
                  stacklevel=3)


def device_seed(device: torch.device) -> Tensor:
    """A fresh 31-bit seed on the device, drawn with the CUDA default generator (no host sync)."""
    return torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.int32)


def n_words(n_literals: int) -> int:
    return (int(n_literals) + 31) // 32


def pack_bits(x: Tensor) -> Tensor:
    """``(R, L)`` 0/1 tensor -> ``(R, ceil(L/32))`` int32, literal ``l`` in bit ``l % 32`` of word
    ``l // 32`` (unused high bits are zero)."""
    R, L = x.shape
    W = n_words(L)
    b = torch.zeros(R, W * 32, dtype=torch.int32, device=x.device)
    b[:, :L] = x.to(torch.int32)
    shifts = torch.arange(32, dtype=torch.int32, device=x.device)
    return (b.view(R, W, 32) << shifts).sum(dim=2, dtype=torch.int32)


if _HAVE_TRITON:

    @triton.jit
    def _uniform(r):
        # uint32 -> float32 in [0, 1) from the top 24 bits
        return (r >> 8).to(tl.float32) * (1.0 / 16777216.0)

    @triton.jit
    def _binomial(n, u, q, log2_1mq, thr_ptr, TABLE: tl.constexpr, LOG2W: tl.constexpr):
        # Binomial(n, q) by inversion: #{k : u >= cdf[n, k]}. The first HEAD steps walk the CDF
        # arithmetically (pmf(0) = (1-q)^n, pmf(k+1) = pmf(k) (n-k)/(k+1) q/(1-q)), so the usual
        # small counts need no memory access; only draws past HEAD bisect the float32 table
        # (width 2**LOG2W > n; entries k >= n are 2.0, so the search stops at n). Steps that no
        # element of the tile needs are skipped for the whole tile.
        HEAD: tl.constexpr = 8
        n_max = tl.max(tl.max(n, axis=1), axis=0)
        lo = tl.zeros_like(n)
        if n_max > 0:
            nf = n.to(tl.float32)
            pmf = tl.exp2(nf * log2_1mq)
            cdf = pmf
            ratio = q / (1.0 - q)
            for k in tl.static_range(HEAD):
                if n_max > k:
                    lo += ((u >= cdf) & (k < n)).to(lo.dtype)
                    pmf = pmf * (nf - k) * (ratio / (k + 1))
                    cdf = cdf + pmf
            tail = lo == HEAD
            if tl.max(tl.max(tail.to(tl.int32), axis=1), axis=0) > 0:
                base = n * TABLE
                for i in tl.static_range(LOG2W):
                    cand = lo + (1 << (LOG2W - 1 - i))
                    thr = tl.load(thr_ptr + base + cand - 1, mask=tail & (cand <= n), other=2.0)
                    lo = tl.where(tail & (u >= thr), cand, lo)
        return lo

    @triton.jit
    def _uniforms(seed, offs, BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr):
        # one Philox call per 4 automata (it returns four 32-bit words): two independent
        # uniforms per automaton from a (BLOCK_C, BLOCK_L // 2) call
        r0, r1, r2, r3 = tl.randint4x(seed, offs)
        ua = tl.reshape(tl.join(_uniform(r0), _uniform(r1)), (BLOCK_C, BLOCK_L))
        ub = tl.reshape(tl.join(_uniform(r2), _uniform(r3)), (BLOCK_C, BLOCK_L))
        return ua, ub

    @triton.jit
    def _commit_kernel(
        state_ptr, include_ptr, bits_ptr, count_new_ptr, count_old_ptr,
        lit_ptr, sel_i_ptr, sel_ii_ptr, p_i_ptr, p_ii_ptr, n_ia_ptr, n_ib_ptr, active_ptr,
        thr_mem_ptr, thr_forget_ptr, seed_ptr,
        B, P, C, L, W, N, budget, q_mem, log2_1mq_mem, q_forget, log2_1mq_forget,
        BLOCK_C: tl.constexpr, BLOCK_L: tl.constexpr, KB: tl.constexpr, CONV: tl.constexpr,
        MEM_SAMPLE: tl.constexpr, MEM_FLIP: tl.constexpr, FORGET_SAMPLE: tl.constexpr,
        FORGET_FLIP: tl.constexpr, HAS_ACTIVE: tl.constexpr, HAS_BUDGET: tl.constexpr,
        LOG2W: tl.constexpr, TABLE: tl.constexpr,
    ):
        pc = tl.program_id(0)
        pl = tl.program_id(1)
        rc = pc * BLOCK_C + tl.arange(0, BLOCK_C)
        rl = pl * BLOCK_L + tl.arange(0, BLOCK_L)
        cm = rc < C
        lm = rl < L
        nt = tl.zeros((BLOCK_C, BLOCK_L), dtype=tl.float32)
        n2 = tl.zeros((BLOCK_C, BLOCK_L), dtype=tl.float32)
        if CONV:
            # every clause is updated from its own random matching patch of each example
            for b in range(0, B):
                s1 = tl.load(sel_i_ptr + b * C + rc, mask=cm, other=0.0).to(tl.float32)
                s2 = tl.load(sel_ii_ptr + b * C + rc, mask=cm, other=0.0).to(tl.float32)
                q1 = tl.load(p_i_ptr + b * C + rc, mask=cm, other=0)
                q2 = tl.load(p_ii_ptr + b * C + rc, mask=cm, other=0)
                r1 = (b * P + q1).to(tl.int64) * L
                r2 = (b * P + q2).to(tl.int64) * L
                g1 = tl.load(lit_ptr + r1[:, None] + rl[None, :],
                             mask=(s1[:, None] > 0) & lm[None, :], other=0.0).to(tl.float32)
                g2 = tl.load(lit_ptr + r2[:, None] + rl[None, :],
                             mask=(s2[:, None] > 0) & lm[None, :], other=1.0).to(tl.float32)
                nt += s1[:, None] * g1
                n2 += s2[:, None] * (1.0 - g2)
        else:
            # flat: counts = sel^T @ literals on tensor cores, K = the batch
            for b0 in range(0, B, KB):
                rb = b0 + tl.arange(0, KB)
                bm = rb < B
                a1 = tl.load(sel_i_ptr + rb[:, None] * C + rc[None, :],
                             mask=bm[:, None] & cm[None, :], other=0.0)
                a2 = tl.load(sel_ii_ptr + rb[:, None] * C + rc[None, :],
                             mask=bm[:, None] & cm[None, :], other=0.0)
                lt = tl.load(lit_ptr + rb[:, None] * L + rl[None, :],
                             mask=bm[:, None] & lm[None, :], other=0.0)
                nt += tl.dot(tl.trans(a1), lt)
                n2 += tl.dot(tl.trans(a2), (1.0 - lt.to(tl.float32)).to(tl.float16))
        n_ia = tl.load(n_ia_ptr + rc, mask=cm, other=0.0)
        n_ib = tl.load(n_ib_ptr + rc, mask=cm, other=0.0)
        nf = n_ia[:, None] - nt + n_ib[:, None]
        if HAS_BUDGET:  # an oversized matching clause gets Type Ib instead of Type Ia
            over = tl.load(count_old_ptr + rc, mask=cm, other=0) > budget
            nf = tl.where(over[:, None], n_ia[:, None] + n_ib[:, None], nf)
            nt = tl.where(over[:, None], 0.0, nt)
        nt_i = nt.to(tl.int32)
        nf_i = nf.to(tl.int32)
        offs = rc[:, None] * L + rl[None, :]
        seed = tl.load(seed_ptr)
        # Philox counters unique per (row, tile, pair of automata), whatever L is
        half = ((rc[:, None] * tl.num_programs(1) + pl) * (BLOCK_L // 2)
                + tl.arange(0, BLOCK_L // 2)[None, :])
        u_mem, u_forget = _uniforms(seed, half, BLOCK_C, BLOCK_L)
        if MEM_SAMPLE:
            k = _binomial(nt_i, u_mem, q_mem, log2_1mq_mem, thr_mem_ptr, TABLE, LOG2W)
            inc = nt_i - k if MEM_FLIP else k
        else:
            inc = nt_i
        if FORGET_SAMPLE:
            k = _binomial(nf_i, u_forget, q_forget, log2_1mq_forget, thr_forget_ptr, TABLE, LOG2W)
            dec = nf_i - k if FORGET_FLIP else k
        else:
            dec = nf_i
        mask = cm[:, None] & lm[None, :]
        old = tl.load(state_ptr + offs, mask=mask, other=0).to(tl.int32)
        inc2 = tl.minimum(n2.to(tl.int32), tl.maximum(N - old, 0))
        delta = inc - dec + inc2
        if HAS_ACTIVE:
            act = tl.load(active_ptr + rl, mask=lm, other=0.0)
            delta = tl.where(act[None, :] > 0, delta, 0)
        new = tl.minimum(tl.maximum(old + delta, 0), 2 * N - 1)
        tl.store(state_ptr + offs, new.to(state_ptr.dtype.element_ty), mask=mask)
        included = (new >= N) & mask
        tl.store(include_ptr + offs, included.to(include_ptr.dtype.element_ty), mask=mask)
        tl.atomic_add(count_new_ptr + rc, tl.sum(included.to(tl.int32), axis=1), mask=cm)
        # bit-packed include mask: BLOCK_L // 32 words per row of the tile
        words = tl.reshape(included.to(tl.int32), (BLOCK_C, BLOCK_L // 32, 32))
        shifts = tl.arange(0, 32)
        packed = tl.sum(words << shifts[None, None, :], axis=2)
        rw = pl * (BLOCK_L // 32) + tl.arange(0, BLOCK_L // 32)
        tl.store(bits_ptr + rc[:, None] * W + rw[None, :], packed,
                 mask=cm[:, None] & (rw[None, :] < W))

    @triton.jit
    def _hash32(x):
        # lowbias32 (Wellons): a well-mixed 32-bit bijection, far cheaper than a Philox call
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        x = x * 0x846CA68B
        x = x ^ (x >> 16)
        return x

    @triton.jit
    def _conv_eval_kernel(
        neg_ptr, inc_ptr, out_ptr, p_i_ptr, p_ii_ptr, seed_ptr, P, C,
        W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DRAW: tl.constexpr,
    ):
        b = tl.program_id(0)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        nm = rn < C
        hit = tl.zeros((BN,), dtype=tl.int32)
        # best (key << 32 | patch) per clause; a uniformly random matching patch is the one
        # with the largest key, and the max over packed values yields its index directly
        best1 = tl.zeros((BN,), dtype=tl.int64)
        best2 = tl.zeros((BN,), dtype=tl.int64)
        if DRAW:
            seed = _hash32(tl.load(seed_ptr).to(tl.uint32) + b.to(tl.uint32) * 0x9E3779B9)
        for m0 in range(0, P, BM):
            rm = m0 + tl.arange(0, BM)
            mm = rm < P
            viol = tl.zeros((BM, BN), dtype=tl.int32)
            for w in tl.static_range(W):
                a = tl.load(neg_ptr + (b * P + rm) * W + w, mask=mm, other=-1)
                c = tl.load(inc_ptr + rn * W + w, mask=nm, other=0)
                viol = viol | (a[:, None] & c[None, :])
            match = (viol == 0) & mm[:, None]
            hit = tl.maximum(hit, tl.max(match.to(tl.int32), axis=0))
            if DRAW:
                # keys hash (seed, patch, clause): independent ones for Type Ia and Type II;
                # 31 bits, | 1 keeps a matching patch above the 0 of non-matching ones (and the
                # packed value positive)
                x = (rm[:, None] * C + rn[None, :]).to(tl.uint32) * 2 + seed
                k1 = ((_hash32(x) >> 1) | 1).to(tl.int64)
                k2 = ((_hash32(x + 1) >> 1) | 1).to(tl.int64)
                pidx = rm[:, None].to(tl.int64)
                v1 = tl.where(match, (k1 << 32) | pidx, 0)
                v2 = tl.where(match, (k2 << 32) | pidx, 0)
                best1 = tl.maximum(best1, tl.max(v1, axis=0))
                best2 = tl.maximum(best2, tl.max(v2, axis=0))
        tl.store(out_ptr + b * C + rn, hit.to(tl.int8), mask=nm)
        if DRAW:
            tl.store(p_i_ptr + b * C + rn, (best1 & 0xFFFFFFFF).to(tl.int32), mask=nm)
            tl.store(p_ii_ptr + b * C + rn, (best2 & 0xFFFFFFFF).to(tl.int32), mask=nm)


def conv_eval(neg_bits: Tensor, inc_bits: Tensor, B: int, P: int,
              draw: bool) -> Optional[Tuple[Tensor, Optional[Tensor], Optional[Tensor]]]:
    """Clause outputs of a convolutional machine from bit-packed *False* literals
    ``(B*P, W)`` and include masks ``(C, W)``: ``(B, C)`` bool, the OR over patches of "no
    included literal is False" (empty clauses match). With ``draw``, also ``(B, C)`` int32
    indices of a uniformly random matching patch for Type Ia and an independent one for Type II
    (arbitrary where nothing matches). ``None`` if the kernel cannot run."""
    if "conv_eval" in _FAILED:
        return None
    C, W = inc_bits.shape
    dev = inc_bits.device
    out = torch.empty(B, C, dtype=torch.int8, device=dev)
    p_i = torch.empty(B, C, dtype=torch.int32, device=dev) if draw else out
    p_ii = torch.empty(B, C, dtype=torch.int32, device=dev) if draw else out
    seed = device_seed(dev) if draw else out
    BN = 256  # tuned on an RTX 3090 (B7-sized model): 8 x 256 tiles, 4 warps
    try:
        _conv_eval_kernel[(B, triton.cdiv(C, BN))](
            neg_bits, inc_bits, out, p_i, p_ii, seed, P, C, W=W, BM=8, BN=BN, DRAW=bool(draw),
            num_warps=4)
    except Exception as err:  # pragma: no cover - toolchain dependent
        _failed("conv_eval", err)
        return None
    return out.bool(), (p_i if draw else None), (p_ii if draw else None)


_TABLES: Dict[Tuple[float, torch.device], Tensor] = {}


def _table(p: float, device: torch.device) -> Tensor:
    key = (float(p), device)
    t = _TABLES.get(key)
    if t is None:
        from . import functional as F

        t = _TABLES[key] = F._wide_thresholds(p, device)
    return t


def fused_commit(
    state: Tensor, include: Tensor, include_bits: Tensor, include_count: Tensor,
    literals: Tensor, sel_i: Tensor, sel_ii: Tensor, p_i: Optional[Tensor], p_ii: Optional[Tensor],
    n_ia: Tensor, n_ib: Tensor, *, n_states: int, s: float, boost_true_positive: bool,
    max_included_literals: Optional[int], literal_active: Optional[Tensor], max_events: int,
) -> bool:
    """Apply one update's feedback from its events (see the module docstring). ``literals`` is
    ``(B, 2F)`` (flat) or ``(B, P, 2F)`` with ``p_i`` / ``p_ii`` the patch of each (example,
    clause) (convolutional); ``sel_i`` / ``sel_ii`` the ``(B, C)`` 0/1 Type Ia / Type II masks;
    ``n_ia`` / ``n_ib`` the ``(C,)`` Type Ia / Type Ib event counts; ``max_events`` a bound on any
    count (the examples in the update). Updates ``state``, ``include``, ``include_bits`` and
    ``include_count`` in place (soft budget). Returns ``False`` (nothing modified) if it cannot
    run."""
    if "fused_commit" in _FAILED or max_events >= TABLE_N or state.numel() >= 2**31:
        return False
    if state.dtype not in (torch.uint8, torch.int16, torch.int32):
        return False
    C, L = state.shape
    W = include_bits.shape[1]
    dev = state.device
    conv = literals.dim() == 3
    B = literals.shape[0]
    P = literals.shape[1] if conv else 1
    p_mem = 1.0 if boost_true_positive else (float(s) - 1.0) / float(s)
    p_forget = 1.0 / float(s)
    if p_mem <= 0.0:
        return False
    mem_sample = p_mem < 1.0
    forget_sample = p_forget < 1.0
    q_mem, q_forget = min(p_mem, 1 - p_mem), min(p_forget, 1 - p_forget)
    thr_mem = _table(q_mem, dev) if mem_sample else n_ia
    thr_forget = _table(q_forget, dev) if forget_sample else n_ia
    log2w = max(4, int(max_events).bit_length())  # bisection width 2**log2w > max_events
    lit = literals.to(torch.float16).contiguous()
    s1 = sel_i.to(torch.float16).contiguous()
    s2 = sel_ii.to(torch.float16).contiguous()
    count_new = torch.zeros(C, dtype=torch.int32, device=dev)
    active = literal_active.reshape(-1).to(torch.float32).contiguous() if literal_active is not None else n_ia
    BLOCK_C, BLOCK_L = 16, 64  # small tiles: ~100 registers per thread, enough warps per SM
    grid = (triton.cdiv(C, BLOCK_C), triton.cdiv(L, BLOCK_L))
    try:
        _commit_kernel[grid](
            state, include, include_bits, count_new, include_count,
            lit, s1, s2, p_i if conv else s1, p_ii if conv else s1,
            n_ia.to(torch.float32).contiguous(), n_ib.to(torch.float32).contiguous(), active,
            thr_mem, thr_forget, device_seed(dev),
            B, P, C, L, W, int(n_states),
            -1 if max_included_literals is None else int(max_included_literals),
            q_mem, math.log2(1.0 - q_mem) if q_mem < 1.0 else 0.0,
            q_forget, math.log2(1.0 - q_forget) if q_forget < 1.0 else 0.0,
            BLOCK_C=BLOCK_C, BLOCK_L=BLOCK_L, KB=32, CONV=conv,
            MEM_SAMPLE=mem_sample, MEM_FLIP=p_mem > 0.5, FORGET_SAMPLE=forget_sample,
            FORGET_FLIP=p_forget > 0.5, HAS_ACTIVE=literal_active is not None,
            HAS_BUDGET=max_included_literals is not None, LOG2W=log2w, TABLE=TABLE_N,
            num_warps=4,
        )
    except Exception as err:  # pragma: no cover - toolchain dependent
        _failed("fused_commit", err)
        return False
    include_count.copy_(count_new)
    return True
