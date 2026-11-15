"""From-scratch Tsetlin machine with delta-tolerant clauses (THEORY_HTM H1; PLAN_HTM E-H1).

A clause fires iff at most ``delta`` of its included literals are false (``delta = 0``: the
ordinary clause). Everything else follows torchtsetlin's ``TsetlinMachine`` /
``Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)`` (read, not modified):

- class-wise clauses, first ``ceil(n/2)`` of each class positive, the rest negative;
- votes ``sum polarity * weight * out``, clamped to [-T, T]; one uniform random negative class
  per example; clause selection with ``(T - v_y)/2T`` and ``(T + v_neg)/2T``, independently per
  clause;
- Type Ia: true literals memorise with ``p_mem`` (1 with ``boost_true_positive``), false
  literals forget with ``1/s``; Type Ib: every literal forgets with ``1/s``; Type II: every
  false literal moves one step towards inclusion, never past the include boundary;
- conv columns: Type Ia and Type II each use one uniformly random matching column, drawn
  independently per (example, clause, event);
- batch feedback: all events of a mini-batch are summed and applied once; states clamped to
  [0, 2N-1]; init N-1 (all excluded); weights +1 per Type Ia, -1 per Type II, floor 1.
- empty clauses fire while learning, not when predicting.

With tolerance, a firing clause can have up to ``delta`` included false literals; Type Ia
pushes them out with ``1/s`` (THEORY_HTM H1.4).
"""
from __future__ import annotations

import math
from typing import Optional

import torch


class TolerantTM:
    def __init__(self, n_classes: int, n_clauses: int, T: float, s: float, delta: int = 0, *,
                 weighted: bool = True, n_states: int = 128, boost_true_positive: bool = True,
                 device="cuda", seed: Optional[int] = None):
        self.K, self.n, self.T, self.s, self.delta = n_classes, n_clauses, float(T), float(s), int(delta)
        self.weighted, self.N, self.boost = weighted, n_states, boost_true_positive
        self.dev = torch.device(device)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed if seed is not None else 0)
        n_pos = (n_clauses + 1) // 2
        pol = torch.where(torch.arange(n_clauses) < n_pos, 1.0, -1.0)
        self.pol = pol.repeat(n_classes).to(self.dev)  # (C,)
        self.C = n_classes * n_clauses
        self.state = None
        self.w = torch.ones(self.C, device=self.dev)
        self.training = True
        self.fast_binomial = False

    # ------------------------------------------------------------------ evaluation
    def _init(self, F2):
        self.state = torch.full((self.C, F2), self.N - 1, dtype=torch.int16, device=self.dev)

    @property
    def include(self):
        return (self.state >= self.N).to(torch.float16 if self.dev.type == "cuda" else torch.float32)

    def _lits(self, x):
        """x: (B, F) flat or (B, F, P) columns -> literals (B, P, 2F) float."""
        if x.dim() == 2:
            x = x.unsqueeze(2)
        x = x.transpose(1, 2).bool()
        dt = torch.float16 if self.dev.type == "cuda" else torch.float32
        return torch.cat([x, ~x], 2).to(dt)

    def _fire(self, lit, inc, learning):
        """(B, P, C) firing per column."""
        cnt = inc.sum(1)  # (C,)
        true_inc = torch.matmul(lit, inc.t())  # (B, P, C) included literals that are true
        miss = cnt.view(1, 1, -1) - true_inc
        fire = miss <= self.delta + 0.5
        if not learning:
            fire &= (cnt > 0).view(1, 1, -1)
        return fire

    def votes(self, out):
        v = out.float() * (self.pol * self.w).view(1, -1)
        return v.view(out.shape[0], self.K, self.n).sum(2)

    @torch.no_grad()
    def predict(self, x, bs=256):
        res = []
        for k in range(0, x.shape[0], bs):
            lit = self._lits(x[k:k + bs].to(self.dev))
            if self.state is None:
                self._init(lit.shape[2])
            out = self._fire(lit, self.include, learning=False).any(1)
            res.append(self.votes(out).argmax(1))
        return torch.cat(res)

    # ------------------------------------------------------------------ learning
    def _draw(self, fire, sel):
        """One uniformly random matching column per (example, clause): (B, C) indices."""
        r = torch.rand(fire.shape, device=self.dev, generator=self.gen)
        r = torch.where(fire, r, torch.full_like(r, -1.0))
        return r.argmax(1)

    def _drawn_counts(self, lit, fire, sel):
        """sum_b sel[b, c] * lit[b, col(b, c)] -> (C, 2F) and sum_b sel[b, c] -> (C,)."""
        B, P, F2 = lit.shape
        if P == 1:
            s = sel.to(lit.dtype)
            return torch.matmul(s.t(), lit[:, 0]).float(), s.sum(0).float()
        col = self._draw(fire, sel)  # (B, C)
        out = torch.zeros(self.C, F2, device=self.dev)
        for b in range(B):  # gather drawn columns example by example: (C, 2F) each
            m = sel[b]
            if m.any():
                rows = lit[b][col[b]]  # (C, 2F)
                out += rows.float() * m.float().unsqueeze(1)
        return out, sel.float().sum(0)

    def _binom(self, n, p):
        if p >= 1.0:
            return n
        n = n.clamp(min=0)
        if self.fast_binomial:  # the library's compiled inverse-CDF sampler (read-only use), global RNG
            from torchtsetlin.functional import _binomial
            return _binomial(n.contiguous(), p, None)
        return torch.binomial(n, torch.full_like(n, p), generator=self.gen)

    @torch.no_grad()
    def update(self, x, y):
        lit = self._lits(x.to(self.dev))
        y = y.to(self.dev)
        if self.state is None:
            self._init(lit.shape[2])
        B = lit.shape[0]
        inc = self.include
        fire = self._fire(lit, inc, learning=True)  # (B, P, C)
        out = fire.any(1)  # (B, C)
        v = self.votes(out).clamp(-self.T, self.T)
        off = torch.randint(1, self.K, (B,), device=self.dev, generator=self.gen)
        yn = (y + off) % self.K
        p_t = (self.T - v.gather(1, y.view(-1, 1))) / (2 * self.T)
        p_n = (self.T + v.gather(1, yn.view(-1, 1))) / (2 * self.T)
        p_cls = torch.zeros(B, self.K, device=self.dev)
        p_cls.scatter_(1, y.view(-1, 1), p_t)
        p_cls.scatter_(1, yn.view(-1, 1), p_n)
        sel = torch.rand(B, self.K, self.n, device=self.dev, generator=self.gen) < p_cls.unsqueeze(2)
        is_t = torch.zeros(B, self.K, dtype=torch.bool, device=self.dev).scatter_(1, y.view(-1, 1), True)
        is_n = torch.zeros_like(is_t).scatter_(1, yn.view(-1, 1), True)
        pos = (self.pol.view(self.K, self.n) > 0).unsqueeze(0)
        t1 = (sel & ((is_t.unsqueeze(2) & pos) | (is_n.unsqueeze(2) & ~pos))).view(B, -1)
        t2 = (sel & ((is_t.unsqueeze(2) & ~pos) | (is_n.unsqueeze(2) & pos))).view(B, -1)
        ia, ib, ii = t1 & out, t1 & ~out, t2 & out
        n_true, n_ia = self._drawn_counts(lit, fire, ia)
        n_false = n_ia.unsqueeze(1) - n_true
        n2_true, n_ii = self._drawn_counts(lit, fire, ii)
        n2 = n_ii.unsqueeze(1) - n2_true  # Type II events with the literal false
        n_ib = ib.float().sum(0)
        p_mem = 1.0 if self.boost else (self.s - 1) / self.s
        inc_ = self._binom(n_true, p_mem)
        dec = self._binom(n_false + n_ib.unsqueeze(1), 1.0 / self.s)
        room = (self.N - self.state.float()).clamp(min=0)
        inc2 = torch.minimum(n2, room)
        new = (self.state.float() + inc_ - dec + inc2).clamp(0, 2 * self.N - 1)
        self.state = new.to(torch.int16)
        if self.weighted:
            self.w = (self.w + ia.float().sum(0) - ii.float().sum(0)).clamp(min=1)
        return v

    def n_included(self):
        return float((self.state >= self.N).sum(1).float().mean())

    def clause_lengths(self):
        return (self.state >= self.N).sum(1)


def check_formula():  # tiny self-test of H1.3's recall formula against the tolerant firing rule
    k, eta, trials = 12, 0.1, 200_000
    flips = torch.rand(trials, k) < eta
    for d in range(4):
        emp = ((flips.sum(1)) <= d).float().mean().item()
        th = sum(math.comb(k, i) * eta ** i * (1 - eta) ** (k - i) for i in range(d + 1))
        print(d, round(emp, 4), round(th, 4))


if __name__ == "__main__":
    check_formula()
