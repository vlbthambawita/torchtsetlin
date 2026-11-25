"""Joint feedback for a team of convolutional Tsetlin machines (plan B4), without touching the library.

Feedback in ``TsetlinMachineBase._accumulate_literals`` depends on the other members only through
the vote sum, so a joint update needs two overrides: ``_votes`` adds the other members' votes
(an *offset*, written into a fixed buffer before the member's ``update``) and ``_select_feedback``
uses the joint threshold and, in shared mode, the negative class the team drew.

Rules (``c_ij`` = weight of member j in the score member i sees; ``T_i`` = its threshold):

    sum:  score = sum_j v_j,               threshold sum_j T_j   (c_ij = 1)
    one:  score = T_i * sum_j v_j / T_j,   threshold T_i         (c_ij = T_i / T_j)
    none: one member, plain library update (arms I and W)

CUDA graphs: a replay runs no Python and reads the buffers the capture read. Offset, negative
class and the feedback counter are therefore registered buffers of a fixed size, written in place;
a graphed update is always a single chunk, so the cursor is 0 there. ``_joint_T``,
``_joint_shared`` and ``_joint_on`` are constants of a captured graph: never change them on a
member that has already been updated.

Drop clause (cifar10_joint_teams, 2026-11-23): a joint update evaluates every member twice (pass A
for the votes, then the member's own ``update``), and both passes must see the same clause mask.
Members are therefore built with ``drop_granularity="epoch"`` (their ``update`` never resamples),
``JointTeam.update`` draws one mask per member and update, and ``learning_votes`` applies it. With
``drop_clause_p = 0`` no mask and no random number is drawn: the update is the one above.
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

import torchtsetlin as tt
from torchtsetlin import _triton
from torchtsetlin import functional as F


class _JointMixin:
    def joint_setup(self, max_batch: int) -> "_JointMixin":
        """Allocate the joint buffers (call before ``.to(device)``)."""
        K = self.n_classes
        self.register_buffer("_joint_offset", torch.zeros(max_batch, K), persistent=False)
        self.register_buffer("_joint_neg", torch.zeros(max_batch, dtype=torch.long), persistent=False)
        # Type I selections, Type II selections, examples since the last read
        self.register_buffer("_joint_fb", torch.zeros(3, dtype=torch.float64), persistent=False)
        self._joint_on = False
        self._joint_cur = 0
        self._joint_T = float(self.T)
        self._joint_shared = True
        self._joint_trace: Optional[list] = None      # tests only (graphs off)
        return self

    def _votes(self, clause_out: Tensor, clamp: bool = True) -> Tensor:
        if not self._joint_on:                        # forward(), plain updates
            return super()._votes(clause_out, clamp)
        v = super()._votes(clause_out, clamp=False)
        B, cur = v.shape[0], self._joint_cur
        self._joint_cur = cur + B
        v = v + self._joint_offset[cur:cur + B]
        return v.clamp_(-self._joint_T, self._joint_T)

    def _count(self, type_i: Tensor, type_ii: Tensor) -> None:
        B = float(type_i.shape[0])
        self._joint_fb.add_(torch.stack([type_i.sum(), type_ii.sum(),
                                         torch.full((), B, device=type_i.device, dtype=torch.long)]))

    def _select_feedback(self, votes: Tensor, y: Tensor, clause_out: Tensor) -> Tuple[Tensor, Tensor, Any]:
        if not self._joint_on:
            type_i, type_ii, aux = super()._select_feedback(votes, y, clause_out)
            self._count(type_i, type_ii)
            return type_i, type_ii, aux
        # TsetlinMachine._select_feedback with the joint threshold and the team's negative class.
        B, dev, T = votes.shape[0], votes.device, self._joint_T
        if self._joint_shared:
            cur = self._joint_cur - B                 # _votes advanced the cursor for this chunk
            y_neg = self._joint_neg[cur:cur + B]
        else:
            y_neg = F.sample_negative_classes(y, self.n_classes, votes=votes, focused=False, T=T)
        yc, nc = y.view(B, 1), y_neg.view(B, 1)
        p_tgt = F.feedback_probabilities(votes.gather(1, yc), T, target=True)
        p_neg = F.feedback_probabilities(votes.gather(1, nc), T, target=False)
        p_class = torch.zeros(B, self.n_classes, device=dev)
        p_class.scatter_(1, yc, p_tgt)
        p_class.scatter_(1, nc, p_neg)
        r = torch.rand(B, self.n_classes, self.n_clauses, device=dev)
        selected = r < p_class.unsqueeze(2)
        is_tgt = torch.zeros(B, self.n_classes, dtype=torch.bool, device=dev).scatter_(1, yc, True)
        is_neg = torch.zeros_like(is_tgt).scatter_(1, nc, True)
        positive = (self.clause_polarity.view(self.n_classes, self.n_clauses) > 0).unsqueeze(0)
        type_i = (selected & ((is_tgt.unsqueeze(2) & positive) | (is_neg.unsqueeze(2) & ~positive))).view(B, -1)
        type_ii = (selected & ((is_tgt.unsqueeze(2) & ~positive) | (is_neg.unsqueeze(2) & positive))).view(B, -1)
        self._count(type_i, type_ii)
        if self._joint_trace is not None:
            self._joint_trace.append((votes.clone(), y_neg.clone()))
        return type_i, type_ii, None

    def read_feedback(self) -> dict:
        """Feedback selections since the last call (and reset)."""
        a = self._joint_fb.tolist()
        self._joint_fb.zero_()
        return {"type_i": a[0], "type_ii": a[1], "examples": a[2]}


class JointConv(_JointMixin, tt.ConvTsetlinMachine):
    """A 2D convolutional member. Without ``_joint_on`` it is the library model plus a counter."""


@torch.no_grad()
def learning_votes(m, x) -> Tensor:
    """Pass A: raw, unclamped vote sums of ``m`` with learning semantics (an empty clause is
    True), RNG-free. On CUDA this is the fused kernel the update itself evaluates with. Clauses
    switched off by the member's current drop-clause mask do not vote, as in its update."""
    assert not m._joint_on and m.drop_literal_p == 0.0
    active = m._clause_active
    xb = m._prepare(x)
    B = xb.shape[0]
    P = m._patches_per_example(xb)
    fused = m._fused_eval_ok(P)
    step = m._fused_chunk_size(xb) if fused else m._chunk_size(xb)
    out = []
    for i in range(0, B, step):
        lit = m._encode(xb[i:i + step])
        res = _triton.conv_eval(m._neg_bits(lit), m.include_bits(), lit.shape[0], P, draw=False) if fused else None
        co = res[0] if res is not None else m._evaluate_dense(lit, True)[0]
        if active is not None:
            co = co & active.unsqueeze(0)
        out.append(m._votes(co, clamp=False))
    return torch.cat(out, dim=0)


class JointTeam:
    """Members that learn from one joint vote sum (or, with ``rule="none"``, one plain member)."""

    def __init__(self, names: Sequence[str], members: Sequence[_JointMixin], rule: str,
                 shared_negative: bool = True) -> None:
        assert rule in ("sum", "one", "none") and len(names) == len(members)
        assert rule != "none" or len(members) == 1
        self.names, self.members, self.rule = list(names), list(members), rule
        self.shared = bool(shared_negative)
        Ts = [float(m.T) for m in members]
        M = len(members)
        if rule == "one":
            self.coef = [[Ts[i] / Ts[j] for j in range(M)] for i in range(M)]
            self.T_joint = list(Ts)
            self.weights = [1.0 / t for t in Ts]      # prediction: argmax of sum_j v_j / T_j
        else:
            self.coef = [[1.0] * M for _ in range(M)]
            self.T_joint = [sum(Ts)] * M
            self.weights = [1.0] * M
        if rule != "none":
            for m, tj in zip(members, self.T_joint):
                m._joint_T = float(tj)
                m._joint_shared = self.shared

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    def resample_dropout(self) -> None:
        """One new clause mask per member, shared by pass A and the member's update."""
        for m in self.members:
            if m.drop_clause_p > 0.0:
                assert m.drop_granularity == "epoch", "joint members must not resample in update()"
                m.resample_dropout()

    def offsets(self, vs: List[Tensor]) -> List[Tensor]:
        """What each member adds to its own votes: sum over the others of c_ij * v_j."""
        out = []
        for i in range(len(vs)):
            off = torch.zeros_like(vs[i])
            for j, v in enumerate(vs):
                if j != i:
                    off = off + self.coef[i][j] * v
            out.append(off)
        return out

    def combine(self, sums: Sequence[Tensor]):
        """The team's score from the members' vote sums (tensors or arrays)."""
        return sum(w * v for w, v in zip(self.weights, sums))

    @torch.no_grad()
    def update(self, xs: Sequence[Tensor], y: Tensor) -> Tensor:
        """One joint update; returns the team's pre-update score (learning semantics)."""
        self.resample_dropout()
        if self.rule == "none":
            return self.members[0].update(xs[0], y)
        vs = [learning_votes(m, x) for m, x in zip(self.members, xs)]
        B = y.shape[0]
        neg = F.sample_negative_classes(y, self.members[0].n_classes) if self.shared else None
        for m, x, off in zip(self.members, xs, self.offsets(vs)):
            m._joint_offset[:B].copy_(off)
            if neg is not None:
                m._joint_neg[:B].copy_(neg)
            m._joint_on, m._joint_cur = True, 0
            try:
                m.update(x, y)
            finally:
                m._joint_on = False
        return self.combine(vs)

    @torch.no_grad()
    def member_sums(self, xs: Sequence[Tensor], batch: int = 1000) -> List[Tensor]:
        """Raw class sums of every member in prediction mode (empty clauses are False)."""
        self.eval()
        try:
            return [torch.cat([m(x[i:i + batch]) for i in range(0, x.shape[0], batch)])
                    for m, x in zip(self.members, xs)]
        finally:
            self.train()
