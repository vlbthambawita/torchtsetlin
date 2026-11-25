"""Members and teams (PLAN_KVASIR §7).

A *member* is one model with its own encoding and convolution. A *team* adds the members'
scores, each divided by its own ``T``. Teams of one member are the ordinary heads.

Member kinds (``spec["conv"]``):
    flat     two-class ``TsetlinMachine`` on rows (RPN objectness, mask, dense)
    ml       multi-label ``CoalescedTsetlinMachine`` on rows (RPN deltas)
    conv2d   ``ConvTsetlinMachine`` on crops (RoI classifier)
    conv3d   ``Conv3dTsetlinMachine`` on crops, the encoding's levels as depth
    convml   multi-label ``ConvCoalescedTsetlinMachine`` on crops (box refinement)
    grad     a small gradient-trained network on the same inputs (baseline iv)

``combine="joint"`` trains the members on one shared vote sum (rule ``sum`` of
``joint_wide.py``, copied from ``experiments/joint_wide_tm/code/joint.py``); a composite is not
trained here at all: it loads independently trained members (``Team.load_members``).

A score is a *margin*: positive means "class 1" (two-class members: ``(v1 - v0) / T``) or
"output on" (multi-label members: ``v / T``). The sums are never clipped (decision D-3).
"""

from typing import List, Optional

import torch
import torch.nn as nn
from joint_wide import JointConv, JointTeam, _JointMixin

import torchtsetlin as tt

TM_KW = dict(n_states=100, boost_true_positive=True, clamp_prediction_votes=False)
EVAL_BATCH = {"flat": 8192, "ml": 4096, "conv2d": 512, "conv3d": 256, "convml": 256, "grad": 2048}
JOINT_MAX_BATCH = 256


class JointFlat(_JointMixin, tt.TsetlinMachine):
    """A flat two-class member that can learn from a joint vote sum (test T7 pins that without
    the joint switch it is the library model)."""


@torch.no_grad()
def flat_learning_votes(m, x):
    """Raw, unclamped vote sums with learning semantics (an empty clause is True), RNG-free."""
    assert not m._joint_on and m.drop_literal_p == 0.0 and m._clause_active is None
    xb = m._prepare(x)
    out = []
    step = max(1, m._chunk_size(xb))
    for i in range(0, xb.shape[0], step):
        co, _ = m._evaluate(m._encode(xb[i:i + step]), True)
        out.append(m._votes(co, clamp=False))
    return torch.cat(out, 0)


class FlatJointTeam(JointTeam):
    """``JointTeam`` for flat members: only pass A differs (no patches)."""

    @torch.no_grad()
    def update(self, xs, y):
        from torchtsetlin import functional as F
        vs = [flat_learning_votes(m, x) for m, x in zip(self.members, xs)]
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


class GradMember(nn.Module):
    """Baseline (iv): a small network trained by gradient descent on exactly a member's input.
    Rows get an MLP, crops a three-layer CNN. Same ``update`` / ``forward`` contract as a TM."""

    T = 1.0

    def __init__(self, n_out, multi_label, width=128, lr=1e-3):
        super().__init__()
        self.n_out, self.multi_label, self.width, self.lr = n_out, multi_label, width, lr
        self.net: Optional[nn.Module] = None
        self.opt = None

    def _build(self, x):
        if x.dim() == 2:
            net = nn.Sequential(nn.Linear(x.shape[1], 2 * self.width), nn.ReLU(), nn.Linear(2 * self.width, self.width),
                                nn.ReLU(), nn.Linear(self.width, self.n_out))
        else:
            c, w = x.shape[1], self.width // 2
            net = nn.Sequential(nn.Conv2d(c, w, 3, padding=1), nn.ReLU(), nn.Conv2d(w, 2 * w, 3, stride=2, padding=1),
                                nn.ReLU(), nn.Conv2d(2 * w, 2 * w, 3, stride=2, padding=1), nn.ReLU(),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2 * w, self.n_out))
        self.net = net.to(x.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def _in(self, x):
        x = x.float()
        return x.flatten(1, 2) if x.dim() == 5 else x

    def update(self, x, y):
        x = self._in(x)
        if self.net is None:
            self._build(x)
        self.net.train()
        with torch.enable_grad():
            out = self.net(x)
            loss = (nn.functional.binary_cross_entropy_with_logits(out, y.float()) if self.multi_label
                    else nn.functional.cross_entropy(out, y.long()))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return out.detach()

    def forward(self, x):
        x = self._in(x)
        if self.net is None:
            self._build(x)
        self.net.eval()
        with torch.no_grad():
            return self.net(x)

    def state_dict(self, *a, **k):                   # the optimiser state resumes with the weights
        sd = super().state_dict(*a, **k)
        if self.opt is not None:
            sd["__opt__"] = self.opt.state_dict()
        return sd

    def load_state_dict(self, sd, strict=True):
        sd = dict(sd)
        opt = sd.pop("__opt__", None)
        if self.net is None:
            first = sd["net.0.weight"]
            dummy = torch.zeros((1, first.shape[1]) if first.dim() == 2 else (1, first.shape[1], 8, 8), device=first.device)
            self._build(dummy)
        out = super().load_state_dict(sd, strict)
        if opt is not None:
            self.opt.load_state_dict(opt)
        return out


def build_member(spec, n_out, joint=False):
    """One untrained member. ``n_out`` is the number of outputs of multi-label members."""
    kind, c, T, s = spec["conv"], int(spec["clauses"]), float(spec["T"]), float(spec.get("s", 10.0))
    kw = dict(TM_KW)
    if spec.get("budget"):
        kw["max_included_literals"] = int(spec["budget"])
    if kind == "flat":
        cls = JointFlat if joint else tt.TsetlinMachine
        return cls(None, 2, c, T, s, weighted=True, **kw)
    if kind == "ml":
        return tt.CoalescedTsetlinMachine(None, n_out, c, T, s, multi_label=True, **kw)
    if kind == "conv2d":
        cls = JointConv if joint else tt.ConvTsetlinMachine
        return cls(2, c, T, s, patch_size=int(spec["patch"]), position_encoding=True, weighted=True, **kw)
    if kind == "conv3d":
        p = int(spec["patch"])
        return tt.Conv3dTsetlinMachine(2, c, T, s, patch_size=(int(spec["kd"]), p, p), position_encoding=True,
                                       weighted=True, **kw)
    if kind == "convml":
        return tt.ConvCoalescedTsetlinMachine(n_out, c, T, s, patch_size=int(spec["patch"]), position_encoding=True,
                                              multi_label=True, **kw)
    if kind == "grad":
        return GradMember(n_out if n_out else 2, multi_label=bool(n_out), width=int(spec.get("width", 128)))
    raise ValueError(f"unknown member kind {kind!r}")


class Team:
    """Members plus the rule that adds their scores."""

    def __init__(self, specs: List[dict], n_out=0, combine="single", device="cpu"):
        assert combine in ("single", "joint", "composite")
        assert combine != "single" or len(specs) == 1
        self.specs, self.n_out, self.combine, self.device = specs, n_out, combine, torch.device(device)
        joint = combine == "joint"
        if joint:
            kinds = {s["conv"] for s in specs}
            assert kinds <= {"flat"} or kinds <= {"conv2d"}, "joint teams: all flat or all conv2d members"
        self.members = [build_member(s, n_out, joint) for s in specs]
        if joint:
            for m in self.members:
                m.joint_setup(JOINT_MAX_BATCH)
        self.members = [m.to(self.device) for m in self.members]
        self._joint = None
        if joint:
            cls = FlatJointTeam if specs[0]["conv"] == "flat" else JointTeam
            self._joint = cls([f"m{i}" for i in range(len(specs))], self.members, "sum")
        self.multi_label = bool(n_out)

    def train(self):
        for m in self.members:
            m.train()

    def eval(self):
        for m in self.members:
            m.eval()

    @torch.no_grad()
    def update(self, xs, y):
        """One update of every member on its own view ``xs[i]`` of the same examples."""
        if self._joint is not None:
            B = y.shape[0]
            for i in range(0, B, JOINT_MAX_BATCH):
                self._joint.update([x[i:i + JOINT_MAX_BATCH] for x in xs], y[i:i + JOINT_MAX_BATCH])
            return None
        first = None
        for m, x in zip(self.members, xs):
            v = m.update(x, y)
            first = v if first is None else first
        return first                                 # pre-update votes of member 0 (running metric)

    def _weights(self):
        if self._joint is not None:                  # rule "sum": one threshold, the sum of the T
            tj = sum(float(m.T) for m in self.members)
            return [1.0 / tj] * len(self.members)
        return [1.0 / float(m.T) for m in self.members]

    @torch.no_grad()
    def member_margin(self, i, x):
        """Margin of member ``i`` in prediction mode (refuses training mode: rule C5)."""
        m = self.members[i]
        if m.training and not isinstance(m, GradMember):
            raise RuntimeError("scoring a Tsetlin machine in training mode (empty clauses would fire); call eval()")
        step = EVAL_BATCH[self.specs[i]["conv"]]
        out = []
        for j in range(0, x.shape[0], step):
            v = m(x[j:j + step]).float()
            out.append(v if self.multi_label else v[:, 1] - v[:, 0])
        if not out:
            return torch.zeros((0, self.n_out) if self.multi_label else (0,), device=self.device)
        return torch.cat(out, 0) * self._weights()[i]

    def margin(self, xs):
        return sum(self.member_margin(i, x) for i, x in enumerate(xs))

    # ------------------------------------------------------------------ state
    def state(self):
        return [m.state_dict() for m in self.members]

    def load_state(self, states):
        for m, sd in zip(self.members, states):
            m.load_state_dict(sd)

    def stats(self):
        """Included literals per clause (mean over members) for the Tsetlin members."""
        vals = [float(m.include_count.float().mean()) for m in self.members if hasattr(m, "include_count")
                and getattr(m, "include_count", None) is not None and m.include_count.numel()]
        out = {"literals_per_clause": sum(vals) / len(vals)} if vals else {}
        if self._joint is not None:
            for i, m in enumerate(self.members):
                fb = m.read_feedback()
                if fb["examples"]:
                    out[f"feedback_per_example_m{i}"] = (fb["type_i"] + fb["type_ii"]) / fb["examples"]
        return out
