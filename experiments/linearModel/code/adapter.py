"""Spec §6 adapter over ``torchtsetlin`` source models and the frozen gate sets.

A *gate set* maps the float view ``X`` (``(N, d)``, float64) to Boolean gates ``G`` of shape
``(N, K*M)`` in bank order: column ``k*M + j`` is clause ``j`` of output ``k`` — the same order
as the library's ``clause_class = arange(C) // n_clauses``. Three kinds:

- :class:`FrozenRules` — spec §6.3 snapshot of a flat ``TsetlinMachine``: exported literals,
  evaluated by our own code (literal-export deployment mode, spec §15.2).
- :class:`LibraryGates` — a ``ConvTsetlinMachine``'s clause outputs via ``evaluate_clauses``
  (library-evaluator deployment mode, D6): the OR over patches cannot be exported as one
  conjunction (spec §6.2).
- :class:`ConstantGate` — one always-true gate per output (A2: a global linear model).

Every gate set carries a ``fingerprint`` over everything that determines ``G``.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

import torchtsetlin as tt

from .data import REPO


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def src_diff_stat() -> str:
    try:
        return subprocess.check_output(["git", "diff", "--stat", "src/"], cwd=REPO, text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _h(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode() + str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


# --------------------------------------------------------------------------- source TM contract
def snapshot_identity(model: tt.TsetlinMachine) -> str:
    """SNAPSHOT_IDENTITY: hash of the automata, weights and config that define the clauses."""
    w = getattr(model, "weights", None)
    conv = {}
    if hasattr(model, "patch_size"):
        conv = dict(patch=model.patch_size, stride=model.stride, pos=model.position_encoding,
                    shape=tuple(getattr(model, "input_shape", ()) or ()))
    return _h(type(model).__name__, model.ta_state, w if isinstance(w, torch.Tensor) else None,
              model.n_states, model.n_classes, model.n_clauses, model.n_features, conv, tt.__version__)


def clause_polarity(model) -> np.ndarray:
    M = model.n_clauses
    within = np.arange(model.n_classes * M) % M
    return np.where(within < (M + 1) // 2, 1, -1).astype(np.int8)


def clause_weight(model) -> Optional[np.ndarray]:
    if getattr(model, "weighted", False):
        return model.weights.detach().cpu().numpy().astype(np.int64)
    return None


def raw_clause_outputs(model, B: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    """EVALUATE_RAW_CLAUSES in prediction mode (empty clause = False) -> ``(N, C)`` bool."""
    was = model.training
    model.eval()
    try:
        outs = [model.evaluate_clauses(B[i:i + batch]) for i in range(0, B.shape[0], batch)]
    finally:
        model.train(was)
    return torch.cat(outs, 0).bool()


# --------------------------------------------------------------------------- frozen rules
@dataclass(frozen=True)
class FrozenRule:
    source_output_id: int
    source_clause_id: int                         # global clause index k*M + j
    source_polarity: int                          # metadata only (spec §4.2)
    source_clause_weight: Optional[int]           # metadata only
    included_literals: Tuple[Tuple[int, int], ...]  # (boolean feature u, +1 POSITIVE / -1 NEGATIVE)
    empty_clause_output: int
    semantic_fingerprint: str


def evaluate_frozen_clause(rule: FrozenRule, b: np.ndarray) -> int:
    """Spec §6.3 EVALUATE_FROZEN_CLAUSE, literally (used by the tests as the reference)."""
    if len(rule.included_literals) == 0:
        return rule.empty_clause_output
    for u, pol in rule.included_literals:
        v = bool(b[u])
        if pol < 0:
            v = not v
        if not v:
            return 0
    return 1


@dataclass
class FrozenRules:
    """Immutable class-specific banks of frozen conjunctions over ``x > threshold``."""
    K: int
    M: int
    n_features: int
    threshold: float
    include: np.ndarray                 # (K*M, 2F) bool; [:, :F] = x_u, [:, F:] = NOT x_u
    empty_clause_output: np.ndarray     # (K*M,) uint8
    polarity: np.ndarray                # (K*M,) int8 metadata
    weight: Optional[np.ndarray]        # (K*M,) int metadata
    source_identity: str
    kind: str = "frozen_rules"
    fingerprint: str = field(default="", init=False)

    def __post_init__(self):
        self.include = np.array(self.include, dtype=bool, copy=True)
        self.include.setflags(write=False)
        self.empty_clause_output = np.array(self.empty_clause_output, dtype=np.uint8, copy=True)
        self.empty_clause_output.setflags(write=False)
        self.fingerprint = _h("FrozenRules", self.K, self.M, self.n_features, float(self.threshold),
                              "x>t", self.include, self.empty_clause_output)

    @property
    def n_gates(self) -> int:
        return self.K * self.M

    def rules(self) -> List[FrozenRule]:
        F = self.n_features
        out = []
        for c in range(self.n_gates):
            pos = np.flatnonzero(self.include[c, :F])
            neg = np.flatnonzero(self.include[c, F:])
            lits = tuple(sorted([(int(u), 1) for u in pos] + [(int(u), -1) for u in neg]))
            out.append(FrozenRule(
                source_output_id=c // self.M, source_clause_id=c,
                source_polarity=int(self.polarity[c]),
                source_clause_weight=None if self.weight is None else int(self.weight[c]),
                included_literals=lits, empty_clause_output=int(self.empty_clause_output[c]),
                semantic_fingerprint=_h(lits, int(self.empty_clause_output[c]), float(self.threshold), "x>t")))
        return out

    def encode(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32) > self.threshold   # == tt.data.Binarizer(t)

    def evaluate_bool(self, B: torch.Tensor, device: str = "cpu", batch: int = 8192) -> np.ndarray:
        """EVALUATE_FROZEN_BANKS on Boolean features -> ``(N, K*M)`` bool.

        A clause fires iff none of its included literals is false; the count of false included
        literals is an exact small-integer float32 matmul."""
        inc = torch.tensor(self.include, device=device, dtype=torch.float32)
        empty = torch.as_tensor(self.include.sum(1) == 0, device=device)
        eo = torch.as_tensor(self.empty_clause_output.astype(bool), device=device)
        out = []
        for i in range(0, B.shape[0], batch):
            b = B[i:i + batch].to(device).bool()
            lit_false = torch.cat([~b, b], dim=1).float()              # literal is False
            fires = (lit_false @ inc.T) == 0
            fires = torch.where(empty, eo, fires)
            out.append(fires.cpu())
        return torch.cat(out, 0).numpy()

    def __call__(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        return self.evaluate_bool(self.encode(X), device=device)

    def state(self) -> dict:
        return dict(kind=self.kind, K=self.K, M=self.M, n_features=self.n_features,
                    threshold=float(self.threshold), include=np.packbits(self.include, axis=1),
                    include_cols=self.include.shape[1], empty_clause_output=self.empty_clause_output,
                    polarity=self.polarity, weight=self.weight, source_identity=self.source_identity,
                    fingerprint=self.fingerprint)

    @classmethod
    def from_state(cls, st: dict) -> FrozenRules:
        inc = np.unpackbits(st["include"], axis=1, count=int(st["include_cols"])).astype(bool)
        obj = cls(K=int(st["K"]), M=int(st["M"]), n_features=int(st["n_features"]),
                  threshold=float(st["threshold"]), include=inc,
                  empty_clause_output=st["empty_clause_output"], polarity=st["polarity"],
                  weight=st["weight"], source_identity=str(st["source_identity"]))
        if obj.fingerprint != st["fingerprint"]:
            raise RuntimeError("frozen-rule fingerprint mismatch on load")
        return obj


def freeze_rules(model: tt.TsetlinMachine, threshold: float, probe_X: np.ndarray,
                 device: str = "cpu") -> FrozenRules:
    """Spec §6.3 FREEZE_RULES for a flat ``TsetlinMachine`` with probe-equivalence check."""
    if hasattr(model, "patch_size"):
        raise TypeError("convolutional clauses are not conjunctions over x; use LibraryGates")
    ident = snapshot_identity(model)
    inc = model.included_mask().detach().cpu().numpy().astype(bool)
    assert inc.shape == (model.n_classes * model.n_clauses, 2 * model.n_features)
    rules = FrozenRules(K=model.n_classes, M=model.n_clauses, n_features=model.n_features,
                        threshold=float(threshold), include=inc,
                        empty_clause_output=np.zeros(inc.shape[0], np.uint8),   # prediction mode
                        polarity=clause_polarity(model), weight=clause_weight(model),
                        source_identity=ident)
    B = rules.encode(probe_X)
    expected = raw_clause_outputs(model, B.to(model.device)).cpu().numpy()
    observed = rules.evaluate_bool(B, device=device)
    if not np.array_equal(expected, observed):
        bad = np.argwhere(expected != observed)
        raise AssertionError(f"frozen rules disagree with the source TM at {len(bad)} entries, e.g. {bad[:5]}")
    if snapshot_identity(model) != ident:
        raise AssertionError("source TM changed while freezing")
    return rules


def threshold_probes(X: np.ndarray, threshold: float, n: int = 64, seed: int = 0) -> np.ndarray:
    """Training examples with pixels set just below / at / just above the threshold
    (float32 neighbours, since the encoder compares in float32)."""
    rng = np.random.default_rng(seed)
    t = np.float32(threshold)
    vals = [np.nextafter(t, np.float32(-1)), t, np.nextafter(t, np.float32(2))]
    rows = X[rng.choice(len(X), size=n, replace=len(X) < n)].copy()
    out = []
    for v in vals:
        r = rows.copy()
        mask = rng.random(r.shape) < 0.3
        r[mask] = float(v)
        out.append(r)
    return np.concatenate(out, 0)


# --------------------------------------------------------------------------- other gate sets
class LibraryGates:
    """Gates = the source model's own clause outputs (prediction semantics) — used for
    ``ConvTsetlinMachine`` (D6). Holds a private frozen copy of the source model."""
    kind = "library"

    def __init__(self, model, threshold: float, input_shape=(1, 28, 28)):
        import copy
        self.model = copy.deepcopy(model).eval()
        for b in self.model.buffers():
            b.requires_grad_(False)
        self.K, self.M = model.n_classes, model.n_clauses
        self.threshold = float(threshold)
        self.input_shape = tuple(input_shape)
        self.source_identity = snapshot_identity(self.model)
        self.polarity = clause_polarity(model)
        self.weight = clause_weight(model)
        self.fingerprint = _h("LibraryGates", self.source_identity, self.threshold, "x>t", self.input_shape)

    @property
    def n_gates(self) -> int:
        return self.K * self.M

    def __call__(self, X: np.ndarray, device: Optional[str] = None, batch: int = 2048) -> np.ndarray:
        if snapshot_identity(self.model) != self.source_identity:
            raise RuntimeError("library gate model changed")
        B = torch.as_tensor(X, dtype=torch.float32).reshape(-1, *self.input_shape) > self.threshold
        return raw_clause_outputs(self.model, B.to(self.model.device), batch=batch).cpu().numpy()

    def state(self) -> dict:
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in self.model.state_dict().items()}, buf)
        cfg = dict(n_classes=self.K, n_clauses=self.M, T=self.model.T, s=self.model.s,
                   patch_size=self.model.patch_size, stride=self.model.stride,
                   position_encoding=self.model.position_encoding, weighted=self.model.weighted)
        return dict(kind=self.kind, cfg=cfg, threshold=self.threshold, input_shape=self.input_shape,
                    model_state=np.frombuffer(buf.getvalue(), np.uint8),
                    source_identity=self.source_identity, fingerprint=self.fingerprint)

    @classmethod
    def from_state(cls, st: dict, device: str = "cpu") -> LibraryGates:
        cfg = dict(st["cfg"])
        m = tt.ConvTsetlinMachine(input_shape=tuple(st["input_shape"]), **cfg)
        sd = torch.load(io.BytesIO(bytes(st["model_state"])), map_location=device)
        m.load_state_dict(sd)
        m.to(device)
        obj = cls(m, st["threshold"], st["input_shape"])
        if obj.fingerprint != st["fingerprint"]:
            raise RuntimeError("library gate fingerprint mismatch on load")
        return obj


class ConstantGate:
    """One always-active gate per output (A2/A2c: ``beta_k + w_k^T x`` with the layout's
    clause intercept switched off)."""
    kind = "constant"

    def __init__(self, K: int):
        self.K, self.M = K, 1
        self.polarity = np.ones(K, np.int8)
        self.weight = None
        self.source_identity = "none"
        self.fingerprint = _h("ConstantGate", K)

    @property
    def n_gates(self) -> int:
        return self.K

    def __call__(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        return np.ones((X.shape[0], self.K), dtype=bool)

    def state(self) -> dict:
        return dict(kind=self.kind, K=self.K, fingerprint=self.fingerprint)


def gates_from_state(st: dict, device: str = "cpu"):
    kind = str(st["kind"])
    if kind == "frozen_rules":
        return FrozenRules.from_state(st)
    if kind == "library":
        return LibraryGates.from_state(st, device=device)
    if kind == "constant":
        return ConstantGate(int(st["K"]))
    if kind == "random_rules":
        from .gates_random import RandomRules
        return RandomRules.from_state(st)
    raise ValueError(kind)
