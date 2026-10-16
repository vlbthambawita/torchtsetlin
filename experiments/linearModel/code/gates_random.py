"""A5 random gates (D5, plan addition): random conjunctions matched clause-by-clause to a
source bank's literal counts and calibrated to its training activation rates.

For source clause ``c`` with ``L_c`` literals and training activation rate ``p_c``: draw an anchor
training example uniformly and take ``L_c`` of the ``F`` literals true for it (so the clause fires
at least once), sampled without replacement with weight ``freq(literal)**alpha`` (Gumbel top-k).
``freq`` is the label-free training frequency of the literal. ``alpha`` is bisected per clause so
the candidates' rate approaches ``p_c``; the closest candidate is kept, and one within ±10%
relative ends the search (D12: uniform sampling, alpha = 0, gave rates ~100x too low). Empty source clauses
stay empty (never fire in prediction mode). The achieved match is reported, not assumed.
"""

from __future__ import annotations

import numpy as np
import torch

from .adapter import FrozenRules, _h


class RandomRules(FrozenRules):
    kind = "random_rules"

    def __init__(self, *a, seed: int = 0, match_report: dict = None, **kw):
        super().__init__(*a, **kw)
        self.kind = "random_rules"          # the dataclass field would otherwise say frozen_rules
        self.seed = seed
        self.match_report = match_report or {}
        self.fingerprint = _h("RandomRules", self.fingerprint, seed)

    def state(self) -> dict:
        st = super().state()
        st.update(kind=self.kind, seed=self.seed)
        return st

    @classmethod
    def from_state(cls, st: dict) -> RandomRules:
        inc = np.unpackbits(st["include"], axis=1, count=int(st["include_cols"])).astype(bool)
        obj = cls(K=int(st["K"]), M=int(st["M"]), n_features=int(st["n_features"]),
                  threshold=float(st["threshold"]), include=inc, empty_clause_output=st["empty_clause_output"],
                  polarity=st["polarity"], weight=st["weight"], source_identity=str(st["source_identity"]),
                  seed=int(st["seed"]))
        if obj.fingerprint != st["fingerprint"]:
            raise RuntimeError("random-rule fingerprint mismatch on load")
        return obj


def random_rules_like(src: FrozenRules, X: np.ndarray, seed: int, rounds: int = 16, per_round: int = 64,
                      tol: float = 0.10, alpha_max: float = 256.0, device: str = "cuda") -> RandomRules:
    rng = np.random.default_rng(seed)
    F = src.n_features
    B = src.encode(X)                                               # (N, F) bool
    N = B.shape[0]
    Bd = B.to(device)
    lit_false = torch.cat([~Bd, Bd], 1).half()                      # exact for counts <= 2048
    target = np.asarray(src.evaluate_bool(B, device=device).mean(0))   # (C,)
    L = src.include.sum(1)
    Bn = B.numpy()
    TL = np.concatenate([Bn, ~Bn], 1)                               # (N, 2F) literal truth
    logp = np.log(np.clip(TL.mean(0), 1.0 / N, 1.0))                # label-free literal frequencies
    inc = np.zeros_like(src.include)
    achieved = np.zeros(len(L))
    alphas = np.zeros(len(L))
    for c in range(len(L)):
        if L[c] == 0:
            continue
        best_err, best_row = np.inf, None
        lo, hi = 0.0, alpha_max
        for _ in range(rounds):
            a = 0.5 * (lo + hi)
            anchors = rng.integers(0, N, per_round)
            keys = a * logp[None, :] + rng.gumbel(size=(per_round, 2 * F))
            keys[~TL[anchors]] = -np.inf                            # only literals true for the anchor
            top = np.argpartition(-keys, int(L[c]) - 1, axis=1)[:, :int(L[c])]
            cand = np.zeros((per_round, 2 * F), bool)
            np.put_along_axis(cand, top, True, axis=1)
            Cd = torch.as_tensor(cand, device=device).half()
            rate = ((lit_false @ Cd.T) == 0).float().mean(0).cpu().numpy()
            err = np.abs(rate - target[c]) / max(target[c], 1.0 / N)
            i = int(err.argmin())
            if err[i] < best_err:
                best_err, best_row, achieved[c], alphas[c] = err[i], cand[i], rate[i], a
            if best_err <= tol:
                break
            if np.median(rate) < target[c]:
                lo = a                                              # favour more frequent literals
            else:
                hi = a
        inc[c] = best_row
    nz = L > 0
    rel = np.abs(achieved - target)[nz] / np.maximum(target[nz], 1.0 / N)
    report = dict(method="anchor+frequency-tilted (D12)", seed=seed, tol=tol, rounds=rounds, per_round=per_round,
                  frac_within_tol=float((rel <= tol).mean()) if nz.any() else 1.0,
                  median_rel_err=float(np.median(rel)) if nz.any() else 0.0,
                  target_mean_rate=float(target.mean()), achieved_mean_rate=float(achieved.mean()),
                  alpha_median=float(np.median(alphas[nz])) if nz.any() else 0.0)
    return RandomRules(K=src.K, M=src.M, n_features=F, threshold=src.threshold, include=inc,
                       empty_clause_output=src.empty_clause_output, polarity=src.polarity, weight=src.weight,
                       source_identity="random-like:" + src.source_identity, seed=seed, match_report=report)
