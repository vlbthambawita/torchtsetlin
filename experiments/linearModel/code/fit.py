"""Spec §10.2 FIT_COEFFICIENTS and §10.3 AUDIT_NEIGHBORS, shared by both engines.

The engine (``ref`` = spec-literal lazy columns, ``gram`` = Gram form) only runs the inner
coordinate loop. Everything else — permutations, residual rebuild, objective/drift checks,
integrity checks, validation selection, audits — is here and identical for both.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hybrid import (
    HybridModel,
    Layout,
    column_sq_norms,
    design_T,
    gates_by_output,
    objective,
    scores_direct,
    validation_key,
)

# Declared numerical consistency tolerances (spec §10.4), separate from accept_tol.
DRIFT_ABS, DRIFT_REL = 1e-9, 1e-9


@dataclass
class FitConfig:
    max_sweeps: int = 10
    accept_tol: float = 1e-12
    seed: int = 0
    engine: str = "ref"                 # "ref" | "gram"
    line: bool = False                  # A3-L integer line search (D5)
    log_every: int = 1


@dataclass
class FitResult:
    best_states: np.ndarray
    best_sweep: int
    best_key: tuple
    terminal_states: np.ndarray
    terminal_sweep: int
    stop_reason: str
    trace: List[Dict] = field(default_factory=list)
    terminal_audit: Dict = field(default_factory=dict)
    selected_audit: Dict = field(default_factory=dict)
    timings: Dict = field(default_factory=dict)
    freeze_id: str = ""
    moves: Optional[List[np.ndarray]] = None


class IntegrityError(RuntimeError):
    pass


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def best_move_vec(q: np.ndarray, g: np.ndarray, h: np.ndarray, delta: float, Q: int):
    """Vectorised BEST_ONE_STATE_MOVE (``-1`` tried first, strict improvement)."""
    best_m = np.zeros(q.shape, np.int64)
    best_c = np.zeros(q.shape)
    qi = q.astype(np.int64)
    for m in (-1, 1):
        feas = (qi + m >= -Q) & (qi + m <= Q)
        s = delta * m
        c = s * g + (s * s / 2) * h
        better = feas & (c < best_c)
        best_m[better] = m
        best_c[better] = c[better]
    return best_m, best_c


def gradient(X, Gk, R, q, layout: Layout, delta: float, lam: float) -> np.ndarray:
    return design_T(X, Gk, R, layout) / X.shape[0] + lam * delta * q.astype(np.float64)


def audit_neighbors(q, X, G, Y, layout: Layout, delta, Q, lam, accept_tol, h=None) -> Dict:
    """Spec §10.3 on a checkpoint (does not modify it)."""
    Gk = gates_by_output(G, layout)
    R = scores_direct(X, G, layout, q, delta) - Y
    g = gradient(X, Gk, R, q, layout, delta, lam)
    if h is None:
        h = column_sq_norms(X, Gk, layout) / X.shape[0] + lam
    _, c = best_move_vec(q, g, h, delta, Q)
    n_imp = int((c < -accept_tol).sum())
    return dict(stationary_within_tolerance=n_imp == 0, improving_coordinate_count=n_imp,
                best_available_objective_change=float(c.min(initial=0.0)),
                objective=objective(R, q, delta, lam))


def fit_coefficients(model: HybridModel, Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray,
                     yva: np.ndarray, cfg: FitConfig, Gtr: np.ndarray = None, Gva: np.ndarray = None,
                     gram: np.ndarray = None, record_moves: bool = False, progress=None) -> FitResult:
    lay, hc = model.layout, model.cfg
    delta, Q, lam = hc.delta, hc.Q, hc.lam
    assert Xtr.shape[0] > 0 and Xva.shape[0] > 0
    assert delta > 0 and Q >= 1 and lam >= 0 and cfg.max_sweeps >= 1 and cfg.accept_tol >= 0
    if np.any(model.states != 0):
        raise ValueError("initial coefficient states must be zero (spec §1)")
    t_all = time.perf_counter()
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float64)
    Xva = np.ascontiguousarray(Xva, dtype=np.float64)
    Ytr = np.ascontiguousarray(Ytr, dtype=np.float64)
    assert np.isfinite(Xtr).all() and np.isfinite(Ytr).all()
    t0 = time.perf_counter()
    Gtr = model.gate_values(Xtr) if Gtr is None else Gtr
    Gva = model.gate_values(Xva) if Gva is None else Gva
    t_gates = time.perf_counter() - t0
    gate_fp = model.gates.fingerprint
    freeze_id = _sha(Gtr, Xtr, Ytr) + ":" + gate_fp + f":{delta}:{Q}:{lam}:{lay}"
    N = Xtr.shape[0]
    Gk = gates_by_output(Gtr, lay)

    q = np.zeros((lay.K, lay.D), np.int16)
    R = -Ytr.copy()
    J = objective(R, q, delta, lam)
    h = column_sq_norms(Xtr, Gk, lay) / N + lam

    t0 = time.perf_counter()
    if cfg.engine == "ref":
        from .opt_ref import RefEngine
        eng = RefEngine(Xtr, Gk, lay, h)
        eng.start(R, q, delta, lam)
    elif cfg.engine == "gram":
        from .opt_gram import GramEngine
        eng = GramEngine(Xtr, Gk, lay, h, H=gram)
        eng.start(R, q, delta, lam, g0=gradient(Xtr, Gk, R, q, lay, delta, lam))
    else:
        raise ValueError(cfg.engine)
    t_engine = time.perf_counter() - t0

    regression = yva.ndim == 2                         # (N, 1) float targets -> regression key
    vkey = (lambda S: (float(((S - yva) ** 2).mean()), 0.0)) if regression else (lambda S: validation_key(S, yva))
    best_key = vkey(np.zeros((Xva.shape[0], lay.K)))
    best_q, best_sweep = q.copy(), 0
    trace = [dict(sweep=0, J=J, accepted=0, val_acc=-best_key[0], val_mse=best_key[1],
                  frac_at_bound=0.0, frac_nonzero=0.0, seconds=0.0, drift=0.0)]
    stop = "SWEEP_BUDGET"
    rng = np.random.default_rng(cfg.seed)
    moves = [] if record_moves else None
    t_opt = 0.0
    sweep = 0
    for sweep in range(1, cfg.max_sweeps + 1):
        ts = time.perf_counter()
        start_J = J
        perms = np.stack([rng.permutation(lay.D) for _ in range(lay.K)]).astype(np.int64)
        q_before = q.copy() if record_moves else None
        acc, pred = eng.sweep(perms, q, delta, Q, lam, cfg.accept_tol, cfg.line)
        accepted, predicted = int(acc.sum()), float(pred.sum())
        if record_moves:
            moves.append((perms, q_before, q.copy()))
        if np.abs(q.astype(np.int64)).max() > Q:
            raise IntegrityError("state bound violated")
        # rebuild residuals from the states (drift removal, spec §10.2)
        R = scores_direct(Xtr, Gtr, lay, q, delta) - Ytr
        J = objective(R, q, delta, lam)
        g_rebuilt = gradient(Xtr, Gk, R, q, lay, delta, lam) if cfg.engine == "gram" else None
        drift = eng.resync(R, g_rebuilt)
        tol = DRIFT_ABS + DRIFT_REL * max(1.0, abs(start_J))
        if abs(J - (start_J + predicted)) > tol:
            raise IntegrityError(f"sweep {sweep}: J={J!r} but start+predicted={start_J + predicted!r}")
        if J > start_J + tol:
            raise IntegrityError(f"sweep {sweep}: objective rose {start_J!r} -> {J!r}")
        if model.gates.fingerprint != gate_fp or _sha(Gtr, Xtr, Ytr) != freeze_id.split(":")[0]:
            raise IntegrityError("frozen components changed during fitting")
        Sva = scores_direct(Xva, Gva, lay, q, delta)
        key = vkey(Sva)
        if key < best_key:
            best_key, best_q, best_sweep = key, q.copy(), sweep
        dt = time.perf_counter() - ts
        t_opt += dt
        qa = np.abs(q)
        trace.append(dict(sweep=sweep, J=J, accepted=accepted, val_acc=-key[0], val_mse=key[1],
                          frac_at_bound=float((qa == Q).mean()), frac_nonzero=float((qa > 0).mean()),
                          seconds=dt, drift=drift))
        if progress is not None and (sweep % cfg.log_every == 0 or accepted == 0):
            progress(trace[-1])
        if accepted == 0:
            stop = "NO_ACCEPTED_MOVES"
            break

    t0 = time.perf_counter()
    term_audit = audit_neighbors(q, Xtr, Gtr, Ytr, lay, delta, Q, lam, cfg.accept_tol, h)
    sel_audit = audit_neighbors(best_q, Xtr, Gtr, Ytr, lay, delta, Q, lam, cfg.accept_tol, h)
    t_audit = time.perf_counter() - t0
    return FitResult(best_states=best_q, best_sweep=best_sweep, best_key=best_key, terminal_states=q.copy(),
                     terminal_sweep=sweep, stop_reason=stop, trace=trace, terminal_audit=term_audit,
                     selected_audit=sel_audit, freeze_id=freeze_id, moves=moves,
                     timings=dict(gates=t_gates, engine_setup=t_engine, gram_build=getattr(eng, "build_seconds", 0.0),
                                  optimize=t_opt, audit=t_audit, total=time.perf_counter() - t_all))
