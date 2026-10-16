"""One gate set -> several arms/configurations -> one ``results/<group>/<run_id>.json`` each.

A *group* shares the gates and the Gram matrices, so a whole hyperparameter grid for one
source model costs one Gram build. Arms:

    A0   source model's own prediction                 (no fit)
    A1   frozen gates, constants only, integer search  (layout slopes=False)
    A2   no gates, global affine, integer search       (ConstantGate, clause_bias=False)
    A3   frozen gates, affine, integer search          (the proposed model)
    A3L  as A3, integer line search (D5)
    A4   as A3, continuous ridge;  A1c / A2c likewise for A1 / A2
    A4q  A4 rounded to the state grid (D8, diagnostic)
    A5   random gates (gates_random.py), affine, integer search

    python -m experiments.linearModel.code.run --spec configs/<file>.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import adapter as AD
from . import source_tm as ST
from .data import EXP, load_split, read_manifest
from .fit import FitConfig, audit_neighbors, fit_coefficients
from .hybrid import (
    HybridConfig,
    HybridModel,
    Layout,
    gates_by_output,
    objective,
    scores_direct,
    validation_key,
)
from .opt_gram import build_gram
from .ridge import quantization_bound, ridge_solve

RESULTS = EXP / "results"
ART = EXP / "artifacts" / "hybrid"


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def make_gates(spec: dict, tr, device: str):
    """``spec["gates"]``: ``"source"`` | ``"constant"`` | ``"random:<seed>"``."""
    kind = spec["gates"]
    if kind == "constant":
        return AD.ConstantGate(10), None, None
    model, srec = ST.load(spec["source"], device=device)
    if kind == "source":
        if srec["kind"] == "conv":
            return AD.LibraryGates(model, srec["threshold"]), model, srec
        va = load_split(srec["manifest"], "valid")
        probes = np.concatenate([tr.X, va.X, AD.threshold_probes(tr.X, srec["threshold"])])
        return AD.freeze_rules(model, srec["threshold"], probes, device=device), model, srec
    if kind.startswith("random:"):
        from .gates_random import random_rules_like
        src = AD.freeze_rules(model, srec["threshold"], tr.X, device=device)
        return random_rules_like(src, tr.X, int(kind.split(":")[1]), device=device), model, srec
    raise ValueError(kind)


def layout_for(arm: str, gates) -> Layout:
    if arm in ("A2", "A2c"):
        return Layout(10, 1, 784, clause_bias=False, slopes=True)
    if arm in ("A1", "A1c"):
        return Layout(10, gates.M, 784, clause_bias=True, slopes=False)
    return Layout(10, gates.M, 784, clause_bias=True, slopes=True)


def metrics(S: np.ndarray, y: np.ndarray) -> dict:
    pred = S.argmax(1)
    cm = np.zeros((10, 10), np.int64)
    np.add.at(cm, (y, pred), 1)
    acc, mse = validation_key(S, y)
    return dict(acc=-acc, mse=mse, per_class_acc=(np.diag(cm) / cm.sum(1)).tolist(), confusion=cm.tolist())


def gate_stats(G: np.ndarray, K: int, M: int) -> dict:
    rate = G.mean(0)
    return dict(activation_rate=rate.tolist(), mean_activation=float(rate.mean()),
                n_never_active=int((rate == 0).sum()), n_always_active=int((rate == 1).sum()),
                mean_active_per_example=float(G.reshape(len(G), K, M).sum(2).mean()))


def run_group(spec: dict, out_dir: Path, device: str = "cuda", force: bool = False) -> List[dict]:
    t_group = time.perf_counter()
    manifest = spec["manifest"]
    tr, va = load_split(manifest, "train"), load_split(manifest, "valid")
    split_sha = read_manifest(manifest)["sha256"]
    out_dir.mkdir(parents=True, exist_ok=True)
    gates, src_model, srec = make_gates(spec, tr, device)
    t0 = time.perf_counter()
    Gtr, Gva = gates(tr.X), gates(va.X)
    t_gates = time.perf_counter() - t0
    common = dict(group=spec["name"], manifest=manifest, split_sha=split_sha, gates=spec["gates"],
                  gate_kind=gates.kind, gate_fingerprint=gates.fingerprint, source=spec.get("source"),
                  source_record={k: v for k, v in (srec or {}).items() if k != "history"},
                  n_train=len(tr), n_valid=len(va), git_rev=AD.git_rev(), src_diff=AD.src_diff_stat(),
                  hardware=dict(cpu_threads=torch.get_num_threads(),
                                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    gstats = dict(train=gate_stats(Gtr, gates.K, gates.M), valid=gate_stats(Gva, gates.K, gates.M))
    if isinstance(gates, AD.FrozenRules) or getattr(gates, "include", None) is not None:
        inc = gates.include
        gstats["literals_per_clause"] = inc.sum(1).tolist()
        gstats["n_included_literals"] = int(inc.sum())
    if hasattr(gates, "match_report"):
        gstats["random_match"] = gates.match_report
    records = []

    # ---- A0: the source model itself
    if "A0" in spec["arms"] and src_model is not None:
        rid = f"A0__{spec['source']}"
        if force or not (out_dir / f"{rid}.json").exists():
            B = ST.bool_view(va.X, srec["threshold"], srec["kind"], device)
            t0 = time.perf_counter()
            with torch.no_grad():
                src_model.eval()
                V = torch.cat([src_model(B[i:i + 2048]) for i in range(0, len(B), 2048)]).float().cpu().numpy()
            t_inf = time.perf_counter() - t0
            rec = dict(run_id=rid, arm="A0", **common, val=metrics(V, va.y), inference_seconds_val=t_inf,
                       sizes=dict(n_clauses=srec["M"] * 10, n_included_literals=srec["n_included_literals"]))
            (out_dir / f"{rid}.json").write_text(json.dumps(rec))
            records.append(rec)

    # ---- the fitted arms
    grams: Dict[tuple, np.ndarray] = {}

    def gram_for(lay: Layout, G: np.ndarray):
        key = (lay.clause_bias, lay.slopes, lay.M, G.shape[1])
        if key not in grams:
            grams.clear()                                   # keep at most one Gram in memory
            t0 = time.perf_counter()
            H = build_gram(tr.X, gates_by_output(G, lay), lay)
            grams[key] = (H, time.perf_counter() - t0)
        return grams[key]

    for c in spec["configs"]:
        arm = c["arm"]
        lay = layout_for(arm, gates)
        g_used = AD.ConstantGate(10) if arm in ("A2", "A2c") else gates
        Gt = np.ones((len(tr), 10), bool) if arm in ("A2", "A2c") else Gtr
        Gv = np.ones((len(va), 10), bool) if arm in ("A2", "A2c") else Gva
        hc = HybridConfig(delta=c.get("delta", 1.0), Q=c.get("Q", 1), lam=c["lam"])
        tag = f"{arm}__{c.get('engine', 'ridge')}__d{hc.delta:g}_Q{hc.Q}_l{hc.lam:g}_sw{c.get('max_sweeps', 0)}_seed{c.get('seed', 0)}"
        rid = f"{tag}__{spec['name']}"
        if not force and (out_dir / f"{rid}.json").exists():
            continue
        print(f"[{spec['name']}] {rid}", flush=True)
        rec = dict(run_id=rid, arm=arm, **common, config=c, layout=dict(K=lay.K, M=lay.M, d=lay.d, D_k=lay.D,
                   D_total=lay.D_total, clause_bias=lay.clause_bias, slopes=lay.slopes), gate_stats=gstats,
                   timings=dict(gates=t_gates))
        if arm in ("A4", "A1c", "A2c", "A4q"):
            H, t_build = gram_for(lay, Gt)
            t0 = time.perf_counter()
            sol = ridge_solve(tr.X, Gt, tr.Y, lay, hc.lam, gram=H, with_mu=c.get("with_mu", False))
            rec["timings"].update(gram_build=t_build, ridge=time.perf_counter() - t0)
            theta = sol["theta"]
            if arm == "A4q":
                qd = HybridConfig(c["delta"], c["Q"], hc.lam)
                q = np.clip(np.round(theta / qd.delta), -qd.Q, qd.Q).astype(np.int16)
                Sva, Str = scores_direct(va.X, Gv, lay, q, qd.delta), scores_direct(tr.X, Gt, lay, q, qd.delta)
                rec["frac_clipped"] = float((np.abs(np.round(theta / qd.delta)) > qd.Q).mean())
                rec["train_J"] = objective(Str - tr.Y, q, qd.delta, hc.lam)
                rec["audit"] = audit_neighbors(q, tr.X, Gt, tr.Y, lay, qd.delta, qd.Q, hc.lam, 1e-12)
                model = HybridModel(g_used, lay, qd, q, metadata=dict(run_id=rid))
            else:
                Sva, Str = scores_direct(va.X, Gv, lay, theta, 1.0), scores_direct(tr.X, Gt, lay, theta, 1.0)
                rec["train_J"] = objective(Str - tr.Y, theta, 1.0, hc.lam)
                rec["mu"] = sol["mu"].tolist()
                rec["max_abs_theta"] = float(np.abs(theta).max())
                rec["frac_abs_theta_gt4"] = float((np.abs(theta) > 4).mean())
                (ART.parent / "ridge").mkdir(parents=True, exist_ok=True)
                np.save(ART.parent / "ridge" / f"{rid}.npy", theta)
                rec["sizes"] = dict(n_coefficients=lay.D_total, artifact=f"artifacts/ridge/{rid}.npy")
                model = None
            rec["train"] = dict(acc=float((Str.argmax(1) == tr.y).mean()))
            rec["val"] = metrics(Sva, va.y)
        else:
            engine = c.get("engine", "gram")
            model = HybridModel(g_used, lay, hc, metadata=dict(run_id=rid))
            fc = FitConfig(max_sweeps=c["max_sweeps"], accept_tol=c.get("accept_tol", 1e-12), seed=c.get("seed", 0),
                           engine=engine, line=(arm == "A3L"), log_every=c.get("log_every", 50))
            H, t_b = gram_for(lay, Gt) if engine == "gram" else (None, 0.0)
            res = fit_coefficients(model, tr.X, tr.Y, va.X, va.y, fc, Gtr=Gt, Gva=Gv, gram=H,
                                   record_moves=c.get("record_moves", False),
                                   progress=lambda t: print(f"  sweep {t['sweep']} J {t['J']:.6f} acc {t['accepted']} "
                                                            f"val {t['val_acc']:.4f} {t['seconds']:.2f}s", flush=True))
            res.timings["gram_build"] = t_b
            model.states = res.best_states
            Sva = scores_direct(va.X, Gv, lay, res.best_states, hc.delta)
            Str = scores_direct(tr.X, Gt, lay, res.best_states, hc.delta)
            qa = np.abs(res.best_states)
            rec.update(stop_reason=res.stop_reason, best_sweep=res.best_sweep, terminal_sweep=res.terminal_sweep,
                       trace=res.trace, terminal_audit=res.terminal_audit, selected_audit=res.selected_audit,
                       freeze_id=res.freeze_id, train=dict(acc=float((Str.argmax(1) == tr.y).mean())),
                       train_J_selected=objective(Str - tr.Y, res.best_states, hc.delta, hc.lam),
                       val=metrics(Sva, va.y), frac_at_bound=float((qa == hc.Q).mean()),
                       frac_nonzero=float((qa > 0).mean()))
            rec["timings"].update(res.timings)
            if c.get("record_moves"):
                np.savez_compressed(out_dir / f"{rid}__moves.npz",
                                    **{f"s{i}_{n}": a for i, mv in enumerate(res.moves) for n, a in zip(("perm", "qb", "qa"), mv)})
            if c.get("with_bound"):
                Hb = gram_for(lay, Gt)[0]
                sol = ridge_solve(tr.X, Gt, tr.Y, lay, hc.lam, gram=Hb, with_mu=True)
                Jstar = objective(scores_direct(tr.X, Gt, lay, sol["theta"], 1.0) - tr.Y, sol["theta"], 1.0, hc.lam)
                rec["bound_13_3"] = dict(J_terminal=res.trace[-1]["J"], J_star=Jstar,
                                         gap=res.trace[-1]["J"] - Jstar, mu=sol["mu"].tolist(),
                                         bound=quantization_bound(Hb, hc.lam, hc.delta, sol["mu"], fc.accept_tol),
                                         any_at_bound=bool((np.abs(res.terminal_states) == hc.Q).any()))
        if model is not None:
            ART.mkdir(parents=True, exist_ok=True)
            p = ART / f"{rid}.npz"
            nbytes = model.save(p)
            t0 = time.perf_counter()
            re = HybridModel.load(p, device="cpu")
            S2 = re.predict_scores(va.X)
            rec["inference_seconds_val"] = time.perf_counter() - t0       # load-free: gates + scores on 10k
            if not np.array_equal(S2.argmax(1), Sva.argmax(1)) or np.abs(S2 - Sva).max() > 1e-9:
                raise RuntimeError("reloaded hybrid disagrees with the fitted one")
            rec["sizes"] = dict(serialized_bytes=nbytes, n_coefficients=lay.D_total, artifact=str(p.relative_to(EXP)),
                                checksum=model.checksum())
        rec["peak_rss_gb"] = peak_rss_gb()
        (out_dir / f"{rid}.json").write_text(json.dumps(rec))
        records.append(rec)
        print(f"  -> val acc {rec['val']['acc']:.4f}", flush=True)
    print(f"[{spec['name']}] done in {time.perf_counter() - t_group:.1f}s", flush=True)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    for s in (spec if isinstance(spec, list) else [spec]):
        run_group(s, RESULTS / s.get("out", s["name"]), device=a.device, force=a.force)


if __name__ == "__main__":
    main()
