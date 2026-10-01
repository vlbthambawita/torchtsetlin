"""Gate G0 — does a Tsetlin machine's vote sum estimate P(y=1 | context)?

Tests the fixed point derived in VALIDITY.md sec.1:  v* = T(2p - 1),  p = (v + T) / 2T.
Every run writes one JSON record to results/g0/. Nothing here touches src/torchtsetlin.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from torchtsetlin import CoalescedTsetlinMachine, TsetlinMachine
from torchtsetlin.utils import seed_everything

import metrics as M
import synth as S

ROOT = Path(__file__).resolve().parents[2]          # experiments/generation
OUT = ROOT / "results" / "g0"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def write_record(sweep, arm, config, result, wall):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {
        "sweep": sweep, "arm": arm, "config": config, "result": result,
        "wall_s": round(wall, 2), "git_sha": git_sha(),
        "torch": torch.__version__, "host": platform.node(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stamp = f"{sweep}__{arm}__{int(time.time()*1000)}.json"
    (OUT / stamp).write_text(json.dumps(rec, indent=2))
    return rec


def train_stream(model, ds, n_examples, batch, gen, sequential=None):
    model.train()
    for _ in range(n_examples // batch):
        x, y = ds.sample(batch, gen)
        model.update(x, y, sequential=sequential)


@torch.no_grad()
def score_binary(model, X, T):
    """Class-1 vote under PREDICTION semantics (empty clause = False)."""
    model.eval()
    return model(X)[:, 1].float()


@torch.no_grad()
def score_both(model, X, T):
    """Class-1 vote under both empty-clause semantics.

    The fixed point v* = T(2p-1) is established while *learning*, where an empty clause
    evaluates True. Reading it back under prediction semantics (empty = False) shifts every
    vote by the number of empty clauses, so both are recorded and compared.
    """
    model.eval();  v_eval = model(X)[:, 1].float()
    model.train(); v_train = model(X)[:, 1].float()
    model.eval()
    return v_eval, v_train


def calib_both(v_eval, v_train, p_true, T):
    r = M.binary_calibration(v_eval, p_true, T)
    rt = M.binary_calibration(v_train, p_true, T)
    r["train_semantics"] = rt
    r["mae_train_semantics"] = rt["mae"]
    r["mean_abs_vote_shift"] = (v_train - v_eval).abs().mean().item()
    return r


@torch.no_grad()
def validation_votes(model, ds, n, gen, chunk=200_000):
    """A held-out stream of (vote, sampled label) pairs for fitting a calibrator."""
    model.eval()
    vs, ys = [], []
    left = n
    while left > 0:
        k = min(chunk, left)
        x, y = ds.sample(k, gen)
        vs.append(model(x)[:, 1].float())
        ys.append(y)
        left -= k
    return torch.cat(vs), torch.cat(ys)


def empty_clause_stats(model):
    ic = model.include_count
    return {"empty_clauses": int((ic == 0).sum().item()),
            "n_clauses_total": int(ic.numel()),
            "mean_included": float(ic.float().mean().item())}


def build_binary(n_features, n_clauses, T, s, device, **kw):
    return TsetlinMachine(n_features, 2, n_clauses, T, s, **kw).to(device)


# ----------------------------------------------------------------------------- sweeps
def p01(a):
    """T sweep on a fixed-capacity machine: does v/T track 2p-1 with slope 1?"""
    for seed in a.seeds:
      for clauses in a.clause_values:
        for T in a.t_values:
            seed_everything(seed)
            g = torch.Generator(device=a.device).manual_seed(seed)
            ds = S.BernoulliContexts(a.n_features, a.n_contexts, seed, a.device)
            m = build_binary(a.n_features, clauses, T, a.s, a.device)
            t0 = time.time()
            train_stream(m, ds, a.n_examples, a.batch, g)
            v, v_tr = score_both(m, ds.contexts, T)
            res = calib_both(v, v_tr, ds.p_true, T)
            res["per_level"] = M.per_level(((v + T) / (2 * T)).clamp(0, 1), ds.p_true)
            vv, vy = validation_votes(m, ds, a.n_val, g)
            f_iso, f_platt = M.fit_calibrators(vv, vy)
            res.update(M.calibrated_errors(v, ds.p_true, T, f_iso, f_platt))
            res.update(empty_clause_stats(m))
            cfg = dict(T=T, clauses=clauses, s=a.s, n_features=a.n_features,
                       n_contexts=a.n_contexts, n_examples=a.n_examples,
                       batch=a.batch, seed=seed, feedback_mode="batch")
            r = write_record("p01", f"C{clauses}_T{T}_seed{seed}", cfg, res, time.time() - t0)
            print(f"[p01] C={clauses:>5} T={T:>4} seed={seed}  mae analytic={res['mae_analytic']:.4f} "
                  f"iso={res['mae_isotonic']:.4f} platt={res['mae_platt']:.4f} | "
                  f"slope={res['slope']:.3f} r2={res['r2']:.3f} "
                  f"|v|max={res['vote_abs_max']:.0f}/T={T:.0f} empty={res['empty_clauses']} "
                  f"({r['wall_s']}s)", flush=True)


def p02(a):
    """Batch size and feedback mode: does batched aggregation bias the fixed point?"""
    combos = [("batch", b) for b in a.batch_values]
    if a.fixed_updates == 0:
        combos += [("sequential", b) for b in (50, 500)]
    for seed in a.seeds:
        for mode, batch in combos:
            # fixed_updates: hold the number of update() calls constant instead of the number
            # of examples, so batch size is not confounded with learning steps.
            n_ex = batch * a.fixed_updates if a.fixed_updates else a.n_examples
            seed_everything(seed)
            g = torch.Generator(device=a.device).manual_seed(seed)
            ds = S.BernoulliContexts(a.n_features, a.n_contexts, seed, a.device)
            m = build_binary(a.n_features, a.clauses, a.T, a.s, a.device,
                             feedback_mode=mode, n_states=a.n_states)
            t0 = time.time()
            train_stream(m, ds, n_ex, batch, g)
            v = score_binary(m, ds.contexts, a.T)
            res = M.binary_calibration(v, ds.p_true, a.T)
            res.update(empty_clause_stats(m))
            cfg = dict(T=a.T, clauses=a.clauses, s=a.s, n_features=a.n_features,
                       n_contexts=a.n_contexts, n_examples=n_ex,
                       batch=batch, seed=seed, feedback_mode=mode,
                       fixed_updates=a.fixed_updates, n_states=a.n_states)
            r = write_record("p02", f"{mode}_b{batch}_seed{seed}", cfg, res, time.time() - t0)
            print(f"[p02] states={a.n_states:>4} batch={batch:>4} updates={n_ex//batch:>7} seed={seed}  "
                  f"mae={res['mae']:.4f} slope={res['slope']:.3f} "
                  f"empty={res['empty_clauses']}  ({r['wall_s']}s)", flush=True)


def p03a(a):
    """Repetition: total examples fixed, number of distinct contexts varied."""
    for seed in a.seeds:
        for M_ctx in a.context_counts:
            seed_everything(seed)
            g = torch.Generator(device=a.device).manual_seed(seed)
            ds = S.BernoulliContexts(a.n_features_wide, M_ctx, seed, a.device)
            m = build_binary(a.n_features_wide, a.clauses, a.T, a.s, a.device)
            t0 = time.time()
            train_stream(m, ds, a.n_examples, a.batch, g)
            probe = ds.contexts[: min(M_ctx, 4096)]
            ptrue = ds.p_true[: probe.shape[0]]
            v = score_binary(m, probe, a.T)
            res = M.binary_calibration(v, ptrue, a.T)
            res["repetitions_per_context"] = a.n_examples / M_ctx
            res.update(empty_clause_stats(m))
            cfg = dict(T=a.T, clauses=a.clauses, s=a.s, n_features=a.n_features_wide,
                       n_contexts=M_ctx, n_examples=a.n_examples, batch=a.batch, seed=seed)
            r = write_record("p03a", f"M{M_ctx}_seed{seed}", cfg, res, time.time() - t0)
            print(f"[p03a] contexts={M_ctx:>7} reps={a.n_examples/M_ctx:>9.1f} seed={seed}  "
                  f"mae={res['mae']:.4f} slope={res['slope']:.3f}  ({r['wall_s']}s)", flush=True)


def p03b(a):
    """Fragmentation: p depends on 4 bits; pad with irrelevant bits (canvas vs window)."""
    for seed in a.seeds:
        for n_noise in a.noise_counts:
            seed_everything(seed)
            g = torch.Generator(device=a.device).manual_seed(seed)
            ds = S.SignalPlusNoise(a.n_signal, n_noise, seed, a.device)
            m = build_binary(ds.n_features, a.clauses, a.T, a.s, a.device)
            t0 = time.time()
            train_stream(m, ds, a.n_examples, a.batch, g)
            X, ptrue, grp = ds.probes(a.n_completions, g)
            v = score_binary(m, X, a.T)
            res = M.binary_calibration(v, ptrue, a.T)
            p_hat = ((v + a.T) / (2 * a.T)).clamp(0, 1)
            spread = torch.stack([p_hat[grp == k].std(unbiased=False)
                                  for k in range(ds.n_groups)])
            res["within_group_spread"] = spread.mean().item()   # spurious noise-bit sensitivity
            res["per_level"] = M.per_level(p_hat, ptrue)
            res.update(empty_clause_stats(m))
            cfg = dict(T=a.T, clauses=a.clauses, s=a.s, n_signal=a.n_signal, n_noise=n_noise,
                       n_features=ds.n_features, n_examples=a.n_examples,
                       batch=a.batch, seed=seed)
            r = write_record("p03b", f"noise{n_noise}_seed{seed}", cfg, res, time.time() - t0)
            print(f"[p03b] noise_bits={n_noise:>3} seed={seed}  mae={res['mae']:.4f} "
                  f"spread={res['within_group_spread']:.4f} slope={res['slope']:.3f}  "
                  f"({r['wall_s']}s)", flush=True)


def p04(a):
    """Multi-class vs coalesced multi-label: are ALL K outputs calibrated, or only argmax?"""
    for seed in a.seeds:
        for K in a.k_values:
            ds = S.CategoricalContexts(a.n_features, a.n_contexts_k, K, 4, seed, a.device)
            variants = {
                "classowned_multiclass": lambda: TsetlinMachine(
                    a.n_features, K, a.clauses_k, a.T_k, a.s).to(a.device),
                "coalesced_multiclass": lambda: CoalescedTsetlinMachine(
                    a.n_features, K, a.shared_clauses, a.T_k, a.s).to(a.device),
                "coalesced_multilabel": lambda: CoalescedTsetlinMachine(
                    a.n_features, K, a.shared_clauses, a.T_k, a.s,
                    multi_label=True, negative_scale=1.0).to(a.device),
            }
            for name, ctor in variants.items():
                seed_everything(seed)
                g = torch.Generator(device=a.device).manual_seed(seed)
                m = ctor()
                t0 = time.time()
                m.train()
                for _ in range(a.n_examples_k // a.batch):
                    x, y = ds.sample(a.batch, g)
                    if name.endswith("multilabel"):
                        y = torch.nn.functional.one_hot(y, K).to(torch.bool)
                    m.update(x, y)
                m.eval()
                with torch.no_grad():
                    votes = m(ds.contexts).float()
                p_lin = (0.5 * (1.0 + votes / m.T)).clamp_min(1e-9)
                p_lin = p_lin / p_lin.sum(dim=1, keepdim=True)
                res = M.categorical_calibration(p_lin, ds.p_true)
                res.update(empty_clause_stats(m))
                res["vote_std_across_outputs"] = votes.std(dim=1).mean().item()
                cfg = dict(K=K, variant=name, T=a.T_k, s=a.s, n_features=a.n_features,
                           n_contexts=a.n_contexts_k, n_examples=a.n_examples_k,
                           batch=a.batch, seed=seed,
                           clauses=a.clauses_k if "classowned" in name else a.shared_clauses)
                r = write_record("p04", f"K{K}_{name}_seed{seed}", cfg, res, time.time() - t0)
                print(f"[p04] K={K:>4} {name:<22} seed={seed}  mae_all={res['mae_all_outputs']:.5f} "
                      f"TV={res['total_variation']:.3f} KL={res['kl_true_to_hat']:.3f} "
                      f"top1={res['top1_match']:.2f}  ({r['wall_s']}s)", flush=True)


def p05(a):
    """train() vs eval() scoring: the empty-clause vote shift (silent-bug check)."""
    seed = a.seeds[0]
    seed_everything(seed)
    g = torch.Generator(device=a.device).manual_seed(seed)
    ds = S.BernoulliContexts(a.n_features, a.n_contexts, seed, a.device)
    m = build_binary(a.n_features, a.clauses, a.T, a.s, a.device)
    t0 = time.time()
    train_stream(m, ds, a.n_examples, a.batch, g)
    with torch.no_grad():
        m.eval();  v_eval = m(ds.contexts)[:, 1].float()
        m.train(); v_train = m(ds.contexts)[:, 1].float()
        m.eval()
    r_eval = M.binary_calibration(v_eval, ds.p_true, a.T)
    r_train = M.binary_calibration(v_train, ds.p_true, a.T)
    res = {"eval": r_eval, "train": r_train,
           "mean_abs_vote_shift": (v_train - v_eval).abs().mean().item(),
           "max_abs_vote_shift": (v_train - v_eval).abs().max().item(),
           "mae_penalty_if_wrong_mode": r_train["mae"] - r_eval["mae"]}
    res.update(empty_clause_stats(m))
    cfg = dict(T=a.T, clauses=a.clauses, s=a.s, n_features=a.n_features,
               n_contexts=a.n_contexts, n_examples=a.n_examples, batch=a.batch, seed=seed)
    r = write_record("p05", f"mode_seed{seed}", cfg, res, time.time() - t0)
    print(f"[p05] empty_clauses={res['empty_clauses']}/{res['n_clauses_total']} "
          f"vote_shift={res['mean_abs_vote_shift']:.3f} "
          f"mae eval={r_eval['mae']:.4f} train={r_train['mae']:.4f}  ({r['wall_s']}s)", flush=True)


def p06(a):
    """Sampling fidelity: draw from p_hat and compare the empirical rate to p_true."""
    seed = a.seeds[0]
    seed_everything(seed)
    g = torch.Generator(device=a.device).manual_seed(seed)
    ds = S.BernoulliContexts(a.n_features, a.n_contexts, seed, a.device)
    m = build_binary(a.n_features, a.clauses, a.T, a.s, a.device)
    t0 = time.time()
    train_stream(m, ds, a.n_examples, a.batch, g)
    v = score_binary(m, ds.contexts, a.T)
    p_hat = ((v + a.T) / (2 * a.T)).clamp(0, 1)
    draws = a.n_draws
    emp = (torch.rand(ds.n_contexts, draws, generator=g, device=a.device)
           < p_hat.unsqueeze(1)).float().mean(dim=1)
    res = {"mae_emp_vs_phat": (emp - p_hat).abs().mean().item(),
           "mae_emp_vs_ptrue": (emp - ds.p_true).abs().mean().item(),
           "draws_per_context": draws}
    cfg = dict(T=a.T, clauses=a.clauses, s=a.s, n_examples=a.n_examples, seed=seed)
    r = write_record("p06", f"sampling_seed{seed}", cfg, res, time.time() - t0)
    print(f"[p06] emp vs p_hat={res['mae_emp_vs_phat']:.5f} "
          f"emp vs p_true={res['mae_emp_vs_ptrue']:.4f}  ({r['wall_s']}s)", flush=True)


def p07(a):
    """Capacity and specificity: s x clause budget."""
    # T is set by the Finding-1 rule (T ~ clauses/32) so specificity is not confounded by it.
    for seed in a.seeds:
        for s in a.s_values:
            for c in a.clause_values:
                T = max(5.0, c / 32.0)
                seed_everything(seed)
                g = torch.Generator(device=a.device).manual_seed(seed)
                ds = S.BernoulliContexts(a.n_features, a.n_contexts, seed, a.device)
                m = build_binary(a.n_features, c, T, s, a.device)
                t0 = time.time()
                train_stream(m, ds, a.n_examples, a.batch, g)
                v, v_tr = score_both(m, ds.contexts, T)
                res = calib_both(v, v_tr, ds.p_true, T)
                vv, vy = validation_votes(m, ds, a.n_val, g)
                f_iso, f_platt = M.fit_calibrators(vv, vy)
                res.update(M.calibrated_errors(v, ds.p_true, T, f_iso, f_platt))
                res.update(empty_clause_stats(m))
                cfg = dict(T=T, clauses=c, s=s, n_features=a.n_features,
                           n_contexts=a.n_contexts, n_examples=a.n_examples,
                           batch=a.batch, seed=seed)
                r = write_record("p07", f"s{s}_c{c}_seed{seed}", cfg, res, time.time() - t0)
                print(f"[p07] s={s:>4} clauses={c:>5} T={T:>6.0f} seed={seed}  "
                      f"mae={res['mae_analytic']:.4f} iso={res['mae_isotonic']:.4f} "
                      f"slope={res['slope']:.3f} empty={res['empty_clauses']}  "
                      f"({r['wall_s']}s)", flush=True)


SWEEPS = {"p01": p01, "p02": p02, "p03a": p03a, "p03b": p03b,
          "p04": p04, "p05": p05, "p06": p06, "p07": p07}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", nargs="+", default=["p01"], choices=list(SWEEPS) + ["all"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-examples", type=int, default=2_000_000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--n-features", type=int, default=10)
    ap.add_argument("--n-contexts", type=int, default=112)
    ap.add_argument("--clauses", type=int, default=800)
    ap.add_argument("--T", type=float, default=50.0)
    ap.add_argument("--s", type=float, default=10.0)
    ap.add_argument("--t-values", type=float, nargs="+", default=[10, 25, 50, 100, 200])
    ap.add_argument("--batch-values", type=int, nargs="+", default=[1, 10, 50, 100, 500])
    ap.add_argument("--n-features-wide", type=int, default=20)
    ap.add_argument("--context-counts", type=int, nargs="+",
                    default=[16, 112, 1000, 10000, 100000])
    ap.add_argument("--n-signal", type=int, default=4)
    ap.add_argument("--noise-counts", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    ap.add_argument("--n-completions", type=int, default=64)
    ap.add_argument("--k-values", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--n-contexts-k", type=int, default=64)
    ap.add_argument("--clauses-k", type=int, default=100)
    ap.add_argument("--shared-clauses", type=int, default=2000)
    ap.add_argument("--T-k", type=float, default=50.0)
    ap.add_argument("--n-examples-k", type=int, default=1_000_000)
    ap.add_argument("--n-draws", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=400_000)
    ap.add_argument("--s-values", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0])
    ap.add_argument("--n-states", type=int, default=128)
    ap.add_argument("--fixed-updates", type=int, default=0,
                    help="if >0, n_examples = batch * this, holding update() calls constant")
    ap.add_argument("--clause-values", type=int, nargs="+", default=[200, 800, 3200])
    a = ap.parse_args()
    a.device = torch.device(a.device)
    names = list(SWEEPS) if "all" in a.sweep else a.sweep
    for n in names:
        print(f"\n===== {n} =====", flush=True)
        SWEEPS[n](a)


if __name__ == "__main__":
    main()
