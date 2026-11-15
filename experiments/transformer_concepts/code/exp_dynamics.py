"""PROOF_PLAN Part II, D1: the drawn pair is uniform and the Type Ia mean drift is
rho - (1 - rho)/(s * p_mem) (THEORY C1, C2).

One clause (class 1, positive slot 0) includes only feature f0, which is 1 on every column; the
example has m columns, f1 = 1 on r of them (rho = r/m). Every update uses B identical positive
examples. After each update the clause row of ``ta_state`` is read and reset, so the clause never
changes between updates. Under uniform draws among matching columns and the library's
probabilities (p_mem to memorise a true literal: 1 with ``boost_true_positive=True``, the library
default, else (s-1)/s; 1/s to forget a false one):

    E[d f1] / E[d f0]     = rho - (1 - rho)/(s * p_mem)
    E[d not f1] / E[d f0] = (1 - rho) - rho/(s * p_mem)

The first run of this script assumed p_mem = (s-1)/s for the default model and failed in 29 of 48
cells (results/dynamics/D1_first_run.json): the default boosts true positives.

Writes results/dynamics/D1.json.  Run: python -m experiments.transformer_concepts.code.exp_dynamics
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from experiments.transformer_concepts.code.hand import init_model, load_clauses

import torchtsetlin as tt

OUT = Path(__file__).resolve().parents[1] / "results" / "dynamics"


def run_cell(m: int, r: int, s: float, device: str, sequential: bool, U: int, B: int, seed: int,
             boost: bool = True):
    torch.manual_seed(seed)
    x = torch.zeros(B, 2, m, dtype=torch.bool)
    x[:, 0, :] = True
    x[:, 1, :r] = True
    y = torch.ones(B, dtype=torch.long)
    model = tt.Conv1dTsetlinMachine(2, 2, T=4, s=s, kernel_size=1, position_encoding=False,
                                    feedback_mode="sequential" if sequential else "batch",
                                    boost_true_positive=boost).to(device)
    init_model(model, x[:1].to(device))
    load_clauses(model, {(1, 0): [0]})
    model.train()
    row = 1 * model.n_clauses + 0
    F = model.n_features
    init = model.ta_state.clone()
    xd, yd = x.to(device), y.to(device)
    d = np.zeros((U, 3))
    for u in range(U):
        model.update(xd, yd)
        st = model.ta_state[row].long().cpu()
        base = init[row].long().cpu()
        d[u] = [(st[0] - base[0]).item(), (st[1] - base[1]).item(), (st[F + 1] - base[F + 1]).item()]
        with torch.no_grad():
            model.ta_state.copy_(init)
        model._refresh_include()
    rho = r / m
    p_mem = 1.0 if boost else (s - 1) / s
    pred_f1 = rho - (1 - rho) / (s * p_mem)
    pred_nf1 = (1 - rho) - rho / (s * p_mem)
    tot = d.sum(0)
    ratio_f1, ratio_nf1 = tot[1] / tot[0], tot[2] / tot[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, U, U)
        t = d[idx].sum(0)
        boots.append((t[1] / t[0], t[2] / t[0]))
    se = np.std(np.array(boots), axis=0)
    z_f1 = (ratio_f1 - pred_f1) / max(se[0], 1e-9)
    z_nf1 = (ratio_nf1 - pred_nf1) / max(se[1], 1e-9)
    return dict(m=m, r=r, rho=rho, s=s, device=device, sequential=sequential, boost=boost, updates=U, batch=B,
                events_f0_total=float(tot[0]), ratio_f1=ratio_f1, pred_f1=pred_f1, se_f1=float(se[0]),
                z_f1=float(z_f1), ratio_not_f1=ratio_nf1, pred_not_f1=pred_nf1, se_not_f1=float(se[1]),
                z_not_f1=float(z_nf1), within_3se=bool(abs(z_f1) <= 3 and abs(z_nf1) <= 3))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    cells = []
    t0 = time.time()
    for boost in (True, False):
        for dev in devices:
            for m in (4, 16):
                for r in sorted({0, m // 4, m // 2, 3 * m // 4, m}):
                    for s in (3.0, 10.0):
                        cells.append(run_cell(m, r, s, dev, False, U=400, B=64, seed=len(cells), boost=boost))
                        print(json.dumps({k: cells[-1][k] for k in ("boost", "device", "m", "r", "s", "ratio_f1", "pred_f1", "z_f1", "z_not_f1")}))
            # sequential mode (one example per update; on CUDA this is the fused single-example commit)
            for r in (0, 4, 8, 16):
                cells.append(run_cell(16, r, 10.0, dev, True, U=3000, B=1, seed=len(cells), boost=boost))
                print(json.dumps({k: cells[-1][k] for k in ("boost", "device", "sequential", "r", "ratio_f1", "pred_f1", "z_f1")}))
    n_ok = sum(c["within_3se"] for c in cells)
    rec = dict(id="D1", claims=["C1", "C2-premise"], cells=cells, n_cells=len(cells), n_within_3se=n_ok,
               pass_=n_ok >= len(cells) - max(1, len(cells) // 50), seconds=round(time.time() - t0, 1),
               summary=f"{n_ok}/{len(cells)} cells within 3 SE of rho - (1-rho)/(s p_mem)")
    rec["pass"] = rec.pop("pass_")
    (OUT / "D1.json").write_text(json.dumps(rec, indent=1))
    print(rec["summary"])


if __name__ == "__main__":
    main()
