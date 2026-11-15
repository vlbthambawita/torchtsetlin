"""PLAN_HTM.md E-H2b: a literal spatial pooler (from scratch) as the value layer of attention.

Stage 1 (unsupervised, training images only). Each 4x4 therm4 CIFAR-10 patch token (192 bits)
becomes literals [u, NOT u] (384). ``m`` columns each have a random potential pool (half of the
literals) with permanences; a literal is connected iff its permanence >= 0.5. Overlap = number
of connected literals that are true, times a boost factor that pushes every column's duty cycle
towards k/m (HTM boosting, so no column dies). The ``k`` columns with the largest boosted overlap
win (k-WTA: density exactly k/m). Winners' permanences move +inc on true and -dec on false pool
literals (THEORY_HTM H2.3: a soft Hamming k-medians quantiser).

Stage 2 (supervised, library). Per token: [patch bits | position bits | for each of the 16 edge
types of cifar.TokenGraphs, the OR of the senders' SP codes]. A
``Conv1dTsetlinMachine(kernel_size=1, position_encoding=False)`` over the 64 token columns
(THEORY_HTM H2.4: a conv TM over fixed per-token vectors). D15 settings.

    python -m experiments.transformer_concepts.code.htm.spatial_pooler jobs | freeze | final
    CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.htm.spatial_pooler work jobs.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

import torchtsetlin as tt
from experiments.transformer_concepts.code import cifar, runner

R = runner.RESULTS / "htm"
ROOT = runner.ROOT
M_COLS = 256
KS = (1, 3, 5, 10, 20, 40, 80)  # densities 0.4 % .. 31 % (D21)
S_GRID = (2.0, 10.0)
FINAL_SEEDS = (1, 2, 3)


class LiteralSP:
    def __init__(self, n_in, m, k, seed, device, pool=0.5, inc=0.03, dec=0.015, boost=3.0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.m, self.k, self.inc, self.dec, self.boost_strength = m, k, inc, dec, boost
        self.pool = (torch.rand(m, 2 * n_in, generator=g) < pool).to(device)
        self.perm = (0.5 + 0.1 * torch.randn(m, 2 * n_in, generator=g)).clamp(0, 1).to(device) * self.pool
        self.duty = torch.full((m,), k / m, device=device)
        self.boost = torch.ones(m, device=device)

    def connected(self):
        return ((self.perm >= 0.5) & self.pool).half()

    def codes(self, u, learn=False):
        """u: (N, n_in) bool -> (N, m) bool with exactly k ones per row."""
        lit = torch.cat([u, ~u], 1).half()
        ov = torch.matmul(lit, self.connected().t()).float() * self.boost.unsqueeze(0)
        ov = ov + 1e-3 * torch.rand_like(ov)  # random tie-break
        win = torch.zeros_like(ov, dtype=torch.bool)
        win.scatter_(1, ov.topk(self.k, 1).indices, True)
        if learn:
            w = win.float()
            n = w.sum(0).clamp(min=1).unsqueeze(1)
            lf = lit.float()
            frac_true = torch.matmul(w.t(), lf) / n  # (m, 2n_in) share of won inputs with the literal true
            delta = self.inc * frac_true - self.dec * (1 - frac_true)
            act = (w.sum(0) > 0).unsqueeze(1)
            self.perm = torch.where(act & self.pool, (self.perm + delta).clamp(0, 1), self.perm)
            self.duty = 0.99 * self.duty + 0.01 * w.mean(0)
            self.boost = torch.exp(self.boost_strength * (self.k / self.m - self.duty) / (self.k / self.m))
        return win


def edge_adjacency(device):
    """(16, 64, 64) float: A[t, i, j] = 1 iff j sends to i over type t (cifar.TokenGraphs)."""
    tg = cifar.TokenGraphs("cpu")
    A = torch.zeros(cifar.N_EDGE_TYPES, 64, 64)
    A[tg.etype, tg.edges[:, 1], tg.edges[:, 0]] = 1.0
    return A.to(device)


def effective_pstar(A):
    """Density p maximising the mean inbox entropy over (type, receiver) slots with >= 1 sender."""
    n = A.sum(2).flatten()
    n = n[n > 0].cpu().numpy()
    ps = np.geomspace(1e-3, 0.5, 400)
    H = []
    for p in ps:
        q = 1 - (1 - p) ** n
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        H.append(np.mean(h))
    return float(ps[int(np.argmax(H))]), dict(senders_min=float(n.min()), senders_mean=float(n.mean()),
                                             senders_max=float(n.max()))


def train_sp(k, seed, device, epochs=2, bs=256):
    path = R / "h2b_sp" / f"sp_m{M_COLS}_k{k}_seed{seed}.pt"
    meta = path.with_suffix(".json")
    if path.exists():
        return torch.load(path, map_location=device), json.loads(meta.read_text())
    D = cifar.load(device)
    sp = LiteralSP(192, M_COLS, k, seed, device)
    g = torch.Generator().manual_seed(seed)
    prev = None
    for ep in range(epochs):
        perm = torch.randperm(len(D["xtr"]), generator=g).to(device)
        for i in range(0, len(perm), bs):
            t = cifar.tokens(D["xtr"][perm[i:i + bs]]).reshape(-1, 192)
            sp.codes(t, learn=True)
        probe = cifar.tokens(D["xva"][:500]).reshape(-1, 192)
        cur = sp.codes(probe)
        stab = float((cur & prev).sum() / cur.sum()) if prev is not None else None
        prev = cur
    usage = cur.float().mean(0)
    info = dict(k=k, m=M_COLS, density=float(cur.float().mean()), target=k / M_COLS,
                dead_columns=int((usage == 0).sum()), usage_max=float(usage.max()),
                code_overlap_last_two_epochs=stab, epochs=epochs, seed=seed)
    state = dict(pool=sp.pool, perm=sp.perm, boost=sp.boost, duty=sp.duty)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    meta.write_text(json.dumps(info, indent=1))
    return state, info


class SPFeatures:
    def __init__(self, state, k, device):
        self.sp = LiteralSP(192, M_COLS, k, 0, device)
        self.sp.pool, self.sp.perm, self.sp.boost = state["pool"], state["perm"], state["boost"]
        self.A = edge_adjacency(device)
        r = torch.arange(64)
        self.pos = torch.cat([cifar._therm(r // 8, 0, 7), cifar._therm(r % 8, 0, 7)], 1).to(device)

    @torch.no_grad()
    def __call__(self, x):
        t = cifar.tokens(x)  # (B, 64, 192)
        B = t.shape[0]
        c = self.sp.codes(t.reshape(-1, 192)).view(B, 64, M_COLS).float()
        inbox = torch.einsum("tij,bjm->bitm", self.A, c) > 0  # (B, 64, 16, m)
        f = torch.cat([t, self.pos.unsqueeze(0).expand(B, -1, -1), inbox.reshape(B, 64, -1)], 2)
        return f.transpose(1, 2).contiguous()  # (B, F, 64)


def run(stage, key, k, s, seed, epochs=30, test=False, device="cuda"):
    path = R / stage / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    state, info = train_sp(k, 0, device)  # one SP per density (seed 0), shared by every TM seed
    feats = SPFeatures(state, k, device)
    D = cifar.load(device)
    tt.seed_everything(seed)
    c = cifar.COMMON
    model = tt.Conv1dTsetlinMachine(cifar.N_CLASSES, c["n_per_class"], c["T"], s, kernel_size=1,
                                    position_encoding=False,
                                    max_included_literals=c["max_included_literals"]).to(device)
    g = torch.Generator().manual_seed(seed)
    curve, best, best_ep, best_state, t_train = [], -1.0, -1, None, 0.0
    inbox_density = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(D["xtr"]), generator=g).to(device)
        t0 = time.perf_counter()
        for i in range(0, len(perm), c["batch"]):
            idx = perm[i:i + c["batch"]]
            model.update(feats(D["xtr"][idx]), D["ytr"][idx])
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        acc = float((cifar.predict(model, feats, D["xva"]) == D["yva"]).float().mean())
        curve.append(round(acc, 5))
        print(f"{key} epoch {ep} val {acc:.4f}", flush=True)
        if acc > best:
            best, best_ep = acc, ep
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
    with torch.no_grad():
        f = feats(D["xva"][:256])
        inbox_density = float(f[:, 206:].float().mean())
    ps, sd = effective_pstar(feats.A)
    rec = dict(stage=stage, key=key, k=k, m=M_COLS, sp_density=k / M_COLS, s=s, seed=seed, epochs=epochs,
               **{kk: v for kk, v in c.items()}, sp=info, inbox_density=inbox_density,
               effective_p_star=ps, senders=sd, val_curve=curve, best_val=best, best_epoch=best_ep,
               train_seconds=round(t_train, 1), gpu=torch.cuda.get_device_name(), torchtsetlin=tt.__version__)
    if test:
        model.load_state_dict(best_state)
        pred = cifar.predict(model, feats, D["xte"]).cpu()
        runner.log_test_access(f"htm/{stage}/{key}", len(pred))
        rec["test_acc"] = float((pred == D["yte"]).float().mean())
        rec["test_acc_per_class"] = [float(((pred == D["yte"]) & (D["yte"] == cc)).sum() / (D["yte"] == cc).sum())
                                     for cc in range(10)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1))
    return rec


def jobs():
    out = [dict(stage="h2b_search", key=f"sp_k{k}_s{int(s)}_seed0", k=k, s=s, seed=0, epochs=10)
           for k in KS for s in S_GRID]
    (ROOT / "jobs_htm_h2b_search.json").write_text(json.dumps(out, indent=1))
    print(len(out))


def freeze():
    p = R / "frozen_h2b.json"
    if p.exists():
        raise SystemExit("freeze exists")
    recs = [json.loads((R / "h2b_search" / f"sp_k{k}_s{int(s)}_seed0.json").read_text()) for k in KS for s in S_GRID]
    b = max(recs, key=lambda r: (r["best_val"], -r["s"]))
    p.write_text(json.dumps(dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), k=b["k"], s=b["s"], best_val=b["best_val"],
                                 grid={f"k{r['k']}_s{r['s']:g}": r["best_val"] for r in recs},
                                 rule="best validation over (k, s), seed 0"), indent=1))
    print(b["k"], b["s"], b["best_val"])


def final():
    fz = json.loads((R / "frozen_h2b.json").read_text())
    out = [dict(stage="h2b_final", key=f"sp_k{fz['k']}_s{fz['s']:g}_seed{sd}", k=fz["k"], s=fz["s"], seed=sd, test=True)
           for sd in FINAL_SEEDS]
    (ROOT / "jobs_htm_h2b_final.json").write_text(json.dumps(out, indent=1))
    print(len(out))


def work(paths):
    for path in paths:
        for job in json.loads(open(path).read()):
            d = R / job["stage"]
            d.mkdir(parents=True, exist_ok=True)
            if (d / f"{job['key']}.json").exists():
                continue
            try:
                os.close(os.open(d / f"{job['key']}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            try:
                r = run(**job)
                (d / f"{job['key']}.lock").unlink(missing_ok=True)
                print(f"[{time.strftime('%H:%M:%S')}] {job['key']} val={r['best_val']:.4f}", flush=True)
            except Exception:
                (d / f"{job['key']}.err").write_text(traceback.format_exc())
                print("FAILED", job["key"], flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "work":
        work(sys.argv[2:])
    else:
        {"jobs": jobs, "freeze": freeze, "final": final}[cmd]()
