"""Phase 1 arms: autoregressive MNIST generation with a Tsetlin machine, plus baselines.

Baselines run first and define the NLL floor; no TM arm is scored without them (PLAN C3).
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as Fn

from torchtsetlin import TsetlinMachine
from torchtsetlin.utils import seed_everything

import common as C

W = C.W


# --------------------------------------------------------------------------- streams
def sample_batch(flat, labels, prepped, ctx, n, gen, device):
    idx = torch.randint(0, flat.shape[0], (n,), generator=gen, device=device)
    pos = torch.randint(0, C.NPIX, (n,), generator=gen, device=device)
    x = ctx.encode(prepped, idx, pos, labels[idx])
    y = flat[idx, pos].long()
    return x, y


@torch.no_grad()
def teacher_forced_votes(model, flat, labels, prepped, ctx, positions, chunk, device):
    """Vote sums and targets for every (image, position) pair, teacher-forced."""
    n_img = flat.shape[0]
    vs, ys = [], []
    for p in positions:
        pos = torch.full((n_img,), int(p), device=device, dtype=torch.long)
        for lo in range(0, n_img, chunk):
            sl = slice(lo, min(lo + chunk, n_img))
            idx = torch.arange(sl.start, sl.stop, device=device)
            x = ctx.encode(prepped, idx, pos[sl], labels[idx])
            vs.append(model(x)[:, 1].float())
            ys.append(flat[idx, pos[sl]].long())
    return torch.cat(vs), torch.cat(ys)


def nll_nats_per_image(p, y, n_img):
    p = p.clamp(1e-6, 1 - 1e-6)
    ll = torch.where(y.bool(), p.log(), (1 - p).log())
    return float(-ll.sum() / n_img)


# --------------------------------------------------------------------------- baselines
def arm_marginal(a, data, device):
    """P(ink | position, digit) from training counts. No model. The NLL floor."""
    t0 = time.time()
    xtr, ytr = data["train"]
    flat = xtr.reshape(-1, C.NPIX).float()
    counts = torch.zeros(10, C.NPIX, device=device)
    tot = torch.zeros(10, 1, device=device)
    for d in range(10):
        m = ytr == d
        counts[d] = flat[m].sum(0)
        tot[d] = m.sum()
    p = (counts + 1.0) / (tot + 2.0)                       # Laplace

    res = {}
    for split in ("val", "test"):
        x, y = data[split]
        f = x.reshape(-1, C.NPIX)
        pp = p[y]                                          # (N, 784)
        ll = torch.where(f, pp.log(), (1 - pp).log())
        res[f"nll_{split}"] = float(-ll.sum() / f.shape[0])
    res["n_test_images"] = int(data["test"][0].shape[0])
    C.write_record("mnist-marginal", dict(arm="mnist-marginal", smoothing="laplace"),
                   res, time.time() - t0)
    print(f"[mnist-marginal] test NLL = {res['nll_test']:.2f} nats/image "
          f"(val {res['nll_val']:.2f})", flush=True)
    return p


def arm_logreg(a, data, ctx, device):
    """Logistic regression on the IDENTICAL features — isolates 'TM' from 'these features'."""
    t0 = time.time()
    seed_everything(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    xtr, ytr = data["train"]
    flat_tr = xtr.reshape(-1, C.NPIX)
    prep_tr = ctx.pad(xtr)

    lin = torch.nn.Linear(ctx.n_features, 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    steps = a.examples // a.logreg_batch
    for i in range(steps):
        x, y = sample_batch(flat_tr, ytr, prep_tr, ctx, a.logreg_batch, gen, device)
        loss = Fn.binary_cross_entropy_with_logits(lin(x.float()).squeeze(1), y.float())
        opt.zero_grad(); loss.backward(); opt.step()

    res = {}
    with torch.no_grad():
        for split in ("val", "test"):
            xs, ys = data[split]
            xs, ys = xs[: a.eval_images], ys[: a.eval_images]
            fl, pr = xs.reshape(-1, C.NPIX), ctx.pad(xs)
            lls, n = 0.0, xs.shape[0]
            for p in range(C.NPIX):
                pos = torch.full((n,), p, device=device, dtype=torch.long)
                idx = torch.arange(n, device=device)
                xx = ctx.encode(pr, idx, pos, ys[idx])
                pp = torch.sigmoid(lin(xx.float()).squeeze(1)).clamp(1e-6, 1 - 1e-6)
                tt = fl[idx, pos].bool()
                lls += float(torch.where(tt, pp.log(), (1 - pp).log()).sum())
            res[f"nll_{split}"] = -lls / n
    res["n_eval_images"] = int(a.eval_images)
    cfg = dict(arm="mnist-logreg", context=ctx.name, n_features=ctx.n_features,
               examples=a.examples, batch=a.logreg_batch, seed=a.seed)
    C.write_record(f"mnist-logreg-{ctx.name}", cfg, res, time.time() - t0)
    print(f"[mnist-logreg-{ctx.name}] test NLL = {res['nll_test']:.2f} nats/image "
          f"(val {res['nll_val']:.2f})", flush=True)


# --------------------------------------------------------------------------- TM arm
def arm_tm(a, data, ctx, device):
    t0 = time.time()
    seed_everything(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    flat_tr, prep_tr = xtr.reshape(-1, C.NPIX), ctx.pad(xtr)
    flat_va, prep_va = xva.reshape(-1, C.NPIX), ctx.pad(xva)

    T = a.T if a.T else max(5.0, a.clauses / 32.0)          # G0 Finding 1
    tm = TsetlinMachine(ctx.n_features, 2, a.clauses, T, a.s,
                        weighted=a.weighted).to(device)
    print(f"[{a.arm}] F={ctx.n_features} clauses/class={a.clauses} T={T:.0f} s={a.s} "
          f"batch={a.batch} weighted={a.weighted}", flush=True)

    steps = a.examples // a.batch
    log_every = max(1, steps // a.checks)
    history = []
    tm.train()
    for i in range(1, steps + 1):
        x, y = sample_batch(flat_tr, ytr, prep_tr, ctx, a.batch, gen, device)
        tm.update(x, y)
        if i % log_every == 0 or i == steps:
            empty = int((tm.include_count == 0).sum())
            with torch.no_grad():
                tm.eval()
                xv, yv = sample_batch(flat_va, yva, prep_va, ctx, 20000, gen, device)
                v = tm(xv)[:, 1].float()
                p = C.analytic_p(v, T)
                nll = float(-torch.where(yv.bool(), p.log(), (1 - p).log()).mean())
                vmax = float(v.abs().max())
                out, _ = tm._evaluate(tm._encode(tm._prepare(xv)), empty_value=False)
                raw = tm._votes(out, clamp=False)[:, 1].float()
                raw_p99 = float(raw.abs().quantile(0.99))
                tm.train()
            history.append(dict(step=i, examples=i * a.batch, empty=empty,
                                val_bit_nll=nll, vmax_over_T=vmax / T, raw_p99=raw_p99))
            print(f"    step {i:>7}/{steps}  empty={empty:>5}  |v|max/T={vmax/T:.2f}  "
                  f"raw|v|p99={raw_p99:>7.0f} (T={T:.0f})  val bit-NLL={nll:.4f}", flush=True)

    empty = int((tm.include_count == 0).sum())
    frac = empty / tm.n_clauses_total
    # G0 Finding 4 gate, as a FRACTION: real collapses killed 5-50% of the pool; a handful of
    # empty clauses out of thousands is noise, not collapse.
    if frac > a.max_empty_frac and not a.allow_empty:
        raise SystemExit(f"[{a.arm}] ABORT: {empty}/{tm.n_clauses_total} clauses empty "
                         f"({frac:.1%} > {a.max_empty_frac:.1%}) — G0 Finding 4 says this model "
                         f"is collapsed. Re-tune before scoring.")

    # calibrate on validation (sampled labels only, never p_true)
    tm.eval()
    xv, yv = sample_batch(flat_va, yva, prep_va, ctx, a.calib_examples, gen, device)
    with torch.no_grad():
        vv = tm(xv)[:, 1].float()
    iso = C.fit_isotonic(vv, yv)

    # held-out teacher-forced NLL
    res = {"T": T, "empty_clauses": empty, "history": history}
    for split in ("val", "test"):
        xs, ys = data[split]
        xs, ys = xs[: a.eval_images], ys[: a.eval_images]
        v, y = teacher_forced_votes(tm, xs.reshape(-1, C.NPIX), ys, ctx.pad(xs), ctx,
                                    range(C.NPIX), a.eval_chunk, device)
        n = xs.shape[0]
        res[f"nll_{split}_analytic"] = nll_nats_per_image(C.analytic_p(v, T), y, n)
        res[f"nll_{split}_isotonic"] = nll_nats_per_image(iso(v), y, n)
    res["n_eval_images"] = int(a.eval_images)

    cfg = dict(arm=a.arm, context=ctx.name, n_features=ctx.n_features, clauses=a.clauses,
               T=T, s=a.s, batch=a.batch, examples=a.examples, seed=a.seed,
               weighted=bool(a.weighted),
               rows_above=getattr(ctx, "rows_above", None), left=getattr(ctx, "left", None))
    torch.save({"state": tm.state_dict(), "cfg": cfg,
                "iso_x": vv.cpu(), "iso_y": yv.cpu()},
               C.OUT / f"{a.arm}_seed{a.seed}.pt")
    C.write_record(a.arm, cfg, res, time.time() - t0)
    print(f"[{a.arm}] test NLL: analytic {res['nll_test_analytic']:.2f} / "
          f"isotonic {res['nll_test_isotonic']:.2f} nats/image  (empty={empty})", flush=True)


# --------------------------------------------------------------------------- CLI
def build_ctx(a):
    if a.context == "canvas":
        return C.CanvasContext()
    return C.WindowContext(a.rows_above, a.left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="mnist-tm-window")
    ap.add_argument("--stage", default="tm",
                    choices=["marginal", "logreg", "tm"])
    ap.add_argument("--context", default="window", choices=["window", "canvas"])
    ap.add_argument("--rows-above", type=int, default=2)
    ap.add_argument("--left", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clauses", type=int, default=2000)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--s", type=float, default=40.0)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--examples", type=int, default=20_000_000)
    ap.add_argument("--calib-examples", type=int, default=2_000_000)
    ap.add_argument("--eval-images", type=int, default=2000)
    ap.add_argument("--eval-chunk", type=int, default=10000)
    ap.add_argument("--logreg-batch", type=int, default=512)
    ap.add_argument("--checks", type=int, default=10)
    ap.add_argument("--allow-empty", action="store_true")
    ap.add_argument("--max-empty-frac", type=float, default=0.01)
    ap.add_argument("--weighted", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device)
    data = C.load_mnist(dev)
    ctx = build_ctx(a)
    C.selftest_no_leakage(ctx, dev)
    if a.stage == "marginal":
        arm_marginal(a, data, dev)
    elif a.stage == "logreg":
        arm_logreg(a, data, ctx, dev)
    else:
        arm_tm(a, data, ctx, dev)


if __name__ == "__main__":
    main()
