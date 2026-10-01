"""Diagnostic: where does the vote actually sit, and which clauses carry it?"""
from __future__ import annotations
import argparse, time, torch
from torchtsetlin import TsetlinMachine
from torchtsetlin.utils import seed_everything
import synth as S


@torch.no_grad()
def decompose(m, X):
    """Raw (unclamped) class-1 vote plus the positive/negative firing counts behind it."""
    m.eval()
    out, _ = m._evaluate(m._encode(m._prepare(X)), empty_value=False)   # (B, C)
    raw = m._votes(out, clamp=False)[:, 1]
    cls1 = (m.clause_class == 1)
    pos = cls1 & (m.clause_polarity > 0)
    neg = cls1 & (m.clause_polarity < 0)
    return (raw.float(), out[:, pos].sum(1).float(), out[:, neg].sum(1).float(),
            int(pos.sum()), int(neg.sum()))


def run(n_examples, clauses, T, s, batch, seed, device, n_states=128, mode="batch", boost=True):
    seed_everything(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    ds = S.BernoulliContexts(10, 112, seed, device)
    m = TsetlinMachine(10, 2, clauses, T, s, n_states=n_states,
                       feedback_mode=mode, boost_true_positive=boost).to(device)
    m.train()
    t0 = time.time()
    for _ in range(n_examples // batch):
        x, y = ds.sample(batch, g)
        m.update(x, y)
    raw, fpos, fneg, npos, nneg = decompose(m, ds.contexts)
    v = m(ds.contexts)[:, 1].float()
    p_hat = ((v + T) / (2 * T)).clamp(0, 1)
    ic = m.include_count
    cls1 = (m.clause_class == 1)
    emp_pos = int(((ic == 0) & cls1 & (m.clause_polarity > 0)).sum())
    emp_neg = int(((ic == 0) & cls1 & (m.clause_polarity < 0)).sum())
    lo, hi = ds.p_true < 0.2, ds.p_true > 0.8
    print(f"N={n_examples:>8} C={clauses:>4} T={T:>5.0f} s={s:>4} states={n_states} "
          f"batch={batch:>4} {mode:<10} boost={int(boost)} | "
          f"raw v1: lo-p {raw[lo].mean():>8.1f}  hi-p {raw[hi].mean():>8.1f} | "
          f"fire pos/neg: lo {fpos[lo].mean():>6.1f}/{fneg[lo].mean():>6.1f} "
          f"hi {fpos[hi].mean():>6.1f}/{fneg[hi].mean():>6.1f} | "
          f"empty pos/neg {emp_pos}/{emp_neg} of {npos}/{nneg} | "
          f"mae={(p_hat - ds.p_true).abs().mean():.4f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    print("--- is it undertrained? ---")
    for n in (100_000, 500_000, 2_000_000):
        run(n, 800, 50, 10.0, 50, 0, dev)
    print("--- clause budget vs T (T fixed at 50) ---")
    for c in (40, 100, 200, 800):
        run(500_000, c, 50, 10.0, 50, 0, dev)
    print("--- T sweep at 100 clauses/class ---")
    for T in (5, 15, 25, 50):
        run(500_000, 100, T, 10.0, 50, 0, dev)
    print("--- specificity / states / boost / feedback mode ---")
    for s in (2.0, 5.0, 20.0):
        run(500_000, 100, 25, s, 50, 0, dev)
    for st in (32, 256):
        run(500_000, 100, 25, 10.0, 50, 0, dev, n_states=st)
    run(500_000, 100, 25, 10.0, 50, 0, dev, boost=False)
    run(200_000, 100, 25, 10.0, 50, 0, dev, mode="sequential")
