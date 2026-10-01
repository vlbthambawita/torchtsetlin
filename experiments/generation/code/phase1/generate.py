"""Autoregressive sampling from a trained TM, plus sample-quality evaluation.

Sampling is serial over the 784 positions, so we batch across *samples*, not steps.
"""
from __future__ import annotations

import argparse
import time

import torch

from torchtsetlin import TsetlinMachine
import common as C


@torch.no_grad()
def generate(tm, ctx, calib, digits, T, gen, device):
    """digits: (B,) long. Returns (B, 28, 28) bool."""
    B = digits.shape[0]
    canvas = torch.zeros(B, C.H, C.W, dtype=torch.bool, device=device)
    idx = torch.arange(B, device=device)
    tm.eval()
    for i in range(C.NPIX):
        prep = ctx.pad(canvas)
        pos = torch.full((B,), i, device=device, dtype=torch.long)
        x = ctx.encode(prep, idx, pos, digits)
        v = tm(x)[:, 1].float()
        p = calib(v) if calib is not None else C.analytic_p(v, T)
        bit = torch.rand(B, generator=gen, device=device) < p
        canvas[:, i // C.W, i % C.W] = bit
    return canvas


# --------------------------------------------------------------------------- judge
class Judge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(64 * 7 * 7, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def train_judge(data, device, epochs=3, seed=0):
    """CNN trained on the SAME binarised train split. Used only to score generated samples."""
    torch.manual_seed(seed)
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    m = Judge().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    n = xtr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, 256):
            b = perm[lo:lo + 256]
            loss = torch.nn.functional.cross_entropy(
                m(xtr[b].unsqueeze(1).float()), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(xte.unsqueeze(1).float()).argmax(1) == yte).float().mean())
    return m, acc


# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def hamming_to_train(samples, train_flat, chunk=8192):
    """Min Hamming distance from each sample to any training image."""
    s = samples.reshape(samples.shape[0], -1).float()
    best = torch.full((s.shape[0],), float("inf"), device=s.device)
    for lo in range(0, train_flat.shape[0], chunk):
        t = train_flat[lo:lo + chunk].float()
        d = s @ (1 - t).T + (1 - s) @ t.T
        best = torch.minimum(best, d.min(dim=1).values)
    return best


@torch.no_grad()
def sample_metrics(samples, digits, judge, data, device):
    flat = samples.reshape(samples.shape[0], -1)
    res = {}
    pred = judge(samples.unsqueeze(1).float()).argmax(1)
    res["judge_accuracy"] = float((pred == digits).float().mean())
    res["ink_fraction"] = float(flat.float().mean())
    res["ink_fraction_real"] = float(data["test"][0].float().mean())

    per_digit_distinct, per_digit_pair = [], []
    for d in range(10):
        m = digits == d
        f = flat[m].float()
        per_digit_distinct.append(int(torch.unique(flat[m], dim=0).shape[0]))
        pd = f @ (1 - f).T + (1 - f) @ f.T
        k = f.shape[0]
        per_digit_pair.append(float(pd.sum() / max(1, k * (k - 1))))
    res["distinct_per_digit_min"] = min(per_digit_distinct)
    res["distinct_per_digit_mean"] = sum(per_digit_distinct) / 10
    res["pairwise_hamming_mean"] = sum(per_digit_pair) / 10
    res["judge_accuracy_per_digit"] = [
        float((pred[digits == d] == d).float().mean()) for d in range(10)]

    tf = data["train"][0].reshape(-1, C.NPIX)
    res["nn_train_hamming_mean"] = float(hamming_to_train(samples, tf).mean())
    real = data["test"][0][:500]
    res["nn_train_hamming_real_reference"] = float(hamming_to_train(real, tf).mean())
    return res


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-per-digit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrator", default="isotonic", choices=["isotonic", "analytic"])
    a = ap.parse_args()
    dev = torch.device(a.device)
    t0 = time.time()

    ck = torch.load(a.checkpoint, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    ctx = (C.CanvasContext() if cfg["context"] == "canvas"
           else C.WindowContext(cfg["rows_above"], cfg["left"]))
    tm = TsetlinMachine(ctx.n_features, 2, cfg["clauses"], cfg["T"], cfg["s"],
                        weighted=cfg.get("weighted", False)).to(dev)
    tm.load_state_dict(ck["state"])
    calib = (C.fit_isotonic(ck["iso_x"].to(dev), ck["iso_y"].to(dev))
             if a.calibrator == "isotonic" else None)

    data = C.load_mnist(dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    digits = torch.arange(10, device=dev).repeat_interleave(a.n_per_digit)
    samples = generate(tm, ctx, calib, digits, cfg["T"], gen, dev)

    judge, judge_acc = train_judge(data, dev)
    res = sample_metrics(samples, digits, judge, data, dev)
    res["judge_test_accuracy_on_real"] = judge_acc
    res["calibrator"] = a.calibrator
    res["n_samples"] = int(digits.shape[0])

    torch.save({"samples": samples.cpu(), "digits": digits.cpu()},
               C.OUT / f"{cfg['arm']}_seed{cfg['seed']}_samples.pt")
    C.write_record(f"{cfg['arm']}-gen", cfg, res, time.time() - t0)
    print(f"[{cfg['arm']}] judge acc on generated = {res['judge_accuracy']:.3f} "
          f"(judge is {judge_acc:.3f} on real) | distinct/digit "
          f"{res['distinct_per_digit_mean']:.0f} | ink {res['ink_fraction']:.3f} "
          f"vs real {res['ink_fraction_real']:.3f} | NN-train Hamming "
          f"{res['nn_train_hamming_mean']:.1f} (real ref "
          f"{res['nn_train_hamming_real_reference']:.1f})", flush=True)


if __name__ == "__main__":
    main()
