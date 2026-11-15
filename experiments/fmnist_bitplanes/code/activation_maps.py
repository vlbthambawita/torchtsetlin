"""Where in the image does a saved model find the evidence for its prediction? (user, 2026-11-14)

For one test image per class that the model gets right (row 1) and one it gets wrong (row 2),
draw the image with the evidence for the *predicted* class on top.

Attribution (exact, no saliency heuristic): a conv TM's vote for class k is
``v_k = sum_j w_j * OR_p match_j(p)`` over the clauses j of class k (signed weight w_j, positive or
negative polarity), where ``match_j(p)`` says clause j matches the 10x10 window at patch p.
Each firing clause's vote w_j is split equally over the n_j windows where it matches, and each
window's share is spread over its 100 pixels. The pixel map therefore sums exactly to the
(unclamped) class vote: positive = evidence for the predicted class, negative = against it. The
check ``map.sum() == v_k`` runs for every image.

Writes figures/activation/<arm>/<tag>.png and results/activation/<arm>/<tag>.json, and logs the
figure to the run's W&B run (id fm-<arm>-<tag>, resumed) unless --no-wandb.

Usage: python code/activation_maps.py models/aug/<tag>.pt [--device cuda:1] [--no-wandb]
"""

import argparse
import os

import numpy as np
import torch

from common import CACHE, PROG, RESULTS, load_data, new_record, write_json
from infer import load_model, to_model_input

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
           "Bag", "Ankle boot"]
FOR, AGAINST = np.array([0.90, 0.38, 0.05]), np.array([0.13, 0.42, 0.85])   # orange / blue


@torch.no_grad()
def evidence_maps(model, x, k, hp):
    """x: (B, ...) model input; k: (B,) class per image -> (B, H, W) pixel evidence maps, (B,) votes."""
    M = model.patch_clause_outputs(x, empty_value=False)               # (B, [Pd,] Py, Px, C)
    if M.dim() == 5:
        M = M.any(dim=1)                                               # conv3 kd = D: Pd = 1
    B, Py, Px, C = M.shape
    n_cls = model.n_clauses
    w = model._signed_weights().view(model.n_classes, n_cls)            # (K, n) signed weights
    maps = torch.zeros(B, Py, Px, device=x.device)
    votes = torch.zeros(B, device=x.device)
    for b in range(B):
        m = M[b].reshape(Py * Px, model.n_classes, n_cls)[:, int(k[b])].float()   # (P, n)
        n = m.sum(0)                                                    # windows per clause
        share = torch.where(n > 0, w[int(k[b])] / n.clamp(min=1), torch.zeros_like(n))
        maps[b] = (m @ share).view(Py, Px)
        votes[b] = (w[int(k[b])] * (n > 0).float()).sum()
    ph = hp.get("patch", 10)
    pix = torch.nn.functional.conv_transpose2d(maps[:, None], torch.ones(1, 1, ph, ph, device=x.device))
    return pix[:, 0] / (ph * ph), votes


def pick(y, pred):
    """Per class: the first correctly classified test image, the first misclassified one."""
    right, wrong = [], []
    for k in range(10):
        r = np.flatnonzero((y == k) & (pred == k))
        w = np.flatnonzero((y == k) & (pred != k))
        right.append(int(r[0]) if len(r) else None)
        wrong.append(int(w[0]) if len(w) else None)
    return right, wrong


def draw(raw, maps, rows, y, pred, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f2328", "#59636e"
    fig, ax = plt.subplots(2, 10, figsize=(16, 4.6), facecolor="white")
    for r, (label, idx) in enumerate(rows):
        for k in range(10):
            a = ax[r, k]
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#d0d7de")
            i = idx[k]
            if i is None:
                a.text(0.5, 0.5, "none", ha="center", va="center", color=muted, transform=a.transAxes)
                continue
            img = raw[i].astype(float) / 255
            m = maps[i]
            lim = max(1e-9, float(np.abs(m).max()))
            v = np.clip(m / lim, -1, 1)
            rgb = np.where(v[..., None] >= 0, FOR, AGAINST)
            alpha = 0.85 * np.abs(v)[..., None]
            base = np.repeat((0.15 + 0.7 * img)[..., None], 3, axis=2)          # item light on dark
            a.imshow(base * (1 - alpha) + rgb * alpha, interpolation="nearest")
            if r == 0:
                a.set_title(CLASSES[k], fontsize=10, color=ink)
            a.set_xlabel(f"pred: {CLASSES[pred[i]]}" if r == 1 else f"#{i}", fontsize=8,
                         color=ink if r == 1 else muted)
        ax[r, 0].set_ylabel(label, fontsize=10, color=ink)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Overlay = evidence for the PREDICTED class (orange = votes for, blue = votes "
             "against), exact split of each firing clause's vote over its matching 10x10 windows; "
             "scaled per image. Columns = true class; row 2 shows what the wrong prediction was.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--eval-batch", type=int, default=500)
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, st = load_model(args.model, dev)
    hp, tag, arm = st["hp"], st["tag"], st["arm"]
    d = load_data(hp["enc"])
    n_te = hp["data"]["n_test"]
    y = d["y_test"][:n_te]
    x = torch.from_numpy(np.ascontiguousarray(d["x_test"][:n_te])).to(dev).bool()
    xin = to_model_input(hp["arch"], x)
    model.eval()
    with torch.no_grad():
        sums = torch.cat([model(xin[i:i + args.eval_batch]) for i in range(0, n_te, args.eval_batch)])
    pred = sums.argmax(1).cpu().numpy()
    acc = 100 * float((pred == y).mean())
    if abs(acc - st["test_acc"]) > 1e-9:
        raise SystemExit(f"reloaded model gives {acc:.2f}, saved {st['test_acc']:.2f}")
    right, wrong = pick(y, pred)
    chosen = sorted({i for i in right + wrong if i is not None})
    sel = torch.tensor(chosen, device=dev)
    pix, votes = evidence_maps(model, xin[sel], torch.from_numpy(pred[chosen]).to(dev), hp)
    T = hp["T"]
    clamped = sums[sel, torch.from_numpy(pred[chosen]).to(dev)]
    for j, i in enumerate(chosen):                                    # attribution check
        assert abs(float(pix[j].sum()) - float(votes[j])) <= 1e-3 * max(1.0, abs(float(votes[j]))), i
        if abs(float(votes[j])) < T:
            assert abs(float(votes[j]) - float(clamped[j])) < 1e-3, (i, float(votes[j]), float(clamped[j]))
    maps = {i: pix[j].cpu().numpy() for j, i in enumerate(chosen)}
    raw = d["raw_test"]
    title = (f"{tag}  (best epoch {st['epoch']}, test {acc:.2f}%)")
    png = PROG / "figures" / "activation" / arm / f"{tag}.png"
    draw(raw, maps, [("correct", right), ("wrong", wrong)], y, pred, title, png)
    rec = new_record("activation", device=dev, tag=tag, arm=arm, model=str(args.model),
                     epoch=st["epoch"], test_acc=acc, figure=str(png.relative_to(PROG)),
                     correct_idx=right, wrong_idx=wrong,
                     predicted={str(i): int(pred[i]) for i in chosen},
                     vote_unclamped={str(i): float(votes[j]) for j, i in enumerate(chosen)})
    write_json(rec, RESULTS / "activation" / arm / f"{tag}.json")
    print(f"{tag}: {png}", flush=True)
    if not args.no_wandb and os.environ.get("FMB_WANDB", "1") != "0":
        try:
            import wandb
            from tracking import ENTITY, PROJECT, run_id
            run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(arm, tag), resume="allow",
                             dir=str(CACHE), settings=wandb.Settings(init_timeout=180, console="off"))
            run.log({"activation_maps": wandb.Image(str(png), caption=title)})
            run.summary["activation_maps_epoch"] = st["epoch"]
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] activation map upload failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
