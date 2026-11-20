"""The three figures drawn every 20 epochs (plan §7): evidence panel, clause health, curves.

Evidence (exact, as cifar10_joint_teams/code/viz.py): the vote of the ConvTM for class k is
``v_k = sum_j w_j * OR_p match_j(p)``. Each firing clause's signed weight ``w_j`` is split
equally over the windows where it matches and each window's share over its 3 x 3 pixels, so the
pixel map of a class sums to its (unclamped) class sum. This identity is asserted.

``draw`` is called by ``train.py`` (not timed). On a saved model:
    python code/viz.py models/factorial/<tag>.pt [--device cuda:0]
"""

import argparse

import numpy as np
import torch
from common import CLASSES, FIGURES, HEALTH_IMAGES, load_cifar, panel_indices

FOR, AGAINST = np.array([0.90, 0.38, 0.05]), np.array([0.13, 0.42, 0.85])   # orange / blue
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
C_TRAIN, C_VAL, C_RUN, C_GAP = "#b3541e", "#1f6feb", "#8c959f", "#1f2328"


@torch.no_grad()
def evidence(model, x, k, top=3):
    """x: bool (B, 24, 32, 32); k: (B,) class per image (model in eval mode).

    Returns pixel maps (B, 32, 32), votes (B,), and per image the ``top`` strongest matching
    positive clauses of class k: [(weight, n literals, clause index, pixel map (32, 32))]."""
    ph, pw = model.patch_size[-2:]
    step = 1 if model.n_clauses_total > 100_000 else 2          # (step, P, C) patch matches
    ones = torch.ones(1, 1, ph, pw, device=x.device)
    w = model._signed_weights().view(model.n_classes, model.n_clauses)
    lens = model.include_count.view(model.n_classes, model.n_clauses)
    pix, votes, tops = [], [], []
    for i in range(0, x.shape[0], step):
        M = model.patch_clause_outputs(x[i:i + step], empty_value=False)       # (b, Py, Px, C)
        b, Py, Px, _ = M.shape
        maps = torch.zeros(b, Py, Px, device=x.device)
        for j in range(b):
            kk = int(k[i + j])
            m = M[j].reshape(Py * Px, model.n_classes, model.n_clauses)[:, kk].float()     # (P, n)
            n = m.sum(0)                                                       # windows per clause
            share = torch.where(n > 0, w[kk] / n.clamp(min=1), torch.zeros_like(n))
            maps[j] = (m @ share).view(Py, Px)
            fired = n > 0
            votes.append((w[kk] * fired.float()).sum())
            score = torch.where(fired, w[kk], torch.full_like(w[kk], float("-inf")))
            t = []
            for c in score.topk(min(top, score.numel())).indices.tolist():
                if not bool(fired[c]) or float(w[kk, c]) <= 0:
                    continue
                cm = (m[:, c] * share[c]).view(1, 1, Py, Px)
                t.append((float(w[kk, c]), int(lens[kk, c]), kk * model.n_clauses + c,
                          (torch.nn.functional.conv_transpose2d(cm, ones)[0, 0] / (ph * pw)).cpu().numpy()))
            tops.append(t)
        pix.append(torch.nn.functional.conv_transpose2d(maps[:, None], ones)[:, 0] / (ph * pw))
    return torch.cat(pix), torch.stack(votes), tops


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _overlay(ax, img, m):
    lim = max(float(np.abs(m).max()), 1e-9)
    v = np.clip(m / lim, -1, 1)
    rgb = np.where(v[..., None] >= 0, FOR, AGAINST)
    alpha = 0.85 * np.abs(v)[..., None]
    gray = img.astype(float).mean(axis=2, keepdims=True) / 255
    base = np.repeat(0.15 + 0.7 * gray, 3, axis=2)
    ax.imshow(base * (1 - alpha) + rgb * alpha, interpolation="nearest")


def _clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


@torch.no_grad()
def panel(model, x, raw, y, title, path, top=3):
    """Evidence panel for images ``raw`` (uint8 (N, 32, 32, 3)), inputs ``x``, labels ``y``.
    Columns: image | evidence for the predicted class | evidence for the true class (only when
    the prediction is wrong) | the ``top`` strongest matching clauses. Returns the panel accuracy."""
    plt = _plt()
    sums = model(x).float()
    pred = sums.argmax(1)
    yt = torch.as_tensor(y, device=x.device)
    pm, v, tops = evidence(model, x, pred, top)
    raw_vote = sums.gather(1, pred.view(-1, 1)).squeeze(1)
    assert torch.allclose(v, raw_vote, atol=1e-2, rtol=1e-4), "evidence does not sum to the class vote"
    assert torch.allclose(pm.flatten(1).sum(1), raw_vote, atol=1.0, rtol=1e-3)
    tm = evidence(model, x, yt, 0)[0].cpu().numpy()
    pm, pred = pm.cpu().numpy(), pred.cpu().numpy()
    N, cols = len(raw), 3 + top
    fig, ax = plt.subplots(N, cols, figsize=(1.45 * cols + 0.6, 1.45 * N + 1.0), facecolor="white", squeeze=False)
    heads = ["image", "predicted class", "true class"] + [f"top clause {i + 1}" for i in range(top)]
    for i in range(N):
        for c in range(cols):
            _clean(ax[i, c])
            if i == 0:
                ax[i, c].set_title(heads[c], fontsize=9, color=INK)
        ok = pred[i] == y[i]
        ax[i, 0].imshow(raw[i], interpolation="nearest")
        ax[i, 0].set_ylabel(f"{CLASSES[y[i]]}\n-> {CLASSES[pred[i]]}", fontsize=7.5, color=INK if ok else "#b3261e")
        _overlay(ax[i, 1], raw[i], pm[i])
        ax[i, 1].set_xlabel(f"sum {float(raw_vote[i]):.0f}", fontsize=6.5, color=INK)
        if ok:
            ax[i, 2].text(0.5, 0.5, "same", ha="center", va="center", color=MUTED, transform=ax[i, 2].transAxes)
        else:
            _overlay(ax[i, 2], raw[i], tm[i])
            ax[i, 2].set_xlabel(f"sum {float(sums[i, y[i]]):.0f}", fontsize=6.5, color=INK)
        for t in range(top):
            a = ax[i, 3 + t]
            if t < len(tops[i]):
                w, n, c, cm = tops[i][t]
                _overlay(a, raw[i], cm)
                a.set_xlabel(f"#{c}  w {w:.0f}  {n} lit", fontsize=6.5, color=INK)
            else:
                a.text(0.5, 0.5, "none", ha="center", va="center", color=MUTED, transform=a.transAxes)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.004, "Overlay = evidence for a class (orange = for, blue = against): each firing clause's "
             "weight split over its matching windows; a map sums to the class sum.\nTop clauses: strongest "
             "matching clauses of the predicted class (clause, weight, literals). Row label: true class -> "
             "prediction (red = wrong).", fontsize=7, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return 100 * float((pred == np.asarray(y)).mean())


@torch.no_grad()
def firing_rate(model, x, batch=250):
    """Per clause, the fraction of images ``x`` on which it fires (prediction mode) -> (C,)."""
    n = torch.zeros(model.n_clauses_total, device=x.device)
    for i in range(0, x.shape[0], batch):
        n += model.clause_outputs(model._encode(x[i:i + batch]), empty_value=False).float().sum(0)
    return (n / x.shape[0]).cpu().numpy()


@torch.no_grad()
def health(model, x_val, val_sums, y_val, T, title, path):
    """Clause health: clause length, clause weight, firing rate, validation vote margin.
    ``val_sums``: raw class sums of the whole validation set. Returns scalars for the record."""
    plt = _plt()
    lens = model.include_count.cpu().numpy()
    w = model.weights.cpu().numpy()
    rate = firing_rate(model, x_val[:HEALTH_IMAGES])
    s = np.sort(val_sums, axis=1)
    top, margin = s[:, -1] / T, (s[:, -1] - s[:, -2]) / T
    ok = val_sums.argmax(1) == y_val
    out = {"frac_never_firing": float((rate == 0).mean()), "frac_firing_over_half": float((rate > 0.5).mean()),
           "frac_top_sum_at_T": float((top >= 1).mean()), "median_margin_over_T": float(np.median(margin)),
           "max_weight": int(w.max()), "median_weight": float(np.median(w))}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), facecolor="white")
    ax[0].hist(lens, bins=np.arange(-0.5, max(lens.max(), 33) + 1.5), color=C_VAL)
    ax[0].set_xlabel("included literals per clause")
    ax[0].set_title(f"clause length (mean {lens.mean():.1f}, empty {100 * (lens == 0).mean():.1f} %)")
    ax[1].hist(w, bins=50, color=C_VAL)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("clause weight")
    ax[1].set_title(f"clause weights (median {out['median_weight']:.0f}, max {out['max_weight']})")
    ax[2].hist(rate, bins=np.linspace(0, 1, 51), color=C_VAL)
    ax[2].set_yscale("log")
    ax[2].set_xlabel(f"fraction of {min(HEALTH_IMAGES, len(x_val))} validation images a clause fires on")
    ax[2].set_title(f"firing rate (never: {100 * out['frac_never_firing']:.1f} % of clauses)")
    bins = np.linspace(min(0.0, float(margin.min())), max(1.0, float(margin.max())), 41)
    ax[3].hist([margin[ok], margin[~ok]], bins=bins, stacked=True, color=[C_VAL, C_TRAIN],
               label=["correct", "wrong"])
    ax[3].set_xlabel("(top class sum - runner-up) / T, validation")
    ax[3].set_title(f"vote margin (top sum >= T: {100 * out['frac_top_sum_at_T']:.1f} % of images)")
    ax[3].legend(frameon=False, fontsize=8)
    for a in ax:
        a.set_ylabel("clauses" if a is not ax[3] else "images")
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.title.set_fontsize(9)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return out


def curves(curve, title, path):
    """Accuracy on the clean training subset, validation accuracy and their gap against epoch."""
    plt = _plt()
    e = [r["epoch"] for r in curve]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), facecolor="white")
    ax[0].plot(e, [r["train_acc_running"] for r in curve], color=C_RUN, linewidth=1.2,
               label="train, running (as trained)")
    ax[0].plot(e, [r["train_eval_acc"] for r in curve], color=C_TRAIN, linewidth=1.8, label="train, clean subset")
    ax[0].plot(e, [r["val_acc"] for r in curve], color=C_VAL, linewidth=1.8, label="validation")
    best = max(curve, key=lambda r: (r["val_acc"], -r["epoch"]))
    ax[0].plot([best["epoch"]], [best["val_acc"]], "o", color=C_VAL, markersize=5)
    ax[0].annotate(f"best {best['val_acc']:.2f} @ {best['epoch']}", (best["epoch"], best["val_acc"]),
                   textcoords="offset points", xytext=(-6, -13), ha="right", fontsize=8, color=C_VAL)
    ax[0].set_ylabel("accuracy (%)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].plot(e, [r["gap"] for r in curve], color=C_GAP, linewidth=1.8)
    ax[1].axhline(0, color=GRID, linewidth=1)
    ax[1].set_ylabel("train (clean) - validation (pp)")
    ax[1].set_title("overfitting gap", fontsize=9)
    for a in ax:
        a.set_xlabel("epoch")
        a.xaxis.get_major_locator().set_params(integer=True)
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def draw(model, pan, x_val, val_sums, y_val, T, curve, title, folder, stem):
    """All three figures of one epoch -> ({W&B key: (png, caption)}, scalars). The model is put in
    eval mode for the drawing and restored."""
    was_training = model.training
    model.eval()
    try:
        files = {k: folder / f"{stem}_{k}.png" for k in ("panel", "health", "curves")}
        acc = panel(model, pan["x"], pan["raw"], pan["y"], title, files["panel"])
        scalars = health(model, x_val, val_sums, y_val, T, title, files["health"])
        curves(curve, title, files["curves"])
    finally:
        if was_training:
            model.train()
    return {f"viz/{k}": (p, title) for k, p in files.items()}, dict(scalars, panel_acc=acc)


def main():
    from train import class_sums, gpu_encode, load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="models/<phase>/<tag>.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    model, st = load_model(args.model, dev)
    hp = st["hp"]
    data = load_cifar(hp["data"]["subset"])
    xva, yva = gpu_encode(data["val"][0], dev), data["val"][1]
    sums = class_sums(model, xva, 1000).float().cpu().numpy()
    idx = panel_indices(yva)
    pan = {"x": xva[torch.as_tensor(idx, device=dev)], "raw": data["val"][0][idx], "y": yva[idx]}
    folder = FIGURES / "saved" / st["tag"]
    title = f"{st['tag']}  best epoch {st['epoch']}  val {st['val_acc']:.2f} %  test {st['test_acc']:.2f} %"
    model.eval()
    acc = panel(model, pan["x"], pan["raw"], pan["y"], title, folder / "panel.png")
    h = health(model, xva, sums, yva, hp["T"], title, folder / "health.png")
    print(f"val {100 * float((sums.argmax(1) == yva).mean()):.2f} % (saved: {st['val_acc']:.2f} %), panel {acc:.0f} %, {h}")
    print(f"figures: {folder}")


if __name__ == "__main__":
    main()
