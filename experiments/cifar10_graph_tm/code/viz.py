"""Figures drawn every 20 epochs and for the kept model (plan §7): evidence panel, clause health,
curves, and the strongest rules.

Evidence (exact): the vote for class k is ``v_k = sum_j W[j, k] * OR_n match_j(n)`` over the
nodes n (image windows). Each firing clause's signed weight is split equally over the nodes where
its whole (deep) clause matches, and each node's share over the pixels of its window, so the pixel
map of a class sums to its (unclamped) class sum. This identity is asserted. With message passing
a node's match also depends on its neighbours; the map shows where the deep clause holds.

``draw`` is called by ``train.py`` (not timed). On a kept model:
    python code/viz.py models/depth/<tag>.pt [--device cuda:0]
"""

import argparse
import textwrap

import numpy as np
import torch
from common import CLASSES, FIGURES, HEALTH_IMAGES, panel_indices

FOR, AGAINST = np.array([0.90, 0.38, 0.05]), np.array([0.13, 0.42, 0.85])   # orange / blue
INK, MUTED, GRID = "#1f2328", "#59636e", "#d0d7de"
C_TRAIN, C_VAL, C_RUN, C_GAP = "#b3541e", "#1f6feb", "#8c959f", "#1f2328"


@torch.no_grad()
def node_matches(model, inp, x):
    """(b, P, C) bool: where each clause holds, prediction semantics (model in eval mode)."""
    if hasattr(model, "node_clause_outputs"):
        return model.node_clause_outputs(inp(x), empty_value=False)
    return model.patch_clause_outputs(inp(x), empty_value=False).flatten(1, 2)


@torch.no_grad()
def evidence(model, inp, x, k, top=3):
    """x: image planes (B, Z, 32, 32); k: (B,) class per image.

    Returns pixel maps (B, 32, 32), votes (B,), and per image the ``top`` strongest matching
    positive clauses of class k: [(weight, n literals, clause index, pixel map (32, 32))]."""
    g, ph, st = inp.grid, inp.patch, inp.stride
    ones = torch.ones(1, 1, ph, ph, device=x.device)
    W = model.weights.float()                                          # (C, K)
    lens = model.include_count

    def pixels(node_map):                                              # (1, 1, g, g) -> (32, 32)
        return torch.nn.functional.conv_transpose2d(node_map, ones, stride=st)[0, 0] / (ph * ph)

    pix, votes, tops = [], [], []
    for i in range(x.shape[0]):
        m = node_matches(model, inp, x[i:i + 1])[0].float()            # (P, C)
        w = W[:, int(k[i])]
        n = m.sum(0)                                                   # nodes per clause
        fired = n > 0
        share = torch.where(fired, w / n.clamp(min=1), torch.zeros_like(n))
        pix.append(pixels((m @ share).view(1, 1, g, g)))
        votes.append((w * fired.float()).sum())
        score = torch.where(fired, w, torch.full_like(w, float("-inf")))
        t = []
        for c in score.topk(min(top, score.numel())).indices.tolist():
            if bool(fired[c]) and float(w[c]) > 0:
                t.append((float(w[c]), int(lens[c]), c, pixels((m[:, c] * share[c]).view(1, 1, g, g)).cpu().numpy()))
        tops.append(t)
    return torch.stack(pix), torch.stack(votes), tops


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


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    _plt().close(fig)


@torch.no_grad()
def panel(model, inp, x, raw, y, title, path, top=3):
    """Evidence panel for images ``raw`` (uint8 (N, 32, 32, 3)), planes ``x``, labels ``y``.
    Columns: image | evidence for the predicted class | evidence for the true class (only when
    the prediction is wrong) | the ``top`` strongest matching clauses. Returns the panel accuracy."""
    plt = _plt()
    sums = model(inp(x)).float()
    pred = sums.argmax(1)
    yt = torch.as_tensor(y, device=x.device)
    pm, v, tops = evidence(model, inp, x, pred, top)
    raw_vote = sums.gather(1, pred.view(-1, 1)).squeeze(1)
    assert torch.allclose(v, raw_vote, atol=1e-2, rtol=1e-4), "evidence does not sum to the class vote"
    assert torch.allclose(pm.flatten(1).sum(1), raw_vote, atol=1.0, rtol=1e-3)
    tm = evidence(model, inp, x, yt, 0)[0].cpu().numpy()
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
             "signed weight split over the nodes (windows) where it holds; a map sums to the class sum.\nTop "
             "clauses: strongest matching clauses voting for the predicted class (clause, weight, literals). "
             "Row label: true class -> prediction (red = wrong).", fontsize=7, color=MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    _save(fig, path)
    return 100 * float((pred == np.asarray(y)).mean())


@torch.no_grad()
def firing_rate(model, inp, x, batch=4):
    """Per clause, the fraction of images ``x`` on which it fires (prediction mode) -> (C,)."""
    n = torch.zeros(model.n_clauses_total, device=x.device)
    for i in range(0, x.shape[0], batch):
        n += node_matches(model, inp, x[i:i + batch]).any(1).float().sum(0)
    return (n / x.shape[0]).cpu().numpy()


def layer_lengths(model):
    """(C, depth) included literals per clause and layer (own features, then each inbox)."""
    depth = getattr(model, "depth", 1)
    if depth == 1:
        return model.include_count.view(-1, 1).cpu().numpy()
    Fn = int(model.n_features)
    b = model._layer_bounds()
    inc = model.ta_state >= model.n_states
    return torch.stack([inc[:, b[i]:b[i + 1]].sum(1) + inc[:, Fn + b[i]:Fn + b[i + 1]].sum(1)
                        for i in range(depth)], dim=1).cpu().numpy()


@torch.no_grad()
def health(model, inp, x_held, sums, y_held, T, title, path, held="val"):
    """Clause health: clause length (per layer), clause weights, firing rate, vote margin.
    ``sums``: raw class sums of the whole held-out set. Returns scalars for the record."""
    plt = _plt()
    L = layer_lengths(model)
    lens = L.sum(1)
    w = model.weights.cpu().numpy()                                    # (C, K), signed
    wmax = np.abs(w).max(1)
    rate = firing_rate(model, inp, x_held[:HEALTH_IMAGES])
    s = np.sort(sums, axis=1)
    top, margin = s[:, -1] / T, (s[:, -1] - s[:, -2]) / T
    ok = sums.argmax(1) == y_held
    out = {"frac_never_firing": float((rate == 0).mean()), "frac_firing_over_half": float((rate > 0.5).mean()),
           "frac_top_sum_at_T": float((top >= 1).mean()), "median_margin_over_T": float(np.median(margin)),
           "max_weight": int(wmax.max()), "median_weight": float(np.median(wmax))}
    if L.shape[1] > 1:
        msg = L[:, 1:].sum(1)
        out.update(frac_clauses_with_messages=float((msg > 0).mean()),
                   mean_message_literals=float(msg.mean()),
                   **{f"mean_len_layer{i}": float(L[:, i].mean()) for i in range(L.shape[1])})
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), facecolor="white")
    bins = np.arange(-0.5, max(lens.max(), 33) + 1.5)
    if L.shape[1] > 1:
        for i, col in zip(range(L.shape[1]), (C_VAL, C_TRAIN, C_GAP)):
            ax[0].hist(L[:, i], bins=bins, histtype="step", linewidth=1.5, color=col,
                       label="own features" if i == 0 else f"inbox of layer {i}")
        ax[0].legend(frameon=False, fontsize=7)
        ax[0].set_title(f"literals per layer (mean total {lens.mean():.1f}; "
                        f"{100 * out['frac_clauses_with_messages']:.0f} % read messages)")
    else:
        ax[0].hist(lens, bins=bins, color=C_VAL)
        ax[0].set_title(f"clause length (mean {lens.mean():.1f}, empty {100 * (lens == 0).mean():.1f} %)")
    ax[0].set_xlabel("included literals per clause")
    ax[1].hist(w.ravel(), bins=60, color=C_VAL)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("signed clause weight (one per clause and class)")
    ax[1].set_title(f"clause weights (median max |w| {out['median_weight']:.0f}, max {out['max_weight']})")
    ax[2].hist(rate, bins=np.linspace(0, 1, 51), color=C_VAL)
    ax[2].set_yscale("log")
    ax[2].set_xlabel(f"fraction of {min(HEALTH_IMAGES, len(x_held))} {held} images a clause fires on")
    ax[2].set_title(f"firing rate (never: {100 * out['frac_never_firing']:.1f} % of clauses)")
    mb = np.linspace(min(0.0, float(margin.min())), max(1.0, float(margin.max())), 41)
    ax[3].hist([margin[ok], margin[~ok]], bins=mb, stacked=True, color=[C_VAL, C_TRAIN], label=["correct", "wrong"])
    ax[3].set_xlabel(f"(top class sum - runner-up) / T, {held}")
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
    _save(fig, path)
    return out


def curves(curve, title, path):
    """Accuracies against epoch; with a validation set also the overfitting gap."""
    plt = _plt()
    e = [r["epoch"] for r in curve]
    early = "val_acc" in curve[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), facecolor="white")
    ax[0].plot(e, [r["train_acc_running"] for r in curve], color=C_RUN, linewidth=1.2, label="train, running")
    if early:
        ax[0].plot(e, [r["train_eval_acc"] for r in curve], color=C_TRAIN, linewidth=1.8,
                   label="train, 5 000 fixed images")
        ax[0].plot(e, [r["val_acc"] for r in curve], color=C_VAL, linewidth=1.8, label="validation")
        best = max(curve, key=lambda r: (r["val_acc"], -r["epoch"]))
        ax[0].plot([best["epoch"]], [best["val_acc"]], "o", color=C_VAL, markersize=5)
        ax[0].annotate(f"best {best['val_acc']:.2f} @ {best['epoch']}", (best["epoch"], best["val_acc"]),
                       textcoords="offset points", xytext=(-6, -13), ha="right", fontsize=8, color=C_VAL)
        ax[1].plot(e, [r["gap"] for r in curve], color=C_GAP, linewidth=1.8)
        ax[1].axhline(0, color=GRID, linewidth=1)
        ax[1].set_ylabel("train (5 000) - validation (pp)")
        ax[1].set_title("overfitting gap", fontsize=9)
    else:
        ax[0].plot(e, [r["test_acc"] for r in curve], color=C_VAL, linewidth=1.8, label="test (the paper's protocol)")
        ax[1].plot(e, [r["mean_clause_len"] for r in curve], color=C_GAP, linewidth=1.8)
        ax[1].set_ylabel("mean included literals per clause")
        ax[1].set_title("clause length", fontsize=9)
    ax[0].set_ylabel("accuracy (%)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        a.set_xlabel("epoch")
        a.xaxis.get_major_locator().set_params(integer=True)
        a.grid(color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


def rules(model, inp, title, path, top=3):
    """The ``top`` strongest clauses as rules, one line per layer. With message passing the
    clauses with message literals are preferred. A message literal ``right1:57`` means: clause 57
    held (up to the previous layer) at the node a right-going edge comes from. Hashed messages
    share inbox positions, so a decoded message can also be another clause with the same
    positions; positions that decode to no message are printed raw (``m1[k]``)."""
    plt = _plt()
    L = layer_lengths(model)
    wmax, cls = (t.cpu().numpy() for t in model.weights.abs().max(1))
    has_msg = L[:, 1:].sum(1) > 0 if L.shape[1] > 1 else np.zeros(len(L), dtype=bool)
    score = wmax + 1e9 * has_msg
    names = inp.feature_names() if hasattr(inp, "feature_names") else None
    lines = []
    for j in np.argsort(-score)[:top].tolist():
        w = int(model.weights[j, cls[j]])
        lines.append(f"clause {j}: strongest vote {'for' if w > 0 else 'against'} {CLASSES[cls[j]]} (weight {w}), "
                     f"literals per layer {L[j].tolist()}")
        for i in range(L.shape[1]):
            if hasattr(model, "depth"):
                e = model.clause_expression(j, feature_names=names, layers=[i])
            else:
                e = model.clause_expression(j)
            what = "own window" if i == 0 else f"inbox of layer {i}"
            lines += textwrap.wrap(f"  {what}: {e}", 150, subsequent_indent="      ")[:6]
        lines.append("")
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".txt").write_text(title + "\n\n" + text)
    fig = plt.figure(figsize=(13, 0.15 * len(lines) + 0.9), facecolor="white")
    fig.text(0.01, 0.98, title, fontsize=10, color=INK, va="top")
    fig.text(0.01, 0.98 - 0.6 / fig.get_figheight(), text, fontsize=7.5, color=INK, va="top", family="monospace")
    _save(fig, path)


@torch.no_grad()
def draw(model, inp, pan, x_held, sums, y_held, T, curve, title, folder, stem, held="val"):
    """All figures of one epoch -> ({W&B key: (png, caption)}, scalars). ``curve=None`` skips the
    curves. The model is put in eval mode for the drawing and restored."""
    was_training = model.training
    model.eval()
    try:
        kinds = ["panel", "health", "rules"] + (["curves"] if curve else [])
        files = {k: folder / f"{stem}_{k}.png" for k in kinds}
        acc = panel(model, inp, pan["x"], pan["raw"], pan["y"], title, files["panel"])
        scalars = health(model, inp, x_held, sums, y_held, T, title, files["health"], held)
        rules(model, inp, title, files["rules"])
        if curve:
            curves(curve, title, files["curves"])
    finally:
        if was_training:
            model.train()
    return {f"viz/{k}": (p, title) for k, p in files.items()}, dict(scalars, panel_acc=acc)


def main():
    from train import class_sums, load_data, load_model
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="models/<phase>/<tag>.pt")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    model, inp, st = load_model(a.model, dev)
    hp = st["hp"]
    paper = hp["kind"] != "grid"
    held = "test" if paper else "val"
    args = argparse.Namespace(kind=hp["kind"], subset=hp["data"]["subset"])
    _, _, sets, raw, n_full = load_data(args, dev)
    x, y = sets[held]
    sums = class_sums(model, inp, x, 256).float().cpu().numpy()
    idx = panel_indices(y, held, n_full)
    pan = {"x": x[torch.as_tensor(idx, device=dev)], "raw": raw[idx], "y": y[idx]}
    acc = 100 * float((sums.argmax(1) == y).mean())
    title = f"{st['tag']}  kept model of epoch {st['epoch']}  {held} {acc:.2f} %"
    _, sc = draw(model, inp, pan, x, sums, y, hp["T"], None, title, FIGURES / "saved" / st["tag"], "kept", held)
    print(f"{held} {acc:.2f} % (saved: {st.get('val_acc' if not paper else 'test_acc'):.2f} %), {sc}")
    print(f"figures: {FIGURES / 'saved' / st['tag']}")


if __name__ == "__main__":
    main()
