"""Explainable clauses on top of images (plan §10): where does each member, and the team, find
its evidence, and which clauses carry it?

Attribution (exact, as joint_wide_tm/code/activation_maps.py): a conv member's vote for class k
is ``v_k = sum_j w_j * OR_p match_j(p)``. Each firing clause's signed vote ``w_j`` is split
equally over the windows where it matches and each window's share over its pixels, so a member's
pixel map sums to its (unclamped) class vote, and the members' maps add up to the team score
(rule ``sum``). Both identities are asserted. A 3D member's window spans all bit planes, so its
map is spatial as well.

Panel figure (one row per image): image | team map | one map per member | the three strongest
matching clauses of the team for the predicted class (their match windows). Orange = votes for the
predicted class, blue = against.

During training ``train.py`` calls ``panel`` every 10 epochs and sends the figure to W&B. On a
saved team:  python code/viz.py models/grid/M_b32_full_s0.pt [--device cuda:0]
"""

import argparse
from itertools import combinations

import numpy as np
import torch
from common import CLASSES, FIGURES, load_cifar, panel_indices

FOR, AGAINST = np.array([0.90, 0.38, 0.05]), np.array([0.13, 0.42, 0.85])   # orange / blue


def _matches(model, x):
    """(b, Py, Px, C) bool patch matches in prediction mode, for 2D and (depth-1 grid) 3D members."""
    M = model.patch_clause_outputs(x, empty_value=False)
    if M.dim() == 5:                                   # (b, Pd, Ph, Pw, C): OR over depth positions
        M = M.any(dim=1)
    return M


@torch.no_grad()
def evidence(model, x, k, top=3, step=2):
    """x: member input (B, ...); k: (B,) class per image.

    Returns pixel maps (B, 32, 32), votes (B,), and per image the ``top`` strongest matching
    clauses of class k: [(signed weight, n literals, clause index, pixel map (32, 32))]."""
    ph, pw = model.patch_size[-2:]
    if model.n_clauses_total > 100_000:                # (step, P, C) patch matches: one image at a time
        step = 1
    ones = torch.ones(1, 1, ph, pw, device=x.device)
    w = model._signed_weights().view(model.n_classes, model.n_clauses)
    lens = model.include_count.view(model.n_classes, model.n_clauses)
    pix, votes, tops = [], [], []
    for i in range(0, x.shape[0], step):
        M = _matches(model, x[i:i + step])
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


def _overlay(ax, img, m, lim):
    v = np.clip(m / max(lim, 1e-9), -1, 1)
    rgb = np.where(v[..., None] >= 0, FOR, AGAINST)
    alpha = 0.85 * np.abs(v)[..., None]
    gray = img.astype(float).mean(axis=2, keepdims=True) / 255
    base = np.repeat(0.15 + 0.7 * gray, 3, axis=2)
    ax.imshow(base * (1 - alpha) + rgb * alpha, interpolation="nearest")


@torch.no_grad()
def panel(team, xs, raw, y, title, path, top=3):
    """Draw the panel for images ``raw`` (uint8 (N, 32, 32, 3)) with member inputs ``xs`` and
    labels ``y``; returns {"map_correlation", "panel_acc"}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    was_training = team.members[0].training
    sums = team.member_sums(xs)
    score = team.combine(sums)
    pred = score.argmax(1)
    team.eval()
    try:
        maps, tops = [], []
        for m, x, wgt, s in zip(team.members, xs, team.weights, sums):
            pm, v, t = evidence(m, x, pred, top)
            raw_vote = s.gather(1, pred.view(-1, 1)).squeeze(1).float()
            assert torch.allclose(v, raw_vote, atol=1e-2, rtol=1e-4), "member map does not sum to its vote"
            assert torch.allclose(pm.flatten(1).sum(1), raw_vote, atol=1.0, rtol=1e-3)
            maps.append((wgt * pm).cpu().numpy())
            tops.append([[(wgt * a, n, c, wgt * cm) for a, n, c, cm in ti] for ti in t])
    finally:
        if was_training:
            team.train()
    team_map = sum(maps)
    tscore = score.gather(1, pred.view(-1, 1)).squeeze(1).float().cpu().numpy()
    assert np.allclose(team_map.reshape(len(raw), -1).sum(1), tscore, atol=1.0, rtol=1e-3)
    pred = pred.cpu().numpy()
    N, R = len(raw), len(team.names)
    cols = 2 + R + top
    ink, muted = "#1f2328", "#59636e"
    fig, ax = plt.subplots(N, cols, figsize=(1.45 * cols + 0.6, 1.45 * N + 1.0), facecolor="white", squeeze=False)
    heads = ["image", "team"] + list(team.names) + [f"top clause {i + 1}" for i in range(top)]
    for i in range(N):
        best = sorted(((a, n, c, cm, team.names[r]) for r in range(R) for a, n, c, cm in tops[r][i]),
                      key=lambda t: -t[0])[:top]
        lim_m = max(float(np.abs(m[i]).max()) for m in maps)
        for c in range(cols):
            a = ax[i, c]
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#d0d7de")
            if i == 0:
                a.set_title(heads[c], fontsize=9, color=ink)
        ax[i, 0].imshow(raw[i], interpolation="nearest")
        ok = pred[i] == y[i]
        ax[i, 0].set_ylabel(f"{CLASSES[y[i]]}\n-> {CLASSES[pred[i]]}", fontsize=7.5,
                            color=ink if ok else "#b3261e")
        _overlay(ax[i, 1], raw[i], team_map[i], float(np.abs(team_map[i]).max()))
        for r in range(R):
            _overlay(ax[i, 2 + r], raw[i], maps[r][i], lim_m)
        for t in range(top):
            a = ax[i, 2 + R + t]
            if t < len(best):
                w, n, c, cm, name = best[t]
                _overlay(a, raw[i], cm, float(np.abs(cm).max()))
                a.set_xlabel(f"{name} #{c}  w {w:.0f}  {n} lit", fontsize=6.5, color=ink)
            else:
                a.text(0.5, 0.5, "none", ha="center", va="center", color=muted, transform=a.transAxes)
    fig.suptitle(title, fontsize=10, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.004, "Overlay = evidence for the PREDICTED class (orange = for, blue = against): each firing "
             "clause's vote split over its matching windows.\nMember columns share one colour scale per image and add "
             "up to the team map. Top clauses: strongest matching clauses (member, clause, weight, literals).\n"
             "Row label: true class -> prediction (red = wrong).", fontsize=7, color=muted, va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    cs = []
    for i in range(N):
        for a, b in combinations(range(R), 2):
            u, v = maps[a][i].ravel(), maps[b][i].ravel()
            if u.std() > 0 and v.std() > 0:
                cs.append(float(np.corrcoef(u, v)[0, 1]))
    return {"map_correlation": float(np.mean(cs)) if cs else None, "panel_acc": 100 * float((pred == y).mean())}


def main():
    from encoders import encode_fixed
    from team import load_team
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = torch.device(args.device)
    team, st = load_team(args.model, dev)
    hp = st["hp"]
    data = load_cifar(hp["data"]["subset"])
    encs = [hp["members"][n]["enc"] for n in team.names]
    fixed = {e: encode_fixed(e, data["test"][0], dev) for e in set(encs)}
    score = team.combine(team.member_sums([fixed[e] for e in encs])).argmax(1).cpu().numpy()
    acc = 100 * float((score == data["test"][1]).mean())
    if abs(acc - st["test_acc"]) > 1e-9:
        raise SystemExit(f"reloaded team gives {acc:.2f}, saved {st['test_acc']:.2f}")
    xv, yv = data["val"]
    idx = panel_indices(yv)
    xs = [encode_fixed(e, xv[idx], dev) for e in encs]
    out = FIGURES / "activation" / st["phase"] / f"{st['tag']}_best.png"
    r = panel(team, xs, xv[idx], yv[idx], f"{st['tag']}  best epoch {st['epoch']}  test {st['test_acc']:.2f} %", out)
    print(f"[{st['tag']}] epoch {st['epoch']}: test {acc:.2f} (saved {st['test_acc']:.2f}) OK | {out} | "
          f"map correlation {r['map_correlation']}")


if __name__ == "__main__":
    main()
