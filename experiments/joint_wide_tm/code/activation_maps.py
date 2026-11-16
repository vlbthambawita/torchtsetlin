"""Where in the image does each member of a saved team find its evidence? (plan B6, phase 3)

Attribution (exact, as fmnist_bitplanes/code/activation_maps.py): a conv member's vote for class k
is ``v_k = sum_j w_j * OR_p match_j(p)``. Each firing clause's signed vote w_j is split equally
over the windows where it matches and each window's share over its pixels, so a member's pixel map
sums to its (unclamped) class vote. The team map is the members' maps added with the team's
combining weights; it sums to the team score. Both identities are asserted for every image.

For one correctly and one wrongly classified test image per class (the team's prediction), draws
one row per member and one for the team: figures/activation/<phase>/<arm>/<tag>_{correct,wrong}.png.
``--indices-from`` reuses the images of another model's record (joint team vs composite on the
same images). Also reports how alike the members' maps are: the mean pairwise Pearson correlation
of their pixel maps (predicted class) over ``--corr-n`` test images -> results/<phase>/activation/.

Usage: python code/activation_maps.py models/main/J/S_sum_seed0.pt [--indices-from <record.json>]
"""

import argparse
import os
from itertools import combinations

import numpy as np
import torch
from common import CACHE, FIGURES, PROG, RESULTS, load_all, new_record, read_json, write_json
from infer import load_team

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
FOR, AGAINST = np.array([0.90, 0.38, 0.05]), np.array([0.13, 0.42, 0.85])   # orange / blue


@torch.no_grad()
def evidence_maps(model, x, k, step=2):
    """x: (B, planes, 28, 28); k: (B,) class per image -> (B, 28, 28) pixel maps, (B,) votes."""
    ph, pw = model.patch_size
    w = model._signed_weights().view(model.n_classes, model.n_clauses)
    pix, votes = [], []
    for i in range(0, x.shape[0], step):
        M = model.patch_clause_outputs(x[i:i + step], empty_value=False)     # (b, Py, Px, C)
        b, Py, Px, _ = M.shape
        maps = torch.zeros(b, Py, Px, device=x.device)
        for j in range(b):
            kk = int(k[i + j])
            m = M[j].reshape(Py * Px, model.n_classes, model.n_clauses)[:, kk].float()   # (P, n)
            n = m.sum(0)                                                     # windows per clause
            share = torch.where(n > 0, w[kk] / n.clamp(min=1), torch.zeros_like(n))
            maps[j] = (m @ share).view(Py, Px)
            votes.append((w[kk] * (n > 0).float()).sum())
        pix.append(torch.nn.functional.conv_transpose2d(maps[:, None], torch.ones(1, 1, ph, pw, device=x.device))[:, 0]
                   / (ph * pw))
    return torch.cat(pix), torch.stack(votes)


def pick(y, pred):
    right, wrong = [], []
    for k in range(10):
        r, w = np.flatnonzero((y == k) & (pred == k)), np.flatnonzero((y == k) & (pred != k))
        right.append(int(r[0]) if len(r) else None)
        wrong.append(int(w[0]) if len(w) else None)
    return right, wrong


def draw(raw, rows, idx, pred, title, path):
    """rows: [(label, {image index: map})]; the member rows of one image share a colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#1f2328", "#59636e"
    R = len(rows)
    fig, ax = plt.subplots(R, 10, figsize=(16, 1.75 * R + 1.1), facecolor="white", squeeze=False)
    for k in range(10):
        i = idx[k]
        lim_m = max([1e-9] + [float(np.abs(m[i]).max()) for _, m in rows[:-1]]) if i is not None else 1.0
        for r, (label, maps) in enumerate(rows):
            a = ax[r, k]
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_color("#d0d7de")
            if k == 0:
                a.set_ylabel(label, fontsize=9, color=ink)
            if r == 0:
                a.set_title(CLASSES[k], fontsize=10, color=ink)
            if i is None:
                a.text(0.5, 0.5, "none", ha="center", va="center", color=muted, transform=a.transAxes)
                continue
            m = maps[i]
            lim = lim_m if r < R - 1 else max(1e-9, float(np.abs(m).max()))
            v = np.clip(m / lim, -1, 1)
            rgb = np.where(v[..., None] >= 0, FOR, AGAINST)
            alpha = 0.85 * np.abs(v)[..., None]
            base = np.repeat((0.15 + 0.7 * raw[i].astype(float) / 255)[..., None], 3, axis=2)
            a.imshow(base * (1 - alpha) + rgb * alpha, interpolation="nearest")
            if r == R - 1:
                a.set_xlabel(f"#{i}  pred: {CLASSES[pred[i]]}", fontsize=8, color=ink)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Overlay = evidence for the team's PREDICTED class (orange = votes for, blue = against): each "
             "firing clause's vote split over its matching windows. Member rows = weighted share of the team score, "
             "one colour scale per image; last row = their sum (own scale). Columns = true class.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--indices-from", default=None, help="an activation record whose images to reuse")
    ap.add_argument("--corr-n", type=int, default=1000, help="test images for the map correlation (0 = skip)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    team, st = load_team(args.model, dev)
    hp, tag, arm, phase = st["hp"], st["tag"], st["arm"], st["phase"]
    encs = [hp["members"][n]["enc"] for n in team.names]
    x, y, raw, _ = load_all(encs, dev, hp["data"]["subset"])
    xs, yt = [x[e]["test"] for e in encs], y["test"]
    sums = team.member_sums(xs)
    score = team.combine(sums)
    pred = score.argmax(1).cpu().numpy()
    acc = 100 * float((pred == yt).mean())
    if abs(acc - st["test_acc"]) > 1e-9:
        raise SystemExit(f"reloaded team gives {acc:.2f}, saved {st['test_acc']:.2f}")
    if args.indices_from:
        src = read_json(args.indices_from)
        right, wrong = src["correct_idx"], src["wrong_idx"]
    else:
        right, wrong = pick(yt, pred)
    chosen = sorted({i for i in right + wrong if i is not None})
    corr_idx = list(range(min(args.corr_n, len(yt))))
    todo = sorted(set(chosen) | set(corr_idx))
    sel = torch.tensor(todo, device=dev)
    k = torch.from_numpy(pred[todo]).to(dev)
    for m in team.members:
        m.eval()
    maps = []
    for m, xm, w, s in zip(team.members, xs, team.weights, sums):
        pix, votes = evidence_maps(m, xm[sel], k)
        own = s[sel, k].float()                       # the member's class sum from forward()
        assert torch.allclose(votes, own, atol=1e-3, rtol=1e-5), "clause attribution != member vote"
        assert torch.allclose(pix.sum((1, 2)), votes, atol=1e-2, rtol=1e-4), "pixel map != member vote"
        maps.append((w * pix).cpu().numpy())
    total = sum(maps)
    assert np.allclose(total.sum((1, 2)), score[sel, k].cpu().numpy(), atol=1e-2, rtol=1e-4), "team map != team score"
    pos = {i: j for j, i in enumerate(todo)}
    rows = [(n, {i: mp[pos[i]] for i in chosen}) for n, mp in zip(team.names, maps)]
    rows.append(("team", {i: total[pos[i]] for i in chosen}))
    title = f"{arm}/{tag}  ({phase}, best epoch {st['epoch']}, test {acc:.2f}%)"
    figs = {}
    for name, idx in (("correct", right), ("wrong", wrong)):
        figs[name] = FIGURES / "activation" / phase / arm / f"{tag}_{name}.png"
        draw(raw["test"], rows, idx, pred, f"{title} — {name}", figs[name])
    corr = {}
    if corr_idx and len(maps) > 1:                    # how alike are the members' maps?
        flat = [mp[[pos[i] for i in corr_idx]].reshape(len(corr_idx), -1) for mp in maps]
        for (a, fa), (b, fb) in combinations(zip(team.names, flat), 2):
            fa0, fb0 = fa - fa.mean(1, keepdims=True), fb - fb.mean(1, keepdims=True)
            r = (fa0 * fb0).sum(1) / np.maximum(1e-12, np.linalg.norm(fa0, axis=1) * np.linalg.norm(fb0, axis=1))
            corr[f"{a}-{b}"] = {"mean": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r)))}
        corr["mean_pairwise"] = float(np.mean([v["mean"] for v in corr.values()]))
    rec = new_record("activation", device=dev, tag=tag, arm=arm, phase=phase, model=str(args.model), epoch=st["epoch"],
                     test_acc=acc, figures={k: str(v.relative_to(PROG)) for k, v in figs.items()},
                     correct_idx=right, wrong_idx=wrong, indices_from=args.indices_from,
                     predicted={str(i): int(pred[i]) for i in chosen}, map_correlation=corr, corr_n=len(corr_idx),
                     attribution_checked=len(todo))
    write_json(rec, RESULTS / phase / "activation" / arm / f"{tag}.json")
    print(f"{arm}/{tag}: {figs['correct']} | map correlation {corr.get('mean_pairwise')}", flush=True)
    if not args.no_wandb and os.environ.get("JW_WANDB", "1") != "0" and phase != "smoke":
        try:
            import wandb
            from tracking import ENTITY, PROJECT, run_id
            run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(phase, arm, tag), name=f"jw-{arm}-{tag}",
                             group=f"jw-{arm}", tags=["joint_wide"], resume="allow", dir=str(CACHE),
                             settings=wandb.Settings(init_timeout=180, console="off"))
            run.log({f"activation_maps_{k}": wandb.Image(str(v), caption=title) for k, v in figs.items()})
            run.finish()
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] activation map upload failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
