"""H3: supervised coarse-to-fine logical decoder with out-of-fold coarse vote features.

``cfg["stack"]`` keys: ``factor`` (4), ``coarse`` (dict of TM settings: spec, patch, clauses,
T_ratio, s, epochs, q), ``taus`` ([-0.5, 0, 0.5]), ``upsample`` (nearest|bilinear), ``folds``
(3), ``mode`` (oof | shuffled | oracle | insample), ``targets`` (majority | center).

The coarse stage is trained with its own ordinary supervised TM feedback for a *fixed*
number of epochs (no checkpoint selection on validation), then frozen. Training-set coarse
features are out-of-fold: each fold model never saw the images it predicts (folds by group:
duplicate group for Kvasir, patient for ACDC). Validation features come from a deployment
coarse model trained on all training images. ``oracle`` (ground-truth coarse maps) is a
labelled diagnostic, never a competitive arm.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F

import features
import metrics as M
import tm
import torchtsetlin as tt

IGNORE = 255


def downsample_images(x: torch.Tensor, f: int) -> torch.Tensor:
    return F.avg_pool2d(x.float(), f, ceil_mode=True).round().clamp(0, 255).to(torch.uint8)


def coarse_labels(y: torch.Tensor, f: int, K: int, mode: str = "majority") -> torch.Tensor:
    """Majority valid label per ``f x f`` cell (all-void -> IGNORE); ``center`` takes the
    cell-centre pixel's label."""
    if mode == "center":
        return y[:, f // 2 :: f, f // 2 :: f].clone()
    valid = y != IGNORE
    yy = torch.where(valid, y, torch.zeros_like(y)).long()
    oh = F.one_hot(yy, K).permute(0, 3, 1, 2).float() * valid.unsqueeze(1).float()
    cnt = F.avg_pool2d(oh, f, ceil_mode=True)
    lab = cnt.argmax(1).to(torch.uint8)
    lab[cnt.sum(1) == 0] = IGNORE
    return lab


def train_coarse(cc: Dict, x: torch.Tensor, y: torch.Tensor, K: int, dev: str, seed: int):
    tt.seed_everything(seed)
    C = cc["clauses"]
    m = tm.ExpSegTM(K, n_clauses=C, T=float(cc["T_ratio"] * C), s=float(cc["s"]), patch_size=cc["patch"],
                    patches_per_commit=cc["q"], n_states=100, weighted=True, ignore_index=IGNORE).to(dev)
    m.init_order("perm", tuple(cc["sample"]) if cc.get("sample") else None, seed=seed + 7919)
    if cc.get("sample"):
        m.max_chunk_elements, m.gather = 2**40, True
    gen = torch.Generator().manual_seed(seed + 104729)
    for _ in range(cc["epochs"]):
        m.train()
        perm = torch.randperm(len(x), generator=gen)
        bs = cc.get("batch", 8)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            m.update(features.encode(x[idx].to(dev), cc["spec"]), y[idx].to(dev).long())
    m.eval()
    return m


@torch.no_grad()
def coarse_votes(m, cc, x, dev, bs=64):
    out = []
    for i in range(0, len(x), bs):
        out.append(m.vote_map(features.encode(x[i : i + bs].to(dev), cc["spec"])).float().cpu())
    return torch.cat(out)


def vote_planes(votes: torch.Tensor, T: float, taus, size, upsample: str) -> torch.Tensor:
    """Normalised votes ``v/T`` thresholded at each tau -> Boolean planes on the fine grid
    (class-major: all taus of class 0, then class 1, ...)."""
    vn = votes / T
    if upsample == "bilinear":
        vn = F.interpolate(vn, size=size, mode="bilinear", align_corners=False)
    else:
        vn = F.interpolate(vn, size=size, mode="nearest")
    planes = torch.stack([vn >= t for t in taus], 2)  # (N, K, nt, H, W)
    N, K, nt, H, W = planes.shape
    return planes.reshape(N, K * nt, H, W)


def _folds(groups, n_folds, seed):
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[g]: int(i % n_folds) for i, g in enumerate(perm)}
    return np.asarray([fold_of[g] for g in groups])


def deploy_features(cfg, rec, d, K, dev):
    """Coarse vote planes for a new split from the saved deployment coarse model."""
    import os

    import record

    st = cfg["stack"]
    f, cc, taus = st.get("factor", 4), st["coarse"], st.get("taus", [-0.5, 0.0, 0.5])
    C = cc["clauses"]
    m = tm.ExpSegTM(K, n_clauses=C, T=float(cc["T_ratio"] * C), s=float(cc["s"]), patch_size=cc["patch"],
                    patches_per_commit=cc["q"], n_states=100, weighted=True, ignore_index=IGNORE).to(dev)
    xc = downsample_images(d["x"], f)
    m._prepare(features.encode(xc[:1].to(dev), cc["spec"]))
    m.load_state_dict(torch.load(os.path.join(record.HERE, rec["stack_info"]["coarse_checkpoint"])))
    m.eval()
    v = coarse_votes(m, cc, xc, dev)
    return vote_planes(v, float(cc["T_ratio"] * C), taus, tuple(d["x"].shape[-2:]), st.get("upsample", "nearest"))


def add_coarse_features(cfg, dev, tr, va, K, rec):
    st = cfg["stack"]
    f, cc, taus = st.get("factor", 4), st["coarse"], st.get("taus", [-0.5, 0.0, 0.5])
    up, mode = st.get("upsample", "nearest"), st.get("mode", "oof")
    size = tuple(tr["x"].shape[-2:])
    t0 = time.time()
    xc_tr, xc_va = downsample_images(tr["x"], f), downsample_images(va["x"], f)
    yc_tr = coarse_labels(tr["y"], f, K, st.get("targets", "majority"))
    yc_va = coarse_labels(va["y"], f, K, "majority")
    info: Dict = {"factor": f, "mode": mode, "taus": taus, "upsample": up}
    Tc = float(cc["T_ratio"] * cc["clauses"])
    if mode == "oracle":
        def oh(yc):
            yy = torch.where(yc == IGNORE, torch.zeros_like(yc), yc).long()
            return F.interpolate(F.one_hot(yy, K).permute(0, 3, 1, 2).float(), size=size, mode="nearest") > 0.5
        ex_tr, ex_va = oh(yc_tr), oh(yc_va)
    else:
        seed = cfg["seed"] * 1000 + 17
        dep = train_coarse(cc, xc_tr, yc_tr, K, dev, seed)
        if rec.get("run_id"):  # the deployment coarse model is needed to evaluate new splits
            import os

            import record

            p = os.path.join(record.ARTIFACTS, "ckpt", f"{rec['run_id']}_coarse.pt")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            torch.save({k: v.cpu() for k, v in dep.state_dict().items()}, p)
            info["coarse_checkpoint"] = os.path.relpath(p, record.HERE)
        v_va = coarse_votes(dep, cc, xc_va, dev)
        if mode == "insample":
            v_tr = coarse_votes(dep, cc, xc_tr, dev)
        else:
            fold = _folds(tr["groups"], st.get("folds", 3), seed)
            v_tr = torch.zeros(len(xc_tr), K, *xc_tr.shape[-2:])
            fold_m = []
            for k in range(st.get("folds", 3)):
                tr_idx = torch.from_numpy(np.where(fold != k)[0])
                ho_idx = torch.from_numpy(np.where(fold == k)[0])
                mk = train_coarse(cc, xc_tr[tr_idx], yc_tr[tr_idx], K, dev, seed + k + 1)
                v_tr[ho_idx] = coarse_votes(mk, cc, xc_tr[ho_idx], dev)
                ev = M.SegEvaluator(K, boundaries=False)
                for i, j in enumerate(ho_idx.tolist()):
                    ev.add(v_tr[j].argmax(0), yc_tr[j])
                fold_m.append(ev.summary()["miou"])
            info["fold_heldout_coarse_miou"] = fold_m
            info["fold_sizes"] = [int((fold == k).sum()) for k in range(st.get("folds", 3))]
        ev_tr, ev_va = M.SegEvaluator(K, boundaries=False), M.SegEvaluator(K, boundaries=False)
        for i in range(len(v_tr)):
            ev_tr.add(v_tr[i].argmax(0), yc_tr[i])
        for i in range(len(v_va)):
            ev_va.add(v_va[i].argmax(0), yc_va[i])
        info["train_feature_coarse_miou"] = ev_tr.summary()["miou"]
        info["val_deploy_coarse_miou"] = ev_va.summary()["miou"]
        # coarse-only control: upsampled coarse votes evaluated on the fine validation grid
        evf = M.SegEvaluator(K, boundaries=False)
        upv = F.interpolate(v_va, size=size, mode="bilinear", align_corners=False)
        for i in range(len(upv)):
            evf.add(upv[i].argmax(0), va["y"][i])
        info["coarse_only_val_fine_miou"] = evf.summary()["miou"]
        ex_tr = vote_planes(v_tr, Tc, taus, size, up)
        ex_va = vote_planes(v_va, Tc, taus, size, up)
        info["bit_rate_train_oof"] = float(ex_tr.float().mean())
        info["bit_rate_val_deploy"] = float(ex_va.float().mean())
        info["coarse_automaton_states"] = int(dep.ta_state.numel())
        if mode == "shuffled":
            g = torch.Generator().manual_seed(seed + 99)
            ex_tr = ex_tr[torch.randperm(len(ex_tr), generator=g)]
            ex_va = ex_va[torch.randperm(len(ex_va), generator=g)]
    info["coarse_stage_s"] = time.time() - t0
    rec["stack_info"] = info
    print("stack", {k: v for k, v in info.items() if not isinstance(v, list)}, flush=True)
    tr, va = dict(tr), dict(va)
    tr["extra"], va["extra"] = ex_tr, ex_va
    return tr, va
