"""Train/validate one configuration and write an immutable run record.

    python code/run.py '<json config>' [--device cuda:0]
    python code/run.py --file configs/xyz.json --index 3

Config keys (defaults in ``DEFAULTS``): ``name`` (arm id), ``dataset``
(kvasir|acdc|scenes|shapes|longrange), ``size``, ``seed``, ``family``
(tm|coalesced|unet|localcnn|linear|majority|rowprior), family hyper-parameters, ``phase``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import sys
import time
import traceback
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines  # noqa: E402
import data  # noqa: E402
import features  # noqa: E402
import metrics as M  # noqa: E402
import record  # noqa: E402
import synth  # noqa: E402
import tm  # noqa: E402

import torchtsetlin as tt  # noqa: E402

IGNORE = 255
K_OF = {"kvasir": 2, "acdc": 4, "scenes": 4, "shapes": 3, "longrange": 3}
DEFAULTS = {
    "family": "tm", "size": 128, "seed": 0, "phase": "E0", "epochs": 10, "batch": 8,
    # TM
    "spec": {"bits": 4}, "patch": 3, "clauses": 300, "T_ratio": 0.2, "s": 10.0, "max_lits": None,
    "q": 128, "order": "perm", "sample": None, "class_fb": None, "weighted": True, "n_states": 100,
    "position": False, "train_frac": 1.0,
    # CNN
    "input": "raw", "width": 16, "lr": 1e-3, "wd": 1e-4, "dice_w": 0.0, "max_time": 900,
    "save_votes": False, "native_eval": False,
}


def load_data(cfg, split):
    ds = cfg["dataset"]
    if ds in ("kvasir", "acdc"):
        d = data.load_split(ds, cfg["size"], split)
        if ds == "kvasir":
            man = data.kvasir_manifest()
            grp = {}
            for g in man["multi_member_groups"]:
                for s in g:
                    grp[s] = g[0]
            d["groups"] = [grp.get(s, s) for s in d["ids"]]
        else:
            d["ids"] = list(range(len(d["x"])))
            d["groups"] = [d["vols"][v]["patient"] for v in d["vol"].tolist()]
        return d
    d = synth.load(ds, split)
    d["groups"] = d["ids"]
    return d


def model_input(cfg, x, dev, extra=None):
    """uint8 batch -> model input (Boolean planes for TM / bool-CNN; floats for raw CNN).

    ``extra`` are precomputed Boolean planes (H3 coarse-vote features) appended last."""
    x = x.to(dev, non_blocking=False)
    if cfg["family"] in ("tm", "coalesced") or cfg["input"] == "bool":
        out = features.encode(x, cfg["spec"])
        if extra is not None:
            out = torch.cat([out, extra.to(dev).bool()], 1)
        return out if cfg["family"] in ("tm", "coalesced") else out.float()
    return x.float() / 255.0


# ------------------------------------------------------------------------------ TM
def build_tm(cfg, K, dev, n_planes):
    cls = tm.ExpCoalescedSegTM if cfg["family"] == "coalesced" else tm.ExpSegTM
    C = cfg["clauses"]
    T = cfg["T_ratio"] * (C if cfg["family"] == "tm" else C / K)
    kw = dict(
        T=float(T), s=float(cfg["s"]), patch_size=cfg["patch"], patches_per_commit=cfg["q"],
        n_states=cfg["n_states"], weighted=cfg["weighted"], ignore_index=IGNORE,
        max_included_literals=cfg["max_lits"], position_encoding=cfg["position"],
    )
    if cfg["family"] == "coalesced":
        kw.pop("weighted")  # coalesced clauses always carry integer per-class weights
    m = cls(K, n_clauses=C, **kw)
    if cfg.get("cross_image"):
        # lift the chunk budget so one chunk holds the whole image batch: commits can then mix
        # pixels of different images (LIBRARY_GAPS LG-001); literals of the batch stay resident.
        m.max_chunk_elements = 2**40
        m.gather = bool(cfg.get("gather", True))
    m = m.to(dev)
    m.init_order(cfg["order"], tuple(cfg["sample"]) if cfg["sample"] else None, seed=cfg["seed"] + 7919)
    return m


@torch.no_grad()
def predict_votes(cfg, model, d, dev, bs=16):
    """Score maps ``(N, K, S, S)`` float16 on CPU (TM: vote sums; CNN: logits)."""
    if cfg["family"] in ("tm", "coalesced"):
        bs = 4  # (bs*P, 2F) literals: 16 images of wide features are > 1 GB
    out = []
    for i in range(0, len(d["x"]), bs):
        ex = d["extra"][i : i + bs] if d.get("extra") is not None else None
        xb = model_input(cfg, d["x"][i : i + bs], dev, ex)
        if cfg["family"] in ("tm", "coalesced"):
            v = model.vote_map(xb)
        else:
            v = model(xb)
        out.append(v.float().cpu().half())
    return torch.cat(out)


def eval_screen(scores, d, K, boundaries=True):
    ev = M.SegEvaluator(K, boundaries=boundaries)
    pred = scores.float().argmax(1)
    for i in range(len(pred)):
        ev.add(pred[i], d["y"][i], d["ids"][i])
    st = ev.stats()
    st["groups"] = d["groups"]
    return ev.summary(), st


def eval_native(cfg, scores, d, split):
    """Restore score maps to the native grid, argmax there, and evaluate (PLAN section 6)."""
    K = K_OF[cfg["dataset"]]
    ev = M.SegEvaluator(K, tols=(1, 2, "n"))
    S = cfg["size"]

    def ntol(g):  # 2 model-grid pixels expressed in native pixels
        return max(1, round(2 * max(int(g[0]), int(g[1])) / S))

    if cfg["dataset"] == "kvasir":
        for i, stem in enumerate(d["ids"]):
            s = data.restore_scores(scores[i].float(), d["geom"][i])
            ev.add(s.argmax(0), torch.from_numpy(data.load_kvasir_mask(stem)), stem, ntol(d["geom"][i]))
        st = ev.stats()
        st["groups"] = d["groups"]
        return {"summary": ev.summary(), "stats": st}
    # ACDC: per-slice native maps -> volumes
    gts = data.acdc_native_masks(split)
    vols = d["vols"]
    preds = [np.zeros(v["shape"], np.uint8) for v in vols]
    for i in range(len(scores)):
        s = data.restore_scores(scores[i].float(), d["geom"][i]).argmax(0).numpy().astype(np.uint8)
        vi, z = int(d["vol"][i]), int(d["slice"][i])
        preds[vi][:, :, z] = s
    vol_metrics = []
    for vi, v in enumerate(vols):
        for z in range(v["shape"][2]):
            ev.add(preds[vi][:, :, z], gts[vi][:, :, z], f"{v['patient']}_{v['phase']}_{z}", ntol(v["shape"]))
        vm = M.acdc_volume_metrics(preds[vi], gts[vi], v["spacing"])
        vol_metrics.append({"patient": v["patient"], "phase": v["phase"], "group": v["group"], **vm})
    st = ev.stats()
    st["groups"] = [i.split("_")[0] for i in st["ids"]]
    summ = ev.summary()
    for key in ("dice_RV", "dice_MYO", "dice_LV", "hd95_RV", "hd95_MYO", "hd95_LV"):
        vals = [r[key] for r in vol_metrics if r[key] is not None]
        summ[f"vol_{key}"] = float(np.mean(vals)) if vals else None
        summ[f"vol_{key}_undefined"] = sum(r[key] is None for r in vol_metrics)
    fg = [np.mean([r[k] for k in ("dice_RV", "dice_MYO", "dice_LV") if r[k] is not None]) for r in vol_metrics]
    summ["vol_fg_dice"] = float(np.mean(fg))
    return {"summary": summ, "stats": st, "volumes": vol_metrics}


def fit_tm(cfg, dev, tr, va, K, rec):
    gen = torch.Generator().manual_seed(cfg["seed"] + 104729)
    tt.seed_everything(cfg["seed"])
    n_pl = features.n_planes(cfg["spec"], tr["x"].shape[1]) + (0 if tr.get("extra") is None else tr["extra"].shape[1])
    model = build_tm(cfg, K, dev, n_pl)
    if cfg.get("class_fb") == "balanced":
        model.set_class_feedback_p(tt.data.balanced_class_probabilities(tr["y"].long(), K, ignore_index=IGNORE, floor=0.05).to(dev))
    best, best_state, best_scores, hist = -1.0, None, None, []
    ntr = len(tr["x"])
    t_train = 0.0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(ntr, generator=gen)
        if cfg["train_frac"] < 1.0:
            perm = perm[: int(round(cfg["train_frac"] * ntr))]
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        t0 = time.time()
        for i in range(0, len(perm), cfg["batch"]):
            idx = perm[i : i + cfg["batch"]]
            ex = tr["extra"][idx] if tr.get("extra") is not None else None
            model.update(model_input(cfg, tr["x"][idx], dev, ex), tr["y"][idx].to(dev).long())
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        t_ep = time.time() - t0
        t_train += t_ep
        model.eval()
        t1 = time.time()
        scores = predict_votes(cfg, model, va, dev)
        t_eval = time.time() - t1
        summ, _ = eval_screen(scores, va, K, boundaries=False)
        h = {"epoch": ep + 1, "screen_miou": summ["miou"], "screen_iou": summ["iou"], "train_s": t_ep, "eval_s": t_eval,
             "pixels_fed": model.counts["pixels_fed"], "commits": model.counts["commits"],
             "type_i": float(model.counts["type_i"]), "type_ii": float(model.counts["type_ii"]),
             **tm.state_stats(model)}
        hist.append(h)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k != "screen_iou"}), flush=True)
        if summ["miou"] is not None and summ["miou"] > best:  # strict > : earliest epoch wins ties
            best, best_scores = summ["miou"], scores
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rec["selected_epoch"] = ep + 1
    rec["history"] = hist
    rec["train_time_s"] = t_train
    rec["model_info"] = {"n_features": int(model.n_features), "n_planes": n_pl, "n_clauses_total": int(model.n_clauses_total),
                         "automaton_states": int(model.ta_state.numel()), "state_dtype": str(model.ta_state.dtype),
                         "T": float(model.T)}
    return best_state, best_scores, model


def fit_cnn(cfg, dev, tr, va, K, rec):
    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 104729)
    in_ch = model_input(cfg, tr["x"][:1], "cpu").shape[1]
    fam = cfg["family"]
    if fam == "unet":
        net = baselines.UNet(in_ch, K, cfg["width"])
    elif fam == "localcnn":
        net = baselines.LocalCNN(in_ch, K, 64, cfg.get("rf", 3))
    else:
        net = baselines.PixelLinear(in_ch, K)
    net = net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best, best_state, best_scores, hist = -1.0, None, None, []
    t_train, ntr = 0.0, len(tr["x"])
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(ntr, generator=gen)
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        t0 = time.time()
        for i in range(0, ntr, cfg["batch"]):
            idx = perm[i : i + cfg["batch"]]
            logits = net(model_input(cfg, tr["x"][idx], dev))
            loss = baselines.masked_loss(logits, tr["y"][idx].to(dev).long(), K, cfg["dice_w"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if dev.startswith("cuda"):
            torch.cuda.synchronize(dev)
        t_ep = time.time() - t0
        t_train += t_ep
        net.eval()
        scores = predict_votes(cfg, net, va, dev)
        summ, _ = eval_screen(scores, va, K, boundaries=False)
        hist.append({"epoch": ep + 1, "screen_miou": summ["miou"], "train_s": t_ep, "loss": float(loss.detach())})
        if summ["miou"] is not None and summ["miou"] > best:
            best, best_scores = summ["miou"], scores
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            rec["selected_epoch"] = ep + 1
        if t_train > cfg["max_time"]:
            rec["stopped"] = f"max_time after epoch {ep + 1}"
            break
    rec["history"] = hist
    rec["train_time_s"] = t_train
    rec["model_info"] = {"params": sum(p.numel() for p in net.parameters()), "in_ch": in_ch}
    return best_state, best_scores, net


def fit_trivial(cfg, tr, va, K, rec):
    fam = cfg["family"]
    y = tr["y"].long()
    if fam == "majority":
        cnt = torch.bincount(y[y != IGNORE], minlength=K)
        s = torch.zeros(len(va["x"]), K, *va["y"].shape[1:])
        s[:, int(cnt.argmax())] = 1
    else:  # rowprior: most common class per row of the training label maps
        S = y.shape[1]
        oh = torch.zeros(S, K)
        for k in range(K):
            oh[:, k] = ((y == k).sum((0, 2))).float()
        s = (oh / oh.sum(1, keepdim=True).clamp_min(1)).t().view(1, K, S, 1).expand(len(va["x"]), K, S, S).contiguous()
    rec["train_time_s"] = 0.0
    return None, s.half(), None


def run(cfg: Dict, dev: str) -> Dict:
    cfg = {**copy.deepcopy(DEFAULTS), **cfg}
    K = K_OF[cfg["dataset"]]
    rid = cfg.get("run_id") or f"{cfg['phase']}_{cfg['dataset']}_{cfg['name']}_s{cfg['seed']}"
    cfg["run_id"] = rid
    rec = {"run_id": rid, "config": cfg, "argv": sys.argv, "status": "running", "env": record.environment(dev),
           "manifest_sha256": record.manifest_hash(cfg["dataset"]), "seeds": {"init/global": cfg["seed"],
           "image_order": cfg["seed"] + 104729, "pixel_order/sampler": cfg["seed"] + 7919}}
    if os.path.exists(os.path.join(record.RESULTS, f"{rid}.json")):
        print("exists", rid)
        return {}
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    try:
        tr, va = load_data(cfg, "train"), load_data(cfg, "val")
        if cfg.get("stack"):
            import hierarchy

            tr, va = hierarchy.add_coarse_features(cfg, dev, tr, va, K, rec)
        if cfg["family"] in ("tm", "coalesced"):
            state, scores, model = fit_tm(cfg, dev, tr, va, K, rec)
        elif cfg["family"] in ("unet", "localcnn", "linear"):
            state, scores, model = fit_cnn(cfg, dev, tr, va, K, rec)
        else:
            state, scores, model = fit_trivial(cfg, tr, va, K, rec)
        summ, st = eval_screen(scores, va, K)
        rec["val_screen"] = summ
        rec["val_screen_stats"] = st
        if cfg["native_eval"] and cfg["dataset"] in ("kvasir", "acdc"):
            nat = eval_native(cfg, scores, va, "val")
            rec["val_native"] = nat["summary"]
            rec["val_native_stats"] = nat["stats"]
            if "volumes" in nat:
                rec["val_volumes"] = nat["volumes"]
        if state is not None:
            os.makedirs(os.path.join(record.ARTIFACTS, "ckpt"), exist_ok=True)
            p = os.path.join(record.ARTIFACTS, "ckpt", f"{rid}.pt")
            torch.save(state, p)
            rec["checkpoint"] = {"path": os.path.relpath(p, record.HERE), "bytes": os.path.getsize(p)}
        if cfg["save_votes"]:
            os.makedirs(os.path.join(record.ARTIFACTS, "votes"), exist_ok=True)
            p = os.path.join(record.ARTIFACTS, "votes", f"{rid}_val.pt")
            torch.save(scores, p)
            rec["val_votes"] = os.path.relpath(p, record.HERE)
        rec["status"] = "completed"
    except torch.cuda.OutOfMemoryError as e:  # recorded outcome, not a missing row
        rec["status"], rec["failure"] = "oom", str(e)[:500]
    except RuntimeError as e:  # torch.AcceleratorError "CUDA error: out of memory" (allocation in .to())
        oom = "out of memory" in str(e)
        rec["status"] = "oom" if oom else "failed"
        rec["failure"] = str(e)[:500] if oom else traceback.format_exc()[-2000:]
    except Exception as e:
        rec["status"], rec["failure"] = "failed", traceback.format_exc()[-2000:]
    rec["wall_s"] = time.time() - t0
    rec["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if dev.startswith("cuda"):
        rec["gpu_peak_alloc_mb"] = torch.cuda.max_memory_allocated(dev) / 2**20
        rec["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(dev) / 2**20
    path = record.write_record(rec)
    print(rec["status"], rid, rec.get("val_screen", {}).get("miou"), path, flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--index", type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.file:
        with open(a.file) as f:
            cfgs = json.load(f)
        cfg = cfgs[a.index]
    else:
        cfg = json.loads(a.config)
    run(cfg, a.device)
