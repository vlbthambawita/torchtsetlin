"""CNN baselines (PLAN_KVASIR §10): ``python baselines.py --tag <tag>``.

    mrcnn_i    (i)    torchvision Mask R-CNN R50-FPN, COCO weights, fine-tuned
    mrcnn_iis  (ii-s) Mask R-CNN ResNet-18-FPN from scratch
    mrcnn_iii  (iii)  Mask R-CNN R50-FPN from scratch on the T4 Boolean planes (12 input channels)
    unet       (v-b)  the segmentation programme's compact U-Net, connected components as instances

Same split, augmentation views, evaluator, early stopping (patience 75 on validation mask AP50,
earliest epoch on ties), checkpoints, W&B run and prediction panels as the Tsetlin heads. The
Mask R-CNN inputs are the 128 px images up-sampled 4x to 512 px so that the stock anchors fit;
predictions are brought back to the 128 px grid before scoring. The learning rate is constant
after a linear warm-up (early stopping decides the length); ``schedule="steps"`` is the original
fixed schedule of MR B.3 (3 000 iterations, x0.1 at 67 % and 89 %), run once as a check.
"""

import argparse
import os
import sys
import time

import boolenc as E
import jobs as JOBS
import torch
import torch.nn as nn
import torch.nn.functional as TF
from augment import make_view
from common import (
    DEVICE,
    IGNORE,
    MAX_EPOCHS,
    PANEL_EVERY,
    PATIENCE,
    S,
    env_info,
    model_dir,
    read_json,
    record_path,
    set_seed,
    stop_requested,
    write_json,
)
from data import Data
from detection import rois as R
from pipeline import MIN_PIXELS, evaluate
from tracking import Tracker, panel_path, render

UP = 4
BATCH = {"mrcnn_i": 8, "mrcnn_iis": 8, "mrcnn_iii": 8, "unet": 16}
INTERRUPTED = 143


class UNet(nn.Module):
    """Compact U-Net of ``experiments/segmentation/code/baselines.py`` (width 16, 4 levels)."""

    def __init__(self, in_ch=3, K=2, width=16, levels=4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        ch = [width * 2 ** i for i in range(levels)]
        self.down = nn.ModuleList([block(in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(levels - 1, 0, -1)])
        self.dec = nn.ModuleList([block(2 * ch[i - 1], ch[i - 1]) for i in range(levels - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], K, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = TF.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


def build(model):
    from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    size = S * UP
    if model == "unet":
        return UNet()
    if model == "mrcnn_i":
        m = maskrcnn_resnet50_fpn(weights="COCO_V1", min_size=size, max_size=size)
        m.roi_heads.box_predictor = FastRCNNPredictor(1024, 2)
        m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
        return m
    if model == "mrcnn_iis":
        bb = resnet_fpn_backbone(backbone_name="resnet18", weights=None, trainable_layers=5)
        return MaskRCNN(bb, num_classes=2, min_size=size, max_size=size)
    if model == "mrcnn_iii":
        z = E.Z["T4"]
        m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2, min_size=size, max_size=size,
                                  image_mean=[0.0] * z, image_std=[1.0] * z, trainable_backbone_layers=5)
        m.backbone.body.conv1 = nn.Conv2d(z, 64, 7, stride=2, padding=3, bias=False)
        return m
    raise ValueError(model)


def inputs(model, x):
    """The network input of a batch of uint8 images ``(B, 3, S, S)``."""
    if model == "unet":
        return x.float() / 255.0
    f = E.planes(x, "T4").float() if model == "mrcnn_iii" else x.float() / 255.0
    return TF.interpolate(f, scale_factor=UP, mode="nearest" if model == "mrcnn_iii" else "bilinear")


def targets(view, idx):
    out = []
    for i in idx:
        b = view.boxes[i]
        ok = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1) if len(b) else torch.zeros(0, dtype=torch.bool)
        masks = torch.stack([view.inst[i] == j + 1 for j in range(len(b))]) if len(b) else torch.zeros(0, S, S, dtype=torch.bool, device=view.x.device)
        masks = masks[ok] if len(b) else masks
        up = masks.repeat_interleave(UP, 1).repeat_interleave(UP, 2).to(torch.uint8)
        out.append({"boxes": (b[ok] if len(b) else b.reshape(0, 4)) * UP, "labels": torch.ones(int(ok.sum()) if len(b) else 0, dtype=torch.long, device=view.x.device),
                    "masks": up})
    return out


@torch.no_grad()
def detect(model, net, x, inst, batch=8):
    """Detections on the 128 px grid, in the format of ``pipeline.Pipeline.detect``."""
    net.eval()
    out = []
    for i in range(0, x.shape[0], batch):
        xb, ib = x[i:i + batch], inst[i:i + batch]
        if model == "unet":
            import numpy as np
            from scipy import ndimage
            lg = net(inputs(model, xb))
            dm = lg[:, 1] - lg[:, 0]
            sem = (dm > 0) & (ib != IGNORE)
            for j in range(xb.shape[0]):
                lab, n = ndimage.label(sem[j].cpu().numpy(), structure=np.ones((3, 3)))
                lab = torch.from_numpy(lab).to(x.device)
                masks = [lab == k for k in range(1, n + 1) if int((lab == k).sum()) >= MIN_PIXELS]
                sc = torch.stack([dm[j][m].mean() for m in masks]) if masks else torch.zeros(0, device=x.device)
                masks = torch.stack(masks) if masks else torch.zeros(0, S, S, dtype=torch.bool, device=x.device)
                out.append({"boxes": R.mask_boxes(masks), "masks": masks, "cls": sc, "obj": torch.zeros_like(sc),
                            "mmask": sc.clone(), "semantic": sem[j]})
            continue
        with torch.autocast("cuda", enabled=x.is_cuda):
            res = net(list(inputs(model, xb)))
        for j, r in enumerate(res):
            pm = TF.avg_pool2d(r["masks"].float(), UP)[:, 0] > 0.5 if len(r["masks"]) else torch.zeros(0, S, S, dtype=torch.bool, device=x.device)
            pm = pm & (ib[j] != IGNORE)
            sc = r["scores"].float()
            out.append({"boxes": r["boxes"].float() / UP, "masks": pm, "cls": sc, "obj": torch.zeros_like(sc),
                        "mmask": torch.zeros_like(sc)})
    return out


def detect_tag(job, data, dev, x, inst):
    """Detections of a finished baseline run (the sealed evaluation)."""
    net = build(job["model"]).to(dev)
    net.load_state_dict(torch.load(model_dir(job["tag"]) / "best.pt", map_location=dev, weights_only=False)["state"])
    return detect(job["model"], net, x, inst)


def panel_items(dets):
    items = []
    for d in dets:
        if "semantic" in d:
            items.append({"pred_mask": d["semantic"].cpu()})
            continue
        keep = d["cls"] >= 0.5
        if len(d["cls"]) and not bool(keep.any()):
            keep[d["cls"].argmax()] = True
        pm = d["masks"][keep].any(0) if bool(keep.any()) else torch.zeros(S, S, dtype=torch.bool)
        items.append({"pred_mask": pm.cpu(), "pred_boxes": d["boxes"][keep].cpu(), "scores": d["cls"][keep].cpu()})
    return items


def save(obj, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    job = JOBS.find(args.tag)
    assert job["kind"] == "baseline"
    tag, model, dev = job["tag"], job["model"], torch.device(args.device)
    rec_path, mdir = record_path(tag), model_dir(tag)
    if (read_json(rec_path) or {}).get("status") == "done":
        print(f"{tag}: already done")
        return 0
    mdir.mkdir(parents=True, exist_ok=True)
    set_seed(job["seed"])
    data = Data(dev)
    tr, va = data.image_sets(job.get("fold"))
    if job.get("subset"):
        tr, va = tr[: int(job["subset"])], va[: max(8, int(job["subset"]) // 4)]
    val_view = make_view(data, va, "none")
    net = build(model).to(dev)
    steps = job.get("schedule") == "steps"
    if model == "unet":
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    else:
        opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and model != "unet")
    bs, warm = BATCH[model], 500
    per_epoch = max(1, len(tr) // bs)
    max_epochs = job.get("max_epochs") or (-(-3000 // per_epoch) if steps else MAX_EPOCHS)
    patience = PATIENCE if not (job.get("max_epochs") or steps) else 10 ** 9
    st = {"epoch": 0, "it": 0, "best": None, "best_epoch": 0, "wait": 0, "history": [], "wall": 0.0, "last_panel": -10 ** 9,
          "best_metrics": None}
    last = mdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        st = ck["st"]
        torch.set_rng_state(ck["rng"].cpu())
        print(f"{tag}: resuming after epoch {st['epoch']}", flush=True)
    env = env_info(dev)
    tracker = Tracker(dict(job, head=model), {**{k: v for k, v in job.items() if k != "needs"}, **env, "patience": patience,
                                              "batch": bs, "input_px": S * UP if model != "unet" else S},
                      enabled=not args.no_wandb)

    def record(status, **extra):
        write_json({"tag": tag, "status": status, "job": job, "env": env, "best": st["best"], "best_epoch": st["best_epoch"],
                    "epochs": st["epoch"], "wall_s": st["wall"], "metrics": st["best_metrics"], "history": st["history"],
                    "wandb_url": tracker.url, **extra}, rec_path)

    def panels(epoch, final=False):
        out = []
        for split in ("test", "val"):
            x, inst, _ = data.panel(split)
            items = panel_items(detect(model, net, x, inst))
            p = render(x.cpu(), inst.cpu(), items, f"{tag}  {split}  epoch {epoch}", panel_path(tag, split, None if final else epoch))
            tracker.panel(epoch, split, x.cpu(), inst.cpu(), items, p)
            out.append(str(p))
        return out

    record("running")
    stopped_by = None
    while st["epoch"] < max_epochs:
        epoch = st["epoch"] + 1
        t0 = time.time()
        view = make_view(data, tr, job.get("aug", "sym"), job["seed"], epoch)
        perm = torch.randperm(len(view), generator=torch.Generator().manual_seed(13 * job["seed"] + epoch))
        net.train()
        tot, nb = 0.0, 0
        for b in range(0, len(perm) - bs + 1, bs):
            idx = perm[b:b + bs].tolist()
            lr = (1e-3 if model == "unet" else 0.01) * min(1.0, (st["it"] + 1) / warm)
            if steps:
                lr *= 0.1 ** ((st["it"] >= 0.67 * 3000) + (st["it"] >= 0.89 * 3000))
            for g in opt.param_groups:
                g["lr"] = lr
            xb = inputs(model, view.x[idx])
            if model == "unet":
                y = torch.where(view.inst[idx] == IGNORE, torch.full_like(view.inst[idx], IGNORE), (view.inst[idx] > 0).to(torch.uint8)).long()
                loss = TF.cross_entropy(net(xb), y, ignore_index=IGNORE)
            else:
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    loss = sum(net(list(xb), targets(view, idx)).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, nb, st["it"] = tot + float(loss.detach()), nb + 1, st["it"] + 1
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = time.time()
        metrics, stats = evaluate(detect(model, net, val_view.x, val_view.inst), val_view.inst, val_view.boxes)
        t2 = time.time()
        mon = metrics["mask_AP50"]
        improved = st["best"] is None or mon > st["best"]
        if improved:
            st.update(best=mon, best_epoch=epoch, wait=0, best_metrics=metrics)
            save({"state": net.state_dict(), "job": job, "epoch": epoch, "metrics": metrics}, mdir / "best.pt")
            stats["group"] = data.group[va].tolist()
            from evals import stats_path
            write_json(stats, stats_path(tag))
        else:
            st["wait"] += 1
        row = {"epoch": epoch, "train_loss": tot / max(nb, 1), **{f"val_{k}": v for k, v in metrics.items()},
               "val_monitor": mon, "best_val_monitor": st["best"], "epochs_since_best": st["wait"], "train_s": t1 - t0,
               "eval_s": t2 - t1, "examples_per_s": nb * bs / max(t1 - t0, 1e-9), "lr": lr}
        if dev.type == "cuda":
            row["peak_gpu_mib"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        st["history"].append(row)
        st["epoch"] = epoch
        st["wall"] += time.time() - t0
        tracker.log(epoch, row)
        if epoch % PANEL_EVERY == 0 or (improved and epoch - st["last_panel"] >= PANEL_EVERY):
            panels(epoch)
            st["last_panel"] = epoch
        save({"state": net.state_dict(), "opt": opt.state_dict(), "st": st, "rng": torch.get_rng_state()}, last)
        record("running")
        print(f"{tag} epoch {epoch}: loss {tot / max(nb, 1):.4f} mask AP50 {mon:.4f} (best {st['best']:.4f} at {st['best_epoch']}, "
              f"wait {st['wait']}/{patience})  train {t1 - t0:.1f}s eval {t2 - t1:.1f}s", flush=True)
        if st["wait"] >= patience:
            stopped_by = "patience"
            break
        if stop_requested():
            print(f"{tag}: STOP file present, stopping after epoch {epoch} (resumable)", flush=True)
            tracker.finish()
            return INTERRUPTED
    stopped_by = stopped_by or ("schedule" if steps else "max_epochs" if job.get("max_epochs") else "safety_cap")
    net.load_state_dict(torch.load(mdir / "best.pt", map_location=dev, weights_only=False)["state"])
    pan = panels(st["best_epoch"], final=True)
    s_ep = [h["train_s"] + h["eval_s"] for h in st["history"]]
    record("done", stopped_by=stopped_by, panels=pan, n_train=len(tr), n_val=len(va),
           s_per_epoch=sum(s_ep[1:]) / max(len(s_ep) - 1, 1) if len(s_ep) > 1 else s_ep[0])
    tracker.summary({"best_epoch": st["best_epoch"], "best_val_monitor": st["best"], "epochs": st["epoch"], "stopped_by": stopped_by})
    tracker.finish()
    print(f"{tag}: done, best mask AP50 {st['best']:.4f} at epoch {st['best_epoch']} of {st['epoch']} ({stopped_by})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
