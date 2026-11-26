"""Train one Tsetlin machine of the job list (plan section 5.4) and write its record.

    python code/train.py --tag n400.k3__s0 --device cuda:0

* After every epoch the whole validation split is scored. Early stopping on validation mIoU:
  ``SS_PATIENCE`` epochs (30), cap ``SS_MAX_EPOCHS`` (300); a job may fix ``max_epochs`` (pricing, smoke).
* ``models/<tag>/best.pt`` holds the automata of the best validation epoch. Validation, test and
  the full training split are scored once from it at the end (rule C4).
* ``models/<tag>/last.pt`` is written every epoch (automata, history, generator states), so a
  restarted run continues where it stopped. A STOP file ends the run after the current epoch
  (exit code 143: the runner does not count it as a failure).
"""

import argparse
import os
import sys
import time

import torch
import torchtsetlin as tt
from common import (
    MAX_EPOCHS,
    PANEL_EVERY,
    PATIENCE,
    env_info,
    model_dir,
    record_path,
    set_seed,
    stop_requested,
    write_json,
)
from data import Data
from models import Confusion, Net, margin
from tracking import Tracker, panel_path, render

INTERRUPTED = 143


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def score(net, data, split, boundary=False):
    """Metrics of a whole split, and the predictions."""
    t0 = time.time()
    pred = net.predict(data.x[split], split)
    sync(net.dev)
    m = Confusion().add(pred, data.y[split]).metrics()
    m["ms_per_image"] = 1000 * (time.time() - t0) / len(pred)
    if boundary:
        m["boundary_f1"] = float(tt.metrics.boundary_f1(pred.cpu(), data.y[split].cpu(), tolerance=2))
    return m, pred


def run(job, device, wandb_on=True):
    dev = torch.device(device)
    tag, mdir = job["tag"], model_dir(job["tag"])
    data = Data(dev, size=job["px"], limit=job.get("limit"))
    set_seed(job["seed"])
    net = Net(job, dev)
    max_epochs = job.get("max_epochs") or MAX_EPOCHS
    patience = PATIENCE if not job.get("max_epochs") else 10 ** 9
    st = {"epoch": 0, "best": -1.0, "best_epoch": 0, "history": [], "wall_s": 0.0}
    last = mdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load(ck["state"], data.x["val"])
        st = ck["st"]
        torch.set_rng_state(ck["rng"].cpu())
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), dev)
        print(f"resumed {tag} after epoch {st['epoch']}", flush=True)
    config = {**{k: v for k, v in job.items() if k not in ("needs", "tag")}, "T": margin(job), "patience": patience,
              "max_epochs": max_epochs, "train_images": len(data.y["train"])}
    tracker = Tracker(job, config, enabled=wandb_on and job["phase"] != "smoke")
    panel = data.panel()
    base = {"tag": tag, "job": job, "config": config, "env": env_info(dev), "wandb": tracker.url}
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def show(epoch, pred, note=""):
        p = render(data.x["val"][panel], data.y["val"][panel], pred[panel], panel_path(tag, epoch),
                   f"{tag}  epoch {epoch}{note}")
        tracker.image(epoch, "panel/val", p, f"validation, epoch {epoch}{note}")

    reason = None
    while reason is None:
        if st["epoch"] >= max_epochs:
            reason = "max_epochs"
        elif st["epoch"] - st["best_epoch"] >= patience:
            reason = "patience"
        elif stop_requested() and job["phase"] != "smoke":
            write_json({**base, "status": "stopped", **{k: st[k] for k in ("epoch", "best", "best_epoch")}}, record_path(tag))
            tracker.finish()
            print(f"stopped {tag} after epoch {st['epoch']} (STOP)", flush=True)
            return INTERRUPTED
        if reason:
            break
        epoch = st["epoch"] + 1
        t0 = time.time()
        conf = net.train_epoch(data.x["train"], data.y["train"])
        sync(dev)
        t1 = time.time()
        val, pred = score(net, data, "val")
        t2 = time.time()
        row = {"epoch": epoch, "train_s": t1 - t0, "eval_s": t2 - t1,
               **{f"train_running_{k}": v for k, v in conf.metrics().items()},
               **{f"val_{k}": v for k, v in val.items() if k != "ms_per_image"}, **net.info()}
        if val["mIoU"] > st["best"]:
            st["best"], st["best_epoch"] = val["mIoU"], epoch
            save({"state": net.state(), "epoch": epoch, "val": val, "job": job}, mdir / "best.pt")
        st["epoch"] = epoch
        st["wall_s"] += time.time() - t0
        st["history"].append(row)
        tracker.log(epoch, {**row, "best_val_mIoU": st["best"], "epochs_since_best": epoch - st["best_epoch"]})
        if epoch % PANEL_EVERY == 0:
            show(epoch, pred)
        save({"state": net.state(), "st": st, "rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None}, last)
        write_json({**base, "status": "running", "epochs": epoch, "best": st["best"], "best_epoch": st["best_epoch"],
                    "history": st["history"]}, record_path(tag))
        print(f"{tag} epoch {epoch:4d}  train {row['train_running_mIoU']:.3f}  val mIoU {val['mIoU']:.4f} "
              f"polyp {val['IoU_polyp']:.3f}  best {st['best']:.4f}@{st['best_epoch']}  "
              f"{row['train_s']:.1f}+{row['eval_s']:.1f} s", flush=True)

    best = torch.load(mdir / "best.pt", map_location=dev, weights_only=False)
    net.load(best["state"], data.x["val"])
    final = {}
    for sp in ("val", "test", "train"):
        final[sp], pred = score(net, data, sp, boundary=sp != "train")
        if sp == "val":
            show(st["best_epoch"], pred, " (best)")
    h = st["history"]
    rec = {**base, "status": "done", "stop_reason": reason, "epochs": st["epoch"], "best": st["best"],
           "best_epoch": st["best_epoch"], "val": final["val"], "test": final["test"], "train": final["train"],
           "gap_train_minus_val": final["train"]["mIoU"] - final["val"]["mIoU"], "model": net.info(),
           "s_per_epoch": sum(r["train_s"] + r["eval_s"] for r in h) / max(len(h), 1),
           "train_s_per_epoch": sum(r["train_s"] for r in h) / max(len(h), 1), "wall_s": st["wall_s"],
           "peak_gpu_mib": torch.cuda.max_memory_allocated(dev) / 2 ** 20 if dev.type == "cuda" else None,
           "history": h}
    write_json(rec, record_path(tag))
    tracker.summary({"best_val_mIoU": st["best"], "best_epoch": st["best_epoch"], "stop_reason": reason,
                     **{f"final_{sp}_{k}": v for sp in final for k, v in final[sp].items()},
                     "gap_train_minus_val": rec["gap_train_minus_val"], "s_per_epoch": rec["s_per_epoch"]})
    tracker.finish()
    print(f"done {tag}: val mIoU {final['val']['mIoU']:.4f}, test {final['test']['mIoU']:.4f}, "
          f"{st['epoch']} epochs ({reason})", flush=True)
    return 0


if __name__ == "__main__":
    import jobs as JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-wandb", action="store_true")
    a = ap.parse_args()
    sys.exit(run(JOBS.find(a.tag), a.device, wandb_on=not a.no_wandb))
