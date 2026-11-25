"""Train one head (PLAN_KVASIR §8, §9): ``python train.py --tag <tag>``.

* One epoch is one pass over the training images with fresh sampling (``heads.py``).
* Early stopping: patience 75 epochs (``KV_PATIENCE``) on the head's validation metric; any
  improvement counts, the earliest epoch wins ties, the best state is what is saved as
  ``models/<tag>/best.pt``. Safety cap ``KV_MAX_EPOCHS`` (1 500), recorded when hit.
* Resumable (rule C9): ``last.pt`` is written every epoch with the team, the early-stopping
  state, the history and the random-number states; a restarted job continues from it and keeps
  its W&B run.
* Panels: every 15 epochs (``KV_PANEL_EVERY``), at a new best epoch at most once per 15, and at
  the end, for 16 test and 16 validation images (rule C8: the test panel is visual only; this
  file computes no metric on test images).
* ``touch STOP`` in the programme folder: the run stops after the current epoch (exit 143).
"""

import argparse
import os
import sys
import time

import jobs as JOBS
import torch
from common import (
    DEVICE,
    MAX_EPOCHS,
    PANEL_EVERY,
    PATIENCE,
    env_info,
    model_dir,
    read_json,
    record_path,
    set_seed,
    stop_requested,
    write_json,
)
from data import Data
from heads import HEADS
from tracking import Tracker, panel_path, render

INTERRUPTED = 143


def save(obj, path):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def panel_items(head, team, x, inst, job, data, device):
    """The head's own panel, or the end-to-end prediction when the job names the other heads."""
    refs = job.get("pipeline")
    if not refs or head.name != "mask":
        return head.panel(team, x, inst)
    from data import boxes_of, valid_rect
    from pipeline import Pipeline, run
    pipe = Pipeline(data, device, refs, live={"mask": (head, team)})
    rect = torch.stack([valid_rect(m) for m in inst])
    dets = run(lambda a, b, c, d: pipe.detect(a, b, c), x, inst, rect, [boxes_of(m) for m in inst])
    items = []
    for d in dets:
        if len(d["cls"]) == 0:
            items.append({"pred_mask": torch.zeros(inst.shape[1:], dtype=torch.bool)})
            continue
        order = torch.argsort(d["obj"], descending=True, stable=True)
        order = order[torch.argsort(d["cls"][order], descending=True, stable=True)]
        keep = d["cls"] > 0                          # the top detection, and every one the classifier accepts
        keep[order[0]] = True
        items.append({"pred_mask": d["masks"][keep].any(0).cpu(), "pred_boxes": d["boxes"][keep].cpu(),
                      "scores": d["cls"][keep].cpu()})
    return items


def log_panels(tracker, head, team, job, data, device, epoch, final=False):
    out = []
    team.eval()
    for split in ("test", "val"):
        x, inst, _ = data.panel(split)
        items = panel_items(head, team, x, inst, job, data, device)
        path = render(x.cpu(), inst.cpu(), items, f"{job['tag']}  {split}  epoch {epoch}",
                      panel_path(job["tag"], split, None if final else epoch))
        tracker.panel(epoch, split, x.cpu(), inst.cpu(), items, path)
        out.append(str(path))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--no-graphs", action="store_true", help="CUDA graphs off (resume test)")
    args = ap.parse_args()
    job = JOBS.find(args.tag)
    assert job["kind"] == "train", f"{args.tag} is a {job['kind']} job: run evals.py or baselines.py"
    if args.no_graphs:
        os.environ["TORCHTSETLIN_CUDA_GRAPHS"] = "0"
    tag, dev = job["tag"], torch.device(args.device)
    rec_path, mdir = record_path(tag), model_dir(tag)
    rec = read_json(rec_path)
    if rec and rec.get("status") == "done":
        print(f"{tag}: already done")
        return 0
    mdir.mkdir(parents=True, exist_ok=True)
    max_epochs = args.max_epochs or job.get("max_epochs") or MAX_EPOCHS
    patience = PATIENCE if not job.get("max_epochs") else 10 ** 9

    set_seed(job["seed"])
    data = Data(dev)
    head = HEADS[job["head"]](job, data, dev)
    team = head.team()
    st = {"epoch": 0, "best": None, "best_epoch": 0, "wait": 0, "history": [], "wall": 0.0, "last_panel": -10 ** 9}
    last = mdir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        team.load_state(ck["states"])
        st = ck["st"]
        torch.set_rng_state(ck["rng"].cpu())
        if dev.type == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), dev)
        print(f"{tag}: resuming after epoch {st['epoch']} (best {st['best']} at {st['best_epoch']})", flush=True)

    env = env_info(dev)
    config = {**{k: v for k, v in job.items() if k not in ("needs",)}, **env, "patience": patience,
              "max_epochs": max_epochs, "n_members": len(job["members"])}
    for k in ("enc", "conv", "patch", "scales", "crop", "clauses", "T", "s"):
        vals = [m.get(k) for m in job["members"]]
        config[k] = vals[0] if len(vals) == 1 else str(vals)
    tracker = Tracker(job, config, enabled=not args.no_wandb)      # setup.sh runs the smoke test with KV_WANDB=0

    def record(status, **extra):
        write_json({"tag": tag, "status": status, "job": job, "env": env, "best": st["best"],
                    "best_epoch": st["best_epoch"], "epochs": st["epoch"], "wall_s": st["wall"],
                    "patience": patience, "history": st["history"], "wandb_url": tracker.url, **extra}, rec_path)

    record("running")
    stopped_by, t_run = None, time.time()
    while st["epoch"] < max_epochs:
        epoch = st["epoch"] + 1
        t0 = time.time()
        tr = head.train_epoch(team, epoch)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t1 = time.time()
        team.eval()
        val = head.validate(team)
        t2 = time.time()
        improved = st["best"] is None or val["monitor"] > st["best"]
        if improved:
            st.update(best=val["monitor"], best_epoch=epoch, wait=0)
            save({"states": team.state(), "job": job, "epoch": epoch, "val": val}, mdir / "best.pt")
        else:
            st["wait"] += 1
        row = {"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}, **tr, "best_val_monitor": st["best"],
               "epochs_since_best": st["wait"], "train_s": t1 - t0, "eval_s": t2 - t1,
               "examples_per_s": tr["train_examples"] / max(t1 - t0, 1e-9), **team.stats()}
        if dev.type == "cuda":
            row["peak_gpu_mib"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        st["history"].append(row)
        st["epoch"] = epoch
        st["wall"] += time.time() - t0
        tracker.log(epoch, row)
        if epoch % PANEL_EVERY == 0 or (improved and epoch - st["last_panel"] >= PANEL_EVERY):
            log_panels(tracker, head, team, job, data, dev, epoch)
            st["last_panel"] = epoch
        save({"states": team.state(), "st": st, "rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None}, last)
        record("running")
        print(f"{tag} epoch {epoch}: monitor {val['monitor']:.4f} (best {st['best']:.4f} at {st['best_epoch']}, "
              f"wait {st['wait']}/{patience})  train {t1 - t0:.1f}s eval {t2 - t1:.1f}s", flush=True)
        if st["wait"] >= patience:
            stopped_by = "patience"
            break
        if stop_requested():
            print(f"{tag}: STOP file present, stopping after epoch {epoch} (resumable)", flush=True)
            tracker.finish()
            return INTERRUPTED
    stopped_by = stopped_by or ("max_epochs" if job.get("max_epochs") or args.max_epochs else "safety_cap")

    best = torch.load(mdir / "best.pt", map_location=dev, weights_only=False)
    team.load_state(best["states"])
    team.eval()
    panels = log_panels(tracker, head, team, job, data, dev, st["best_epoch"], final=True)
    final_val = head.validate(team)
    extra = {}
    for k in ("out_of_range",):
        if hasattr(head, k):
            extra[k] = getattr(head, k)
    s_ep = [h["train_s"] + h["eval_s"] for h in st["history"]]
    record("done", stopped_by=stopped_by, val=final_val, panels=panels, n_train=len(head.tr), n_val=len(head.va),
           s_per_epoch=sum(s_ep[1:]) / max(len(s_ep) - 1, 1) if len(s_ep) > 1 else s_ep[0], **extra)
    tracker.summary({"best_epoch": st["best_epoch"], "best_val_monitor": st["best"], "epochs": st["epoch"],
                     "wall_s": st["wall"], "stopped_by": stopped_by, "record": str(rec_path)})
    tracker.finish()
    print(f"{tag}: done, best {st['best']:.4f} at epoch {st['best_epoch']} of {st['epoch']} ({stopped_by}), "
          f"{time.time() - t_run:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
