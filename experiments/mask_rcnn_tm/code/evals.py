"""Jobs that train nothing: ``python evals.py --tag <tag>`` (PLAN_KVASIR §7, §10).

    combine  a composite: independently trained members added at inference, validated as a head
    select   a pre-registered selection rule over finished records (validation metrics only)
    props    the proposal cache: out-of-fold for training images, full models for validation
    e2e      end-to-end detections on validation images, scored (gate K4; oracle swaps)
    k3       the mask arms on the same validation regions (gate K3 / hypothesis H1)
    oracle   the ceilings of the design (gate K0b) and the frozen anchor manifest
    sealed   the sealed split, once per arm (rule C3): refuses to run when its record exists
"""

import argparse
import sys
import time

import jobs as JOBS
import numpy as np
import torch
from augment import make_view
from common import (
    CACHE,
    DEVICE,
    MANIFESTS,
    RESULTS,
    S,
    env_info,
    model_dir,
    read_json,
    record_path,
    set_seed,
    write_json,
)
from data import Data, oof_halves
from detection import boxes as B
from detection import evaluate as EV
from detection import rois as R
from heads import HEADS, chunks
from pipeline import Pipeline, best_path, dense_cc, evaluate, load_head, run


def stats_path(tag):
    return RESULTS / "stats" / f"{tag}.json"


def monitor_of(tag, metric=None):
    r = read_json(record_path(tag))
    if r is None or r.get("status") != "done":
        raise SystemExit(f"record of {tag} is not done")
    if metric:
        return float(r["metrics"][metric])
    return float(r["best"] if r.get("best") is not None else r["metrics"]["mask_AP50"])


# --------------------------------------------------------------------------- combine
def do_combine(job, data, dev):
    head = HEADS[job["head"]](job, data, dev)
    team = head.team()
    states = []
    for t in job["member_runs"]:
        states += torch.load(best_path(t), map_location=dev, weights_only=False)["states"]
    team.load_state(states)
    team.eval()
    val = head.validate(team)
    model_dir(job["tag"]).mkdir(parents=True, exist_ok=True)
    torch.save({"states": team.state(), "job": job, "epoch": 0, "val": val}, best_path(job["tag"]))
    members = {t: monitor_of(t) for t in job["member_runs"]}
    return {"best": val["monitor"], "val": val, "members": members}


# --------------------------------------------------------------------------- select
def do_select(job):
    metric = job.get("metric")
    rows = []
    for c in job["candidates"]:
        vals = [monitor_of(t, metric) for t in c["tags"]]
        rows.append({"name": c["name"], "cfg": c["cfg"], "tags": c["tags"], "values": vals, "mean": float(np.mean(vals))})
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["mean"], i))      # ties: the earlier candidate
    ranking = [rows[i] for i in order]
    out = {"rule": job["rule"], "ranking": [{k: r[k] for k in ("name", "values", "mean")} for r in ranking]}
    if job["rule"] == "rank":
        out.update(best=ranking[0]["cfg"], top=[r["cfg"] for r in ranking])
    elif job["rule"] == "families":
        # the best member of each of the three best different (encoding, convolution) families
        fam, seen = [], set()
        for r in ranking:
            m = r["cfg"]["members"][0]
            key = (m["enc"], m["conv"], m.get("patch"), tuple(m.get("scales", ())), m.get("kd"), m.get("crop"))
            if key not in seen:
                seen.add(key)
                fam.append(r["cfg"])
        out.update(best=ranking[0]["cfg"], top=[r["cfg"] for r in ranking], families=fam[:3])
    elif job["rule"] == "team":
        by = {r["name"]: r for r in rows}
        teams = [r for r in ranking if r["name"] not in ("single", "W")]
        team, W, single = teams[0], by["W"], by["single"]
        diff = [a - b for a, b in zip(team["values"], W["values"])]
        adopt = all(d > 0 for d in diff)             # gate K5: on every seed
        pick = team if adopt else (W if W["mean"] > single["mean"] else single)
        out.update(best=pick["cfg"], picked=pick["name"], adopted=adopt, best_team=team["name"], team_minus_W=diff,
                   joint_minus_composite=(by["joint"]["mean"] - max(by[k]["mean"] for k in by if k.startswith("team"))
                                          if "joint" in by else None))
    elif job["rule"] == "wins_all":
        by = {r["name"]: r for r in rows}
        diff = [a - b for a, b in zip(by["alb"]["values"], by["sym"]["values"])]
        out.update(level="alb" if all(d > 0 for d in diff) else "sym", alb_minus_sym=diff, best=by["sym"]["cfg"])
    else:
        raise SystemExit(f"unknown selection rule {job['rule']!r}")
    return out


# --------------------------------------------------------------------------- props
def do_props(job, data, dev, post=100):
    tr, va = data.image_sets(job.get("fold"))
    if job.get("subset"):
        tr, va = tr[: int(job["subset"])], va[: max(8, int(job["subset"]) // 4)]
    n = data.x.shape[0]
    boxes = torch.zeros(n, post, 4)
    obj = torch.full((n, post), float("-inf"))
    half = oof_halves(data.group)

    def fill(refs, idx, store=True):
        pipe = Pipeline(data, dev, refs, post=post)
        v = make_view(data, idx, "none")
        images = []
        for c in chunks(len(idx), 8):
            for j, (b, s) in zip(c, pipe.propose(v.x[c], v.inst[c], v.rect[c])):
                if store:
                    boxes[idx[j], : len(b)] = b.cpu()
                    obj[idx[j], : len(b)] = s.cpu()
                images.append((s.cpu().numpy(), B.box_iou(b.double(), v.boxes[j].double()).cpu().numpy()))
        return EV.recall_at(images, ks=(1, 5, 10, 30, 100))

    rec = {"val": fill(job["full"], va)}
    oof = {}
    for j in (0, 1):
        idx = tr[half[tr] == j]
        if len(idx):
            oof[j] = (fill(job["oof"][j], idx), len(idx))
    rec["train_oof"] = {k: float(sum(r[k] * m for r, m in oof.values()) / sum(m for _, m in oof.values()))
                        for k in (1, 5, 10, 30, 100)}
    rec["train_in_sample"] = fill(job["full"], tr, store=False)
    rec["gap_in_sample_minus_val@30"] = rec["train_in_sample"][30] - rec["val"][30]      # decision D-5
    path = CACHE / "props" / f"{job['name']}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"boxes": boxes, "obj": obj, "job": job, "train": tr, "val": va}, path)
    return {"recall": {k: {str(a): b for a, b in v.items()} if isinstance(v, dict) else v for k, v in rec.items()},
            "cache": str(path), "best": rec["val"][30]}


# --------------------------------------------------------------------------- e2e / sealed
def detector_of(job, data, dev):
    if job["variant"] == "dense_cc":
        head, team = load_head(job["refs"]["dense"], data, dev)
        return lambda x, inst, rect, gt: dense_cc(head, team, x, inst)
    pipe = Pipeline(data, dev, job["refs"], **job.get("opts", {}))
    oracle = tuple(job.get("oracle", ()))
    return lambda x, inst, rect, gt: pipe.detect(x, inst, rect, gt_boxes=gt, oracle=oracle)


def do_e2e(job, data, dev):
    _, va = data.image_sets(job.get("fold"))
    if job.get("subset"):
        va = va[: max(8, int(job["subset"]) // 4)]
    v = make_view(data, va, "none")
    t0 = time.time()
    dets = run(detector_of(job, data, dev), v.x, v.inst, v.rect, v.boxes)
    dt = time.time() - t0
    metrics, stats = evaluate(dets, v.inst, v.boxes)
    stats["group"] = data.group[va].tolist()
    write_json(stats, stats_path(job["tag"]))
    return {"metrics": metrics, "best": metrics["mask_AP50"], "n_images": len(va), "ms_per_image": 1000 * dt / len(va)}


def do_sealed(job, data, dev):
    """The sealed split, scored once. The semantic threshold is the one chosen on validation."""
    src = read_json(record_path(job["source"]))
    sjob = src["job"]
    panel = set(read_json(MANIFESTS / "panel.json")["test"])
    if sjob["kind"] == "baseline":
        import baselines
        dets = baselines.detect_tag(sjob, data, dev, data.test_x, data.test_inst)
    else:
        from data import valid_rect
        rect = torch.stack([valid_rect(m) for m in data.test_inst])
        dets = run(detector_of(sjob, data, dev), data.test_x, data.test_inst, rect, data.test_boxes)
    thr = src["metrics"]["semantic_threshold"]
    metrics, stats = evaluate(dets, data.test_inst, data.test_boxes, threshold=thr)
    stats["group"] = data.test_group.tolist()
    write_json(stats, stats_path(job["tag"]))
    unseen = [i for i, s in enumerate(data.test_stems) if s not in panel]
    m2, _ = evaluate([dets[i] for i in unseen], data.test_inst[unseen], [data.test_boxes[i] for i in unseen], threshold=thr)
    return {"metrics": metrics, "metrics_never_viewed": m2, "n_images": len(dets), "n_never_viewed": len(unseen),
            "best": metrics["mask_AP50"], "validation": src["metrics"]}


# --------------------------------------------------------------------------- K3
def do_k3(job, data, dev):
    """Arms on the same validation regions: A the RoI mask head, C the same without position
    bits, B the dense map cut to the box, the filled box, and the mask head of baseline (iv)."""
    first = read_json(record_path(job["arms"]["A"][0]))["job"]
    ref = HEADS["mask"](dict(first, src="gt", fold=None, subset=None), data, dev)
    v, (bidx, boxes, _, m) = ref.val_view, ref._val
    group = data.group[ref.va][bidx.cpu()].numpy()
    per = {}
    for arm, tags in job["arms"].items():
        rows = []
        for t in tags:
            head, team = load_head(t, data, dev)
            if arm == "B":
                inside = R.paste(torch.ones(len(boxes), 2, 2, device=dev), boxes, S)
                masks = (head.predict(team, v.x)[bidx] > 0) & inside
            else:
                masks = R.paste(head.predict(team, v.x, bidx, boxes), boxes, S)
            rows.append(ref.region_iou(masks, v.inst, bidx, m).cpu().numpy())
        per[arm] = np.mean(rows, 0)                   # per region, mean over seeds
    per["filled_box"] = ref.region_iou(R.paste(torch.ones(len(boxes), 2, 2, device=dev), boxes, S), v.inst, bidx, m).cpu().numpy()

    def boot(d, n=2000):
        rng = np.random.default_rng(0)
        ug = np.unique(group)
        idx = {g: np.where(group == g)[0] for g in ug}
        vals = [d[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    out = {"mean_iou": {k: float(a.mean()) for k, a in per.items()}, "n_regions": int(len(boxes))}
    for name, (a, b) in {"A-B": ("A", "B"), "C-B": ("C", "B"), "A-filled": ("A", "filled_box"), "A-iv": ("A", "iv")}.items():
        d = per[a] - per[b]
        out[name] = {"mean": float(d.mean()), "ci95": boot(d)}
    out["H1_supported"] = out["A-B"]["ci95"][0] > 0
    best_tm = max(out["mean_iou"]["A"], out["mean_iou"]["B"], out["mean_iou"]["C"])
    out["required_margin_over_filled_box"] = best_tm >= out["mean_iou"]["filled_box"] + 0.05
    out["mask_head_for_pipeline"] = "mask" if out["mean_iou"]["A"] > out["mean_iou"]["B"] else "dense"
    out["best"] = out["mean_iou"]["A"]
    return out


# --------------------------------------------------------------------------- oracle ceilings
def do_oracle(job, data, dev):
    tr, va = data.image_sets(None)
    anchors = B.Anchors(S, device=dev)
    bins = B.DeltaBins()
    out = {}
    for name, idx in (("train", tr), ("val", va)):
        v = make_view(data, idx, "none")
        best, n_pos, nms_hit, d_iou, paste_iou, filled = [], [], [], [], {16: [], 28: [], 32: []}, []
        gen = torch.Generator().manual_seed(0)
        for i in range(len(v)):
            gt = v.boxes[i]
            if not len(gt):
                continue
            valid = anchors.valid(v.inst[i])
            iou = B.box_iou(anchors.boxes, gt) * valid.unsqueeze(1)
            best += iou.max(0).values.tolist()
            n_pos += (iou >= 0.7).sum(0).tolist()
            keep = B.nms(gt, torch.rand(len(gt), generator=gen).to(dev), 0.7)
            nms_hit += (B.box_iou(gt[keep], gt).max(0).values >= 0.5).tolist()
            top = iou.argmax(0)                       # deltas of the best anchor, through the bins
            ref = anchors.boxes[top]
            t, _ = bins.targets(B.encode(ref, gt))
            d_iou += torch.diag(B.box_iou(B.decode(ref, bins.decode(t.float() * 2 - 1)), gt)).tolist()
            gm = torch.stack([v.inst[i] == j + 1 for j in range(len(gt))])
            ar = torch.arange(len(gt), device=dev)
            for size in paste_iou:
                c = R.crop(gm[:, None].float(), ar, gt, size, mode="mask")[:, 0]
                back = R.paste(c.float() * 2 - 1, gt, S)
                paste_iou[size] += torch.diag(R.mask_iou(back, gm)).tolist()
            fb = R.paste(torch.ones(len(gt), 2, 2, device=dev), gt, S)
            filled += torch.diag(R.mask_iou(fb, gm)).tolist()
        best, d_iou = np.asarray(best), np.asarray(d_iou)
        out[name] = {"instances": int(len(best)),
                     "O1_best_anchor_recall@0.5": float((best >= 0.5).mean()),
                     "best_anchor_recall@0.7": float((best >= 0.7).mean()),
                     "median_positive_anchors_per_instance": float(np.median(n_pos)),
                     "O2_recall_after_nms_of_true_boxes": float(np.mean(nms_hit)),
                     "O3_mask_AP75_through_crop_and_paste": {str(k): float((np.asarray(a) >= 0.75).mean()) for k, a in paste_iou.items()},
                     "O4_box_AP75_of_true_deltas_through_bins": float((d_iou >= 0.75).mean()),
                     "O5_filled_box_mask_iou": float(np.mean(filled))}
    t = out["train"]
    gates = {"O1": t["O1_best_anchor_recall@0.5"] >= 0.99, "O2": t["O2_recall_after_nms_of_true_boxes"] >= 0.99,
             "O3": t["O3_mask_AP75_through_crop_and_paste"]["32"] >= 0.90, "O4": t["O4_box_AP75_of_true_deltas_through_bins"] >= 0.90}
    write_json({"anchors": anchors.describe(), "audit": out, "frozen": time.strftime("%Y-%m-%d")}, MANIFESTS / "anchors.json")
    return {"ceilings": out, "gates": gates, "K0b_pass": all(gates.values()), "best": t["O1_best_anchor_recall@0.5"]}


KINDS = {"combine": do_combine, "props": do_props, "e2e": do_e2e, "k3": do_k3, "oracle": do_oracle, "sealed": do_sealed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()
    job = JOBS.find(args.tag)
    rec_path = record_path(job["tag"])
    old = read_json(rec_path)
    if old and old.get("status") == "done":
        if job["kind"] == "sealed":
            print(f"{job['tag']}: the sealed split was already evaluated for this arm; refusing to run again (rule C3)")
        return 0
    missing = [t for t in job.get("needs", []) if not JOBS.is_done(t)]
    if missing:
        raise SystemExit(f"{job['tag']} needs {missing[:3]} first")
    t0 = time.time()
    if job["kind"] == "select":
        res, env = do_select(job), {}
    else:
        set_seed(int(job.get("seed", 0)))
        dev = torch.device(args.device)
        res, env = KINDS[job["kind"]](job, Data(dev), dev), env_info(dev)
    write_json({"tag": job["tag"], "status": "done", "job": job, "env": env, "wall_s": time.time() - t0, **res}, rec_path)
    show = {k: v for k, v in res.items() if k in ("best", "picked", "adopted", "level", "K0b_pass", "H1_supported")}
    print(f"{job['tag']}: done {show}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
