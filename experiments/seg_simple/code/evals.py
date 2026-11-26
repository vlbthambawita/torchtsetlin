"""Team jobs (plan section 12.1): combine the decisions of finished members. Nothing is trained.

    python code/evals.py --tag team-W4x400__0 --device cuda:0

Each member's vote map is clipped to its margin and divided by it, so every member weighs the same;
the team's label is the class with the largest sum. A ``greedy`` team is built by forward
selection on validation mIoU (so only its test number is a fair estimate).
"""

import argparse
import itertools
import sys

import torch
from common import env_info, model_dir, record_path, write_json
from data import Data
from models import Confusion, Net


def member_votes(job, data, dev):
    net = Net(job, dev)
    best = torch.load(model_dir(job["tag"]) / "best.pt", map_location=dev, weights_only=False)
    net.load(best["state"], data.x["val"])
    return {sp: (net.votes(data.x[sp]).clamp(-net.T, net.T) / net.T) for sp in ("val", "test")}


def metrics(v, y):
    return Confusion().add(v.argmax(1), y).metrics()


def run(job, device):
    import jobs as JOBS
    dev = torch.device(device)
    data = Data(dev, limit=job.get("limit"))
    index = {j["tag"]: j for j in JOBS.smoke_jobs() + JOBS.all_jobs(JOBS.PHASES)[0]}
    tags = list(job["members"])
    V = {t: member_votes(index[t], data, dev) for t in tags}
    single = {t: {sp: metrics(V[t][sp], data.y[sp]) for sp in ("val", "test")} for t in tags}
    chosen, path = tags, None
    if job.get("greedy"):
        chosen, total, path = [], None, []
        while len(chosen) < job["greedy"]:
            cand = {t: metrics(V[t]["val"] if total is None else total + V[t]["val"], data.y["val"])["mIoU"]
                    for t in tags if t not in chosen}
            t = max(cand, key=cand.get)
            if path and cand[t] <= path[-1]["val_mIoU"]:
                break
            chosen.append(t)
            total = V[t]["val"] if total is None else total + V[t]["val"]
            path.append({"added": t, "val_mIoU": cand[t]})
    team = {sp: metrics(sum(V[t][sp] for t in chosen), data.y[sp]) for sp in ("val", "test")}
    dis = [float((V[a]["val"].argmax(1) != V[b]["val"].argmax(1)).float().mean())
           for a, b in itertools.combinations(chosen, 2)]
    rec = {"tag": job["tag"], "job": job, "status": "done", "env": env_info(dev), "members": chosen,
           "clauses_per_class": sum(index[t]["n"] for t in chosen), "val": team["val"], "test": team["test"],
           "best": team["val"]["mIoU"], "member_metrics": {t: single[t] for t in chosen},
           "best_member_val_mIoU": max(single[t]["val"]["mIoU"] for t in chosen),
           "mean_pairwise_disagreement": sum(dis) / len(dis) if dis else None, "greedy_path": path}
    write_json(rec, record_path(job["tag"]))
    print(f"done {job['tag']}: {len(chosen)} members, val mIoU {team['val']['mIoU']:.4f} "
          f"(best member {rec['best_member_val_mIoU']:.4f}), test {team['test']['mIoU']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    import jobs as JOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sys.exit(run(JOBS.find(a.tag), a.device))
