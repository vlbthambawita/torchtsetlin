"""Reload a saved team file and check it reproduces its recorded test accuracy exactly (plan B8).

A team file (train.py ``save_model`` or composite.py) holds one state_dict per member and the hp
to rebuild them; the team's score is ``sum_m weight_m * sums_m`` with ``hp["combine_weights"]``.
Writes results/<phase>/inference/<arm>/<tag>.json; exit code 1 on a mismatch.

Usage: python code/infer.py models/main/J/S_sum_seed0.pt [--device cuda:1]
"""

import argparse
import sys

import numpy as np
import torch
from common import RESULTS, load_all, new_record, write_json
from train import build_team


def load_team(path, dev):
    """(team in eval-ready state, the file's dict). Members are plain library checkpoints."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    hp = st["hp"]
    cfgs = {n: {k: c[k] for k in ("enc", "patch", "T", "s", "clauses")} for n, c in hp["members"].items()}
    team = build_team(cfgs, "sum" if len(cfgs) > 1 else "none", True, 1, dev)
    for n, m in zip(team.names, team.members):
        m.load_state_dict(st["members"][n])
    team.weights = [float(hp["combine_weights"][n]) for n in team.names]
    return team, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--eval-batch", type=int, default=1000)
    args = ap.parse_args()
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    team, st = load_team(args.model, dev)
    hp = st["hp"]
    encs = [hp["members"][n]["enc"] for n in team.names]
    x, y, _, _ = load_all(encs, dev, hp["data"]["subset"])
    sums = [s.float().cpu().numpy() for s in team.member_sums([x[e]["test"] for e in encs], args.eval_batch)]
    pred = team.combine(sums).argmax(1)
    test_acc = 100 * float((pred == y["test"]).mean())
    conf = np.zeros((10, 10), dtype=int)
    np.add.at(conf, (y["test"], pred), 1)
    ok = abs(test_acc - st["test_acc"]) < 1e-9
    rec = new_record("inference", device=dev, model=str(args.model), phase=st["phase"], arm=st["arm"], tag=st["tag"],
                     epoch=st["epoch"], test_acc=test_acc, test_acc_saved=st["test_acc"], reproduces=bool(ok),
                     member_test_acc={n: 100 * float((s.argmax(1) == y["test"]).mean()) for n, s in zip(team.names, sums)},
                     per_class_acc=[100 * float(conf[k, k] / conf[k].sum()) for k in range(10)], confusion=conf.tolist())
    write_json(rec, RESULTS / st["phase"] / "inference" / st["arm"] / f"{st['tag']}.json")
    print(f"[{st['arm']}/{st['tag']}] epoch {st['epoch']}: test {test_acc:.2f} (saved {st['test_acc']:.2f}) "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
