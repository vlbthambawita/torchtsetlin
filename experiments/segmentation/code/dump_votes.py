"""Save a frozen run's score maps for a split: python code/dump_votes.py RUN_ID [--split val]."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalio  # noqa: E402
import record  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--split", default="val")
ap.add_argument("--device", default="cuda:1")
a = ap.parse_args()
for rid in a.runs:
    rec = evalio.load_run(rid)
    _, _, scores = evalio.predict(rec, a.split, a.device)
    p = os.path.join(record.ARTIFACTS, "votes", f"{rid}_{a.split}.pt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    torch.save(scores, p)
    print(p, tuple(scores.shape))
