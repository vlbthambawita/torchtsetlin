"""Smoke: a run interrupted and resumed from its checkpoint must give exactly the same curve as an
uninterrupted run (team M, B = 32, 2 000 images, augmentation `full`). Writes
results/smoke/resume_check.json. Usage: CUDA_VISIBLE_DEVICES=0 python code/resume_check.py
"""

import os
import subprocess
import sys

from common import CACHE, HERE, RESULTS, read_json, write_json

BASE = [sys.executable, "train.py", "--phase", "resume_check", "--team", "M", "--batch-size", "32",
        "--device", os.environ.get("CJT_DEVICE", "cuda:0"), "--subset", "2000", "--min-epochs", "4",
        "--max-epochs", "4", "--ckpt-every", "1", "--viz-every", "0", "--no-wandb"]
TAG = "M_b32_full_s0"


def main():
    for t in ("_full", "_resumed"):
        (RESULTS / "resume_check" / f"{TAG}{t}.json").unlink(missing_ok=True)
        (CACHE / "ckpt" / "resume_check" / f"{TAG}{t}.pt").unlink(missing_ok=True)
    assert subprocess.call(BASE + ["--tag-suffix", "_full"], cwd=HERE) == 0
    assert subprocess.call(BASE + ["--tag-suffix", "_resumed", "--stop-after", "2"], cwd=HERE) == 3
    assert subprocess.call(BASE + ["--tag-suffix", "_resumed"], cwd=HERE) == 0
    a, b = (read_json(RESULTS / "resume_check" / f"{TAG}{t}.json") for t in ("_full", "_resumed"))
    def curve(r):
        return [[c["val_acc"], c["test_acc"], c["train_acc_running"]]
                + [c["members"][n]["mean_clause_len"] for n in c["members"]] for c in r["curve"]]
    same = curve(a) == curve(b) and a["best_epoch"] == b["best_epoch"]
    write_json({"identical": same, "full": curve(a), "resumed": curve(b), "resumed_at": b.get("resumed_at")},
               RESULTS / "smoke" / "resume_check.json")
    print(f"resume check: {'identical' if same else 'DIFFERENT'}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
