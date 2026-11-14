"""Smoke: a run interrupted mid-patience and resumed from its checkpoint must give exactly the same
curve and stopping epoch as an uninterrupted run (conv, C = 1 000, B = 32, 2 000 images).
Writes results/smoke/resume_check.json. Usage: CUDA_VISIBLE_DEVICES=0 python code/resume_check.py
"""

import os
import subprocess
import sys

from common import CACHE, HERE, RESULTS, read_json, write_json

BASE = [sys.executable, "train.py", "--model", "conv", "--clauses", "1000", "--batch-size", "32",
        "--arm", "resume_check", "--device", os.environ.get("CBC_DEVICE", "cuda:0"), "--subset", "2000", "--min-epochs", "1",
        "--patience", "2", "--max-epochs", "6", "--ckpt-every", "1", "--no-wandb"]


def run(extra):
    return subprocess.call(BASE + extra, cwd=HERE)


def main():
    for t in ("_full", "_resumed"):
        (RESULTS / "resume_check" / f"conv_c1000_b32_Tscaled_s0{t}.json").unlink(missing_ok=True)
        (CACHE / "ckpt" / "resume_check" / f"conv_c1000_b32_Tscaled_s0{t}.pt").unlink(missing_ok=True)
    assert run(["--tag-suffix", "_full"]) == 0
    assert run(["--tag-suffix", "_resumed", "--stop-after", "2"]) == 3
    assert run(["--tag-suffix", "_resumed"]) == 0
    a, b = (read_json(RESULTS / "resume_check" / f"conv_c1000_b32_Tscaled_s0{t}.json") for t in ("_full", "_resumed"))
    keys = ("val_acc", "test_acc", "train_acc_running", "mean_clause_len")
    ca = [[c[k] for k in keys] for c in a["curve"]]
    cb = [[c[k] for k in keys] for c in b["curve"]]
    same = ca == cb and a["best_epoch"] == b["best_epoch"] and a["epochs_run"] == b["epochs_run"]
    write_json({"identical": same, "full": ca, "resumed": cb, "resumed_at": b.get("resumed_at")},
               RESULTS / "smoke" / "resume_check.json")
    print(f"resume check: {'identical' if same else 'DIFFERENT'} ({a['epochs_run']} vs {b['epochs_run']} epochs)")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
