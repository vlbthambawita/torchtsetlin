"""Cost probe (PLAN P2): 2 epochs per arm on T1 and T2 at both lengths; results/probe/*.json."""
from __future__ import annotations

import sys

from experiments.transformer_concepts.code.runner import ARMS, run

if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    for L in (16, 32):
        for task in ("T1", "T2"):
            LL = L + 1 if task == "T2" else L
            for arm in ARMS:
                if arm == "B3_conv_kd1":
                    continue
                key = f"{task}_L{LL}_{arm}"
                r = run("probe", key, task, LL, arm, 15.0, 1, epochs=2, patience=99, device=dev)
                print(f"{key:40s} train_s/epoch={r['train_seconds'] / r['epochs_run']:.2f} "
                      f"wall={r['wall_seconds']:.1f} val={r['val_curve']}", flush=True)
