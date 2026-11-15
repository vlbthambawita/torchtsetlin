#!/usr/bin/env bash
# Diagnostic (2026-11-09): does test accuracy fall with clause count only under batched feedback?
# therm5, 10 epochs: 64k at batch 1 on cuda:0 (vs the main run's batch-50 curve), and
# 2k / 8k / 16k at batch 50 and batch 1 on cuda:1. Records: results/diagnostics/; logs: logs/diagnostics/.
# Progress is mirrored to W&B (code/tracking.py). Usage: nohup bash code/diag_batch.sh &
cd "$(dirname "$0")"
mkdir -p ../logs/diagnostics
run() {
  python train.py --spec therm5 --clauses "$1" --batch-size "$2" --epochs 10 --device "$3" \
    --tag "diagnostics/therm5_c$1_b$2" >> "../logs/diagnostics/therm5_c$1_b$2.log" 2>&1
}
run 64000 1 cuda:0 &
for c in 2000 8000 16000; do for b in 50 1; do run "$c" "$b" cuda:1; done; done
wait
