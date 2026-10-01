#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_b.log
: > $LOG
for T in 250 500; do
  python -u run_arm.py --arm "screen-window-T$T" --stage tm --context window --device cuda:1 \
      --clauses 2000 --T $T --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
# Throughput probe: G0 Finding 5 permits larger batches at s=40, guarded by the empty-clause gate.
python -u run_arm.py --arm "screen-window-b240" --stage tm --context window --device cuda:1 \
    --clauses 2000 --T 500 --batch 240 --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
echo "### screen_b done $(date +%T)" | tee -a $LOG
