#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_c.log
: > $LOG
for T in 1000 2000; do
  python -u run_arm.py --arm "screen-window-T$T" --stage tm --context window --device cuda:0 \
      --clauses 2000 --T $T --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
# Weighted clauses raise the attainable vote amplitude, which G0 Finding 1 ties to resolution.
python -u run_arm.py --arm "screen-window-weighted" --stage tm --context window --device cuda:0 \
    --clauses 2000 --T 500 --weighted --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
echo "### screen_c done $(date +%T)" | tee -a $LOG
