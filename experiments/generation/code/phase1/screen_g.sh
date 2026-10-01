#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_g.log
: > $LOG
# T=16 beat T=62 decisively, so the optimum is BELOW the G0 C/32 rule for this task.
# Push down until it turns over (the mistake to avoid is declaring a winner from one side only).
for T in 4 8 12 24; do
  python -u run_arm.py --arm "screen-window-T$T" --stage tm --context window --device cuda:0 \
      --clauses 2000 --T $T --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
# ...and at batch 240, which was independently better.
for T in 8 16 31; do
  python -u run_arm.py --arm "screen-window-b240-T$T" --stage tm --context window --device cuda:0 \
      --clauses 2000 --T $T --batch 240 --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
echo "### screen_g done $(date +%T)" | tee -a $LOG
