#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_d.log
: > $LOG
# Bracket T below C/32 = 62 (the sweep only explored upward, and upward was monotonically worse).
for T in 16 31 45; do
  python -u run_arm.py --arm "screen-window-T$T" --stage tm --context window --device cuda:1 \
      --clauses 2000 --T $T --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
# Does the G0 clause-scaling law transfer? T follows the C/32 rule at each budget.
python -u run_arm.py --arm "screen-window-C8000" --stage tm --context window --device cuda:1 \
    --clauses 8000 --T 250 --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
python -u run_arm.py --arm "screen-window-C500" --stage tm --context window --device cuda:1 \
    --clauses 500 --T 16 --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
echo "### screen_d done $(date +%T)" | tee -a $LOG
