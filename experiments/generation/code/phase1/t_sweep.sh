#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/t_sweep.log
: > $LOG
for T in 62 125 250 500 1000 2000; do
  python -u run_arm.py --arm "screen-window-T$T" --stage tm --context window --device cuda:0 \
      --clauses 2000 --T $T --examples 4000000 --calib-examples 500000 \
      --eval-images 300 --checks 4 --allow-empty 2>&1 | tee -a $LOG
done
echo "### t_sweep done $(date +%T)" | tee -a $LOG
