#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_f.log
: > $LOG
# Weighted clauses raise the attainable vote amplitude, so their T optimum is elsewhere.
# G0 Finding 8: sweep T per variant until it peaks AND turns over before declaring a winner.
for T in 125 250 500 1000 2000 4000; do
  python -u run_arm.py --arm "screen-wt-b240-T$T" --stage tm --context window --device cuda:1 \
      --clauses 2000 --T $T --weighted --batch 240 --examples 4000000 \
      --calib-examples 500000 --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
echo "### screen_f done $(date +%T)" | tee -a $LOG
