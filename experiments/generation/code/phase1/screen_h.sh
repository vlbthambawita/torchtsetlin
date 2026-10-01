#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_h.log
: > $LOG
# Weighted was still improving at T=4000 (74.76 and falling). Push until it turns over.
for T in 8000 16000 32000 64000; do
  python -u run_arm.py --arm "screen-wt-b240-T$T" --stage tm --context window --device cuda:1 \
      --clauses 2000 --T $T --weighted --batch 240 --examples 4000000 \
      --calib-examples 500000 --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
echo "### screen_h done $(date +%T)" | tee -a $LOG
