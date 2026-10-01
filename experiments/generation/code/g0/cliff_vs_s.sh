#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/cliff_vs_s.log
: > $LOG
echo "### does the batch cliff move with s? C=3200 T=100, fixed 20k updates" | tee -a $LOG
for S in 10 20 40; do
  python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 --s $S \
      --batch-values 40 60 80 120 160 240 --fixed-updates 20000 2>&1 | tee -a $LOG
done
echo "### cliff_vs_s done $(date +%T)" | tee -a $LOG
