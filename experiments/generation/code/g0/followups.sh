#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/followups.log
: > $LOG
echo "### where exactly is the batch cliff? C=3200 T=100, fixed 20k updates" | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 \
    --batch-values 50 60 70 80 90 100 --fixed-updates 20000 2>&1 | tee -a $LOG
echo "### does higher specificity keep helping? s up to 160" | tee -a $LOG
python -u run_g0.py --sweep p07 --device cuda:1 --seeds 0 --clause-values 3200 12800 \
    --s-values 20 40 80 160 --n-examples 2000000 2>&1 | tee -a $LOG
echo "### followups done $(date +%T)" | tee -a $LOG
