#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/final.log
: > $LOG
echo "### does the safe batch size scale with clause budget? (fixed 20k updates)" | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 \
    --batch-values 50 100 200 400 --fixed-updates 20000 2>&1 | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 12800 --T 400 \
    --batch-values 50 200 800 --fixed-updates 20000 2>&1 | tee -a $LOG
echo "### p07: specificity at each budget's tuned T" | tee -a $LOG
python -u run_g0.py --sweep p07 --device cuda:1 --seeds 0 --clause-values 800 3200 \
    --s-values 2 5 10 20 --n-examples 2000000 2>&1 | tee -a $LOG
echo "### final sweeps done $(date +%T)" | tee -a $LOG
