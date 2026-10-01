#!/usr/bin/env bash
# Is the p03a collapse a clause-capacity limit, or just a mis-set T?
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/capacity.log
while [ -d /proc/3455764 ]; do sleep 10; done
: > $LOG
# 1000 contexts at the budget whose tuned T we know (C=12800 -> T=400), vs the C=800 baseline
python -u run_g0.py --sweep p03a --device cuda:1 --seeds 0 --n-examples 2000000 \
    --context-counts 1000 --clauses 12800 --T 400 2>&1 | tee -a $LOG
python -u run_g0.py --sweep p03a --device cuda:1 --seeds 0 --n-examples 2000000 \
    --context-counts 1000 --clauses 3200 --T 100 2>&1 | tee -a $LOG
python -u run_g0.py --sweep p03a --device cuda:1 --seeds 0 --n-examples 2000000 \
    --context-counts 1000 --clauses 800 --T 25 2>&1 | tee -a $LOG
python -u run_g0.py --sweep p03a --device cuda:1 --seeds 0 --n-examples 2000000 \
    --context-counts 10000 --clauses 51200 --T 1600 2>&1 | tee -a $LOG
echo "### capacity test done $(date +%T)" | tee -a $LOG
