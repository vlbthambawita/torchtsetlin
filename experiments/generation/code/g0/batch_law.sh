#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/batch_law.log
: > $LOG
echo "### test the 'stable batch ~ 6s' law, C=3200 T=100, fixed 20k updates" | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 --s 40 \
    --batch-values 240 480 960 1920 --fixed-updates 20000 2>&1 | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 --s 80 \
    --batch-values 240 480 960 --fixed-updates 20000 2>&1 | tee -a $LOG
echo "### best recipe at a large batch: C=12800 T=400 s=40, 3 seeds" | tee -a $LOG
python -u run_g0.py --sweep p01 --device cuda:1 --seeds 0 1 2 --n-examples 2000000 \
    --clause-values 12800 --t-values 400 --s 40 --batch 240 2>&1 | tee -a $LOG
echo "### batch_law done $(date +%T)" | tee -a $LOG
