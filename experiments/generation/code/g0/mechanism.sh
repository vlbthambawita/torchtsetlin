#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/mechanism.log
: > $LOG
echo "### does the batch cliff scale with n_states? C=3200 T=100 s=10, fixed 20k updates" | tee -a $LOG
for ST in 32 64 128 256 512; do
  python -u run_g0.py --sweep p02 --device cuda:1 --seeds 0 --clauses 3200 --T 100 \
      --n-states $ST --batch-values 10 20 40 80 160 320 --fixed-updates 20000 2>&1 | tee -a $LOG
done
echo "### best config, 3 seeds: C=12800 T=400 s=40" | tee -a $LOG
python -u run_g0.py --sweep p01 --device cuda:1 --seeds 0 1 2 --n-examples 2000000 \
    --clause-values 12800 --t-values 400 --s 40 2>&1 | tee -a $LOG
echo "### mechanism done $(date +%T)" | tee -a $LOG
