#!/usr/bin/env bash
# Two controls for confounds found in p02 and p03a: both collapses tracked empty clauses,
# which tracked the number of update() calls, not batch size or context count.
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/controls.log
: > $LOG
echo "### control A: batch size at FIXED update count (20k), C=800 T=25" | tee -a $LOG
python -u run_g0.py --sweep p02 --device cuda:0 --seeds 0 --clauses 800 --T 25 \
    --batch-values 50 100 500 2000 --fixed-updates 20000 2>&1 | tee -a $LOG
echo "### control B: 1000 contexts, trained until no clause is empty" | tee -a $LOG
for N in 8000000 32000000; do
  python -u run_g0.py --sweep p03a --device cuda:0 --seeds 0 --context-counts 1000 \
      --clauses 800 --T 25 --n-examples $N 2>&1 | tee -a $LOG
done
echo "### controls done $(date +%T)" | tee -a $LOG
