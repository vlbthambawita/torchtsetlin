#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/coarse.log
: > $LOG
# Alphabet size vs the count-table baseline, at a fixed clause budget.
for N in 16 32 64 128 256; do
  python -u coarse_test.py --n-coarse $N --clauses 2000 --T 2000 --device cuda:0 2>&1 | tee -a $LOG
done
# And the other lever G0 Finding 6 predicts: more clauses at the hardest alphabet.
for C in 8000 32000; do
  python -u coarse_test.py --n-coarse 256 --clauses $C --T 2000 --device cuda:0 2>&1 | tee -a $LOG
done
echo "### coarse sweep done $(date +%T)" | tee -a $LOG
