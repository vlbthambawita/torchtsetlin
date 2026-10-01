#!/usr/bin/env bash
# P7 re-test: the multi-class / multi-label comparison with T tuned per variant.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/p04_sweep.log
: > $LOG
for T in 25 50 100 200 400; do
  python -u run_g0.py --sweep p04 --device cuda:1 --seeds 0 \
      --k-values 64 256 --T-k $T --n-examples-k 1000000 2>&1 | tee -a $LOG
done
echo "### p04 sweep done $(date +%T)" | tee -a $LOG
