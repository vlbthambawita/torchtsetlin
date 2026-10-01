#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/ns_sweep.log
: > $LOG
# negative_scale x T at N=32, where markov gets 1.270 and the TM got 3.290 with ns=1.0.
for NS in 1.0 0.3 0.1 0.03 0.01; do
  for T in 200 2000; do
    python -u coarse_test.py --n-coarse 32 --clauses 2000 --T $T --negative-scale $NS \
        --device cuda:1 2>&1 | tee -a $LOG
  done
done
echo "### ns_sweep done $(date +%T)" | tee -a $LOG
