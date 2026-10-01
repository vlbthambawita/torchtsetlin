#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/payoff.log
: > $LOG
# Does negative_scale ~ 1/N close the gap at larger alphabets too?
for N in 64 256; do
  NS=$(python -c "print(1.0/$N)")
  python -u coarse_test.py --n-coarse $N --clauses 2000 --T 2000 --negative-scale $NS \
      --device cuda:1 2>&1 | grep -E "^\[coarse" | tee -a $LOG
  python -u coarse_test.py --n-coarse $N --clauses 8000 --T 2000 --negative-scale $NS \
      --device cuda:1 2>&1 | grep -E "^\[coarse" | tee -a $LOG
done
echo "### payoff done $(date +%T)" | tee -a $LOG
