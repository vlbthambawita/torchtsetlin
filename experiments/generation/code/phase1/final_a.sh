#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/final_a.log
: > $LOG
R=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1
# G1 confirmation: the winning window at 3 seeds. A single seed at 0.438 against a 0.40 gate
# is not evidence of passing.
for SEED in 0 1 2; do
  python -u run_arm.py --arm mnist-tm-win48 --stage tm --context window --device cuda:0 \
      --rows-above 4 --left 8 --clauses 2000 --T 8000 --weighted --batch 240 \
      --examples 8000000 --calib-examples 1000000 --eval-images 2000 --checks 3 \
      --seed $SEED 2>&1 | tee -a $LOG
  python -u generate.py --checkpoint $R/mnist-tm-win48_seed${SEED}.pt --device cuda:0 \
      --n-per-digit 100 --seed $SEED 2>&1 | tee -a $LOG
done
echo "### final_a done $(date +%T)" | tee -a $LOG
