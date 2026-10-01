#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/primary.log
: > $LOG
# Winner of the screen: weighted clauses, T=8000 (swept until it turned over), batch 240
# (G0 Finding 5 permits it at s=40; the empty-clause gate is on, default 1%).
for SEED in 0 1 2; do
  python -u run_arm.py --arm mnist-tm-window --stage tm --context window --device cuda:0 \
      --clauses 2000 --T 8000 --weighted --batch 240 --examples 20000000 \
      --calib-examples 2000000 --eval-images 2000 --checks 6 --seed $SEED 2>&1 | tee -a $LOG
done
echo "### primary done $(date +%T)" | tee -a $LOG
