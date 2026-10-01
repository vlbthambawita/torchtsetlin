#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/length.log
: > $LOG
R=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1
# Seed 0 scored judge 0.438 at 4M examples and 0.381 at 8M, with test NLL essentially unchanged
# (76.36 vs 76.48). If that is real rather than seed noise, sample quality peaks EARLIER than
# likelihood, and the right stopping rule for a generator is not the likelihood one.
for EX in 2000000 4000000; do
  for SEED in 0 1 2; do
    ARM="len-$((EX/1000000))M-s$SEED"
    python -u run_arm.py --arm "$ARM" --stage tm --context window --device cuda:0 \
        --rows-above 4 --left 8 --clauses 2000 --T 8000 --weighted --batch 240 \
        --examples $EX --calib-examples 1000000 --eval-images 2000 --checks 2 \
        --seed $SEED --allow-empty 2>&1 | tee -a $LOG
    python -u generate.py --checkpoint $R/${ARM}_seed${SEED}.pt --device cuda:0 \
        --n-per-digit 100 --seed $SEED 2>&1 | tee -a $LOG
  done
done
echo "### length done $(date +%T)" | tee -a $LOG
