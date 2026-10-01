#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/final_b.log
: > $LOG
R=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1
# Second candidate: is the 0.35-0.44 judge band across windows >= (4,8) a real ordering or
# seed noise? Three seeds at (8,16) answers it.
for SEED in 0 1 2; do
  python -u run_arm.py --arm mnist-tm-win816 --stage tm --context window --device cuda:1 \
      --rows-above 8 --left 16 --clauses 2000 --T 8000 --weighted --batch 240 \
      --examples 8000000 --calib-examples 1000000 --eval-images 2000 --checks 3 \
      --seed $SEED 2>&1 | tee -a $LOG
  python -u generate.py --checkpoint $R/mnist-tm-win816_seed${SEED}.pt --device cuda:1 \
      --n-per-digit 100 --seed $SEED 2>&1 | tee -a $LOG
done
# Control: what does calibration actually buy at generation time? (PLAN P1.5)
python -u generate.py --checkpoint $R/mnist-tm-win816_seed0.pt --device cuda:1 \
    --n-per-digit 100 --seed 0 --calibrator analytic 2>&1 | tee -a $LOG
echo "### final_b done $(date +%T)" | tee -a $LOG
