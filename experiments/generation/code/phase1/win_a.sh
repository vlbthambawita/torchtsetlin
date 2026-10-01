#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/win_a.log
: > $LOG
R=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1
# Judge accuracy is the binding G1 metric, so every window size is trained AND sampled.
run () {  # $1 rows_above  $2 left  $3 T
  ARM="win-r$1-l$2-T$3"
  python -u run_arm.py --arm "$ARM" --stage tm --context window --device cuda:0 \
      --rows-above $1 --left $2 --clauses 2000 --T $3 --weighted --batch 240 \
      --examples 4000000 --calib-examples 500000 --eval-images 500 --checks 2 \
      --allow-empty 2>&1 | tee -a $LOG
  python -u generate.py --checkpoint $R/${ARM}_seed0.pt --device cuda:0 \
      --n-per-digit 100 2>&1 | tee -a $LOG
}
run 4 8 8000
run 8 16 8000
run 4 8 24000
echo "### win_a done $(date +%T)" | tee -a $LOG
