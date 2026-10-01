#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/screen_e.log
: > $LOG
# Both probes repeated at the WINNING T=62; the earlier ones were run at T=500 and so were
# confounded by a bad T (G0 Finding 8: never compare variants at a shared, untuned T).
python -u run_arm.py --arm "screen-window-b240-T62" --stage tm --context window --device cuda:0 \
    --clauses 2000 --T 62 --batch 240 --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
python -u run_arm.py --arm "screen-window-weighted-T62" --stage tm --context window --device cuda:0 \
    --clauses 2000 --T 62 --weighted --examples 4000000 --calib-examples 500000 \
    --eval-images 300 --checks 2 --allow-empty 2>&1 | tee -a $LOG
# How much does training length buy at the winning setting?
python -u run_arm.py --arm "screen-window-T62-16M" --stage tm --context window --device cuda:0 \
    --clauses 2000 --T 62 --examples 16000000 --calib-examples 1000000 \
    --eval-images 300 --checks 4 --allow-empty 2>&1 | tee -a $LOG
echo "### screen_e done $(date +%T)" | tee -a $LOG
