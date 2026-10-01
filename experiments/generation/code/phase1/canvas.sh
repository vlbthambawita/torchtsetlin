#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase1
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase1/canvas.log
: > $LOG
# H1 ablation: the full partial canvas (VALIDITY.md E1 applied). It gets its OWN T sweep,
# because G0 Finding 8 showed a shared T makes a comparison meaningless.
for T in 2000 8000 32000; do
  python -u run_arm.py --arm "screen-canvas-T$T" --stage tm --context canvas --device cuda:1 \
      --clauses 2000 --T $T --weighted --batch 240 --examples 2000000 \
      --calib-examples 400000 --eval-images 200 --checks 2 --allow-empty 2>&1 | tee -a $LOG
done
echo "### canvas screen done $(date +%T)" | tee -a $LOG
