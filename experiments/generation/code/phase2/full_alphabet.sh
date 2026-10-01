#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/full_alphabet.log
: > $LOG
# The real Stage B arm, rerun with negative_scale = 1/K. Target: markov1 stage-0 NLL 4.429.
for T in 2000 8000; do
  python -u tm_arm.py --arm "tm-s0-ns-T$T" --stages 0 --T $T --prev-id 1 \
      --examples 1500000 --n-eval 60 --device cuda:0 --allow-empty 2>&1 | tee -a $LOG
done
echo "### full_alphabet done $(date +%T)" | tee -a $LOG
