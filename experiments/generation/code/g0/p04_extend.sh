#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/g0
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0/p04_sweep.log
while [ -n "3456015" ] && [ -d /proc/3456015 ]; do sleep 10; done
for T in 800 1600 3200; do
  python -u run_g0.py --sweep p04 --device cuda:1 --seeds 0 \
      --k-values 64 256 --T-k $T --n-examples-k 1000000 2>&1 | tee -a $LOG
done
echo "### p04 extend done $(date +%T)" | tee -a $LOG
