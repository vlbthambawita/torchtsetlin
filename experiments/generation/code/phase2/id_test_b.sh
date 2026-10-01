#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/id_test_b.log
: > $LOG
# Arm 2: id AND waveform together. If this beats both the id-only arm and the waveform-only
# arm (6.696), the two encodings are complementary and P2 was half right.
for T in 500 2000; do
  python -u tm_arm.py --arm "tm-s0-idwave-T$T" --stages 0 --T $T --prev-id 1 \
      --examples 1500000 --n-eval 60 --device cuda:1 --allow-empty 2>&1 | tee -a $LOG
done
echo "### id_test_b done $(date +%T)" | tee -a $LOG
