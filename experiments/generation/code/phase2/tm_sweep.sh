#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/tm_sweep.log
: > $LOG
# Stage 0 only: the per-stage baseline analysis showed ALL predictability lives there
# (markov1 gets 4.429 nats at stage 0 vs 6.39-6.79 at stages 1-7, i.e. residuals are noise).
# T is swept because the first configuration, carried over from Phase 1 at T=8000, collapsed
# (40 % empty clauses, NLL worse than uniform) -- G0 Finding 8.
for T in 100 500 2000 8000; do
  python -u tm_arm.py --arm "tm-s0-T$T" --stages 0 --T $T --examples 1000000 \
      --n-eval 60 --device cuda:0 --allow-empty 2>&1 | tee -a $LOG
done
echo "### tm_sweep done $(date +%T)" | tee -a $LOG
