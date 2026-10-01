#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/id_test_a.log
: > $LOG
# Does the compositional (waveform) encoding of VALIDITY.md P2 handicap the TM on
# next-token-identity NLL? ecg-markov1 conditions on the previous token's stage-0 id and gets
# 4.429. Arm 1 gives the TM exactly that and nothing else -- a matched fight.
for T in 500 2000; do
  python -u tm_arm.py --arm "tm-s0-idonly-T$T" --stages 0 --T $T --prev-id 1 --no-wave \
      --examples 1500000 --n-eval 60 --device cuda:0 --allow-empty 2>&1 | tee -a $LOG
done
echo "### id_test_a done $(date +%T)" | tee -a $LOG
