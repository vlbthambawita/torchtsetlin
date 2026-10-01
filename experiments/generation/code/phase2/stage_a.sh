#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/stage_a.log
: > $LOG
# Gate G2 needs >=20 dB. A single VQ tops out near 5-8 dB because a block spans 48-130
# significant principal directions (measured) and log2(K) bits cannot address that.
# Residual VQ spends M*log2(K) bits; this sweep finds where, or whether, it clears the gate.
echo "### residual VQ depth sweep, B=25, 8 leads, high-passed" | tee -a $LOG
for M in 1 2 4 8 16 32; do
  python -u run_stage_a.py --channels 8 --B 25 --K 1024 --aligned 0 --highpass 1 \
      --quant rvq$M --n-eval 150 2>&1 | tee -a $LOG
done
echo "### best depth vs block length" | tee -a $LOG
for B in 10 50 100; do
  python -u run_stage_a.py --channels 8 --B $B --K 1024 --aligned 0 --highpass 1 \
      --quant rvq16 --n-eval 150 2>&1 | tee -a $LOG
done
echo "### stage_a done $(date +%T)" | tee -a $LOG
