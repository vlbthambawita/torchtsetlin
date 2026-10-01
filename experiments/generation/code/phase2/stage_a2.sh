#!/usr/bin/env bash
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/generation/code/phase2
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/phase2/stage_a2.log
: > $LOG
# G2 needs min-lead SNR >= 20 dB AND seam ratio <= 2. Best so far (B=10, M=16) reaches
# 23.6 dB mean / 17.4 min-lead but seam 4.98. Two candidate fixes, tested separately so the
# credit is attributable:
#   (a) more residual depth   -> raises SNR, shrinks seams only indirectly
#   (b) overlap-add decoding  -> targets seams directly
echo "### (a) depth at the best block length" | tee -a $LOG
python -u run_stage_a.py --channels 8 --B 10 --K 1024 --aligned 0 --highpass 1 \
    --quant rvq32 --n-eval 150 2>&1 | tee -a $LOG
echo "### (b) overlap-add, same token count" | tee -a $LOG
for M in 16 32; do
  python -u run_stage_a.py --channels 8 --B 10 --K 1024 --aligned 0 --highpass 1 \
      --overlap 1 --quant rvq$M --n-eval 150 2>&1 | tee -a $LOG
done
echo "### stage_a2 done $(date +%T)" | tee -a $LOG
