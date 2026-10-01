#!/usr/bin/env bash
# Remaining G0 sweeps on cuda:0. Waits for the diagnostic job to release the GPU first.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
LOG=/work/vajira/DL2026/torchtsetlin/experiments/generation/results/g0
echo "### starting rest $(date +%T)"

# fast, high-information first
python -u run_g0.py --sweep p05 p06 --device cuda:0 --seeds 0 --n-examples 2000000      2>&1 | tee    $LOG/rest.log
python -u run_g0.py --sweep p03b    --device cuda:0 --seeds 0 1 2 --n-examples 2000000  2>&1 | tee -a $LOG/rest.log
python -u run_g0.py --sweep p04     --device cuda:0 --seeds 0 --n-examples-k 1000000    2>&1 | tee -a $LOG/rest.log
python -u run_g0.py --sweep p03a    --device cuda:0 --seeds 0 --n-examples 2000000      2>&1 | tee -a $LOG/rest.log
python -u run_g0.py --sweep p02     --device cuda:0 --seeds 0 --n-examples 500000       2>&1 | tee -a $LOG/rest.log
python -u run_g0.py --sweep p07     --device cuda:0 --seeds 0 --n-examples 2000000      2>&1 | tee -a $LOG/rest.log
echo "### rest finished $(date +%T)"
