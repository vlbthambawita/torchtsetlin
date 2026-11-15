#!/usr/bin/env bash
# Phase 0 (plan §6 Phase 2), on the RTX 3090 (cuda:0), one job at a time (D8):
#  1. pricing: both models x C in {1k, 8k, 64k} x B in {1, 32, 1024}, up to 2 000 timed updates each
#     after a 10-update warm-up -> results/phase0/*.json
#  2. sanity gate: flat (2k, b32) and conv (2k, b16), seed 0, the full protocol (early stopping);
#     these are the grid's own cells -> results/grid/*.json, mirrored to W&B.
# Usage: nohup bash code/phase0.sh > logs/phase0/driver.txt 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs/phase0 ../logs/grid
for m in flat conv; do for c in 1000 8000 64000; do for b in 1 32 1024; do
  echo "== pricing $m c$c b$b"
  python train.py --model "$m" --clauses "$c" --batch-size "$b" --arm phase0 --calibrate 2000 \
    --device cuda:0 --no-wandb >> ../logs/phase0/pricing.log 2>&1 || echo "   FAILED (exit $?)"
done; done; done
echo "== pricing done"
python train.py --model flat --clauses 2000 --batch-size 32 --seed 0 --arm grid --device cuda:0 \
  >> ../logs/grid/flat_c2000_b32_Tscaled_s0.log 2>&1 || echo "   sanity flat FAILED (exit $?)"
echo "== sanity flat done"
python train.py --model conv --clauses 2000 --batch-size 16 --seed 0 --arm grid --device cuda:0 \
  >> ../logs/grid/conv_c2000_b16_Tscaled_s0.log 2>&1 || echo "   sanity conv FAILED (exit $?)"
echo "== sanity conv done"
