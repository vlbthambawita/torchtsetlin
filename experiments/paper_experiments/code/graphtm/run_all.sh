#!/usr/bin/env bash
# Seed runs of plans/graph-tm-reproduction.md: GraphTM on GPU 0 (RTX 3090), CoTM on GPU 1 (RTX 3080).
# Usage: ./run_all.sh "0 1 2"   (seeds)
set -u
cd "$(dirname "$0")"
SEEDS=${1:-0}
LOG=../../results/graphtm/logs; mkdir -p "$LOG"
export PYTHONUNBUFFERED=1
( for s in $SEEDS; do for e in G1 G2; do
    CUDA_VISIBLE_DEVICES=0 python run_graphtm.py run $e --model graph --seed $s > "$LOG/${e}_graph_seed$s.log" 2>&1
  done; done ) &
( for s in $SEEDS; do for e in G1 G2; do
    CUDA_VISIBLE_DEVICES=1 python run_graphtm.py run $e --model cotm --seed $s > "$LOG/${e}_cotm_seed$s.log" 2>&1
  done; done ) &
wait
echo ALL DONE
