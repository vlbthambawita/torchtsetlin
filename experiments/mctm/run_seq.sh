#!/usr/bin/env bash
# run_seq.sh <gpu> <jobs-file> -- the queue runner without the per-GPU lock, for a second
# stream on a card with room for it. Use queue_runner.py for anything unattended.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$1"
jobs="$2"; shift 2
for seed in "$@"; do
  grep -v '^#' "$jobs" | grep . | while IFS='|' read -r name args; do
    out="results/${name}_seed${seed}.json"
    [ -f "$out" ] && { echo "skip $name seed $seed"; continue; }
    echo "=== $name seed $seed (gpu $CUDA_VISIBLE_DEVICES) ==="
    python run_ideas.py --arm "$name" --seed "$seed" --epochs 30 --epochs-l1 30 \
      $args --out "$out" > "logs/${name}_seed${seed}.log" 2>&1
    echo "    rc=$? -> logs/${name}_seed${seed}.log"
  done
done
echo "=== $jobs done on gpu $CUDA_VISIBLE_DEVICES ==="
