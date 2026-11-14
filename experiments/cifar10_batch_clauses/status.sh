#!/bin/bash
# Progress of all four GPUs (from results/*.json) and current GPU use.
cd "$(dirname "$0")"
. ./env.sh
python code/report.py --status
echo
for k in 0 1 2 3; do
  if [ -f logs/gpu$k.pid ] && kill -0 "$(cat logs/gpu$k.pid)" 2>/dev/null; then s=running; else s=stopped; fi
  echo "GPU $k runner: $s   last: $(grep '^==' logs/gpu$k.log 2>/dev/null | tail -1)"
done
echo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
