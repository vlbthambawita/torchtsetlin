#!/bin/bash
# Progress of every runner (from results/*.json) and current GPU use.
cd "$(dirname "$0")"
. ./env.sh
python code/report.py --status
echo
for f in logs/gpu*.pid; do
  [ -f "$f" ] || continue
  k=${f#logs/gpu}; k=${k%.pid}
  if kill -0 "$(cat "$f")" 2>/dev/null; then s=running; else s=stopped; fi
  echo "GPU $k runner: $s   last: $(grep '^==' "logs/gpu$k.log" 2>/dev/null | tail -1)"
done
echo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
