#!/bin/bash
# Progress of all runners (from results/runs/*.json) and current GPU use.  ./status.sh --price for the cost table.
cd "$(dirname "$0")"
. ./env.sh
python code/report.py ${1:---status}
echo
for f in logs/gpu*.pid; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .pid)
  if kill -0 "$(cat "$f")" 2>/dev/null; then s=running; else s=stopped; fi
  echo "$n runner: $s   last: $(grep '^==' "logs/$n.log" 2>/dev/null | tail -1)"
done
echo
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
