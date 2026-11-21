#!/bin/bash
# ./stop.sh        graceful: every runner finishes its current run, then exits (creates STOP)
# ./stop.sh --now  stop immediately; interrupted runs resume from their last checkpoint (every 5 epochs)
# To continue later: rm STOP; ./run_gpu.sh 0 (and 1, 2, 3)
cd "$(dirname "$0")"
touch STOP
echo "STOP created: runners exit after their current run."
if [ "${1:-}" = "--now" ]; then
  for f in logs/gpu*.pid; do
    [ -f "$f" ] || continue
    k=${f#logs/gpu}; k=${k%.pid}
    kill -TERM -- "-$(cat "$f")" 2>/dev/null && echo "GPU $k: stopped"
  done
fi
