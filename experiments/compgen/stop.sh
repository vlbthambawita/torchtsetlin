#!/bin/bash
# ./stop.sh        graceful: every runner finishes its current run, then exits (creates STOP)
# ./stop.sh --now  stop immediately; interrupted runs resume from their last checkpoint (every 25 epochs)
# To continue later: rm STOP; ./run_gpu0.sh; ./run_gpu1.sh; ./run_gpu2.sh; ./run_gpu3.sh
cd "$(dirname "$0")"
touch STOP
echo "STOP created: runners exit after their current run."
if [ "${1:-}" = "--now" ]; then
  for f in logs/gpu*.pid; do
    [ -f "$f" ] && kill -TERM -- "-$(cat "$f")" 2>/dev/null && echo "$(basename "$f" .pid): stopped"
  done
fi
