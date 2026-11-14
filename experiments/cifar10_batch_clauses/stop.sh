#!/bin/bash
# ./stop.sh        graceful: every runner finishes its current run, then exits (creates STOP)
# ./stop.sh --now  stop immediately; interrupted runs resume from their last checkpoint (every 5 epochs)
# To continue later: rm STOP; ./run_gpu0.sh; ./run_gpu1.sh; ./run_gpu2.sh; ./run_gpu3.sh
cd "$(dirname "$0")"
touch STOP
echo "STOP created: runners exit after their current run."
if [ "${1:-}" = "--now" ]; then
  for k in 0 1 2 3; do
    [ -f logs/gpu$k.pid ] && kill -TERM -- "-$(cat logs/gpu$k.pid)" 2>/dev/null && echo "GPU $k: stopped"
  done
fi
