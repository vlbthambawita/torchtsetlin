#!/bin/bash
# ./stop.sh        graceful: every training run stops after its current epoch, then the runners exit (creates STOP)
# ./stop.sh --now  stop immediately; interrupted runs resume from the checkpoint of their last finished epoch
# To continue later: rm STOP; ./run_gpu0.sh <phases>; ... ./run_gpu3.sh <phases>
cd "$(dirname "$0")"
touch STOP
echo "STOP created: runs stop after their current epoch, runners exit."
if [ "${1:-}" = "--now" ]; then
  for f in logs/gpu*.pid; do
    [ -f "$f" ] && kill -TERM -- "-$(cat "$f")" 2>/dev/null && echo "$(basename "$f" .pid): stopped"
  done
fi
