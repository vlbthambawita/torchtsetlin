#!/usr/bin/env bash
# torchtsetlin CPU timing on B4/B5 with the current release (user, 2026-11-01); starts after the
# B2 batch diagnostic so the timing run has the machine to itself (rule P6).
# The 0.5.0 CPU records stay in results/timing_v0.5.0/. Detached: setsid nohup code/run_tt_cpu.sh &
cd "$(dirname "$0")/.."
while pgrep -f "code/diag_b2_batch.py --jobs" > /dev/null; do sleep 30; done
{
  echo "== tt CPU B4/B5 timing on $(python -c 'import torchtsetlin as t; print(t.__version__)' 2>/dev/null) start $(date -Is)"
  ./reproduce.sh timing --bench B4 B5 --lib tt --device cpu
  ./reproduce.sh report
  echo "== end $(date -Is)"
} >> .cache/logs/run_gpu.txt 2>&1
