#!/usr/bin/env bash
# Re-measure the B1 tt timing cells that overlapped the kernel audit (2026-10-28, load ~2.5),
# once code/run_gpu.sh has finished and the machine is idle again; then rebuild the report.
cd "$(dirname "$0")/.."
while pgrep -f "code/run_gpu.sh" > /dev/null; do sleep 60; done
{
  echo "== B1 tt re-timing $(date -Is)"
  ./reproduce.sh timing --bench B1 --lib tt --device cuda
  ./reproduce.sh report
  echo "== done $(date -Is)"
} >> .cache/logs/run_gpu.txt 2>&1
