#!/usr/bin/env bash
# The full measured run (Phases 3-5, 7), detached:  setsid nohup code/run_all.sh &
# Progress: .cache/logs/run_all.txt (one line per cell). Safe to restart: finished cells are skipped.
cd "$(dirname "$0")/.."
{
  echo "== start $(date -Is)"
  ./reproduce.sh repro
  ./reproduce.sh timing
  ./reproduce.sh seeds
  ./reproduce.sh report
  echo "== end $(date -Is)"
} >> .cache/logs/run_all.txt 2>&1
