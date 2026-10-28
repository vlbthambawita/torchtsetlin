#!/usr/bin/env bash
# GPU-only full benchmark (user decision 2026-10-28): torchtsetlin vs the two other GPU libraries
# (PyTsetlinMachineCUDA, tmu) selected from the smoke records (code/select_gpu_libs.py).
# Detached:  setsid nohup code/run_gpu.sh &      Progress: .cache/logs/run_gpu.txt
# Safe to restart: finished cells are skipped.
cd "$(dirname "$0")/.."
LIBS=(pytmcuda tmu tt)
{
  echo "== start $(date -Is)"
  ./reproduce.sh timing --lib "${LIBS[@]}" --device cuda
  ./reproduce.sh seeds --lib "${LIBS[@]}" --jobs 4
  ./reproduce.sh report
  echo "== end $(date -Is)"
} >> .cache/logs/run_gpu.txt 2>&1
