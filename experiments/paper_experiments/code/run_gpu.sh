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
  # tmu seeds one at a time: four tmu runs sharing the GPU ran ~6x slower each (2026-10-29)
  ./reproduce.sh seeds --lib pytmcuda tt --jobs 4
  ./reproduce.sh seeds --lib tmu --jobs 1
  ./reproduce.sh report
  echo "== end $(date -Is)"
} >> .cache/logs/run_gpu.txt 2>&1
