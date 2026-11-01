#!/usr/bin/env bash
# Re-run of the B2 tt-seq cells on torchtsetlin 0.6.1 (gap G25; 0.6.0 records in
# results/{timing,seeds}_v0.6.0_tt/). Detached: setsid nohup code/run_b2_seq.sh &
cd "$(dirname "$0")/.."
{
  echo "== B2 seq re-run on 0.6.1 start $(date -Is)"
  ./reproduce.sh timing --bench B2 --lib tt --mode seq --device cuda
  ./reproduce.sh seeds --bench B2 --lib tt --mode seq --jobs 4
  ./reproduce.sh report
  echo "== end $(date -Is)"
} >> .cache/logs/run_gpu.txt 2>&1
