#!/usr/bin/env bash
# Start queue workers for a job list and wait for all of them.
#   experiments/IMN/launch.sh "<job-name> [<job-name> ...]" [workers-per-gpu=3] [gpus="0 1"] [extra worker args]
# Several names are drained in order by every worker.
# Logs: experiments/IMN/logs/<job-name>_g<gpu>w<k>.log
set -u
names=$1; name=${names// /+}; per=${2:-3}; gpus=${3:-"0 1"}; extra=${4:-}
cd "$(dirname "$0")/../.."
mkdir -p experiments/IMN/logs
export PYTHONUNBUFFERED=1
# 6 workers share 32 cores: cap BLAS/OpenMP threads per worker (sklearn HGB/RF, torch CPU ops)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-5} MKL_NUM_THREADS=${OMP_NUM_THREADS:-5}
pids=()
for g in $gpus; do
  for k in $(seq 1 "$per"); do
    CUDA_VISIBLE_DEVICES=$g python -m experiments.IMN.code.jobs work --name $names --worker "$g$k" $extra \
      > "experiments/IMN/logs/${name}_g${g}w${k}.log" 2>&1 &
    pids+=($!)
    sleep 2
  done
done
wait "${pids[@]}"
for n in $names; do python -m experiments.IMN.code.jobs status --name "$n"; done
