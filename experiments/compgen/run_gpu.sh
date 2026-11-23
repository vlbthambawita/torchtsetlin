#!/bin/bash
# Start (or resume) the queue on one GPU: ./run_gpu.sh <slot 0-3>   (run_gpu0.sh ... run_gpu3.sh)
# The four runners share one job list, so start one per GPU / tmux session, all in this folder.
#   CG_PHASES=price ./run_gpu0.sh            only the cost table
#   CG_PHASES=price,p1 ./run_gpu0.sh         without the standard-TM runs (phase 2)
#   CG_SEEDS=0 ./run_gpu0.sh                 one seed instead of 0,1,2
#   CG_REF=1 ./run_gpu0.sh                   also the reference-library runs (off by default)
#   CG_BOTH_MAX_EPOCHS=500 ./run_gpu0.sh     shorter cap for the standard-TM runs (default 3000)
#   CG_SLOTS=2 ./run_gpu0.sh                 two runners on this GPU (the models are tiny)
#   CG_FG=1 ./run_gpu0.sh                    stay in the foreground (log also in logs/gpuK.log)
# Use the same options in all four sessions.
set -euo pipefail
K=${1:?usage: run_gpu.sh <gpu slot 0-3>}
cd "$(dirname "$0")"
. ./env.sh
[ -f results/data/seed0.json ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (one GPU per tmux session, SLURM,
# containers), gpu_device.sh uses that GPU / the K-th entry instead of the bare index K.
DEV=$(./gpu_device.sh "$K") || exit 1
export CG_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=$DEV
SUBS=("")
if [ "${CG_SLOTS:-1}" = 2 ]; then SUBS=("" b); fi
if [ "${CG_FG:-0}" = 1 ]; then
  echo "GPU slot $K (device $DEV): running in the foreground; Ctrl-C stops it (resumable)"
  python code/queue_runner.py --gpu "$K" 2>&1 | tee -a "logs/gpu$K.log"
  exit "${PIPESTATUS[0]}"
fi
for S in "${SUBS[@]}"; do
  if pgrep -f "queue_runner.py --gpu $K --sub $S( |$)" >/dev/null; then echo "GPU $K$S: a runner is already running"; continue; fi
  setsid nohup python code/queue_runner.py --gpu "$K" --sub "$S" >> "logs/gpu$K$S.log" 2>&1 < /dev/null &
  echo $! > "logs/gpu$K$S.pid"
  sleep 3
  tail -n 2 "logs/gpu$K$S.log"
  echo "GPU slot $K$S (device $DEV) runner started (pid $(cat "logs/gpu$K$S.pid")). Follow it with: tail -f $PWD/logs/gpu$K$S.log"
done
