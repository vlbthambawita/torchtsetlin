#!/bin/bash
# Start (or resume) the queue on one GPU: ./run_gpu.sh <slot 0-3>   (run_gpu0.sh ... run_gpu3.sh)
# The four runners share one job list, so start one per GPU / tmux session, all in this folder.
#   CJT_PHASES=pricing ./run_gpu0.sh              only the cost table
#   CJT_PHASES=pricing,pilot,grid ./run_gpu0.sh   the default
#   CJT_PHASES=ext ./run_gpu0.sh                  team M at B = 1 and 4
#   CJT_PHASES=large CJT_LARGE_BATCHES=32,128,1024 ./run_gpu0.sh   the five-member team
#   CJT_FG=1 ./run_gpu0.sh                        stay in the foreground (log also in logs/gpuK.log)
set -euo pipefail
K=${1:?usage: run_gpu.sh <gpu slot 0-3>}
cd "$(dirname "$0")"
. ./env.sh
[ -f ../cifar10_batch_clauses/.cache/data/cifar10_uint8.npz ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
if pgrep -f "queue_runner.py --gpu $K( |$)" >/dev/null; then echo "GPU $K: a runner is already running"; exit 1; fi
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (one GPU per tmux session, SLURM,
# containers), gpu_device.sh uses that GPU / the K-th entry instead of the bare index K.
DEV=$(./gpu_device.sh "$K") || exit 1
export CJT_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=$DEV
if [ "${CJT_FG:-0}" = 1 ]; then
  echo "GPU slot $K (device $DEV): running in the foreground; Ctrl-C stops it (resumable)"
  python code/queue_runner.py --gpu "$K" 2>&1 | tee -a "logs/gpu$K.log"
  exit "${PIPESTATUS[0]}"
fi
setsid nohup python code/queue_runner.py --gpu "$K" >> "logs/gpu$K.log" 2>&1 < /dev/null &
echo $! > "logs/gpu$K.pid"
sleep 5
tail -n 3 "logs/gpu$K.log"
echo "GPU slot $K (device $DEV) runner started (pid $(cat logs/gpu$K.pid)). Follow it with: tail -f $PWD/logs/gpu$K.log"
