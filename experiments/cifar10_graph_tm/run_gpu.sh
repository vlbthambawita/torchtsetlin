#!/bin/bash
# Start (or resume) the queue on one GPU:  ./run_gpu.sh <gpu index>
# The plan uses four V100s, one per tmux session:  ./run_gpu.sh 0 ... ./run_gpu.sh 3
# All runners share one job list (.cache/claims/), so start them in this same folder.
# CGT_FG=1 keeps the runner in the foreground (e.g. of a tmux pane); CGT_PHASES=pricing,pilot
# restricts it to some phases; CGT_TS=0.5,5 fixes T / C and s without a pilot; CGT_GATE=off runs
# every cell as planned.
set -euo pipefail
K=${1:?usage: run_gpu.sh <runner number 0-3>}
cd "$(dirname "$0")"
. ./env.sh
[ -f ../cifar10_batch_clauses/.cache/data/cifar10_uint8.npz ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
if pgrep -f "cifar10_graph_tm.*queue_runner.py --gpu $K( |$)" >/dev/null \
   || { [ -f "logs/gpu$K.pid" ] && kill -0 "$(cat "logs/gpu$K.pid")" 2>/dev/null; }; then
  echo "GPU $K: a runner is already running"; exit 1
fi
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (SLURM, containers, one GPU per
# tmux session), gpu_device.sh uses that GPU (or the K-th entry of the list) instead of index K.
DEV=$(./gpu_device.sh "$K") || exit 1
export CGT_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=$DEV
if [ "${CGT_FG:-0}" = 1 ]; then
  echo "GPU slot $K (device $DEV): running in the foreground; Ctrl-C stops it (resumable)"
  python "$PWD/code/queue_runner.py" --gpu "$K" 2>&1 | tee -a "logs/gpu$K.log"
  exit "${PIPESTATUS[0]}"
fi
setsid nohup python "$PWD/code/queue_runner.py" --gpu "$K" >> "logs/gpu$K.log" 2>&1 < /dev/null &
echo $! > "logs/gpu$K.pid"
sleep 5
tail -n 3 "logs/gpu$K.log"
echo "GPU slot $K (device $DEV) runner started (pid $(cat logs/gpu$K.pid)). Follow it with: tail -f $PWD/logs/gpu$K.log"
