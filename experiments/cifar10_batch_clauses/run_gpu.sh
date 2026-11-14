#!/bin/bash
# Start (or resume) the CIFAR-10 batch x clauses queue on one V100: ./run_gpu.sh <gpu-index>
# The four runners share one job list, so start one per GPU (run_gpu0.sh ... run_gpu3.sh).
set -euo pipefail
K=${1:?usage: run_gpu.sh <gpu index 0-3>}
cd "$(dirname "$0")"
. ./env.sh
[ -f .cache/data/cifar10_uint8.npz ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
if pgrep -f "queue_runner.py --gpu $K( |$)" >/dev/null; then echo "GPU $K: a runner is already running"; exit 1; fi
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (SLURM, containers, a set
# CUDA_VISIBLE_DEVICES), use the K-th entry of that list instead of the bare index K.
DEV=$(./gpu_device.sh "$K") || exit 1
CBC_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-} CUDA_VISIBLE_DEVICES=$DEV \
  setsid nohup python code/queue_runner.py --gpu "$K" >> "logs/gpu$K.log" 2>&1 < /dev/null &
echo $! > "logs/gpu$K.pid"
sleep 5
tail -n 3 "logs/gpu$K.log"
echo "GPU slot $K (device $DEV) runner started (pid $(cat logs/gpu$K.pid)). Follow it with: tail -f $PWD/logs/gpu$K.log"
