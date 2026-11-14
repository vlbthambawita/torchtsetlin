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
CUDA_VISIBLE_DEVICES=$K setsid nohup python code/queue_runner.py --gpu "$K" >> "logs/gpu$K.log" 2>&1 < /dev/null &
echo $! > "logs/gpu$K.pid"
sleep 5
tail -n 3 "logs/gpu$K.log"
echo "GPU $K runner started (pid $!). Follow it with: tail -f $PWD/logs/gpu$K.log"
