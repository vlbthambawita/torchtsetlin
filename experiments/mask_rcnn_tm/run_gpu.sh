#!/bin/bash
# Start (or resume) the queue on one GPU: ./run_gpu.sh <slot 0-3> [phases]   (run_gpu0.sh ... run_gpu3.sh)
#   ./run_gpu0.sh                price (default)
#   ./run_gpu0.sh base,p1        the same as KV_PHASES=base,p1 ./run_gpu0.sh      then: p2, alb, p3, final
#   ./run_gpu0.sh base,p1 2      two runners on this GPU (the same as KV_SLOTS=2)
# The four runners share one job list, so start one per GPU / tmux session, all in this folder.
#   KV_PHASES=price ./run_gpu0.sh        3 epochs of everything + the oracle ceilings (default)
#   KV_PHASES=base,p1 ./run_gpu0.sh      baselines and stage A        then: p2, alb, p3, final
#   KV_SLOTS=2 ./run_gpu0.sh             two runners on this GPU (the Tsetlin heads are small)
#   KV_FG=1 ./run_gpu0.sh                stay in the foreground (log also in logs/gpuK.log)
#   KV_WANDB=offline|0 ./run_gpu0.sh     W&B offline (wandb sync later) or off
#   KV_SEEDS=0 ./run_gpu0.sh             one seed for the baselines and controls
# Use the same options in all four sessions.
set -euo pipefail
K=${1:?usage: run_gpu.sh <gpu slot 0-3> [phases] [runners per GPU]}
if [ -n "${2:-}" ]; then export KV_PHASES=$2; fi
if [ -n "${3:-}" ]; then export KV_SLOTS=$3; fi
cd "$(dirname "$0")"
. ./env.sh
[ -f .cache/data/kvasir_128.pt ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (one GPU per tmux session, SLURM,
# containers), gpu_device.sh uses that GPU / the K-th entry instead of the bare index K.
DEV=$(./gpu_device.sh "$K") || exit 1
export CUDA_VISIBLE_DEVICES=$DEV
SUBS=("")
if [ "${KV_SLOTS:-1}" = 2 ]; then SUBS=("" b); fi
if [ "${KV_FG:-0}" = 1 ]; then
  echo "GPU slot $K (device $DEV): running in the foreground; Ctrl-C stops it (resumable)"
  python code/queue_runner.py --gpu "$K" 2>&1 | tee -a "logs/gpu$K.log"
  exit "${PIPESTATUS[0]}"
fi
for S in "${SUBS[@]}"; do
  if pgrep -f "queue_runner.py --gpu $K --sub $S( |$)" >/dev/null; then echo "GPU $K$S: a runner is already running"; continue; fi
  setsid nohup python code/queue_runner.py --gpu "$K" --sub "$S" >> "logs/gpu$K$S.log" 2>&1 < /dev/null &
  echo $! > "logs/gpu$K$S.pid"
  sleep 3
  tail -n 3 "logs/gpu$K$S.log"
  echo "GPU slot $K$S (device $DEV) runner started (pid $(cat "logs/gpu$K$S.pid")), phases ${KV_PHASES:-price}. Follow it with: tail -f $PWD/logs/gpu$K$S.log"
done
