#!/bin/bash
# Start (or resume) the queue on one GPU: ./run_gpu.sh <GPU index> [phases] [runners per GPU]
# Three A100s (GPUs 0, 3 and 1 on that server): ./run_gpu0.sh, ./run_gpu3.sh, ./run_gpu1.sh, one per tmux session.
#   ./run_gpu0.sh                price (default): 3 epochs of every dense shape
#   ./run_gpu0.sh main           then: side, wide, price3d, conv3d (several: main,side)
#   ./run_gpu0.sh main 3         three runners on this GPU (1 to 4; a dense run uses about 1 GiB and leaves the GPU mostly idle)
# All runners share one job list, so start one script per GPU / tmux session, all in this folder.
#   SS_FG=1 ./run_gpu0.sh                stay in the foreground (log also in logs/gpuK.log)
#   SS_WANDB=offline|0 ./run_gpu0.sh     W&B offline (wandb sync later) or off
#   SS_PATIENCE=30 SS_MAX_EPOCHS=300     early stopping (defaults of the plan)
# Use the same options in every session.
set -euo pipefail
K=${1:?usage: run_gpu.sh <GPU index> [phases] [runners per GPU]}
if [ -n "${2:-}" ]; then export SS_PHASES=$2; fi
if [ -n "${3:-}" ]; then export SS_SLOTS=$3; fi
cd "$(dirname "$0")"
. ./env.sh
[ -f .cache/data/kvasir_64.pt ] || { echo "run ./setup.sh first"; exit 1; }
if [ -f STOP ]; then echo "STOP file present: remove it (rm STOP) to run again"; exit 1; fi
# Phases coarse and multi use 256 px images: build that cache once (a lock keeps the other sessions waiting).
case ",${SS_PHASES:-price}," in *,coarse,*|*,multi,*)
  if [ ! -f .cache/data/kvasir_256.pt ]; then
    echo "building the 256 px image cache (once, about a minute)"
    flock .cache/data/.build256.lock python code/data.py || { echo "could not build .cache/data/kvasir_256.pt"; exit 1; }
  fi ;;
esac
mkdir -p logs
# Slot K -> one device. If this shell already restricts the GPUs (one GPU per tmux session, SLURM,
# containers), gpu_device.sh uses that GPU / the K-th entry instead of the bare index K.
DEV=$(./gpu_device.sh "$K") || exit 1
export CUDA_VISIBLE_DEVICES=$DEV
SUBS=("")
case "${SS_SLOTS:-1}" in 1) ;; 2) SUBS=("" b) ;; 3) SUBS=("" b c) ;; 4) SUBS=("" b c d) ;; *) echo "runners per GPU: 1 to 4"; exit 1 ;; esac
if [ "${SS_FG:-0}" = 1 ]; then
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
  echo "GPU slot $K$S (device $DEV) runner started (pid $(cat "logs/gpu$K$S.pid")), phases ${SS_PHASES:-price}. Follow it with: tail -f $PWD/logs/gpu$K$S.log"
done
