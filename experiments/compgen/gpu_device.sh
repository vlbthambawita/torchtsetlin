#!/bin/bash
# Print the CUDA_VISIBLE_DEVICES value for runner slot K (0-3).
#  - The shell sees exactly ONE GPU (e.g. one reserved GPU per tmux session): use that GPU,
#    whatever K is; K then only names the runner (logs/gpuK.log, records).
#  - The shell sees several GPUs through CUDA_VISIBLE_DEVICES (SLURM, containers): the K-th entry.
#  - Otherwise (all GPUs of the machine visible): K itself.
K=${1:?usage: gpu_device.sh <slot>}
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra VIS <<< "$CUDA_VISIBLE_DEVICES"
  if [ "${#VIS[@]}" -eq 1 ]; then echo "${VIS[0]}"; exit 0; fi
  if [ "$K" -ge "${#VIS[@]}" ]; then
    echo "GPU slot $K: this shell only sees CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (${#VIS[@]} GPUs)" >&2
    exit 1
  fi
  echo "${VIS[$K]}"
  exit 0
fi
N=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if [ "$N" -eq 1 ]; then echo 0; exit 0; fi
if [ "$N" -gt 0 ] && [ "$K" -ge "$N" ]; then
  echo "GPU slot $K: this shell only sees $N GPU(s) (nvidia-smi -L)" >&2
  exit 1
fi
echo "$K"
