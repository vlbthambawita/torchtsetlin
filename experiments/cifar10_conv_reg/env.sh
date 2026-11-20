# Sourced by every script here: the Python environment, W&B target, augmentation privacy switches.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# Python environment: CCR_VENV=<path>, or CCR_VENV=none for whatever python is active. Default: this
# folder's .venv; if there is none, the one cifar10_joint_teams already built on this server (it has
# everything needed here); setup.sh creates .venv when neither exists.
if [ -z "${CCR_VENV:-}" ]; then
  if [ -f "$PROG_DIR/.venv/bin/activate" ]; then CCR_VENV=$PROG_DIR/.venv
  elif [ -f "$REPO_DIR/experiments/cifar10_joint_teams/.venv/bin/activate" ]; then
    CCR_VENV=$REPO_DIR/experiments/cifar10_joint_teams/.venv
  else CCR_VENV=$PROG_DIR/.venv; fi
fi
if [ "$CCR_VENV" != none ] && [ -f "$CCR_VENV/bin/activate" ]; then . "$CCR_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_CIFAR
# AlbumentationsX sends usage telemetry unless told not to (code/augment.py sets these as well).
export ALBUMENTATIONS_NO_TELEMETRY=1 ALBUMENTATIONS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
# Three runners share the CPU: keep each one's BLAS/OpenMP pool small.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
