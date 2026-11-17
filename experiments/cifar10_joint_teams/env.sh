# Sourced by every script here: the Python environment, W&B target, augmentation privacy switches.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# CJT_VENV=none uses whatever python is active (e.g. a conda env you set up yourself).
CJT_VENV=${CJT_VENV:-$PROG_DIR/.venv}
if [ "$CJT_VENV" != none ] && [ -f "$CJT_VENV/bin/activate" ]; then . "$CJT_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_CIFAR
# AlbumentationsX sends usage telemetry unless told not to (code/augment.py sets these as well).
export ALBUMENTATIONS_NO_TELEMETRY=1 ALBUMENTATIONS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
# Four runners share the CPU: keep each one's BLAS/OpenMP pool small.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
