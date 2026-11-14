# Sourced by every script here: the Python environment and W&B target.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# CBC_VENV=none uses whatever python is active (e.g. a conda env you set up yourself).
CBC_VENV=${CBC_VENV:-$PROG_DIR/.venv}
if [ "$CBC_VENV" != none ] && [ -f "$CBC_VENV/bin/activate" ]; then . "$CBC_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_CIFAR
