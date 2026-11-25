# Sourced by every script here: the Python environment, the data and the W&B target.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# KV_VENV=none uses whatever python is active (e.g. a conda env you set up yourself).
KV_VENV=${KV_VENV:-$PROG_DIR/.venv}
if [ "$KV_VENV" != none ] && [ -f "$KV_VENV/bin/activate" ]; then . "$KV_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_KVASIR
export KV_WANDB_PROJECT=${KV_WANDB_PROJECT:-TM_KVASIR}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export ALBUMENTATIONS_NO_TELEMETRY=1 ALBUMENTATIONS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
# Kvasir-SEG: /work/vajira/DATA/Kvasir-SEG by default (code/data.py); KV_DATA=/other/path overrides
[ -f "$PROG_DIR/.kv_data" ] && export KV_DATA=${KV_DATA:-$(cat "$PROG_DIR/.kv_data")}
