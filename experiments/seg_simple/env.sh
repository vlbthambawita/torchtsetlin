# Sourced by every script here: the Python environment, the data and the W&B target.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# SS_VENV=none uses whatever python is active (e.g. a conda env you set up yourself).
SS_VENV=${SS_VENV:-$PROG_DIR/.venv}
if [ "$SS_VENV" != none ] && [ -f "$SS_VENV/bin/activate" ]; then . "$SS_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_SEG_SIMPLE
export SS_WANDB_PROJECT=${SS_WANDB_PROJECT:-TM_SEG_SIMPLE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Kvasir-SEG: /work/vajira/data/Kvasir-SEG by default (code/data.py); SS_DATA=/other/path overrides
if [ -f "$PROG_DIR/.ss_data" ]; then export SS_DATA=${SS_DATA:-$(cat "$PROG_DIR/.ss_data")}; fi
