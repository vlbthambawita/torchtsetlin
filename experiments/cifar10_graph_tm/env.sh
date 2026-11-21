# Sourced by every script here: the Python environment and the W&B target.
PROG_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$PROG_DIR/../.." && pwd)
# Python environment: CGT_VENV=<path>, or CGT_VENV=none for whatever python is active. Default: this
# folder's .venv; if there is none, the one an earlier CIFAR-10 programme built on this server
# (setup.sh adds the two packages it may lack); setup.sh creates .venv when none exists.
if [ -z "${CGT_VENV:-}" ]; then
  CGT_VENV=$PROG_DIR/.venv
  for d in cifar10_graph_tm cifar10_batch_clauses cifar10_conv_reg cifar10_joint_teams; do
    if [ -f "$REPO_DIR/experiments/$d/.venv/bin/activate" ]; then CGT_VENV=$REPO_DIR/experiments/$d/.venv; break; fi
  done
fi
if [ "$CGT_VENV" != none ] && [ -f "$CGT_VENV/bin/activate" ]; then . "$CGT_VENV/bin/activate"; fi
export PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.forge.coreweave.com}   # simulamet_mlc/TM_CIFAR
# Four runners share the CPU: keep each one's BLAS/OpenMP pool small.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
