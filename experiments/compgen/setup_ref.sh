#!/bin/bash
# The reference library (arm `ref`): cair/PyHierarchicalTsetlinMachineCUDA at a pinned commit, in
# its own environment (.cache/ref/venv: numpy, scipy, pycuda). Called by setup.sh; safe to re-run.
# pycuda compiles against the CUDA toolkit: `nvcc` must be on PATH, or set CUDA_ROOT=/usr/local/cuda.
# If this fails, run everything else with CG_REF=0 (the reference runs are then skipped).
set -euo pipefail
cd "$(dirname "$0")"
REF=${CG_REF_BASE:-$PWD/.cache/ref}
REPO_URL=$(python3 - <<'EOF'
import sys; sys.path.insert(0, "code")
import common; print(common.REF_REPO, common.REF_COMMIT)
EOF
)
URL=${REPO_URL% *}; COMMIT=${REPO_URL#* }
mkdir -p "$REF"
if [ ! -d "$REF/PyHierarchicalTsetlinMachineCUDA/.git" ]; then
  git clone -q "$URL" "$REF/PyHierarchicalTsetlinMachineCUDA"
fi
git -C "$REF/PyHierarchicalTsetlinMachineCUDA" fetch -q origin
git -C "$REF/PyHierarchicalTsetlinMachineCUDA" checkout -q "$COMMIT"
echo "  reference library at $(git -C "$REF/PyHierarchicalTsetlinMachineCUDA" rev-parse --short HEAD)"
if [ -z "${CUDA_ROOT:-}" ] && ! command -v nvcc >/dev/null; then
  for d in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 /opt/cuda; do
    if [ -x "$d/bin/nvcc" ]; then export CUDA_ROOT=$d; break; fi
  done
fi
if [ -n "${CUDA_ROOT:-}" ]; then export PATH="$CUDA_ROOT/bin:$PATH"; fi
command -v nvcc >/dev/null || { echo "  nvcc not found: set CUDA_ROOT to the CUDA toolkit, or use CG_REF=0"; exit 1; }
echo "$(dirname "$(dirname "$(command -v nvcc)")")" > "$REF/cuda_root"
if [ ! -f "$REF/venv/bin/activate" ]; then
  ${PYTHON:-python3} -m venv "$REF/venv"
fi
. "$REF/venv/bin/activate"
python -m pip install -q -U pip
python -m pip install -q numpy scipy pycuda
python - <<EOF
import sys
sys.path.insert(0, "$REF/PyHierarchicalTsetlinMachineCUDA")
import numpy, scipy, pycuda, pycuda.autoprimaryctx
import pycuda.driver as drv
import PyHierarchicalTsetlinMachineCUDA.tm as tm
print(f"  reference environment: python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, "
      f"pycuda {pycuda.VERSION_TEXT}, GPU {drv.Device(0).name()}")
EOF
