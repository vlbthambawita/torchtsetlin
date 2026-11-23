#!/bin/bash
# The reference library (arm `ref`): cair/PyHierarchicalTsetlinMachineCUDA at a pinned commit, in
# its own environment (.cache/ref/venv: numpy, scipy, pycuda). Called by setup.sh; safe to re-run.
# pycuda compiles against a CUDA toolkit. The newest toolkit that the driver supports and whose nvcc
# works with this machine's compiler is chosen automatically; CUDA_ROOT=<toolkit> forces one.
# Only needed with CG_REF=1; without it the reference runs are not part of the job list.
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
# Choose a CUDA toolkit whose nvcc works with this machine's compiler and whose version the
# driver supports. Clusters often have an old toolkit first on PATH (CUDA 10.1 refuses gcc > 8).
# Candidates: CUDA_ROOT (if set, it is the only one), nvcc on PATH, and the usual install places.
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\)\.\([0-9]*\).*/\2.\3/p' | head -1)
works() {   # works <toolkit root>: nvcc preprocesses a CUDA source with the host compiler
  local t; t=$(mktemp --suffix=.cu)
  echo '#include <cuda_runtime.h>' > "$t"
  "$1/bin/nvcc" --preprocess "$t" --compiler-options -P > /dev/null 2>&1; local rc=$?
  rm -f "$t"; return $rc
}
version_of() { "$1/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1; }
CANDS=()
if [ -n "${CUDA_ROOT:-}" ]; then
  CANDS=("$CUDA_ROOT")
else
  if command -v nvcc >/dev/null; then CANDS+=("$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"); fi
  for d in /cm/shared/apps/cuda*/toolkit/* /usr/local/cuda-* /usr/local/cuda /opt/cuda*; do
    if [ -x "$d/bin/nvcc" ]; then CANDS+=("$(readlink -f "$d")"); fi
  done
fi
BEST=""; BESTV=0
for d in "${CANDS[@]}"; do
  v=$(version_of "$d"); [ -n "$v" ] || continue
  key=$(( ${v%%.*} * 100 + ${v##*.} ))
  if [ -n "$DRIVER_CUDA" ] && [ "$key" -gt $(( ${DRIVER_CUDA%%.*} * 100 + ${DRIVER_CUDA##*.} )) ]; then
    echo "  CUDA $v at $d: newer than the driver supports ($DRIVER_CUDA), skipped"; continue
  fi
  if ! works "$d"; then echo "  CUDA $v at $d: nvcc does not work with this compiler ($(gcc -dumpversion 2>/dev/null)), skipped"; continue; fi
  if [ "$key" -gt "$BESTV" ]; then BEST=$d; BESTV=$key; fi
done
if [ -z "$BEST" ]; then
  echo "  no usable CUDA toolkit found (candidates: ${CANDS[*]:-none})."
  echo "  Load a newer one (e.g. 'module avail cuda', 'module load cuda12.2/toolkit') or set CUDA_ROOT=<toolkit>,"
  echo "  then re-run ./setup.sh; or run without the reference library: CG_REF=0."
  exit 1
fi
echo "  CUDA toolkit for the reference library: $(version_of "$BEST") at $BEST"
export CUDA_ROOT=$BEST PATH="$BEST/bin:$PATH" LD_LIBRARY_PATH="$BEST/lib64:${LD_LIBRARY_PATH:-}"
# pycuda is compiled against one toolkit: rebuild the environment when the toolkit changed.
if [ -f "$REF/cuda_root" ] && [ "$(cat "$REF/cuda_root")" != "$BEST" ]; then
  echo "  toolkit changed (was $(cat "$REF/cuda_root")): rebuilding the reference environment"
  rm -rf "$REF/venv"
fi
echo "$BEST" > "$REF/cuda_root"
if [ ! -f "$REF/venv/bin/activate" ]; then
  ${PYTHON:-python3} -m venv "$REF/venv"
fi
. "$REF/venv/bin/activate"
python -m pip install -q -U pip
python -m pip install -q numpy scipy
python -m pip install -q --no-cache-dir pycuda
python - <<EOF
import sys
sys.path.insert(0, "$REF/PyHierarchicalTsetlinMachineCUDA")
import numpy, scipy, pycuda, pycuda.autoprimaryctx
import pycuda.driver as drv
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from pycuda.compiler import SourceModule
SourceModule("__global__ void f(float *a) { a[0] = 1; }")      # nvcc works end to end
print(f"  reference environment: python {sys.version.split()[0]}, numpy {numpy.__version__}, scipy {scipy.__version__}, "
      f"pycuda {pycuda.VERSION_TEXT}, GPU {drv.Device(0).name()}")
EOF
