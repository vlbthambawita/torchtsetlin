#!/usr/bin/env bash
# Phase 0: one isolated virtualenv per external TM library (PLAN.md §2, rule P7).
#   code/setup_envs.sh [lib ...]      libs: pytm pytmpar pytmcuda tmu gt   (default: all)
# PyCUDA venvs (pytmcuda, tmu) must run with LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:
# pycuda builds against the system libstdc++, the conda python would load an older one.
# Each venv gets the library at its pinned version plus the shared adapter deps.
# Logs go to .cache/logs/setup_<lib>.log; the outcome (ok / error) to results/install_<lib>.json
# is written by code/check_envs.py.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="${BASE_PYTHON:-/home/vajira/anaconda3/envs/pytorch2/bin/python}"   # CPython 3.10.11
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.6}"
export PATH="$CUDA_ROOT/bin:$PATH" CUDA_ROOT
# numpy 2.2.6 everywhere except tmu 0.8.3, which fails under numpy 2
# (OverflowError: np.uint32(~0) in clause_bank.initialize_clauses) -> 1.26.4, as in multi_convtm.
declare -A NUMPY=( [tmu]="numpy==1.26.4" )
COMMON=("psutil==5.9.8" "nvidia-ml-py==13.615.71" "setuptools<81" "wheel")
declare -A PKG=(
  [pytm]="pyTsetlinMachine==0.6.6"
  [pytmpar]="pyTsetlinMachineParallel==0.2.1"
  [pytmcuda]="PyTsetlinMachineCUDA==0.1.9 pycuda==2026.1"
  [tmu]="tmu==0.8.3 pycuda==2026.1"
  [gt]="green-tsetlin==1.1.0"
)
# build-time requirements of sdists installed with --no-build-isolation
declare -A BUILD=( [gt]="pybind11==3.0.1 py-cpuinfo==9.0.0" )
LIBS=("$@"); [ ${#LIBS[@]} -eq 0 ] && LIBS=(pytm pytmpar pytmcuda tmu gt)
mkdir -p "$HERE/.cache/logs"
for lib in "${LIBS[@]}"; do
  venv="$HERE/.cache/venv-$lib"; log="$HERE/.cache/logs/setup_$lib.log"
  {
    echo "== $lib  $(date -Is)  python=$PY  numpy=${NUMPY[$lib]:-numpy==2.2.6}  cuda=$CUDA_ROOT"
    [ -x "$venv/bin/python" ] || "$PY" -m venv "$venv"
    "$venv/bin/python" -m pip install -q --upgrade pip
    "$venv/bin/python" -m pip install -q "${NUMPY[$lib]:-numpy==2.2.6}" "${COMMON[@]}"
    [ -n "${BUILD[$lib]:-}" ] && "$venv/bin/python" -m pip install -q ${BUILD[$lib]}
    # sdists build against the venv's numpy/setuptools, so no build isolation
    "$venv/bin/python" -m pip install --no-build-isolation ${PKG[$lib]}
    echo "== exit $?"
  } >"$log" 2>&1 &
done
wait
echo "done: ${LIBS[*]}"
