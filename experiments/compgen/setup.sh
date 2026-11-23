#!/bin/bash
# One-time setup on the GPU server (plan §5, phase 0). Safe to re-run.
#   1. Python environment (.venv): PyTorch, this repository's torchtsetlin (editable, >= 0.7.4 for
#      the `literals` option), numpy, scipy, matplotlib, wandb.
#   2. GPU check: every visible GPU runs a small update on the fast CUDA path.
#   3. Data: the benchmark's generator for every seed (about 1 min per seed), pattern statistics.
#   4. W&B login check (https://forge.coreweave.com/wandb/simulamet_mlc/TM_COMPGEN).
#   5. Only with CG_REF=1: the reference library in its own environment (setup_ref.sh; needs a
#      CUDA toolkit with a working nvcc). Off by default: only the torchtsetlin arms are run.
#   6. Checks M0-M3 and a smoke test on GPU slot 0.
# Knobs: CG_VENV=none (use the active python), CG_SKIP_INSTALL=1, TORCH_INDEX=<wheel index>,
# PYTHON=python3.12, CG_SKIP_SMOKE=1, CG_SEEDS=0, CG_REF=1, CUDA_ROOT=<CUDA toolkit for pycuda>.
# PyTorch build: the default wheel needs a CUDA 13 driver. When the driver is older (the V100
# server: CUDA 12.2) the CUDA 12.6 build is installed instead (the build the earlier V100 runs
# used), and a wrong build left in .venv by an earlier setup is replaced. TORCH_INDEX overrides.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)
CG_VENV=${CG_VENV:-$PROG_DIR/.venv}

echo "== 1/6 Python environment"
if [ "$CG_VENV" != none ] && [ ! -f "$CG_VENV/bin/activate" ]; then
  PY=${PYTHON:-python3}
  echo "  creating $CG_VENV with $PY ($($PY -V 2>&1))"
  "$PY" -m venv "$CG_VENV"
fi
. ./env.sh
# The newest CUDA version the installed NVIDIA driver supports (nvidia-smi header), e.g. 12.2.
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\)\.\([0-9]*\).*/\2.\3/p' | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "${TORCH_INDEX:-}" ] && [ -n "$DRIVER_CUDA" ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  TORCH_INDEX=https://download.pytorch.org/whl/cu126
  echo "  ${GPU_NAME:-GPU} driver supports CUDA $DRIVER_CUDA: using the CUDA 12.6 PyTorch build ($TORCH_INDEX)"
fi
if [ "${CG_SKIP_INSTALL:-0}" != 1 ]; then
  python -m pip install -q -U pip
  python -m pip install -q torch ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  # An earlier setup may have left a build for a newer CUDA than the driver has: replace it.
  HAVE=$(python -c "import torch; print((torch.version.cuda or '0').split('.')[0])")
  if [ -n "$DRIVER_CUDA" ] && [ "$HAVE" -gt "${DRIVER_CUDA%%.*}" ]; then
    echo "  the installed PyTorch is built for CUDA $HAVE, the driver supports $DRIVER_CUDA: reinstalling"
    python -m pip install -q --force-reinstall torch ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  fi
  python -m pip install -q -e "$REPO_DIR" numpy scipy wandb matplotlib
fi
python - <<'EOF'
import sys, numpy, scipy, torch, torchtsetlin
print(f"  python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchtsetlin {torchtsetlin.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}")
import inspect
assert "literals" in inspect.signature(torchtsetlin.TsetlinMachineBase.__init__).parameters, \
    "this torchtsetlin has no `literals` option (needs >= 0.7.4): git pull"
EOF

echo "== 2/6 GPU check"
python - <<'EOF'
import sys, warnings
sys.path.insert(0, "code")
import torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit("no CUDA GPU visible")
import torchtsetlin as tt
from torchtsetlin import _triton
arch = torch.cuda.get_arch_list()
for i in range(n):
    name, cap = torch.cuda.get_device_name(i), "sm_%d%d" % torch.cuda.get_device_capability(i)
    print(f"  GPU {i}: {name}, {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f} GiB, {cap}")
    if cap not in arch:
        sys.exit(f"this torch build has no kernels for {cap} ({arch}); set TORCH_INDEX to a build that has")
    dev = torch.device(f"cuda:{i}")
    torch.cuda.set_device(dev)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for mode in ("batch", "sequential"):
            m = tt.TsetlinMachine(125, 2, 100, T=10, s=25.0, max_included_literals=100, literals="negated",
                                  feedback_mode=mode).to(dev)
            x, y = torch.rand(256, 125, device=dev) > 0.5, torch.randint(0, 2, (256,), device=dev)
            for _ in range(5):
                m.update(x, y)
        torch.cuda.synchronize(dev)
    bad = {str(m.message)[:160] for m in w if "failed" in str(m.message)}
    for msg in bad:
        print("  WARNING:", msg)
    if not _triton.available(dev) or bad:
        print("  WARNING: the fast CUDA path (Triton kernels, CUDA graphs) is not active on this GPU; "
              "training still works but is slower.")
print(f"  {n} GPU(s) OK")
EOF
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
for K in $(seq 0 $((NGPU < 4 ? NGPU - 1 : 3))); do
  DEV=$(./gpu_device.sh "$K")
  CUDA_VISIBLE_DEVICES=$DEV python -c "import torch; print('  slot $K -> CUDA_VISIBLE_DEVICES=$DEV:', torch.cuda.get_device_name(0))" \
    || { echo "  slot $K (CUDA_VISIBLE_DEVICES=$DEV): no GPU visible to PyTorch"; exit 1; }
done
[ "$NGPU" -ge 4 ] || echo "  $NGPU GPU(s) visible here. One GPU per tmux session: run setup.sh once (any session), then
  run_gpuK.sh in session K (K = 0..3); each runner uses its session's GPU."

echo "== 3/6 data (the reference generator, about 1 min per seed)"
( cd code && python data.py ) | sed 's/^/  /'

echo "== 4/6 W&B"
if python - <<'EOF'
import os, sys, wandb
try:
    ok = wandb.login(host=os.environ["WANDB_BASE_URL"], anonymous="never", timeout=30, verify=True)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
    ok = False
sys.exit(0 if ok else 1)
EOF
then echo "  logged in to $WANDB_BASE_URL"
else
  echo "  NOT logged in. Run:  wandb login --host https://api.forge.coreweave.com   (then re-run ./setup.sh)"
  echo "  (training still works without W&B; records are written to results/ either way)"
fi

mkdir -p logs
DEV0=$(./gpu_device.sh 0)
echo "== 5/6 reference library (arm ref)"
if [ "${CG_REF:-0}" != 1 ]; then
  echo "  skipped: only the torchtsetlin arms are run (CG_REF=1 adds the reference library)"
elif CUDA_VISIBLE_DEVICES=$DEV0 ./setup_ref.sh > logs/setup_ref.log 2>&1; then
  tail -n 2 logs/setup_ref.log
else
  tail -n 5 logs/setup_ref.log | sed 's/^/  /'
  echo "  The reference environment could not be built (logs/setup_ref.log). The torchtsetlin runs do not"
  echo "  need it: start the runners without CG_REF=1, or fix it (CUDA_ROOT=<toolkit with nvcc>) and re-run setup.sh."
fi

if [ "${CG_SKIP_SMOKE:-0}" = 1 ]; then echo "== 6/6 checks and smoke test skipped"; exit 0; fi
echo "== 6/6 checks and smoke test on GPU slot 0 (logs/checks.log, logs/smoke.log, about 3 min)"
( cd code && CUDA_VISIBLE_DEVICES=$DEV0 python checks.py --device cuda:0 ) > logs/checks.log 2>&1 \
  || { grep -E "PASS|FAIL" logs/checks.log; tail -5 logs/checks.log; echo "checks FAILED (logs/checks.log)"; exit 1; }
grep -E "PASS|FAIL" logs/checks.log | sed 's/^/  /'
CUDA_VISIBLE_DEVICES=$DEV0 python code/queue_runner.py --gpu 0 --phases smoke > logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
if [ "${CG_REF:-0}" = 1 ] && [ -x .cache/ref/venv/bin/python ]; then
  ( cd code && CUDA_VISIBLE_DEVICES=$DEV0 PATH="$(cat ../.cache/ref/cuda_root)/bin:$PATH" LD_LIBRARY_PATH="$(cat ../.cache/ref/cuda_root)/lib64:${LD_LIBRARY_PATH:-}" ../.cache/ref/venv/bin/python run_ref.py \
      --phase smoke --seed 0 --subset 4000 --max-epochs 2 ) >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
fi
python - <<'EOF'
import sys
sys.path.insert(0, "code")
from common import RESULTS, jobs, read_json, tag_of, with_ref
bad = []
tags = [j["tag"] for j in jobs("smoke")] + ([tag_of("ref", "seq", 0)] if with_ref() else [])
for t in tags:
    r = read_json(RESULTS / "smoke" / f"{t}.json")
    if r is None and t.startswith("ref"):
        print(f"  {t:14s} not run (no reference environment)")
        continue
    st = r and r.get("status")
    print(f"  {t:14s} {st}" + (f"  test {r['test_acc_last']:.1f} %  {r['train_s_per_epoch_mean']:.2f} s/epoch" if st == "done" else ""))
    if st != "done":
        bad.append(t)
if bad:
    sys.exit(f"smoke test failed: {bad} (see logs/smoke.log, logs/runs/)")
EOF
cat <<EOF

Setup done. Start one runner per GPU / tmux session (each resumes where it stopped if re-run):
  ./run_gpu0.sh    ./run_gpu1.sh    ./run_gpu2.sh    ./run_gpu3.sh
They run: price (3 epochs per cell) -> phase 1 (batching, 15 runs) -> phase 2 (standard TM, 12 runs).
Progress: ./status.sh      Results: python code/report.py (writes REPORT.md)
Stop: ./stop.sh (after current runs) or ./stop.sh --now
EOF
