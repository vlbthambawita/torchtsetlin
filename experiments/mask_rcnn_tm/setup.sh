#!/bin/bash
# One-time setup on the GPU server (PLAN_KVASIR §12). Safe to re-run.
#   ./setup.sh        Kvasir-SEG is read from /work/vajira/DATA/Kvasir-SEG (images/ and masks/);
#                     KV_DATA=/other/path ./setup.sh overrides it
#   1. Python environment (.venv): PyTorch (the build that fits the driver), torchvision, this
#      repository's torchtsetlin (editable), pycocotools, wandb, AlbumentationsX, ...
#   2. GPU check: every visible GPU runs a small update on the fast CUDA path.
#   3. Data: every image and mask is checked against the SHA-256 of the manifest, then the
#      128 px cache, the instance statistics and the panel manifest are built (runners never
#      build shared caches).
#   4. The COCO weights of baseline (i) are downloaded.
#   5. W&B login check (project TM_KVASIR).
#   6. Tests, and a smoke test of every kind of job on GPU slot 0.
# Knobs: KV_VENV=none (use the active python), KV_SKIP_INSTALL=1, TORCH_INDEX=<wheel index>,
# PYTHON=python3.12, KV_SKIP_SMOKE=1.
# PyTorch build: the default wheel needs a CUDA 13 driver. When the driver is older (the V100
# server: CUDA 12.2) the CUDA 12.6 build is installed instead; TORCH_INDEX overrides.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)
KV_VENV=${KV_VENV:-$PROG_DIR/.venv}
if [ -n "${KV_DATA:-}" ]; then echo "$KV_DATA" > .kv_data; fi

echo "== 1/6 Python environment"
if [ "$KV_VENV" != none ] && [ ! -f "$KV_VENV/bin/activate" ]; then
  PY=${PYTHON:-python3}
  echo "  creating $KV_VENV with $PY ($($PY -V 2>&1))"
  "$PY" -m venv "$KV_VENV"
fi
. ./env.sh
DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9][0-9]*\)\.\([0-9]*\).*/\2.\3/p' | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
if [ -z "${TORCH_INDEX:-}" ] && [ -n "$DRIVER_CUDA" ] && [ "${DRIVER_CUDA%%.*}" -lt 13 ]; then
  TORCH_INDEX=https://download.pytorch.org/whl/cu126
  echo "  ${GPU_NAME:-GPU} driver supports CUDA $DRIVER_CUDA: using the CUDA 12.6 PyTorch build ($TORCH_INDEX)"
fi
if [ "${KV_SKIP_INSTALL:-0}" != 1 ]; then
  python -m pip install -q -U pip
  python -m pip install -q torch torchvision ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  HAVE=$(python -c "import torch; print((torch.version.cuda or '0').split('.')[0])")
  if [ -n "$DRIVER_CUDA" ] && [ "$HAVE" -gt "${DRIVER_CUDA%%.*}" ]; then
    echo "  the installed PyTorch is built for CUDA $HAVE, the driver supports $DRIVER_CUDA: reinstalling"
    python -m pip install -q --force-reinstall torch torchvision ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  fi
  python -m pip install -q -e "$REPO_DIR" numpy scipy pillow matplotlib wandb "pycocotools>=2.0.8" opencv-python-headless pytest
  # AlbumentationsX needs Python >= 3.11; older Pythons get the last MIT release (same API here)
  if python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    python -m pip install -q "albumentationsx[headless]" || python -m pip install -q "albumentations==2.0.8"
  else
    python -m pip install -q "albumentations==2.0.8"
  fi
fi
python - <<'EOF'
import sys, numpy, scipy, torch, torchvision, torchtsetlin, albumentations
print(f"  python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), torchvision "
      f"{torchvision.__version__}, torchtsetlin {torchtsetlin.__version__}, albumentations {albumentations.__version__}")
EOF

echo "== 2/6 GPU check"
python - <<'EOF'
import sys, warnings
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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = tt.TsetlinMachine(432, 2, 300, T=60, s=10.0, weighted=True, n_states=100).to(dev)
        x, y = torch.rand(128, 432, device=dev) > 0.5, torch.randint(0, 2, (128,), device=dev)
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
[ "$NGPU" -ge 4 ] || echo "  $NGPU GPU(s) visible here. One GPU per tmux session: run setup.sh once (any session), then
  run_gpuK.sh in session K (K = 0..3); each runner uses its session's GPU."

echo "== 3/6 data (hash check, 128 px cache, panel manifest)"
( cd code && python data.py ) | sed 's/^/  /'

echo "== 4/6 COCO weights of baseline (i)"
python - <<'EOF' || echo "  could not download the COCO weights: baseline (i) will fail until the server has them in ~/.cache/torch/hub/checkpoints"
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
MaskRCNN_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=False)
print("  maskrcnn_resnet50_fpn COCO_V1 weights are in the torch hub cache")
EOF

echo "== 5/6 W&B"
if python - <<'EOF'
import os, sys, wandb
try:
    ok = wandb.login(host=os.environ["WANDB_BASE_URL"], anonymous="never", timeout=30, verify=True)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
    ok = False
sys.exit(0 if ok else 1)
EOF
then echo "  logged in to $WANDB_BASE_URL (entity simulamet_mlc, project $KV_WANDB_PROJECT)"
else
  echo "  NOT logged in. Run:  wandb login --host https://api.forge.coreweave.com   (then re-run ./setup.sh)"
  echo "  Without a connection use KV_WANDB=offline (wandb sync later) or KV_WANDB=0; records and panel"
  echo "  figures are written to results/ and figures/ either way."
fi

mkdir -p logs
DEV0=$(./gpu_device.sh 0)
if [ "${KV_SKIP_SMOKE:-0}" = 1 ]; then echo "== 6/6 tests and smoke test skipped"; exit 0; fi
echo "== 6/6 tests and smoke test on GPU slot 0 (logs/tests.log, logs/smoke.log, a few minutes)"
CUDA_VISIBLE_DEVICES=$DEV0 python -m pytest tests -q -p no:cacheprovider > logs/tests.log 2>&1 \
  || { tail -30 logs/tests.log; echo "tests FAILED (logs/tests.log)"; exit 1; }
tail -1 logs/tests.log | sed 's/^/  tests: /'
rm -f results/runs/*smoke* .cache/props/smoke.pt
rm -rf models/smoke-* models/*smoke* .cache/claims
CUDA_VISIBLE_DEVICES=$DEV0 KV_WANDB=${KV_SMOKE_WANDB:-0} python code/queue_runner.py --gpu 0 --phases smoke > logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
python - <<'EOF'
import sys
sys.path.insert(0, "code")
import jobs
from common import read_json, record_path
bad = []
for j in jobs.smoke_jobs():
    r = read_json(record_path(j["tag"])) or {}
    st = r.get("status")
    b = r.get("best")
    print(f"  {j['kind']:8s} {j['tag'][:70]:70s} {st}" + (f"  {b:.3f}" if isinstance(b, float) else ""))
    if st != "done":
        bad.append(j["tag"])
if bad:
    sys.exit(f"smoke test failed: {bad} (see logs/smoke.log, logs/runs/)")
EOF
CUDA_VISIBLE_DEVICES=$DEV0 python -m pytest tests/test_after_smoke.py -q -p no:cacheprovider >> logs/tests.log 2>&1 \
  || { tail -30 logs/tests.log; echo "tests after the smoke test FAILED (logs/tests.log)"; exit 1; }
tail -1 logs/tests.log | sed 's/^/  tests after the smoke test: /'
cat <<EOF

Setup done. Start one runner per GPU / tmux session (each resumes where it stopped if re-run):
  ./run_gpu0.sh    ./run_gpu1.sh    ./run_gpu2.sh    ./run_gpu3.sh
The default phase is 'price' (3 epochs of everything + the oracle ceilings); then ./status.sh --price
shows the projected cost. Open the next phases with the same value in all four sessions:
  KV_PHASES=base,p1 ./run_gpuK.sh    then p2, alb, p3, final  (read the gate in between: README.md)
Progress: ./status.sh      Report: python code/report.py      Stop: ./stop.sh  (or ./stop.sh --now)
EOF
