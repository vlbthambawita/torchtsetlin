#!/bin/bash
# One-time setup on the GPU server (plan §5, phase 0). Safe to re-run.
#   1. Python environment. If cifar10_joint_teams was set up on this server, its .venv is reused
#      (it has everything). Otherwise a .venv is created here with PyTorch, this repository's
#      torchtsetlin (editable, so records carry the git sha), AlbumentationsX, matplotlib, wandb.
#   2. GPU check: every visible GPU runs a small update with drop clause on the fast CUDA path.
#   3. CIFAR-10: download, md5/sha256 check, 45 000 / 5 000 split (shared with cifar10_batch_clauses).
#   4. W&B login check (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
#   5. Smoke test on one GPU: two tiny runs (with and without augmentation + drop clause) with
#      figures, the resume check, and a reload of a saved model.
# Knobs: CCR_VENV=<path>|none, CCR_SKIP_INSTALL=1, TORCH_INDEX=<url>, PYTHON=python3.12,
# CCR_SKIP_SMOKE=1, CCR_SMOKE_GPU=<index> (default 0), CCR_CIFAR_DIR=<folder with cifar-10-python.tar.gz>.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)

echo "== 1/5 Python environment"
. ./env.sh
if [ "$CCR_VENV" != none ] && [ ! -f "$CCR_VENV/bin/activate" ]; then
  PY=${PYTHON:-}
  if [ -z "$PY" ]; then   # AlbumentationsX needs Python >= 3.11
    for c in python3.13 python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY=$c; break; fi
    done
  fi
  PY=${PY:-python3}
  echo "  creating $CCR_VENV with $PY ($($PY -V 2>&1))"
  "$PY" -m venv "$CCR_VENV"
  . ./env.sh
  NEW_VENV=1
fi
echo "  environment: $CCR_VENV"
if [ "${CCR_SKIP_INSTALL:-0}" != 1 ] && { [ "${NEW_VENV:-0}" = 1 ] || ! python -c 'import torch, torchtsetlin, albumentations, matplotlib, wandb' 2>/dev/null; }; then
  python -m pip install -q -U pip
  python -m pip install -q torch ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  python -m pip install -q -e "$REPO_DIR" numpy wandb matplotlib
  if python -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    python -m pip install -q "albumentationsx[headless]"
  else
    echo "  Python < 3.11: AlbumentationsX is not installable; using albumentations==2.0.8 (same transforms)"
    python -m pip install -q "albumentations==2.0.8" opencv-python-headless
  fi
fi
python - <<'PYEOF'
import sys
sys.path.insert(0, "code")
import augment                                   # sets the telemetry switches before the import below
import albumentations, cv2, matplotlib, torch, torchtsetlin
print(f"  python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchtsetlin {torchtsetlin.__version__} ({torchtsetlin.__file__}), albumentations "
      f"{albumentations.__version__}, opencv {cv2.__version__}")
assert augment.telemetry_off(), "augmentation telemetry is not switched off"
augment.build("geo+col")
print("  augmentation policy builds; telemetry off")
PYEOF

echo "== 2/5 GPU check"
python - <<'PYEOF'
import sys, warnings
sys.path.insert(0, "code")
import torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit("no CUDA GPU visible")
from torchtsetlin import _triton
from train import build_model
arch = torch.cuda.get_arch_list()
for i in range(n):
    name = torch.cuda.get_device_name(i)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 2**30
    free = torch.cuda.mem_get_info(i)[0] / 2**30
    print(f"  GPU {i}: {name}, {mem:.0f} GiB ({free:.0f} GiB free), {cap}")
    if cap not in arch:
        sys.exit(f"this torch build has no kernels for {cap} ({arch}); set TORCH_INDEX to a build that has")
    if free < 0.5 * mem:
        print(f"  NOTE: GPU {i} is in use by another process; do not start a runner on it")
        continue
    dev = torch.device(f"cuda:{i}")
    torch.cuda.set_device(dev)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = build_model(50, 100, 5.0, 0.25).to(dev)
        x = torch.rand(32, 24, 32, 32, device=dev) > 0.5
        y = torch.randint(0, 10, (32,), device=dev)
        for _ in range(5):
            m.update(x, y)
        m.eval()
        m(x)
        torch.cuda.synchronize(dev)
    bad = {str(k.message)[:160] for k in w if "kernel" in str(k.message) and "failed" in str(k.message)}
    for msg in bad:
        print("  WARNING:", msg)
    if not _triton.available(dev) or bad:
        print("  WARNING: the fast CUDA path (Triton kernels) is not active on this GPU; training still "
              "works but is several times slower.")
print(f"  {n} GPU(s) visible")
PYEOF
echo "  CPU cores: $(nproc --all) (each augmenting runner uses a few worker processes; CCR_WORKERS=<n> overrides)"

echo "== 3/5 CIFAR-10"
python ../cifar10_batch_clauses/code/data.py ${CCR_CIFAR_DIR:+--from-dir "$CCR_CIFAR_DIR"}

echo "== 4/5 W&B"
if python - <<'PYEOF'
import os, sys, wandb
try:
    ok = wandb.login(host=os.environ["WANDB_BASE_URL"], anonymous="never", timeout=30, verify=True)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")
    ok = False
sys.exit(0 if ok else 1)
PYEOF
then echo "  logged in to $WANDB_BASE_URL"
else
  echo "  NOT logged in. Run:  wandb login --host https://api.forge.coreweave.com   (then re-run ./setup.sh)"
  echo "  (training still works without W&B; records are written to results/ either way)"
fi

if [ "${CCR_SKIP_SMOKE:-0}" = 1 ]; then echo "== 5/5 smoke test skipped"; exit 0; fi
G=${CCR_SMOKE_GPU:-0}
echo "== 5/5 smoke test on GPU $G (log: logs/smoke.log, about 3 min)"
mkdir -p logs
DEV=$(./gpu_device.sh "$G")
export CCR_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-}
rm -rf results/smoke models/smoke figures/smoke
CUDA_VISIBLE_DEVICES=$DEV python code/queue_runner.py --gpu "$G" --phases smoke > logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV python code/resume_check.py >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
( cd code && CUDA_VISIBLE_DEVICES=$DEV python viz.py ../models/smoke/conv_c1000_b32_geocol_dc0.25_s0.pt --device cuda:0 ) \
  >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
python - <<'PYEOF'
import sys
sys.path.insert(0, "code")
from common import FIGURES, RESULTS, jobs, read_json
bad = []
for j in jobs("smoke"):
    r = read_json(RESULTS / "smoke" / f"{j['tag']}.json")
    st = r and r.get("status")
    figs = all((FIGURES / "smoke" / j["tag"] / f).exists()
               for f in ("e002_panel.png", "e002_health.png", "e002_curves.png", "best_panel.png", "best_health.png"))
    print(f"  {j['tag']:34s} {st}" + (f"  test {r['test_acc_at_best']:.1f} %  train (clean) "
                                      f"{r['train_eval_acc_at_best']:.1f} %  figures {'ok' if figs else 'MISSING'}"
                                      if st == "done" else ""))
    if st != "done" or not figs:
        bad.append(j["tag"])
rc = read_json(RESULTS / "smoke" / "resume_check.json")
print(f"  resume check: {'identical' if rc and rc['identical'] else 'DIFFERENT'}")
if bad or not (rc and rc["identical"]):
    sys.exit(f"smoke test failed: {bad or 'resume check'} (see logs/smoke.log, logs/runs/)")
PYEOF
cat <<MSG

Setup done. Start one runner per GPU (each resumes where it stopped if re-run):
  ./run_gpu.sh 0    ./run_gpu.sh 3    ./run_gpu.sh 1
They run: pricing (8 timings) -> 64 000-clause base and aug runs + the 8 000-clause screen -> p* ->
the rest of the factorial, longest first.
Progress: ./status.sh      Results: python code/report.py (writes REPORT.md)
Stop: ./stop.sh (after current runs) or ./stop.sh --now
MSG
