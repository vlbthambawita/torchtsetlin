#!/bin/bash
# One-time setup on the GPU server (plan §8, phase 0b). Safe to re-run.
#   1. Python environment: venv (Python >= 3.11 if there is one) with PyTorch, this repository's
#      torchtsetlin (editable, so records carry the git sha), AlbumentationsX, OpenCV, matplotlib, wandb.
#   2. GPU check: every visible GPU runs a small 2D and 3D joint update on the fast CUDA path.
#   3. CIFAR-10: download, md5/sha256 check, 45 000 / 5 000 split (shared with cifar10_batch_clauses).
#   4. W&B login check (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
#   5. Correctness checks (joint update, 3D member, encoders, augmentation determinism).
#   6. Smoke test on GPU slot 0: every team at B = 16 and 1024 on 2 000 images, activation maps,
#      resume check, reload of a saved team.
# Environment knobs: CJT_VENV=none (use the active python), CJT_SKIP_INSTALL=1, TORCH_INDEX=<url>,
# PYTHON=python3.12, CJT_SKIP_SMOKE=1, CJT_CIFAR_DIR=<folder with cifar-10-python.tar.gz>.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)
CJT_VENV=${CJT_VENV:-$PROG_DIR/.venv}

echo "== 1/6 Python environment"
if [ "$CJT_VENV" != none ] && [ ! -f "$CJT_VENV/bin/activate" ]; then
  PY=${PYTHON:-}
  if [ -z "$PY" ]; then   # AlbumentationsX needs Python >= 3.11
    for c in python3.13 python3.12 python3.11 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY=$c; break; fi
    done
  fi
  PY=${PY:-python3}
  echo "  creating $CJT_VENV with $PY ($($PY -V 2>&1))"
  "$PY" -m venv "$CJT_VENV"
fi
. ./env.sh
if [ "${CJT_SKIP_INSTALL:-0}" != 1 ]; then
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
python - <<'EOF'
import os, sys
sys.path.insert(0, "code")
import augment                                   # sets the telemetry switches before the import below
import albumentations, cv2, torch, torchtsetlin
from importlib.metadata import PackageNotFoundError, version
lib = "albumentations"
try:
    lib = f"albumentationsx {version('albumentationsx')}"
except PackageNotFoundError:
    lib = f"albumentations {albumentations.__version__} (fallback, not AlbumentationsX)"
print(f"  python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchtsetlin {torchtsetlin.__version__}, {lib}, opencv {cv2.__version__}")
assert augment.telemetry_off(), "augmentation telemetry is not switched off"
for p in augment.POLICIES:
    augment.build(p)
print("  augmentation policies build; telemetry off")
EOF

echo "== 2/6 GPU check"
python - <<'EOF'
import sys, warnings
sys.path.insert(0, "code")
import torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit("no CUDA GPU visible")
from common import MEMBERS
from team import build_team
from torchtsetlin import _triton
arch = torch.cuda.get_arch_list()
for i in range(n):
    name = torch.cuda.get_device_name(i)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 2**30
    print(f"  GPU {i}: {name}, {mem:.0f} GiB, {cap}")
    if cap not in arch:
        sys.exit(f"this torch build has no kernels for {cap} ({arch}); set TORCH_INDEX to a build that has")
    dev = torch.device(f"cuda:{i}")
    torch.cuda.set_device(dev)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        team = build_team({k: dict(MEMBERS[k], clauses=50) for k in ("H3", "B3")}, 32, dev)
        x = torch.rand(32, 3, 8, 32, 32, device=dev) > 0.5
        y = torch.randint(0, 10, (32,), device=dev)
        for _ in range(5):
            team.update([x.reshape(32, 24, 32, 32), x], y)
        team.member_sums([x.reshape(32, 24, 32, 32), x])
        torch.cuda.synchronize(dev)
    bad = {str(m.message)[:160] for m in w if "kernel" in str(m.message) and "failed" in str(m.message)
           or "capture failed" in str(m.message)}
    for msg in bad:
        print("  WARNING:", msg)
    if not _triton.available(dev) or bad:
        print("  WARNING: the fast CUDA path (Triton kernels, CUDA graphs) is not active on this GPU; "
              "training still works but is several times slower.")
print(f"  {n} GPU(s) OK")
EOF
# Each runner sees exactly one GPU (run_gpu.sh): check every slot the same way.
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
for K in $(seq 0 $((NGPU < 4 ? NGPU - 1 : 3))); do
  DEV=$(./gpu_device.sh "$K")
  CUDA_VISIBLE_DEVICES=$DEV python -c "import torch; print('  slot $K -> CUDA_VISIBLE_DEVICES=$DEV:', torch.cuda.get_device_name(0))" \
    || { echo "  slot $K (CUDA_VISIBLE_DEVICES=$DEV): no GPU visible to PyTorch"; exit 1; }
done
[ "$NGPU" -ge 4 ] || echo "  $NGPU GPU(s) visible here. One GPU per tmux session: run setup.sh once (any session), then
  run_gpuK.sh in session K (K = 0..3); each runner uses its session's GPU."
echo "  CPU cores: $(nproc --all) (each runner uses a few augmentation workers; CJT_WORKERS=<n> overrides)"

echo "== 3/6 CIFAR-10"
python ../cifar10_batch_clauses/code/data.py ${CJT_CIFAR_DIR:+--from-dir "$CJT_CIFAR_DIR"}

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

if [ "${CJT_SKIP_SMOKE:-0}" = 1 ]; then echo "== 5/6, 6/6 checks and smoke test skipped"; exit 0; fi
mkdir -p logs
DEV0=$(./gpu_device.sh 0)
export CJT_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-}
echo "== 5/6 correctness checks on GPU slot 0 (log: logs/checks.log, ~4 min)"
{ CUDA_VISIBLE_DEVICES=$DEV0 JW_RESULTS=$PROG_DIR/.cache/joint_wide_tm_checks \
    python ../joint_wide_tm/code/check_joint.py --device cuda:0 \
  && CUDA_VISIBLE_DEVICES=$DEV0 python code/check_team.py --device cuda:0; } > logs/checks.log 2>&1 \
  || { grep -E "PASS|FAIL" logs/checks.log; tail -5 logs/checks.log; echo "checks FAILED (logs/checks.log)"; exit 1; }
grep -E "PASS|FAIL" logs/checks.log | sed 's/^/  /'

echo "== 6/6 smoke test on GPU slot 0 (log: logs/smoke.log, ~5 min)"
CUDA_VISIBLE_DEVICES=$DEV0 python code/queue_runner.py --gpu 0 --phases smoke > logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV0 python code/resume_check.py >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV0 python code/viz.py models/smoke/M_b16_full_s0.pt --device cuda:0 >> logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
python - <<'EOF'
import sys
sys.path.insert(0, "code")
from common import FIGURES, RESULTS, jobs, read_json
bad = []
for j in jobs("smoke"):
    r = read_json(RESULTS / "smoke" / f"{j['tag']}.json")
    st = r and r.get("status")
    fig = FIGURES / "activation" / "smoke" / j["tag"] / "e001.png"
    print(f"  {j['tag']:18s} {st}" + (f"  test {r['test_acc_at_best']:.1f}%  peak {r['peak_gpu_mib']/1024:.1f} GiB  "
                                      f"maps {'ok' if fig.exists() else 'MISSING'}" if st == "done" else ""))
    if st != "done" or not fig.exists():
        bad.append(j["tag"])
rc = read_json(RESULTS / "smoke" / "resume_check.json")
print(f"  resume check: {'identical' if rc and rc['identical'] else 'DIFFERENT'}")
if bad or not (rc and rc["identical"]):
    sys.exit(f"smoke test failed: {bad or 'resume check'} (see logs/smoke.log, logs/runs/)")
EOF
cat <<EOF

Setup done. Start one runner per GPU / tmux session (each resumes where it stopped if re-run):
  ./run_gpu0.sh    ./run_gpu1.sh    ./run_gpu2.sh    ./run_gpu3.sh
They run: pricing (cost table) -> augmentation pilot (3 runs) -> batch-size grid (14 runs).
Progress: ./status.sh      Results: python code/report.py (writes REPORT.md)
Stop: ./stop.sh (after current runs) or ./stop.sh --now
EOF
