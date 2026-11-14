#!/bin/bash
# One-time setup on the V100 server (plan §7, Phase 0). Safe to re-run.
#   1. Python environment: venv with PyTorch (CUDA 12.6 wheels, which support the V100's sm_70),
#      this repository's torchtsetlin (editable, so records carry the git sha), numpy, wandb.
#   2. GPU check: every GPU is visible, is a V100, and runs a small conv/flat/conv3 update.
#   3. CIFAR-10: download, md5/sha256 check, .cache/data/cifar10_uint8.npz, results/split.json.
#   4. W&B login check (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
#   5. Smoke test on GPU 0 (18 tiny runs, 1 epoch on 2 000 images) + resume check.
# Environment knobs: CBC_VENV=none (use the active python), CBC_SKIP_INSTALL=1, TORCH_INDEX=<url>,
# PYTHON=python3.11, CBC_SKIP_SMOKE=1.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)
CBC_VENV=${CBC_VENV:-$PROG_DIR/.venv}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

echo "== 1/5 Python environment"
if [ "$CBC_VENV" != none ] && [ ! -f "$CBC_VENV/bin/activate" ]; then
  ${PYTHON:-python3} -m venv "$CBC_VENV"
fi
. ./env.sh
if [ "${CBC_SKIP_INSTALL:-0}" != 1 ]; then
  python -m pip install -q -U pip
  python -m pip install -q torch --index-url "$TORCH_INDEX"
  python -m pip install -q -e "$REPO_DIR" numpy wandb
fi
python - <<'EOF'
import torch, torchtsetlin, sys
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchtsetlin {torchtsetlin.__version__}")
EOF

echo "== 2/5 GPU check"
python - <<'EOF'
import os, sys, warnings
import torch, torchtsetlin as tt
n = torch.cuda.device_count()
if n == 0:
    sys.exit("no CUDA GPU visible")
arch = torch.cuda.get_arch_list()
for i in range(n):
    name = torch.cuda.get_device_name(i)
    cap = "sm_%d%d" % torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 2**30
    print(f"  GPU {i}: {name}, {mem:.0f} GiB, {cap}")
    if cap not in arch:
        sys.exit(f"this torch build has no kernels for {cap} ({arch}); set TORCH_INDEX to a build that has")
    if "V100" not in name:
        print("  WARNING: not a V100; the plan's timings assume four identical V100s")
    dev = torch.device(f"cuda:{i}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = torch.rand(64, 3, 8, 32, 32, device=dev) > 0.5
        y = torch.randint(0, 10, (64,), device=dev)
        for m, xi in ((tt.ConvTsetlinMachine(10, 50, 100, 5.0, patch_size=3, position_encoding=True,
                                             input_shape=(24, 32, 32), weighted=True), x.reshape(64, 24, 32, 32)),
                      (tt.Conv3dTsetlinMachine(10, 50, 100, 5.0, patch_size=(8, 3, 3), position_encoding=True,
                                               input_shape=(3, 8, 32, 32), weighted=True), x),
                      (tt.TsetlinMachine(9216, 10, 50, 10, 10.0), x.reshape(64, -1)[:, :9216])):
            m = m.to(dev)
            for _ in range(4):
                m.update(xi[:32], y[:32])
            m.eval()
            m(xi)
        torch.cuda.synchronize(dev)
    for msg in {str(x.message) for x in w if "Triton" in str(x.message)}:
        print("  NOTE:", msg[:200])
print(f"  {n} GPU(s) OK")
EOF

# Each runner sees exactly one GPU (run_gpu.sh): check every slot the same way.
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
for K in $(seq 0 $((NGPU < 4 ? NGPU - 1 : 3))); do
  DEV=$(./gpu_device.sh "$K")
  CUDA_VISIBLE_DEVICES=$DEV python -c "import torch; print('  slot $K -> CUDA_VISIBLE_DEVICES=$DEV:', torch.cuda.get_device_name(0))" \
    || { echo "  slot $K (CUDA_VISIBLE_DEVICES=$DEV): no GPU visible to PyTorch"; exit 1; }
done
[ "$NGPU" -ge 4 ] || echo "  $NGPU GPU(s) visible here. One GPU per session (e.g. tmux): run setup.sh once, then
  run_gpuK.sh in session K (K = 0..3); each runner uses its session's GPU."

echo "== 3/5 CIFAR-10"
python code/data.py ${CBC_CIFAR_DIR:+--from-dir "$CBC_CIFAR_DIR"}

echo "== 4/5 W&B"
if python - <<'EOF'
import os, sys, wandb
host = os.environ["WANDB_BASE_URL"]
try:
    ok = wandb.login(host=host, anonymous="never", timeout=30, verify=True)
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

if [ "${CBC_SKIP_SMOKE:-0}" = 1 ]; then echo "== 5/5 smoke test skipped"; exit 0; fi
echo "== 5/5 smoke test on GPU 0 (log: logs/smoke.log)"
mkdir -p logs
DEV0=$(./gpu_device.sh 0)
CUDA_VISIBLE_DEVICES=$DEV0 python code/queue_runner.py --gpu 0 --arm smoke --any-gpu > logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV0 python code/resume_check.py >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
python - <<'EOF'
import sys
sys.path.insert(0, "code")
from common import RESULTS, jobs, read_json
bad = []
for j in jobs("smoke"):
    r = read_json(RESULTS / "smoke" / f"{j['tag']}.json")
    st = r and r.get("status")
    print(f"  {j['tag']:45s} {st}" + (f"  test {r['test_acc_at_best']:.1f}%  peak {r['peak_gpu_mib']/1024:.1f} GiB" if st == "done" else ""))
    if st not in ("done", "oom"):
        bad.append(j["tag"])
rc = read_json(RESULTS / "smoke" / "resume_check.json")
print(f"  resume check: {'identical' if rc and rc['identical'] else 'DIFFERENT'}")
if bad or not (rc and rc["identical"]):
    sys.exit(f"smoke test failed: {bad or 'resume check'} (see logs/smoke.log, logs/runs/)")
EOF
cat <<EOF

Setup done. Start one runner per GPU (each resumes where it stopped if re-run):
  ./run_gpu0.sh; ./run_gpu1.sh; ./run_gpu2.sh; ./run_gpu3.sh
Progress: ./status.sh      Results: python code/report.py (writes REPORT.md)
Stop: ./stop.sh (after current runs) or ./stop.sh --now
EOF
