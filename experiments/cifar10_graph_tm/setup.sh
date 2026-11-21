#!/bin/bash
# One-time setup on the GPU server (plan §5, phase 0). Safe to re-run.
#   1. Python environment. An environment of an earlier CIFAR-10 programme on this server is
#      reused (cifar10_batch_clauses, cifar10_conv_reg, cifar10_joint_teams); otherwise .venv is
#      created here. Needed: PyTorch, this repository's torchtsetlin (editable, so records carry
#      the git sha), numpy, wandb, matplotlib, OpenCV (the paper arm's adaptive threshold).
#   2. GPU check: every visible GPU is listed with its free memory.
#   3. CIFAR-10: download, md5/sha256 check, 45 000 / 5 000 split (shared with cifar10_batch_clauses).
#   4. W&B login check (https://forge.coreweave.com/wandb/simulamet_mlc/TM_CIFAR).
#   5. Phase-0 checks (depth 1 = conv coalesced TM bit for bit; fast CUDA path at depth 1-3) and
#      the smoke test on one GPU: five tiny runs with figures, the resume check, a model reload.
# Knobs: CGT_VENV=<path>|none, CGT_SKIP_INSTALL=1, TORCH_INDEX=<url> (default: the CUDA 12.6
# wheels, which support the V100), PYTHON=python3.11, CGT_SKIP_SMOKE=1, CGT_SMOKE_GPU=<index>
# (default 0), CGT_CIFAR_DIR=<folder with cifar-10-python.tar.gz>.
set -euo pipefail
cd "$(dirname "$0")"
PROG_DIR=$PWD
REPO_DIR=$(cd ../.. && pwd)
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

echo "== 1/5 Python environment"
. ./env.sh
if [ "$CGT_VENV" != none ] && [ ! -f "$CGT_VENV/bin/activate" ]; then
  echo "  creating $CGT_VENV with ${PYTHON:-python3}"
  "${PYTHON:-python3}" -m venv "$CGT_VENV"
  . ./env.sh
fi
echo "  environment: $CGT_VENV"
if [ "${CGT_SKIP_INSTALL:-0}" != 1 ]; then
  python -c 'import torch' 2>/dev/null || { python -m pip install -q -U pip; python -m pip install -q torch --index-url "$TORCH_INDEX"; }
  python -c 'import torchtsetlin, numpy, wandb' 2>/dev/null || python -m pip install -q -e "$REPO_DIR" numpy wandb
  python -c 'import matplotlib' 2>/dev/null || python -m pip install -q matplotlib
  python -c 'import cv2' 2>/dev/null || python -m pip install -q opencv-python-headless
fi
python - <<'PYEOF'
import sys
import cv2, matplotlib, torch, torchtsetlin
print(f"  python {sys.version.split()[0]}, torch {torch.__version__} (CUDA {torch.version.cuda}), "
      f"torchtsetlin {torchtsetlin.__version__} ({torchtsetlin.__file__}), opencv {cv2.__version__}, "
      f"matplotlib {matplotlib.__version__}")
if not hasattr(torchtsetlin, "GraphTsetlinMachine"):
    sys.exit("this torchtsetlin has no GraphTsetlinMachine (needs >= 0.7.0): git pull, then re-run ./setup.sh")
PYEOF

echo "== 2/5 GPU check"
python - <<'PYEOF'
import sys
import torch
n = torch.cuda.device_count()
if n == 0:
    sys.exit("no CUDA GPU visible")
arch = torch.cuda.get_arch_list()
for i in range(n):
    cap = "sm_%d%d" % torch.cuda.get_device_capability(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 2**30
    free = torch.cuda.mem_get_info(i)[0] / 2**30
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.0f} GiB ({free:.0f} GiB free), {cap}")
    if cap not in arch:
        sys.exit(f"this torch build has no kernels for {cap} ({arch}); set TORCH_INDEX to a build that has")
    if free < 0.5 * mem:
        print(f"  NOTE: GPU {i} is in use by another process; do not start a runner on it")
print(f"  {n} GPU(s) visible to this shell")
PYEOF

echo "== 3/5 CIFAR-10"
python ../cifar10_batch_clauses/code/data.py ${CGT_CIFAR_DIR:+--from-dir "$CGT_CIFAR_DIR"}

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

if [ "${CGT_SKIP_SMOKE:-0}" = 1 ]; then echo "== 5/5 checks and smoke test skipped"; exit 0; fi
G=${CGT_SMOKE_GPU:-0}
echo "== 5/5 phase-0 checks and smoke test on GPU $G (log: logs/smoke.log, about 5 min)"
mkdir -p logs
DEV=$(./gpu_device.sh "$G")
export CGT_OUTER_CVD=${CUDA_VISIBLE_DEVICES:-}
rm -rf results/smoke results/resume_check models/smoke figures/smoke
CUDA_VISIBLE_DEVICES=$DEV python code/checks.py 2>&1 | tee logs/smoke.log | grep -E "^  " \
  || { echo "  phase-0 checks FAILED (logs/smoke.log, results/checks.json): do not start the runs"; tail -5 logs/smoke.log; exit 1; }
[ "${PIPESTATUS[0]}" = 0 ] || { echo "  phase-0 checks FAILED (results/checks.json): do not start the runs"; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV python code/queue_runner.py --gpu "$G" --phases smoke >> logs/smoke.log 2>&1 \
  || { tail -20 logs/smoke.log; exit 1; }
CUDA_VISIBLE_DEVICES=$DEV python code/resume_check.py >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
( cd code && CUDA_VISIBLE_DEVICES=$DEV python viz.py ../models/smoke/gtm_d3_c500_m64_rewired_s0.pt --device cuda:0 ) \
  >> logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
python - <<'PYEOF'
import sys
sys.path.insert(0, "code")
from common import FIGURES, RESULTS, jobs, read_json
bad = []
for j in jobs("smoke"):
    r = read_json(RESULTS / "smoke" / f"{j['tag']}.json")
    st = r and r.get("status")
    figs = all((FIGURES / "smoke" / j["tag"] / f"e002_{k}.png").exists() for k in ("panel", "health", "rules", "curves"))
    acc = "" if st != "done" else (f"  test {r['test_acc_last']:.1f} % after 2 epochs, messages "
                                   f"{100 * r['message_literal_share']:.0f} % of literals, figures {'ok' if figs else 'MISSING'}")
    print(f"  {j['tag']:34s} {st}{acc}")
    if st != "done" or not figs:
        bad.append(j["tag"])
rc = read_json(RESULTS / "smoke" / "resume_check.json")
print(f"  resume check: {'identical' if rc and rc['identical'] else 'DIFFERENT'}")
if bad or not (rc and rc["identical"]):
    sys.exit(f"smoke test failed: {bad or 'resume check'} (see logs/smoke.log, logs/runs/)")
PYEOF
cat <<MSG

Setup done. Start one runner per GPU / tmux session (each resumes where it stopped if re-run):
  ./run_gpu.sh 0    ./run_gpu.sh 1    ./run_gpu.sh 2    ./run_gpu.sh 3
They run: pricing and the cost gate -> the paper-setting runs and the pilot (T and s) ->
the depth study, longest first.
Progress: ./status.sh      Results: python code/report.py (writes REPORT.md)
Stop: ./stop.sh (after current runs) or ./stop.sh --now
MSG
