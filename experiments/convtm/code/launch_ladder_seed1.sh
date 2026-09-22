#!/bin/bash
# Detached GPU-0 launcher: the device control, then the seed-1 ladder endpoints (DR-008 D2).
#
#   setsid nohup code/launch_ladder_seed1.sh > logs/ladder_seed1_launcher.log 2>&1 < /dev/null &
#
# `setsid` puts it in its own session so it survives the shell that started it; every wait is
# done INSIDE the GPU lock helper (queue_runner.py --wait), never by holding a turn open.
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/convtm || exit 1
export PYTHONUNBUFFERED=1
PY=$(command -v python)
echo "[launcher] started $(date -Is) pid=$$ ppid=$PPID  gpu 0 (RTX 3090)"

# ---------------------------------------------------------------------------------------
# 0. DEVICE CONTROL, ~10 min.  jobs/p3_ladder_seed1.txt buys the 20 000 point at seed 1 on
#    GPU 0, but its seed 0 ran on GPU 1.  DR-001 makes `device + seed` the reproducibility
#    key, so the device difference has to be controlled, not assumed away.  Three epochs of
#    the identical configuration at seed 0 on GPU 0 must reproduce the recorded seed-0 curve
#    (val 0.5152 / 0.5244 / 0.5800).  Written to results/devcontrol/, NOT to results/: it is
#    a control, it is not 50 epochs, and it must never be read as an arm.
# ---------------------------------------------------------------------------------------
echo "[launcher] === 0: device control (3 epochs, seed 0, GPU 0) === $(date -Is)"
mkdir -p results/devcontrol
CUDA_VISIBLE_DEVICES=0 "$PY" code/run_arm.py --arm ctm-therm5 --label ctm-therm5-devctl \
    --seed 0 --epochs 3 --eval-n 2000 --test-curve-last 25 \
    --out results/devcontrol/ctm-therm5-devctl_seed0.json --force \
    > logs/ctm-therm5-devctl_seed0.log 2>&1
echo "[launcher] device control rc=$?  $(date -Is)"
"$PY" - <<'PYEOF'
import json, sys
want = [0.5152, 0.5244, 0.5800]          # results/ctm-therm5-preflight_seed0.json, epochs 1-3
try:
    got = [round(e["val_acc"], 4) for e in
           json.load(open("results/devcontrol/ctm-therm5-devctl_seed0.json"))["curve"][:3]]
except Exception as exc:                  # noqa: BLE001
    print(f"[devctl] UNREADABLE: {exc}"); sys.exit(0)
ok = all(abs(a - b) < 1e-4 for a, b in zip(got, want))
print(f"[devctl] {'PASS' if ok else '*** FAIL ***'} gpu0 {got} vs gpu1 {want}")
if not ok:
    open("logs/DEVICE_MISMATCH.md", "w").write(
        "# ctm-therm5 is device-dependent\n\n"
        f"GPU 0 (3090) epochs 1-3 val: {got}\nGPU 1 (3080) epochs 1-3 val: {want}\n\n"
        "Consequence: the 20 000 point's seed 0 (3080) and seed 1 (3090) are NOT a matched\n"
        "seed pair, and `device` is part of the reproducibility key for this arm.\n")
PYEOF

# ---------------------------------------------------------------------------------------
# 1. The ladder endpoints at seed 1.  12.71 GPU-h, 80 000 first.  --wait so that a detached
#    launch behind another wave polls the lock instead of dying or, far worse, contending.
#    --poll 300: GPU 0 is free, nothing else is queued on it, so no priority claim is needed.
# ---------------------------------------------------------------------------------------
echo "[launcher] === 1: ladder seed 1, endpoints === $(date -Is)"
"$PY" code/queue_runner.py --gpu 0 --jobs jobs/p3_ladder_seed1.txt --seeds 1 \
      --wait 172800 --poll 300
echo "[launcher] ladder rc=$?  $(date -Is)"
echo "[launcher] done. $(date -Is)"
