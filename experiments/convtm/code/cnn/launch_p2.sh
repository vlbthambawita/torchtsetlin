#!/bin/bash
# Detached P2 launcher. Waits for the shared GPU lock, then runs the CNN baseline waves.
#
#   setsid nohup code/cnn/launch_p2.sh > logs/p2_cnn_launcher.log 2>&1 < /dev/null &
#
# `setsid` puts it in its own session so it survives the shell that started it; all waiting is
# done INSIDE the GPU lock helper (`queue_cnn.py --wait`), never by holding a turn open.
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/convtm || exit 1
export PYTHONUNBUFFERED=1
PY=$(command -v python)

echo "[launcher] started $(date -Is) pid=$$ ppid=$PPID"

# ---------------------------------------------------------------------------------------
# The lock is RELEASED between every invocation below, so a TM wave can take the 3080 at any
# of those points and this launcher simply waits again. One 47-hour lock hold would have made
# the card unusable to the rest of the programme for two days.
#
# Order, per seed: spine -> binary -> sample-efficiency curve -> reference nets.
# Seed-outer, so the complete decomposition lands at seed 0 before any second seed starts.
# The curve is ahead of the reference nets deliberately: ~3 GPU-h against ~21, and it is a
# deliverable the TM side is compared against, whereas the reference nets are numbers nobody
# disputes. Everything is skip-if-exists, so re-ordering later costs nothing.
#
# The FIRST arm the queue runs once it owns an uncontended card is a linear probe, then an
# MLP (~1 minute together) -- which doubles as the GPU smoke of the `mlp` / `linear` code
# paths, verified so far only on CPU. A failure there shows up in
# logs/cnn-hog-linear_seed0.log within a minute of the lock clearing and costs nothing.
# ---------------------------------------------------------------------------------------
rc=0
for seed in 0 1 2; do
  for f in p2_cnn p2_cnn_bin p2_cnn_curve p2_cnn_ref; do
    echo "[launcher] === seed $seed file $f === $(date -Is)"
    "$PY" code/cnn/queue_cnn.py --gpu 1 --jobs "jobs/$f.txt" \
          --seeds "$seed" --wait 172800 --poll 120
    r=$?
    [ $r -ne 0 ] && rc=$r
    echo "[launcher] seed $seed file $f rc=$r  $(date -Is)"
  done
done

# Optional binary variants (binaryconnect / xnor / binary-aug), 1 seed, last.
"$PY" code/cnn/queue_cnn.py --gpu 1 --jobs jobs/p2_cnn_opt.txt \
      --seeds 0 --wait 172800 --poll 120
echo "[launcher] wave 4 rc=$?  overall rc=$rc  $(date -Is)"
echo "[launcher] done."
