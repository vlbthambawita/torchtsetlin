#!/bin/bash
# Detached P3 launcher for the clause-size-budget prediction sweep, Block 2 first.
#
#   setsid nohup code/launch_p3_block2.sh > logs/p3_block2_launcher.log 2>&1 < /dev/null &
#
# `setsid` puts it in its own session so it survives the shell that started it; all waiting is
# done INSIDE the GPU lock helper (`queue_runner.py --wait`), never by holding a turn open.
# Pattern copied from code/cnn/launch_p2.sh, which is working.
set -u
cd /work/vajira/DL2026/torchtsetlin/experiments/convtm || exit 1
export PYTHONUNBUFFERED=1
PY=$(command -v python)

echo "[launcher] started $(date -Is) pid=$$ ppid=$PPID"

# ---------------------------------------------------------------------------------------
# GPU 1 (RTX 3080) is not optional: every arm these are read against was measured there and
# DR-001 makes `device + seed` the reproducibility key. No --epochs override: each job
# carries its own, matching the existing arms exactly.
#
# THE LOCK IS RELEASED BETWEEN EVERY INVOCATION BELOW, and the detached CNN queue
# (code/cnn/queue_cnn.py, launched by the dl-expert) is polling the same lock at 120 s. The
# --poll values here are a deliberate priority order, not an accident:
#
#   wave 1  Block 2, the s-sweep          --poll 5    1.9 GPU-h  the only block that can
#                                                                overturn DR-004's headline
#   wave 2  the CIFAR-2 budget pair       --poll 5    0.9-1.5 h  the theorist ranks it above
#                                                                blocks 3+4; separates bloat
#                                                                from headroom
#   wave 3  blocks 1+3+4, the rest        --poll 120  4.6 h      SAME period as the CNN queue,
#                                                                so from here the two waves
#                                                                interleave at file boundaries
#                                                                rather than one starving the
#                                                                other (DR-005 Decision 2's
#                                                                ordering logic, applied both
#                                                                ways).
#
# Total high-priority hold: ~2.8-3.4 GPU-h. After that the CNN P2 spine competes on equal
# terms. If the orchestrator wants P2 to go first, kill this launcher -- nothing here holds
# the lock while waiting, so killing it between waves costs nothing, and every wave is
# skip-if-exists so a partial run resumes exactly where it stopped.
# ---------------------------------------------------------------------------------------
rc=0

echo "[launcher] === wave 1: block 2, the s-sweep === $(date -Is)"
"$PY" code/queue_runner.py --gpu 1 --jobs jobs/p3_block2_s_sweep.txt --seeds 0 1 2 \
      --wait 172800 --poll 5
r=$?; [ $r -ne 0 ] && rc=$r
echo "[launcher] wave 1 rc=$r  $(date -Is)"

echo "[launcher] === wave 2: the CIFAR-2 discriminator === $(date -Is)"
"$PY" code/queue_runner.py --gpu 1 --jobs jobs/p3_cifar2.txt --seeds 0 1 2 \
      --wait 172800 --poll 5
r=$?; [ $r -ne 0 ] && rc=$r
echo "[launcher] wave 2 rc=$r  $(date -Is)"

# Blocks 1, 3 and 4. Block 2's four s-cells are in that file too and will be skipped as
# "exists" -- the labels are identical by construction, which is why wave 1 reuses them
# verbatim rather than inventing new names.
echo "[launcher] === wave 3: blocks 1, 3 and 4 === $(date -Is)"
"$PY" code/queue_runner.py --gpu 1 --jobs jobs/p3_budget_predictions.txt --seeds 0 1 2 \
      --wait 172800 --poll 120
r=$?; [ $r -ne 0 ] && rc=$r
echo "[launcher] wave 3 rc=$r  $(date -Is)"

echo "[launcher] overall rc=$rc  $(date -Is)"
echo "[launcher] done."
