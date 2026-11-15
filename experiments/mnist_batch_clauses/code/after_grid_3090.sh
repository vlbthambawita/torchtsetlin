#!/bin/bash
# 3090 queue after the flat/conv seed-0 grid (user, 2026-11-13): control_T, then the conv3 bit-plane grid.
cd "$(dirname "$0")/.."
GRID_PID=${1:?pid of the running grid queue_runner}
while kill -0 "$GRID_PID" 2>/dev/null; do sleep 60; done
python code/queue_runner.py --arm control_T --device cuda:0 >> logs/q_control_T_3090.log 2>&1
exec python code/queue_runner.py --arm grid --seeds 0 --models conv3 --device cuda:0 >> logs/q_grid3d_s0.log 2>&1
