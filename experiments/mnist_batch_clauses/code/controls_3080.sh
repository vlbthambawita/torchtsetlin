#!/usr/bin/env bash
# Control arms on the RTX 3080 (cuda:1), in parallel with the seed-0 grid on the 3090 (D8: control
# timings are not reported). D6 sequential check first (4 runs), then the D2 T-fixed control (12).
# flat 64k B=256 needs 15.3 GB (dense commit path) and fails here; its .failed-cuda:1 marker leaves
# it for a 3090 runner after the seed-0 grid:  python code/queue_runner.py --arm control_T --device cuda:0
# Usage: nohup bash code/controls_3080.sh > logs/q_controls_3080.log 2>&1 &
cd "$(dirname "$0")"
python queue_runner.py --arm control_seq --device cuda:1
python queue_runner.py --arm control_T --device cuda:1
