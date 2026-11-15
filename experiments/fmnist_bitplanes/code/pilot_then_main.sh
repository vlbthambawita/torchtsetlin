#!/bin/bash
# Wait for the stage-A pilot, write the s choice (validation), then run the stage-A main arm on the 3080.
# User go-ahead 2026-11-13: "after pilot find best s values and run full".
cd "$(dirname "$0")/.."
while pgrep -f "queue_runner.py --arm pilot" >/dev/null; do sleep 60; done
python code/report.py --arm pilot --choose || exit 1
cat results/pilot/choice.json
python code/queue_runner.py --arm main --stage A --device cuda:1 --dry-run || exit 1
exec python code/queue_runner.py --arm main --stage A --device cuda:1 > logs/q_main.log 2>&1
