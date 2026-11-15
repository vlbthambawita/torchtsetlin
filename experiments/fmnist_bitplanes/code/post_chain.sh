#!/bin/bash
# After code/aug_chain.sh (aug runs + replay) has finished: for every saved best-validation model,
# infer.py (reload check, confusion matrix, shift test) and activation_maps.py (evidence maps for
# one correct + one wrong test image per class, logged to the run's W&B run). User, 2026-11-14.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
while pgrep -f "code/aug_chain.sh" >/dev/null || pgrep -f "queue_runner.py --arm (aug|replay)" >/dev/null; do sleep 120; done
for m in models/aug/*.pt models/replay/*.pt; do
  [ -f "$m" ] || continue
  echo "== $m"
  python code/infer.py "$m" --device cuda:1
  python code/activation_maps.py "$m" --device cuda:1
done
echo "post_chain done"
