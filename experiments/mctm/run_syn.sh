#!/usr/bin/env bash
# run_syn.sh <gpu> <task> -- E6 on one of the two constructed tasks.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$1"
python synthetic_ideas.py --tasks "$2" \
  --arms auto auto-calib calib random credit-ia credit credit-warm credit-warm-p1 credit-cal \
  --seeds 0 1 2 --epochs 20 --n-train 8000 --n-test 2000 \
  --out "results/synthetic_ideas_$2.json"
echo "=== synthetic $2 done ==="
