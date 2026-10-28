#!/usr/bin/env bash
# deep.py / pred.py anatomy for torchtsetlin 0.5.1 (worktree) and the working tree, same GPU
P=/home/vajira/anaconda3/envs/pytorch2/bin/python
for ver in v0.5.1 now; do
  if [ $ver = v0.5.1 ]; then export PYTHONPATH=$PWD/wt051/src; else export PYTHONPATH=/work/vajira/DL2026/torchtsetlin/src; fi
  for a in "B1 batch 300" "B1 seq 3" "B2 seq 2" "B4 batch 60" "B4 b100 40" "B9 batch 20" "B7 batch 30" "B7 b100 15"; do
    echo "$ver $($P deep.py $a 2>&1 | grep -E '^B[0-9]' | cut -d, -f1-2)"
  done
  echo "$ver $($P pred.py 2>&1 | grep 'B7 predict')"
done
