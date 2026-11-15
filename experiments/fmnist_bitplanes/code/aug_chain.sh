#!/bin/bash
# plans/fmnist-bitplanes-augmentation.md, user go-ahead 2026-11-14: the 4 stage-A conv3 cells with
# shift2flip augmentation (no policy pilot: "apply both"), then the no-augmentation replay of the
# best stage-A cell (conv3 gray8 8k) "after augmentation". Both save best-validation models to
# models/ (git-ignored). RTX 3080 only.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2
python code/queue_runner.py --arm aug --device cuda:1 > logs/q_aug.log 2>&1
exec python code/queue_runner.py --arm replay --device cuda:1 > logs/q_replay.log 2>&1
