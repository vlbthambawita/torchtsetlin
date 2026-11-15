#!/bin/bash
# Final CIFAR-10 runs at the searched s (best validation in results/cifar/*_seed0_e30.json),
# seeds 1-3, 30 epochs, test set read once per run. Usage: cifar_final.sh GPU "arm s" ...
cd "$(dirname "$0")/../../.."
gpu=$1; shift
for a in "$@"; do set -- $a
  for seed in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python -m experiments.transformer_concepts.code.cifar run $1 $2 $seed --epochs 30 --test \
      > experiments/transformer_concepts/logs/cifar/final_$1_s$2_seed$seed.log 2>&1
  done
done
