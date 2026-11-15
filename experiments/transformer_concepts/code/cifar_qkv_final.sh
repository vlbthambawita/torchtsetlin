#!/bin/bash
# After the qkv / qkv_d1 searches: pick s by best validation, run the finals (D16, D17).
cd "$(dirname "$0")/../../.."
R=experiments/transformer_concepts/results/cifar
best() { python -c "
import json,sys
c=[(json.load(open(f'$R/$1_s{s}_seed0_e30.json'))['best_val'],s) for s in (2,10)]
print(max(c)[1])"; }
until [ -f $R/qkv_d1_s2_seed0_e30.json ] && [ -f $R/qkv_d1_s10_seed0_e30.json ]; do sleep 60; done
s=$(best qkv_d1)
( for seed in 1 2 3; do CUDA_VISIBLE_DEVICES=1 python -m experiments.transformer_concepts.code.cifar run qkv_d1 $s $seed --epochs 30 --test \
    > experiments/transformer_concepts/logs/cifar/final_qkv_d1_s${s}_seed$seed.log 2>&1; done ) &
until [ -f $R/qkv_s2_seed0_e30.json ] && [ -f $R/qkv_s10_seed0_e30.json ]; do sleep 60; done
s=$(best qkv)
CUDA_VISIBLE_DEVICES=0 python -m experiments.transformer_concepts.code.cifar run qkv $s 1 --epochs 30 --test \
    > experiments/transformer_concepts/logs/cifar/final_qkv_s${s}_seed1.log 2>&1
wait
