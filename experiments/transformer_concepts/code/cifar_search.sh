#!/bin/bash
# s search for the CIFAR-10 arms (validation only, seed 0, 30 epochs). Two workers per GPU.
cd "$(dirname "$0")/../../.."
run() { CUDA_VISIBLE_DEVICES=$1 python -m experiments.transformer_concepts.code.cifar run $2 $3 0 --epochs 30 \
        > experiments/transformer_concepts/logs/cifar/search_$2_s$3.log 2>&1; }
( for a in "pair 2" "flat 2" "flat 10" "flat 40"; do run 0 $a; done ) &
( for a in "pair 10" "pair 40"; do run 0 $a; done ) &
( for a in "pair_local 2" "pair_local 10" "pair_local 40" "ctm 40"; do run 1 $a; done ) &
( for a in "ctm 2" "ctm 10" "pair_self 2" "pair_self 10" "pair_self 40"; do run 1 $a; done ) &
wait
