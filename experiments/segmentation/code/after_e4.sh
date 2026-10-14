#!/bin/bash
# Post-E4 pipeline: sealed evaluation, stress, idle latency bench, report. Run from experiments/segmentation.
set -u
n() { grep -l '"status": "completed"' results/$1*.json 2>/dev/null | wc -l ; }
until [ "$(n E4_kvasir_F_)" -ge 25 ] && [ "$(n E4_acdc_F_)" -ge 25 ]; do sleep 60; done
echo "$(date +%T) E4 complete"
python code/final_eval.py --dataset kvasir --device cuda:1
python code/final_eval.py --dataset acdc --device cuda:1
echo "$(date +%T) final eval done"
python code/stress.py E4_kvasir_F_base_B2_s0 E4_kvasir_F_AB_sampled_wide_s0 E4_kvasir_F_unet_s0 \
    E4_acdc_F_base_B2_s0 E4_acdc_F_AB_sampled_wide_s0 E4_acdc_F_unet_s0 --device cuda:1
echo "$(date +%T) stress done"
# idle GPU for latency: wait until no training job runs on cuda:0
while pgrep -f "code/run.py" > /dev/null; do sleep 60; done
python code/bench.py latency E4_kvasir_F_base_B2_s0 E4_kvasir_F_A_sampled_s0 E4_kvasir_F_B_wide_s0 E4_kvasir_F_AB_sampled_wide_s0 E4_kvasir_F_unet_s0 \
    E4_acdc_F_base_B2_s0 E4_acdc_F_AB_sampled_wide_s0 E4_acdc_F_unet_s0 --device cuda:0
python code/bench.py profile --device cuda:0
echo "$(date +%T) bench done"
python code/report.py
echo "$(date +%T) ALL DONE"
