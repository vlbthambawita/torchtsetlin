#!/usr/bin/env bash
# Smoke test (plan §6 Phase 1): no GPU-hours, nothing reported in the paper.
#  1. resume check: a tiny run interrupted after epoch 4 and resumed must give the same curve,
#     best epoch and stop epoch as an uninterrupted one (CPU and CUDA) -> results/smoke/resume_check.json
#  2. all 112 grid configurations, 1 epoch on 2 000 training images (500 val / 500 test), split over
#     both GPUs; 8 of them mirrored to W&B (group mnist_batch_clauses_smoke) -> results/smoke/*.json
#  3. REPORT_smoke.md
# Usage: bash code/smoke.sh
set -euo pipefail  # run as: nohup bash code/smoke.sh > logs/smoke/driver.txt 2>&1 &
cd "$(dirname "$0")"
mkdir -p ../logs/smoke
rm -rf ../.cache/claims/smoke ../results/resume_check ../.cache/ckpt/resume_check

for dev in cpu cuda:0; do
  A=(--model conv --clauses 100 --batch-size 16 --device "$dev" --subset 2000 --patience 3
     --min-epochs 1 --max-epochs 15 --no-wandb --arm resume_check --ckpt-every 100)
  python train.py "${A[@]}" --tag-suffix "_full_${dev/:/}" > "../logs/smoke/resume_${dev/:/}.log" 2>&1
  python train.py "${A[@]}" --tag-suffix "_int_${dev/:/}" --stop-after 4 >> "../logs/smoke/resume_${dev/:/}.log" 2>&1 || [ $? -eq 3 ]
  python train.py "${A[@]}" --tag-suffix "_int_${dev/:/}" >> "../logs/smoke/resume_${dev/:/}.log" 2>&1
done
python - <<'PY'
import json
from common import RESULTS, write_json
out = {}
for dev in ("cpu", "cuda0"):
    f, i = (json.load(open(RESULTS / "resume_check" / f"conv_c100_b16_Tscaled_s0_{k}_{dev}.json")) for k in ("full", "int"))
    curve = lambda r: [(c["epoch"], c["val_acc"], c["test_acc"]) for c in r["curve"]]
    out[dev] = dict(identical_curve=curve(f) == curve(i), best_epoch=[f["best_epoch"], i["best_epoch"]],
                    epochs_run=[f["epochs_run"], i["epochs_run"]], stop_reason=[f["stop_reason"], i["stop_reason"]],
                    resumed_at=i["resumed_at"][0]["epoch"])
    assert out[dev]["identical_curve"], (dev, out[dev])
write_json(out, RESULTS / "smoke" / "resume_check.json")
print("resume check:", out)
PY

WB="flat_c1000_b1_Tscaled_s0 flat_c64000_b1024_Tscaled_s0 flat_c64000_b1_Tscaled_s0 flat_c8000_b32_Tscaled_s0
    conv_c1000_b1_Tscaled_s0 conv_c64000_b1024_Tscaled_s0 conv_c64000_b1_Tscaled_s0 conv_c8000_b32_Tscaled_s0"
python queue_runner.py --arm smoke --seeds 0 --device cuda:0 --wandb-only $WB > ../logs/smoke/queue_gpu0.log 2>&1 &
python queue_runner.py --arm smoke --seeds 0 --device cuda:1 --wandb-only $WB > ../logs/smoke/queue_gpu1.log 2>&1 &
wait
python report.py --arm smoke
