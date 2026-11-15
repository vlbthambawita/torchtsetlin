#!/usr/bin/env bash
# Smoke test (plan Phase 0): no reported numbers, RTX 3080 only (D11).
#  1. encoding + plane-layout checks            -> results/smoke/encoding_checks.json
#  2. resume check: a tiny conv3 run interrupted after epoch 4 and resumed must give the same
#     curve, best epoch and stop epoch as an uninterrupted one (CPU and cuda:1)
#                                                -> results/smoke/resume_check.json
#  3. every model the plan can build (10 cells x 2 C, conv3 at kd 1/2/4, + bit4/therm4),
#     full size, 1 epoch on 2 000 training images; 4 of them mirrored to W&B (group smoke)
#  4. REPORT_smoke.md
# Usage: nohup bash code/smoke.sh > logs/smoke_driver.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../logs/smoke
rm -rf ../.cache/claims/smoke ../results/resume_check ../.cache/ckpt/resume_check

python check_encodings.py

for dev in cpu cuda:1; do
  A=(--arch conv3 --enc bit8 --clauses 50 --kd 2 --device "$dev" --subset 1000 --patience 3
     --min-epochs 1 --max-epochs 12 --no-wandb --arm resume_check --ckpt-every 100)
  python train.py "${A[@]}" --tag-suffix "_full_${dev/:/}" > "../logs/smoke/resume_${dev/:/}.log" 2>&1
  python train.py "${A[@]}" --tag-suffix "_int_${dev/:/}" --stop-after 4 >> "../logs/smoke/resume_${dev/:/}.log" 2>&1 || [ $? -eq 3 ]
  python train.py "${A[@]}" --tag-suffix "_int_${dev/:/}" >> "../logs/smoke/resume_${dev/:/}.log" 2>&1
done
python - <<'PY'
import json
from common import RESULTS, write_json
out = {}
for dev in ("cpu", "cuda1"):
    f, i = (json.load(open(RESULTS / "resume_check" / f"conv3_bit8_c50_spec5_kd2_seed0_{k}_{dev}.json")) for k in ("full", "int"))
    curve = lambda r: [(c["epoch"], c["val_acc"], c["test_acc"], c["plane_usage"]["n_included"]) for c in r["curve"]]
    out[dev] = dict(identical_curve=curve(f) == curve(i), best_epoch=[f["best_epoch"], i["best_epoch"]],
                    epochs_run=[f["epochs_run"], i["epochs_run"]], stop_reason=[f["stop_reason"], i["stop_reason"]],
                    resumed_at=i["resumed_at"][0]["epoch"])
    assert out[dev]["identical_curve"], (dev, out[dev])
write_json(out, RESULTS / "smoke" / "resume_check.json")
print("resume check:", out)
PY

WB="flat_agt_c2000_spec10_seed0 conv2_agt_c2000_spec5_seed0 conv3_bit8_c2000_spec5_kd2_seed0 conv2_therm8_c8000_spec5_seed0"
python queue_runner.py --arm smoke --device cuda:1 --wandb-only $WB > ../logs/smoke/queue.log 2>&1
python report.py --arm smoke
