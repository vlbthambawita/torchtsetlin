#!/usr/bin/env bash
# Phase 0 (plan B6): no reported numbers, RTX 3080 only.
#  1. correctness checks T0-T6, T8                 -> results/checks/check_joint.json, REPORT_checks.md
#  2. pricing T7 (timed updates of every run kind) -> results/pricing/, REPORT_pricing.md
#  3. resume check: a joint run interrupted after epoch 2 and resumed gives the same curve
#  4. pipeline: every arm, full-size models, 2 epochs on 2 000 training images, then composites,
#     reload checks and activation maps            -> results/smoke/, REPORT_smoke.md
# Usage: nohup bash code/smoke.sh > logs/smoke_driver.txt 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
DEV=${DEV:-cuda:1}
mkdir -p ../logs/smoke
rm -rf ../.cache/claims/smoke ../.cache/claims/pricing ../.cache/ckpt/smoke ../results/smoke ../.cache/sums/smoke \
       ../.cache/epochs/smoke ../models/smoke

python check_joint.py --device "$DEV"
python report.py --phase checks

python queue_runner.py --phase pricing --device "$DEV"
python report.py --phase pricing

A=(--phase smoke --arm J --team S --rule one --device "$DEV" --subset 1000 --epochs 4 --no-wandb --ckpt-every 100)
JW_RESULTS=../results/smoke/resume_full JW_CACHE=../.cache/resume_full JW_MODELS=../.cache/resume_full/models \
  python train.py "${A[@]}" > ../logs/smoke/resume.log 2>&1
JW_RESULTS=../results/smoke/resume_int JW_CACHE=../.cache/resume_int JW_MODELS=../.cache/resume_int/models \
  python train.py "${A[@]}" --stop-after 2 >> ../logs/smoke/resume.log 2>&1 || [ $? -eq 3 ]
JW_RESULTS=../results/smoke/resume_int JW_CACHE=../.cache/resume_int JW_MODELS=../.cache/resume_int/models \
  python train.py "${A[@]}" >> ../logs/smoke/resume.log 2>&1
python - <<'PY'
from common import RESULTS, read_json, write_json
f, i = (read_json(RESULTS / "smoke" / f"resume_{k}" / "smoke" / "J" / "S_one_seed0.json") for k in ("full", "int"))
curve = lambda r: [(c["epoch"], c["val_acc"], c["test_acc"], [m["mean_clause_len"] for m in c["members"].values()])
                   for c in r["curve"]]
out = dict(identical_curve=curve(f) == curve(i), resumed_at=i["resumed_at"][0]["epoch"], epochs=[f["epochs_run"], i["epochs_run"]])
write_json(out, RESULTS / "smoke" / "resume_check.json")
print("resume check:", out)
assert out["identical_curve"], out
PY
rm -rf ../.cache/resume_full ../.cache/resume_int ../results/smoke/resume_full ../results/smoke/resume_int

python queue_runner.py --phase smoke --device "$DEV"
python composite.py --phase smoke --seeds 0
for f in ../models/smoke/*/*.pt; do python infer.py "$f" --device "$DEV"; done
python activation_maps.py ../models/smoke/J/S_sum_seed0.pt --device "$DEV" --corr-n 50 --no-wandb
python activation_maps.py ../models/smoke/Icomp/S_seed0.pt --device "$DEV" --corr-n 50 --no-wandb \
  --indices-from ../results/smoke/activation/J/S_sum_seed0.json
python report.py --phase smoke
echo "smoke: all steps passed"
