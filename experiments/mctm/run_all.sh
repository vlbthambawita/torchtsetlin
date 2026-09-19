#!/usr/bin/env bash
# Run every MCTM arm across seeds, one process per run so a crash costs only that run.
#
# E0 showed layer-1 density and layer-1 accuracy trade off directly against s: s=10 gives the
# most accurate layer 1 but the sparsest maps, s=2 the densest maps but a weaker layer 1.
# The single-layer baselines therefore run at the s that is best FOR THEM (s=10), so the
# comparison is conservative towards the MCTM, and the MCTM is run in both regimes
# (mctm at s1=10, mctm-dense at s1=2) so it is judged on whichever suits it.
#
#   EPOCHS=30 SEEDS="0 1 2" ./run_all.sh
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
SUBSET=${SUBSET:-""}
[ -n "$SUBSET" ] && SUBSET="--subset $SUBSET"

# name|arm|extra args
RUNS=$(cat <<'EOF'
single-l1|single-l1|--s1 10 --s2 10
single-matched|single-matched|--s1 10 --s2 10
single-rf|single-rf|--s1 10 --s2 10
flat-stack|flat-stack|--s1 10 --s2 10
mctm|mctm|--s1 10 --s2 10
mctm-dense|mctm|--s1 2 --s2 2
mctm-random|mctm-random|--s1 10 --s2 10
mctm-nopos|mctm-nopos|--s1 10 --s2 10
mctm-pool4|mctm|--s1 10 --s2 10 --pool 4
mctm-nopool|mctm-nopool|--s1 10 --s2 10
EOF
)

mkdir -p results logs
for seed in $SEEDS; do
  echo "$RUNS" | while IFS='|' read -r name arm args; do
    [ -z "$name" ] && continue
    out="results/${name}_seed${seed}.json"
    if [ -f "$out" ]; then echo "skip $name seed $seed (exists)"; continue; fi
    tag=""
    [ "$name" != "$arm" ] && tag="${name#${arm}}"
    echo "=== $name (arm=$arm) seed=$seed epochs=$EPOCHS $args ==="
    python run.py --arm "$arm" --seed "$seed" --epochs "$EPOCHS" --epochs-l1 "$EPOCHS" \
      $args $SUBSET --tag="$tag" --out "$out" 2>&1 | tee "logs/${name}_seed${seed}.log"
  done
done
echo "=== all done ==="
python report_data.py
