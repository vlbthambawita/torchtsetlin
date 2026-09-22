#!/usr/bin/env bash
# Run the section-7.3 arms, one process per (arm, seed) so a crash costs only that run.
#
#   GPU=0 WAVE=a ./run_ideas_all.sh        (waves: a b c d e f g)
#
# Waves are independent apart from e and f, which use the firing-rate target chosen from the
# wave-b sweep (TARGET, default 0.20 -- the best of 0.005 .. 0.20).
#
# NOTE: bash re-reads a script file as it executes it, so editing the wave list below while a
# wave is running corrupts the running shell. `queue_runner.py` exists for that reason: it
# reads its job list once, at start, and is what the long runs were actually driven with.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${GPU:-0}

EPOCHS=${EPOCHS:-30}
SEEDS=${SEEDS:-"0 1 2"}
TARGET=${TARGET:-0.20}

# --- wave a: idea A (a different objective) and the readouts that measure it ----------
WAVE_A=$(cat <<'X'
mctm-auto|--l1 auto
flat-auto|--l1 auto --head flat
flat-random|--l1 random --head flat
X
)
# --- wave b: idea B, the firing-rate target sweep -------------------------------------
#     Layer 1 is the cached E1 checkpoint in every row: six adjustments of one object.
WAVE_B=$(cat <<'X'
mctm-calib-r005|--l1 greedy --calib 0.005
mctm-calib-r01|--l1 greedy --calib 0.01
mctm-calib-r02|--l1 greedy --calib 0.02
mctm-calib-r05|--l1 greedy --calib 0.05
mctm-calib-r10|--l1 greedy --calib 0.10
mctm-calib-r20|--l1 greedy --calib 0.20
X
)
# --- wave c: idea C, the three rungs of the credit rule, from a random layer 1 ---------
#     ia   the rule as written: Type Ia and Type II only -- both ADD literals
#     ib   + Type Ib, blamed on the literal that actually blocked the match
#     bal  + Type Ib capped at the delivered Type Ia rate
WAVE_C=$(cat <<'X'
mctm-credit-ia|--l1 random --credit ia
mctm-credit-ib|--l1 random --credit ib
mctm-credit-bal|--l1 random --credit bal
X
)
# --- wave d: idea C at its best case -- greedy pretraining, then credit fine-tuning ----
#     credit-warm is the bandwidth result: over a 0.14%-firing layer 1 the path delivers
#     ZERO events, so the step size is only worth sweeping with the controller running
#     (wave f), where layer 1 is dense enough for a layer-2 clause to match on a positive
#     channel literal at all.
WAVE_D=$(cat <<'X'
mctm-credit-warm|--l1 greedy --credit bal --warmup 10
X
)
# --- wave e: idea B's second target -- clause SIZE at the best firing rate -------------
WAVE_E=$(cat <<'X'
mctm-size-k24|--l1 greedy --calib TARGET --max-size 24
mctm-size-k12|--l1 greedy --calib TARGET --max-size 12
mctm-size-k6|--l1 greedy --calib TARGET --max-size 6
mctm-size-k3|--l1 greedy --calib TARGET --max-size 3
X
)
# --- wave g: the same controller over an UNTRAINED layer 1 -----------------------------
#     This is the sweep that clears every criterion; r20 is named mctm-random-calib for
#     historical reasons (it was run first, as a control for wave b).
WAVE_G=$(cat <<'X'
mctm-rc-r05|--l1 random --calib 0.05
mctm-rc-r10|--l1 random --calib 0.10
mctm-random-calib|--l1 random --calib 0.20
mctm-rc-r40|--l1 random --calib 0.40
mctm-rc-r60|--l1 random --calib 0.60
X
)
# --- wave f: the combinations ---------------------------------------------------------
WAVE_F=$(cat <<'X'
mctm-auto-calib|--l1 auto --calib TARGET
mctm-calib-r20-tight|--l1 greedy --calib TARGET --calib-band 1.2
mctm-credit-cal-p1|--l1 greedy --credit bal --warmup 10 --credit-rate 1.0 --calib-every 50 --calib-target TARGET
mctm-credit-cal|--l1 greedy --credit bal --warmup 10 --credit-rate 0.1 --calib-every 50 --calib-target TARGET
mctm-credit-cal-p001|--l1 greedy --credit bal --warmup 10 --credit-rate 0.01 --calib-every 50 --calib-target TARGET
mctm-auto-credit|--l1 auto --credit bal --warmup 10 --credit-rate 0.1 --calib-every 50 --calib-target TARGET
X
)
case "${WAVE:-a}" in
  a) RUNS="$WAVE_A" ;;  b) RUNS="$WAVE_B" ;;  c) RUNS="$WAVE_C" ;;
  d) RUNS="$WAVE_D" ;;  e) RUNS="$WAVE_E" ;;  f) RUNS="$WAVE_F" ;;
  g) RUNS="$WAVE_G" ;;
  *) echo "unknown WAVE"; exit 1 ;;
esac
RUNS=${RUNS//TARGET/$TARGET}

mkdir -p results logs
for seed in $SEEDS; do
  echo "$RUNS" | while IFS='|' read -r name args; do
    [ -z "$name" ] && continue
    out="results/${name}_seed${seed}.json"
    if [ -f "$out" ]; then echo "skip $name seed $seed (exists)"; continue; fi
    echo "=== $name seed=$seed epochs=$EPOCHS $args ==="
    python run_ideas.py --arm "$name" --seed "$seed" --epochs "$EPOCHS" \
      --epochs-l1 "$EPOCHS" $args --out "$out" 2>&1 | tee "logs/${name}_seed${seed}.log"
  done
done
echo "=== wave ${WAVE:-a} done (gpu $CUDA_VISIBLE_DEVICES) ==="
