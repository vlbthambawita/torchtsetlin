#!/usr/bin/env bash
# Reproduce every number of experiments/paper_experiments (PLAN.md §6).
#
#   ./reproduce.sh                  # everything, in order (about 3 days on an RTX 3090 + 3080)
#   ./reproduce.sh setup            # Phase 0: one virtualenv per external library
#   ./reproduce.sh data             # Phase 1: Booleanize every dataset once (.cache/data)
#   ./reproduce.sh smoke            # Phase 2: every cell for 1 epoch (results/smoke)
#   ./reproduce.sh repro            # Phase 3: reproduction checks R1-R4 (R5 = B6 GT rows, R6 = B1 tt-seq)
#   ./reproduce.sh timing           # Phase 4: seed 0 of every cell, one at a time (exclusive machine)
#   ./reproduce.sh seeds            # Phase 5: remaining seeds / splits, concurrently
#   ./reproduce.sh report           # Phase 7: REPORT.md + LaTeX tables for the paper
#
# Extra arguments go to code/launch.py, e.g.  ./reproduce.sh timing --bench B6 --lib gt
# Re-running a stage skips cells whose results/*.json record has status "done".
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/vajira/anaconda3/envs/pytorch2/bin/python
stage="${1:-all}"; shift || true
run() {
  case "$1" in
    setup)  ./code/setup_envs.sh ;;
    data)   $PY code/prepare_data.py ;;
    smoke)  $PY code/launch.py smoke "${@:2}" ;;
    repro)  $PY code/launch.py repro "${@:2}" ;;
    timing) $PY code/launch.py timing "${@:2}" ;;
    seeds)  $PY code/launch.py seeds "${@:2}" ;;
    report) $PY code/report.py "${@:2}" ;;
    *) echo "unknown stage $1"; exit 2 ;;
  esac
}
if [ "$stage" = all ]; then
  for s in setup data smoke repro timing seeds report; do run "$s"; done
else
  run "$stage" "$@"
fi
