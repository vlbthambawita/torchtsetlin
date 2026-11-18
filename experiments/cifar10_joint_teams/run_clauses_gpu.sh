#!/bin/bash
# Clause sweep on one GPU: ./run_clauses_gpu.sh <slot 0-3>   (run_clauses_gpu0.sh ... 3.sh)
# Best team of the batch grid (M = H3 + M10 + B3, joint, augmented) at batch 64 with larger
# members. The four runners share one job list (largest model first), one run per GPU:
#   1. cprice    30 timed updates per size -> s/epoch and peak memory (./status.sh)
#   2. clauses   the full runs (early stopping, patience 30, at most 200 epochs or 120 h)
# Needs the augmentation policy of the pilot (results/pilot/choice.json), so run it in the folder
# of the batch-size study.
#   CJT_CLAUSES=16000,32000 ./run_clauses_gpu0.sh     other sizes (per class and member)
#   CJT_CLAUSE_BATCH=32 ./run_clauses_gpu0.sh         another batch size
#   CJT_T_MODE=scaled ./run_clauses_gpu0.sh           T in proportion to the clauses (tag ..._Tscaled)
#   CJT_TIME_CAP_H=72 ./run_clauses_gpu0.sh           time cap per run
#   CJT_PHASES=cprice ./run_clauses_gpu0.sh           only the cost table
#   CJT_FG=1 ./run_clauses_gpu0.sh                    stay in the foreground
# Use the same options in all four sessions.
export CJT_PHASES=${CJT_PHASES:-cprice,clauses}
exec "$(dirname "$0")/run_gpu.sh" "${1:?usage: run_clauses_gpu.sh <gpu slot 0-3>}"
