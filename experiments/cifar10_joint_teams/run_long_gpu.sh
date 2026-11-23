#!/bin/bash
# Longer training + drop clause on one GPU: ./run_long_gpu.sh <slot 0-3>   (run_long_gpu0.sh ... 3.sh)
# Best team of the clause sweep (M = H3 + M10 + B3, joint, geo+col augmentation, batch 64) with T
# scaled with the clauses, early stopping at patience 75 and NO epoch limit. The four runners
# share one job list, one run per GPU:
#   1. lprice   30 timed updates per model -> s/epoch and peak memory (./status.sh)
#   2. long     stage 1: 32 000 clauses per class and member at drop_clause_p 0, 0.25, 0.1, 0.5
#               stage 2: 64 000 clauses at p = 0 (first free GPU) and at the p chosen from stage 1
#                        (results/long/choice.json, written when the four stage-1 runs are done)
# Needs no pilot: the augmentation policy is fixed. Safe to re-run: finished runs are skipped,
# interrupted runs continue from their last epoch.
#   CJT_LONG_PSTAR=0.25 ./run_long_gpu0.sh     stage 2 does not wait for stage 1, uses this p
#   CJT_LONG_BIG=none ./run_long_gpu0.sh       no stage 2
#   CJT_LONG_PATIENCE=150 ./run_long_gpu0.sh   larger patience; runs that already stopped by
#                                              patience continue from their kept last checkpoint
#   CJT_TIME_CAP_H=120 ./run_long_gpu0.sh      safety time cap per run (default 240 h)
#   CJT_PHASES=lprice ./run_long_gpu0.sh       only the cost table
#   CJT_FG=1 ./run_long_gpu0.sh                stay in the foreground
# Use the same options in all four sessions.
export CJT_PHASES=${CJT_PHASES:-lprice,long}
exec "$(dirname "$0")/run_gpu.sh" "${1:?usage: run_long_gpu.sh <gpu slot 0-3>}"
