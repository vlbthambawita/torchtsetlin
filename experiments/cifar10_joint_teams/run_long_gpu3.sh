#!/bin/bash
# Longer training + drop clause on GPU slot 3: team M, batch 64, scaled T, patience 75, no epoch limit.
# Start one script per tmux session. Safe to re-run: it resumes. Options: see run_long_gpu.sh.
exec "$(dirname "$0")/run_long_gpu.sh" 3
