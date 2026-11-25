#!/bin/bash
# Kvasir-SEG detect-then-segment programme on GPU slot 3. Start one script per tmux session.
# Safe to re-run: it resumes.  ./run_gpu3.sh [phases] [runners per GPU], e.g. ./run_gpu3.sh base,p1 2
# Options: see run_gpu.sh.
exec "$(dirname "$0")/run_gpu.sh" 3 "$@"
