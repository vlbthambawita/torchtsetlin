#!/bin/bash
# Kvasir-SEG detect-then-segment programme on GPU slot 0. Start one script per tmux session.
# Safe to re-run: it resumes. Options: see run_gpu.sh.
exec "$(dirname "$0")/run_gpu.sh" 0
