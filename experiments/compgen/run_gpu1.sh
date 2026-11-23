#!/bin/bash
# compgen benchmark (issue #15) on GPU slot 1. Start one script per tmux session.
# Safe to re-run: it resumes. Options: see run_gpu.sh.
exec "$(dirname "$0")/run_gpu.sh" 1
