#!/bin/bash
# Run the CIFAR-10 joint-team queue on GPU slot 0. Safe to re-run: it resumes.
exec "$(dirname "$0")/run_gpu.sh" 0
