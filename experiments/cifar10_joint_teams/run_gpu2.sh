#!/bin/bash
# Run the CIFAR-10 joint-team queue on GPU slot 2. Safe to re-run: it resumes.
exec "$(dirname "$0")/run_gpu.sh" 2
