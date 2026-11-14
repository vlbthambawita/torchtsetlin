#!/bin/bash
# Run the CIFAR-10 batch x clauses queue on GPU 3 (V100). Safe to re-run: it resumes.
exec "$(dirname "$0")/run_gpu.sh" 3
