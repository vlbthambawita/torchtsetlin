#!/bin/bash
# Run the CIFAR-10 batch x clauses queue on GPU 2 (V100). Safe to re-run: it resumes.
exec "$(dirname "$0")/run_gpu.sh" 2
